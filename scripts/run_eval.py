"""Reproducible 5-configuration offline evaluation runner over the fixed 100-query benchmark.

Runs all 5 required configurations through the exact same `RAGPipeline` over the exact same
100 query IDs (`eval/eval_queries_100.json`):
  1. `lexical`           (BM25)
  2. `dense`             (Local ONNX FastEmbed + FAISS)
  3. `hybrid`            (BM25 + Dense via RRF, k=60)
  4. `hybrid_qe`         (BM25 + Dense via Weighted Multi-Query RRF + Gemini QE)
  5. `hybrid_qe_rerank`  (Strongest Variant: Hybrid+QE top-30 + FlashRank ONNX cross-encoder)

Features:
- Caches query expansions per query ID so `hybrid_qe` and `hybrid_qe_rerank` are strictly
  apples-to-apples and incur expansion LLM calls only once per query.
- Records both observed and effective (uncached-expansion-adjusted) latency/token/cost metrics.
- Checkpoints every evaluated query immediately to `eval/results/<config_id>_runs.jsonl` and
  supports clean, idempotent resume across interruptions or Gemini rate-limit pauses.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RetrievalMode, Settings, estimate_llm_cost_usd, get_settings
from src.generation.answerer import GroundedAnswerer
from src.pipeline import PipelineResult, RAGPipeline
from src.retrieval.query_expansion import BiomedicalQueryExpander, _RateLimiter
from src.retrieval.retriever import UnifiedRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ALL_EVAL_MODES: List[RetrievalMode] = [
    RetrievalMode.LEXICAL,
    RetrievalMode.DENSE,
    RetrievalMode.HYBRID,
    RetrievalMode.HYBRID_QE,
    RetrievalMode.HYBRID_QE_RERANK,
]

QE_MODES = {RetrievalMode.HYBRID_QE, RetrievalMode.HYBRID_QE_RERANK}


# ---------------------------------------------------------------------------
# Eval dataset & checkpoint I/O helpers
# ---------------------------------------------------------------------------


def load_eval_queries(eval_file: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load the fixed 100-query benchmark JSON file and validate basic invariants."""
    if not eval_file.exists():
        raise FileNotFoundError(
            f"Evaluation query file not found: {eval_file}. "
            "Run `python scripts/create_eval_set.py` first."
        )
    with eval_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        metadata: Dict[str, Any] = {"total_queries": len(data)}
        queries = data
    elif isinstance(data, dict) and isinstance(data.get("queries"), list):
        metadata = dict(data.get("metadata") or {})
        queries = list(data["queries"])
    else:
        raise ValueError(f"Unexpected format in evaluation file: {eval_file}")

    seen_ids: set[str] = set()
    for item in queries:
        qid = str(item.get("id", "")).strip()
        if not qid or qid in seen_ids:
            raise ValueError(f"Invalid or duplicate query ID in {eval_file}: {qid!r}")
        seen_ids.add(qid)
        if not item.get("question") or not item.get("relevant_passage_ids"):
            raise ValueError(f"Query {qid} is missing question or relevant_passage_ids.")

    return metadata, queries


def is_record_complete(
    record: Dict[str, Any],
    mode: RetrievalMode,
    require_generation: bool,
    accept_expansion_fallback: bool = False,
) -> bool:
    """Return True if a checkpointed run record completed cleanly and does not need retry."""
    if not isinstance(record, dict):
        return False
    if str(record.get("config_id", "")) != mode.value:
        return False
    if require_generation:
        if not record.get("generation_enabled", False):
            return False
        if record.get("abstention_stage") == "generation_error":
            return False
    if mode in QE_MODES and not accept_expansion_fallback:
        usage = record.get("usage") or {}
        if usage.get("expansion_fallback_used", False):
            return False
    return True


def load_existing_runs(run_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load checkpointed per-query JSONL records keyed by query_id (latest entry wins)."""
    records: Dict[str, Dict[str, Any]] = {}
    if not run_path.exists():
        return records

    with run_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("query_id") is not None:
                records[str(rec["query_id"])] = rec
    return records


def write_ordered_runs_jsonl(
    run_path: Path,
    ordered_query_ids: Sequence[str],
    records_by_id: Dict[str, Dict[str, Any]],
) -> None:
    """Atomically rewrite `<config_id>_runs.jsonl` in exact canonical eval query order."""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = run_path.with_suffix(".tmp.jsonl")
    with tmp_path.open("w", encoding="utf-8") as f:
        for qid in ordered_query_ids:
            rec = records_by_id.get(qid)
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(run_path)


# ---------------------------------------------------------------------------
# Record construction & apples-to-apples effective expansion accounting
# ---------------------------------------------------------------------------


def build_eval_record(
    query_item: Dict[str, Any],
    result: PipelineResult,
    cached_expansion_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a `PipelineResult` + benchmark ground truth into a self-contained eval record."""
    qid = str(query_item["id"])
    relevant_ids = [str(pid) for pid in query_item["relevant_passage_ids"]]
    relevant_set = set(relevant_ids)
    selected_set = set(result.selected_passage_ids)

    retrieved_top_k: List[Dict[str, Any]] = []
    selected_passages: List[Dict[str, Any]] = []

    for p in result.passages:
        is_rel = p.passage_id in relevant_set
        retrieved_top_k.append(
            {
                "rank": p.rank,
                "passage_id": p.passage_id,
                "score": p.score,
                "bm25_score": p.bm25_score,
                "dense_score": p.dense_score,
                "rrf_score": p.rrf_score,
                "rerank_score": p.rerank_score,
                "matched_queries": list(p.matched_queries),
                "selected_for_answer": p.selected_for_answer,
                "cited": p.cited,
                "is_relevant": is_rel,
            }
        )
        if p.passage_id in selected_set:
            selected_passages.append(
                {
                    "rank": p.rank,
                    "passage_id": p.passage_id,
                    "score": p.score,
                    "text": p.text,
                    "pubmed_url": p.pubmed_url,
                    "is_relevant": is_rel,
                }
            )

    raw_latency = result.latency.model_dump()
    raw_usage = result.usage.model_dump()

    # Fair uncached expansion accounting so Config 4 & Config 5 are compared apples-to-apples
    # even when Config 5 (or a resumed Config 4) hits the expansion cache.
    eff_latency = dict(raw_latency)
    eff_usage = dict(raw_usage)

    if (
        result.mode in QE_MODES
        and raw_usage.get("expansion_cached")
        and isinstance(cached_expansion_meta, dict)
    ):
        exp_p = int(cached_expansion_meta.get("prompt_tokens", 0) or 0)
        exp_c = int(cached_expansion_meta.get("completion_tokens", 0) or 0)
        exp_t = exp_p + exp_c
        model_name = str(cached_expansion_meta.get("model_name") or "gemini-2.5-flash")
        exp_cost = float(
            cached_expansion_meta.get(
                "estimated_cost_usd",
                estimate_llm_cost_usd(model_name, exp_p, exp_c),
            )
            or 0.0
        )
        exp_ms = float(cached_expansion_meta.get("latency_ms", 0.0) or 0.0)

        eff_usage["expansion_prompt_tokens"] = exp_p
        eff_usage["expansion_completion_tokens"] = exp_c
        eff_usage["expansion_total_tokens"] = exp_t
        eff_usage["total_tokens"] = exp_t + int(raw_usage.get("generation_total_tokens", 0) or 0)
        eff_usage["expansion_cost_usd"] = round(exp_cost, 8)
        eff_usage["estimated_cost_usd"] = round(
            exp_cost + float(raw_usage.get("generation_cost_usd", 0.0) or 0.0), 8
        )

        if exp_ms > 0.0:
            eff_latency["expansion_ms"] = round(exp_ms, 2)
            eff_latency["retrieval_total_ms"] = round(
                float(raw_latency.get("retrieval_total_ms", 0.0)) + exp_ms, 2
            )
            eff_latency["total_ms"] = round(
                float(raw_latency.get("total_ms", 0.0)) + exp_ms, 2
            )

    return {
        "query_id": qid,
        "category": str(query_item.get("category", "unknown")),
        "config_id": result.mode.value,
        "timestamp": result.timestamp,
        "search_id": result.search_id,
        "question": str(query_item["question"]),
        "gold_answer": str(query_item.get("answer", "")),
        "relevant_passage_ids": relevant_ids,
        "original_query": result.original_query,
        "expanded_queries": list(result.expanded_queries),
        "all_queries": list(result.all_queries),
        "retrieved_top_k": retrieved_top_k,
        "retrieved_passage_ids": [p.passage_id for p in result.passages],
        "selected_passage_ids": list(result.selected_passage_ids),
        "selected_passages": selected_passages,
        "answer": result.answer,
        "sufficient_evidence": result.sufficient_evidence,
        "insufficient_reason": result.insufficient_reason,
        "abstention_stage": result.abstention_stage,
        "citation_validation": (
            result.citation_validation.model_dump()
            if result.citation_validation is not None
            else None
        ),
        "threshold_used": result.threshold_used,
        "top_retrieval_score": result.top_retrieval_score,
        "answer_model": result.answer_model,
        "generation_enabled": result.generation_enabled,
        "latency_ms": raw_latency,
        "effective_latency_ms": eff_latency,
        "usage": raw_usage,
        "effective_usage": eff_usage,
    }


# ---------------------------------------------------------------------------
# Pre-compute shared 100-query expansions for Configs 4 & 5
# ---------------------------------------------------------------------------


def precompute_expansions(
    queries: Sequence[Dict[str, Any]],
    expander: BiomedicalQueryExpander,
    snapshot_path: Path,
    max_consecutive_errors: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Warm the expansion cache for all eval queries and export a reproducible snapshot."""
    snapshot: Dict[str, Dict[str, Any]] = {}
    if snapshot_path.exists():
        try:
            with snapshot_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                snapshot = loaded
        except Exception:
            snapshot = {}

    cached_hits = 0
    fresh_calls = 0
    fallbacks = 0
    consecutive_errors = 0

    for item in tqdm(queries, desc="Precomputing QE (100 queries)"):
        qid = str(item["id"])
        qtext = str(item["question"])

        # Seed expander cache from snapshot if available and not yet in local cache
        if qid in snapshot and expander.get_cached_entry(qtext) is None:
            snap_entry = snapshot[qid]
            if snap_entry.get("expanded_queries") and not snap_entry.get("fallback_used"):
                key = expander._cache_key(" ".join(qtext.strip().split()), expander.num_expansions)
                with expander._cache_lock:
                    expander._cache[key] = {
                        "original_query": " ".join(qtext.strip().split()),
                        "expanded_queries": list(snap_entry["expanded_queries"]),
                        "model_name": snap_entry.get("model_name", expander.model_name),
                        "latency_ms": float(snap_entry.get("latency_ms", 0.0) or 0.0),
                        "prompt_tokens": int(snap_entry.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(snap_entry.get("completion_tokens", 0) or 0),
                        "estimated_cost_usd": float(snap_entry.get("estimated_cost_usd", 0.0) or 0.0),
                    }
                    expander._save_cache()

        res = expander.expand(qtext, use_cache=True)
        cached_meta = expander.get_cached_entry(qtext)

        if res.fallback_used:
            fallbacks += 1
            consecutive_errors += 1
            logger.warning("QE fallback for query %s: %s", qid, res.error)
            if consecutive_errors >= max_consecutive_errors:
                logger.error(
                    "Stopping QE precomputation after %d consecutive API errors.",
                    consecutive_errors,
                )
                break
            continue

        consecutive_errors = 0
        if res.cached:
            cached_hits += 1
        else:
            fresh_calls += 1

        snapshot[qid] = {
            "query_id": qid,
            "original_query": res.original_query,
            "expanded_queries": list(res.expanded_queries),
            "model_name": res.model_name,
            "latency_ms": (
                float(cached_meta.get("latency_ms", 0.0) or 0.0)
                if res.cached and cached_meta
                else res.latency_ms
            ),
            "prompt_tokens": (
                int(cached_meta.get("prompt_tokens", 0) or 0)
                if res.cached and cached_meta
                else res.prompt_tokens
            ),
            "completion_tokens": (
                int(cached_meta.get("completion_tokens", 0) or 0)
                if res.cached and cached_meta
                else res.completion_tokens
            ),
            "estimated_cost_usd": (
                float(cached_meta.get("estimated_cost_usd", 0.0) or 0.0)
                if res.cached and cached_meta
                else res.estimated_cost_usd
            ),
            "fallback_used": False,
        }

        # Persist snapshot after each fresh call so progress is durable
        if not res.cached:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = snapshot_path.with_suffix(".tmp.json")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            tmp.replace(snapshot_path)

    if snapshot:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = snapshot_path.with_suffix(".tmp.json")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        tmp.replace(snapshot_path)

    logger.info(
        "Query expansion cache ready: %d total (%d cached, %d fresh, %d fallbacks) -> %s",
        len(snapshot),
        cached_hits,
        fresh_calls,
        fallbacks,
        snapshot_path,
    )
    return snapshot


# ---------------------------------------------------------------------------
# Inline IR & operational diagnostics (full dual-layer scoring lives in Phase 4)
# ---------------------------------------------------------------------------


def _dcg_at_k(relevances: Sequence[int], k: int) -> float:
    return sum(
        rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]) if rel > 0
    )


def summarize_config_runs(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute quick retrieval, citation, latency, and cost summary for a config's run records."""
    n = len(records)
    if n == 0:
        return {"count": 0}

    r5_sum = 0.0
    r10_sum = 0.0
    mrr10_sum = 0.0
    ndcg10_sum = 0.0
    eff_total_ms: List[float] = []
    eff_ret_ms: List[float] = []
    eff_cost_sum = 0.0
    sufficient_count = 0
    abstention_count = 0
    gen_error_count = 0
    cit_valid_rates: List[float] = []
    zero_halluc_count = 0
    cit_checked = 0

    for rec in records:
        rel_set = set(str(pid) for pid in (rec.get("relevant_passage_ids") or []))
        ret_ids = [str(pid) for pid in (rec.get("retrieved_passage_ids") or [])]

        if rel_set:
            hits_5 = sum(1 for pid in ret_ids[:5] if pid in rel_set)
            hits_10 = sum(1 for pid in ret_ids[:10] if pid in rel_set)
            r5_sum += hits_5 / float(len(rel_set))
            r10_sum += hits_10 / float(len(rel_set))

            rr = 0.0
            for idx, pid in enumerate(ret_ids[:10], start=1):
                if pid in rel_set:
                    rr = 1.0 / float(idx)
                    break
            mrr10_sum += rr

            rels = [1 if pid in rel_set else 0 for pid in ret_ids[:10]]
            ideal_rels = [1] * min(len(rel_set), 10)
            idcg = _dcg_at_k(ideal_rels, 10)
            ndcg10_sum += (_dcg_at_k(rels, 10) / idcg) if idcg > 0 else 0.0

        lat = rec.get("effective_latency_ms") or rec.get("latency_ms") or {}
        eff_total_ms.append(float(lat.get("total_ms", 0.0) or 0.0))
        eff_ret_ms.append(float(lat.get("retrieval_total_ms", 0.0) or 0.0))

        usage = rec.get("effective_usage") or rec.get("usage") or {}
        eff_cost_sum += float(usage.get("estimated_cost_usd", 0.0) or 0.0)

        if rec.get("abstention_stage") == "generation_error":
            gen_error_count += 1
        if rec.get("sufficient_evidence") is True:
            sufficient_count += 1
        elif rec.get("sufficient_evidence") is False:
            abstention_count += 1

        cit = rec.get("citation_validation")
        if isinstance(cit, dict):
            cit_checked += 1
            if cit.get("citation_validity_rate") is not None:
                cit_valid_rates.append(float(cit["citation_validity_rate"]))
            if cit.get("zero_hallucinated_ids", True):
                zero_halluc_count += 1

    sorted_ms = sorted(eff_total_ms)
    p95_idx = min(len(sorted_ms) - 1, int(math.ceil(0.95 * len(sorted_ms))) - 1)

    return {
        "count": n,
        "recall_at_5": round(r5_sum / n, 4),
        "recall_at_10": round(r10_sum / n, 4),
        "mrr_at_10": round(mrr10_sum / n, 4),
        "ndcg_at_10": round(ndcg10_sum / n, 4),
        "avg_retrieval_ms": round(sum(eff_ret_ms) / n, 2),
        "avg_total_ms": round(sum(eff_total_ms) / n, 2),
        "p95_total_ms": round(sorted_ms[p95_idx], 2),
        "est_cost_usd_total": round(eff_cost_sum, 6),
        "sufficient_count": sufficient_count,
        "abstention_count": abstention_count,
        "generation_error_count": gen_error_count,
        "citation_validity_rate": (
            round(sum(cit_valid_rates) / len(cit_valid_rates), 4)
            if cit_valid_rates
            else None
        ),
        "zero_hallucinated_id_rate": (
            round(zero_halluc_count / cit_checked, 4) if cit_checked > 0 else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------


def run_config_evaluation(
    pipeline: RAGPipeline,
    mode: RetrievalMode,
    queries: Sequence[Dict[str, Any]],
    output_dir: Path,
    top_k: int = 10,
    answer_context_k: int = 5,
    generate_answer: bool = True,
    resume: bool = True,
    accept_expansion_fallback: bool = False,
    max_consecutive_errors: int = 5,
    expansion_snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one configuration across all benchmark queries with durable per-query checkpointing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / f"{mode.value}_runs.jsonl"
    ordered_qids = [str(item["id"]) for item in queries]

    records_by_id: Dict[str, Dict[str, Any]] = {}
    if resume and run_path.exists():
        records_by_id = load_existing_runs(run_path)

    already_done = sum(
        1
        for qid in ordered_qids
        if qid in records_by_id
        and is_record_complete(
            records_by_id[qid],
            mode=mode,
            require_generation=generate_answer,
            accept_expansion_fallback=accept_expansion_fallback,
        )
    )
    logger.info(
        "=== Config [%s] | %d/%d already complete (generate_answer=%s) ===",
        mode.value,
        already_done,
        len(queries),
        generate_answer,
    )

    consecutive_errors = 0
    aborted_early = False

    with run_path.open("a", encoding="utf-8") as append_fh:
        for item in tqdm(queries, desc=f"Eval [{mode.value}]"):
            qid = str(item["id"])
            existing = records_by_id.get(qid)
            if existing is not None and is_record_complete(
                existing,
                mode=mode,
                require_generation=generate_answer,
                accept_expansion_fallback=accept_expansion_fallback,
            ):
                continue

            qtext = str(item["question"])
            result: PipelineResult = pipeline.run(
                query=qtext,
                mode=mode,
                top_k=top_k,
                answer_context_k=answer_context_k,
                generate_answer=generate_answer,
                record_search=False,
                use_expansion_cache=True,
            )

            cached_meta: Optional[Dict[str, Any]] = None
            if mode in QE_MODES:
                cached_meta = pipeline.retriever.expander.get_cached_entry(qtext)
                if cached_meta is None and expansion_snapshot and qid in expansion_snapshot:
                    cached_meta = expansion_snapshot[qid]

            record = build_eval_record(
                query_item=item,
                result=result,
                cached_expansion_meta=cached_meta,
            )

            records_by_id[qid] = record
            append_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            append_fh.flush()

            has_error = (
                generate_answer and record.get("abstention_stage") == "generation_error"
            ) or (
                mode in QE_MODES
                and not accept_expansion_fallback
                and bool((record.get("usage") or {}).get("expansion_fallback_used", False))
            )

            if has_error:
                consecutive_errors += 1
                logger.warning(
                    "[%s] Query %s encountered API error (%d/%d consecutive): %s",
                    mode.value,
                    qid,
                    consecutive_errors,
                    max_consecutive_errors,
                    record.get("insufficient_reason"),
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        "[%s] Halting early after %d consecutive API errors. "
                        "Progress is saved; re-run to resume cleanly.",
                        mode.value,
                        consecutive_errors,
                    )
                    aborted_early = True
                    break
            else:
                consecutive_errors = 0

    # Compact & rewrite JSONL in exact canonical benchmark order (deduplicating retries)
    write_ordered_runs_jsonl(run_path, ordered_qids, records_by_id)

    ordered_records = [records_by_id[qid] for qid in ordered_qids if qid in records_by_id]
    summary = summarize_config_runs(ordered_records)
    summary["config_id"] = mode.value
    summary["complete"] = (
        len(ordered_records) == len(queries)
        and all(
            is_record_complete(
                rec,
                mode=mode,
                require_generation=generate_answer,
                accept_expansion_fallback=accept_expansion_fallback,
            )
            for rec in ordered_records
        )
    )
    summary["aborted_early"] = aborted_early
    summary["output_file"] = str(run_path)

    logger.info(
        "[%s] Done (%d/%d) | R@5=%.4f R@10=%.4f MRR@10=%.4f nDCG@10=%.4f | "
        "AvgMs=%.1f | Cost=$%.5f | CitValid=%s | Suff=%d Abstain=%d Err=%d",
        mode.value,
        len(ordered_records),
        len(queries),
        summary.get("recall_at_5", 0.0),
        summary.get("recall_at_10", 0.0),
        summary.get("mrr_at_10", 0.0),
        summary.get("ndcg_at_10", 0.0),
        summary.get("avg_total_ms", 0.0),
        summary.get("est_cost_usd_total", 0.0),
        summary.get("citation_validity_rate"),
        summary.get("sufficient_count", 0),
        summary.get("abstention_count", 0),
        summary.get("generation_error_count", 0),
    )

    return ordered_records, summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible 5-config offline evaluation over the fixed 100-query BioASQ set."
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Fixed evaluation query JSON (default: eval/eval_queries_100.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-config JSONL run outputs (default: eval/results).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[m.value for m in ALL_EVAL_MODES],
        default=[m.value for m in ALL_EVAL_MODES],
        help="Subset of configurations to run (default: all 5).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of passages retrieved per query (default: 10).",
    )
    parser.add_argument(
        "--answer-context-k",
        type=int,
        default=5,
        help="Number of top passages passed to the grounded LLM answerer (default: 5).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM answer generation (useful for quick retrieval verification).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpointed JSONL files and re-run from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of queries (for smoke testing only).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Override Gemini RPM pacing limit (0 disables pacing).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for transient Gemini 429/503 responses (default: 5).",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Stop a config run cleanly after this many consecutive API errors (default: 5).",
    )
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Only warm the 100-query expansion cache and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings: Settings = get_settings()
    if args.rpm is not None:
        settings = settings.model_copy(update={"gemini_rpm_limit": args.rpm})

    eval_file = args.eval_file or (settings.eval_dir / "eval_queries_100.json")
    output_dir = args.output_dir or (settings.eval_dir / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_meta, queries = load_eval_queries(eval_file)
    if args.limit is not None:
        queries = queries[: args.limit]
        logger.warning("Limiting evaluation to first %d queries (--limit).", len(queries))

    selected_modes = [RetrievalMode(name) for name in args.configs]

    logger.info(
        "Loading aligned BM25 + FAISS + canonical passage store for evaluation (%d queries, %d configs)...",
        len(queries),
        len(selected_modes),
    )
    t_load = time.perf_counter()

    # Share one thread-safe rate limiter across expander and answerer to avoid RPM bursts
    shared_limiter = _RateLimiter(settings.gemini_rpm_limit)
    expander = BiomedicalQueryExpander(settings=settings, max_retries=args.max_retries)
    expander._rate_limiter = shared_limiter
    answerer = GroundedAnswerer(settings=settings, max_retries=args.max_retries)
    answerer._rate_limiter = shared_limiter

    retriever = UnifiedRetriever.from_disk(settings=settings, expander=expander)
    pipeline = RAGPipeline(retriever=retriever, answerer=answerer, settings=settings)

    # Verify corpus fingerprint alignment with the fixed eval set if present
    expected_fp = eval_meta.get("corpus_id_fingerprint")
    if expected_fp and expected_fp != retriever.id_fingerprint:
        logger.warning(
            "Corpus fingerprint mismatch: eval set=%s vs loaded index=%s",
            str(expected_fp)[:16],
            retriever.id_fingerprint[:16],
        )

    logger.info(
        "Loaded RAGPipeline (%d passages) in %.2fs.",
        retriever.corpus_size,
        time.perf_counter() - t_load,
    )

    # Warm up local ONNX models (FastEmbed & FlashRank) once so query #1 latency has no cold-start spike
    if any(m != RetrievalMode.LEXICAL for m in selected_modes):
        retriever.dense_index.embedder.embed_queries(["warmup biomedical query"])
    if RetrievalMode.HYBRID_QE_RERANK in selected_modes:
        retriever.reranker.rerank(
            "warmup biomedical query",
            [("warmup", "warmup biomedical passage text")],
            top_k=1,
        )

    # Precompute shared query expansions once if any QE config is requested
    expansion_snapshot: Dict[str, Dict[str, Any]] = {}
    needs_qe = args.precompute_only or any(m in QE_MODES for m in selected_modes)
    if needs_qe:
        expansion_snapshot = precompute_expansions(
            queries=queries,
            expander=expander,
            snapshot_path=output_dir / "expansions_100.json",
            max_consecutive_errors=args.max_consecutive_errors,
        )
        if args.precompute_only:
            return

    all_summaries: Dict[str, Dict[str, Any]] = {}
    t_eval_start = time.perf_counter()

    for mode in selected_modes:
        _, summary = run_config_evaluation(
            pipeline=pipeline,
            mode=mode,
            queries=queries,
            output_dir=output_dir,
            top_k=args.top_k,
            answer_context_k=args.answer_context_k,
            generate_answer=not args.retrieval_only,
            resume=not args.no_resume,
            max_consecutive_errors=args.max_consecutive_errors,
            expansion_snapshot=expansion_snapshot,
        )
        all_summaries[mode.value] = summary

    manifest_path = output_dir / "phase3_run_manifest.json"
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "eval_file": str(eval_file),
        "num_queries": len(queries),
        "corpus_size": retriever.corpus_size,
        "corpus_id_fingerprint": retriever.id_fingerprint,
        "top_k": args.top_k,
        "answer_context_k": args.answer_context_k,
        "generate_answer": not args.retrieval_only,
        "models": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "expansion_model": settings.gemini_expansion_model,
            "generation_model": settings.gemini_generation_model,
        },
        "min_score_thresholds": settings.min_score_thresholds,
        "elapsed_seconds": round(time.perf_counter() - t_eval_start, 2),
        "configs": all_summaries,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 3 Evaluation Run Summary ===")
    header = (
        f"{'Config':<18} {'Done':>6} {'R@5':>7} {'R@10':>7} {'MRR@10':>8} "
        f"{'nDCG@10':>8} {'AvgMs':>8} {'Cost($)':>9} {'CitVal':>7} {'Suff':>5} {'Err':>4}"
    )
    print(header)
    print("-" * len(header))
    for mode in selected_modes:
        s = all_summaries[mode.value]
        cit_str = (
            f"{s['citation_validity_rate']:.3f}"
            if s.get("citation_validity_rate") is not None
            else "   n/a"
        )
        print(
            f"{mode.value:<18} {s.get('count', 0):>6} "
            f"{s.get('recall_at_5', 0.0):>7.4f} {s.get('recall_at_10', 0.0):>7.4f} "
            f"{s.get('mrr_at_10', 0.0):>8.4f} {s.get('ndcg_at_10', 0.0):>8.4f} "
            f"{s.get('avg_total_ms', 0.0):>8.1f} {s.get('est_cost_usd_total', 0.0):>9.5f} "
            f"{cit_str:>7} {s.get('sufficient_count', 0):>5} {s.get('generation_error_count', 0):>4}"
        )


if __name__ == "__main__":
    main()
