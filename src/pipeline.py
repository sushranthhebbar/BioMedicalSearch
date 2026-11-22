"""Single shared end-to-end Biomedical RAG pipeline used by both the UI and offline evaluation."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.config import RetrievalMode, Settings, get_settings
from src.generation.answerer import GenerationResult, GroundedAnswerer
from src.generation.citations import CitationValidationResult
from src.retrieval.retriever import RetrievalResult, RetrievedPassage, UnifiedRetriever


class PipelinePassage(RetrievedPassage):
    """Ranked evidence passage enriched with answer-context and citation flags."""

    selected_for_answer: bool = False
    cited: bool = False


class PipelineLatency(BaseModel):
    """End-to-end latency breakdown across retrieval and generation stages (ms)."""

    expansion_ms: float = 0.0
    lexical_ms: float = 0.0
    dense_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    retrieval_total_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class PipelineUsage(BaseModel):
    """Combined token usage and estimated USD cost across expansion and answer generation."""

    expansion_prompt_tokens: int = 0
    expansion_completion_tokens: int = 0
    expansion_total_tokens: int = 0
    generation_prompt_tokens: int = 0
    generation_completion_tokens: int = 0
    generation_total_tokens: int = 0
    total_tokens: int = 0
    expansion_cost_usd: float = 0.0
    generation_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    expansion_cached: bool = False
    expansion_fallback_used: bool = False
    repair_attempted: bool = False


class PipelineResult(BaseModel):
    """Complete structured output from a single shared RAGPipeline run."""

    search_id: str
    timestamp: str
    mode: RetrievalMode
    original_query: str
    expanded_queries: List[str] = Field(default_factory=list)
    all_queries: List[str] = Field(default_factory=list)
    passages: List[PipelinePassage] = Field(default_factory=list)
    selected_passage_ids: List[str] = Field(default_factory=list)
    answer: Optional[str] = None
    sufficient_evidence: Optional[bool] = None
    insufficient_reason: Optional[str] = None
    abstention_stage: Optional[str] = None
    citation_validation: Optional[CitationValidationResult] = None
    threshold_used: Optional[float] = None
    top_retrieval_score: Optional[float] = None
    latency: PipelineLatency = Field(default_factory=PipelineLatency)
    usage: PipelineUsage = Field(default_factory=PipelineUsage)
    answer_model: Optional[str] = None
    generation_enabled: bool = True


class RAGPipeline:
    """Shared retrieval + grounded generation + citation validation pipeline."""

    def __init__(
        self,
        retriever: UnifiedRetriever,
        answerer: Optional[GroundedAnswerer] = None,
        settings: Optional[Settings] = None,
        search_log_path: Optional[Path] = None,
    ) -> None:
        self.settings = settings or retriever.settings or get_settings()
        self.retriever = retriever
        self.answerer = answerer or GroundedAnswerer(settings=self.settings)
        self.search_log_path = (
            search_log_path or retriever.search_log_path or self.settings.search_log_path
        )
        self._log_lock = threading.Lock()

    @classmethod
    def from_disk(cls, settings: Optional[Settings] = None) -> "RAGPipeline":
        """Load BM25, FAISS, and canonical passages from disk and initialize the pipeline."""
        cfg = settings or get_settings()
        retriever = UnifiedRetriever.from_disk(settings=cfg)
        answerer = GroundedAnswerer(settings=cfg)
        return cls(retriever=retriever, answerer=answerer, settings=cfg)

    def _record_search(self, result: PipelineResult) -> None:
        """Append full search, generation, citation, latency, and cost metadata to JSONL."""
        self.search_log_path.parent.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "search_id": result.search_id,
            "timestamp": result.timestamp,
            "mode": result.mode.value,
            "original_query": result.original_query,
            "expanded_queries": result.expanded_queries,
            "all_queries": result.all_queries,
            "retrieved_passages": [
                {
                    "rank": p.rank,
                    "passage_id": p.passage_id,
                    "score": p.score,
                    "bm25_score": p.bm25_score,
                    "dense_score": p.dense_score,
                    "rrf_score": p.rrf_score,
                    "rerank_score": p.rerank_score,
                    "matched_queries": p.matched_queries,
                    "selected_for_answer": p.selected_for_answer,
                    "cited": p.cited,
                }
                for p in result.passages
            ],
            "selected_passage_ids": result.selected_passage_ids,
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
            "latency_ms": result.latency.model_dump(),
            "usage": result.usage.model_dump(),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._log_lock:
            with self.search_log_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def load_search_logs(self) -> List[Dict[str, Any]]:
        """Read all recorded JSONL search logs (newest-first ordering handled by callers)."""
        if not self.search_log_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self._log_lock:
            with self.search_log_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            records.append(parsed)
                    except json.JSONDecodeError:
                        continue
        return records

    def hydrate_history_record(self, record: Dict[str, Any]) -> PipelineResult:
        """Reconstruct a full `PipelineResult` (including passage texts) from a JSONL log entry."""
        raw_passages = record.get("retrieved_passages", []) or []
        cit_val_raw = record.get("citation_validation")
        citation_validation = (
            CitationValidationResult.model_validate(cit_val_raw)
            if isinstance(cit_val_raw, dict)
            else None
        )
        cited_set = set(citation_validation.valid_citations) if citation_validation else set()
        selected_ids = list(record.get("selected_passage_ids") or [])
        selected_set = set(selected_ids)

        passages: List[PipelinePassage] = []
        for idx, item in enumerate(raw_passages, start=1):
            pid = str(item.get("passage_id", ""))
            canonical = self.retriever.passage_map.get(pid)
            text = canonical.text if canonical else str(item.get("text", ""))
            pubmed_url = (
                canonical.pubmed_url if canonical else item.get("pubmed_url")
            )
            passages.append(
                PipelinePassage(
                    rank=int(item.get("rank", idx)),
                    passage_id=pid,
                    text=text,
                    pubmed_url=pubmed_url,
                    score=float(item.get("score", 0.0)),
                    bm25_score=item.get("bm25_score"),
                    dense_score=item.get("dense_score"),
                    rrf_score=item.get("rrf_score"),
                    rerank_score=item.get("rerank_score"),
                    matched_queries=list(item.get("matched_queries") or []),
                    selected_for_answer=bool(
                        item.get("selected_for_answer", pid in selected_set)
                    ),
                    cited=bool(item.get("cited", pid in cited_set)),
                )
            )

        lat_raw = record.get("latency_ms") or {}
        retrieval_total = float(
            lat_raw.get("retrieval_total_ms", lat_raw.get("total_ms", 0.0)) or 0.0
        )
        gen_ms = float(lat_raw.get("generation_ms", 0.0) or 0.0)
        total_ms = float(lat_raw.get("total_ms", retrieval_total + gen_ms) or 0.0)
        latency = PipelineLatency(
            expansion_ms=float(lat_raw.get("expansion_ms", 0.0) or 0.0),
            lexical_ms=float(lat_raw.get("lexical_ms", 0.0) or 0.0),
            dense_ms=float(lat_raw.get("dense_ms", 0.0) or 0.0),
            fusion_ms=float(lat_raw.get("fusion_ms", 0.0) or 0.0),
            rerank_ms=float(lat_raw.get("rerank_ms", 0.0) or 0.0),
            retrieval_total_ms=round(retrieval_total, 2),
            generation_ms=round(gen_ms, 2),
            total_ms=round(total_ms, 2),
        )

        usage_raw = record.get("usage") or {}
        exp_p = int(usage_raw.get("expansion_prompt_tokens", 0) or 0)
        exp_c = int(usage_raw.get("expansion_completion_tokens", 0) or 0)
        exp_t = int(usage_raw.get("expansion_total_tokens", exp_p + exp_c) or 0)
        gen_p = int(usage_raw.get("generation_prompt_tokens", 0) or 0)
        gen_c = int(usage_raw.get("generation_completion_tokens", 0) or 0)
        gen_t = int(usage_raw.get("generation_total_tokens", gen_p + gen_c) or 0)
        est_cost = float(usage_raw.get("estimated_cost_usd", 0.0) or 0.0)
        exp_cost = float(
            usage_raw.get(
                "expansion_cost_usd",
                est_cost if gen_t == 0 else 0.0,
            )
            or 0.0
        )
        gen_cost = float(usage_raw.get("generation_cost_usd", 0.0) or 0.0)
        usage = PipelineUsage(
            expansion_prompt_tokens=exp_p,
            expansion_completion_tokens=exp_c,
            expansion_total_tokens=exp_t,
            generation_prompt_tokens=gen_p,
            generation_completion_tokens=gen_c,
            generation_total_tokens=gen_t,
            total_tokens=int(usage_raw.get("total_tokens", exp_t + gen_t) or 0),
            expansion_cost_usd=round(exp_cost, 8),
            generation_cost_usd=round(gen_cost, 8),
            estimated_cost_usd=round(est_cost, 8),
            expansion_cached=bool(usage_raw.get("expansion_cached", False)),
            expansion_fallback_used=bool(usage_raw.get("expansion_fallback_used", False)),
            repair_attempted=bool(usage_raw.get("repair_attempted", False)),
        )

        orig_q = str(record.get("original_query", ""))
        exp_qs = list(record.get("expanded_queries") or [])
        all_qs = list(record.get("all_queries") or ([orig_q] + exp_qs if orig_q else []))

        mode_str = str(record.get("mode", self.settings.default_retrieval_mode.value))
        try:
            mode_enum = RetrievalMode(mode_str)
        except ValueError:
            mode_enum = self.settings.default_retrieval_mode

        return PipelineResult(
            search_id=str(record.get("search_id", "")),
            timestamp=str(record.get("timestamp", "")),
            mode=mode_enum,
            original_query=orig_q,
            expanded_queries=exp_qs,
            all_queries=all_qs,
            passages=passages,
            selected_passage_ids=selected_ids,
            answer=record.get("answer"),
            sufficient_evidence=record.get("sufficient_evidence"),
            insufficient_reason=record.get("insufficient_reason"),
            abstention_stage=record.get("abstention_stage"),
            citation_validation=citation_validation,
            threshold_used=record.get("threshold_used"),
            top_retrieval_score=(
                record.get("top_retrieval_score")
                if record.get("top_retrieval_score") is not None
                else (passages[0].score if passages else None)
            ),
            latency=latency,
            usage=usage,
            answer_model=record.get("answer_model"),
            generation_enabled=bool(
                record.get("generation_enabled", record.get("answer") is not None)
            ),
        )

    def run(
        self,
        query: str,
        mode: Optional[Union[RetrievalMode, str]] = None,
        top_k: Optional[int] = None,
        answer_context_k: Optional[int] = None,
        generate_answer: bool = True,
        record_search: bool = True,
        use_expansion_cache: bool = True,
        min_score_threshold: Optional[float] = None,
    ) -> PipelineResult:
        """Run end-to-end biomedical retrieval, grounded generation, and citation validation."""
        t_start = time.perf_counter()
        resolved_mode = (
            self.settings.default_retrieval_mode
            if mode is None
            else (mode if isinstance(mode, RetrievalMode) else RetrievalMode(str(mode)))
        )
        k = self.settings.default_top_k if top_k is None else int(top_k)
        ctx_k = (
            self.settings.default_answer_context_k
            if answer_context_k is None
            else max(1, int(answer_context_k))
        )

        # Stage 1: Retrieval (recording handled once at pipeline level to include answer metadata)
        ret: RetrievalResult = self.retriever.retrieve(
            query=query,
            mode=resolved_mode,
            top_k=k,
            record_search=False,
            use_expansion_cache=use_expansion_cache,
        )

        # Stage 2: Optional Grounded Answer Generation + Citation Validation
        gen: Optional[GenerationResult] = None
        if generate_answer:
            gen = self.answerer.generate(
                query=ret.original_query,
                passages=ret.passages,
                mode=resolved_mode,
                context_k=ctx_k,
                min_score_threshold=min_score_threshold,
            )

        selected_ids = (
            gen.selected_passage_ids
            if gen is not None
            else [p.passage_id for p in ret.passages[:ctx_k]]
        )
        selected_set = set(selected_ids)
        cited_set = (
            set(gen.citation_validation.valid_citations)
            if gen is not None and gen.citation_validation is not None
            else set()
        )

        enriched_passages = [
            PipelinePassage(
                **p.model_dump(),
                selected_for_answer=p.passage_id in selected_set,
                cited=p.passage_id in cited_set,
            )
            for p in ret.passages
        ]

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        latency = PipelineLatency(
            expansion_ms=ret.latency.expansion_ms,
            lexical_ms=ret.latency.lexical_ms,
            dense_ms=ret.latency.dense_ms,
            fusion_ms=ret.latency.fusion_ms,
            rerank_ms=ret.latency.rerank_ms,
            retrieval_total_ms=ret.latency.total_ms,
            generation_ms=gen.latency_ms if gen is not None else 0.0,
            total_ms=total_ms,
        )

        exp_cost = ret.usage.estimated_cost_usd
        gen_cost = gen.estimated_cost_usd if gen is not None else 0.0
        usage = PipelineUsage(
            expansion_prompt_tokens=ret.usage.expansion_prompt_tokens,
            expansion_completion_tokens=ret.usage.expansion_completion_tokens,
            expansion_total_tokens=ret.usage.expansion_total_tokens,
            generation_prompt_tokens=gen.prompt_tokens if gen is not None else 0,
            generation_completion_tokens=gen.completion_tokens if gen is not None else 0,
            generation_total_tokens=gen.total_tokens if gen is not None else 0,
            total_tokens=ret.usage.expansion_total_tokens
            + (gen.total_tokens if gen is not None else 0),
            expansion_cost_usd=round(exp_cost, 8),
            generation_cost_usd=round(gen_cost, 8),
            estimated_cost_usd=round(exp_cost + gen_cost, 8),
            expansion_cached=ret.usage.expansion_cached,
            expansion_fallback_used=ret.usage.expansion_fallback_used,
            repair_attempted=gen.repair_attempted if gen is not None else False,
        )

        result = PipelineResult(
            search_id=ret.search_id,
            timestamp=ret.timestamp,
            mode=resolved_mode,
            original_query=ret.original_query,
            expanded_queries=ret.expanded_queries,
            all_queries=ret.all_queries,
            passages=enriched_passages,
            selected_passage_ids=selected_ids,
            answer=gen.answer if gen is not None else None,
            sufficient_evidence=gen.sufficient_evidence if gen is not None else None,
            insufficient_reason=gen.insufficient_reason if gen is not None else None,
            abstention_stage=gen.abstention_stage if gen is not None else None,
            citation_validation=gen.citation_validation if gen is not None else None,
            threshold_used=gen.threshold_used if gen is not None else None,
            top_retrieval_score=(
                gen.top_retrieval_score
                if gen is not None
                else (ret.passages[0].score if ret.passages else None)
            ),
            latency=latency,
            usage=usage,
            answer_model=gen.model_name if gen is not None else None,
            generation_enabled=generate_answer,
        )

        if record_search:
            self._record_search(result)

        return result
