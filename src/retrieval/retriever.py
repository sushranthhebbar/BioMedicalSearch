"""Unified Biomedical Retriever supporting all 5 configurations and search logging."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from src.config import RetrievalMode, Settings, get_settings
from src.data_loader import Passage, load_passage_map
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex
from src.retrieval.fusion import FusedCandidate, RankedList, reciprocal_rank_fusion
from src.retrieval.query_expansion import BiomedicalQueryExpander, QueryExpansionResult
from src.retrieval.reranker import FlashRankReranker


class RetrievedPassage(BaseModel):
    """Single ranked evidence passage with full score provenance."""

    rank: int
    passage_id: str
    text: str
    pubmed_url: Optional[str] = None
    score: float
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    matched_queries: List[str] = Field(default_factory=list)


class RetrievalLatency(BaseModel):
    """Per-stage retrieval timing breakdown in milliseconds."""

    expansion_ms: float = 0.0
    lexical_ms: float = 0.0
    dense_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


class RetrievalUsage(BaseModel):
    """Token usage and estimated cost incurred during retrieval (e.g. query expansion)."""

    expansion_prompt_tokens: int = 0
    expansion_completion_tokens: int = 0
    expansion_total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    expansion_cached: bool = False
    expansion_fallback_used: bool = False


class RetrievalResult(BaseModel):
    """Complete structured output from a retrieval run across any configured mode."""

    search_id: str
    timestamp: str
    mode: RetrievalMode
    original_query: str
    expanded_queries: List[str] = Field(default_factory=list)
    all_queries: List[str] = Field(default_factory=list)
    passages: List[RetrievedPassage] = Field(default_factory=list)
    latency: RetrievalLatency = Field(default_factory=RetrievalLatency)
    usage: RetrievalUsage = Field(default_factory=RetrievalUsage)


class UnifiedRetriever:
    """Unified retrieval engine over identical BM25 and FAISS passage indexes.

    Supports all 5 benchmark configurations:
      1. `lexical`: BM25 only
      2. `dense`: Dense vector search (FastEmbed + FAISS) only
      3. `hybrid`: BM25 + Dense via RRF (k=60)
      4. `hybrid_qe`: BM25 + Dense via Weighted Multi-Query RRF with Gemini query expansion
      5. `hybrid_qe_rerank`: `hybrid_qe` top-30 candidates reranked by FlashRank cross-encoder
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        dense_index: DenseIndex,
        passage_map: Dict[str, Passage],
        expander: Optional[BiomedicalQueryExpander] = None,
        reranker: Optional[FlashRankReranker] = None,
        settings: Optional[Settings] = None,
        search_log_path: Optional[Path] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.passage_map = passage_map
        self.expander = expander or BiomedicalQueryExpander(settings=self.settings)
        self.reranker = reranker or FlashRankReranker(settings=self.settings)
        self.search_log_path = search_log_path or self.settings.search_log_path

        self._verify_index_alignment()

    def _verify_index_alignment(self) -> None:
        """Verify BM25, FAISS, and the canonical passage map index the exact same passages."""
        if self.bm25_index.passage_ids != self.dense_index.passage_ids:
            raise ValueError(
                "BM25 and FAISS indexes are not aligned on the exact same passage IDs/order "
                f"(bm25={self.bm25_index.size}, faiss={self.dense_index.size})."
            )
        missing_in_store = [
            pid for pid in self.bm25_index.passage_ids if pid not in self.passage_map
        ]
        if missing_in_store:
            raise ValueError(
                f"{len(missing_in_store)} indexed passage IDs are missing from the passage store "
                f"(e.g., {missing_in_store[:3]})."
            )

    @classmethod
    def from_disk(
        cls,
        settings: Optional[Settings] = None,
        expander: Optional[BiomedicalQueryExpander] = None,
        reranker: Optional[FlashRankReranker] = None,
    ) -> "UnifiedRetriever":
        """Load BM25 index, FAISS index, and canonical passages from configured disk paths."""
        cfg = settings or get_settings()
        bm25_index = BM25Index.load(cfg.bm25_dir)
        dense_index = DenseIndex.load(cfg.faiss_dir, settings=cfg)
        passage_map = load_passage_map(cfg.passages_path)
        return cls(
            bm25_index=bm25_index,
            dense_index=dense_index,
            passage_map=passage_map,
            expander=expander,
            reranker=reranker,
            settings=cfg,
        )

    @property
    def corpus_size(self) -> int:
        return self.bm25_index.size

    @property
    def id_fingerprint(self) -> str:
        return self.bm25_index.id_fingerprint

    def _to_retrieved_passage(
        self,
        rank: int,
        passage_id: str,
        score: float,
        bm25_score: Optional[float] = None,
        dense_score: Optional[float] = None,
        rrf_score: Optional[float] = None,
        rerank_score: Optional[float] = None,
        matched_queries: Optional[List[str]] = None,
    ) -> RetrievedPassage:
        passage = self.passage_map[passage_id]
        return RetrievedPassage(
            rank=rank,
            passage_id=passage.id,
            text=passage.text,
            pubmed_url=passage.pubmed_url,
            score=round(float(score), 6),
            bm25_score=round(float(bm25_score), 6) if bm25_score is not None else None,
            dense_score=round(float(dense_score), 6) if dense_score is not None else None,
            rrf_score=round(float(rrf_score), 6) if rrf_score is not None else None,
            rerank_score=round(float(rerank_score), 6) if rerank_score is not None else None,
            matched_queries=list(matched_queries or []),
        )

    def _record_search(self, result: RetrievalResult) -> None:
        """Append retrieval metadata (query, expansions, passage IDs, scores) to JSONL log."""
        self.search_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "search_id": result.search_id,
            "timestamp": result.timestamp,
            "mode": result.mode.value,
            "original_query": result.original_query,
            "expanded_queries": result.expanded_queries,
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
                }
                for p in result.passages
            ],
            "latency_ms": result.latency.model_dump(),
            "usage": result.usage.model_dump(),
        }
        with self.search_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve(
        self,
        query: str,
        mode: Union[RetrievalMode, str] = RetrievalMode.HYBRID_QE_RERANK,
        top_k: Optional[int] = None,
        record_search: bool = True,
        use_expansion_cache: bool = True,
    ) -> RetrievalResult:
        """Execute retrieval for a biomedical query under the requested mode."""
        t_start = time.perf_counter()
        resolved_mode = (
            mode if isinstance(mode, RetrievalMode) else RetrievalMode(str(mode))
        )
        k = top_k if top_k is not None else self.settings.default_top_k
        pool_k = max(k, self.settings.candidate_pool_k)
        rerank_k = max(k, self.settings.rerank_candidate_k)

        orig_query = " ".join((query or "").strip().split())
        search_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()

        latency = RetrievalLatency()
        usage = RetrievalUsage()
        expanded_queries: List[str] = []
        all_queries: List[str] = [orig_query] if orig_query else []

        if not orig_query or k <= 0:
            latency.total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            empty_result = RetrievalResult(
                search_id=search_id,
                timestamp=timestamp,
                mode=resolved_mode,
                original_query=orig_query,
                expanded_queries=[],
                all_queries=all_queries,
                passages=[],
                latency=latency,
                usage=usage,
            )
            if record_search:
                self._record_search(empty_result)
            return empty_result

        # Step 1: Optional Query Expansion
        if resolved_mode in (RetrievalMode.HYBRID_QE, RetrievalMode.HYBRID_QE_RERANK):
            qe_res: QueryExpansionResult = self.expander.expand(
                orig_query, use_cache=use_expansion_cache
            )
            expanded_queries = qe_res.expanded_queries
            all_queries = qe_res.all_queries
            latency.expansion_ms = qe_res.latency_ms
            usage.expansion_prompt_tokens = qe_res.prompt_tokens
            usage.expansion_completion_tokens = qe_res.completion_tokens
            usage.expansion_total_tokens = qe_res.total_tokens
            usage.estimated_cost_usd = qe_res.estimated_cost_usd
            usage.expansion_cached = qe_res.cached
            usage.expansion_fallback_used = qe_res.fallback_used

        passages_out: List[RetrievedPassage] = []

        # Step 2: Execute mode-specific retrieval
        if resolved_mode == RetrievalMode.LEXICAL:
            t0 = time.perf_counter()
            hits = self.bm25_index.search(orig_query, top_k=k)
            latency.lexical_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            passages_out = [
                self._to_retrieved_passage(
                    rank=idx,
                    passage_id=pid,
                    score=score,
                    bm25_score=score,
                    matched_queries=[orig_query],
                )
                for idx, (pid, score) in enumerate(hits, start=1)
            ]

        elif resolved_mode == RetrievalMode.DENSE:
            t0 = time.perf_counter()
            hits = self.dense_index.search(orig_query, top_k=k)
            latency.dense_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            passages_out = [
                self._to_retrieved_passage(
                    rank=idx,
                    passage_id=pid,
                    score=score,
                    dense_score=score,
                    matched_queries=[orig_query],
                )
                for idx, (pid, score) in enumerate(hits, start=1)
            ]

        elif resolved_mode in (
            RetrievalMode.HYBRID,
            RetrievalMode.HYBRID_QE,
            RetrievalMode.HYBRID_QE_RERANK,
        ):
            ranked_lists: List[RankedList] = []
            lexical_elapsed = 0.0
            dense_elapsed = 0.0

            for q_idx, q_text in enumerate(all_queries):
                if resolved_mode == RetrievalMode.HYBRID:
                    q_weight = 1.0
                else:
                    q_weight = (
                        self.settings.qe_original_query_weight
                        if q_idx == 0
                        else self.settings.qe_expanded_query_weight
                    )

                t_lex = time.perf_counter()
                lex_hits = self.bm25_index.search(q_text, top_k=pool_k)
                lexical_elapsed += (time.perf_counter() - t_lex) * 1000.0

                t_den = time.perf_counter()
                den_hits = self.dense_index.search(q_text, top_k=pool_k)
                dense_elapsed += (time.perf_counter() - t_den) * 1000.0

                ranked_lists.append(
                    RankedList(
                        source="lexical",
                        query=q_text,
                        results=lex_hits,
                        weight=q_weight * self.settings.rrf_lexical_weight,
                    )
                )
                ranked_lists.append(
                    RankedList(
                        source="dense",
                        query=q_text,
                        results=den_hits,
                        weight=q_weight * self.settings.rrf_dense_weight,
                    )
                )

            latency.lexical_ms = round(lexical_elapsed, 2)
            latency.dense_ms = round(dense_elapsed, 2)

            t_fuse = time.perf_counter()
            fuse_limit = (
                rerank_k if resolved_mode == RetrievalMode.HYBRID_QE_RERANK else k
            )
            fused: List[FusedCandidate] = reciprocal_rank_fusion(
                ranked_lists=ranked_lists,
                k=self.settings.rrf_k,
                top_k=fuse_limit,
            )
            latency.fusion_ms = round((time.perf_counter() - t_fuse) * 1000.0, 2)

            if resolved_mode == RetrievalMode.HYBRID_QE_RERANK and fused:
                cand_by_id = {c.passage_id: c for c in fused}
                rerank_inputs: List[Tuple[str, str]] = [
                    (c.passage_id, self.passage_map[c.passage_id].text) for c in fused
                ]
                t_rr = time.perf_counter()
                reranked = self.reranker.rerank(
                    query=orig_query,
                    passages=rerank_inputs,
                    top_k=k,
                )
                latency.rerank_ms = round((time.perf_counter() - t_rr) * 1000.0, 2)

                passages_out = [
                    self._to_retrieved_passage(
                        rank=idx,
                        passage_id=pid,
                        score=rr_score,
                        bm25_score=cand_by_id[pid].bm25_score,
                        dense_score=cand_by_id[pid].dense_score,
                        rrf_score=cand_by_id[pid].rrf_score,
                        rerank_score=rr_score,
                        matched_queries=cand_by_id[pid].matched_queries,
                    )
                    for idx, (pid, rr_score) in enumerate(reranked, start=1)
                ]
            else:
                passages_out = [
                    self._to_retrieved_passage(
                        rank=idx,
                        passage_id=cand.passage_id,
                        score=cand.rrf_score,
                        bm25_score=cand.bm25_score,
                        dense_score=cand.dense_score,
                        rrf_score=cand.rrf_score,
                        matched_queries=cand.matched_queries,
                    )
                    for idx, cand in enumerate(fused[:k], start=1)
                ]

        latency.total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        result = RetrievalResult(
            search_id=search_id,
            timestamp=timestamp,
            mode=resolved_mode,
            original_query=orig_query,
            expanded_queries=expanded_queries,
            all_queries=all_queries,
            passages=passages_out,
            latency=latency,
            usage=usage,
        )

        if record_search:
            self._record_search(result)

        return result
