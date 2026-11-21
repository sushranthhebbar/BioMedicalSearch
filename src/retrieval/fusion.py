"""Reciprocal Rank Fusion (RRF) and weighted multi-query rank fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RankedList:
    """Single ranked retrieval output participating in RRF fusion."""

    source: str  # e.g., "lexical" or "dense"
    query: str
    results: List[Tuple[str, float]]
    weight: float = 1.0


@dataclass
class FusedCandidate:
    """Passage candidate aggregated across lexical, dense, and expanded-query runs."""

    passage_id: str
    rrf_score: float = 0.0
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    matched_queries: List[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[RankedList],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[FusedCandidate]:
    """Combine multiple ranked lists via weighted Reciprocal Rank Fusion (RRF).

    Formula:
        score(d) = sum_r (w_r / (k + rank_r(d)))
    where `rank_r(d)` is 1-based rank in ranked list `r`.

    Ties are broken deterministically by ascending `passage_id`.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be positive, got {k}")

    candidates: Dict[str, FusedCandidate] = {}

    for rlist in ranked_lists:
        if rlist.weight <= 0.0 or not rlist.results:
            continue

        # Guard against duplicate passage IDs within a single ranked list
        seen_in_list: set[str] = set()
        rank = 0
        for pid, raw_score in rlist.results:
            if pid in seen_in_list:
                continue
            seen_in_list.add(pid)
            rank += 1

            cand = candidates.get(pid)
            if cand is None:
                cand = FusedCandidate(passage_id=pid)
                candidates[pid] = cand

            cand.rrf_score += rlist.weight / float(k + rank)

            if rlist.source == "lexical":
                if cand.bm25_score is None or raw_score > cand.bm25_score:
                    cand.bm25_score = float(raw_score)
                if cand.bm25_rank is None or rank < cand.bm25_rank:
                    cand.bm25_rank = rank
            elif rlist.source == "dense":
                if cand.dense_score is None or raw_score > cand.dense_score:
                    cand.dense_score = float(raw_score)
                if cand.dense_rank is None or rank < cand.dense_rank:
                    cand.dense_rank = rank

            if rlist.query and rlist.query not in cand.matched_queries:
                cand.matched_queries.append(rlist.query)

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda c: (-c.rrf_score, c.passage_id),
    )

    if top_k is not None:
        return sorted_candidates[: max(0, top_k)]
    return sorted_candidates
