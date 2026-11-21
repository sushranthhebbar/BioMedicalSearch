"""CPU-friendly ONNX cross-encoder reranker backed by FlashRank."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import Settings, get_settings


class FlashRankReranker:
    """Lazy-loading wrapper around `flashrank.Ranker` (`ms-marco-MiniLM-L-12-v2`)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_length: int = 512,
        ranker: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.reranker_model
        self.cache_dir = cache_dir or self.settings.flashrank_cache_dir
        self.max_length = max_length
        self._ranker = ranker

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                max_length=self.max_length,
            )
        return self._ranker

    def rerank(
        self,
        query: str,
        passages: Sequence[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank `(passage_id, passage_text)` pairs against `query`.

        Returns a list of `(passage_id, rerank_score)` sorted descending by score
        with deterministic tie-breaking by `passage_id`.
        """
        if not passages or not query or not query.strip():
            return []

        from flashrank import RerankRequest

        ranker = self._get_ranker()
        payload: List[Dict[str, Any]] = [
            {"id": pid, "text": text} for pid, text in passages
        ]
        request = RerankRequest(query=query.strip(), passages=payload)
        raw_results = ranker.rerank(request)

        scored: List[Tuple[str, float]] = [
            (str(item["id"]), float(item["score"])) for item in raw_results
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))

        if top_k is not None:
            return scored[: max(0, top_k)]
        return scored
