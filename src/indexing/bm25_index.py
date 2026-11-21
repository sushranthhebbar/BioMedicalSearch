"""Biomedical-friendly BM25 lexical index built with `bm25s`."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import bm25s

from src.data_loader import Passage
from src.indexing import compute_id_fingerprint

# Biomedical-friendly compound token pattern: keeps alphanumeric tokens plus
# hyphenated/slashed/plus biomedical entities (e.g., IL-6, COVID-19, TP53, CD4+, PD-1/PD-L1).
BIOMED_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*\+?")

# Concise English stopword set chosen to avoid stripping biomedical symbols/terms.
ENGLISH_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
        "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
        "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
        "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
        "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
        "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
        "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
        "your", "yours", "yourself", "yourselves",
    }
)


def tokenize_biomedical(text: str) -> List[str]:
    """Tokenize biomedical text preserving hyphenated/alphanumeric terms and sub-tokens.

    Examples:
      - "IL-6" -> ["il-6", "il", "6"]
      - "COVID-19" -> ["covid-19", "covid", "19"]
      - "TP53" -> ["tp53"]
      - "CD4+" -> ["cd4+", "cd4"]
    """
    if not text:
        return []

    tokens: List[str] = []
    for raw_tok in BIOMED_TOKEN_RE.findall(text.lower()):
        if raw_tok in ENGLISH_STOPWORDS:
            continue
        # Skip lone single-character non-alphanumeric or pure stopwords
        if len(raw_tok) == 1 and not raw_tok.isalnum():
            continue

        tokens.append(raw_tok)

        # Also add stripped form for trailing '+' (e.g., cd4+ -> cd4)
        base_tok = raw_tok[:-1] if raw_tok.endswith("+") else raw_tok
        if base_tok != raw_tok and base_tok and base_tok not in ENGLISH_STOPWORDS:
            tokens.append(base_tok)

        # For hyphenated or slashed compounds, emit component parts so both
        # "IL-6" and "IL 6" match cleanly.
        if "-" in base_tok or "/" in base_tok:
            for part in re.split(r"[-/]", base_tok):
                if part and part not in ENGLISH_STOPWORDS:
                    tokens.append(part)

    return tokens


class BM25Index:
    """Wrapper around `bm25s.BM25` with deterministic ID mapping and disk persistence."""

    def __init__(
        self,
        retriever: Optional[bm25s.BM25] = None,
        passage_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self.retriever = retriever
        self.passage_ids: List[str] = list(passage_ids or [])
        self.metadata: Dict[str, object] = dict(metadata or {})

    @property
    def size(self) -> int:
        return len(self.passage_ids)

    @property
    def id_fingerprint(self) -> str:
        return str(
            self.metadata.get("id_fingerprint")
            or compute_id_fingerprint(self.passage_ids)
        )

    def build(self, passages: Sequence[Passage], show_progress: bool = False) -> None:
        """Build a BM25 index over the provided canonical passages."""
        if not passages:
            raise ValueError("Cannot build BM25 index on an empty passage list.")

        self.passage_ids = [p.id for p in passages]
        tokenized_corpus = [tokenize_biomedical(p.text) for p in passages]

        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus, show_progress=show_progress)
        self.retriever = retriever
        self.metadata = {
            "index_type": "bm25s",
            "tokenizer": "biomedical_compound_v1",
            "num_passages": len(self.passage_ids),
            "id_fingerprint": compute_id_fingerprint(self.passage_ids),
        }

    def save(self, save_dir: Path) -> None:
        """Persist BM25 index, ordered ID map, and metadata to disk."""
        if self.retriever is None:
            raise RuntimeError("BM25 index has not been built yet.")

        save_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(save_dir), show_progress=False)

        with (save_dir / "id_map.json").open("w", encoding="utf-8") as f:
            json.dump(self.passage_ids, f)

        with (save_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, save_dir: Path) -> "BM25Index":
        """Load a persisted BM25 index and ID mapping from disk."""
        if not save_dir.exists():
            raise FileNotFoundError(f"BM25 index directory not found: {save_dir}")

        id_map_path = save_dir / "id_map.json"
        if not id_map_path.exists():
            raise FileNotFoundError(f"BM25 id_map.json not found in {save_dir}")

        with id_map_path.open("r", encoding="utf-8") as f:
            passage_ids: List[str] = json.load(f)

        meta_path = save_dir / "meta.json"
        metadata: Dict[str, object] = {}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

        retriever = bm25s.BM25.load(str(save_dir), load_corpus=False, show_progress=False)
        return cls(retriever=retriever, passage_ids=passage_ids, metadata=metadata)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the BM25 index and return ranked `(passage_id, score)` pairs with score > 0."""
        if self.retriever is None or not self.passage_ids:
            raise RuntimeError("BM25 index is not loaded or empty.")
        if top_k <= 0:
            return []

        query_tokens = tokenize_biomedical(query)
        if not query_tokens:
            return []

        k = min(top_k, len(self.passage_ids))
        doc_indices, scores = self.retriever.retrieve(
            [query_tokens],
            k=k,
            sorted=True,
            show_progress=False,
        )

        results: List[Tuple[str, float]] = []
        for idx, score in zip(doc_indices[0], scores[0]):
            score_val = float(score)
            if score_val <= 0.0:
                continue
            int_idx = int(idx)
            if 0 <= int_idx < len(self.passage_ids):
                results.append((self.passage_ids[int_idx], score_val))

        return results
