"""Lexical (BM25) and Dense Vector (FastEmbed + FAISS) indexing modules."""

from __future__ import annotations

import hashlib
from typing import Sequence


def compute_id_fingerprint(passage_ids: Sequence[str]) -> str:
    """Compute a deterministic SHA-256 fingerprint of ordered passage IDs."""
    hasher = hashlib.sha256()
    for pid in passage_ids:
        hasher.update(pid.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()
