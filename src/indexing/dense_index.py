"""Local ONNX FastEmbed + FAISS dense vector index with resumable checkpointing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from src.config import Settings, get_settings
from src.data_loader import Passage
from src.indexing import compute_id_fingerprint

logger = logging.getLogger(__name__)


class DenseEmbedder(Protocol):
    """Protocol for dense text embedders (allows fast mock embedders in unit tests)."""

    model_name: str

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 matrix of shape (len(texts), dim)."""
        ...

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Return float32 matrix of shape (len(queries), dim)."""
        ...


class FastEmbedWrapper:
    """Lazy-loading wrapper around `fastembed.TextEmbedding` (local CPU ONNX)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[Path] = None,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            kwargs = {"model_name": self.model_name}
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                kwargs["cache_dir"] = str(self.cache_dir)
            self._model = TextEmbedding(**kwargs)
        return self._model

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._get_model()
        vectors = list(model.passage_embed(texts, batch_size=self.batch_size))
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0), dtype=np.float32)
        model = self._get_model()
        vectors = list(model.query_embed(queries))
        return np.asarray(vectors, dtype=np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return a float32 C-contiguous L2-normalized copy of vectors."""
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


class DenseIndex:
    """FAISS `IndexFlatIP` cosine-similarity index backed by local ONNX FastEmbed."""

    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        passage_ids: Optional[List[str]] = None,
        embedder: Optional[DenseEmbedder] = None,
        metadata: Optional[Dict[str, object]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index = index
        self.passage_ids: List[str] = list(passage_ids or [])
        self.metadata: Dict[str, object] = dict(metadata or {})
        self.embedder: DenseEmbedder = embedder or FastEmbedWrapper(
            model_name=str(
                self.metadata.get("embedding_model") or self.settings.embedding_model
            ),
            cache_dir=self.settings.fastembed_cache_dir,
            batch_size=self.settings.embedding_batch_size,
        )
        self._query_cache: Dict[str, np.ndarray] = {}

    @property
    def size(self) -> int:
        return len(self.passage_ids)

    @property
    def dimension(self) -> int:
        if self.index is not None:
            return int(self.index.d)
        return int(self.metadata.get("dimension", 0))

    @property
    def id_fingerprint(self) -> str:
        return str(
            self.metadata.get("id_fingerprint")
            or compute_id_fingerprint(self.passage_ids)
        )

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        expected_ids: Sequence[str],
    ) -> Tuple[int, Optional[np.ndarray]]:
        """Load a valid resumable embedding checkpoint if it matches the target corpus & model."""
        if not checkpoint_path.exists():
            return 0, None

        try:
            with np.load(checkpoint_path, allow_pickle=False) as data:
                saved_model = str(data["model_name"].item())
                saved_ids = data["passage_ids"].tolist()
                saved_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        except Exception as exc:
            logger.warning("Ignoring unreadable embedding checkpoint %s: %s", checkpoint_path, exc)
            return 0, None

        if saved_model != self.embedder.model_name:
            logger.info(
                "Checkpoint model (%s) differs from configured model (%s); rebuilding from scratch.",
                saved_model,
                self.embedder.model_name,
            )
            return 0, None

        n_saved = len(saved_ids)
        if (
            n_saved == 0
            or n_saved > len(expected_ids)
            or saved_embeddings.shape[0] != n_saved
            or saved_ids != list(expected_ids[:n_saved])
        ):
            logger.info("Checkpoint passage IDs do not match current corpus prefix; rebuilding from scratch.")
            return 0, None

        logger.info("Resuming dense embeddings from checkpoint: %d / %d passages.", n_saved, len(expected_ids))
        return n_saved, saved_embeddings

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        passage_ids: Sequence[str],
        embeddings: np.ndarray,
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(".tmp.npz")
        np.savez(
            tmp_path,
            model_name=np.array(self.embedder.model_name),
            passage_ids=np.array(list(passage_ids), dtype=str),
            embeddings=embeddings,
        )
        tmp_path.replace(checkpoint_path)

    def build(
        self,
        passages: Sequence[Passage],
        batch_size: Optional[int] = None,
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:
        """Embed passages in batches (with optional disk checkpointing) and build `IndexFlatIP`."""
        if not passages:
            raise ValueError("Cannot build dense FAISS index on an empty passage list.")

        bs = batch_size or self.settings.embedding_batch_size
        ckpt_every = checkpoint_interval or self.settings.embedding_checkpoint_interval
        target_ids = [p.id for p in passages]
        total = len(passages)

        start_idx = 0
        chunks: List[np.ndarray] = []

        if checkpoint_path is not None:
            start_idx, cached_emb = self._load_checkpoint(checkpoint_path, target_ids)
            if cached_emb is not None and start_idx > 0:
                chunks.append(cached_emb)

        since_ckpt = 0
        iterator: Iterable[int] = range(start_idx, total, bs)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Embedding ({self.embedder.model_name})",
                initial=start_idx // bs,
                total=(total + bs - 1) // bs,
            )

        for batch_start in iterator:
            batch_end = min(batch_start + bs, total)
            batch_texts = [p.text for p in passages[batch_start:batch_end]]
            batch_vecs = self.embedder.embed_passages(batch_texts)
            batch_vecs = l2_normalize(batch_vecs)
            chunks.append(batch_vecs)
            since_ckpt += batch_end - batch_start

            if checkpoint_path is not None and (since_ckpt >= ckpt_every or batch_end == total):
                merged = np.vstack(chunks)
                self._save_checkpoint(checkpoint_path, target_ids[:batch_end], merged)
                chunks = [merged]
                since_ckpt = 0

        all_embeddings = np.ascontiguousarray(
            chunks[0] if len(chunks) == 1 else np.vstack(chunks),
            dtype=np.float32,
        )
        all_embeddings = l2_normalize(all_embeddings)

        dim = int(all_embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(all_embeddings)

        self.index = index
        self.passage_ids = target_ids
        self.metadata = {
            "index_type": "faiss.IndexFlatIP",
            "embedding_model": self.embedder.model_name,
            "dimension": dim,
            "normalized": True,
            "num_passages": len(self.passage_ids),
            "id_fingerprint": compute_id_fingerprint(self.passage_ids),
        }

    def save(self, save_dir: Path) -> None:
        """Persist FAISS index, ordered ID map, and metadata to disk."""
        if self.index is None:
            raise RuntimeError("FAISS index has not been built yet.")

        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_dir / "index.faiss"))

        with (save_dir / "id_map.json").open("w", encoding="utf-8") as f:
            json.dump(self.passage_ids, f)

        with (save_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        save_dir: Path,
        embedder: Optional[DenseEmbedder] = None,
        settings: Optional[Settings] = None,
    ) -> "DenseIndex":
        """Load persisted FAISS index and ID mapping from disk."""
        index_path = save_dir / "index.faiss"
        id_map_path = save_dir / "id_map.json"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not id_map_path.exists():
            raise FileNotFoundError(f"FAISS id_map.json not found: {id_map_path}")

        index = faiss.read_index(str(index_path))
        with id_map_path.open("r", encoding="utf-8") as f:
            passage_ids: List[str] = json.load(f)

        meta_path = save_dir / "meta.json"
        metadata: Dict[str, object] = {}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(
            index=index,
            passage_ids=passage_ids,
            embedder=embedder,
            metadata=metadata,
            settings=settings,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed and L2-normalize a single query (with in-memory caching)."""
        key = query.strip()
        if key not in self._query_cache:
            vec = self.embedder.embed_queries([key])
            self._query_cache[key] = l2_normalize(vec)
        return self._query_cache[key]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search the FAISS index by cosine similarity and return `(passage_id, score)` pairs."""
        if self.index is None or not self.passage_ids:
            raise RuntimeError("FAISS index is not loaded or empty.")
        if top_k <= 0 or not query or not query.strip():
            return []

        q_vec = self.embed_query(query)
        k = min(top_k, len(self.passage_ids))
        scores, indices = self.index.search(q_vec, k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            int_idx = int(idx)
            if int_idx < 0 or int_idx >= len(self.passage_ids):
                continue
            results.append((self.passage_ids[int_idx], float(score)))
        return results
