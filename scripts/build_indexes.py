"""Offline ingestion and dual-index build script for `rag-datasets/rag-mini-bioasq`.

Builds both the BM25 (`bm25s`) and Dense (`fastembed` + `faiss-cpu`) indexes over
the exact same cleaned, canonically ordered passage corpus.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RetrievalMode, get_settings
from src.data_loader import load_hf_corpus, load_passages_jsonl, save_passages_jsonl
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex
from src.retrieval.retriever import UnifiedRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download rag-mini-bioasq corpus and build aligned BM25 + FAISS indexes."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of passages (for smoke testing only; omit for full corpus).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override dense embedding batch size.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override passage count between embedding checkpoints.",
    )
    parser.add_argument(
        "--reuse-passages",
        action="store_true",
        help="Reuse existing data/indexes/passages.jsonl instead of reloading from Hugging Face.",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip building BM25 index (assumes it already exists).",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip building Dense FAISS index (assumes it already exists).",
    )
    parser.add_argument(
        "--verify-query",
        type=str,
        default="Is Hirschsprung disease a mendelian or a multifactorial disorder?",
        help="Sample biomedical query used to verify built indexes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    settings.ensure_directories()

    t_all = time.perf_counter()

    # 1. Load & persist canonical corpus
    if args.reuse_passages and settings.passages_path.exists():
        logger.info("Reusing existing canonical passages from %s", settings.passages_path)
        passages = load_passages_jsonl(settings.passages_path)
    else:
        logger.info(
            "Loading Hugging Face dataset %s (%s / %s)...",
            settings.hf_dataset_name,
            settings.hf_corpus_config,
            settings.hf_corpus_split,
        )
        t0 = time.perf_counter()
        passages, stats = load_hf_corpus(settings=settings)
        logger.info(
            "Loaded and cleaned corpus in %.2fs | stats=%s",
            time.perf_counter() - t0,
            stats,
        )
        if args.limit is not None:
            passages = passages[: args.limit]
            logger.warning("Limiting corpus to first %d passages (--limit).", len(passages))

        save_passages_jsonl(passages, settings.passages_path)
        logger.info(
            "Saved %d canonical passages to %s", len(passages), settings.passages_path
        )

    # 2. Build & persist BM25 lexical index
    if not args.skip_bm25:
        logger.info("Building BM25 lexical index over %d passages...", len(passages))
        t0 = time.perf_counter()
        bm25 = BM25Index()
        bm25.build(passages, show_progress=True)
        bm25.save(settings.bm25_dir)
        logger.info(
            "Built & saved BM25 index to %s in %.2fs (fingerprint=%s...)",
            settings.bm25_dir,
            time.perf_counter() - t0,
            bm25.id_fingerprint[:12],
        )

    # 3. Build & persist Dense FAISS index (local CPU ONNX fastembed)
    if not args.skip_dense:
        logger.info(
            "Building Dense FAISS index with %s over %d passages...",
            settings.embedding_model,
            len(passages),
        )
        t0 = time.perf_counter()
        dense = DenseIndex(settings=settings)
        dense.build(
            passages,
            batch_size=args.batch_size or settings.embedding_batch_size,
            checkpoint_path=settings.embeddings_checkpoint_path,
            checkpoint_interval=args.checkpoint_interval
            or settings.embedding_checkpoint_interval,
            show_progress=True,
        )
        dense.save(settings.faiss_dir)
        logger.info(
            "Built & saved FAISS index (dim=%d) to %s in %.2fs (fingerprint=%s...)",
            dense.dimension,
            settings.faiss_dir,
            time.perf_counter() - t0,
            dense.id_fingerprint[:12],
        )

    # 4. Reload & verify strict 1-to-1 alignment across both indexes
    logger.info("Verifying disk load and 1-to-1 alignment between BM25 and FAISS...")
    retriever = UnifiedRetriever.from_disk(settings=settings)
    logger.info(
        "Alignment verified: %d passages | shared fingerprint=%s",
        retriever.corpus_size,
        retriever.id_fingerprint[:16],
    )

    if args.verify_query:
        for mode in (RetrievalMode.LEXICAL, RetrievalMode.DENSE, RetrievalMode.HYBRID):
            res = retriever.retrieve(
                args.verify_query,
                mode=mode,
                top_k=3,
                record_search=False,
            )
            top_ids = [(p.passage_id, round(p.score, 4)) for p in res.passages]
            logger.info(
                "Sanity [%s] (%.1f ms) -> top-3: %s",
                mode.value,
                res.latency.total_ms,
                top_ids,
            )

    logger.info("Index build completed in %.2fs.", time.perf_counter() - t_all)


if __name__ == "__main__":
    main()
