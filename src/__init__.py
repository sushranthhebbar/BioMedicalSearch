"""Biomedical Hybrid Search & RAG Application."""

# On Windows, loading `faiss` before `scipy`/`bm25s` avoids OpenMP DLL initialization
# conflicts when both lexical and dense indexes are imported in the same process.
try:
    import faiss as _faiss  # noqa: F401
except Exception:
    pass
