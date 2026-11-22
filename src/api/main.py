"""FastAPI application for Biomedical Hybrid Search, Grounded RAG, and Admin Dashboard."""

from __future__ import annotations

import csv
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import PROJECT_ROOT, RetrievalMode, Settings, get_settings
from src.pipeline import PipelineResult, RAGPipeline

logger = logging.getLogger(__name__)

FRONTEND_DIR = PROJECT_ROOT / "frontend"

MODE_METADATA: List[Dict[str, Any]] = [
    {
        "id": RetrievalMode.LEXICAL.value,
        "label": "Lexical (BM25)",
        "short_label": "BM25",
        "description": "Exact biomedical keyword & compound-term matching via bm25s.",
        "uses_expansion": False,
        "uses_reranker": False,
    },
    {
        "id": RetrievalMode.DENSE.value,
        "label": "Dense Vector",
        "short_label": "Dense",
        "description": "Semantic cosine search via local ONNX BGE-small embeddings + FAISS.",
        "uses_expansion": False,
        "uses_reranker": False,
    },
    {
        "id": RetrievalMode.HYBRID.value,
        "label": "Hybrid (RRF)",
        "short_label": "Hybrid",
        "description": "Reciprocal Rank Fusion (k=60) combining BM25 and Dense rankings.",
        "uses_expansion": False,
        "uses_reranker": False,
    },
    {
        "id": RetrievalMode.HYBRID_QE.value,
        "label": "Hybrid + Query Expansion",
        "short_label": "Hybrid + QE",
        "description": "Weighted multi-query RRF over original query (2x) + 3 biomedical expansions.",
        "uses_expansion": True,
        "uses_reranker": False,
    },
    {
        "id": RetrievalMode.HYBRID_QE_RERANK.value,
        "label": "Hybrid + QE + Rerank",
        "short_label": "Hybrid + QE + Rerank",
        "description": "Strongest pipeline: top-30 Hybrid+QE candidates reranked by FlashRank ONNX cross-encoder.",
        "uses_expansion": True,
        "uses_reranker": True,
    },
]


class SearchRequest(BaseModel):
    """Payload accepted by `POST /api/search`."""

    query: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[RetrievalMode] = None
    top_k: int = Field(default=10, ge=1, le=50)
    answer_context_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_expansion_cache: bool = Field(default=True)

    @field_validator("query")
    @classmethod
    def _validate_non_blank(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("Query must not be empty or whitespace-only.")
        return cleaned


class HistoryItem(BaseModel):
    """Compact summary item rendered in the left search-history sidebar."""

    search_id: str
    timestamp: str
    original_query: str
    mode: str
    sufficient_evidence: Optional[bool] = None
    top_score: Optional[float] = None
    passage_count: int = 0
    valid_citation_count: int = 0
    total_ms: float = 0.0
    estimated_cost_usd: float = 0.0


class HistoryListResponse(BaseModel):
    total: int
    items: List[HistoryItem]


def _extract_top_score(record: Dict[str, Any]) -> Optional[float]:
    if record.get("top_retrieval_score") is not None:
        return float(record["top_retrieval_score"])
    passages = record.get("retrieved_passages") or []
    if passages and isinstance(passages[0], dict) and passages[0].get("score") is not None:
        return float(passages[0]["score"])
    return None


def _extract_latency_ms(record: Dict[str, Any]) -> Dict[str, float]:
    lat = record.get("latency_ms") or {}
    retrieval_total = float(
        lat.get("retrieval_total_ms", lat.get("total_ms", 0.0)) or 0.0
    )
    generation_ms = float(lat.get("generation_ms", 0.0) or 0.0)
    total_ms = float(lat.get("total_ms", retrieval_total + generation_ms) or 0.0)
    return {
        "expansion_ms": float(lat.get("expansion_ms", 0.0) or 0.0),
        "lexical_ms": float(lat.get("lexical_ms", 0.0) or 0.0),
        "dense_ms": float(lat.get("dense_ms", 0.0) or 0.0),
        "fusion_ms": float(lat.get("fusion_ms", 0.0) or 0.0),
        "rerank_ms": float(lat.get("rerank_ms", 0.0) or 0.0),
        "retrieval_total_ms": round(retrieval_total, 2),
        "generation_ms": round(generation_ms, 2),
        "total_ms": round(total_ms, 2),
    }


def _extract_usage(record: Dict[str, Any]) -> Dict[str, Any]:
    u = record.get("usage") or {}
    exp_t = int(u.get("expansion_total_tokens", 0) or 0)
    gen_t = int(u.get("generation_total_tokens", 0) or 0)
    return {
        "expansion_total_tokens": exp_t,
        "generation_total_tokens": gen_t,
        "total_tokens": int(u.get("total_tokens", exp_t + gen_t) or 0),
        "estimated_cost_usd": float(u.get("estimated_cost_usd", 0.0) or 0.0),
        "expansion_cached": bool(u.get("expansion_cached", False)),
        "expansion_fallback_used": bool(u.get("expansion_fallback_used", False)),
        "repair_attempted": bool(u.get("repair_attempted", False)),
    }


def _load_eval_summary(eval_dir: Path) -> Dict[str, Any]:
    """Load offline 100-query evaluation summary JSON or CSV if available."""
    results_dir = eval_dir / "results"
    json_path = results_dir / "summary_table.json"
    csv_path = results_dir / "summary_table.csv"

    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rows = data if isinstance(data, list) else data.get("rows", [])
            return {
                "available": True,
                "source": str(json_path.relative_to(PROJECT_ROOT))
                if json_path.is_relative_to(PROJECT_ROOT)
                else str(json_path),
                "rows": rows,
            }
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", json_path, exc)

    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(row) for row in reader]
            return {
                "available": True,
                "source": str(csv_path.relative_to(PROJECT_ROOT))
                if csv_path.is_relative_to(PROJECT_ROOT)
                else str(csv_path),
                "rows": rows,
            }
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", csv_path, exc)

    return {
        "available": False,
        "source": None,
        "rows": [],
        "message": (
            "Offline 100-query benchmark results have not been generated yet. "
            "Run Phase 3/4 evaluation (scripts/run_eval.py) to populate this table."
        ),
    }


def create_app(
    pipeline: Optional[RAGPipeline] = None,
    settings: Optional[Settings] = None,
    frontend_dir: Optional[Path] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = settings or (pipeline.settings if pipeline is not None else get_settings())
    ui_dir = frontend_dir or FRONTEND_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "pipeline", None) is None:
            logger.info("Loading BM25, FAISS, and canonical passage store from disk...")
            app.state.pipeline = RAGPipeline.from_disk(settings=cfg)
            logger.info(
                "RAGPipeline ready (%d passages indexed).",
                app.state.pipeline.retriever.corpus_size,
            )
        yield

    app = FastAPI(
        title="Biomedical Hybrid Search & RAG",
        description="Hybrid BM25 + FAISS + Query Expansion + FlashRank Cross-Encoder RAG over rag-mini-bioasq.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.pipeline = pipeline

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_pipeline(request: Request) -> RAGPipeline:
        pipe: Optional[RAGPipeline] = getattr(request.app.state, "pipeline", None)
        if pipe is None:
            raise HTTPException(
                status_code=503,
                detail="Retrieval indexes are still loading or unavailable.",
            )
        return pipe

    # -----------------------------------------------------------------------
    # Core Search & History API
    # -----------------------------------------------------------------------

    @app.get("/api/health")
    def get_health(request: Request) -> Dict[str, Any]:
        pipe: Optional[RAGPipeline] = getattr(request.app.state, "pipeline", None)
        if pipe is None:
            return {
                "status": "initializing",
                "ready": False,
                "corpus_size": 0,
                "bm25_ready": False,
                "faiss_ready": False,
                "id_fingerprint": None,
                "default_mode": cfg.default_retrieval_mode.value,
            }
        ret = pipe.retriever
        return {
            "status": "ok",
            "ready": True,
            "corpus_size": ret.corpus_size,
            "bm25_ready": ret.bm25_index.retriever is not None,
            "faiss_ready": ret.dense_index.index is not None,
            "id_fingerprint": ret.id_fingerprint,
            "default_mode": pipe.settings.default_retrieval_mode.value,
        }

    @app.get("/api/config")
    def get_config(request: Request) -> Dict[str, Any]:
        active_cfg = getattr(request.app.state, "settings", cfg)
        return {
            "default_mode": active_cfg.default_retrieval_mode.value,
            "modes": MODE_METADATA,
            "default_top_k": active_cfg.default_top_k,
            "default_answer_context_k": active_cfg.default_answer_context_k,
            "min_score_thresholds": active_cfg.min_score_thresholds,
            "models": {
                "embedding_model": active_cfg.embedding_model,
                "reranker_model": active_cfg.reranker_model,
                "expansion_model": active_cfg.gemini_expansion_model,
                "generation_model": active_cfg.gemini_generation_model,
                "judge_model": active_cfg.gemini_judge_model,
            },
        }

    @app.post("/api/search", response_model=PipelineResult)
    def post_search(payload: SearchRequest, request: Request) -> PipelineResult:
        pipe = _get_pipeline(request)
        return pipe.run(
            query=payload.query,
            mode=payload.mode or pipe.settings.default_retrieval_mode,
            top_k=payload.top_k,
            answer_context_k=payload.answer_context_k,
            generate_answer=True,
            record_search=True,
            use_expansion_cache=payload.use_expansion_cache,
        )

    @app.get("/api/history", response_model=HistoryListResponse)
    def get_history(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        q: Optional[str] = Query(default=None),
    ) -> HistoryListResponse:
        pipe = _get_pipeline(request)
        records = list(reversed(pipe.load_search_logs()))

        if q and q.strip():
            needle = q.strip().lower()
            records = [
                r
                for r in records
                if needle in str(r.get("original_query", "")).lower()
                or needle in str(r.get("mode", "")).lower()
            ]

        total = len(records)
        items: List[HistoryItem] = []
        for rec in records[:limit]:
            lat = _extract_latency_ms(rec)
            usage = _extract_usage(rec)
            cit = rec.get("citation_validation") or {}
            valid_cits = cit.get("valid_citations") or []
            items.append(
                HistoryItem(
                    search_id=str(rec.get("search_id", "")),
                    timestamp=str(rec.get("timestamp", "")),
                    original_query=str(rec.get("original_query", "")),
                    mode=str(rec.get("mode", "")),
                    sufficient_evidence=rec.get("sufficient_evidence"),
                    top_score=_extract_top_score(rec),
                    passage_count=len(rec.get("retrieved_passages") or []),
                    valid_citation_count=len(valid_cits),
                    total_ms=lat["total_ms"],
                    estimated_cost_usd=usage["estimated_cost_usd"],
                )
            )

        return HistoryListResponse(total=total, items=items)

    @app.get("/api/history/{search_id}", response_model=PipelineResult)
    def get_history_item(search_id: str, request: Request) -> PipelineResult:
        pipe = _get_pipeline(request)
        for rec in reversed(pipe.load_search_logs()):
            if str(rec.get("search_id", "")) == search_id:
                return pipe.hydrate_history_record(rec)
        raise HTTPException(
            status_code=404, detail=f"Search history item '{search_id}' not found."
        )

    # -----------------------------------------------------------------------
    # Admin & Observability API
    # -----------------------------------------------------------------------

    @app.get("/api/admin/stats")
    def get_admin_stats(request: Request) -> Dict[str, Any]:
        pipe = _get_pipeline(request)
        ret = pipe.retriever
        active_cfg = pipe.settings
        records = pipe.load_search_logs()

        total_searches = len(records)
        grounded_count = 0
        insufficient_count = 0
        retrieval_only_count = 0
        total_ms_sum = 0.0
        ret_ms_sum = 0.0
        gen_ms_sum = 0.0
        gen_latency_count = 0
        total_tokens = 0
        total_cost = 0.0
        cit_rates: List[float] = []
        zero_halluc_count = 0
        cit_checked_count = 0
        mode_counts: Dict[str, int] = {m.value: 0 for m in RetrievalMode}

        for rec in records:
            m = str(rec.get("mode", ""))
            mode_counts[m] = mode_counts.get(m, 0) + 1

            suff = rec.get("sufficient_evidence")
            if suff is True:
                grounded_count += 1
            elif suff is False:
                insufficient_count += 1
            else:
                retrieval_only_count += 1

            lat = _extract_latency_ms(rec)
            total_ms_sum += lat["total_ms"]
            ret_ms_sum += lat["retrieval_total_ms"]
            if lat["generation_ms"] > 0 or suff is not None:
                gen_ms_sum += lat["generation_ms"]
                gen_latency_count += 1

            usage = _extract_usage(rec)
            total_tokens += usage["total_tokens"]
            total_cost += usage["estimated_cost_usd"]

            cit = rec.get("citation_validation")
            if isinstance(cit, dict):
                cit_checked_count += 1
                if cit.get("citation_validity_rate") is not None:
                    cit_rates.append(float(cit["citation_validity_rate"]))
                if cit.get("zero_hallucinated_ids", True):
                    zero_halluc_count += 1

        answered_total = grounded_count + insufficient_count
        return {
            "corpus_size": ret.corpus_size,
            "id_fingerprint": ret.id_fingerprint,
            "bm25_metadata": ret.bm25_index.metadata,
            "faiss_metadata": ret.dense_index.metadata,
            "default_mode": active_cfg.default_retrieval_mode.value,
            "default_top_k": active_cfg.default_top_k,
            "default_answer_context_k": active_cfg.default_answer_context_k,
            "min_score_thresholds": active_cfg.min_score_thresholds,
            "models": {
                "embedding_model": active_cfg.embedding_model,
                "reranker_model": active_cfg.reranker_model,
                "expansion_model": active_cfg.gemini_expansion_model,
                "generation_model": active_cfg.gemini_generation_model,
                "judge_model": active_cfg.gemini_judge_model,
            },
            "metrics": {
                "total_searches": total_searches,
                "grounded_count": grounded_count,
                "insufficient_count": insufficient_count,
                "retrieval_only_count": retrieval_only_count,
                "sufficient_evidence_rate": (
                    round(grounded_count / float(answered_total), 4)
                    if answered_total > 0
                    else None
                ),
                "avg_total_ms": (
                    round(total_ms_sum / float(total_searches), 2)
                    if total_searches > 0
                    else 0.0
                ),
                "avg_retrieval_ms": (
                    round(ret_ms_sum / float(total_searches), 2)
                    if total_searches > 0
                    else 0.0
                ),
                "avg_generation_ms": (
                    round(gen_ms_sum / float(gen_latency_count), 2)
                    if gen_latency_count > 0
                    else 0.0
                ),
                "total_tokens": total_tokens,
                "total_estimated_cost_usd": round(total_cost, 6),
                "avg_citation_validity_rate": (
                    round(sum(cit_rates) / float(len(cit_rates)), 4)
                    if cit_rates
                    else None
                ),
                "zero_hallucinated_rate": (
                    round(zero_halluc_count / float(cit_checked_count), 4)
                    if cit_checked_count > 0
                    else None
                ),
                "mode_counts": mode_counts,
            },
        }

    @app.get("/api/admin/logs")
    def get_admin_logs(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        mode: Optional[str] = Query(default=None),
        sufficient: Optional[str] = Query(default=None),
        q: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        pipe = _get_pipeline(request)
        records = list(reversed(pipe.load_search_logs()))

        if mode and mode.strip() and mode.strip().lower() != "all":
            target_mode = mode.strip().lower()
            records = [r for r in records if str(r.get("mode", "")).lower() == target_mode]

        if sufficient and sufficient.strip().lower() in ("true", "false"):
            want = sufficient.strip().lower() == "true"
            records = [r for r in records if r.get("sufficient_evidence") is want]

        if q and q.strip():
            needle = q.strip().lower()
            records = [
                r
                for r in records
                if needle in str(r.get("original_query", "")).lower()
                or needle in str(r.get("search_id", "")).lower()
                or any(needle in str(e).lower() for e in (r.get("expanded_queries") or []))
            ]

        total = len(records)
        page = records[offset : offset + limit]

        enriched: List[Dict[str, Any]] = []
        for rec in page:
            enriched.append(
                {
                    "search_id": rec.get("search_id"),
                    "timestamp": rec.get("timestamp"),
                    "mode": rec.get("mode"),
                    "original_query": rec.get("original_query"),
                    "expanded_queries": rec.get("expanded_queries") or [],
                    "top_score": _extract_top_score(rec),
                    "threshold_used": rec.get("threshold_used"),
                    "sufficient_evidence": rec.get("sufficient_evidence"),
                    "insufficient_reason": rec.get("insufficient_reason"),
                    "abstention_stage": rec.get("abstention_stage"),
                    "answer": rec.get("answer"),
                    "citation_validation": rec.get("citation_validation"),
                    "retrieved_passages": rec.get("retrieved_passages") or [],
                    "latency_ms": _extract_latency_ms(rec),
                    "usage": _extract_usage(rec),
                }
            )

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": enriched,
        }

    @app.get("/api/admin/eval-summary")
    def get_admin_eval_summary(request: Request) -> Dict[str, Any]:
        active_cfg = getattr(request.app.state, "settings", cfg)
        return _load_eval_summary(active_cfg.eval_dir)

    # -----------------------------------------------------------------------
    # Static Frontend Serving (Main UI + Admin UI)
    # -----------------------------------------------------------------------

    def _serve_frontend_file(filename: str):
        path = ui_dir / filename
        if not path.exists() or not path.is_file():
            return JSONResponse(
                status_code=404,
                content={"detail": f"Frontend asset '{filename}' not found."},
            )
        return FileResponse(path)

    @app.get("/", include_in_schema=False)
    def serve_index():
        return _serve_frontend_file("index.html")

    @app.get("/admin", include_in_schema=False)
    def serve_admin():
        return _serve_frontend_file("admin.html")

    @app.get("/admin.html", include_in_schema=False)
    def serve_admin_html():
        return _serve_frontend_file("admin.html")

    @app.get("/index.html", include_in_schema=False)
    def serve_index_html():
        return _serve_frontend_file("index.html")

    @app.get("/styles.css", include_in_schema=False)
    def serve_styles():
        return _serve_frontend_file("styles.css")

    @app.get("/app.js", include_in_schema=False)
    def serve_app_js():
        return _serve_frontend_file("app.js")

    @app.get("/admin.js", include_in_schema=False)
    def serve_admin_js():
        return _serve_frontend_file("admin.js")

    return app


app = create_app()
