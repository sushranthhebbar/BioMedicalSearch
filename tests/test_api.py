"""Integration and contract tests for Phase 2 FastAPI endpoints and shared RAGPipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import RetrievalMode, Settings
from src.data_loader import Passage
from src.generation.answerer import GroundedAnswerer
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex
from src.pipeline import RAGPipeline
from src.retrieval.query_expansion import BiomedicalQueryExpander
from src.retrieval.reranker import FlashRankReranker
from src.retrieval.retriever import UnifiedRetriever
from tests.test_retrieval import (
    FakeFlashRank,
    FakeGeminiClient,
    KeywordVectorEmbedder,
    SAMPLE_PASSAGES,
)


class DynamicFakeAnswerClient:
    """Deterministic mock Gemini answerer switching between grounded and abstaining answers."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.models = self

    def generate_content(self, model: str, contents: str, config=None):
        self.calls.append(contents)
        lower = contents.lower()
        if "lava cake" in lower or "chocolate" in lower:
            payload = {
                "answer": "Insufficient evidence: the retrieved biomedical passages do not contain baking recipes.",
                "sufficient_evidence": False,
                "insufficient_reason": "Query is outside the indexed biomedical corpus.",
                "cited_ids": [],
            }
        else:
            payload = {
                "answer": (
                    "Tocilizumab targets interleukin-6 (IL-6) in rheumatoid arthritis and "
                    "severe COVID-19 [100002], whereas anti-TNF antibodies target joint inflammation [100005]."
                ),
                "sufficient_evidence": True,
                "insufficient_reason": None,
                "cited_ids": ["100002", "100005"],
            }
        return SimpleNamespace(
            parsed=SimpleNamespace(**payload),
            text=json.dumps(payload),
            usage_metadata=SimpleNamespace(prompt_token_count=210, candidates_token_count=48),
        )


def _build_test_client(tmp_path: Path) -> TestClient:
    bm25 = BM25Index()
    bm25.build(SAMPLE_PASSAGES)

    dense = DenseIndex(embedder=KeywordVectorEmbedder())
    dense.build(SAMPLE_PASSAGES)

    passage_map: Dict[str, Passage] = {p.id: p for p in SAMPLE_PASSAGES}
    settings = Settings(
        gemini_rpm_limit=0,
        data_dir=tmp_path / "data",
        eval_dir=tmp_path / "eval",
    )

    expander = BiomedicalQueryExpander(
        settings=settings,
        client=FakeGeminiClient(
            alternates=[
                "interleukin-6 cytokine tocilizumab",
                "IL-6 receptor antagonist rheumatoid arthritis",
                "anti-IL-6 therapy COVID-19",
            ]
        ),
        cache_path=tmp_path / "qe_cache.json",
    )
    reranker = FlashRankReranker(ranker=FakeFlashRank(boost_id="100002"))
    log_path = tmp_path / "data" / "logs" / "searches.jsonl"

    retriever = UnifiedRetriever(
        bm25_index=bm25,
        dense_index=dense,
        passage_map=passage_map,
        expander=expander,
        reranker=reranker,
        settings=settings,
        search_log_path=log_path,
    )
    answerer = GroundedAnswerer(settings=settings, client=DynamicFakeAnswerClient())
    pipeline = RAGPipeline(
        retriever=retriever,
        answerer=answerer,
        settings=settings,
        search_log_path=log_path,
    )
    app = create_app(pipeline=pipeline, settings=settings)
    return TestClient(app)


def test_health_config_and_static_ui_routes(tmp_path: Path):
    client = _build_test_client(tmp_path)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    health = r_health.json()
    assert health["ready"] is True
    assert health["corpus_size"] == len(SAMPLE_PASSAGES)
    assert health["bm25_ready"] is True
    assert health["faiss_ready"] is True

    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg = r_cfg.json()
    assert cfg["default_mode"] == RetrievalMode.HYBRID_QE_RERANK.value
    assert len(cfg["modes"]) == 5
    assert "min_score_thresholds" in cfg

    # Static UI files served cleanly
    for route in ("/", "/admin", "/styles.css", "/app.js", "/admin.js"):
        resp = client.get(route)
        assert resp.status_code == 200, f"Expected 200 for {route}, got {resp.status_code}"


def test_search_grounded_and_insufficient_evidence_and_history(tmp_path: Path):
    client = _build_test_client(tmp_path)

    # 1. Validation error on blank query
    r_bad = client.post("/api/search", json={"query": "   "})
    assert r_bad.status_code == 422

    # 2. Grounded answerable search
    r_ok = client.post(
        "/api/search",
        json={
            "query": "Which drug targets IL-6 in rheumatoid arthritis?",
            "mode": "hybrid_qe_rerank",
            "top_k": 5,
        },
    )
    assert r_ok.status_code == 200
    data = r_ok.json()
    assert data["sufficient_evidence"] is True
    assert data["mode"] == "hybrid_qe_rerank"
    assert len(data["expanded_queries"]) == 3
    assert data["all_queries"][0] == "Which drug targets IL-6 in rheumatoid arthritis?"
    assert data["citation_validation"]["is_valid"] is True
    assert data["citation_validation"]["zero_hallucinated_ids"] is True
    assert "100002" in data["citation_validation"]["valid_citations"]
    assert any(p["passage_id"] == "100002" and p["cited"] is True for p in data["passages"])
    assert data["usage"]["total_tokens"] > 0
    assert data["latency"]["total_ms"] >= 0

    search_id_1 = data["search_id"]

    # 3. Explicit insufficient-evidence search (pre-generation lexical threshold gate on unmatched terms)
    r_insuff = client.post(
        "/api/search",
        json={
            "query": "What is the recipe for chocolate lava cake?",
            "mode": "lexical",
            "top_k": 5,
        },
    )
    assert r_insuff.status_code == 200
    insuff = r_insuff.json()
    assert insuff["sufficient_evidence"] is False
    assert insuff["abstention_stage"] == "pre_generation"
    assert insuff["insufficient_reason"] is not None

    # 4. History listing & full hydration by search_id
    r_hist = client.get("/api/history")
    assert r_hist.status_code == 200
    hist = r_hist.json()
    assert hist["total"] == 2
    # Newest first
    assert hist["items"][0]["search_id"] == insuff["search_id"]
    assert hist["items"][1]["search_id"] == search_id_1

    r_item = client.get(f"/api/history/{search_id_1}")
    assert r_item.status_code == 200
    hydrated = r_item.json()
    assert hydrated["search_id"] == search_id_1
    assert hydrated["answer"] == data["answer"]
    assert len(hydrated["passages"]) == len(data["passages"])
    assert hydrated["passages"][0]["text"] != ""

    r_missing = client.get("/api/history/does_not_exist")
    assert r_missing.status_code == 404


def test_admin_stats_logs_and_eval_summary(tmp_path: Path):
    client = _build_test_client(tmp_path)

    client.post(
        "/api/search",
        json={"query": "How does tocilizumab target IL-6?", "mode": "hybrid_qe_rerank", "top_k": 5},
    )
    client.post(
        "/api/search",
        json={"query": "chocolate lava cake baking recipe", "mode": "lexical", "top_k": 5},
    )

    r_stats = client.get("/api/admin/stats")
    assert r_stats.status_code == 200
    stats = r_stats.json()
    assert stats["corpus_size"] == len(SAMPLE_PASSAGES)
    assert stats["metrics"]["total_searches"] == 2
    assert stats["metrics"]["grounded_count"] == 1
    assert stats["metrics"]["insufficient_count"] == 1
    assert stats["metrics"]["sufficient_evidence_rate"] == 0.5
    assert stats["metrics"]["avg_citation_validity_rate"] == 1.0

    # Filter logs by mode and sufficiency
    r_logs_grounded = client.get("/api/admin/logs?sufficient=true")
    assert r_logs_grounded.status_code == 200
    assert r_logs_grounded.json()["total"] == 1

    r_logs_lex = client.get("/api/admin/logs?mode=lexical")
    assert r_logs_lex.status_code == 200
    assert r_logs_lex.json()["total"] == 1

    # Eval summary before Phase 3/4 exists
    r_eval_empty = client.get("/api/admin/eval-summary")
    assert r_eval_empty.status_code == 200
    assert r_eval_empty.json()["available"] is False

    # Eval summary once summary_table.csv is created
    eval_results_dir = tmp_path / "eval" / "results"
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    (eval_results_dir / "summary_table.csv").write_text(
        "config,recall_at_5,recall_at_10,mrr_at_10\nlexical,0.55,0.68,0.51\nhybrid_qe_rerank,0.82,0.91,0.79\n",
        encoding="utf-8",
    )
    r_eval_ready = client.get("/api/admin/eval-summary")
    assert r_eval_ready.status_code == 200
    eval_data = r_eval_ready.json()
    assert eval_data["available"] is True
    assert len(eval_data["rows"]) == 2
    assert eval_data["rows"][1]["config"] == "hybrid_qe_rerank"
