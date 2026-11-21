"""Unit tests for Phase 1: data loading, BM25, FAISS, RRF fusion, query expansion, reranker, and retriever."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import pytest

from src.config import RetrievalMode, Settings
from src.data_loader import (
    Passage,
    build_pubmed_url,
    clean_corpus_records,
    clean_question_records,
    load_passages_jsonl,
    normalize_id,
    parse_relevant_passage_ids,
    save_passages_jsonl,
)
from src.indexing.bm25_index import BM25Index, tokenize_biomedical
from src.indexing.dense_index import DenseIndex
from src.retrieval.fusion import RankedList, reciprocal_rank_fusion
from src.retrieval.query_expansion import BiomedicalQueryExpander, deduplicate_queries
from src.retrieval.reranker import FlashRankReranker
from src.retrieval.retriever import UnifiedRetriever


# ---------------------------------------------------------------------------
# Helpers / Mocks for fast deterministic offline tests
# ---------------------------------------------------------------------------

SAMPLE_PASSAGES = [
    Passage(
        id="100001",
        text="Hirschsprung disease is a multifactorial genetic disorder involving RET proto-oncogene mutations affecting enteric ganglia.",
        pubmed_url=build_pubmed_url("100001"),
    ),
    Passage(
        id="100002",
        text="Interleukin-6 (IL-6) is a pro-inflammatory cytokine targeted by tocilizumab in rheumatoid arthritis and severe COVID-19.",
        pubmed_url=build_pubmed_url("100002"),
    ),
    Passage(
        id="100003",
        text="TP53 encodes the p53 tumor suppressor protein regulating cell cycle arrest, DNA repair, and apoptosis.",
        pubmed_url=build_pubmed_url("100003"),
    ),
    Passage(
        id="100004",
        text="Metformin lowers hepatic gluconeogenesis via AMPK activation and is first-line therapy for type 2 diabetes mellitus.",
        pubmed_url=build_pubmed_url("100004"),
    ),
    Passage(
        id="100005",
        text="Anti-TNF monoclonal antibodies such as infliximab and adalimumab reduce joint inflammation in Crohn disease and arthritis.",
        pubmed_url=build_pubmed_url("100005"),
    ),
]


class KeywordVectorEmbedder:
    """Deterministic bag-of-concepts embedder for fast offline FAISS tests."""

    model_name = "mock-biomed-embedder"
    CONCEPTS = [
        ("hirschsprung", "ret", "enteric", "multifactorial", "ganglia"),
        ("il-6", "interleukin", "cytokine", "tocilizumab", "covid-19", "inflammation"),
        ("tp53", "p53", "tumor", "apoptosis", "suppressor", "cancer"),
        ("metformin", "ampk", "diabetes", "gluconeogenesis", "insulin"),
        ("anti-tnf", "infliximab", "adalimumab", "arthritis", "crohn"),
    ]

    def _encode_one(self, text: str) -> np.ndarray:
        lower = text.lower()
        vec = np.zeros(len(self.CONCEPTS) + 1, dtype=np.float32)
        for idx, terms in enumerate(self.CONCEPTS):
            vec[idx] = float(sum(1 for t in terms if t in lower))
        vec[-1] = 0.05  # tiny non-zero floor so norm > 0
        return vec

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._encode_one(t) for t in texts])

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        return np.vstack([self._encode_one(q) for q in queries])


class FakeGeminiClient:
    """Mock `google.genai.Client` returning deterministic structured query expansions."""

    def __init__(self, alternates: List[str], should_fail: bool = False) -> None:
        self.alternates = alternates
        self.should_fail = should_fail
        self.call_count = 0
        self.models = self

    def generate_content(self, model: str, contents: str, config=None):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("Simulated API outage")
        return SimpleNamespace(
            parsed=SimpleNamespace(alternate_queries=self.alternates),
            text=json.dumps({"alternate_queries": self.alternates}),
            usage_metadata=SimpleNamespace(
                prompt_token_count=42,
                candidates_token_count=28,
            ),
        )


class FakeFlashRank:
    """Mock FlashRank Ranker that scores passages by exact overlap with a target keyword."""

    def __init__(self, boost_id: str) -> None:
        self.boost_id = boost_id

    def rerank(self, request):
        out = []
        for idx, p in enumerate(request.passages):
            pid = str(p["id"])
            score = 0.99 if pid == self.boost_id else round(0.5 - idx * 0.05, 4)
            out.append({"id": pid, "text": p["text"], "score": score})
        return out


# ---------------------------------------------------------------------------
# 1. Data Loading & Normalization Tests
# ---------------------------------------------------------------------------


def test_data_normalization_and_cleaning(tmp_path: Path):
    assert normalize_id(123456) == "123456"
    assert normalize_id(123456.0) == "123456"
    assert normalize_id(" 123456.0 ") == "123456"
    assert normalize_id("nan") is None
    assert normalize_id(float("nan")) is None
    assert normalize_id("") is None

    assert build_pubmed_url("23151010") == "https://pubmed.ncbi.nlm.nih.gov/23151010/"
    assert build_pubmed_url("non_numeric_id") is None

    assert parse_relevant_passage_ids("[12345, '67890', 12345]") == ["12345", "67890"]
    assert parse_relevant_passage_ids("['111', 222.0]") == ["111", "222"]
    assert parse_relevant_passage_ids("nan") == []

    raw_records = [
        {"id": 1001, "passage": "Valid biomedical   passage about TP53."},
        {"id": "1001", "passage": "Duplicate ID should be dropped."},
        {"id": 1002, "passage": "   "},
        {"id": 1003, "passage": "nan"},
        {"id": None, "passage": "Missing ID"},
        {"id": "1004.0", "passage": "Second valid passage."},
    ]
    passages, stats = clean_corpus_records(raw_records)
    assert [p.id for p in passages] == ["1001", "1004"]
    assert passages[0].text == "Valid biomedical passage about TP53."
    assert passages[0].pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/1001/"
    assert stats["kept"] == 2
    assert stats["dropped_duplicate_id"] == 1
    assert stats["dropped_empty_text"] == 2
    assert stats["dropped_invalid_id"] == 1

    # Round-trip JSONL persistence
    jsonl_path = tmp_path / "passages.jsonl"
    save_passages_jsonl(passages, jsonl_path)
    reloaded = load_passages_jsonl(jsonl_path)
    assert reloaded == passages

    # Question cleaning against valid indexed passage IDs
    raw_questions = [
        {
            "id": 1,
            "question": "What does TP53 encode?",
            "answer": "p53 protein.",
            "relevant_passage_ids": "[1001, 9999]",
        },
        {
            "id": 2,
            "question": "Unindexed question?",
            "answer": "None",
            "relevant_passage_ids": "[8888]",
        },
    ]
    questions, qstats = clean_question_records(raw_questions, valid_passage_ids={"1001", "1004"})
    assert len(questions) == 1
    assert questions[0].relevant_passage_ids == ["1001"]
    assert qstats["dropped_no_indexed_gold"] == 1


# ---------------------------------------------------------------------------
# 2. Biomedical Tokenization & BM25 Tests
# ---------------------------------------------------------------------------


def test_biomedical_tokenization():
    tokens = tokenize_biomedical("The role of IL-6, TP53, CD4+ cells, and COVID-19 in ARDS.")
    assert "il-6" in tokens
    assert "il" in tokens
    assert "6" in tokens
    assert "tp53" in tokens
    assert "covid-19" in tokens
    assert "cd4+" in tokens
    assert "cd4" in tokens
    assert "the" not in tokens
    assert "of" not in tokens


def test_bm25_build_save_load_and_search(tmp_path: Path):
    bm25 = BM25Index()
    bm25.build(SAMPLE_PASSAGES)
    save_dir = tmp_path / "bm25"
    bm25.save(save_dir)

    loaded = BM25Index.load(save_dir)
    assert loaded.size == len(SAMPLE_PASSAGES)
    assert loaded.id_fingerprint == bm25.id_fingerprint

    hits = loaded.search("IL-6 tocilizumab COVID-19", top_k=3)
    assert len(hits) >= 1
    assert hits[0][0] == "100002"
    assert all(score > 0.0 for _, score in hits)

    # Empty or stopword-only query returns [] cleanly
    assert loaded.search("the of and", top_k=5) == []


# ---------------------------------------------------------------------------
# 3. Dense FAISS Index Tests (with checkpointing & round-trip)
# ---------------------------------------------------------------------------


def test_dense_index_build_checkpoint_save_load_and_search(tmp_path: Path):
    embedder = KeywordVectorEmbedder()
    ckpt_path = tmp_path / "embeddings_checkpoint.npz"
    faiss_dir = tmp_path / "faiss"

    # Build first with small checkpoint interval to exercise checkpoint saving
    dense = DenseIndex(embedder=embedder)
    dense.build(
        SAMPLE_PASSAGES,
        batch_size=2,
        checkpoint_path=ckpt_path,
        checkpoint_interval=2,
    )
    assert ckpt_path.exists()
    dense.save(faiss_dir)

    # Rebuild using the existing checkpoint (verifying resume path)
    dense_resumed = DenseIndex(embedder=embedder)
    dense_resumed.build(
        SAMPLE_PASSAGES,
        batch_size=2,
        checkpoint_path=ckpt_path,
        checkpoint_interval=2,
    )
    assert dense_resumed.id_fingerprint == dense.id_fingerprint

    # Reload from disk and verify cosine search
    loaded = DenseIndex.load(faiss_dir, embedder=embedder)
    assert loaded.size == len(SAMPLE_PASSAGES)
    assert loaded.dimension == len(embedder.CONCEPTS) + 1

    hits = loaded.search("metformin AMPK diabetes therapy", top_k=3)
    assert hits[0][0] == "100004"
    assert -1.01 <= hits[0][1] <= 1.01


# ---------------------------------------------------------------------------
# 4. RRF & Weighted Multi-Query Fusion Tests
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_math_and_weights():
    r1 = RankedList(
        source="lexical",
        query="original query",
        results=[("A", 10.0), ("B", 5.0)],
        weight=2.0,
    )
    r2 = RankedList(
        source="dense",
        query="original query",
        results=[("B", 0.9), ("C", 0.8)],
        weight=2.0,
    )
    r3 = RankedList(
        source="dense",
        query="expanded query",
        results=[("C", 0.95), ("A", 0.7)],
        weight=1.0,
    )

    fused = reciprocal_rank_fusion([r1, r2, r3], k=60)
    scores = {c.passage_id: c.rrf_score for c in fused}

    expected_a = 2.0 / 61.0 + 1.0 / 62.0
    expected_b = 2.0 / 62.0 + 2.0 / 61.0
    expected_c = 2.0 / 62.0 + 1.0 / 61.0

    assert math.isclose(scores["A"], expected_a, rel_tol=1e-9)
    assert math.isclose(scores["B"], expected_b, rel_tol=1e-9)
    assert math.isclose(scores["C"], expected_c, rel_tol=1e-9)
    # B appears in both original lexical & dense lists with weight 2.0 -> must rank #1
    assert fused[0].passage_id == "B"
    assert fused[0].bm25_score == 5.0
    assert fused[0].dense_score == 0.9
    assert fused[0].matched_queries == ["original query"]


# ---------------------------------------------------------------------------
# 5. Query Expansion Tests (retention, dedup, cache, fallback)
# ---------------------------------------------------------------------------


def test_query_expansion_retains_original_dedups_caches_and_falls_back(tmp_path: Path):
    assert deduplicate_queries(
        "What is IL-6?",
        ["what is il-6?", "Interleukin-6 cytokine", "interleukin-6 cytokine", "IL-6 receptor tocilizumab"],
        max_alternates=3,
    ) == ["Interleukin-6 cytokine", "IL-6 receptor tocilizumab"]

    cache_path = tmp_path / "qe_cache.json"
    fake_client = FakeGeminiClient(
        alternates=[
            "What is IL-6?",  # duplicate of original -> must be stripped from expanded_queries
            "Interleukin-6 cytokine inflammation",
            "IL-6 tocilizumab rheumatoid arthritis",
            "interleukin-6 cytokine inflammation",  # case-insensitive duplicate
            "gp130 STAT3 IL-6 signaling",
        ]
    )
    settings = Settings(gemini_rpm_limit=0)
    expander = BiomedicalQueryExpander(
        settings=settings,
        client=fake_client,
        cache_path=cache_path,
        use_cache=True,
    )

    res1 = expander.expand("What is IL-6?")
    assert res1.all_queries[0] == "What is IL-6?"
    assert len(res1.expanded_queries) == 3
    assert res1.all_queries == [res1.original_query] + res1.expanded_queries
    assert res1.cached is False
    assert res1.fallback_used is False
    assert res1.total_tokens == 70
    assert fake_client.call_count == 1

    # Second call hits disk/memory cache -> no additional LLM call
    res2 = expander.expand("  what is il-6? ")
    assert res2.cached is True
    assert res2.all_queries[0] == "what is il-6?"
    assert res2.expanded_queries == res1.expanded_queries
    assert fake_client.call_count == 1

    # Graceful fallback on API failure
    failing_expander = BiomedicalQueryExpander(
        settings=settings,
        client=FakeGeminiClient([], should_fail=True),
        cache_path=tmp_path / "empty_cache.json",
        use_cache=False,
        max_retries=1,
    )
    res_fail = failing_expander.expand("What causes Hirschsprung disease?")
    assert res_fail.fallback_used is True
    assert res_fail.all_queries == ["What causes Hirschsprung disease?"]
    assert res_fail.expanded_queries == []


# ---------------------------------------------------------------------------
# 6. Unified Retriever (All 5 Modes + Alignment Guard + Search Logging)
# ---------------------------------------------------------------------------


def test_unified_retriever_all_five_modes_and_logging(tmp_path: Path):
    bm25 = BM25Index()
    bm25.build(SAMPLE_PASSAGES)

    dense = DenseIndex(embedder=KeywordVectorEmbedder())
    dense.build(SAMPLE_PASSAGES)

    passage_map: Dict[str, Passage] = {p.id: p for p in SAMPLE_PASSAGES}
    settings = Settings(gemini_rpm_limit=0)

    expander = BiomedicalQueryExpander(
        settings=settings,
        client=FakeGeminiClient(
            alternates=[
                "interleukin-6 cytokine tocilizumab",
                "IL-6 severe COVID-19 arthritis",
                "anti-IL-6 receptor blockade",
            ]
        ),
        cache_path=tmp_path / "qe_cache.json",
    )
    reranker = FlashRankReranker(ranker=FakeFlashRank(boost_id="100005"))
    log_path = tmp_path / "searches.jsonl"

    retriever = UnifiedRetriever(
        bm25_index=bm25,
        dense_index=dense,
        passage_map=passage_map,
        expander=expander,
        reranker=reranker,
        settings=settings,
        search_log_path=log_path,
    )

    query = "How does tocilizumab target IL-6 in rheumatoid arthritis?"
    for mode in (
        RetrievalMode.LEXICAL,
        RetrievalMode.DENSE,
        RetrievalMode.HYBRID,
        RetrievalMode.HYBRID_QE,
        RetrievalMode.HYBRID_QE_RERANK,
    ):
        result = retriever.retrieve(query, mode=mode, top_k=3, record_search=True)
        assert result.mode == mode
        assert result.original_query == query
        assert len(result.passages) >= 1
        assert result.passages[0].pubmed_url is not None

        if mode in (RetrievalMode.HYBRID_QE, RetrievalMode.HYBRID_QE_RERANK):
            assert result.all_queries[0] == query
            assert len(result.expanded_queries) == 3
        else:
            assert result.expanded_queries == []

        if mode == RetrievalMode.HYBRID_QE_RERANK:
            # Our fake reranker boosts 100005 to rank 1
            assert result.passages[0].passage_id == "100005"
            assert result.passages[0].rerank_score == 0.99
        else:
            assert result.passages[0].passage_id == "100002"

    # Verify all 5 searches were recorded to JSONL with passage IDs and scores
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    records = [json.loads(line) for line in lines]
    assert [r["mode"] for r in records] == [m.value for m in RetrievalMode]
    for rec in records:
        assert rec["original_query"] == query
        assert len(rec["retrieved_passages"]) >= 1
        assert "passage_id" in rec["retrieved_passages"][0]
        assert "score" in rec["retrieved_passages"][0]


def test_unified_retriever_rejects_misaligned_indexes():
    bm25 = BM25Index()
    bm25.build(SAMPLE_PASSAGES)

    dense = DenseIndex(embedder=KeywordVectorEmbedder())
    dense.build(SAMPLE_PASSAGES[:-1])  # missing last passage!

    passage_map = {p.id: p for p in SAMPLE_PASSAGES}
    with pytest.raises(ValueError, match="not aligned"):
        UnifiedRetriever(bm25_index=bm25, dense_index=dense, passage_map=passage_map)
