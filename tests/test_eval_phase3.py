"""Unit and integration tests for Phase 3 benchmark creation and 5-config offline evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

from scripts.create_eval_set import (
    CATEGORY_QUOTAS,
    classify_question,
    create_stratified_eval_set,
)
from scripts.run_eval import (
    ALL_EVAL_MODES,
    load_eval_queries,
    load_existing_runs,
    precompute_expansions,
    run_config_evaluation,
)
from src.config import PROJECT_ROOT, RetrievalMode, Settings
from src.data_loader import BioASQQuestion, Passage
from src.generation.answerer import GroundedAnswerer
from src.indexing.bm25_index import BM25Index
from src.indexing.dense_index import DenseIndex
from src.pipeline import RAGPipeline
from src.retrieval.query_expansion import BiomedicalQueryExpander
from src.retrieval.reranker import FlashRankReranker
from src.retrieval.retriever import UnifiedRetriever
from tests.test_api import DynamicFakeAnswerClient
from tests.test_retrieval import (
    FakeFlashRank,
    FakeGeminiClient,
    KeywordVectorEmbedder,
    SAMPLE_PASSAGES,
)


def test_question_category_classification():
    assert classify_question("Is the protein Papilin secreted?", "Yes, papilin is secreted.") == "yes_no"
    assert classify_question("Is Hirschsprung disease mendelian or multifactorial?", "Yes, multifactorial") != "yes_no"
    assert classify_question("What is an exosome?", "Exosomes are extracellular vesicles.") == "definition"
    assert classify_question("What does iBAQ stand for in proteomics?", "Intensity-based absolute quantification.") == "definition"
    assert classify_question("List Parkin binding partners", "PINK1, USP30") == "list"
    assert classify_question("Which drug is considered as the first line treatment of fibromyalgia?", "Pregabalin.") == "treatment"
    assert classify_question("What is the mode of action of bedaquiline?", "Inhibits ATP synthase.") == "mechanism"
    assert classify_question("What is the effect of resveratrol on mTOR activity?", "Inhibits mTOR.") == "effect"


def test_stratified_eval_set_is_deterministic_and_balanced():
    synthetic: List[BioASQQuestion] = []
    idx = 1
    templates = {
        "yes_no": [
            ("Is protein X{i} secreted?", "Yes, protein X{i} is secreted."),
            ("Does gene Y{i} mutate in cancer?", "No, gene Y{i} does not mutate."),
        ],
        "definition": [("What is Biomarker{i}?", "Biomarker{i} is a protein marker.")],
        "list": [("List genes associated with disorder D{i}", "GENE1, GENE2")],
        "treatment": [("Which drug is used for treatment of disease T{i}?", "DrugT{i}")],
        "mechanism": [("What is the mode of action of compound M{i}?", "Inhibits kinase M{i}.")],
        "effect": [("What is the effect of drug E{i} on heart rate?", "Lowers heart rate.")],
    }
    for cat, pairs in templates.items():
        for i in range(10):
            q_tpl, a_tpl = pairs[i % len(pairs)]
            synthetic.append(
                BioASQQuestion(
                    id=str(idx),
                    question=q_tpl.format(i=i),
                    answer=a_tpl.format(i=i),
                    relevant_passage_ids=[f"1000{idx}"],
                )
            )
            idx += 1

    small_quotas = {cat: 3 for cat in CATEGORY_QUOTAS}
    run1, _ = create_stratified_eval_set(synthetic, quotas=small_quotas, seed=42)
    run2, _ = create_stratified_eval_set(synthetic, quotas=small_quotas, seed=42)

    assert run1 == run2
    assert len(run1) == 18
    assert len({q["id"] for q in run1}) == 18
    for cat in small_quotas:
        assert sum(1 for q in run1 if q["category"] == cat) == 3


def test_committed_eval_queries_100_integrity():
    eval_path = PROJECT_ROOT / "eval" / "eval_queries_100.json"
    assert eval_path.exists(), "eval/eval_queries_100.json must exist after create_eval_set.py"

    meta, queries = load_eval_queries(eval_path)
    assert len(queries) == 100
    assert meta["seed"] == 42
    assert len({q["id"] for q in queries}) == 100

    counts: Dict[str, int] = {}
    for q in queries:
        cat = q["category"]
        counts[cat] = counts.get(cat, 0) + 1
        assert q["question"].strip()
        assert q["answer"].strip()
        assert len(q["relevant_passage_ids"]) >= 1

    assert counts == CATEGORY_QUOTAS


class _FlakyAnswerClient:
    """Fails once for a target query, then succeeds on resume."""

    def __init__(self, fail_once_substring: str) -> None:
        self.fail_once_substring = fail_once_substring.lower()
        self.failed_once = False
        self.calls = 0
        self.models = self

    def generate_content(self, model: str, contents: str, config=None):
        self.calls += 1
        if self.fail_once_substring in contents.lower() and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("429 RESOURCE_EXHAUSTED simulated")
        payload = {
            "answer": "Tocilizumab inhibits IL-6 signaling [100002].",
            "sufficient_evidence": True,
            "insufficient_reason": None,
            "cited_ids": ["100002"],
        }
        return SimpleNamespace(
            parsed=SimpleNamespace(**payload),
            text=json.dumps(payload),
            usage_metadata=SimpleNamespace(prompt_token_count=150, candidates_token_count=30),
        )


def test_run_eval_all_five_configs_caching_and_resume(tmp_path: Path):
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
    qe_client = FakeGeminiClient(
        alternates=[
            "interleukin-6 cytokine tocilizumab",
            "IL-6 receptor antagonist arthritis",
            "anti-IL-6 therapy COVID-19",
        ]
    )
    expander = BiomedicalQueryExpander(
        settings=settings,
        client=qe_client,
        cache_path=tmp_path / "qe_cache.json",
    )
    reranker = FlashRankReranker(ranker=FakeFlashRank(boost_id="100002"))
    retriever = UnifiedRetriever(
        bm25_index=bm25,
        dense_index=dense,
        passage_map=passage_map,
        expander=expander,
        reranker=reranker,
        settings=settings,
        search_log_path=tmp_path / "searches.jsonl",
    )
    flaky_answer_client = _FlakyAnswerClient(fail_once_substring="metformin")
    answerer = GroundedAnswerer(settings=settings, client=flaky_answer_client, max_retries=1)
    pipeline = RAGPipeline(
        retriever=retriever,
        answerer=answerer,
        settings=settings,
        search_log_path=tmp_path / "searches.jsonl",
    )

    eval_queries = [
        {
            "id": "q1",
            "category": "treatment",
            "question": "How does tocilizumab target IL-6 in rheumatoid arthritis?",
            "answer": "Tocilizumab targets IL-6.",
            "relevant_passage_ids": ["100002"],
        },
        {
            "id": "q2",
            "category": "mechanism",
            "question": "How does metformin lower gluconeogenesis via AMPK?",
            "answer": "Metformin activates AMPK.",
            "relevant_passage_ids": ["100004"],
        },
    ]

    out_dir = tmp_path / "eval" / "results"

    # 1. Precompute expansions once for the 2 queries
    snapshot = precompute_expansions(
        queries=eval_queries,
        expander=expander,
        snapshot_path=out_dir / "expansions_100.json",
    )
    assert len(snapshot) == 2
    assert qe_client.call_count == 2

    # 2. First pass of hybrid_qe_rerank hits simulated error on q2, saves checkpoint
    recs1, sum1 = run_config_evaluation(
        pipeline=pipeline,
        mode=RetrievalMode.HYBRID_QE_RERANK,
        queries=eval_queries,
        output_dir=out_dir,
        top_k=5,
        answer_context_k=3,
        generate_answer=True,
        resume=True,
        expansion_snapshot=snapshot,
    )
    assert sum1["complete"] is False
    assert sum1["generation_error_count"] == 1
    assert qe_client.call_count == 2  # zero extra QE calls (used cache)

    # 3. Resume pass retries ONLY q2 and completes cleanly
    calls_before_resume = flaky_answer_client.calls
    recs2, sum2 = run_config_evaluation(
        pipeline=pipeline,
        mode=RetrievalMode.HYBRID_QE_RERANK,
        queries=eval_queries,
        output_dir=out_dir,
        top_k=5,
        answer_context_k=3,
        generate_answer=True,
        resume=True,
        expansion_snapshot=snapshot,
    )
    assert sum2["complete"] is True
    assert sum2["generation_error_count"] == 0
    assert flaky_answer_client.calls == calls_before_resume + 1  # q1 was skipped!
    assert [r["query_id"] for r in recs2] == ["q1", "q2"]

    # Effective usage preserves fair uncached expansion token cost even though cache hit!
    assert recs2[0]["usage"]["expansion_cached"] is True
    assert recs2[0]["effective_usage"]["expansion_total_tokens"] == 70
    assert recs2[0]["effective_usage"]["expansion_cost_usd"] > 0.0
    assert recs2[0]["effective_usage"]["estimated_cost_usd"] > recs2[0]["usage"]["estimated_cost_usd"]

    # 4. Run all 5 configs cleanly with healthy answerer
    healthy_pipeline = RAGPipeline(
        retriever=retriever,
        answerer=GroundedAnswerer(settings=settings, client=DynamicFakeAnswerClient()),
        settings=settings,
        search_log_path=tmp_path / "searches2.jsonl",
    )
    out_dir_all = tmp_path / "eval" / "results_all"
    for mode in ALL_EVAL_MODES:
        recs, summary = run_config_evaluation(
            pipeline=healthy_pipeline,
            mode=mode,
            queries=eval_queries,
            output_dir=out_dir_all,
            top_k=5,
            answer_context_k=3,
            generate_answer=True,
            resume=False,
            expansion_snapshot=snapshot,
        )
        assert len(recs) == 2
        assert summary["complete"] is True
        saved = load_existing_runs(out_dir_all / f"{mode.value}_runs.jsonl")
        assert set(saved.keys()) == {"q1", "q2"}
        # Eval runs must NOT pollute live searches.jsonl
        assert not (tmp_path / "searches2.jsonl").exists()
