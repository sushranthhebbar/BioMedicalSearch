"""Unit tests for Phase 2: deterministic citation validation and grounded answer generation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import List

from src.config import RetrievalMode, Settings
from src.generation.answerer import GroundedAnswerer
from src.generation.citations import extract_citation_ids, validate_citations
from src.retrieval.retriever import RetrievedPassage


def _make_passage(rank: int, pid: str, text: str, score: float) -> RetrievedPassage:
    return RetrievedPassage(
        rank=rank,
        passage_id=pid,
        text=text,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
        score=score,
        rerank_score=score,
    )


class ScriptedAnswerGeminiClient:
    """Deterministic mock `google.genai.Client` for testing GroundedAnswerer flows."""

    def __init__(self, responses: List[dict], should_fail: bool = False) -> None:
        self.responses = list(responses)
        self.should_fail = should_fail
        self.calls: List[str] = []
        self.models = self

    def generate_content(self, model: str, contents: str, config=None):
        self.calls.append(contents)
        if self.should_fail:
            raise RuntimeError("Simulated Gemini generation failure")
        payload = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return SimpleNamespace(
            parsed=SimpleNamespace(
                answer=payload.get("answer", ""),
                sufficient_evidence=payload.get("sufficient_evidence", False),
                insufficient_reason=payload.get("insufficient_reason"),
                cited_ids=payload.get("cited_ids", []),
            ),
            text=json.dumps(payload),
            usage_metadata=SimpleNamespace(
                prompt_token_count=180,
                candidates_token_count=60,
            ),
        )


# ---------------------------------------------------------------------------
# 1. Deterministic Inline Citation Extraction & Validation
# ---------------------------------------------------------------------------


def test_citation_extraction_and_biomedical_brackets_preserved():
    text = (
        "Intracellular [Ca2+] and [18F]FDG uptake were assessed [p < 0.05] "
        "in Hirschsprung disease [16986122, 2309705] (see [Passage ID: 17965226] "
        "and [PubMed](https://pubmed.ncbi.nlm.nih.gov/16986122/))."
    )
    allowed = ["16986122", "2309705", "17965226"]
    extracted = extract_citation_ids(text, allowed_ids=allowed)
    assert extracted == ["16986122", "2309705", "17965226"]

    res = validate_citations(text, allowed_ids=allowed, sufficient_evidence=True)
    assert res.is_valid is True
    assert res.zero_hallucinated_ids is True
    assert res.valid_citations == ["16986122", "2309705", "17965226"]
    assert res.invalid_citations == []
    assert res.citation_validity_rate == 1.0
    # Biomedical notation and markdown links remain untouched; multi-ID bracket normalized
    assert "[Ca2+]" in res.sanitized_answer
    assert "[18F]FDG" in res.sanitized_answer
    assert "[p < 0.05]" in res.sanitized_answer
    assert "[16986122] [2309705]" in res.sanitized_answer
    assert "[17965226]" in res.sanitized_answer


def test_citation_validator_detects_and_strips_hallucinated_and_numbered_ids():
    allowed = ["100001", "100002"]
    raw = (
        "RET mutations drive Hirschsprung disease [100001, 99999999]. "
        "Another study confirmed this [1] and [PMID: 88888888]."
    )
    res = validate_citations(raw, allowed_ids=allowed, sufficient_evidence=True)

    assert res.valid_citations == ["100001"]
    assert res.invalid_citations == ["99999999", "1", "88888888"]
    assert res.total_citation_occurrences == 4
    assert res.valid_citation_occurrences == 1
    assert res.invalid_citation_occurrences == 3
    assert res.citation_validity_rate == 0.25
    assert res.zero_hallucinated_ids is False
    assert res.is_valid is False
    assert "[100001]" in res.sanitized_answer
    assert "99999999" not in res.sanitized_answer
    assert "[1]" not in res.sanitized_answer
    assert "88888888" not in res.sanitized_answer


def test_citation_validator_missing_citations_vs_abstention():
    uncited = validate_citations(
        "Metformin activates AMPK in hepatocytes.",
        allowed_ids=["100004"],
        sufficient_evidence=True,
    )
    assert uncited.missing_citations is True
    assert uncited.citation_validity_rate == 0.0
    assert uncited.is_valid is False

    abstained = validate_citations(
        "Insufficient evidence in the retrieved passages to answer this question.",
        allowed_ids=["100004"],
        sufficient_evidence=False,
    )
    assert abstained.missing_citations is False
    assert abstained.citation_validity_rate == 1.0
    assert abstained.is_valid is True


# ---------------------------------------------------------------------------
# 2. GroundedAnswerer — Two-Stage Insufficient Evidence Gate & Repair
# ---------------------------------------------------------------------------


def test_pre_generation_gate_short_circuits_empty_and_low_score():
    client = ScriptedAnswerGeminiClient(
        responses=[{"answer": "Should never be called [100001].", "sufficient_evidence": True}]
    )
    settings = Settings(gemini_rpm_limit=0)
    answerer = GroundedAnswerer(settings=settings, client=client)

    # Empty retrieved passages -> 0 LLM calls
    res_empty = answerer.generate(
        query="What is the recipe for chocolate lava cake?",
        passages=[],
        mode=RetrievalMode.HYBRID_QE_RERANK,
    )
    assert res_empty.sufficient_evidence is False
    assert res_empty.abstention_stage == "pre_generation"
    assert len(client.calls) == 0
    assert res_empty.total_tokens == 0

    # Below calibrated threshold (0.15 for hybrid_qe_rerank) -> 0 LLM calls
    low_passages = [_make_passage(1, "100001", "Unrelated abstract text.", score=0.04)]
    res_low = answerer.generate(
        query="What is the recipe for chocolate lava cake?",
        passages=low_passages,
        mode=RetrievalMode.HYBRID_QE_RERANK,
    )
    assert res_low.sufficient_evidence is False
    assert res_low.abstention_stage == "pre_generation"
    assert "threshold" in (res_low.insufficient_reason or "").lower()
    assert len(client.calls) == 0
    assert res_low.estimated_cost_usd == 0.0


def test_grounded_answerer_enforces_top_k_context_and_validates_citations():
    passages = [
        _make_passage(i, f"20000{i}", f"Biomedical fact number {i}.", score=0.95 - i * 0.05)
        for i in range(1, 8)
    ]
    # Model cites rank-1 (allowed) AND rank-6 (outside top-5 context -> must be stripped!)
    client = ScriptedAnswerGeminiClient(
        responses=[
            {
                "answer": "Fact one is established [200001], and rank six mentions more [200006].",
                "sufficient_evidence": True,
                "cited_ids": ["200001", "200006"],
            }
        ]
    )
    settings = Settings(gemini_rpm_limit=0, default_answer_context_k=5)
    answerer = GroundedAnswerer(settings=settings, client=client)

    res = answerer.generate(
        query="What is fact one?",
        passages=passages,
        mode=RetrievalMode.HYBRID_QE_RERANK,
        context_k=5,
    )

    assert len(client.calls) == 1
    # Verify only top-5 passages were sent in the LLM prompt
    assert "[Passage ID: 200005]" in client.calls[0]
    assert "[Passage ID: 200006]" not in client.calls[0]
    assert res.selected_passage_ids == ["200001", "200002", "200003", "200004", "200005"]
    assert res.sufficient_evidence is True
    assert res.citation_validation.valid_citations == ["200001"]
    assert res.citation_validation.invalid_citations == ["200006"]
    assert "[200006]" not in res.answer
    assert "[200001]" in res.answer
    assert res.total_tokens == 240
    assert res.estimated_cost_usd > 0.0


def test_grounded_answerer_model_abstention():
    passages = [
        _make_passage(1, "100001", "Hirschsprung disease involves RET mutations.", score=0.82)
    ]
    client = ScriptedAnswerGeminiClient(
        responses=[
            {
                "answer": "Insufficient evidence: the retrieved passages do not report the exact 5-year survival rate.",
                "sufficient_evidence": False,
                "insufficient_reason": "No survival-rate statistics appear in the retrieved passages.",
                "cited_ids": [],
            }
        ]
    )
    answerer = GroundedAnswerer(settings=Settings(gemini_rpm_limit=0), client=client)
    res = answerer.generate(
        query="What is the exact 5-year survival rate of Hirschsprung disease in adults?",
        passages=passages,
        mode=RetrievalMode.HYBRID_QE_RERANK,
    )
    assert res.sufficient_evidence is False
    assert res.abstention_stage == "model_abstention"
    assert "survival-rate" in (res.insufficient_reason or "")
    assert res.citation_validation.is_valid is True


def test_grounded_answerer_repairs_missing_citations_once():
    passages = [
        _make_passage(1, "100004", "Metformin activates AMPK to reduce hepatic gluconeogenesis.", score=0.91)
    ]
    client = ScriptedAnswerGeminiClient(
        responses=[
            # First draft omits inline citations
            {
                "answer": "Metformin lowers hepatic gluconeogenesis via AMPK activation.",
                "sufficient_evidence": True,
                "cited_ids": [],
            },
            # Repair call adds valid inline citation
            {
                "answer": "Metformin lowers hepatic gluconeogenesis via AMPK activation [100004].",
                "sufficient_evidence": True,
                "cited_ids": ["100004"],
            },
        ]
    )
    answerer = GroundedAnswerer(settings=Settings(gemini_rpm_limit=0), client=client)
    res = answerer.generate(
        query="How does metformin work?",
        passages=passages,
        mode=RetrievalMode.HYBRID_QE_RERANK,
    )
    assert len(client.calls) == 2
    assert res.repair_attempted is True
    assert res.sufficient_evidence is True
    assert res.citation_validation.valid_citations == ["100004"]
    assert res.citation_validation.is_valid is True
    assert res.total_tokens == 480


def test_grounded_answerer_downgrades_if_still_uncited_after_repair():
    passages = [
        _make_passage(1, "100004", "Metformin activates AMPK.", score=0.91)
    ]
    client = ScriptedAnswerGeminiClient(
        responses=[
            {"answer": "Metformin activates AMPK.", "sufficient_evidence": True, "cited_ids": []},
            {"answer": "Metformin still has no citation.", "sufficient_evidence": True, "cited_ids": []},
        ]
    )
    answerer = GroundedAnswerer(settings=Settings(gemini_rpm_limit=0), client=client)
    res = answerer.generate(
        query="How does metformin work?",
        passages=passages,
        mode=RetrievalMode.HYBRID_QE_RERANK,
    )
    assert len(client.calls) == 2
    assert res.repair_attempted is True
    assert res.sufficient_evidence is False
    assert res.abstention_stage == "citation_enforcement"
    assert "insufficient evidence" in res.answer.lower()
