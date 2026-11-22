"""Strictly grounded Gemini answer generator with two-stage insufficient-evidence gate."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.config import RetrievalMode, Settings, estimate_llm_cost_usd, get_settings
from src.generation.citations import CitationValidationResult, validate_citations
from src.retrieval.query_expansion import _RateLimiter
from src.retrieval.retriever import RetrievedPassage

logger = logging.getLogger(__name__)

GROUNDED_ANSWER_SYSTEM_PROMPT = (
    "You are a strictly grounded biomedical question-answering assistant. "
    "You MUST answer the user's question using ONLY the retrieved biomedical passages provided below. "
    "Do NOT use any outside medical knowledge, prior training memory, or assumptions not explicitly stated "
    "in the provided passages.\n\n"
    "Strict rules:\n"
    "1. Every factual claim in your answer MUST be directly supported by one or more provided passages "
    "and include inline citations using the exact passage ID in square brackets, e.g., [16986122] or [16986122] [2309705].\n"
    "2. NEVER invent, guess, or cite any passage ID that is not explicitly present in the provided `[Passage ID: ...]` headers. "
    "NEVER use sequential numbers like [1] or [2] unless that exact string is the Passage ID.\n"
    "3. If the provided passages do NOT contain enough direct evidence to answer the question accurately "
    "(for example, the passages are off-topic, tangential, or omit the key clinical/biological fact asked), "
    "you MUST set `sufficient_evidence` to false, explain specifically what evidence is missing in `insufficient_reason`, "
    "and state clearly in `answer` that there is insufficient evidence in the retrieved passages.\n"
    "4. Keep the answer concise, accurate, objective, and directly responsive to the question."
)

REPAIR_SYSTEM_PROMPT = (
    "You are a strict biomedical citation verifier and editor. "
    "The previous draft answer omitted valid inline `[PASSAGE_ID]` citations. "
    "Rewrite the answer using ONLY the provided passages so every factual claim has an exact inline `[PASSAGE_ID]` "
    "matching one of the provided `[Passage ID: ...]` headers. "
    "If the provided passages do NOT actually support answering the question, set `sufficient_evidence` to false "
    "and explain what evidence is missing."
)


class _AnswerSchema(BaseModel):
    """Structured JSON response schema requested from Gemini."""

    answer: str = Field(
        description=(
            "Grounded biomedical answer with inline [PASSAGE_ID] citations after claims, "
            "or a concise insufficient-evidence statement when abstaining."
        )
    )
    sufficient_evidence: bool = Field(
        description="True only if the provided passages contain sufficient direct evidence to answer the question."
    )
    insufficient_reason: Optional[str] = Field(
        default=None,
        description="Specific explanation of what evidence is missing when sufficient_evidence is false.",
    )
    cited_ids: List[str] = Field(
        default_factory=list,
        description="List of exact passage IDs cited inline in the answer.",
    )


class GenerationResult(BaseModel):
    """Complete structured output from grounded generation and citation enforcement."""

    answer: str
    sufficient_evidence: bool
    insufficient_reason: Optional[str] = None
    abstention_stage: Optional[str] = None  # None | pre_generation | model_abstention | citation_enforcement | generation_error
    selected_passage_ids: List[str] = Field(default_factory=list)
    citation_validation: CitationValidationResult
    threshold_used: float = 0.0
    top_retrieval_score: Optional[float] = None
    model_name: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    repair_attempted: bool = False
    error: Optional[str] = None


def format_passages_for_prompt(passages: Sequence[RetrievedPassage]) -> str:
    """Format selected retrieved passages with unambiguous `[Passage ID: ...]` headers."""
    blocks: List[str] = []
    for p in passages:
        blocks.append(f"[Passage ID: {p.passage_id}]\n{p.text}")
    return "\n\n".join(blocks)


class GroundedAnswerer:
    """Generates strictly grounded answers with two-stage abstention and citation enforcement."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
        max_retries: int = 3,
        allow_repair: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.gemini_generation_model
        self.max_retries = max_retries
        self.allow_repair = allow_repair
        self._client = client
        self._rate_limiter = _RateLimiter(self.settings.gemini_rpm_limit)

    def _get_client(self):
        if self._client is None:
            api_key = (self.settings.gemini_api_key or "").strip()
            if not api_key or api_key == "your_gemini_api_key_here":
                raise RuntimeError("GEMINI_API_KEY is not configured.")
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
        return self._client

    def _call_llm(
        self,
        system_instruction: str,
        user_prompt: str,
    ) -> Tuple[_AnswerSchema, int, int]:
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_AnswerSchema,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter.wait()
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )

                parsed_schema: Optional[_AnswerSchema] = None
                parsed = getattr(response, "parsed", None)
                if isinstance(parsed, _AnswerSchema):
                    parsed_schema = parsed
                elif parsed is not None and hasattr(parsed, "answer"):
                    parsed_schema = _AnswerSchema(
                        answer=str(getattr(parsed, "answer", "") or ""),
                        sufficient_evidence=bool(
                            getattr(parsed, "sufficient_evidence", False)
                        ),
                        insufficient_reason=getattr(parsed, "insufficient_reason", None),
                        cited_ids=[
                            str(x) for x in (getattr(parsed, "cited_ids", None) or [])
                        ],
                    )
                elif getattr(response, "text", None):
                    payload = json.loads(response.text)
                    parsed_schema = _AnswerSchema.model_validate(payload)

                if parsed_schema is None:
                    raise ValueError("Gemini returned an empty or unparseable answer payload.")

                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                completion_tokens = int(
                    getattr(usage, "candidates_token_count", 0) or 0
                )
                return parsed_schema, prompt_tokens, completion_tokens

            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                is_transient = any(
                    code in msg
                    for code in ("429", "resource_exhausted", "503", "unavailable", "rate")
                )
                if is_transient and attempt + 1 < self.max_retries:
                    backoff = 2.0 ** (attempt + 1)
                    logger.warning(
                        "Transient Gemini generation error (attempt %d/%d): %s. Retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise

        assert last_exc is not None
        raise last_exc

    def get_threshold(self, mode: RetrievalMode) -> float:
        """Return the calibrated pre-generation minimum relevance score threshold for `mode`."""
        return float(self.settings.min_score_thresholds.get(mode.value, 0.0))

    def generate(
        self,
        query: str,
        passages: Sequence[RetrievedPassage],
        mode: RetrievalMode = RetrievalMode.HYBRID_QE_RERANK,
        context_k: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
    ) -> GenerationResult:
        """Generate a strictly grounded answer from the top-`context_k` retrieved passages."""
        t_start = time.perf_counter()
        k = (
            self.settings.default_answer_context_k
            if context_k is None
            else max(1, int(context_k))
        )
        selected = list(passages[:k])
        selected_ids = [p.passage_id for p in selected]
        threshold = (
            self.get_threshold(mode)
            if min_score_threshold is None
            else float(min_score_threshold)
        )
        top_score = max((p.score for p in selected), default=None)

        # Stage 1: Pre-generation threshold gate
        if not selected:
            reason = "No relevant passages were retrieved from the indexed biomedical corpus for this query."
            abstention_msg = (
                "Insufficient evidence: no matching biomedical passages were retrieved for this question."
            )
            validation = validate_citations(
                abstention_msg, allowed_ids=[], sufficient_evidence=False
            )
            return GenerationResult(
                answer=abstention_msg,
                sufficient_evidence=False,
                insufficient_reason=reason,
                abstention_stage="pre_generation",
                selected_passage_ids=[],
                citation_validation=validation,
                threshold_used=threshold,
                top_retrieval_score=None,
                model_name=self.model_name,
                latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
            )

        assert top_score is not None
        if top_score < threshold:
            reason = (
                f"Top retrieved passage score ({top_score:.4f}) fell below the minimum "
                f"relevance threshold ({threshold:.4f}) for '{mode.value}' retrieval."
            )
            abstention_msg = (
                "Insufficient evidence: the retrieved biomedical passages did not meet "
                "the minimum relevance threshold needed to answer this question reliably."
            )
            validation = validate_citations(
                abstention_msg, allowed_ids=selected_ids, sufficient_evidence=False
            )
            return GenerationResult(
                answer=abstention_msg,
                sufficient_evidence=False,
                insufficient_reason=reason,
                abstention_stage="pre_generation",
                selected_passage_ids=selected_ids,
                citation_validation=validation,
                threshold_used=threshold,
                top_retrieval_score=round(float(top_score), 6),
                model_name=self.model_name,
                latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
            )

        # Stage 2: Strictly grounded LLM generation over selected top-K passages ONLY
        context_block = format_passages_for_prompt(selected)
        user_prompt = (
            f"Question: {query.strip()}\n\n"
            f"Allowed Passage IDs: {', '.join(selected_ids)}\n\n"
            f"Retrieved Biomedical Passages:\n{context_block}\n\n"
            "Write a grounded answer citing exact `[PASSAGE_ID]` values inline, "
            "or set `sufficient_evidence` to false if the passages do not answer the question."
        )

        total_prompt_tokens = 0
        total_completion_tokens = 0
        repair_attempted = False

        try:
            schema, p_tok, c_tok = self._call_llm(
                GROUNDED_ANSWER_SYSTEM_PROMPT, user_prompt
            )
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
        except Exception as exc:
            logger.warning("Grounded generation failed: %s", exc)
            reason = f"Answer generation failed due to an LLM API error: {exc}"
            abstention_msg = (
                "Insufficient evidence: unable to generate a verified grounded answer "
                "at this time due to a temporary generation service error."
            )
            validation = validate_citations(
                abstention_msg, allowed_ids=selected_ids, sufficient_evidence=False
            )
            return GenerationResult(
                answer=abstention_msg,
                sufficient_evidence=False,
                insufficient_reason=reason,
                abstention_stage="generation_error",
                selected_passage_ids=selected_ids,
                citation_validation=validation,
                threshold_used=threshold,
                top_retrieval_score=round(float(top_score), 6),
                model_name=self.model_name,
                latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                error=str(exc),
            )

        sufficient = bool(schema.sufficient_evidence)
        insufficient_reason = (schema.insufficient_reason or "").strip() or None
        raw_answer = (schema.answer or "").strip()

        # Stage 3: Deterministic citation validation + optional one-shot repair
        validation = validate_citations(
            raw_answer,
            allowed_ids=selected_ids,
            sufficient_evidence=sufficient,
        )

        if sufficient and validation.missing_citations and self.allow_repair:
            repair_attempted = True
            repair_prompt = (
                f"Question: {query.strip()}\n\n"
                f"Allowed Passage IDs: {', '.join(selected_ids)}\n\n"
                f"Retrieved Biomedical Passages:\n{context_block}\n\n"
                f"Previous Draft (missing valid inline citations):\n{raw_answer}\n\n"
                "Rewrite the answer so every claim is backed by an inline `[PASSAGE_ID]` "
                "from the Allowed Passage IDs, or set `sufficient_evidence` to false."
            )
            try:
                repaired_schema, rp_tok, rc_tok = self._call_llm(
                    REPAIR_SYSTEM_PROMPT, repair_prompt
                )
                total_prompt_tokens += rp_tok
                total_completion_tokens += rc_tok

                sufficient = bool(repaired_schema.sufficient_evidence)
                insufficient_reason = (
                    (repaired_schema.insufficient_reason or "").strip() or None
                )
                raw_answer = (repaired_schema.answer or "").strip()

                repaired_validation = validate_citations(
                    raw_answer,
                    allowed_ids=selected_ids,
                    sufficient_evidence=sufficient,
                )
                # Preserve any invalid IDs detected on the initial draft for audit completeness
                if validation.invalid_citations:
                    merged_invalid = list(
                        dict.fromkeys(
                            validation.invalid_citations
                            + repaired_validation.invalid_citations
                        )
                    )
                    repaired_validation.invalid_citations = merged_invalid
                validation = repaired_validation
            except Exception as exc:
                logger.warning("Citation repair call failed: %s", exc)

        abstention_stage: Optional[str] = None
        final_answer = validation.sanitized_answer

        if not sufficient:
            abstention_stage = "model_abstention"
            if not insufficient_reason:
                insufficient_reason = (
                    "The retrieved passages did not contain sufficient direct evidence "
                    "to answer this biomedical question."
                )
            if not final_answer:
                final_answer = f"Insufficient evidence: {insufficient_reason}"
        elif validation.missing_citations:
            # Hard code-level safety fallback: never emit a non-abstaining uncited answer
            sufficient = False
            abstention_stage = "citation_enforcement"
            insufficient_reason = (
                "The generated answer did not include verifiable inline citations to "
                "the selected retrieved passages and was downgraded to insufficient evidence."
            )
            final_answer = (
                "Insufficient evidence: a fully cited answer grounded in the selected "
                "retrieved passages could not be verified."
            )
            validation.sanitized_answer = final_answer

        total_tokens = total_prompt_tokens + total_completion_tokens
        cost_usd = estimate_llm_cost_usd(
            self.model_name, total_prompt_tokens, total_completion_tokens
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return GenerationResult(
            answer=final_answer,
            sufficient_evidence=sufficient,
            insufficient_reason=insufficient_reason,
            abstention_stage=abstention_stage,
            selected_passage_ids=selected_ids,
            citation_validation=validation,
            threshold_used=threshold,
            top_retrieval_score=round(float(top_score), 6),
            model_name=self.model_name,
            latency_ms=latency_ms,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
            repair_attempted=repair_attempted,
        )
