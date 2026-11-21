"""Biomedical query expansion using Google Gemini with caching, pacing, and fallback."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config import Settings, estimate_llm_cost_usd, get_settings

logger = logging.getLogger(__name__)

EXPANSION_SYSTEM_PROMPT = (
    "You are an expert biomedical information retrieval specialist (PubMed / BioASQ). "
    "Given a biomedical question, generate concise, high-signal alternate search queries "
    "to maximize recall over PubMed abstracts. "
    "Use authoritative biomedical synonyms, MeSH terms, gene/protein/drug aliases, "
    "clinical terminology, and acronym expansions (or acronyms where helpful). "
    "Keep each alternate query focused on the exact clinical/biological intent of the original "
    "question without introducing unrelated topics. "
    "Do NOT repeat the original question verbatim."
)


class _ExpansionSchema(BaseModel):
    """Structured JSON response schema requested from Gemini."""

    alternate_queries: List[str] = Field(
        description="3 complementary biomedical alternate search queries."
    )


class QueryExpansionResult(BaseModel):
    """Result returned by the query expander, always preserving the original query first."""

    original_query: str
    expanded_queries: List[str] = Field(default_factory=list)
    all_queries: List[str] = Field(default_factory=list)
    model_name: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cached: bool = False
    fallback_used: bool = False
    error: Optional[str] = None


class _RateLimiter:
    """Thread-safe minimum-interval rate limiter for Gemini Free Tier RPM pacing."""

    def __init__(self, rpm_limit: int) -> None:
        self.rpm_limit = max(0, int(rpm_limit))
        self._lock = threading.Lock()
        self._last_call_ts: float = 0.0

    def wait(self) -> None:
        if self.rpm_limit <= 0:
            return
        min_interval = 60.0 / float(self.rpm_limit)
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_ts
            if self._last_call_ts > 0.0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_call_ts = time.monotonic()


def deduplicate_queries(
    original_query: str,
    candidates: List[str],
    max_alternates: int = 3,
) -> List[str]:
    """Clean and case-insensitively deduplicate alternate queries, excluding the original."""
    orig_clean = " ".join(original_query.strip().split())
    seen = {orig_clean.lower()}
    alternates: List[str] = []

    for q in candidates:
        if not isinstance(q, str):
            continue
        cleaned = " ".join(q.strip().strip('"').strip("'").split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        alternates.append(cleaned)
        if len(alternates) >= max_alternates:
            break

    return alternates


class BiomedicalQueryExpander:
    """Generates biomedical alternate queries via Gemini while always retaining the original query."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.gemini_expansion_model
        self.num_expansions = self.settings.qe_num_expansions
        self.temperature = self.settings.qe_temperature
        self.cache_path = cache_path or self.settings.query_expansion_cache_path
        self.use_cache = use_cache
        self.max_retries = max_retries
        self._client = client
        self._rate_limiter = _RateLimiter(self.settings.gemini_rpm_limit)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        if self.use_cache:
            self._load_cache()

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

    def _cache_key(self, query: str, num_expansions: int) -> str:
        normalized = " ".join(query.strip().lower().split())
        return f"{self.model_name}::{num_expansions}::{normalized}"

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._cache = data
        except Exception as exc:
            logger.warning("Failed to load query expansion cache %s: %s", self.cache_path, exc)

    def _save_cache(self) -> None:
        if not self.use_cache:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_path.with_suffix(".tmp.json")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self.cache_path)
        except Exception as exc:
            logger.warning("Failed to persist query expansion cache: %s", exc)

    def _call_llm(self, query: str, num_expansions: int) -> tuple[List[str], int, int]:
        from google.genai import types

        client = self._get_client()
        user_prompt = (
            f"Original biomedical question: {query}\n\n"
            f"Generate exactly {num_expansions} complementary biomedical alternate search queries."
        )

        config = types.GenerateContentConfig(
            system_instruction=EXPANSION_SYSTEM_PROMPT,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=_ExpansionSchema,
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

                raw_alts: List[str] = []
                parsed = getattr(response, "parsed", None)
                if parsed is not None and hasattr(parsed, "alternate_queries"):
                    raw_alts = list(parsed.alternate_queries)
                elif getattr(response, "text", None):
                    payload = json.loads(response.text)
                    if isinstance(payload, dict) and isinstance(
                        payload.get("alternate_queries"), list
                    ):
                        raw_alts = [str(x) for x in payload["alternate_queries"]]

                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                completion_tokens = int(
                    getattr(usage, "candidates_token_count", 0) or 0
                )
                return raw_alts, prompt_tokens, completion_tokens
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
                        "Transient Gemini expansion error (attempt %d/%d): %s. Retrying in %.1fs.",
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

    def expand(
        self,
        query: str,
        num_expansions: Optional[int] = None,
        use_cache: Optional[bool] = None,
    ) -> QueryExpansionResult:
        """Expand a biomedical query into `[original_query] + alternate_queries`.

        Always preserves `original_query` at index 0 and gracefully falls back to
        `[original_query]` on any API error.
        """
        orig_clean = " ".join((query or "").strip().split())
        n_alts = self.num_expansions if num_expansions is None else max(0, int(num_expansions))
        cache_enabled = self.use_cache if use_cache is None else use_cache

        if not orig_clean or n_alts == 0:
            return QueryExpansionResult(
                original_query=orig_clean,
                expanded_queries=[],
                all_queries=[orig_clean] if orig_clean else [],
                model_name=self.model_name,
            )

        key = self._cache_key(orig_clean, n_alts)
        if cache_enabled:
            with self._cache_lock:
                entry = self._cache.get(key)
            if entry is not None:
                alternates = deduplicate_queries(
                    orig_clean, entry.get("expanded_queries", []), max_alternates=n_alts
                )
                return QueryExpansionResult(
                    original_query=orig_clean,
                    expanded_queries=alternates,
                    all_queries=[orig_clean] + alternates,
                    model_name=str(entry.get("model_name", self.model_name)),
                    latency_ms=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    cached=True,
                    fallback_used=False,
                )

        t0 = time.perf_counter()
        try:
            raw_alts, prompt_tokens, completion_tokens = self._call_llm(orig_clean, n_alts)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            alternates = deduplicate_queries(orig_clean, raw_alts, max_alternates=n_alts)
            cost_usd = estimate_llm_cost_usd(self.model_name, prompt_tokens, completion_tokens)

            if cache_enabled and alternates:
                with self._cache_lock:
                    self._cache[key] = {
                        "original_query": orig_clean,
                        "expanded_queries": alternates,
                        "model_name": self.model_name,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                    self._save_cache()

            return QueryExpansionResult(
                original_query=orig_clean,
                expanded_queries=alternates,
                all_queries=[orig_clean] + alternates,
                model_name=self.model_name,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost_usd=cost_usd,
                cached=False,
                fallback_used=False,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.warning("Query expansion failed; falling back to original query: %s", exc)
            return QueryExpansionResult(
                original_query=orig_clean,
                expanded_queries=[],
                all_queries=[orig_clean],
                model_name=self.model_name,
                latency_ms=latency_ms,
                cached=False,
                fallback_used=True,
                error=str(exc),
            )
