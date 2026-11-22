/**
 * BioMedSearch — Perplexity-Style 3-Pane Frontend Controller
 */
(() => {
  "use strict";

  // Resolve configurable API_BASE_URL (supports local FastAPI & standalone Netlify deployment)
  const metaApi = document.querySelector('meta[name="api-base-url"]')?.getAttribute("content") || "";
  const urlApi = new URLSearchParams(window.location.search).get("api") || "";
  const API_BASE_URL = (window.BIOMED_API_BASE_URL || urlApi || metaApi || "").replace(/\/+$/, "");

  const state = {
    config: null,
    history: [],
    activeResult: null,
    evidenceFilter: "all",
    allExpanded: false,
    isLoading: false,
  };

  // DOM References
  const el = {
    appShell: document.getElementById("appShell"),
    brandHomeLink: document.getElementById("brandHomeLink"),
    newSearchBtn: document.getElementById("newSearchBtn"),
    historyFilterInput: document.getElementById("historyFilterInput"),
    clearHistoryFilterBtn: document.getElementById("clearHistoryFilterBtn"),
    historyCount: document.getElementById("historyCount"),
    historyList: document.getElementById("historyList"),
    healthDot: document.getElementById("healthDot"),
    healthText: document.getElementById("healthText"),
    themeToggleBtn: document.getElementById("themeToggleBtn"),

    errorBanner: document.getElementById("errorBanner"),
    landingSection: document.getElementById("landingSection"),
    landingForm: document.getElementById("landingForm"),
    landingQueryInput: document.getElementById("landingQueryInput"),
    landingModeSelect: document.getElementById("landingModeSelect"),
    landingTopKSelect: document.getElementById("landingTopKSelect"),
    landingSubmitBtn: document.getElementById("landingSubmitBtn"),
    examplesGrid: document.getElementById("examplesGrid"),

    threadSection: document.getElementById("threadSection"),
    activeSearchIdPill: document.getElementById("activeSearchIdPill"),
    toggleDrawerBtn: document.getElementById("toggleDrawerBtn"),
    threadForm: document.getElementById("threadForm"),
    threadQueryInput: document.getElementById("threadQueryInput"),
    threadModeSelect: document.getElementById("threadModeSelect"),
    threadTopKSelect: document.getElementById("threadTopKSelect"),
    threadSubmitBtn: document.getElementById("threadSubmitBtn"),

    loadingCard: document.getElementById("loadingCard"),
    resultContainer: document.getElementById("resultContainer"),
    resultQueryHeadline: document.getElementById("resultQueryHeadline"),
    telemetryStrip: document.getElementById("telemetryStrip"),
    expansionsCard: document.getElementById("expansionsCard"),
    expansionMetaTag: document.getElementById("expansionMetaTag"),
    queryChips: document.getElementById("queryChips"),

    insufficientCallout: document.getElementById("insufficientCallout"),
    abstentionStageBadge: document.getElementById("abstentionStageBadge"),
    insufficientAnswerText: document.getElementById("insufficientAnswerText"),
    insufficientReasonBox: document.getElementById("insufficientReasonBox"),

    answerCard: document.getElementById("answerCard"),
    answerModelTag: document.getElementById("answerModelTag"),
    answerBody: document.getElementById("answerBody"),
    citationAuditBar: document.getElementById("citationAuditBar"),

    citationsFooter: document.getElementById("citationsFooter"),
    citedCountLabel: document.getElementById("citedCountLabel"),
    citationsGrid: document.getElementById("citationsGrid"),

    evidenceCountPill: document.getElementById("evidenceCountPill"),
    expandAllBtn: document.getElementById("expandAllBtn"),
    evidenceList: document.getElementById("evidenceList"),
    filterTabs: Array.from(document.querySelectorAll(".filter-tab")),
  };

  // -------------------------------------------------------------------------
  // Utilities
  // -------------------------------------------------------------------------

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatLatency(ms) {
    if (ms == null || Number.isNaN(Number(ms))) return "—";
    const n = Number(ms);
    return n >= 1000 ? `${(n / 1000).toFixed(2)}s` : `${Math.round(n)} ms`;
  }

  function formatScore(val, digits = 4) {
    if (val == null || Number.isNaN(Number(val))) return "—";
    return Number(val).toFixed(digits);
  }

  function formatCost(usd) {
    if (usd == null || Number.isNaN(Number(usd))) return "$0.0000";
    const n = Number(usd);
    if (n === 0) return "$0.0000";
    if (n < 0.0001) return `$${n.toFixed(6)}`;
    return `$${n.toFixed(4)}`;
  }

  function formatTimestamp(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return "";
    }
  }

  function getModeLabel(modeId) {
    const found = state.config?.modes?.find((m) => m.id === modeId);
    return found ? found.short_label : modeId;
  }

  function formatAbstentionStage(stage) {
    const map = {
      pre_generation: "Pre-Generation Relevance Gate",
      model_abstention: "LLM Grounding Abstention",
      citation_enforcement: "Code Citation Enforcement",
      generation_error: "Generation Service Fallback",
    };
    return map[stage] || "Insufficient Evidence";
  }

  function showError(msg) {
    if (!msg) {
      el.errorBanner.classList.add("hidden");
      el.errorBanner.textContent = "";
      return;
    }
    el.errorBanner.textContent = msg;
    el.errorBanner.classList.remove("hidden");
  }

  async function apiFetch(path, options = {}) {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try {
        const errBody = await res.json();
        if (errBody?.detail) {
          detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
        }
      } catch {
        // ignore JSON parse error
      }
      throw new Error(detail);
    }
    return res.json();
  }

  // -------------------------------------------------------------------------
  // Theme Management
  // -------------------------------------------------------------------------

  function initTheme() {
    const saved = localStorage.getItem("biomed_theme") || "dark";
    applyTheme(saved);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("biomed_theme", theme);
    el.themeToggleBtn.textContent = theme === "dark" ? "☀" : "☾";
  }

  // -------------------------------------------------------------------------
  // Initialization: Health, Config, History
  // -------------------------------------------------------------------------

  async function loadHealth() {
    try {
      const data = await apiFetch("/api/health");
      if (data.ready) {
        el.healthDot.className = "status-dot grounded";
        el.healthText.textContent = `${Number(data.corpus_size).toLocaleString()} PubMed passages`;
      } else {
        el.healthDot.className = "status-dot insufficient";
        el.healthText.textContent = "Index initializing...";
      }
    } catch {
      el.healthDot.className = "status-dot insufficient";
      el.healthText.textContent = "API unreachable";
    }
  }

  async function loadConfig() {
    try {
      const cfg = await apiFetch("/api/config");
      state.config = cfg;
      const optionsHtml = cfg.modes
        .map(
          (m) =>
            `<option value="${escapeHtml(m.id)}" ${
              m.id === cfg.default_mode ? "selected" : ""
            }>${escapeHtml(m.label)}${m.id === cfg.default_mode ? " (Default)" : ""}</option>`
        )
        .join("");

      el.landingModeSelect.innerHTML = optionsHtml;
      el.threadModeSelect.innerHTML = optionsHtml;
      el.landingTopKSelect.value = String(cfg.default_top_k || 10);
      el.threadTopKSelect.value = String(cfg.default_top_k || 10);
    } catch (err) {
      showError(`Unable to load configuration from API: ${err.message}`);
    }
  }

  async function loadHistory(filterQuery = "") {
    try {
      const qs = new URLSearchParams({ limit: "60" });
      if (filterQuery.trim()) qs.set("q", filterQuery.trim());
      const data = await apiFetch(`/api/history?${qs.toString()}`);
      state.history = data.items || [];
      el.historyCount.textContent = String(data.total ?? state.history.length);
      renderHistoryList();
    } catch {
      // keep existing list if offline
    }
  }

  function renderHistoryList() {
    if (!state.history.length) {
      el.historyList.innerHTML = `<div class="history-empty">No matching searches found.</div>`;
      return;
    }

    const activeId = state.activeResult?.search_id || null;
    el.historyList.innerHTML = state.history
      .map((item) => {
        const dotClass =
          item.sufficient_evidence === true
            ? "grounded"
            : item.sufficient_evidence === false
            ? "insufficient"
            : "neutral";
        const dotTitle =
          item.sufficient_evidence === true
            ? "Grounded answer"
            : item.sufficient_evidence === false
            ? "Insufficient evidence"
            : "Retrieval only";

        return `
          <button
            type="button"
            class="history-item ${item.search_id === activeId ? "active" : ""}"
            data-search-id="${escapeHtml(item.search_id)}"
            role="listitem"
          >
            <div class="history-item-query">${escapeHtml(item.original_query)}</div>
            <div class="history-item-meta">
              <span class="status-dot ${dotClass}" title="${dotTitle}"></span>
              <span>${escapeHtml(getModeLabel(item.mode))}</span>
              <span>·</span>
              <span>${escapeHtml(formatTimestamp(item.timestamp))}</span>
            </div>
          </button>
        `;
      })
      .join("");
  }

  // -------------------------------------------------------------------------
  // Search Execution & History Restoration
  // -------------------------------------------------------------------------

  function syncModeAndTopK(mode, topK) {
    if (mode) {
      el.landingModeSelect.value = mode;
      el.threadModeSelect.value = mode;
    }
    if (topK) {
      el.landingTopKSelect.value = String(topK);
      el.threadTopKSelect.value = String(topK);
    }
  }

  function setLoadingState(loading, pendingQuery = "") {
    state.isLoading = loading;
    el.landingSubmitBtn.disabled = loading;
    el.threadSubmitBtn.disabled = loading;

    if (loading) {
      showError("");
      el.landingSection.classList.add("hidden");
      el.threadSection.classList.remove("hidden");
      el.loadingCard.classList.remove("hidden");
      el.resultContainer.classList.add("hidden");
      el.threadQueryInput.value = pendingQuery;
      el.evidenceList.innerHTML = `<div class="evidence-empty">Retrieving and ranking PubMed evidence...</div>`;
    } else {
      el.loadingCard.classList.add("hidden");
    }
  }

  async function executeSearch(query, mode, topK) {
    const cleaned = (query || "").trim();
    if (!cleaned || state.isLoading) return;

    syncModeAndTopK(mode, topK);
    setLoadingState(true, cleaned);

    try {
      const result = await apiFetch("/api/search", {
        method: "POST",
        body: JSON.stringify({
          query: cleaned,
          mode,
          top_k: Number(topK) || 10,
        }),
      });
      state.activeResult = result;
      state.evidenceFilter = "all";
      state.allExpanded = false;
      updateFilterTabUI();
      renderResult(result);
      await loadHistory(el.historyFilterInput.value);
    } catch (err) {
      showError(`Search failed: ${err.message}`);
      el.resultContainer.classList.add("hidden");
    } finally {
      setLoadingState(false, cleaned);
      if (state.activeResult) {
        el.resultContainer.classList.remove("hidden");
      }
    }
  }

  async function restoreHistoryItem(searchId) {
    if (!searchId || state.isLoading) return;
    showError("");
    el.landingSection.classList.add("hidden");
    el.threadSection.classList.remove("hidden");
    el.loadingCard.classList.remove("hidden");
    el.resultContainer.classList.add("hidden");

    try {
      const result = await apiFetch(`/api/history/${encodeURIComponent(searchId)}`);
      state.activeResult = result;
      state.evidenceFilter = "all";
      state.allExpanded = false;
      updateFilterTabUI();
      syncModeAndTopK(result.mode, Math.max(5, Math.min(15, result.passages?.length || 10)));
      el.threadQueryInput.value = result.original_query || "";
      renderResult(result);
      renderHistoryList();
    } catch (err) {
      showError(`Could not load past search: ${err.message}`);
    } finally {
      el.loadingCard.classList.add("hidden");
      if (state.activeResult) {
        el.resultContainer.classList.remove("hidden");
      }
    }
  }

  function resetToNewSearch() {
    showError("");
    state.activeResult = null;
    el.threadSection.classList.add("hidden");
    el.landingSection.classList.remove("hidden");
    el.landingQueryInput.value = "";
    el.landingQueryInput.focus();
    el.evidenceCountPill.textContent = "0";
    el.evidenceList.innerHTML = `
      <div class="evidence-empty">
        Run a biomedical search to inspect ranked passages, score breakdowns, and PubMed source links.
      </div>
    `;
    renderHistoryList();
  }

  // -------------------------------------------------------------------------
  // Rendering: Expansions, Grounded Answer / Insufficient Evidence, Citations, Drawer
  // -------------------------------------------------------------------------

  function renderAnswerWithCitations(answerText, passageById) {
    const raw = String(answerText || "").trim();
    if (!raw) return "<p>No answer text returned.</p>";

    // Escape HTML first, then convert inline [ID] tokens into interactive citation buttons
    const paragraphs = raw.split(/\n{2,}/);
    return paragraphs
      .map((para) => {
        const lines = para.split("\n").map((l) => l.trim()).filter(Boolean);
        const isList = lines.length > 1 && lines.every((l) => /^[-*•]\s+/.test(l));
        if (isList) {
          const items = lines
            .map((l) => `<li>${injectCitationBadges(l.replace(/^[-*•]\s+/, ""), passageById)}</li>`)
            .join("");
          return `<ul>${items}</ul>`;
        }
        return `<p>${injectCitationBadges(lines.join(" "), passageById)}</p>`;
      })
      .join("");
  }

  function injectCitationBadges(text, passageById) {
    const escaped = escapeHtml(text);
    // Replace bold **text** cleanly
    const withBold = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // Replace inline [ID] citations matching retrieved passages
    return withBold.replace(/\[([A-Za-z0-9_-]+)\]/g, (full, pid) => {
      const passage = passageById.get(pid);
      if (!passage) return full;
      const tip = `Rank #${passage.rank} · Score ${formatScore(passage.score)} · Click to inspect passage`;
      return `<button type="button" class="cite-badge" data-cite-id="${escapeHtml(pid)}" title="${escapeHtml(tip)}">[${escapeHtml(pid)}]</button>`;
    });
  }

  function renderResult(result) {
    const passages = result.passages || [];
    const passageById = new Map(passages.map((p) => [p.passage_id, p]));
    const lat = result.latency || {};
    const usage = result.usage || {};
    const citVal = result.citation_validation || null;

    el.activeSearchIdPill.textContent = `Search #${result.search_id || "—"}`;
    el.resultQueryHeadline.textContent = result.original_query || "";

    // Telemetry Pills
    const telemetryParts = [
      `<span class="pill pill-accent">${escapeHtml(getModeLabel(result.mode))}</span>`,
      `<span class="pill mono" title="Retrieval: ${formatLatency(lat.retrieval_total_ms)} | Generation: ${formatLatency(lat.generation_ms)}">⏱ ${escapeHtml(formatLatency(lat.total_ms))}</span>`,
      `<span class="pill mono" title="Prompt + Completion tokens">${Number(usage.total_tokens || 0).toLocaleString()} tok · ${escapeHtml(formatCost(usage.estimated_cost_usd))}</span>`,
    ];
    if (result.sufficient_evidence === true) {
      telemetryParts.push(`<span class="pill pill-accent">✓ Grounded</span>`);
    } else if (result.sufficient_evidence === false) {
      telemetryParts.push(`<span class="pill pill-warning">⚠ Insufficient Evidence</span>`);
    } else {
      telemetryParts.push(`<span class="pill">Retrieval Only</span>`);
    }
    el.telemetryStrip.innerHTML = telemetryParts.join("");

    // Query & Expansions Card
    const expanded = result.expanded_queries || [];
    const usesQE = result.mode === "hybrid_qe" || result.mode === "hybrid_qe_rerank";
    if (usesQE) {
      const cacheLabel = usage.expansion_cached
        ? "cached"
        : usage.expansion_fallback_used
        ? "fallback"
        : formatLatency(lat.expansion_ms);
      el.expansionMetaTag.textContent = `${expanded.length} expansion${expanded.length === 1 ? "" : "s"} · ${cacheLabel}`;
    } else {
      el.expansionMetaTag.textContent = "expansion disabled in this mode";
    }

    const chipsHtml = [
      `
        <span class="query-chip original">
          <span class="chip-weight">${usesQE ? "Orig 2×" : "Original"}</span>
          <span>${escapeHtml(result.original_query)}</span>
        </span>
      `,
      ...expanded.map(
        (eq, idx) => `
          <span class="query-chip">
            <span class="chip-weight">Exp ${idx + 1}</span>
            <span>${escapeHtml(eq)}</span>
          </span>
        `
      ),
    ];
    el.queryChips.innerHTML = chipsHtml.join("");

    // Grounded Answer vs Explicit Insufficient Evidence Callout
    if (result.sufficient_evidence === false) {
      el.answerCard.classList.add("hidden");
      el.insufficientCallout.classList.remove("hidden");
      el.abstentionStageBadge.textContent = formatAbstentionStage(result.abstention_stage);
      el.insufficientAnswerText.textContent =
        result.answer ||
        "Insufficient evidence in the retrieved biomedical literature to answer this question.";

      let reasonHtml = `<strong>Reason:</strong> ${escapeHtml(
        result.insufficient_reason || "Retrieved passages did not sufficiently answer the query."
      )}`;
      if (result.top_retrieval_score != null && result.threshold_used != null) {
        reasonHtml += ` <span class="mono">(Top score: ${formatScore(result.top_retrieval_score)} · Threshold: ${formatScore(result.threshold_used)})</span>`;
      }
      el.insufficientReasonBox.innerHTML = reasonHtml;
    } else if (result.answer) {
      el.insufficientCallout.classList.add("hidden");
      el.answerCard.classList.remove("hidden");
      el.answerModelTag.textContent = result.answer_model || "gemini-2.5-flash";
      el.answerBody.innerHTML = renderAnswerWithCitations(result.answer, passageById);

      // Citation Audit Bar
      if (citVal) {
        const validCount = citVal.valid_citations?.length || 0;
        const invalidCount = citVal.invalid_citations?.length || 0;
        const pct = Math.round((citVal.citation_validity_rate ?? 1) * 100);
        const badgeClass = invalidCount === 0 && citVal.is_valid ? "pill-accent" : "pill-warning";
        const auditText =
          invalidCount === 0
            ? `✓ ${validCount} source${validCount === 1 ? "" : "s"} verified · ${pct}% valid · 0 hallucinated IDs`
            : `⚠ ${validCount} valid · ${invalidCount} hallucinated ID${invalidCount === 1 ? "" : "s"} stripped (${citVal.invalid_citations.join(", ")})`;

        el.citationAuditBar.innerHTML = `
          <span class="pill ${badgeClass}">${escapeHtml(auditText)}</span>
          <span class="mono">Context window: Top ${result.selected_passage_ids?.length || 0} passages</span>
        `;
      } else {
        el.citationAuditBar.innerHTML = "";
      }
    } else {
      // Historical retrieval-only log entry (e.g. from Phase 1 smoke test)
      el.insufficientCallout.classList.add("hidden");
      el.answerCard.classList.remove("hidden");
      el.answerModelTag.textContent = "retrieval-only log";
      el.answerBody.innerHTML = `<p style="color:var(--text-secondary);">This historical search was recorded in retrieval-only mode before answer generation was enabled. Click <strong>Search</strong> above to generate a grounded cited answer.</p>`;
      el.citationAuditBar.innerHTML = "";
    }

    // Citations Footer (Bottom of Answer)
    const citedIds = citVal?.valid_citations || [];
    const citedPassages = citedIds.map((id) => passageById.get(id)).filter(Boolean);

    if (citedPassages.length > 0 && result.sufficient_evidence !== false) {
      el.citationsFooter.classList.remove("hidden");
      el.citedCountLabel.textContent = `${citedPassages.length} cited passage${citedPassages.length === 1 ? "" : "s"}`;
      el.citationsGrid.innerHTML = citedPassages
        .map(
          (p) => `
            <div class="citation-footer-card" data-jump-id="${escapeHtml(p.passage_id)}" tabindex="0" role="button">
              <div class="citation-footer-top">
                <span class="pill pill-accent mono">[${escapeHtml(p.passage_id)}]</span>
                ${
                  p.pubmed_url
                    ? `<a href="${escapeHtml(p.pubmed_url)}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">PubMed ↗</a>`
                    : `<span class="mono" style="font-size:11px;color:var(--text-muted);">#${p.rank}</span>`
                }
              </div>
              <div class="citation-footer-snippet">${escapeHtml(p.text)}</div>
              <div class="history-item-meta">
                <span>Rank #${p.rank}</span>
                <span>·</span>
                <span class="mono">Score ${formatScore(p.score)}</span>
              </div>
            </div>
          `
        )
        .join("");
    } else {
      el.citationsFooter.classList.add("hidden");
      el.citationsGrid.innerHTML = "";
    }

    el.resultContainer.classList.remove("hidden");
    renderEvidenceDrawer();
  }

  function updateFilterTabUI() {
    el.filterTabs.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.filter === state.evidenceFilter);
    });
  }

  function renderEvidenceDrawer() {
    const result = state.activeResult;
    const allPassages = result?.passages || [];
    el.evidenceCountPill.textContent = String(allPassages.length);

    let filtered = allPassages;
    if (state.evidenceFilter === "cited") {
      filtered = allPassages.filter((p) => p.cited);
    } else if (state.evidenceFilter === "context") {
      filtered = allPassages.filter((p) => p.selected_for_answer);
    }

    if (!filtered.length) {
      el.evidenceList.innerHTML = `<div class="evidence-empty">No passages match the "${escapeHtml(state.evidenceFilter)}" filter.</div>`;
      return;
    }

    const origLower = (result?.original_query || "").trim().toLowerCase();

    el.evidenceList.innerHTML = filtered
      .map((p) => {
        const statusPill = p.cited
          ? `<span class="pill pill-accent">Cited</span>`
          : p.selected_for_answer
          ? `<span class="pill pill-info">Context</span>`
          : `<span class="pill">Retrieved</span>`;

        const scoreCells = [
          { label: "Final", val: formatScore(p.score) },
          ...(p.rerank_score != null ? [{ label: "Rerank", val: formatScore(p.rerank_score) }] : []),
          ...(p.rrf_score != null ? [{ label: "RRF", val: formatScore(p.rrf_score, 5) }] : []),
          ...(p.bm25_score != null ? [{ label: "BM25", val: formatScore(p.bm25_score, 3) }] : []),
          ...(p.dense_score != null ? [{ label: "Dense", val: formatScore(p.dense_score, 4) }] : []),
        ];

        const matchedQueries = p.matched_queries || [];

        return `
          <div class="evidence-card ${p.cited ? "cited-card" : ""} ${state.allExpanded ? "expanded" : ""}" data-passage-id="${escapeHtml(p.passage_id)}">
            <button type="button" class="evidence-card-header" aria-expanded="${state.allExpanded ? "true" : "false"}">
              <div class="evidence-top-row">
                <div class="evidence-id-group">
                  <span class="rank-badge">#${p.rank}</span>
                  <span class="pmid-label">PMID ${escapeHtml(p.passage_id)}</span>
                  ${statusPill}
                </div>
                <span class="score-pill">${formatScore(p.score)}</span>
              </div>
              <div class="evidence-preview">${escapeHtml(p.text)}</div>
              <div class="evidence-card-footer-mini">
                <span>${p.pubmed_url ? "PubMed source available" : "Corpus passage"}</span>
                <span class="evidence-chevron">▼</span>
              </div>
            </button>
            <div class="evidence-card-body">
              <p class="evidence-full-text">${escapeHtml(p.text)}</p>
              <div class="score-grid">
                ${scoreCells
                  .map(
                    (c) => `
                      <div class="score-cell">
                        <span>${escapeHtml(c.label)}</span>
                        <span class="mono">${escapeHtml(c.val)}</span>
                      </div>
                    `
                  )
                  .join("")}
              </div>
              ${
                matchedQueries.length
                  ? `
                    <div class="matched-queries-block">
                      <div class="matched-queries-label">Matched Query Provenance</div>
                      <div class="matched-query-pills">
                        ${matchedQueries
                          .map((mq) => {
                            const isOrig = mq.trim().toLowerCase() === origLower;
                            return `<span class="matched-query-pill">${isOrig ? "★ Orig: " : "Exp: "}${escapeHtml(mq)}</span>`;
                          })
                          .join("")}
                      </div>
                    </div>
                  `
                  : ""
              }
              ${
                p.pubmed_url
                  ? `<a class="pubmed-btn" href="${escapeHtml(p.pubmed_url)}" target="_blank" rel="noopener noreferrer">Open on PubMed (PMID ${escapeHtml(p.passage_id)}) ↗</a>`
                  : ""
              }
            </div>
          </div>
        `;
      })
      .join("");
  }

  function jumpToEvidencePassage(passageId) {
    if (!passageId) return;
    // Ensure drawer is open
    el.appShell.classList.remove("drawer-collapsed");

    let card = el.evidenceList.querySelector(`.evidence-card[data-passage-id="${CSS.escape(passageId)}"]`);
    if (!card && state.evidenceFilter !== "all") {
      state.evidenceFilter = "all";
      updateFilterTabUI();
      renderEvidenceDrawer();
      card = el.evidenceList.querySelector(`.evidence-card[data-passage-id="${CSS.escape(passageId)}"]`);
    }
    if (!card) return;

    card.classList.add("expanded");
    const hdr = card.querySelector(".evidence-card-header");
    if (hdr) hdr.setAttribute("aria-expanded", "true");

    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    card.classList.remove("flash-highlight");
    void card.offsetWidth;
    card.classList.add("flash-highlight");
    setTimeout(() => card.classList.remove("flash-highlight"), 1800);
  }

  // -------------------------------------------------------------------------
  // Event Listeners
  // -------------------------------------------------------------------------

  function bindEvents() {
    el.themeToggleBtn.addEventListener("click", () => {
      const curr = document.documentElement.getAttribute("data-theme") || "dark";
      applyTheme(curr === "dark" ? "light" : "dark");
    });

    el.newSearchBtn.addEventListener("click", resetToNewSearch);
    el.brandHomeLink.addEventListener("click", (e) => {
      e.preventDefault();
      resetToNewSearch();
    });

    el.toggleDrawerBtn.addEventListener("click", () => {
      el.appShell.classList.toggle("drawer-collapsed");
    });

    // Landing form submit
    el.landingForm.addEventListener("submit", (e) => {
      e.preventDefault();
      executeSearch(
        el.landingQueryInput.value,
        el.landingModeSelect.value,
        el.landingTopKSelect.value
      );
    });

    // Thread form submit
    el.threadForm.addEventListener("submit", (e) => {
      e.preventDefault();
      executeSearch(
        el.threadQueryInput.value,
        el.threadModeSelect.value,
        el.threadTopKSelect.value
      );
    });

    // Enter to submit (Shift+Enter for newline)
    [
      [el.landingQueryInput, el.landingForm],
      [el.threadQueryInput, el.threadForm],
    ].forEach(([textarea, form]) => {
      textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          form.requestSubmit();
        }
      });
    });

    // Example cards
    el.examplesGrid.addEventListener("click", (e) => {
      const btn = e.target.closest(".example-card");
      if (!btn) return;
      const query = btn.dataset.query || "";
      el.landingQueryInput.value = query;
      executeSearch(query, el.landingModeSelect.value, el.landingTopKSelect.value);
    });

    // History filter & click
    let filterTimer = null;
    el.historyFilterInput.addEventListener("input", () => {
      const val = el.historyFilterInput.value;
      el.clearHistoryFilterBtn.classList.toggle("hidden", !val);
      clearTimeout(filterTimer);
      filterTimer = setTimeout(() => loadHistory(val), 180);
    });

    el.clearHistoryFilterBtn.addEventListener("click", () => {
      el.historyFilterInput.value = "";
      el.clearHistoryFilterBtn.classList.add("hidden");
      loadHistory("");
    });

    el.historyList.addEventListener("click", (e) => {
      const item = e.target.closest(".history-item");
      if (!item) return;
      restoreHistoryItem(item.dataset.searchId);
    });

    // Inline citation badge clicks inside answer
    el.answerBody.addEventListener("click", (e) => {
      const badge = e.target.closest(".cite-badge");
      if (!badge) return;
      jumpToEvidencePassage(badge.dataset.citeId);
    });

    // Bottom Citations Footer clicks
    el.citationsGrid.addEventListener("click", (e) => {
      const card = e.target.closest(".citation-footer-card");
      if (!card) return;
      jumpToEvidencePassage(card.dataset.jumpId);
    });

    // Evidence drawer accordion & filters
    el.evidenceList.addEventListener("click", (e) => {
      const header = e.target.closest(".evidence-card-header");
      if (!header) return;
      const card = header.closest(".evidence-card");
      if (!card) return;
      const expanded = card.classList.toggle("expanded");
      header.setAttribute("aria-expanded", expanded ? "true" : "false");
    });

    el.filterTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        state.evidenceFilter = tab.dataset.filter || "all";
        updateFilterTabUI();
        renderEvidenceDrawer();
      });
    });

    el.expandAllBtn.addEventListener("click", () => {
      state.allExpanded = !state.allExpanded;
      renderEvidenceDrawer();
    });
  }

  // Boot
  async function init() {
    initTheme();
    bindEvents();
    await Promise.all([loadHealth(), loadConfig(), loadHistory()]);
  }

  init();
})();
