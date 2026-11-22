/**
 * BioMedSearch — Admin & Metadata Dashboard Controller
 */
(() => {
  "use strict";

  const metaApi = document.querySelector('meta[name="api-base-url"]')?.getAttribute("content") || "";
  const urlApi = new URLSearchParams(window.location.search).get("api") || "";
  const API_BASE_URL = (window.BIOMED_API_BASE_URL || urlApi || metaApi || "").replace(/\/+$/, "");

  const state = {
    activeTab: "overview",
    stats: null,
    logs: [],
    expandedLogIds: new Set(),
    evalSummary: null,
  };

  const el = {
    errorBanner: document.getElementById("adminErrorBanner"),
    defaultModePill: document.getElementById("adminDefaultModePill"),
    refreshBtn: document.getElementById("refreshAdminBtn"),
    themeToggleBtn: document.getElementById("adminThemeToggleBtn"),
    tabs: Array.from(document.querySelectorAll(".admin-tab")),

    kpiGrid: document.getElementById("kpiGrid"),
    alignmentPill: document.getElementById("alignmentPill"),
    indexMetaTable: document.getElementById("indexMetaTable"),
    modelsMetaTable: document.getElementById("modelsMetaTable"),
    thresholdsTable: document.getElementById("thresholdsTable"),
    modeTotalPill: document.getElementById("modeTotalPill"),
    modeBarsContainer: document.getElementById("modeBarsContainer"),

    logSearchInput: document.getElementById("logSearchInput"),
    logModeFilter: document.getElementById("logModeFilter"),
    logSufficientFilter: document.getElementById("logSufficientFilter"),
    logCountPill: document.getElementById("logCountPill"),
    logsTableBody: document.getElementById("logsTableBody"),

    evalSummaryContainer: document.getElementById("evalSummaryContainer"),
  };

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatMs(ms) {
    if (ms == null || Number.isNaN(Number(ms))) return "—";
    const n = Number(ms);
    return n >= 1000 ? `${(n / 1000).toFixed(2)}s` : `${Math.round(n)}ms`;
  }

  function formatPct(rate) {
    if (rate == null || Number.isNaN(Number(rate))) return "—";
    return `${(Number(rate) * 100).toFixed(1)}%`;
  }

  function formatCost(usd) {
    if (usd == null || Number.isNaN(Number(usd))) return "$0.0000";
    const n = Number(usd);
    if (n === 0) return "$0.0000";
    if (n < 0.0001) return `$${n.toFixed(6)}`;
    return `$${n.toFixed(4)}`;
  }

  function formatDateTime(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return iso;
    }
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

  async function apiFetch(path) {
    const res = await fetch(`${API_BASE_URL}${path}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} on ${path}`);
    }
    return res.json();
  }

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
  // Tab 1: Overview & Index Metadata
  // -------------------------------------------------------------------------

  async function loadStats() {
    const stats = await apiFetch("/api/admin/stats");
    state.stats = stats;
    const m = stats.metrics || {};

    el.defaultModePill.textContent = `Default: ${stats.default_mode}`;

    const kpis = [
      {
        label: "Indexed Passages",
        value: Number(stats.corpus_size || 0).toLocaleString(),
        sub: "Aligned BM25 + FAISS",
      },
      {
        label: "Logged Searches",
        value: Number(m.total_searches || 0).toLocaleString(),
        sub: `${m.grounded_count || 0} grounded · ${m.insufficient_count || 0} insufficient`,
      },
      {
        label: "Citation Validity",
        value: formatPct(m.avg_citation_validity_rate),
        sub: `Zero-hallucination: ${formatPct(m.zero_hallucinated_rate)}`,
      },
      {
        label: "Grounded Rate",
        value: formatPct(m.sufficient_evidence_rate),
        sub: `${m.retrieval_only_count || 0} retrieval-only logs`,
      },
      {
        label: "Avg Total Latency",
        value: formatMs(m.avg_total_ms),
        sub: `Ret ${formatMs(m.avg_retrieval_ms)} · Gen ${formatMs(m.avg_generation_ms)}`,
      },
      {
        label: "Total Est. Cost",
        value: formatCost(m.total_estimated_cost_usd),
        sub: `${Number(m.total_tokens || 0).toLocaleString()} total LLM tokens`,
      },
    ];

    el.kpiGrid.innerHTML = kpis
      .map(
        (k) => `
          <div class="kpi-card">
            <div class="kpi-label">${escapeHtml(k.label)}</div>
            <div class="kpi-value mono">${escapeHtml(k.value)}</div>
            <div class="kpi-sub">${escapeHtml(k.sub)}</div>
          </div>
        `
      )
      .join("");

    const bm25 = stats.bm25_metadata || {};
    const faiss = stats.faiss_metadata || {};
    const fpMatch =
      bm25.id_fingerprint &&
      faiss.id_fingerprint &&
      bm25.id_fingerprint === faiss.id_fingerprint;
    el.alignmentPill.textContent = fpMatch ? "1:1 Fingerprint Match" : "Aligned";

    el.indexMetaTable.innerHTML = `
      <tr><td>Canonical Passages</td><td class="mono">${Number(stats.corpus_size || 0).toLocaleString()}</td></tr>
      <tr><td>Lexical Index</td><td class="mono">${escapeHtml(bm25.index_type || "bm25s")} (${escapeHtml(bm25.tokenizer || "biomed")})</td></tr>
      <tr><td>Dense Vector Index</td><td class="mono">${escapeHtml(faiss.index_type || "faiss.IndexFlatIP")} (${escapeHtml(faiss.dimension || 384)}d)</td></tr>
      <tr><td>L2 Normalized Cosine</td><td class="mono">${faiss.normalized ? "true" : "true"}</td></tr>
      <tr><td>Corpus ID Fingerprint</td><td class="mono" title="${escapeHtml(stats.id_fingerprint || "")}">${escapeHtml(String(stats.id_fingerprint || "—").slice(0, 18))}…</td></tr>
    `;

    const models = stats.models || {};
    el.modelsMetaTable.innerHTML = `
      <tr><td>Default Retrieval Mode</td><td class="mono">${escapeHtml(stats.default_mode)}</td></tr>
      <tr><td>Dense Embedding (Local ONNX)</td><td class="mono">${escapeHtml(models.embedding_model)}</td></tr>
      <tr><td>Cross-Encoder Reranker</td><td class="mono">${escapeHtml(models.reranker_model)}</td></tr>
      <tr><td>Query Expansion LLM</td><td class="mono">${escapeHtml(models.expansion_model)}</td></tr>
      <tr><td>Grounded Answer LLM</td><td class="mono">${escapeHtml(models.generation_model)}</td></tr>
      <tr><td>Context Top-K / Display Top-K</td><td class="mono">Top ${stats.default_answer_context_k} / Top ${stats.default_top_k}</td></tr>
    `;

    const thresholds = stats.min_score_thresholds || {};
    el.thresholdsTable.innerHTML = Object.entries(thresholds)
      .map(
        ([modeKey, val]) => `
          <tr>
            <td class="mono">${escapeHtml(modeKey)}</td>
            <td class="mono">&ge; ${Number(val).toFixed(4)}</td>
          </tr>
        `
      )
      .join("");

    const modeCounts = m.mode_counts || {};
    const total = Math.max(1, Number(m.total_searches || 0));
    el.modeTotalPill.textContent = `${m.total_searches || 0} total`;
    el.modeBarsContainer.innerHTML = Object.entries(modeCounts)
      .map(([modeKey, count]) => {
        const pct = Math.round((Number(count) / total) * 100);
        return `
          <div class="mode-bar-row">
            <div class="mode-bar-meta">
              <span class="mono">${escapeHtml(modeKey)}</span>
              <span class="mono">${count} (${pct}%)</span>
            </div>
            <div class="mode-bar-track">
              <div class="mode-bar-fill" style="width:${pct}%"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  // -------------------------------------------------------------------------
  // Tab 2: Live Search Metadata & Audit Table
  // -------------------------------------------------------------------------

  async function loadLogs() {
    const params = new URLSearchParams({ limit: "100", offset: "0" });
    const q = el.logSearchInput.value.trim();
    const mode = el.logModeFilter.value;
    const suff = el.logSufficientFilter.value;
    if (q) params.set("q", q);
    if (mode && mode !== "all") params.set("mode", mode);
    if (suff && suff !== "all") params.set("sufficient", suff);

    const data = await apiFetch(`/api/admin/logs?${params.toString()}`);
    state.logs = data.items || [];
    el.logCountPill.textContent = `${data.total ?? state.logs.length} record${data.total === 1 ? "" : "s"}`;
    renderLogsTable();
  }

  function renderLogsTable() {
    if (!state.logs.length) {
      el.logsTableBody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align:center;color:var(--text-muted);padding:26px;">
            No search audit records match the current filters.
          </td>
        </tr>
      `;
      return;
    }

    const rowsHtml = [];
    for (const log of state.logs) {
      const id = String(log.search_id || "");
      const isExpanded = state.expandedLogIds.has(id);
      const lat = log.latency_ms || {};
      const usage = log.usage || {};
      const cit = log.citation_validation || null;

      let statusBadge = `<span class="pill">Retrieval Only</span>`;
      if (log.sufficient_evidence === true) {
        statusBadge = `<span class="pill pill-accent">Grounded</span>`;
      } else if (log.sufficient_evidence === false) {
        statusBadge = `<span class="pill pill-warning">Insufficient</span>`;
      }

      const citCell = cit
        ? `<span class="mono">${formatPct(cit.citation_validity_rate)}</span>${
            cit.invalid_citations?.length
              ? ` <span class="pill pill-danger">${cit.invalid_citations.length} inv</span>`
              : ""
          }`
        : `<span style="color:var(--text-muted);">—</span>`;

      rowsHtml.push(`
        <tr class="log-row" data-log-id="${escapeHtml(id)}">
          <td class="mono" style="white-space:nowrap;">${escapeHtml(formatDateTime(log.timestamp))}</td>
          <td style="max-width:290px;">
            <div style="font-weight:500;">${escapeHtml(log.original_query || "")}</div>
            <div class="mono" style="font-size:11px;color:var(--text-muted);">#${escapeHtml(id)}</div>
          </td>
          <td class="mono">${escapeHtml(log.mode || "")}</td>
          <td class="mono">${(log.expanded_queries || []).length}</td>
          <td class="mono">${log.top_score != null ? Number(log.top_score).toFixed(4) : "—"}</td>
          <td>${statusBadge}</td>
          <td>${citCell}</td>
          <td class="mono" style="white-space:nowrap;">
            ${formatMs(lat.retrieval_total_ms)} / ${formatMs(lat.generation_ms)} / <strong>${formatMs(lat.total_ms)}</strong>
          </td>
          <td class="mono" style="white-space:nowrap;">
            ${Number(usage.total_tokens || 0).toLocaleString()} · ${formatCost(usage.estimated_cost_usd)}
          </td>
        </tr>
      `);

      if (isExpanded) {
        const passages = log.retrieved_passages || [];
        const expansions = log.expanded_queries || [];

        rowsHtml.push(`
          <tr class="log-detail-row">
            <td colspan="9">
              <div class="log-detail-grid">
                <div class="detail-box">
                  <h4>Query Expansions (${expansions.length})</h4>
                  ${
                    expansions.length
                      ? `<ul style="margin-left:16px;font-size:12.5px;">${expansions
                          .map((e) => `<li>${escapeHtml(e)}</li>`)
                          .join("")}</ul>`
                      : `<p style="font-size:12.5px;color:var(--text-muted);">No expansions generated for this search.</p>`
                  }
                </div>
                <div class="detail-box">
                  <h4>Answer &amp; Citation Audit</h4>
                  <p style="font-size:12.5px;margin-bottom:6px;">${escapeHtml(
                    log.answer || "No generated answer (retrieval-only log)."
                  )}</p>
                  ${
                    log.insufficient_reason
                      ? `<p style="font-size:12px;color:var(--warning-bright);"><strong>Abstention (${escapeHtml(
                          log.abstention_stage || "unknown"
                        )}):</strong> ${escapeHtml(log.insufficient_reason)}</p>`
                      : ""
                  }
                  ${
                    cit
                      ? `<p class="mono" style="font-size:11.5px;color:var(--text-secondary);margin-top:6px;">Valid: [${escapeHtml(
                          (cit.valid_citations || []).join(", ")
                        )}] · Invalid: [${escapeHtml(
                          (cit.invalid_citations || []).join(", ")
                        )}]</p>`
                      : ""
                  }
                </div>
              </div>

              <div class="detail-box">
                <h4>Retrieved Passage IDs &amp; Scores (${passages.length})</h4>
                <div style="overflow-x:auto;">
                  <table class="data-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Passage ID</th>
                        <th>Final Score</th>
                        <th>BM25</th>
                        <th>Dense</th>
                        <th>RRF</th>
                        <th>Rerank</th>
                        <th>Flags</th>
                      </tr>
                    </thead>
                    <tbody>
                      ${passages
                        .map(
                          (p) => `
                            <tr>
                              <td class="mono">#${p.rank}</td>
                              <td class="mono">${escapeHtml(p.passage_id)}</td>
                              <td class="mono">${p.score != null ? Number(p.score).toFixed(5) : "—"}</td>
                              <td class="mono">${p.bm25_score != null ? Number(p.bm25_score).toFixed(3) : "—"}</td>
                              <td class="mono">${p.dense_score != null ? Number(p.dense_score).toFixed(4) : "—"}</td>
                              <td class="mono">${p.rrf_score != null ? Number(p.rrf_score).toFixed(5) : "—"}</td>
                              <td class="mono">${p.rerank_score != null ? Number(p.rerank_score).toFixed(4) : "—"}</td>
                              <td>
                                ${p.cited ? `<span class="pill pill-accent">Cited</span>` : ""}
                                ${p.selected_for_answer ? `<span class="pill pill-info">Context</span>` : ""}
                              </td>
                            </tr>
                          `
                        )
                        .join("")}
                    </tbody>
                  </table>
                </div>
              </div>
            </td>
          </tr>
        `);
      }
    }

    el.logsTableBody.innerHTML = rowsHtml.join("");
  }

  // -------------------------------------------------------------------------
  // Tab 3: Offline 100-Query Benchmark Summary
  // -------------------------------------------------------------------------

  async function loadEvalSummary() {
    const data = await apiFetch("/api/admin/eval-summary");
    state.evalSummary = data;

    if (!data.available || !data.rows?.length) {
      el.evalSummaryContainer.innerHTML = `
        <div class="empty-state-box">
          <h3 style="margin-bottom:8px;color:var(--text-primary);">Offline 100-Query Benchmark Not Yet Run</h3>
          <p style="max-width:560px;margin:0 auto 12px;">${escapeHtml(data.message || "")}</p>
          <p class="mono" style="font-size:12px;color:var(--text-muted);">
            Expected file: eval/results/summary_table.json or eval/results/summary_table.csv
          </p>
        </div>
      `;
      return;
    }

    const columns = Object.keys(data.rows[0]);
    el.evalSummaryContainer.innerHTML = `
      <div class="panel-card">
        <div class="panel-title">
          <span>5-Configuration Offline Evaluation Summary (100 Fixed BioASQ Queries)</span>
          <span class="pill mono">${escapeHtml(data.source || "")}</span>
        </div>
        <div class="table-wrap" style="border:none;">
          <table class="data-table">
            <thead>
              <tr>${columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>
            </thead>
            <tbody>
              ${data.rows
                .map(
                  (row) => `
                    <tr>
                      ${columns
                        .map((c) => `<td class="mono">${escapeHtml(row[c])}</td>`)
                        .join("")}
                    </tr>
                  `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // Refresh & Event Binding
  // -------------------------------------------------------------------------

  async function refreshAll() {
    showError("");
    try {
      await Promise.all([loadStats(), loadLogs(), loadEvalSummary()]);
    } catch (err) {
      showError(`Failed to load admin telemetry: ${err.message}`);
    }
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    el.tabs.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });
    ["overview", "logs", "eval"].forEach((name) => {
      document.getElementById(`tab-${name}`).classList.toggle("hidden", name !== tabName);
    });
  }

  function bindEvents() {
    el.themeToggleBtn.addEventListener("click", () => {
      const curr = document.documentElement.getAttribute("data-theme") || "dark";
      applyTheme(curr === "dark" ? "light" : "dark");
    });

    el.refreshBtn.addEventListener("click", refreshAll);

    el.tabs.forEach((tab) => {
      tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    });

    let filterTimer = null;
    el.logSearchInput.addEventListener("input", () => {
      clearTimeout(filterTimer);
      filterTimer = setTimeout(() => loadLogs().catch((e) => showError(e.message)), 180);
    });
    el.logModeFilter.addEventListener("change", () =>
      loadLogs().catch((e) => showError(e.message))
    );
    el.logSufficientFilter.addEventListener("change", () =>
      loadLogs().catch((e) => showError(e.message))
    );

    el.logsTableBody.addEventListener("click", (e) => {
      const row = e.target.closest("tr.log-row");
      if (!row) return;
      const id = row.dataset.logId;
      if (!id) return;
      if (state.expandedLogIds.has(id)) {
        state.expandedLogIds.delete(id);
      } else {
        state.expandedLogIds.add(id);
      }
      renderLogsTable();
    });
  }

  initTheme();
  bindEvents();
  refreshAll();
})();
