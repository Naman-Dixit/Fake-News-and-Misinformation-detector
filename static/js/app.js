/* ==========================================================================
   Veritas — app.js
   Handles mode switching, API mode/key inputs, single-article analysis,
   bulk CSV analysis, the animated pipeline tracker, credibility gauge,
   verdict stamp, collapsible claim cards, and the bulk results table.
   ========================================================================== */

(function () {
  "use strict";

  // ---------------------------------------------------------------------
  // Mode switching
  // ---------------------------------------------------------------------
  const modeBtns = document.querySelectorAll(".mode-btn");
  const panels = { single: document.getElementById("panel-single"), bulk: document.getElementById("panel-bulk") };

  modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeBtns.forEach((b) => { b.classList.remove("is-active"); b.setAttribute("aria-selected", "false"); });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");
      const mode = btn.dataset.mode;
      Object.entries(panels).forEach(([key, panel]) => panel.classList.toggle("is-active", key === mode));
    });
  });

  // ---------------------------------------------------------------------
  // Engine / API key selector
  // ---------------------------------------------------------------------
  const apiModeSelect = document.getElementById("apiMode");
  const apiKeyInput = document.getElementById("apiKey");

  apiModeSelect.addEventListener("change", () => {
    const mode = apiModeSelect.value;
    if (mode === "offline") {
      apiKeyInput.classList.add("hidden");
    } else {
      apiKeyInput.classList.remove("hidden");
      apiKeyInput.placeholder = mode === "nvidia" ? "Paste NVIDIA API key…" : "Paste Anthropic API key…";
    }
  });

  function currentApiCreds() {
    return { api_mode: apiModeSelect.value, api_key: apiKeyInput.value.trim() };
  }

  // ---------------------------------------------------------------------
  // Toast
  // ---------------------------------------------------------------------
  let toastEl = null;
  function showToast(message) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "toast";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = message;
    toastEl.classList.add("is-visible");
    clearTimeout(toastEl._timer);
    toastEl._timer = setTimeout(() => toastEl.classList.remove("is-visible"), 5000);
  }

  // ---------------------------------------------------------------------
  // Pipeline tracker animation
  // ---------------------------------------------------------------------
  const pipelineSteps = Array.from(document.querySelectorAll("#pipelineTracker .pipeline-step"));

  function resetPipeline() {
    pipelineSteps.forEach((s) => s.classList.remove("is-active", "is-done"));
  }

  async function animatePipeline() {
    resetPipeline();
    for (let i = 0; i < pipelineSteps.length; i++) {
      pipelineSteps[i].classList.add("is-active");
      await new Promise((r) => setTimeout(r, 420));
      pipelineSteps[i].classList.remove("is-active");
      pipelineSteps[i].classList.add("is-done");
    }
  }

  // ---------------------------------------------------------------------
  // Single article analysis
  // ---------------------------------------------------------------------
  const singleForm = document.getElementById("singleForm");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const singleEmptyState = document.getElementById("singleEmptyState");
  const singleResults = document.getElementById("singleResults");

  const gaugeFill = document.getElementById("gaugeFill");
  const gaugeNeedle = document.getElementById("gaugeNeedle");
  const gaugeScore = document.getElementById("gaugeScore");
  const verdictStamp = document.getElementById("verdictStamp");
  const verdictMeta = document.getElementById("verdictMeta");
  const metricTags = document.getElementById("metricTags");
  const semanticGrid = document.getElementById("semanticGrid");
  const claimsList = document.getElementById("claimsList");
  const claimCardTemplate = document.getElementById("claimCardTemplate");

  const GAUGE_CIRCUMFERENCE = 283; // matches the arc path length approximation

  function statusClass(status) {
    return "status-" + status.toLowerCase().replace(/\s+/g, "-");
  }

  function voteClass(vote) {
    return "vote-" + vote.toLowerCase();
  }

  function renderGauge(score) {
    const pct = Math.max(0, Math.min(100, score)) / 100;
    const offset = GAUGE_CIRCUMFERENCE * (1 - pct);
    // slight delay so the transition is visible after being set to full offset
    gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = offset;
    });

    let color = "var(--verdict-false)";
    if (score >= 70) color = "var(--verdict-true)";
    else if (score >= 40) color = "var(--verdict-suspect)";
    gaugeFill.style.stroke = color;

    // Needle sweeps from -90deg (score 0) to +90deg (score 100)
    const angle = -90 + pct * 180;
    gaugeNeedle.style.transform = `rotate(${angle}deg)`;
    gaugeScore.textContent = Math.round(score);
  }

  function renderVerdictStamp(verdict) {
    verdictStamp.textContent = verdict.toUpperCase();
    verdictStamp.classList.remove("stamp-true", "stamp-suspect", "stamp-false");
    verdictStamp.classList.add("stamp-" + verdict.toLowerCase());
  }

  function renderMetricTags(result) {
    metricTags.innerHTML = "";
    const tags = [
      ["Source Tier", result.source_tier_label],
      ["Authority ×", result.authority_multiplier.toFixed(2)],
      ["Claims Found", result.claim_count],
      ["Model Disagreement", `${result.model_disagreement.label} (${result.model_disagreement.value})`],
    ];
    tags.forEach(([label, value]) => {
      const el = document.createElement("span");
      el.className = "metric-tag";
      el.innerHTML = `${label}: <b>${value}</b>`;
      metricTags.appendChild(el);
    });
  }

  const SEMANTIC_LABELS = {
    bias: "Bias",
    propaganda: "Propaganda",
    logical_fallacies: "Logical Fallacies",
    emotional_manipulation: "Emotional Manipulation",
    context_stripping: "Context Stripping",
  };

  function renderSemanticGrid(semantics) {
    semanticGrid.innerHTML = "";
    Object.entries(SEMANTIC_LABELS).forEach(([key, label]) => {
      const item = semantics[key];
      if (!item) return;
      const card = document.createElement("div");
      card.className = "semantic-item" + (item.detected ? " flagged" : "");
      card.innerHTML = `
        <div class="semantic-item-head">
          <span class="name">${label}</span>
          <span class="semantic-flag ${item.detected ? "yes" : "no"}">${item.detected ? "Flagged" : "Clear"}</span>
        </div>
        <p>${item.detail}</p>
      `;
      semanticGrid.appendChild(card);
    });

    const consistencyCard = document.createElement("div");
    consistencyCard.className = "semantic-item";
    consistencyCard.innerHTML = `
      <div class="semantic-item-head"><span class="name">Consistency Score</span></div>
      <p>${semantics.consistency_score} / 5 — used as the "L" term in the ensemble score.</p>
    `;
    semanticGrid.appendChild(consistencyCard);
  }

  function renderClaims(claims) {
    claimsList.innerHTML = "";
    claims.forEach((claim) => {
      const node = claimCardTemplate.content.cloneNode(true);
      const card = node.querySelector(".claim-card");
      node.querySelector(".claim-id").textContent = claim.claim_id;
      node.querySelector(".claim-statement").textContent = claim.statement;

      const badge = node.querySelector(".claim-status-badge");
      badge.textContent = claim.status;
      badge.classList.add(statusClass(claim.status));

      node.querySelector(".claim-alleged-source").textContent = claim.alleged_source;
      node.querySelector(".claim-category").textContent = claim.verification_category;
      node.querySelector(".claim-authority").textContent = `Tier ${claim.domain_authority}`;
      node.querySelector(".claim-confidence").textContent = `${Math.round(claim.confidence * 100)}%`;
      node.querySelector(".claim-score").textContent = `${claim.claim_score} / 100`;
      node.querySelector(".claim-evidence").textContent = claim.evidence_summary;

      node.querySelector(".claim-card-head").addEventListener("click", () => {
        card.classList.toggle("is-open");
      });

      claimsList.appendChild(node);
    });
  }

  singleForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const { api_mode, api_key } = currentApiCreds();
    if (api_mode !== "offline" && !api_key) {
      showToast(`Please enter your ${api_mode === "nvidia" ? "NVIDIA" : "Anthropic"} API key, or switch to Offline mode.`);
      return;
    }

    const payload = {
      title: document.getElementById("f-title").value.trim(),
      text: document.getElementById("f-text").value.trim(),
      source: document.getElementById("f-source").value.trim(),
      topic: document.getElementById("f-topic").value.trim(),
      api_mode,
      api_key,
    };

    analyzeBtn.disabled = true;
    analyzeBtn.querySelector("span").textContent = "Analyzing…";
    singleEmptyState.classList.add("hidden");
    singleResults.classList.add("hidden");

    const pipelinePromise = animatePipeline();

    try {
      const [resp] = await Promise.all([
        fetch("/api/single/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
        pipelinePromise,
      ]);

      const data = await resp.json();
      if (!resp.ok) {
        showToast(data.error || "Analysis failed.");
        singleEmptyState.classList.remove("hidden");
        return;
      }

      renderGauge(data.credibility_score);
      renderVerdictStamp(data.verdict);
      verdictMeta.innerHTML = `
        <div><b>${data.article.title}</b></div>
        <div>${data.article.source} · ${data.article.topic}</div>
        <div>Engine: ${data.api_mode}</div>
      `;
      renderMetricTags(data);
      renderSemanticGrid(data.semantic_analysis);
      renderClaims(data.claims);

      singleResults.classList.remove("hidden");
    } catch (err) {
      showToast("Network error while contacting the analysis pipeline.");
      singleEmptyState.classList.remove("hidden");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector("span").textContent = "Analyze Article";
      resetPipeline();
    }
  });

  // ---------------------------------------------------------------------
  // Bulk detection
  // ---------------------------------------------------------------------
  const dropzone = document.getElementById("dropzone");
  const csvInput = document.getElementById("csvInput");
  const dropzoneFile = document.getElementById("dropzoneFile");
  const bulkAnalyzeBtn = document.getElementById("bulkAnalyzeBtn");
  const bulkProgressWrap = document.getElementById("bulkProgressWrap");
  const bulkProgressFill = document.getElementById("bulkProgressFill");
  const bulkProgressLabel = document.getElementById("bulkProgressLabel");
  const bulkResults = document.getElementById("bulkResults");
  const metricsRow = document.getElementById("metricsRow");
  const confusionWrap = document.getElementById("confusionWrap");
  const resultsTableBody = document.getElementById("resultsTableBody");
  const downloadCsvBtn = document.getElementById("downloadCsvBtn");

  let selectedFile = null;
  let lastDownloadId = null;

  dropzone.addEventListener("click", () => csvInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("is-dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
    if (e.dataTransfer.files.length) {
      selectedFile = e.dataTransfer.files[0];
      dropzoneFile.textContent = selectedFile.name;
    }
  });
  csvInput.addEventListener("change", () => {
    if (csvInput.files.length) {
      selectedFile = csvInput.files[0];
      dropzoneFile.textContent = selectedFile.name;
    }
  });

  function fakeProgressTicker() {
    let pct = 0;
    bulkProgressFill.style.width = "0%";
    return setInterval(() => {
      pct = Math.min(pct + Math.random() * 9, 92);
      bulkProgressFill.style.width = pct + "%";
    }, 260);
  }

  bulkAnalyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showToast("Please select a CSV file first.");
      return;
    }

    const { api_mode, api_key } = currentApiCreds();
    if (api_mode !== "offline" && !api_key) {
      showToast(`Please enter your ${api_mode === "nvidia" ? "NVIDIA" : "Anthropic"} API key, or switch to Offline mode.`);
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("api_mode", api_mode);
    formData.append("api_key", api_key);

    bulkAnalyzeBtn.disabled = true;
    bulkProgressWrap.classList.remove("hidden");
    bulkProgressLabel.textContent = "Processing rows through the AI pipeline…";
    const ticker = fakeProgressTicker();

    try {
      const resp = await fetch("/api/bulk/analyze", { method: "POST", body: formData });
      const data = await resp.json();

      clearInterval(ticker);
      bulkProgressFill.style.width = "100%";
      bulkProgressLabel.textContent = data.row_count ? `Processed ${data.row_count} articles.` : "Done.";

      if (!resp.ok) {
        showToast(data.error || "Bulk analysis failed.");
        return;
      }

      lastDownloadId = data.download_id;
      renderBulkResults(data.results, data.metrics);
      bulkResults.classList.remove("hidden");
    } catch (err) {
      clearInterval(ticker);
      showToast("Network error while contacting the bulk analysis endpoint.");
    } finally {
      bulkAnalyzeBtn.disabled = false;
      setTimeout(() => bulkProgressWrap.classList.add("hidden"), 1200);
    }
  });

  function renderBulkResults(results, metrics) {
    metricsRow.innerHTML = "";
    confusionWrap.innerHTML = "";

    if (metrics) {
      const cards = [
        ["Accuracy", (metrics.accuracy * 100).toFixed(1) + "%"],
        ["Precision", (metrics.precision * 100).toFixed(1) + "%"],
        ["Recall", (metrics.recall * 100).toFixed(1) + "%"],
        ["F1 Score", (metrics.f1_score * 100).toFixed(1) + "%"],
      ];
      cards.forEach(([name, value]) => {
        const card = document.createElement("div");
        card.className = "metric-card";
        card.innerHTML = `<div class="metric-value">${value}</div><div class="metric-name">${name}</div>`;
        metricsRow.appendChild(card);
      });

      const labels = metrics.labels;
      let table = `<h4>Confusion Matrix</h4><table class="confusion-table"><thead><tr><th>Actual \\ Predicted</th>`;
      labels.forEach((l) => (table += `<th>${l}</th>`));
      table += `</tr></thead><tbody>`;
      labels.forEach((actual) => {
        table += `<tr><th>${actual}</th>`;
        labels.forEach((predicted) => {
          const val = metrics.confusion_matrix[actual][predicted];
          table += `<td class="${actual === predicted ? "diag" : ""}">${val}</td>`;
        });
        table += `</tr>`;
      });
      table += `</tbody></table>`;
      confusionWrap.innerHTML = table;
    } else {
      metricsRow.innerHTML = `<div class="metric-card"><div class="metric-value">—</div><div class="metric-name">No ground_truth column found — metrics unavailable</div></div>`;
    }

    resultsTableBody.innerHTML = "";
    results.forEach((row) => {
      const tr = document.createElement("tr");
      let matchCell = '<span class="match-na">—</span>';
      if (row.match === true) matchCell = '<span class="match-yes">✓ Match</span>';
      else if (row.match === false) matchCell = '<span class="match-no">✗ Miss</span>';

      tr.innerHTML = `
        <td>${row.article_id || ""}</td>
        <td>${row.title || ""}</td>
        <td>${row.topic || ""}</td>
        <td>${row.source || ""}</td>
        <td>${row.credibility_score}</td>
        <td><span class="vote-pill ${voteClass(row.ensemble_vote)}">${row.ensemble_vote}</span></td>
        <td>${row.claim_count}</td>
        <td>${row.model_disagreement} (${row.model_disagreement_value})</td>
        <td>${row.ground_truth || "—"}</td>
        <td>${matchCell}</td>
      `;
      resultsTableBody.appendChild(tr);
    });
  }

  downloadCsvBtn.addEventListener("click", () => {
    if (!lastDownloadId) {
      showToast("Run a bulk analysis first.");
      return;
    }
    window.location.href = `/api/bulk/download/${lastDownloadId}`;
  });
})();
