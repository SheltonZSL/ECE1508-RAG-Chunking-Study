function byId(id) {
  return document.getElementById(id);
}

function text(id, value) {
  const node = byId(id);
  if (node) node.textContent = value;
}

function num(id, fallback) {
  const value = Number(byId(id)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function numFloat(id, fallback) {
  const value = Number.parseFloat(byId(id)?.value ?? "");
  return Number.isFinite(value) ? value : fallback;
}

function boolValue(id, fallback = false) {
  const raw = String(byId(id)?.value ?? "").trim().toLowerCase();
  if (raw === "true" || raw === "1" || raw === "yes" || raw === "on") return true;
  if (raw === "false" || raw === "0" || raw === "no" || raw === "off") return false;
  return fallback;
}

function rerankerLabel(settings) {
  if (!settings || !settings.reranker_enabled) return "reranker: off";
  const type = settings.reranker_type || "overlap";
  const candidateK = Number(settings.reranker_candidate_k || 0);
  const alpha = Number(settings.reranker_alpha || 0);
  return `reranker: ${type} (k=${candidateK}, alpha=${alpha.toFixed(2)})`;
}

function setBusy(isBusy, mode = "single") {
  const askBtn = byId("askBtn");
  const compareBtn = byId("compareBtn");
  if (askBtn) {
    askBtn.disabled = isBusy;
    askBtn.textContent = isBusy && mode === "single" ? "Running..." : "Run Query";
  }
  if (compareBtn) {
    compareBtn.disabled = isBusy;
    compareBtn.textContent = isBusy && mode === "compare" ? "Comparing..." : "Compare A vs B";
  }
}

function setHealth(kind, label) {
  const node = byId("demoHealth");
  if (!node) return;
  node.classList.remove("status-ok", "status-warn", "status-err");
  node.classList.add(kind);
  node.textContent = label;
}

function setNotice(kind, label) {
  const node = byId("demoNotice");
  if (!node) return;
  node.classList.remove("is-loading", "is-success", "is-warn", "is-error");
  node.classList.add(kind);
  node.textContent = label;
}

function setOutputLoading(isLoading) {
  byId("outputPanel")?.classList.toggle("is-loading", isLoading);
  byId("demoOutputSkeleton")?.classList.toggle("hidden", !isLoading);
}

function syncRerankerControls(suffix = "") {
  const scope = suffix === "B" ? "B" : "A";
  const enabled = boolValue(`rerankerEnabledInput${suffix}`, false);
  document
    .querySelectorAll(`.reranker-advanced[data-reranker-scope="${scope}"]`)
    .forEach((node) => {
      node.classList.toggle("hidden", !enabled);
      node.querySelectorAll("input,select,textarea,button").forEach((control) => {
        control.disabled = !enabled;
      });
    });
}

function syncTopKCandidateGuard(suffix = "") {
  const topK = num(`topKInput${suffix}`, 5);
  const candidate = byId(`rerankerCandidateKInput${suffix}`);
  if (!candidate) return;
  candidate.min = String(Math.max(1, topK));
  const value = Number(candidate.value || 0);
  if (!Number.isFinite(value) || value < topK) {
    candidate.value = String(topK);
  }
}

function escapeHtml(input) {
  return String(input)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function highlightText(textValue, queryValue) {
  const safe = escapeHtml(textValue || "");
  const words = [...new Set((queryValue || "").toLowerCase().match(/[a-z0-9]{4,}/g) || [])].slice(0, 8);
  if (!words.length) return safe;
  const escapedWords = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`\\b(${escapedWords.join("|")})\\b`, "gi");
  return safe.replace(re, "<mark>$1</mark>");
}

function setSelectOptions(id, values, selectedValue) {
  const select = byId(id);
  if (!select) return;
  select.innerHTML = "";
  for (const value of values || []) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  if (selectedValue && (values || []).includes(selectedValue)) {
    select.value = selectedValue;
  }
}

function renderSettingsBadges(settings, wrapId = "settingsBadges") {
  const wrap = byId(wrapId);
  if (!wrap) return;
  wrap.innerHTML = "";
  if (!settings) return;

  const items = [
    `backend: ${settings.backend}`,
    `strategy: ${settings.strategy}`,
    `chunk: ${settings.chunk_size}`,
    `overlap: ${settings.overlap}`,
    `top_k: ${settings.top_k}`,
    rerankerLabel(settings),
    `generation: ${settings.with_generation ? "on" : "off"}`,
  ];
  items.forEach((item) => {
    const span = document.createElement("span");
    span.className = "badge";
    span.textContent = item;
    wrap.appendChild(span);
  });
}

function renderPromptChips(examples) {
  const wrap = byId("promptChips");
  if (!wrap) return;
  wrap.innerHTML = "";
  const defaults = [
    "Who discovered penicillin?",
    "What is the capital of Japan?",
    "When was the first iPhone released?",
    "Which planet is known as the Red Planet?",
  ];
  const prompts = [...new Set([...(examples || []), ...defaults])].slice(0, 8);
  prompts.forEach((prompt) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "prompt-chip";
    chip.textContent = prompt;
    chip.addEventListener("click", () => {
      const input = byId("questionInput");
      if (input) input.value = prompt;
      input?.focus();
    });
    wrap.appendChild(chip);
  });
}

function hitCard(hit, maxScore, queryText) {
  const score = Number(hit.score || 0);
  const ratio = maxScore > 0 ? Math.max(0.04, Math.min(1, score / maxScore)) : 0.1;
  const details = document.createElement("details");
  details.className = "hit-card";
  details.open = hit.rank <= 2;

  details.innerHTML = `
    <summary>
      <div class="hit-main">
        <span class="hit-rank">#${hit.rank}</span>
        <span class="hit-score">score ${score.toFixed(4)}</span>
        <span class="hit-id">${escapeHtml(hit.chunk_id || "")}</span>
      </div>
      <div class="score-bar"><span style="width:${(ratio * 100).toFixed(1)}%"></span></div>
    </summary>
    <div class="hit-body">
      <div class="hit-meta">
        <span>document: ${escapeHtml(hit.document_id || "")}</span>
        <span>query: ${escapeHtml(hit.query_id || "")}</span>
      </div>
      <pre>${highlightText(hit.chunk_text || "", queryText)}</pre>
    </div>
  `;
  return details;
}

function renderHits(hits, queryText, wrapId = "hitsOutput", limit = null) {
  const wrap = byId(wrapId);
  if (!wrap) return;
  wrap.innerHTML = "";
  if (!Array.isArray(hits) || !hits.length) {
    wrap.innerHTML = "<p>No hits returned.</p>";
    return;
  }
  const rows = limit == null ? hits : hits.slice(0, limit);
  const maxScore = Math.max(...rows.map((h) => Number(h.score || 0)));
  rows.forEach((hit) => wrap.appendChild(hitCard(hit, maxScore, queryText)));
}

function renderResponse(payload, requestPayload) {
  text("o-retrieval-latency", `${Number(payload.timings_ms?.retrieval || 0).toFixed(2)} ms`);
  text("o-generation-latency", `${Number(payload.timings_ms?.generation || 0).toFixed(2)} ms`);
  text("o-total-latency", `${Number(payload.timings_ms?.total || 0).toFixed(2)} ms`);
  text("o-context-len", `${payload.context_char_len || 0} chars`);
  text("answerOutput", payload.answer || "(empty answer)");
  text("responseTag", `Run complete | ${payload.settings?.experiment_name || ""}`);
  text("payloadOutput", JSON.stringify(requestPayload, null, 2));
  renderSettingsBadges(payload.settings, "settingsBadges");
  renderHits(payload.hits || [], requestPayload.question, "hitsOutput");
}

function renderCompareResult(slot, payload, requestPayload) {
  const prefix = slot === "A" ? "c-a" : "c-b";
  text(`${prefix}-answer`, payload.answer || "(empty answer)");
  text(
    `${prefix}-meta`,
    `${payload.settings?.backend || "-"} / ${payload.settings?.strategy || "-"} | ${rerankerLabel(
      payload.settings
    )} | total ${Number(
      payload.timings_ms?.total || 0
    ).toFixed(2)} ms`
  );
  renderSettingsBadges(payload.settings, `${prefix}-settings`);
  renderHits(payload.hits || [], requestPayload.question, `${prefix}-hits`, 3);
}

function buildPayloadFromInputs(suffix = "") {
  const question = (byId("questionInput")?.value || "").trim();
  const topK = num(`topKInput${suffix}`, 5);
  const rerankerEnabled = boolValue(`rerankerEnabledInput${suffix}`, false);
  const rerankerCandidateK = Math.max(topK, num(`rerankerCandidateKInput${suffix}`, 20));
  const rerankerAlpha = Math.max(0, Math.min(1, numFloat(`rerankerAlphaInput${suffix}`, 0.5)));
  return {
    question,
    backend: byId(`backendInput${suffix}`)?.value || "dense",
    strategy: byId(`strategyInput${suffix}`)?.value || "fixed",
    chunk_size: num(`chunkSizeInput${suffix}`, 256),
    overlap: num(`overlapInput${suffix}`, 32),
    top_k: topK,
    reranker_enabled: rerankerEnabled,
    reranker_type: byId(`rerankerTypeInput${suffix}`)?.value || "overlap",
    reranker_candidate_k: rerankerCandidateK,
    reranker_alpha: rerankerAlpha,
    config: byId("configInput")?.value || "configs/portable_interactive.yaml",
    with_generation: !!byId("generateInput")?.checked,
  };
}

async function askApi(requestPayload) {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload),
  });
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload.error || "Request failed.");
  }
  return payload;
}

async function bootstrap() {
  setNotice("is-loading", "Loading service defaults...");
  setHealth("status-warn", "Checking...");
  try {
    const health = await fetch("/api/health", { cache: "no-store" });
    if (!health.ok) throw new Error("Health check failed");
    setHealth("status-ok", "Online");
  } catch {
    setHealth("status-err", "Offline");
  }

  try {
    const defaultsRes = await fetch("/api/defaults", { cache: "no-store" });
    if (defaultsRes.ok) {
      const payload = await defaultsRes.json();
      const opts = payload.options || {};
      const defs = payload.defaults || {};
      const backends = opts.backends || ["dense", "bm25"];
      const strategies = opts.strategies || ["fixed", "structure", "adaptive"];
      const rerankerEnableds = opts.reranker_enableds || [false, true];
      const rerankerTypes = opts.reranker_types || ["overlap"];

      setSelectOptions("backendInput", backends, defs.backend);
      setSelectOptions("strategyInput", strategies, defs.strategy);
      setSelectOptions("backendInputB", backends, backends.length > 1 ? backends[1] : defs.backend);
      setSelectOptions("strategyInputB", strategies, defs.strategy);
      setSelectOptions(
        "rerankerEnabledInput",
        rerankerEnableds.map((value) => String(Boolean(value))),
        String(Boolean(defs.reranker_enabled))
      );
      setSelectOptions(
        "rerankerEnabledInputB",
        rerankerEnableds.map((value) => String(Boolean(value))),
        String(Boolean(defs.reranker_enabled))
      );
      setSelectOptions("rerankerTypeInput", rerankerTypes, defs.reranker_type || "overlap");
      setSelectOptions("rerankerTypeInputB", rerankerTypes, defs.reranker_type || "overlap");

      if (byId("chunkSizeInput")) byId("chunkSizeInput").value = defs.chunk_size ?? 256;
      if (byId("overlapInput")) byId("overlapInput").value = defs.overlap ?? 32;
      if (byId("topKInput")) byId("topKInput").value = defs.top_k ?? 5;
      if (byId("rerankerCandidateKInput")) {
        byId("rerankerCandidateKInput").value = defs.reranker_candidate_k ?? Math.max(defs.top_k ?? 5, 20);
      }
      if (byId("rerankerAlphaInput")) byId("rerankerAlphaInput").value = defs.reranker_alpha ?? 0.5;

      if (byId("chunkSizeInputB")) byId("chunkSizeInputB").value = defs.chunk_size ?? 256;
      if (byId("overlapInputB")) byId("overlapInputB").value = defs.overlap ?? 32;
      if (byId("topKInputB")) byId("topKInputB").value = defs.top_k ?? 5;
      if (byId("rerankerCandidateKInputB")) {
        byId("rerankerCandidateKInputB").value = defs.reranker_candidate_k ?? Math.max(defs.top_k ?? 5, 20);
      }
      if (byId("rerankerAlphaInputB")) byId("rerankerAlphaInputB").value = defs.reranker_alpha ?? 0.5;

      if (byId("configInput")) byId("configInput").value = payload.config_path || "configs/portable_interactive.yaml";
      text("activeConfig", payload.config_path || "-");
      text("availableBackends", backends.join(", ") || "-");
      text("availableStrategies", strategies.join(", ") || "-");
      const rerankerModesLabel = rerankerEnableds.some((value) => Boolean(value)) ? "off/on" : "off";
      text("availableRerankers", `${rerankerModesLabel} | ${rerankerTypes.join(", ") || "overlap"}`);
      syncTopKCandidateGuard("");
      syncTopKCandidateGuard("B");
      syncRerankerControls("");
      syncRerankerControls("B");
      setNotice("is-success", "Service ready.");
    } else {
      setNotice("is-error", "Failed to load defaults. Check backend service.");
    }
  } catch {
    text("activeConfig", "Unavailable");
    text("availableBackends", "-");
    text("availableStrategies", "-");
    text("availableRerankers", "-");
    setNotice("is-error", "Failed to load defaults. Check backend service.");
  }

  try {
    const examplesRes = await fetch("/api/examples?limit=6", { cache: "no-store" });
    if (examplesRes.ok) {
      const payload = await examplesRes.json();
      renderPromptChips(payload.examples || []);
      if ((payload.examples || []).length && byId("questionInput") && !byId("questionInput").value.trim()) {
        byId("questionInput").value = payload.examples[0];
      }
      return;
    }
  } catch {
    // noop
  }
  renderPromptChips([]);
}

async function ask(event) {
  event.preventDefault();
  const requestPayload = buildPayloadFromInputs("");
  if (!requestPayload.question) {
    text("demoStatus", "Please enter a question.");
    setNotice("is-warn", "Question is required.");
    return;
  }

  setBusy(true, "single");
  setOutputLoading(true);
  setNotice("is-loading", "Running single query...");
  text("demoStatus", "Running query...");
  text("responseTag", "Request in progress...");

  try {
    const payload = await askApi(requestPayload);
    renderResponse(payload, requestPayload);
    setNotice("is-success", "Query finished successfully.");
    text(
      "demoStatus",
      `Done | backend=${payload.settings.backend}, strategy=${payload.settings.strategy}, ${rerankerLabel(
        payload.settings
      )}`
    );
  } catch (err) {
    setNotice("is-error", `Query failed: ${err.message}`);
    text("demoStatus", `Error: ${err.message}`);
    text("responseTag", "Run failed");
  } finally {
    setOutputLoading(false);
    setBusy(false);
  }
}

async function askCompare() {
  const requestA = buildPayloadFromInputs("");
  const requestB = buildPayloadFromInputs("B");

  if (!requestA.question) {
    text("demoStatus", "Please enter a question.");
    setNotice("is-warn", "Question is required.");
    return;
  }

  setBusy(true, "compare");
  setOutputLoading(true);
  setNotice("is-loading", "Running A/B comparison...");
  text("demoStatus", "Comparing A vs B...");
  text("compareSummary", "Comparison in progress...");
  text("responseTag", "A/B request in progress...");

  const [resA, resB] = await Promise.allSettled([askApi(requestA), askApi(requestB)]);

  try {
    if (resA.status === "fulfilled") {
      renderCompareResult("A", resA.value, requestA);
      renderResponse(resA.value, requestA);
    } else {
      text("c-a-meta", `Failed: ${resA.reason?.message || "unknown error"}`);
      text("c-a-answer", "Run A failed.");
      byId("c-a-hits").innerHTML = "";
      byId("c-a-settings").innerHTML = "";
    }

    if (resB.status === "fulfilled") {
      renderCompareResult("B", resB.value, requestB);
    } else {
      text("c-b-meta", `Failed: ${resB.reason?.message || "unknown error"}`);
      text("c-b-answer", "Run B failed.");
      byId("c-b-hits").innerHTML = "";
      byId("c-b-settings").innerHTML = "";
    }

    if (resA.status === "fulfilled" && resB.status === "fulfilled") {
      const aLatency = Number(resA.value.timings_ms?.total || 0);
      const bLatency = Number(resB.value.timings_ms?.total || 0);
      const faster = aLatency <= bLatency ? "A" : "B";
      const delta = Math.abs(aLatency - bLatency).toFixed(2);
      text("compareSummary", `Completed. Run ${faster} is faster by ${delta} ms.`);
      text("demoStatus", "Comparison done.");
      text("responseTag", "A/B comparison complete");
      setNotice("is-success", "A/B comparison completed.");
    } else if (resA.status === "rejected" && resB.status === "rejected") {
      throw new Error("Both A and B failed.");
    } else {
      text("compareSummary", "Comparison partially completed (one run failed).");
      text("demoStatus", "Comparison partially completed.");
      text("responseTag", "A/B comparison partial");
      setNotice("is-warn", "A/B completed with partial failure.");
    }
  } catch (err) {
    setNotice("is-error", `A/B failed: ${err.message}`);
    text("demoStatus", `Error: ${err.message}`);
    text("compareSummary", "Comparison failed.");
    text("responseTag", "A/B comparison failed");
  } finally {
    setOutputLoading(false);
    setBusy(false);
  }
}

function clearOutput() {
  setOutputLoading(false);
  setNotice("is-success", "Ready.");
  text("o-retrieval-latency", "-");
  text("o-generation-latency", "-");
  text("o-total-latency", "-");
  text("o-context-len", "-");
  text("answerOutput", "No output yet.");
  text("responseTag", "No response yet");
  text("payloadOutput", "-");
  text("compareSummary", "No comparison yet.");
  text("c-a-meta", "-");
  text("c-b-meta", "-");
  text("c-a-answer", "No output yet.");
  text("c-b-answer", "No output yet.");
  const hits = byId("hitsOutput");
  if (hits) hits.innerHTML = "";
  const badges = byId("settingsBadges");
  if (badges) badges.innerHTML = "";
  const compareHitsA = byId("c-a-hits");
  if (compareHitsA) compareHitsA.innerHTML = "";
  const compareHitsB = byId("c-b-hits");
  if (compareHitsB) compareHitsB.innerHTML = "";
  const compareSettingsA = byId("c-a-settings");
  if (compareSettingsA) compareSettingsA.innerHTML = "";
  const compareSettingsB = byId("c-b-settings");
  if (compareSettingsB) compareSettingsB.innerHTML = "";
}

function install() {
  byId("askForm")?.addEventListener("submit", ask);
  byId("compareBtn")?.addEventListener("click", askCompare);
  byId("clearBtn")?.addEventListener("click", clearOutput);
  byId("rerankerEnabledInput")?.addEventListener("change", () => syncRerankerControls(""));
  byId("rerankerEnabledInputB")?.addEventListener("change", () => syncRerankerControls("B"));
  byId("topKInput")?.addEventListener("change", () => syncTopKCandidateGuard(""));
  byId("topKInputB")?.addEventListener("change", () => syncTopKCandidateGuard("B"));
  byId("rerankerCandidateKInput")?.addEventListener("change", () => syncTopKCandidateGuard(""));
  byId("rerankerCandidateKInputB")?.addEventListener("change", () => syncTopKCandidateGuard("B"));
  byId("copyAnswerBtn")?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(byId("answerOutput")?.textContent || "");
      text("demoStatus", "Answer copied to clipboard.");
    } catch {
      text("demoStatus", "Copy failed.");
    }
  });

  byId("questionInput")?.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
      event.preventDefault();
      byId("askForm")?.requestSubmit();
    }
  });

  syncTopKCandidateGuard("");
  syncTopKCandidateGuard("B");
  syncRerankerControls("");
  syncRerankerControls("B");
}

install();
bootstrap();
