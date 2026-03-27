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
    `${payload.settings?.backend || "-"} / ${payload.settings?.strategy || "-"} | total ${Number(
      payload.timings_ms?.total || 0
    ).toFixed(2)} ms`
  );
  renderSettingsBadges(payload.settings, `${prefix}-settings`);
  renderHits(payload.hits || [], requestPayload.question, `${prefix}-hits`, 3);
}

function buildPayloadFromInputs(suffix = "") {
  const question = (byId("questionInput")?.value || "").trim();
  return {
    question,
    backend: byId(`backendInput${suffix}`)?.value || "dense",
    strategy: byId(`strategyInput${suffix}`)?.value || "fixed",
    chunk_size: num(`chunkSizeInput${suffix}`, 256),
    overlap: num(`overlapInput${suffix}`, 32),
    top_k: num(`topKInput${suffix}`, 5),
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

      setSelectOptions("backendInput", backends, defs.backend);
      setSelectOptions("strategyInput", strategies, defs.strategy);
      setSelectOptions("backendInputB", backends, backends.length > 1 ? backends[1] : defs.backend);
      setSelectOptions("strategyInputB", strategies, defs.strategy);

      if (byId("chunkSizeInput")) byId("chunkSizeInput").value = defs.chunk_size ?? 256;
      if (byId("overlapInput")) byId("overlapInput").value = defs.overlap ?? 32;
      if (byId("topKInput")) byId("topKInput").value = defs.top_k ?? 5;

      if (byId("chunkSizeInputB")) byId("chunkSizeInputB").value = defs.chunk_size ?? 256;
      if (byId("overlapInputB")) byId("overlapInputB").value = defs.overlap ?? 32;
      if (byId("topKInputB")) byId("topKInputB").value = defs.top_k ?? 5;

      if (byId("configInput")) byId("configInput").value = payload.config_path || "configs/portable_interactive.yaml";
      text("activeConfig", payload.config_path || "-");
      text("availableBackends", backends.join(", ") || "-");
      text("availableStrategies", strategies.join(", ") || "-");
    }
  } catch {
    text("activeConfig", "Unavailable");
    text("availableBackends", "-");
    text("availableStrategies", "-");
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
    return;
  }

  setBusy(true, "single");
  text("demoStatus", "Running query...");
  text("responseTag", "Request in progress...");

  try {
    const payload = await askApi(requestPayload);
    renderResponse(payload, requestPayload);
    text("demoStatus", `Done | backend=${payload.settings.backend}, strategy=${payload.settings.strategy}`);
  } catch (err) {
    text("demoStatus", `Error: ${err.message}`);
    text("responseTag", "Run failed");
  } finally {
    setBusy(false);
  }
}

async function askCompare() {
  const requestA = buildPayloadFromInputs("");
  const requestB = buildPayloadFromInputs("B");

  if (!requestA.question) {
    text("demoStatus", "Please enter a question.");
    return;
  }

  setBusy(true, "compare");
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
    } else if (resA.status === "rejected" && resB.status === "rejected") {
      throw new Error("Both A and B failed.");
    } else {
      text("compareSummary", "Comparison partially completed (one run failed).");
      text("demoStatus", "Comparison partially completed.");
      text("responseTag", "A/B comparison partial");
    }
  } catch (err) {
    text("demoStatus", `Error: ${err.message}`);
    text("compareSummary", "Comparison failed.");
    text("responseTag", "A/B comparison failed");
  } finally {
    setBusy(false);
  }
}

function clearOutput() {
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
}

install();
bootstrap();
