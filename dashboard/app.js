const DATA_CANDIDATES = {
  metrics: [
    "/results/runs/baseline_lite/metrics.json",
    "../results/runs/baseline_lite/metrics.json",
    "results/runs/baseline_lite/metrics.json",
    "/results/baseline_lite/metrics.json",
    "../results/baseline_lite/metrics.json",
    "results/baseline_lite/metrics.json",
  ],
  matrix: [
    "/results/summaries/baseline_lite_matrix_summary.json",
    "../results/summaries/baseline_lite_matrix_summary.json",
    "results/summaries/baseline_lite_matrix_summary.json",
    "/results/summaries/baseline_lite_reranker_matrix_summary.json",
    "../results/summaries/baseline_lite_reranker_matrix_summary.json",
    "results/summaries/baseline_lite_reranker_matrix_summary.json",
    "/results/summaries/reranker_ablation_lite_matrix_summary.json",
    "../results/summaries/reranker_ablation_lite_matrix_summary.json",
    "results/summaries/reranker_ablation_lite_matrix_summary.json",
    "/results/summaries/baseline_lite_bm25_matrix_summary.json",
    "../results/summaries/baseline_lite_bm25_matrix_summary.json",
    "results/summaries/baseline_lite_bm25_matrix_summary.json",
    "/results/baseline_lite_matrix_summary.json",
    "../results/baseline_lite_matrix_summary.json",
    "results/baseline_lite_matrix_summary.json",
    "/results/baseline_lite_reranker_matrix_summary.json",
    "../results/baseline_lite_reranker_matrix_summary.json",
    "results/baseline_lite_reranker_matrix_summary.json",
    "/results/reranker_ablation_lite_matrix_summary.json",
    "../results/reranker_ablation_lite_matrix_summary.json",
    "results/reranker_ablation_lite_matrix_summary.json",
    "/results/baseline_lite_bm25_matrix_summary.json",
    "../results/baseline_lite_bm25_matrix_summary.json",
    "results/baseline_lite_bm25_matrix_summary.json",
    "/results/baseline_dense_matrix_summary.json",
    "../results/baseline_dense_matrix_summary.json",
    "results/baseline_dense_matrix_summary.json",
    "/results/baseline_bm25_matrix_summary.json",
    "../results/baseline_bm25_matrix_summary.json",
    "results/baseline_bm25_matrix_summary.json",
  ],
};

const SERIES_COLORS = [
  "#0f8269",
  "#d27e28",
  "#0f5f84",
  "#8d4f24",
  "#4b7e5a",
  "#6f4b9b",
];

const state = {
  metrics: {},
  rows: [],
};

function byId(id) {
  return document.getElementById(id);
}

function num(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function toBool(value, fallback = false) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (["true", "1", "yes", "on"].includes(text)) return true;
    if (["false", "0", "no", "off", ""].includes(text)) return false;
  }
  if (typeof value === "number") return value !== 0;
  return fallback;
}

function fixed(value, digits = 4) {
  return num(value).toFixed(digits);
}

function escapeHtml(input) {
  return String(input ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setText(id, text) {
  const node = byId(id);
  if (node) node.textContent = text;
}

function setStatus(kind, text) {
  const node = byId("dataStatus");
  if (!node) return;
  node.classList.remove("status-ok", "status-warn", "status-err");
  node.classList.add(kind);
  node.textContent = text;
}

function setNotice(id, kind, textValue) {
  const node = byId(id);
  if (!node) return;
  node.classList.remove("is-loading", "is-success", "is-warn", "is-error", "hidden");
  node.classList.add(kind);
  node.textContent = textValue;
}

function setHidden(id, isHidden) {
  const node = byId(id);
  if (!node) return;
  node.classList.toggle("hidden", isHidden);
}

function compositeScore(row) {
  const recall = num(row.recall_at_k);
  const mrr = num(row.mrr);
  const f1 = num(row.f1);
  const latency = num(row.avg_query_latency_ms);
  return recall * 0.5 + mrr * 0.3 + f1 * 0.2 - latency * 0.001;
}

async function fetchFirstJson(paths) {
  for (const path of paths) {
    try {
      const res = await fetch(path, { cache: "no-store" });
      if (!res.ok) continue;
      return { path, payload: await res.json() };
    } catch {
      // try next path
    }
  }
  throw new Error(`Unable to load json from: ${paths.join(", ")}`);
}

async function fetchAllJson(paths) {
  const loaded = [];
  const seen = new Set();

  for (const path of paths) {
    if (seen.has(path)) continue;
    seen.add(path);
    try {
      const res = await fetch(path, { cache: "no-store" });
      if (!res.ok) continue;
      loaded.push({ path, payload: await res.json() });
    } catch {
      // try next path
    }
  }

  if (!loaded.length) {
    throw new Error(`Unable to load any json from: ${paths.join(", ")}`);
  }
  return loaded;
}

function normalizeMatrixRow(row) {
  const rerankerEnabled = toBool(row.reranker_enabled, false);
  const rerankerType = rerankerEnabled ? String(row.reranker_type || "overlap").trim().toLowerCase() : "none";
  const rerankerCandidateK = row.reranker_candidate_k == null ? 20 : num(row.reranker_candidate_k);
  const rerankerAlpha = row.reranker_alpha == null ? 0.5 : num(row.reranker_alpha);
  const rerankerLabel = rerankerEnabled
    ? `${rerankerType}/k${rerankerCandidateK.toFixed(0)}/a${rerankerAlpha.toFixed(2)}`
    : "none";
  return {
    ...row,
    backend: String(row.backend || "").trim().toLowerCase(),
    strategy: String(row.strategy || "").trim().toLowerCase(),
    reranker_enabled: rerankerEnabled,
    reranker_type: rerankerType,
    reranker_candidate_k: rerankerCandidateK,
    reranker_alpha: rerankerAlpha,
    reranker_label: rerankerLabel,
    chunk_size: num(row.chunk_size),
    overlap: num(row.overlap),
    top_k: num(row.top_k),
  };
}

function mergeMatrixRows(loadedFiles) {
  const merged = new Map();

  for (const file of loadedFiles) {
    if (!Array.isArray(file.payload)) continue;
    for (const rawRow of file.payload) {
      const row = normalizeMatrixRow(rawRow);
      if (!row.backend || !row.strategy) continue;

      const key = [
        row.backend,
        row.strategy,
        row.reranker_enabled ? "1" : "0",
        row.reranker_type,
        row.reranker_candidate_k,
        row.reranker_alpha,
        row.chunk_size,
        row.overlap,
        row.top_k,
      ].join("|");

      if (!merged.has(key)) {
        merged.set(key, row);
        continue;
      }

      const existing = merged.get(key);
      // Prefer rows that include QA fields when duplicates exist.
      const existingHasQa = Number.isFinite(Number(existing.f1)) && Number.isFinite(Number(existing.em));
      const currentHasQa = Number.isFinite(Number(row.f1)) && Number.isFinite(Number(row.em));
      if (!existingHasQa && currentHasQa) {
        merged.set(key, row);
      }
    }
  }

  return Array.from(merged.values());
}

function animateValue(target, value, suffix = "", digits = 4) {
  if (!target) return;
  const end = num(value);
  const start = 0;
  const duration = 560;
  const startTs = performance.now();

  const step = (ts) => {
    const t = Math.min(1, (ts - startTs) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const v = start + (end - start) * eased;
    target.textContent = `${v.toFixed(digits)}${suffix}`;
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function sortRowsForDisplay(rows) {
  return [...rows]
    .map((row) => ({ ...row, composite_score: compositeScore(row) }))
    .sort((a, b) => num(b.composite_score) - num(a.composite_score));
}

function renderOverview(rows) {
  const baseline = state.metrics || {};
  const ranked = rows
    .map((row) => ({ ...row, composite_score: compositeScore(row) }))
    .sort((a, b) => num(b.composite_score) - num(a.composite_score));
  const best = ranked[0];
  const fastest = [...rows].sort((a, b) => num(a.avg_query_latency_ms) - num(b.avg_query_latency_ms))[0];

  animateValue(byId("m-em"), baseline.em, "", 4);
  animateValue(byId("m-f1"), baseline.f1, "", 4);
  animateValue(byId("m-recall"), baseline.recall_at_k, "", 4);
  animateValue(byId("m-mrr"), baseline.mrr, "", 4);
  setText("m-runs", String(rows.length));
  setText("m-best-score", best ? fixed(best.composite_score, 4) : "-");
  setText(
    "m-best-config",
    best
      ? `${best.backend}/${best.strategy} ${best.reranker_label} c${num(best.chunk_size).toFixed(0)} o${num(
          best.overlap
        ).toFixed(0)} k${num(best.top_k).toFixed(0)}`
      : "-"
  );
  setText("m-best-latency", fastest ? `${fixed(fastest.avg_query_latency_ms, 3)} ms` : "-");
}

function badge(text) {
  return `<span class="badge">${text}</span>`;
}

function renderTable(rows) {
  const tbody = document.querySelector("#runsTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${badge(escapeHtml(row.backend))}</td>
      <td>${badge(escapeHtml(row.strategy))}</td>
      <td>${badge(row.reranker_enabled ? "on" : "off")}</td>
      <td>${num(row.top_k).toFixed(0)}</td>
      <td>${fixed(row.recall_at_k, 4)}</td>
      <td>${fixed(row.mrr, 4)}</td>
      <td>${fixed(row.f1, 4)}</td>
      <td>${fixed(row.avg_query_latency_ms, 3)}</td>
    `;
    tbody.appendChild(tr);
  }
}

function aggregateRows(rows, xKey, yKey, seriesKeys) {
  const map = new Map();
  for (const row of rows) {
    const series = seriesKeys.map((key) => row[key]).join(" | ");
    const x = num(row[xKey]);
    const y = num(row[yKey]);
    const id = `${series}::${x}`;
    if (!map.has(id)) map.set(id, { series, x, sum: 0, count: 0 });
    const bucket = map.get(id);
    bucket.sum += y;
    bucket.count += 1;
  }

  const grouped = {};
  for (const bucket of map.values()) {
    if (!grouped[bucket.series]) grouped[bucket.series] = [];
    grouped[bucket.series].push({ x: bucket.x, y: bucket.sum / bucket.count });
  }
  for (const series of Object.keys(grouped)) {
    grouped[series].sort((a, b) => a.x - b.x);
  }
  return grouped;
}

function pickBestRows(rows, groupKeys, scoreFn) {
  const best = new Map();
  for (const row of rows) {
    const key = groupKeys.map((keyName) => String(row[keyName])).join("|");
    const current = best.get(key);
    const score = scoreFn(row);
    if (!current || score > current.score) {
      best.set(key, { row, score });
    }
  }
  return Array.from(best.values(), (item) => item.row);
}

function prepareCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const displayWidth = canvas.clientWidth || canvas.width;
  const displayHeight = canvas.clientHeight || canvas.height;
  canvas.width = Math.round(displayWidth * ratio);
  canvas.height = Math.round(displayHeight * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: displayWidth, height: displayHeight };
}

function drawChart(canvasId, grouped, yLabel) {
  const canvas = byId(canvasId);
  if (!canvas) return;
  const { ctx, width, height } = prepareCanvas(canvas);

  const pad = { l: 52, r: 18, t: 18, b: 36 };
  const points = Object.values(grouped).flat();
  ctx.clearRect(0, 0, width, height);

  const cardGrad = ctx.createLinearGradient(0, 0, 0, height);
  cardGrad.addColorStop(0, "#ffffff");
  cardGrad.addColorStop(1, "#f7fcfa");
  ctx.fillStyle = cardGrad;
  ctx.fillRect(0, 0, width, height);

  if (!points.length) {
    ctx.fillStyle = "#56716d";
    ctx.font = "14px Space Grotesk";
    ctx.fillText("No experiment rows available", 22, 40);
    return;
  }

  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(...points.map((p) => p.x));
  const minY = Math.min(...points.map((p) => p.y));
  const maxY = Math.max(...points.map((p) => p.y));
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1e-6;

  const xToPx = (x) => pad.l + ((x - minX) / xSpan) * (width - pad.l - pad.r);
  const yToPx = (y) => height - pad.b - ((y - minY) / ySpan) * (height - pad.t - pad.b);

  ctx.strokeStyle = "#d9e8e4";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = minY + (ySpan * i) / 4;
    const py = yToPx(y);
    ctx.beginPath();
    ctx.moveTo(pad.l, py);
    ctx.lineTo(width - pad.r, py);
    ctx.stroke();
    ctx.fillStyle = "#607a76";
    ctx.font = "11px Space Grotesk";
    ctx.fillText(y.toFixed(3), 6, py + 4);
  }

  ctx.strokeStyle = "#91aca6";
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, height - pad.b);
  ctx.lineTo(width - pad.r, height - pad.b);
  ctx.stroke();

  let idx = 0;
  for (const [series, seriesPoints] of Object.entries(grouped)) {
    const color = SERIES_COLORS[idx % SERIES_COLORS.length];
    idx += 1;
    const sorted = [...seriesPoints].sort((a, b) => a.x - b.x);

    ctx.beginPath();
    sorted.forEach((p, i) => {
      const x = xToPx(p.x);
      const y = yToPx(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.stroke();

    for (const point of sorted) {
      const x = xToPx(point.x);
      const y = yToPx(point.y);
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  ctx.fillStyle = "#31504d";
  ctx.font = "11px Space Grotesk";
  ctx.fillText(yLabel, 8, 13);

  const legendX = pad.l + 2;
  let legendY = pad.t + 3;
  idx = 0;
  for (const series of Object.keys(grouped)) {
    const color = SERIES_COLORS[idx % SERIES_COLORS.length];
    idx += 1;
    ctx.fillStyle = color;
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = "#1e3936";
    ctx.font = "11px Space Grotesk";
    ctx.fillText(series, legendX + 14, legendY + 9);
    legendY += 14;
  }
}

function drawBarChart(canvasId, items, options) {
  const canvas = byId(canvasId);
  if (!canvas) return;
  const { ctx, width, height } = prepareCanvas(canvas);
  const pad = { l: 52, r: 18, t: 24, b: 54 };
  ctx.clearRect(0, 0, width, height);

  const cardGrad = ctx.createLinearGradient(0, 0, 0, height);
  cardGrad.addColorStop(0, "#ffffff");
  cardGrad.addColorStop(1, "#f7f9ff");
  ctx.fillStyle = cardGrad;
  ctx.fillRect(0, 0, width, height);

  if (!items.length) {
    ctx.fillStyle = "#56716d";
    ctx.font = "14px Plus Jakarta Sans";
    ctx.fillText("No experiment rows available", 22, 40);
    return;
  }

  const values = items.map((item) => num(item.value));
  const maxValue = Math.max(...values, 0.0001);
  const chartHeight = height - pad.t - pad.b;
  const chartWidth = width - pad.l - pad.r;
  const barWidth = Math.min(110, chartWidth / Math.max(items.length * 1.6, 1));
  const gap = (chartWidth - barWidth * items.length) / Math.max(items.length + 1, 1);

  ctx.strokeStyle = "#d9e2f1";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.t + (chartHeight * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(width - pad.r, y);
    ctx.stroke();
    const tickValue = maxValue - (maxValue * i) / 4;
    ctx.fillStyle = "#607a76";
    ctx.font = "11px Plus Jakarta Sans";
    ctx.fillText(tickValue.toFixed(options.decimals ?? 3), 6, y + 4);
  }

  items.forEach((item, index) => {
    const x = pad.l + gap * (index + 1) + barWidth * index;
    const barHeight = (num(item.value) / maxValue) * chartHeight;
    const y = height - pad.b - barHeight;
    const color = options.colors[index % options.colors.length];

    ctx.fillStyle = color;
    ctx.fillRect(x, y, barWidth, barHeight);

    ctx.fillStyle = "#1e293b";
    ctx.font = "12px Plus Jakarta Sans";
    ctx.textAlign = "center";
    ctx.fillText(num(item.value).toFixed(options.decimals ?? 3), x + barWidth / 2, y - 8);
    ctx.fillText(item.label, x + barWidth / 2, height - pad.b + 18);

    if (item.meta) {
      ctx.fillStyle = "#64748b";
      ctx.font = "11px Plus Jakarta Sans";
      ctx.fillText(item.meta, x + barWidth / 2, height - pad.b + 34);
    }
  });

  ctx.textAlign = "left";
  ctx.fillStyle = "#31504d";
  ctx.font = "11px Plus Jakarta Sans";
  ctx.fillText(options.yLabel, 8, 13);
}

function renderCharts(rows) {
  const backendRows = pickBestRows(rows, ["backend", "top_k"], (row) => compositeScore(row));
  const groupedRecall = aggregateRows(backendRows, "top_k", "recall_at_k", ["backend"]);
  drawChart("recallChart", groupedRecall, "recall@k");

  const highestTopK = Math.max(...rows.map((row) => num(row.top_k)), 0);
  const highestTopKRows = rows.filter((row) => num(row.top_k) === highestTopK);
  const metricKey = highestTopKRows.some((row) => num(row.f1) > 0) ? "f1" : "mrr";
  const bestStrategyRows = pickBestRows(highestTopKRows, ["strategy"], (row) => compositeScore(row)).sort((a, b) =>
    String(a.strategy).localeCompare(String(b.strategy))
  );
  drawBarChart(
    "mrrChart",
    bestStrategyRows.map((row) => ({
      label: String(row.strategy),
      value: num(row[metricKey]),
      meta: `${fixed(row.avg_query_latency_ms, 3)} ms`,
    })),
    {
      yLabel: metricKey,
      decimals: metricKey === "f1" ? 4 : 3,
      colors: ["#0f766e", "#7c3aed", "#ea580c", "#2563eb"],
    }
  );
}

function renderFindings(rows) {
  const championGrid = byId("championGrid");
  const insightList = byId("insightList");
  if (!championGrid || !insightList) return;

  championGrid.innerHTML = "";
  insightList.innerHTML = "";
  if (!rows.length) {
    insightList.innerHTML = "<li>No experiment rows are available.</li>";
    return;
  }

  const topRows = [...rows]
    .sort((a, b) => num(b.composite_score) - num(a.composite_score))
    .slice(0, 3);

  for (const row of topRows) {
    const card = document.createElement("article");
    card.className = "champion-card";
    card.innerHTML = `
      <header>
        <h4>${escapeHtml(`${row.backend}/${row.strategy}`)}</h4>
        <span class="badge">${escapeHtml(row.backend)}</span>
      </header>
      <p>${escapeHtml(row.reranker_label)} | c${num(row.chunk_size).toFixed(0)} | o${num(row.overlap).toFixed(
      0
    )} | k${num(row.top_k).toFixed(0)}</p>
      <p>Recall ${fixed(row.recall_at_k)} | MRR ${fixed(row.mrr)} | F1 ${fixed(row.f1)} | Score ${fixed(
      row.composite_score
    )}</p>
    `;
    championGrid.appendChild(card);
  }

  const topRecall = [...rows].sort((a, b) => num(b.recall_at_k) - num(a.recall_at_k))[0];
  const topMrr = [...rows].sort((a, b) => num(b.mrr) - num(a.mrr))[0];
  const fast = [...rows].sort((a, b) => num(a.avg_query_latency_ms) - num(b.avg_query_latency_ms))[0];

  const topKBuckets = aggregateRows(rows, "top_k", "recall_at_k", ["backend"]);
  const topKTrend = Object.values(topKBuckets)
    .flat()
    .sort((a, b) => a.x - b.x);
  const trendNote =
    topKTrend.length >= 2 && topKTrend[topKTrend.length - 1].y > topKTrend[0].y
      ? "Increasing top-k tends to improve recall on your current slice."
      : "Recall does not consistently increase with top-k in this filtered subset.";

  const notes = [
    `Best recall run is ${topRecall.backend}/${topRecall.strategy} at c${num(topRecall.chunk_size).toFixed(
      0
    )}, o${num(topRecall.overlap).toFixed(0)}, k${num(topRecall.top_k).toFixed(0)} (${topRecall.reranker_label}).`,
    `Best MRR run is ${topMrr.backend}/${topMrr.strategy}, indicating strongest early-ranked evidence.`,
    `${trendNote}`,
    `Fastest run latency is ${fixed(fast.avg_query_latency_ms, 3)} ms (${fast.backend}/${fast.strategy}, ${fast.reranker_label}).`,
  ];

  for (const note of notes) {
    const li = document.createElement("li");
    li.textContent = note;
    insightList.appendChild(li);
  }
}

function updateResultsSummary(rows) {
  const denseCount = rows.filter((row) => String(row.backend) === "dense").length;
  const bm25Count = rows.filter((row) => String(row.backend) === "bm25").length;
  const strategyCount = new Set(rows.map((row) => String(row.strategy))).size;
  setText(
    "resultsSummary",
    `${rows.length} runs loaded across ${strategyCount} chunking strategies. Dense: ${denseCount} | BM25: ${bm25Count}.`
  );
}

function attachEvents() {
  window.addEventListener("resize", () => renderCharts(state.rows));
}

async function init() {
  setNotice("overviewNotice", "is-loading", "Loading experiment artifacts...");
  setHidden("overviewSkeleton", false);
  try {
    const [metrics, matrixFiles] = await Promise.all([
      fetchFirstJson(DATA_CANDIDATES.metrics),
      fetchAllJson(DATA_CANDIDATES.matrix),
    ]);

    state.metrics = metrics.payload || {};
    const mergedRows = mergeMatrixRows(matrixFiles);
    state.rows = sortRowsForDisplay(mergedRows);

    const backends = [...new Set(state.rows.map((row) => String(row.backend)))].filter(Boolean);

    setStatus("status-ok", "Ready");
    setText("metricsSource", metrics.path);
    if (matrixFiles.length === 1) {
      setText("matrixSource", matrixFiles[0].path);
    } else {
      setText("matrixSource", `${matrixFiles.length} files merged`);
    }
    setText("uniqueBackends", backends.join(", ") || "-");

    renderOverview(state.rows);
    updateResultsSummary(state.rows);
    renderTable(state.rows);
    renderCharts(state.rows);
    renderFindings(state.rows);
    attachEvents();
    setHidden("overviewSkeleton", true);
    setNotice("overviewNotice", "is-success", "Overview ready. Complete results are shown automatically.");
  } catch (err) {
    console.error(err);
    setStatus("status-err", "Load failed");
    setText("metricsSource", "Unavailable");
    setText("matrixSource", "Unavailable");
    setText("uniqueBackends", "-");
    setHidden("overviewSkeleton", true);
    setNotice("overviewNotice", "is-error", "Failed to load results files. Check paths under /results.");
  }
}

init();


