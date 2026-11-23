const state = {
  summary: null,
  health: null,
  start: null,
  end: null,
};

function formatInteger(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat("en-US").format(number)
    : "--";
}

function formatDecimal(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "--";
}

function formatTimestamp(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch (_) {
      // Keep the HTTP status fallback.
    }
    throw new Error(detail);
  }
  return response.json();
}

function setApiError(message) {
  const statusDot = document.getElementById("statusDot");
  const sidebarStatus = document.getElementById("sidebarStatus");
  const sidebarModels = document.getElementById("sidebarModels");
  const modelStatus = document.getElementById("modelStatus");

  statusDot.className = "status-dot offline";
  sidebarStatus.textContent = "API unavailable";
  sidebarModels.textContent = message;
  modelStatus.textContent = "Offline";
  modelStatus.style.color = "var(--danger)";

  document.getElementById("lastUpdated").textContent = message;
}

function renderHealth(health) {
  state.health = health;
  const statusDot = document.getElementById("statusDot");
  statusDot.className = "status-dot online";
  document.getElementById("sidebarStatus").textContent = "Engine healthy";
  document.getElementById("sidebarModels").textContent =
    `${health.model_count} models / ${health.sensor_count} sensors`;
  document.getElementById("modelStatus").textContent = "Operational";
  document.getElementById("thresholdLabel").textContent =
    `Threshold ${formatDecimal(health.anomaly_threshold, 3)}`;
}

function renderSummary(summary) {
  state.summary = summary;

  document.getElementById("kpiAnomalyRate").textContent =
    `${formatDecimal(summary.anomaly_rate_percent, 3)}%`;
  document.getElementById("kpiAnomalyCount").textContent =
    `${formatInteger(summary.anomaly_count)} anomalies`;

  document.getElementById("kpiTestRecords").textContent =
    formatInteger(summary.test_row_count);
  document.getElementById("kpiPredictionRecords").textContent =
    `${formatInteger(summary.prediction_row_count)} predictions`;

  document.getElementById("kpiPeakScore").textContent =
    formatDecimal(summary.anomaly_score_max, 3);
  document.getElementById("kpiMeanScore").textContent =
    `Mean ${formatDecimal(summary.anomaly_score_mean, 3)}`;

  document.getElementById("kpiEpisodes").textContent =
    formatInteger(summary.episode_count);
  document.getElementById("kpiEpisodePoints").textContent =
    `${formatInteger(summary.flagged_episode_points)} flagged points`;

  const range = summary.historical_range || {};
  const startInput = document.getElementById("startDate");
  const endInput = document.getElementById("endDate");
  if (range.start) {
    startInput.min = toDateTimeLocal(range.start);
    endInput.min = toDateTimeLocal(range.start);
  }
  if (range.end) {
    startInput.max = toDateTimeLocal(range.end);
    endInput.max = toDateTimeLocal(range.end);
  }

  const sensorSelect = document.getElementById("sensorSelect");
  sensorSelect.replaceChildren(
    ...(summary.sensors || []).map((sensor) => {
      const option = document.createElement("option");
      option.value = sensor;
      option.textContent = sensor.replaceAll("_", " ");
      return option;
    })
  );

  if (!summary.sensors || summary.sensors.length === 0) {
    sensorSelect.innerHTML =
      '<option value="">No sensors available</option>';
  }

  document.getElementById("lastUpdated").textContent =
    `Updated ${new Intl.DateTimeFormat("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(new Date())}`;
}

function renderEpisodes(payload) {
  const table = document.getElementById("episodesTable");
  const records = payload.records || [];

  if (records.length === 0) {
    table.innerHTML =
      '<tr><td colspan="5" class="table-state">No anomaly episodes available</td></tr>';
    return;
  }

  table.replaceChildren(
    ...records.slice(0, 6).map((episode) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${escapeHtml(formatTimestamp(episode.start_time))}</td>
        <td>${escapeHtml(formatDuration(episode.duration_minutes))}</td>
        <td>${escapeHtml(formatInteger(episode.flagged_points))}</td>
        <td><span class="episode-score">${escapeHtml(formatDecimal(episode.max_anomaly_score, 3))}</span></td>
        <td>${escapeHtml((episode.primary_sensor_contributor || "--").replaceAll("_", " "))}</td>
      `;
      return row;
    })
  );
}

function formatDuration(minutes) {
  const value = Number(minutes);
  if (!Number.isFinite(value)) return "--";
  if (value < 60) return `${formatInteger(value)} min`;
  const hours = Math.floor(value / 60);
  const remainder = value % 60;
  return remainder ? `${hours}h ${remainder}m` : `${hours}h`;
}

function renderPerformance(payload) {
  const list = document.getElementById("modelList");
  const records = payload.records || [];

  if (records.length === 0) {
    list.innerHTML = '<div class="sensor-loading">No model metrics available</div>';
    return;
  }

  list.replaceChildren(
    ...records.map((metric) => {
      const row = document.createElement("div");
      row.className = "model-row";

      const name = document.createElement("span");
      name.className = "model-name";
      name.textContent = (metric.Sensor || "--").replaceAll("_", " ");

      const value = document.createElement("span");
      value.className = "model-value";
      value.innerHTML =
        `<span>Test R2</span>${escapeHtml(formatDecimal(metric["Test R2"], 4))}`;

      row.append(name, value);
      return row;
    })
  );
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function toDateTimeLocal(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const offset = date.getTimezoneOffset();
  return new Date(date.getTime() - offset * 60000)
    .toISOString()
    .slice(0, 16);
}

function buildHistoryQuery(extra = {}) {
  const params = new URLSearchParams({ limit: "5000", offset: "0" });
  if (state.start) params.set("start", state.start);
  if (state.end) params.set("end", state.end);
  Object.entries(extra).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      params.set(key, value);
    }
  });
  return params.toString();
}

async function loadHistoricalCharts() {
  const scoreLoading = document.getElementById("scoreChartLoading");
  const sensorLoading = document.getElementById("sensorChartLoading");
  scoreLoading.classList.remove("hidden");
  sensorLoading.classList.remove("hidden");

  const sensorSelect = document.getElementById("sensorSelect");
  const sensor = sensorSelect.value;
  if (!sensor) {
    scoreLoading.textContent = "Sensor selection unavailable";
    sensorLoading.textContent = "Sensor selection unavailable";
    return;
  }

  const [scoresResult, predictionsResult] = await Promise.allSettled([
    fetchJson(`/api/history/scores?${buildHistoryQuery()}`),
    fetchJson(
      `/api/history/predictions?${buildHistoryQuery({ sensor })}`
    ),
  ]);

  if (scoresResult.status === "fulfilled") {
    const records = scoresResult.value.records || [];
    window.CycloneCharts.renderScore(
      records,
      state.health ? state.health.anomaly_threshold : 0
    );
    scoreLoading.classList.add("hidden");
  } else {
    scoreLoading.textContent = scoresResult.reason.message;
  }

  if (predictionsResult.status === "fulfilled") {
    const records = predictionsResult.value.records || [];
    window.CycloneCharts.renderSensor(records, sensor);
    sensorLoading.classList.add("hidden");
  } else {
    sensorLoading.textContent = predictionsResult.reason.message;
  }
}

async function loadDashboard() {
  document.body.classList.add("is-loading");
  const refreshButton = document.getElementById("refreshButton");
  refreshButton.disabled = true;

  const results = await Promise.allSettled([
    fetchJson("/health"),
    fetchJson("/api/history/summary"),
    fetchJson("/api/history/episodes?limit=6&offset=0"),
    fetchJson("/api/history/performance?limit=6&offset=0"),
  ]);

  const [health, summary, episodes, performance] = results;

  if (health.status === "fulfilled") renderHealth(health.value);
  if (summary.status === "fulfilled") renderSummary(summary.value);
  if (episodes.status === "fulfilled") renderEpisodes(episodes.value);
  if (performance.status === "fulfilled") renderPerformance(performance.value);

  if (
    health.status === "fulfilled"
    && summary.status === "fulfilled"
  ) {
    await loadHistoricalCharts();
  }

  if (health.status === "rejected" || summary.status === "rejected") {
    const failures = results
      .filter((result) => result.status === "rejected")
      .map((result) => result.reason.message);
    setApiError(failures[0] || "Could not refresh dashboard data.");
  }

  document.body.classList.remove("is-loading");
  refreshButton.disabled = false;
}

function setupNavigation() {
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebarBackdrop");
  const menuButton = document.getElementById("menuButton");

  function closeNavigation() {
    sidebar.classList.remove("open");
    backdrop.classList.remove("open");
    menuButton.setAttribute("aria-expanded", "false");
  }

  menuButton.addEventListener("click", () => {
    const isOpen = sidebar.classList.toggle("open");
    backdrop.classList.toggle("open", isOpen);
    menuButton.setAttribute("aria-expanded", String(isOpen));
  });

  backdrop.addEventListener("click", closeNavigation);
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((link) => {
        link.classList.remove("active");
        link.removeAttribute("aria-current");
      });
      item.classList.add("active");
      item.setAttribute("aria-current", "page");
      closeNavigation();
    });
  });
}

function setupChartFilters() {
  const filterToggle = document.getElementById("filterToggle");
  const filterBar = document.getElementById("chartFilterBar");
  const applyButton = document.getElementById("applyFilters");
  const clearButton = document.getElementById("clearFilters");
  const startInput = document.getElementById("startDate");
  const endInput = document.getElementById("endDate");
  const sensorSelect = document.getElementById("sensorSelect");

  filterToggle.addEventListener("click", () => {
    const isOpen = filterBar.classList.toggle("open");
    filterToggle.setAttribute("aria-expanded", String(isOpen));
  });

  applyButton.addEventListener("click", async () => {
    if (startInput.value && endInput.value && startInput.value > endInput.value) {
      document.getElementById("scoreChartLoading").textContent =
        "Start must be earlier than or equal to end.";
      document.getElementById("sensorChartLoading").textContent =
        "Invalid date range.";
      return;
    }

    state.start = startInput.value || null;
    state.end = endInput.value || null;
    await loadHistoricalCharts();
  });

  clearButton.addEventListener("click", async () => {
    startInput.value = "";
    endInput.value = "";
    state.start = null;
    state.end = null;
    await loadHistoricalCharts();
  });

  sensorSelect.addEventListener("change", loadHistoricalCharts);
}

document.getElementById("refreshButton").addEventListener("click", loadDashboard);
document.getElementById("viewEpisodes").addEventListener("click", () => {
  document.getElementById("episodes").scrollIntoView({ behavior: "smooth" });
});

setupNavigation();
setupChartFilters();
loadDashboard();
