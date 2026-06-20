const DATA_URL = "../data/processed/dashboard_payload.json";

let payload;
let map;
let markerLayer;
let selectedZoneId;
let currentHotspots = [];

const priorityOrder = ["Immediate patrol", "High watchlist", "Scheduled patrol", "Monitor"];

const els = {
  dateRange: document.getElementById("dateRange"),
  resetButton: document.getElementById("resetButton"),
  priorityFilter: document.getElementById("priorityFilter"),
  stationFilter: document.getElementById("stationFilter"),
  minScoreFilter: document.getElementById("minScoreFilter"),
  minScoreValue: document.getElementById("minScoreValue"),
  searchFilter: document.getElementById("searchFilter"),
  metricViolations: document.getElementById("metricViolations"),
  metricHotspots: document.getElementById("metricHotspots"),
  metricHighZones: document.getElementById("metricHighZones"),
  metricPeakShare: document.getElementById("metricPeakShare"),
  visibleCount: document.getElementById("visibleCount"),
  hotspotList: document.getElementById("hotspotList"),
  selectedClass: document.getElementById("selectedClass"),
  selectedDetails: document.getElementById("selectedDetails"),
  generatedAt: document.getElementById("generatedAt"),
  priorityBreakdown: document.getElementById("priorityBreakdown"),
  stationsTab: document.getElementById("tab-stations"),
  hoursTab: document.getElementById("tab-hours"),
  violationsTab: document.getElementById("tab-violations"),
};

async function init() {
  payload = await loadPayload();
  populateMetrics();
  populateFilters();
  initMap();
  renderSummaries();
  applyFilters();
  bindEvents();
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

async function loadPayload() {
  const response = await fetch(DATA_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${DATA_URL}: ${response.status}`);
  }
  return response.json();
}

function populateMetrics() {
  const { metadata, metrics } = payload;
  els.dateRange.textContent = `${formatDate(metadata.date_min)} to ${formatDate(metadata.date_max)} · ${metadata.city}`;
  els.metricViolations.textContent = compact(metrics.violations);
  els.metricHotspots.textContent = compact(metrics.hotspots);
  els.metricHighZones.textContent = compact(metrics.high_watchlist_zones + metrics.immediate_zones);
  els.metricPeakShare.textContent = `${metrics.peak_hour_share}%`;
  els.generatedAt.textContent = formatDate(metadata.generated_at);
}

function populateFilters() {
  const priorities = ["All priorities", ...priorityOrder.filter((item) =>
    payload.hotspots.some((hotspot) => hotspot.priority_class === item)
  )];
  fillSelect(els.priorityFilter, priorities);

  const stations = Array.from(new Set(payload.hotspots.map((hotspot) => hotspot.police_station)))
    .filter(Boolean)
    .sort((a, b) => a.localeCompare(b));
  fillSelect(els.stationFilter, ["All stations", ...stations]);
}

function fillSelect(select, values) {
  select.innerHTML = values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
}

function initMap() {
  map = L.map("map", {
    zoomControl: true,
    preferCanvas: true,
  }).setView([payload.map_center.latitude, payload.map_center.longitude], 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);

  markerLayer = L.layerGroup().addTo(map);
}

function bindEvents() {
  [els.priorityFilter, els.stationFilter, els.minScoreFilter, els.searchFilter].forEach((control) => {
    control.addEventListener("input", applyFilters);
  });

  els.resetButton.addEventListener("click", () => {
    els.priorityFilter.value = "All priorities";
    els.stationFilter.value = "All stations";
    els.minScoreFilter.value = "0";
    els.searchFilter.value = "";
    applyFilters();
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
    });
  });
}

function applyFilters() {
  const priority = els.priorityFilter.value;
  const station = els.stationFilter.value;
  const minScore = Number(els.minScoreFilter.value);
  const query = els.searchFilter.value.trim().toLowerCase();

  els.minScoreValue.textContent = String(minScore);

  currentHotspots = payload.hotspots.filter((hotspot) => {
    if (priority !== "All priorities" && hotspot.priority_class !== priority) {
      return false;
    }
    if (station !== "All stations" && hotspot.police_station !== station) {
      return false;
    }
    if (Number(hotspot.priority_score) < minScore) {
      return false;
    }
    if (query) {
      const haystack = [
        hotspot.zone_id,
        hotspot.police_station,
        hotspot.junction_name,
        hotspot.location,
        hotspot.primary_violation,
        hotspot.dominant_vehicle_type,
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    }
    return true;
  });

  if (!currentHotspots.some((hotspot) => hotspot.zone_id === selectedZoneId)) {
    selectedZoneId = currentHotspots[0]?.zone_id;
  }

  renderHotspotList();
  renderMarkers();
  renderSelected();
}

function renderHotspotList() {
  els.visibleCount.textContent = `${currentHotspots.length.toLocaleString()} zones`;
  els.hotspotList.innerHTML = currentHotspots
    .slice(0, 120)
    .map((hotspot) => {
      const badgeClass = priorityClassName(hotspot.priority_class);
      const active = hotspot.zone_id === selectedZoneId ? " active" : "";
      return `
        <button class="hotspot-item${active}" type="button" data-zone="${escapeHtml(hotspot.zone_id)}">
          <div class="hotspot-topline">
            <span class="hotspot-title">${escapeHtml(hotspot.junction_name || hotspot.location)}</span>
            <span class="class-badge ${badgeClass}">${escapeHtml(hotspot.priority_class)}</span>
          </div>
          <div class="score-row">
            <div class="score-bar"><span style="width:${clamp(hotspot.priority_score, 0, 100)}%"></span></div>
            <strong>${Number(hotspot.priority_score).toFixed(1)}</strong>
          </div>
          <div class="hotspot-meta">
            <span>${escapeHtml(hotspot.police_station)}</span>
            <span>${Number(hotspot.violation_count).toLocaleString()} violations</span>
          </div>
          <div class="hotspot-meta">
            <span>${escapeHtml(hotspot.primary_violation)}</span>
            <span>${Number(hotspot.peak_share * 100).toFixed(0)}% peak</span>
          </div>
        </button>
      `;
    })
    .join("");

  els.hotspotList.querySelectorAll(".hotspot-item").forEach((button) => {
    button.addEventListener("click", () => {
      selectedZoneId = button.dataset.zone;
      renderHotspotList();
      renderSelected();
      focusMarker(selectedZoneId);
    });
  });
}

function renderMarkers() {
  markerLayer.clearLayers();
  const bounds = [];

  currentHotspots.forEach((hotspot) => {
    const marker = L.circleMarker([hotspot.latitude, hotspot.longitude], {
      radius: markerRadius(hotspot),
      color: priorityColor(hotspot.priority_class),
      fillColor: priorityColor(hotspot.priority_class),
      fillOpacity: hotspot.zone_id === selectedZoneId ? 0.84 : 0.58,
      opacity: 0.95,
      weight: hotspot.zone_id === selectedZoneId ? 3 : 1,
    });

    marker.bindPopup(`
      <div class="popup-title">${escapeHtml(hotspot.junction_name || hotspot.location)}</div>
      <div class="popup-row">${escapeHtml(hotspot.police_station)} · ${escapeHtml(hotspot.primary_violation)}</div>
      <div class="popup-row">Priority ${Number(hotspot.priority_score).toFixed(1)} · ${Number(hotspot.violation_count).toLocaleString()} violations</div>
    `);

    marker.on("click", () => {
      selectedZoneId = hotspot.zone_id;
      renderHotspotList();
      renderSelected();
      marker.openPopup();
    });

    markerLayer.addLayer(marker);
    bounds.push([hotspot.latitude, hotspot.longitude]);
  });

  if (bounds.length) {
    map.fitBounds(bounds, { padding: [28, 28], maxZoom: 14 });
  }
}

function focusMarker(zoneId) {
  const hotspot = currentHotspots.find((item) => item.zone_id === zoneId);
  if (!hotspot) {
    return;
  }
  map.setView([hotspot.latitude, hotspot.longitude], Math.max(map.getZoom(), 15), { animate: true });
}

function renderSelected() {
  const hotspot = currentHotspots.find((item) => item.zone_id === selectedZoneId) || payload.hotspots[0];
  if (!hotspot) {
    els.selectedClass.textContent = "-";
    els.selectedDetails.innerHTML = "";
    return;
  }

  els.selectedClass.textContent = hotspot.priority_class;
  els.selectedDetails.innerHTML = `
    ${detail("Priority", Number(hotspot.priority_score).toFixed(1))}
    ${detail("Severity", Number(hotspot.severity_score).toFixed(1))}
    ${detail("Impact", Number(hotspot.congestion_impact_score).toFixed(1))}
    ${detail("Violations", Number(hotspot.violation_count).toLocaleString())}
    ${detail("Active days", Number(hotspot.active_days).toLocaleString())}
    ${detail("Peak share", `${Number(hotspot.peak_share * 100).toFixed(1)}%`)}
    ${detail("Station", hotspot.police_station)}
    ${detail("Violation", hotspot.primary_violation)}
    ${detail("Junction", hotspot.junction_name, true)}
    ${detail("Action", hotspot.recommended_action, true)}
  `;
}

function detail(label, value, wide = false) {
  return `
    <div class="${wide ? "selected-wide" : ""}">
      <dt>${escapeHtml(label)}</dt>
      <dd>${escapeHtml(value ?? "-")}</dd>
    </div>
  `;
}

function renderSummaries() {
  renderSummaryRows(
    els.stationsTab,
    payload.station_summary,
    "police_station",
    "violation_count",
    (row) => `${compact(row.violation_count)} · max ${Number(row.max_priority_score).toFixed(1)}`
  );

  renderSummaryRows(
    els.hoursTab,
    payload.hourly_summary,
    "hour",
    "violation_count",
    (row) => `${String(row.hour).padStart(2, "0")}:00 · ${compact(row.violation_count)}`,
    (row) => `${String(row.hour).padStart(2, "0")}:00`
  );

  renderSummaryRows(
    els.violationsTab,
    payload.violation_summary,
    "violation",
    "violation_count",
    (row) => compact(row.violation_count)
  );

  const maxPriorityCount = Math.max(...payload.priority_summary.map((row) => Number(row.zone_count)), 1);
  els.priorityBreakdown.innerHTML = payload.priority_summary
    .map((row) => {
      const width = (Number(row.zone_count) / maxPriorityCount) * 100;
      return `
        <div class="breakdown-row">
          <span class="summary-label">${escapeHtml(row.priority_class)}</span>
          <span class="summary-value">${Number(row.zone_count).toLocaleString()} zones</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%; background:${priorityColor(row.priority_class)}"></div></div>
        </div>
      `;
    })
    .join("");
}

function renderSummaryRows(container, rows, labelKey, valueKey, valueFormatter, labelFormatter = null) {
  const maxValue = Math.max(...rows.map((row) => Number(row[valueKey])), 1);
  container.innerHTML = rows
    .map((row) => {
      const label = labelFormatter ? labelFormatter(row) : row[labelKey];
      const value = valueFormatter(row);
      const width = (Number(row[valueKey]) / maxValue) * 100;
      return `
        <div class="summary-row">
          <span class="summary-label">${escapeHtml(label)}</span>
          <span class="summary-value">${escapeHtml(value)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        </div>
      `;
    })
    .join("");
}

function markerRadius(hotspot) {
  return clamp(5 + Math.sqrt(Number(hotspot.violation_count)) / 5, 5, 20);
}

function priorityColor(priority) {
  if (priority === "Immediate patrol") return "#9f1f28";
  if (priority === "High watchlist") return "#c93838";
  if (priority === "Scheduled patrol") return "#c97912";
  return "#7c8793";
}

function priorityClassName(priority) {
  if (priority === "Immediate patrol") return "immediate";
  if (priority === "High watchlist") return "high";
  if (priority === "Scheduled patrol") return "scheduled";
  return "monitor";
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  return date.toLocaleDateString("en-IN", { year: "numeric", month: "short", day: "2-digit" });
}

function compact(value) {
  return new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(Number(value));
}

function clamp(value, min, max) {
  return Math.min(Math.max(Number(value), min), max);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

init().catch((error) => {
  document.body.innerHTML = `<main class="load-error"><h1>Dashboard failed to load</h1><p>${escapeHtml(error.message)}</p></main>`;
  console.error(error);
});
