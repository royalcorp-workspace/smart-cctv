/**
 * Smart CCTV 2.0 - Multi-Camera Grid View Manager (NVR / VMS Matrix)
 * Manages grid layout, size transitions, click-to-focus, and stream detachment.
 */

const GRID_VIEW_MODE_KEY = "smart_cctv_view_mode";
const GRID_SIZE_KEY = "smart_cctv_grid_size";

let currentViewMode = "grid"; // Default landing view: "grid"
let currentGridSize = "2x2";  // Default size: "2x2" (4 slots)
let cachedCamerasList = [];
let gridPollTimer = null;

// Pagination & Auto-Tour / Patrol State
let currentGridPage = 1;
let isAutoTourActive = false;
let autoTourIntervalSec = 10;
let autoTourTimer = null;

const GRID_CAPACITY_MAP = {
  "2x2": 4,
  "3x3": 9,
  "4x4": 16,
  "5x5": 25,
  "6x6": 36,
};

/**
 * Initialize Grid View State and Controls
 */
async function initGridView() {
  // Restore saved view mode & grid size, defaulting to "grid" and "2x2"
  const savedMode = localStorage.getItem(GRID_VIEW_MODE_KEY);
  if (savedMode === "single" || savedMode === "grid") {
    currentViewMode = savedMode;
  } else {
    currentViewMode = "grid";
  }

  const savedSize = localStorage.getItem(GRID_SIZE_KEY);
  if (savedSize && GRID_CAPACITY_MAP[savedSize]) {
    currentGridSize = savedSize;
  } else {
    currentGridSize = "2x2";
  }

  // Sync size dropdown UI
  const sizeSelect = document.getElementById("gridSizeSelect");
  if (sizeSelect) {
    sizeSelect.value = currentGridSize;
  }

  // Fetch registered cameras and render matrix
  await reloadGridCameras(true);
  applyViewMode(currentViewMode, false);

  // Periodic lightweight camera & FPS refresh
  if (gridPollTimer) clearInterval(gridPollTimer);
  gridPollTimer = setInterval(refreshGridTelemetry, 3000);
}

/**
 * Switch between "grid" and "single" view modes
 */
function switchViewMode(mode) {
  if (mode !== "grid" && mode !== "single") return;
  currentViewMode = mode;
  localStorage.setItem(GRID_VIEW_MODE_KEY, mode);
  applyViewMode(mode, true);
}

/**
 * Apply the requested view mode to the DOM
 */
function applyViewMode(mode, userInitiated = true) {
  const dashboardMain = document.getElementById("dashboardMain") || document.querySelector(".dashboard-main");
  const btnGrid = document.getElementById("btnModeGrid");
  const btnSingle = document.getElementById("btnModeSingle");
  const sizeSelectorWrap = document.getElementById("gridSizeSelectorWrap");
  const singleStreamFeed = document.getElementById("streamFeed");

  if (mode === "grid") {
    // 1. Switch layout classes on body and main container
    document.body.classList.add("view-mode-grid");
    document.body.classList.remove("view-mode-single");
    document.body.style.overflow = "hidden";

    if (dashboardMain) {
      dashboardMain.classList.remove("mode-single");
      dashboardMain.classList.add("mode-grid");
    }

    // 2. Update navbar switcher active states
    const camSelectWrap = document.getElementById("cameraSelectWrap");
    const btnCalibrate = document.getElementById("btnToggleEditor");
    if (btnGrid) btnGrid.classList.add("active");
    if (btnSingle) btnSingle.classList.remove("active");
    if (sizeSelectorWrap) sizeSelectorWrap.style.display = "inline-flex";
    if (camSelectWrap) camSelectWrap.style.display = "none";
    if (btnCalibrate) btnCalibrate.style.display = "none";
    if (typeof toggleZoneEditor === "function") {
      const tb = document.getElementById("zoneToolbar");
      if (tb && tb.style.display !== "none") toggleZoneEditor(false);
    }

    // 3. Detach single view stream to prevent duplicate backend decoding
    if (singleStreamFeed) {
      singleStreamFeed.src = "";
    }

    // 4. Attach grid feeds
    attachAllGridStreams();
  } else {
    // 1. Switch layout classes on body and main container
    document.body.classList.add("view-mode-single");
    document.body.classList.remove("view-mode-grid");
    document.body.style.overflow = "hidden";

    if (dashboardMain) {
      dashboardMain.classList.remove("mode-grid");
      dashboardMain.classList.add("mode-single");
    }

    // 2. Update navbar switcher active states
    const camSelectWrap = document.getElementById("cameraSelectWrap");
    const btnCalibrate = document.getElementById("btnToggleEditor");
    if (btnGrid) btnGrid.classList.remove("active");
    if (btnSingle) btnSingle.classList.add("active");
    if (sizeSelectorWrap) sizeSelectorWrap.style.display = "none";
    if (camSelectWrap) camSelectWrap.style.display = "flex";
    if (btnCalibrate) btnCalibrate.style.display = "inline-flex";

    // 3. Detach all grid streams to save bandwidth and client CPU
    detachAllGridStreams();

    // 4. Attach active single camera feed
    const targetCam = (typeof activeCamera !== "undefined" && activeCamera) ? activeCamera : "cam_01";
    if (singleStreamFeed) {
      singleStreamFeed.src = `/video_feed/${targetCam}`;
    }
  }
}

/**
 * Change grid matrix layout size (2x2, 3x3, 4x4, 5x5, 6x6)
 */
/**
 * Change grid matrix layout size (2x2, 3x3, 4x4, 5x5, 6x6)
 */
function onGridSizeChange(newSize) {
  if (!GRID_CAPACITY_MAP[newSize]) return;
  currentGridSize = newSize;
  currentGridPage = 1; // Reset to page 1 on layout change
  localStorage.setItem(GRID_SIZE_KEY, newSize);
  renderGridMatrix();
  if (currentViewMode === "grid") {
    attachAllGridStreams();
  }
}

/**
 * Render the Grid Matrix DOM elements for the active page
 */
function renderGridMatrix() {
  const matrixContainer = document.getElementById("cameraGridMatrix");
  const countBadge = document.getElementById("gridActiveCamCount");
  const layoutBadge = document.getElementById("gridCurrentLayoutBadge");
  if (!matrixContainer) return;

  const totalSlots = GRID_CAPACITY_MAP[currentGridSize] || 4;
  const activeCount = (cachedCamerasList || []).length;
  const totalPages = Math.max(1, Math.ceil(activeCount / totalSlots));

  // Ensure currentGridPage is bounded within valid range
  if (currentGridPage > totalPages) {
    currentGridPage = 1;
  }

  if (countBadge) countBadge.textContent = `${activeCount} Kamera Aktif`;
  if (layoutBadge) layoutBadge.textContent = `Tata Letak: ${currentGridSize} (${totalSlots} Slot)`;

  // Update Pagination & Auto-Tour UI Controls
  const navControls = document.getElementById("gridNavControls");
  const pageIndicator = document.getElementById("gridPageIndicator");
  if (navControls) {
    if (totalPages > 1) {
      navControls.style.display = "inline-flex";
      if (pageIndicator) {
        pageIndicator.textContent = `Hal ${currentGridPage} / ${totalPages}`;
      }
    } else {
      navControls.style.display = "none";
      if (isAutoTourActive) {
        stopAutoTour(false);
      }
    }
  }

  // Slice cameras for the active page
  const startIndex = (currentGridPage - 1) * totalSlots;
  const pageCameras = (cachedCamerasList || []).slice(startIndex, startIndex + totalSlots);

  // Update grid-template class
  matrixContainer.className = `camera-grid-matrix grid-matrix-${currentGridSize}`;
  matrixContainer.innerHTML = "";

  for (let i = 0; i < totalSlots; i++) {
    if (i < pageCameras.length) {
      const cam = pageCameras[i];
      const cell = document.createElement("div");
      cell.className = "grid-cell active-cam";
      cell.id = `gridCell_${cam.id}`;
      cell.title = `Klik untuk Layar Penuh (${cam.name || cam.id})`;
      cell.onclick = () => toggleCameraFullscreen(cam.id);

      cell.innerHTML = `
        <div class="grid-cell-header">
          <div class="grid-cam-info">
            <span class="grid-live-indicator" id="gridLiveDot_${cam.id}"></span>
            <span class="grid-cam-id">${cam.id}</span>
            <span class="grid-cam-name">${cam.name || cam.id}</span>
          </div>
          <div class="grid-cell-kpi">
            <span class="grid-fps-pill" id="gridFps_${cam.id}">-- FPS</span>
          </div>
        </div>

        <img class="grid-stream-feed" id="gridImg_${cam.id}" data-cam-id="${cam.id}" alt="Stream ${cam.id}" />

        <div class="grid-hover-hint">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
          </svg>
          <span>Layar Penuh</span>
        </div>
      `;
      matrixContainer.appendChild(cell);
    } else {
      // Empty placeholder slot (zero network overhead, pure static HTML/SVG)
      const globalSlotIndex = startIndex + i + 1;
      const emptyCell = document.createElement("div");
      emptyCell.className = "grid-cell grid-cell-empty";
      emptyCell.innerHTML = `
        <div class="grid-empty-body">
          <svg class="grid-empty-icon" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M23 7l-7 5 7 5V7z"></path>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            <line x1="1" y1="1" x2="23" y2="23"></line>
          </svg>
          <span class="grid-empty-title">Slot #${globalSlotIndex} Kosong</span>
          <span class="grid-empty-sub">Tidak Ada Sinyal</span>
        </div>
      `;
      matrixContainer.appendChild(emptyCell);
    }
  }
}

/**
 * Navigate to previous grid page (with wrap-around)
 */
function prevGridPage() {
  const totalSlots = GRID_CAPACITY_MAP[currentGridSize] || 4;
  const totalPages = Math.max(1, Math.ceil((cachedCamerasList || []).length / totalSlots));
  if (totalPages <= 1) return;

  detachAllGridStreams();
  currentGridPage = currentGridPage > 1 ? currentGridPage - 1 : totalPages;
  renderGridMatrix();
  if (currentViewMode === "grid") {
    attachAllGridStreams();
  }
}

/**
 * Navigate to next grid page (with wrap-around)
 */
function nextGridPage() {
  const totalSlots = GRID_CAPACITY_MAP[currentGridSize] || 4;
  const totalPages = Math.max(1, Math.ceil((cachedCamerasList || []).length / totalSlots));
  if (totalPages <= 1) return;

  detachAllGridStreams();
  currentGridPage = currentGridPage < totalPages ? currentGridPage + 1 : 1;
  renderGridMatrix();
  if (currentViewMode === "grid") {
    attachAllGridStreams();
  }
}

/**
 * Toggle automated patrol / tour between grid camera pages
 */
function toggleAutoTour(forceState) {
  const totalSlots = GRID_CAPACITY_MAP[currentGridSize] || 4;
  const totalPages = Math.max(1, Math.ceil((cachedCamerasList || []).length / totalSlots));

  if (totalPages <= 1) {
    if (typeof showToast === "function") {
      showToast("Auto-Tour", "Semua kamera sudah muat dalam satu halaman.", "info", 2500);
    }
    return;
  }

  if (typeof forceState === "boolean") {
    isAutoTourActive = forceState;
  } else {
    isAutoTourActive = !isAutoTourActive;
  }

  updateTourControlsUI();

  if (isAutoTourActive) {
    startAutoTourTimer();
    if (typeof showToast === "function") {
      showToast("Auto-Tour Aktif", `Rotasi kamera otomatis setiap ${autoTourIntervalSec} detik.`, "success", 2500);
    }
  } else {
    stopAutoTour(true);
  }
}

function startAutoTourTimer() {
  if (autoTourTimer) clearInterval(autoTourTimer);
  autoTourTimer = setInterval(() => {
    if (currentViewMode === "grid") {
      nextGridPage();
    }
  }, autoTourIntervalSec * 1000);
}

function stopAutoTour(notify = false) {
  isAutoTourActive = false;
  if (autoTourTimer) {
    clearInterval(autoTourTimer);
    autoTourTimer = null;
  }
  updateTourControlsUI();
  if (notify && typeof showToast === "function") {
    showToast("Auto-Tour Berhenti", "Patroli otomatis dinonaktifkan.", "info", 2000);
  }
}

function updateTourControlsUI() {
  const btn = document.getElementById("btnGridTourToggle");
  const iconPlay = document.getElementById("tourIconPlay");
  const iconPause = document.getElementById("tourIconPause");
  const text = document.getElementById("tourStatusText");

  if (btn) {
    if (isAutoTourActive) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  }
  if (iconPlay) iconPlay.style.display = isAutoTourActive ? "none" : "inline-block";
  if (iconPause) iconPause.style.display = isAutoTourActive ? "inline-block" : "none";
  if (text) text.textContent = isAutoTourActive ? "Patroli" : "Auto-Tour";
}

function onTourIntervalChange(sec) {
  const parsed = parseInt(sec, 10);
  if (!isNaN(parsed) && parsed > 0) {
    autoTourIntervalSec = parsed;
    if (isAutoTourActive) {
      startAutoTourTimer();
      if (typeof showToast === "function") {
        showToast("Interval Diperbarui", `Durasi rotasi diatur ke ${autoTourIntervalSec} detik.`, "info", 2000);
      }
    }
  }
}

/**
 * Click-to-Fullscreen: Open clicked camera in native browser Fullscreen mode
 */
function toggleCameraFullscreen(cameraId) {
  if (!cameraId) return;
  const cell = document.getElementById(`gridCell_${cameraId}`);
  if (!cell) return;

  if (document.fullscreenElement) {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  } else {
    if (cell.requestFullscreen) {
      cell.requestFullscreen();
    } else if (cell.webkitRequestFullscreen) {
      cell.webkitRequestFullscreen();
    } else if (cell.msRequestFullscreen) {
      cell.msRequestFullscreen();
    }
  }
}

/**
 * Attach streams to all active grid cells
 */
function attachAllGridStreams() {
  const images = document.querySelectorAll(".grid-stream-feed");
  images.forEach((img) => {
    const camId = img.getAttribute("data-cam-id");
    if (camId && (!img.src || img.src.endsWith("/") || img.src === window.location.href)) {
      img.src = `/video_feed/${camId}`;
    }
  });
}

/**
 * Detach all grid streams to save bandwidth and server resources
 */
function detachAllGridStreams() {
  const images = document.querySelectorAll(".grid-stream-feed");
  images.forEach((img) => {
    img.src = "";
  });
}

/**
 * Dynamically re-sync registered cameras into the Grid Matrix
 * @param {boolean} autoExpandIfNeeded - Auto-scale grid size if camera count exceeds capacity
 */
async function reloadGridCameras(autoExpandIfNeeded = true) {
  try {
    const resp = await fetch("/api/cameras");
    if (!resp.ok) return;
    const cams = await resp.json();
    if (!Array.isArray(cams)) return;

    cachedCamerasList = cams;

    if (!cachedCamerasList || cachedCamerasList.length === 0) {
      cachedCamerasList = [
        { id: "cam_01", name: "Koridor Utama" },
        { id: "cam_02", name: "Area Parkir POS-2" },
        { id: "cam_03", name: "Jalur Logistik Arah POS-1" },
        { id: "cam_04", name: "Area Timbangan Truk" },
      ];
    }

    // Auto-scale grid layout if total cameras exceed capacity of current grid size
    if (autoExpandIfNeeded) {
      const curCapacity = GRID_CAPACITY_MAP[currentGridSize] || 4;
      if (cachedCamerasList.length > curCapacity) {
        const sizeOrder = ["2x2", "3x3", "4x4", "5x5", "6x6"];
        for (const size of sizeOrder) {
          if (GRID_CAPACITY_MAP[size] >= cachedCamerasList.length) {
            currentGridSize = size;
            break;
          }
        }
        localStorage.setItem(GRID_SIZE_KEY, currentGridSize);
        const sizeSelect = document.getElementById("gridSizeSelect");
        if (sizeSelect) sizeSelect.value = currentGridSize;
      }
    }

    // Re-render DOM matrix and attach streams if in grid view
    renderGridMatrix();
    if (currentViewMode === "grid") {
      attachAllGridStreams();
    }
  } catch (err) {
    console.warn("[GridManager] Failed reloading grid cameras:", err);
  }
}

/**
 * Lightweight telemetry polling for grid cells & dynamic lifecycle sync
 */
async function refreshGridTelemetry() {
  if (currentViewMode !== "grid") return;

  try {
    const resp = await fetch("/api/cameras");
    if (resp.ok) {
      const cams = await resp.json();
      if (!Array.isArray(cams)) return;

      // Detect if camera additions or removals occurred in background
      const currentIds = (cachedCamerasList || []).map((c) => c.id).join(",");
      const newIds = cams.map((c) => c.id).join(",");
      if (currentIds !== newIds) {
        console.log("[GridManager] Camera lifecycle change detected. Updating grid matrix...");
        await reloadGridCameras(true);
        return;
      }

      // Smoothly update FPS & indicators on existing cells without interrupting video streams
      for (const cam of cams) {
        const fpsPill = document.getElementById(`gridFps_${cam.id}`);
        if (fpsPill && typeof cam.fps !== "undefined") {
          fpsPill.textContent = `${Number(cam.fps).toFixed(1)} FPS`;
        }
        const liveDot = document.getElementById(`gridLiveDot_${cam.id}`);
        if (liveDot) {
          if (cam.online) {
            liveDot.classList.remove("offline");
          } else {
            liveDot.classList.add("offline");
          }
        }
      }
    }
  } catch (e) {
    // Suppress background poll errors
  }
}

// Global Keyboard Navigation for Grid Pagination (ArrowLeft / ArrowRight)
document.addEventListener("keydown", (e) => {
  if (currentViewMode !== "grid") return;
  const activeEl = document.activeElement;
  if (activeEl && (activeEl.tagName === "INPUT" || activeEl.tagName === "TEXTAREA" || activeEl.tagName === "SELECT")) {
    return;
  }
  if (e.key === "ArrowLeft") {
    prevGridPage();
  } else if (e.key === "ArrowRight") {
    nextGridPage();
  }
});
