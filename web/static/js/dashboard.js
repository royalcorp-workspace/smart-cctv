/**
 * Smart CCTV 2.0 - Enterprise Web Dashboard Client Logic
 * Handles dual-theme toggle (Dark/Light), camera switching, telemetry polling, and fullscreen.
 */

let activeCamera = "cam_01";
let pollTimer = null;

/* ==========================================================================
   1. THEME SWITCHER (DARK / LIGHT DUAL-MODE)
   ========================================================================== */

const THEME_STORAGE_KEY = "smart_cctv_theme";

/**
 * Apply theme to document element and update accessibility labels.
 */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const toggleBtn = document.getElementById("themeToggle");
  if (toggleBtn) {
    const isDark = theme === "dark";
    toggleBtn.setAttribute("aria-label", isDark ? "Ganti ke Mode Terang" : "Ganti ke Mode Gelap");
    toggleBtn.setAttribute("title", isDark ? "Ganti ke Mode Terang" : "Ganti ke Mode Gelap");
  }
}

/**
 * Initialize theme based on stored preference or OS color scheme.
 */
function initTheme() {
  const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === "light" || storedTheme === "dark") {
    applyTheme(storedTheme);
  } else {
    // Detect OS preference
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme = prefersDark ? "dark" : "light";
    applyTheme(initialTheme);
  }

  // Listen to OS scheme change if user hasn't explicitly overridden
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
      if (!localStorage.getItem(THEME_STORAGE_KEY)) {
        applyTheme(e.matches ? "dark" : "light");
      }
    });
  }
}

/**
 * Toggle between dark and light mode.
 */
function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_STORAGE_KEY, newTheme);
  applyTheme(newTheme);
}

/* ==========================================================================
   2. DIGITAL CLOCK (WIB)
   ========================================================================== */

function updateClock() {
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0] + " WIB";
  const clockEl = document.getElementById("systemClock");
  if (clockEl) {
    clockEl.textContent = timeStr;
  }
}
setInterval(updateClock, 1000);
updateClock();

/* ==========================================================================
   3. CAMERA LIST & SWITCHER
   ========================================================================== */

/**
 * Fetch registered cameras from FastAPI backend and populate dropdown selector.
 */
async function loadCameraList() {
  try {
    const resp = await fetch("/api/cameras");
    if (!resp.ok) return;
    const cameras = await resp.json();
    const select = document.getElementById("cameraSelect");
    if (select && cameras && cameras.length > 0) {
      const currentVal = select.value;
      select.innerHTML = "";
      cameras.forEach((cam) => {
        const opt = document.createElement("option");
        opt.value = cam.id;
        opt.textContent = `${cam.id} (${cam.name || "Kamera"})`;
        if (cam.id === currentVal || cam.id === activeCamera) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });
      if (!cameras.some((c) => c.id === activeCamera)) {
        activeCamera = cameras[0].id;
        select.value = activeCamera;
      }
    }
  } catch (err) {
    console.warn("Gagal memuat daftar kamera:", err);
  }
}

/**
 * Dropdown change handler for instant camera feed switching.
 */
function onCameraChange() {
  const select = document.getElementById("cameraSelect");
  if (!select) return;
  const newCam = select.value;
  if (newCam === activeCamera) return;

  activeCamera = newCam;
  const badgeCam = document.getElementById("badgeCamId");
  if (badgeCam) {
    badgeCam.textContent = activeCamera;
  }

  // Instantly swap MJPEG stream src without page reload
  const feedImg = document.getElementById("streamFeed");
  if (feedImg) {
    feedImg.src = `/video_feed/${activeCamera}?t=${Date.now()}`;
  }

  // Reset telemetry display immediately
  const fpsEl = document.getElementById("statFps");
  if (fpsEl) fpsEl.textContent = "--";
  const tracksEl = document.getElementById("statTracks");
  if (tracksEl) tracksEl.textContent = "0";
  const violEl = document.getElementById("statViolations");
  if (violEl) violEl.textContent = "0";

  pollTelemetry();
}

function onStreamError() {
  const pill = document.getElementById("pillStatus");
  if (pill) {
    pill.textContent = "Disconnected";
    pill.className = "status-badge disconnected";
  }
}

function onStreamLoad() {
  const pill = document.getElementById("pillStatus");
  if (pill) {
    pill.textContent = "Connected";
    pill.className = "status-badge connected";
  }
}

/* ==========================================================================
   4. PERIODIC TELEMETRY POLLING
   ========================================================================== */

/**
 * Poll camera telemetry every 1000ms and update KPI cards.
 */
async function pollTelemetry() {
  try {
    const resp = await fetch(`/api/status/${activeCamera}`);
    if (!resp.ok) return;
    const data = await resp.json();

    // 1. Update camera metadata name
    const metaCam = document.getElementById("metaCamName");
    if (metaCam && data.name) {
      metaCam.textContent = data.name;
    }

    // 2. Update Pipeline FPS & Active Tracks
    const statFps = document.getElementById("statFps");
    if (statFps) {
      statFps.textContent = typeof data.fps === "number" ? data.fps.toFixed(1) : "--";
    }
    const statTracks = document.getElementById("statTracks");
    if (statTracks) {
      statTracks.textContent = data.active_tracks !== undefined ? data.active_tracks : "0";
    }

    // 3. Update RTSP Status Pill
    const pillStatus = document.getElementById("pillStatus");
    if (pillStatus) {
      if (data.online) {
        pillStatus.textContent = data.rtsp_status || "Connected";
        pillStatus.className = "status-badge connected";
      } else {
        pillStatus.textContent = data.rtsp_status || "Disconnected";
        pillStatus.className = "status-badge disconnected";
      }
    }

    // 4. Update Clear Area Violations Card
    const violations = data.violations !== undefined ? data.violations : (data.clear_area_count || 0);
    const statViol = document.getElementById("statViolations");
    if (statViol) {
      statViol.textContent = violations;
    }

    const pillViolations = document.getElementById("pillViolations");
    if (pillViolations) {
      if (violations > 0) {
        pillViolations.textContent = `${violations} Pelanggaran!`;
        pillViolations.className = "status-badge disconnected";
      } else {
        pillViolations.textContent = "Aman";
        pillViolations.className = "status-badge connected";
      }
    }

    // 5. Update Biometric Identified Faces List (SFace)
    const facesList = document.getElementById("facesList");
    const faces = data.identified_faces || [];
    const faceCount = document.getElementById("faceCount");
    if (faceCount) {
      faceCount.textContent = `${faces.length} Terdeteksi`;
    }

    if (facesList) {
      if (faces.length === 0) {
        facesList.innerHTML = '<div class="empty-state">Belum ada wajah terdeteksi pada feed saat ini</div>';
      } else {
        facesList.innerHTML = faces.map((face) => {
          const rawName = face.name || "Unknown";
          const isKnown = rawName.toLowerCase() !== "unknown";
          const initials = isKnown ? rawName.substring(0, 2).toUpperCase() : "??";
          const tagClass = isKnown ? "tag-badge authorized" : "tag-badge unknown";
          const tagText = isKnown ? "Terdaftar" : "Tamu / Unknown";
          const conf = (isKnown && face.confidence) ? `(${(face.confidence * 100).toFixed(0)}%)` : "";
          const zoneText = face.zone || "Area Pantau";

          return `
            <div class="person-card">
              <div class="person-profile">
                <div class="person-avatar ${isKnown ? "" : "unknown"}">${initials}</div>
                <div class="person-meta">
                  <span class="person-name">${rawName} ${conf}</span>
                  <span class="person-zone">${zoneText}</span>
                </div>
              </div>
              <span class="${tagClass}">${tagText}</span>
            </div>
          `;
        }).join("");
      }
    }

  } catch (err) {
    console.warn("Telemetry polling warning:", err);
  }
}

/* ==========================================================================
   5. FULLSCREEN HANDLER & TOAST NOTIFICATION SYSTEM
   ========================================================================== */

function toggleFullscreen() {
  const container = document.getElementById("videoContainer");
  if (!container) return;

  if (!document.fullscreenElement) {
    container.requestFullscreen().catch((err) => {
      showToast("Layar Penuh Gagal", err.message, "error");
    });
  } else {
    document.exitFullscreen();
  }
}

/**
 * Enterprise Non-Blocking Toast Notification.
 * @param {string} title - Header text
 * @param {string} message - Descriptive notification body
 * @param {'success'|'error'|'warning'} type - Accent style
 * @param {number} duration - Auto-dismiss timeout (ms)
 */
function showToast(title, message, type = "success", duration = 3500) {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const iconSvg =
    type === "success"
      ? `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`
      : type === "error"
      ? `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
      : `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;

  toast.innerHTML = `
    ${iconSvg}
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      <div class="toast-message">${message}</div>
    </div>
    <button type="button" class="toast-close-btn" aria-label="Tutup">&times;</button>
  `;

  const closeBtn = toast.querySelector(".toast-close-btn");
  const dismiss = () => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(40px)";
    setTimeout(() => toast.remove(), 250);
  };

  closeBtn.addEventListener("click", dismiss);
  container.appendChild(toast);

  if (duration > 0) {
    setTimeout(dismiss, duration);
  }
}

/* ==========================================================================
   6. INTERACTIVE WEB ZONE EDITOR (ENTERPRISE VMS CALIBRATION)
   ========================================================================== */

let zoneEditorActive = false;
let zoneData = {};
let originalZoneData = {};
let historyStack = [];
let selectedZone = "zone_2_transit";
let zoneMode = "drag"; // "drag" | "add"
let selectedVertexIndex = -1;
let draggedPointIndex = -1;
let hoverPointIndex = -1;
let ghostMidpoint = null;
let isMouseDown = false;
let hasMovedDuringDrag = false;

// Strict Enterprise Palette
const ZONE_COLOR_SCHEMES = {
  zone_2_transit: {
    name: "Zona 2: Transit Steril",
    stroke: "#10b981", // Emerald Green
    fill: "rgba(16, 185, 129, 0.15)", // 15% Transparent
    handle: "#34d399",
    glow: "rgba(16, 185, 129, 0.50)",
    dot: "#10b981"
  },
  zone_1_koridor: {
    name: "Zona 1: Koridor Akses",
    stroke: "#f59e0b", // Amber / Warm Orange
    fill: "rgba(245, 158, 11, 0.10)", // 10% Transparent
    handle: "#fbbf24",
    glow: "rgba(245, 158, 11, 0.50)",
    dot: "#f59e0b"
  },
  invalid: {
    stroke: "#ef4444", // Fiery Red
    fill: "rgba(239, 68, 68, 0.25)",
    handle: "#f87171",
    glow: "rgba(239, 68, 68, 0.65)",
    dot: "#ef4444"
  },
  default: {
    name: "Area Khusus",
    stroke: "#3b82f6",
    fill: "rgba(59, 130, 246, 0.15)",
    handle: "#60a5fa",
    glow: "rgba(59, 130, 246, 0.50)",
    dot: "#3b82f6"
  }
};

function getActiveZoneColors(zKey, isInvalid = false) {
  if (isInvalid) return ZONE_COLOR_SCHEMES.invalid;
  return ZONE_COLOR_SCHEMES[zKey] || ZONE_COLOR_SCHEMES.default;
}

/**
 * Save snapshot of current zone configuration to undo history.
 */
function pushHistory() {
  historyStack.push(JSON.parse(JSON.stringify(zoneData)));
  if (historyStack.length > 25) {
    historyStack.shift();
  }
}

/**
 * Undo last polygon action (Ctrl+Z).
 */
function undoZoneAction() {
  if (historyStack.length === 0) {
    showToast("Undo", "Tidak ada perubahan yang dapat diurungkan.", "warning", 2000);
    return;
  }
  zoneData = historyStack.pop();
  selectedVertexIndex = -1;
  draggedPointIndex = -1;
  hoverPointIndex = -1;
  ghostMidpoint = null;
  renderZoneCanvas();
  showToast("Undo", "Perubahan sebelumnya berhasil diurungkan.", "warning", 2000);
}

/**
 * Convert mouse client coordinates to 1920x1080 native coordinates.
 */
function getNativeCoords(e) {
  const canvas = document.getElementById("zoneEditorCanvas");
  if (!canvas) return { x: 0, y: 0 };
  const rect = canvas.getBoundingClientRect();
  const scaleX = 1920 / rect.width;
  const scaleY = 1080 / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);
  return {
    x: Math.max(0, Math.min(1920, x)),
    y: Math.max(0, Math.min(1080, y))
  };
}

/**
 * Toggle interactive zone editor overlay and floating toolbar.
 */
async function toggleZoneEditor(forceState) {
  const canvas = document.getElementById("zoneEditorCanvas");
  const toolbar = document.getElementById("zoneToolbar");
  const btn = document.getElementById("btnToggleEditor");
  if (!canvas || !toolbar) return;

  zoneEditorActive = typeof forceState === "boolean" ? forceState : !zoneEditorActive;

  if (zoneEditorActive) {
    canvas.classList.add("active");
    toolbar.style.display = "flex";
    if (btn) btn.classList.add("active");
    await fetchCameraZones();
    setupCanvasEvents();
    renderZoneCanvas();
    showToast(
      "Mode Kalibrasi Aktif",
      "Klik dan geser titik untuk mengatur area. Tekan Esc untuk keluar.",
      "success",
      3000
    );
  } else {
    canvas.classList.remove("active");
    toolbar.style.display = "none";
    if (btn) btn.classList.remove("active");
    selectedVertexIndex = -1;
    draggedPointIndex = -1;
    hoverPointIndex = -1;
    ghostMidpoint = null;
  }
}

/**
 * Fetch ROI zones from backend for active camera.
 */
async function fetchCameraZones() {
  try {
    const resp = await fetch(`/api/zones/${activeCamera}`);
    if (!resp.ok) throw new Error("Gagal mengambil data zona kamera.");
    const data = await resp.json();
    zoneData = JSON.parse(JSON.stringify(data.zones || {}));
    originalZoneData = JSON.parse(JSON.stringify(zoneData));
    historyStack = [];

    // Populate zone dropdown selector
    const selector = document.getElementById("zoneSelector");
    if (selector) {
      selector.innerHTML = "";
      const zoneKeys = Object.keys(zoneData);
      zoneKeys.forEach((zk) => {
        const opt = document.createElement("option");
        opt.value = zk;
        const cfg = ZONE_COLOR_SCHEMES[zk] || ZONE_COLOR_SCHEMES.default;
        opt.textContent = cfg.name || zk;
        selector.appendChild(opt);
      });
      if (zoneKeys.length > 0) {
        if (!zoneKeys.includes(selectedZone)) {
          selectedZone = zoneKeys[0];
        }
        selector.value = selectedZone;
      }
      updateToolbarDot();
    }
  } catch (err) {
    console.error("Error fetching zones:", err);
    showToast("Gagal Memuat Zona", "Tidak dapat terhubung ke server kamera.", "error");
  }
}

function updateToolbarDot() {
  const dot = document.getElementById("zoneColorDot");
  if (dot) {
    const colors = getActiveZoneColors(selectedZone);
    dot.style.backgroundColor = colors.stroke;
    dot.style.boxShadow = `0 0 8px ${colors.stroke}`;
  }
}

function onZoneSelectChange() {
  const selector = document.getElementById("zoneSelector");
  if (selector) {
    selectedZone = selector.value;
    selectedVertexIndex = -1;
    draggedPointIndex = -1;
    hoverPointIndex = -1;
    ghostMidpoint = null;
    updateToolbarDot();
    renderZoneCanvas();
  }
}

function setZoneMode(mode) {
  zoneMode = mode;
  document.getElementById("btnModeDrag")?.classList.toggle("active", mode === "drag");
  document.getElementById("btnModeAdd")?.classList.toggle("active", mode === "add");
  renderZoneCanvas();
}

function resetCurrentZone() {
  if (originalZoneData[selectedZone]) {
    pushHistory();
    zoneData[selectedZone] = JSON.parse(JSON.stringify(originalZoneData[selectedZone]));
    selectedVertexIndex = -1;
    draggedPointIndex = -1;
    hoverPointIndex = -1;
    ghostMidpoint = null;
    renderZoneCanvas();
    showToast("Reset Selesai", `Koordinat ${selectedZone} dikembalikan ke konfigurasi tersimpan.`, "warning", 2500);
  }
}

function deleteSelectedPoint() {
  const pts = zoneData[selectedZone] || [];
  if (selectedVertexIndex < 0 || selectedVertexIndex >= pts.length) {
    showToast("Pilih Titik", "Klik salah satu titik sudut terlebih dahulu sebelum menghapus.", "warning", 2500);
    return;
  }
  if (pts.length <= 3) {
    showToast("Batas Minimum Titik", "Zona poligon harus memiliki minimal 3 titik koordinat!", "warning", 3000);
    return;
  }

  pushHistory();
  pts.splice(selectedVertexIndex, 1);
  selectedVertexIndex = -1;
  hoverPointIndex = -1;
  renderZoneCanvas();
  showToast("Titik Dihapus", "Titik poligon berhasil dihapus.", "success", 2000);
}

/**
 * Find point near mouse within threshold.
 */
function findVertexNear(nativeX, nativeY, pts, thresholdPx = 30) {
  for (let i = 0; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - nativeX, pts[i][1] - nativeY);
    if (d <= thresholdPx) {
      return i;
    }
  }
  return -1;
}

/**
 * Distance and projection from point (px, py) to line segment (x1, y1)-(x2, y2).
 */
function projectOnSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const l2 = dx * dx + dy * dy;
  if (l2 === 0) return { d: Math.hypot(px - x1, py - y1), x: x1, y: y1, t: 0 };
  let t = ((px - x1) * dx + (py - y1) * dy) / l2;
  t = Math.max(0.06, Math.min(0.94, t));
  const projX = x1 + t * dx;
  const projY = y1 + t * dy;
  return {
    d: Math.hypot(px - projX, py - projY),
    x: Math.round(projX),
    y: Math.round(projY),
    t: t
  };
}

/**
 * Computational Geometry: Check polygon self-intersection.
 */
function ccw(A, B, C) {
  return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0]);
}

function segmentsIntersect(A, B, C, D) {
  if ((A[0] === C[0] && A[1] === C[1]) || (A[0] === D[0] && A[1] === D[1]) ||
      (B[0] === C[0] && B[1] === C[1]) || (B[0] === D[0] && B[1] === D[1])) {
    return false;
  }
  return (ccw(A, C, D) !== ccw(B, C, D)) && (ccw(A, B, C) !== ccw(A, B, D));
}

function isPolygonSelfIntersecting(points) {
  const n = points.length;
  if (n < 4) return false;
  for (let i = 0; i < n; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % n];
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(i - j) <= 1 || (i === 0 && j === n - 1)) continue;
      const p3 = points[j];
      const p4 = points[(j + 1) % n];
      if (segmentsIntersect(p1, p2, p3, p4)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Setup canvas mouse event listeners for dragging, midpoint insertion, and deleting vertices.
 */
let canvasEventsInitialized = false;
function setupCanvasEvents() {
  if (canvasEventsInitialized) return;
  const canvas = document.getElementById("zoneEditorCanvas");
  if (!canvas) return;

  canvas.addEventListener("mousemove", (e) => {
    if (!zoneEditorActive) return;
    const { x, y } = getNativeCoords(e);
    const coordDisplay = document.getElementById("zoneCoordDisplay");
    if (coordDisplay) {
      coordDisplay.textContent = `X: ${x}, Y: ${y}`;
    }

    const pts = zoneData[selectedZone] || [];

    if (isMouseDown && draggedPointIndex >= 0 && draggedPointIndex < pts.length) {
      hasMovedDuringDrag = true;
      pts[draggedPointIndex] = [x, y];
      ghostMidpoint = null;
      renderZoneCanvas();
    } else {
      const prevHover = hoverPointIndex;
      hoverPointIndex = findVertexNear(x, y, pts, 32);

      // Midpoint detection: check if hovering near an edge
      if (hoverPointIndex === -1 && pts.length >= 3) {
        let bestEdge = null;
        let minDist = Infinity;
        for (let i = 0; i < pts.length; i++) {
          const p1 = pts[i];
          const p2 = pts[(i + 1) % pts.length];
          const res = projectOnSegment(x, y, p1[0], p1[1], p2[0], p2[1]);
          if (res.d < minDist && res.d <= 22) {
            minDist = res.d;
            bestEdge = { edgeIdx: i, x: res.x, y: res.y };
          }
        }
        ghostMidpoint = bestEdge;
      } else {
        ghostMidpoint = null;
      }

      if (prevHover !== hoverPointIndex || ghostMidpoint !== null) {
        renderZoneCanvas();
      }
    }
  });

  canvas.addEventListener("mousedown", (e) => {
    if (!zoneEditorActive) return;
    isMouseDown = true;
    hasMovedDuringDrag = false;
    const { x, y } = getNativeCoords(e);
    const pts = zoneData[selectedZone] || [];

    const hitVertex = findVertexNear(x, y, pts, 32);

    if (hitVertex >= 0) {
      // User clicked existing vertex
      selectedVertexIndex = hitVertex;
      draggedPointIndex = hitVertex;
      pushHistory();
      renderZoneCanvas();
    } else if (ghostMidpoint !== null) {
      // User clicked smart ghost midpoint -> insert new vertex
      pushHistory();
      pts.splice(ghostMidpoint.edgeIdx + 1, 0, [ghostMidpoint.x, ghostMidpoint.y]);
      selectedVertexIndex = ghostMidpoint.edgeIdx + 1;
      draggedPointIndex = selectedVertexIndex;
      ghostMidpoint = null;
      renderZoneCanvas();
      showToast("Titik Baru Disisipkan", "Titik poligon baru ditambahkan. Geser untuk memposisikan.", "success", 2000);
    } else if (zoneMode === "add") {
      // General add mode: click anywhere to append
      pushHistory();
      pts.push([x, y]);
      selectedVertexIndex = pts.length - 1;
      draggedPointIndex = selectedVertexIndex;
      renderZoneCanvas();
    } else {
      selectedVertexIndex = -1;
      renderZoneCanvas();
    }
  });

  window.addEventListener("mouseup", () => {
    if (!zoneEditorActive) return;
    if (isMouseDown && hasMovedDuringDrag) {
      // Point movement completed
    }
    isMouseDown = false;
    draggedPointIndex = -1;
    renderZoneCanvas();
  });

  // Global Keyboard Shortcuts (Esc to close, Delete/Backspace to delete vertex, Ctrl+Z to undo)
  window.addEventListener("keydown", (e) => {
    if (!zoneEditorActive) return;

    if (e.key === "Escape") {
      toggleZoneEditor(false);
    } else if (e.key === "Delete" || e.key === "Backspace") {
      // Avoid intercepting if focus is in an input or select
      if (document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        return;
      }
      if (selectedVertexIndex >= 0) {
        e.preventDefault();
        deleteSelectedPoint();
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      undoZoneAction();
    }
  });

  canvasEventsInitialized = true;
}

/**
 * Render polygon overlays, handles, and labels on the 1080p canvas.
 */
function renderZoneCanvas() {
  const canvas = document.getElementById("zoneEditorCanvas");
  if (!canvas || !zoneEditorActive) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw non-selected zones in background (Semi-transparent reference)
  Object.keys(zoneData).forEach((zKey) => {
    if (zKey === selectedZone) return;
    const pts = zoneData[zKey];
    if (!pts || pts.length < 3) return;
    const colors = getActiveZoneColors(zKey);

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(148, 163, 184, 0.08)";
    ctx.fill();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // 2. Draw currently selected active zone
  const activePts = zoneData[selectedZone];
  if (!activePts || activePts.length === 0) return;

  const selfIntersect = isPolygonSelfIntersecting(activePts);
  const colors = getActiveZoneColors(selectedZone, selfIntersect);

  // Update Toolbar Validation Pill and Save Button State
  const valPill = document.getElementById("zoneValidationPill");
  const saveBtn = document.getElementById("btnSaveZones");
  if (valPill) {
    if (selfIntersect) {
      valPill.textContent = "⚠ Garis Bersilangan";
      valPill.className = "glass-validation-pill invalid";
    } else {
      valPill.textContent = "Poligon Valid";
      valPill.className = "glass-validation-pill valid";
    }
  }
  if (saveBtn) {
    saveBtn.disabled = selfIntersect;
    saveBtn.title = selfIntersect ? "Perbaiki garis bersilangan sebelum menyimpan." : "Simpan & Terapkan (Atomic Hot-Reload)";
  }

  // Draw Polygon Body
  ctx.beginPath();
  ctx.moveTo(activePts[0][0], activePts[0][1]);
  for (let i = 1; i < activePts.length; i++) {
    ctx.lineTo(activePts[i][0], activePts[i][1]);
  }
  ctx.closePath();

  ctx.fillStyle = colors.fill;
  ctx.fill();
  ctx.strokeStyle = colors.stroke;
  ctx.lineWidth = selfIntersect ? 4 : 3;
  ctx.stroke();

  // 3. Draw Smart Midpoint (+) Ghost Handle if hovering over an edge
  if (ghostMidpoint !== null && !isMouseDown) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(ghostMidpoint.x, ghostMidpoint.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = colors.stroke;
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Plus sign inside
    ctx.beginPath();
    ctx.moveTo(ghostMidpoint.x - 4, ghostMidpoint.y);
    ctx.lineTo(ghostMidpoint.x + 4, ghostMidpoint.y);
    ctx.moveTo(ghostMidpoint.x, ghostMidpoint.y - 4);
    ctx.lineTo(ghostMidpoint.x, ghostMidpoint.y + 4);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Small tooltip
    ctx.font = "bold 11px -apple-system, sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
    ctx.fillRect(ghostMidpoint.x + 12, ghostMidpoint.y - 12, 80, 20);
    ctx.fillStyle = "#ffffff";
    ctx.fillText("+ Tambah Titik", ghostMidpoint.x + 16, ghostMidpoint.y + 2);
    ctx.restore();
  }

  // 4. Draw Vertex Handles with Glow Rings and Tooltips
  activePts.forEach((pt, idx) => {
    const isHover = idx === hoverPointIndex;
    const isSelected = idx === selectedVertexIndex;
    const isDragged = idx === draggedPointIndex;

    // Radius scaling: 5px normal -> 8px hover -> 10px selected/dragged
    const radius = (isSelected || isDragged) ? 10 : (isHover ? 8 : 5);

    // Glow Ring
    if (isSelected || isDragged || isHover) {
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], radius + 6, 0, 2 * Math.PI);
      ctx.fillStyle = colors.glow;
      ctx.fill();
    }

    // Outer Circle
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], radius, 0, 2 * Math.PI);
    ctx.fillStyle = (isSelected || isDragged) ? "#ffffff" : (isHover ? colors.handle : colors.stroke);
    ctx.fill();
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Real-Time Coordinate Tooltip (above handle)
    if (isDragged || isSelected) {
      const coordText = `X: ${Math.round(pt[0])}, Y: ${Math.round(pt[1])}`;
      ctx.font = "bold 12px ui-monospace, Consolas, monospace";
      const textWidth = ctx.measureText(coordText).width;
      const padX = 8;
      const padY = 4;
      const badgeW = textWidth + padX * 2;
      const badgeH = 22;
      const badgeX = pt[0] - badgeW / 2;
      const badgeY = pt[1] - radius - badgeH - 6;

      // Dark Glass Pill
      ctx.save();
      ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
      ctx.strokeStyle = colors.stroke;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 5);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#f8fafc";
      ctx.fillText(coordText, badgeX + padX, badgeY + 15);
      ctx.restore();
    }
  });

  // 5. If self-intersecting, draw bold warning banner at bottom
  if (selfIntersect) {
    ctx.save();
    ctx.font = "bold 16px -apple-system, sans-serif";
    const warnMsg = "⚠ Poligon Bersilangan! Garis tidak boleh memotong satu sama lain.";
    const wText = ctx.measureText(warnMsg).width;
    const bx = (1920 - wText) / 2 - 16;
    const by = 1010;

    ctx.fillStyle = "rgba(239, 68, 68, 0.92)";
    ctx.beginPath();
    ctx.roundRect(bx, by, wText + 32, 38, 8);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.fillText(warnMsg, bx + 16, by + 24);
    ctx.restore();
  }
}

/**
 * Save updated zones to FastAPI backend with hot-reload.
 */
async function saveZonesToServer() {
  // Validate all zones before submitting
  for (const [zKey, pts] of Object.entries(zoneData)) {
    if (pts.length < 3) {
      showToast("Gagal Menyimpan", `Zona '${zKey}' memiliki titik kurang dari 3! Tambahkan titik.`, "warning");
      return;
    }
    if (isPolygonSelfIntersecting(pts)) {
      showToast("Geometri Bersilangan", `Garis pada zona '${zKey}' saling bersilangan! Harap rapikan titik sebelum menyimpan.`, "error");
      return;
    }
  }

  const saveBtn = document.getElementById("btnSaveZones");
  if (saveBtn) {
    saveBtn.disabled = true;
    saveBtn.innerHTML = `<span>Menyimpan...</span>`;
  }

  try {
    const resp = await fetch("/api/zones/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_id: activeCamera,
        zones: zoneData,
        base_resolution: [1920, 1080]
      })
    });

    const res = await resp.json();
    if (!resp.ok) {
      throw new Error(res.detail || "Gagal menyimpan konfigurasi zona.");
    }

    originalZoneData = JSON.parse(JSON.stringify(zoneData));
    showToast(
      "Zona Berhasil Disimpan & Diterapkan",
      "Perubahan poligon telah dimuat ulang secara instan ke thread pipeline.",
      "success",
      3500
    );
    toggleZoneEditor(false);
  } catch (err) {
    console.error("Save zones error:", err);
    showToast("Gagal Menyimpan", err.message, "error");
  } finally {
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
        <span>Simpan</span>
      `;
    }
  }
}


/* ==========================================================================
   7. INCIDENT DVR PLAYBACK & EVENT LOGS
   ========================================================================== */

/**
 * Fetch and render recent incident events in the DVR playback table.
 */
async function loadRecentEvents() {
  const tbody = document.getElementById("incidentsTableBody");
  if (!tbody) return;

  try {
    const resp = await fetch(`/api/events?camera_id=${activeCamera}&limit=25`);
    if (!resp.ok) throw new Error("Gagal mengambil riwayat event");
    const data = await resp.json();
    const events = data.events || [];

    if (events.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-muted">Belum ada insiden tercatat pada sistem.</td></tr>`;
      return;
    }

    tbody.innerHTML = events.map((ev) => {
      const dt = ev.trigger_time ? new Date(ev.trigger_time).toLocaleString("id-ID") : "--";
      const dwell = ev.dwell_duration ? `${(ev.dwell_duration / 60).toFixed(1)} menit (${ev.dwell_duration.toFixed(0)}s)` : "--";
      const statusText = ev.is_resolved ? `<span class="tag-badge authorized">Selesai</span>` : `<span class="tag-badge unknown">Aktif</span>`;
      const owner = ev.owner_name && ev.owner_name.toLowerCase() !== "unknown"
        ? `<strong>${ev.owner_name}</strong>`
        : `<span class="text-subtle">Belum Teridentifikasi</span>`;

      const clipBtn = ev.clip_path
        ? `<button class="btn-play-clip" onclick="openVideoPlayer('${ev.clip_path}', '${ev.camera_id}', '${dt}', '${dwell}')">
             <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Putar Video
           </button>`
        : `<span class="btn-no-clip">Tidak Ada Klip</span>`;

      return `
        <tr>
          <td>#${ev.id}</td>
          <td>${dt}</td>
          <td><code>${ev.camera_id}</code></td>
          <td>${ev.zone_id}</td>
          <td>${dwell}</td>
          <td>${statusText}</td>
          <td>${owner}</td>
          <td>${clipBtn}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.warn("Gagal memuat log insiden:", err);
  }
}

/**
 * Open HTML5 video modal player for incident clip.
 */
function openVideoPlayer(clipPath, camId, timestamp, dwell) {
  const modal = document.getElementById("videoModal");
  const player = document.getElementById("incidentVideoPlayer");
  const title = document.getElementById("modalClipTitle");
  const meta = document.getElementById("modalClipMeta");
  const dlBtn = document.getElementById("modalDownloadBtn");
  if (!modal || !player) return;

  // Extract clean filename from clipPath
  const filename = clipPath.split(/[\\/]/).pop();
  const videoUrl = `/api/clips/${filename}`;

  if (title) title.textContent = `Rekaman Insiden [${camId}] - ${timestamp}`;
  if (meta) meta.textContent = `Kamera: ${camId} | Dwell: ${dwell} | H.264 Ring Buffer`;
  if (dlBtn) {
    dlBtn.href = videoUrl;
    dlBtn.setAttribute("download", filename);
  }

  player.src = videoUrl;
  modal.style.display = "flex";
  player.load();
  player.play().catch((err) => console.warn("Autoplay dicegah browser:", err));
}

function closeVideoModal() {
  const modal = document.getElementById("videoModal");
  const player = document.getElementById("incidentVideoPlayer");
  if (player) {
    player.pause();
    player.removeAttribute("src");
    player.load();
  }
  if (modal) {
    modal.style.display = "none";
  }
}

/* ==========================================================================
   8. LIFECYCLE INITIALIZATION
   ========================================================================== */

window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadCameraList();
  pollTelemetry();
  loadRecentEvents();
  pollTimer = setInterval(pollTelemetry, 1000);
  setInterval(loadCameraList, 8000);
  setInterval(loadRecentEvents, 10000);
});

