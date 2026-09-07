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
          const conf = face.confidence ? `(${(face.confidence * 100).toFixed(0)}%)` : "";
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
   5. FULLSCREEN HANDLER
   ========================================================================== */

function toggleFullscreen() {
  const container = document.getElementById("videoContainer");
  if (!container) return;

  if (!document.fullscreenElement) {
    container.requestFullscreen().catch((err) => {
      alert(`Error mengaktifkan layar penuh: ${err.message}`);
    });
  } else {
    document.exitFullscreen();
  }
}

/* ==========================================================================
   6. LIFECYCLE INITIALIZATION
   ========================================================================== */

window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadCameraList();
  pollTelemetry();
  pollTimer = setInterval(pollTelemetry, 1000);
  setInterval(loadCameraList, 8000);
});
