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
   6. INTERACTIVE WEB ZONE EDITOR (1080p MATRIX SCALE MAPPING)
   ========================================================================== */

let zoneEditorActive = false;
let zoneData = {};
let originalZoneData = {};
let selectedZone = "";
let zoneMode = "drag"; // "drag" | "add" | "delete"
let draggedPointIndex = -1;
let hoverPointIndex = -1;
let isMouseDown = false;

const ZONE_COLORS = {
  zone_1_koridor: { stroke: "#3b82f6", fill: "rgba(59, 130, 246, 0.22)", handle: "#60a5fa" },
  zone_2_transit: { stroke: "#ef4444", fill: "rgba(239, 68, 68, 0.22)", handle: "#f87171" },
  default: { stroke: "#10b981", fill: "rgba(16, 185, 129, 0.22)", handle: "#34d399" }
};

function getZoneColor(zId) {
  return ZONE_COLORS[zId] || ZONE_COLORS.default;
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
  } else {
    canvas.classList.remove("active");
    toolbar.style.display = "none";
    if (btn) btn.classList.remove("active");
    draggedPointIndex = -1;
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

    // Populate zone dropdown selector
    const selector = document.getElementById("zoneSelector");
    if (selector) {
      selector.innerHTML = "";
      const zoneKeys = Object.keys(zoneData);
      zoneKeys.forEach((zk) => {
        const opt = document.createElement("option");
        opt.value = zk;
        opt.textContent = zk;
        selector.appendChild(opt);
      });
      if (zoneKeys.length > 0) {
        selectedZone = zoneKeys[0];
        selector.value = selectedZone;
      }
      const countBadge = document.getElementById("toolbarZoneCount");
      if (countBadge) countBadge.textContent = `${zoneKeys.length} Zona`;
    }
  } catch (err) {
    console.error("Error fetching zones:", err);
    alert("Gagal memuat koordinat zona dari backend.");
  }
}

function onZoneSelectChange() {
  const selector = document.getElementById("zoneSelector");
  if (selector) {
    selectedZone = selector.value;
    draggedPointIndex = -1;
    hoverPointIndex = -1;
    renderZoneCanvas();
  }
}

function setZoneMode(mode) {
  zoneMode = mode;
  document.getElementById("btnModeDrag")?.classList.toggle("active", mode === "drag");
  document.getElementById("btnModeAdd")?.classList.toggle("active", mode === "add");
  document.getElementById("btnModeDelete")?.classList.toggle("active", mode === "delete");
  renderZoneCanvas();
}

function resetCurrentZone() {
  if (originalZoneData[selectedZone]) {
    zoneData[selectedZone] = JSON.parse(JSON.stringify(originalZoneData[selectedZone]));
    draggedPointIndex = -1;
    hoverPointIndex = -1;
    renderZoneCanvas();
  }
}

/**
 * Find point near mouse within threshold.
 */
function findVertexNear(nativeX, nativeY, pts, thresholdPx = 28) {
  for (let i = 0; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - nativeX, pts[i][1] - nativeY);
    if (d <= thresholdPx) {
      return i;
    }
  }
  return -1;
}

/**
 * Distance from point (px, py) to line segment (x1, y1)-(x2, y2).
 */
function distToSegment(px, py, x1, y1, x2, y2) {
  const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
  if (l2 === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
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
 * Setup canvas mouse event listeners for dragging, adding, and deleting vertices.
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
      coordDisplay.textContent = `X: ${x}, Y: ${y} (1080p)`;
    }

    const pts = zoneData[selectedZone] || [];

    if (isMouseDown && zoneMode === "drag" && draggedPointIndex >= 0 && draggedPointIndex < pts.length) {
      pts[draggedPointIndex] = [x, y];
      renderZoneCanvas();
    } else {
      const prevHover = hoverPointIndex;
      hoverPointIndex = findVertexNear(x, y, pts, 26);
      if (prevHover !== hoverPointIndex) {
        renderZoneCanvas();
      }
    }
  });

  canvas.addEventListener("mousedown", (e) => {
    if (!zoneEditorActive) return;
    isMouseDown = true;
    const { x, y } = getNativeCoords(e);
    const pts = zoneData[selectedZone] || [];

    if (zoneMode === "drag") {
      draggedPointIndex = findVertexNear(x, y, pts, 26);
      renderZoneCanvas();
    } else if (zoneMode === "delete") {
      const targetIdx = findVertexNear(x, y, pts, 26);
      if (targetIdx >= 0) {
        if (pts.length <= 3) {
          alert("Zona harus memiliki minimal 3 titik koordinat!");
        } else {
          pts.splice(targetIdx, 1);
          hoverPointIndex = -1;
          renderZoneCanvas();
        }
      }
    } else if (zoneMode === "add") {
      // Find closest edge to insert the new vertex
      if (pts.length < 3) {
        pts.push([x, y]);
      } else {
        let bestEdgeIdx = 0;
        let minDist = Infinity;
        for (let i = 0; i < pts.length; i++) {
          const p1 = pts[i];
          const p2 = pts[(i + 1) % pts.length];
          const d = distToSegment(x, y, p1[0], p1[1], p2[0], p2[1]);
          if (d < minDist) {
            minDist = d;
            bestEdgeIdx = i;
          }
        }
        // Insert point between bestEdgeIdx and bestEdgeIdx + 1
        pts.splice(bestEdgeIdx + 1, 0, [x, y]);
      }
      renderZoneCanvas();
    }
  });

  window.addEventListener("mouseup", () => {
    if (!zoneEditorActive) return;
    isMouseDown = false;
    draggedPointIndex = -1;
    renderZoneCanvas();
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

  // 1. Draw non-selected zones in background
  Object.keys(zoneData).forEach((zKey) => {
    if (zKey === selectedZone) return;
    const pts = zoneData[zKey];
    if (!pts || pts.length < 3) return;
    const col = getZoneColor(zKey);

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(148, 163, 184, 0.12)";
    ctx.fill();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.6)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // 2. Draw currently selected active zone
  const activePts = zoneData[selectedZone];
  if (!activePts || activePts.length === 0) return;
  const col = getZoneColor(selectedZone);

  // Check self-intersection for warning highlight
  const selfIntersect = isPolygonSelfIntersecting(activePts);

  ctx.beginPath();
  ctx.moveTo(activePts[0][0], activePts[0][1]);
  for (let i = 1; i < activePts.length; i++) {
    ctx.lineTo(activePts[i][0], activePts[i][1]);
  }
  ctx.closePath();

  ctx.fillStyle = selfIntersect ? "rgba(239, 68, 68, 0.35)" : col.fill;
  ctx.fill();
  ctx.strokeStyle = selfIntersect ? "#ef4444" : col.stroke;
  ctx.lineWidth = 4;
  ctx.stroke();

  // 3. Draw vertices and handles
  activePts.forEach((pt, idx) => {
    const isHover = idx === hoverPointIndex;
    const isDragged = idx === draggedPointIndex;
    const radius = isDragged ? 14 : (isHover ? 12 : 9);

    ctx.beginPath();
    ctx.arc(pt[0], pt[1], radius, 0, 2 * Math.PI);
    ctx.fillStyle = isDragged ? "#fbbf24" : (isHover ? "#ffffff" : col.handle);
    ctx.fill();
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 3;
    ctx.stroke();

    // Draw vertex index badge
    ctx.font = "bold 13px ui-monospace, sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.fillText(`${idx + 1}`, pt[0] + 12, pt[1] - 8);
  });

  // 4. If self-intersecting, draw warning tag
  if (selfIntersect) {
    ctx.font = "bold 20px sans-serif";
    ctx.fillStyle = "#ef4444";
    ctx.fillText("⚠ Poligon bersilangan! Harap luruskan titik.", activePts[0][0] + 15, activePts[0][1] + 35);
  }
}

/**
 * Save updated zones to FastAPI backend with hot-reload.
 */
async function saveZonesToServer() {
  // Validate all zones before submitting
  for (const [zKey, pts] of Object.entries(zoneData)) {
    if (pts.length < 3) {
      alert(`Zona '${zKey}' memiliki titik kurang dari 3! Tambahkan titik poligon.`);
      return;
    }
    if (isPolygonSelfIntersecting(pts)) {
      alert(`Zona '${zKey}' memiliki garis yang saling bersilangan (self-intersecting)! Mohon rapikan titik terlebih dahulu.`);
      return;
    }
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
    alert(`✅ Berhasil! ${res.message || "Zona berhasil disimpan dan dimuat ulang secara instan."}`);
    toggleZoneEditor(false);
  } catch (err) {
    console.error("Save zones error:", err);
    alert(`❌ Gagal menyimpan zona: ${err.message}`);
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

