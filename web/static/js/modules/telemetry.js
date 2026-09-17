/**
 * Smart CCTV 2.0 - Telemetry Polling, Camera Switcher & KPI Updates
 */

let activeCamera = "cam_01";
let _isTelemetryPolling = false;

async function loadCameraList() {
  try {
    const resp = await fetch("/api/cameras");
    if (!resp.ok) return;
    const cameras = await resp.json();
    const select = document.getElementById("cameraSelect");
    if (select && cameras && cameras.length > 0) {
      const currentVal = select.value || activeCamera;

      select.innerHTML = "";
      cameras.forEach((cam) => {
        const opt = document.createElement("option");
        opt.value = cam.id;
        const statusIcon = cam.online ? "🟢" : "🔴";
        const fpsText = cam.online && cam.fps ? ` (${cam.fps.toFixed(1)} FPS)` : "";
        opt.textContent = `${statusIcon} ${cam.id} - ${cam.name || "Kamera"}${fpsText}`;
        if (cam.id === currentVal) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });
      if (select.value !== currentVal && cameras.some((c) => c.id === currentVal)) {
        select.value = currentVal;
      } else if (!cameras.some((c) => c.id === activeCamera)) {
        activeCamera = cameras[0].id;
        select.value = activeCamera;
      }
      // Render Quick Camera Switcher strip in sidebar
      renderQuickCameraSwitcher(cameras);
    }
  } catch (err) {
    console.warn("Gagal memuat daftar kamera:", err);
  }
}

function renderQuickCameraSwitcher(cameras) {
  const container = document.getElementById("quickCamSwitcher");
  const countBadge = document.getElementById("quickCamCount");
  if (!container || !cameras) return;

  if (countBadge) {
    countBadge.textContent = `${cameras.length} Kamera`;
  }

  container.innerHTML = cameras.map((cam) => {
    const isActive = cam.id === activeCamera;
    const isOnline = cam.online !== false;
    return `
      <button type="button" class="quick-cam-btn ${isActive ? "active" : ""} ${isOnline ? "" : "inactive"}" 
              onclick="selectQuickCamera('${cam.id}')" title="${cam.name || cam.id}">
        <div class="quick-cam-btn-left">
          <span class="quick-cam-dot"></span>
          <span class="quick-cam-id">${cam.id}</span>
        </div>
        <span class="quick-cam-name">${cam.name || cam.id}</span>
      </button>
    `;
  }).join("");
}

function selectQuickCamera(camId) {
  if (!camId) return;
  const select = document.getElementById("cameraSelect");
  if (select) {
    select.value = camId;
  }
  onCameraChange();
}

function onCameraChange() {
  const select = document.getElementById("cameraSelect");
  if (!select) return;
  const newCam = select.value;
  if (newCam === activeCamera) return;

  activeCamera = newCam;

  // Sync Video Header Badges
  const badgeCam = document.getElementById("badgeCamId");
  if (badgeCam) {
    badgeCam.textContent = activeCamera;
  }

  let camNameText = "Kamera";
  if (select.selectedOptions && select.selectedOptions[0]) {
    const optText = select.selectedOptions[0].textContent || "";
    const match = optText.match(/-\s*([^(]+)/) || optText.match(/\(([^)]+)\)/);
    if (match && match[1]) {
      camNameText = match[1].trim();
    }
  }
  const badgeCamName = document.getElementById("badgeCamName");
  if (badgeCamName) {
    badgeCamName.textContent = camNameText;
  }

  const metaCam = document.getElementById("metaCamName");
  if (metaCam) {
    metaCam.textContent = camNameText;
  }

  // Update quick switcher active state
  document.querySelectorAll(".quick-cam-btn").forEach((btn) => {
    const idEl = btn.querySelector(".quick-cam-id");
    if (idEl && idEl.textContent.trim() === activeCamera) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Terminate old stream
  const feedImg = document.getElementById("streamFeed");
  if (feedImg) {
    feedImg.src = "";
  }

  // Reset telemetry indicators
  const fpsEl = document.getElementById("statFps");
  if (fpsEl) fpsEl.textContent = "--";
  const tracksEl = document.getElementById("statTracks");
  if (tracksEl) tracksEl.textContent = "0";
  const violEl = document.getElementById("statViolations");
  if (violEl) violEl.textContent = "0";

  // Re-bind stream with query cachebuster
  if (window._camSwitchTimer) {
    clearTimeout(window._camSwitchTimer);
  }
  window._camSwitchTimer = setTimeout(() => {
    if (feedImg) {
      feedImg.src = `/api/stream/${activeCamera}?t=${Date.now()}`;
    }
  }, 50);

  // If Zone Editor canvas is active, reload zones for new camera
  if (typeof zoneEditorActive !== "undefined" && zoneEditorActive) {
    if (typeof setCameraCalibrationMode === "function") {
      setCameraCalibrationMode(activeCamera, true);
    }
    if (typeof fetchCameraZones === "function") {
      fetchCameraZones().then(() => {
        if (typeof syncCanvasWithVideoFeed === "function") {
          syncCanvasWithVideoFeed();
        }
        if (typeof renderZoneCanvas === "function") {
          renderZoneCanvas();
        }
      });
    }
  }

  // Refresh active zones summary & activity ticker
  updateActiveZonesSummary(activeCamera);
  updateQuickEventTicker(activeCamera);

  setTimeout(() => {
    pollTelemetry();
    if (typeof loadRecentEvents === "function") {
      loadRecentEvents();
    }
  }, 100);
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
  if (typeof syncCanvasWithVideoFeed === "function" && typeof zoneEditorActive !== "undefined" && zoneEditorActive) {
    syncCanvasWithVideoFeed();
    if (typeof renderZoneCanvas === "function") renderZoneCanvas();
  }
}

async function pollTelemetry() {
  if (_isTelemetryPolling) return;
  _isTelemetryPolling = true;

  try {
    let resp;
    if (window.AbortSignal && AbortSignal.timeout) {
      resp = await fetch(`/api/status/${activeCamera}`, { signal: AbortSignal.timeout(2000) });
    } else {
      resp = await fetch(`/api/status/${activeCamera}`);
    }
    if (!resp || !resp.ok) return;
    const data = await resp.json();

    const metaCam = document.getElementById("metaCamName");
    if (metaCam && data.name) {
      metaCam.textContent = data.name;
    }

    const statFps = document.getElementById("statFps");
    if (statFps) {
      statFps.textContent = typeof data.fps === "number" ? data.fps.toFixed(1) : "--";
    }
    const statTracks = document.getElementById("statTracks");
    if (statTracks) {
      statTracks.textContent = data.active_tracks !== undefined ? data.active_tracks : "0";
    }

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

    const statViolations = document.getElementById("statViolations");
    if (statViolations) {
      const vCount = data.violations !== undefined ? data.violations : 0;
      statViolations.textContent = vCount;
      if (vCount > 0) {
        statViolations.className = "kpi-value alert-text";
      } else {
        statViolations.className = "kpi-value success-text";
      }
    }

    // Biometric Faces List
    const facesList = document.getElementById("facesList");
    if (facesList && Array.isArray(data.identified_faces)) {
      if (data.identified_faces.length === 0) {
        facesList.innerHTML = `<div class="empty-state">Tidak ada personil terdeteksi</div>`;
      } else {
        facesList.innerHTML = data.identified_faces
          .map((f) => {
            const isKnown = f.name && f.name.toLowerCase() !== "unknown";
            const initial = isKnown ? f.name.charAt(0).toUpperCase() : "?";
            const tagClass = isKnown ? "authorized" : "unknown";
            const tagText = isKnown ? "Terverifikasi" : "Belum Dikenal";
            return `
              <div class="person-card">
                <div class="person-profile">
                  <div class="person-avatar ${isKnown ? "" : "unknown"}">${initial}</div>
                  <div class="person-meta">
                    <span class="person-name">${f.name || "Unknown"}</span>
                    <span class="person-detail">Track #${f.track_id} &bull; ${(f.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <span class="tag-badge ${tagClass}">${tagText}</span>
              </div>
            `;
          })
          .join("");
      }
    }
    // Periodic ticker and zone refresh (every ~4 polls)
    if (!window._pollTickCounter) window._pollTickCounter = 0;
    window._pollTickCounter++;
    if (window._pollTickCounter % 3 === 0) {
      updateQuickEventTicker(activeCamera);
    }
  } catch (err) {
    // Silent fail on transient poll errors
  } finally {
    _isTelemetryPolling = false;
  }
}

/**
 * Fetch and render active zone / tripwire summary for active camera
 */
async function updateActiveZonesSummary(camId) {
  const container = document.getElementById("sidebarZoneList");
  if (!container) return;

  try {
    const resp = await fetch(`/api/zones/${camId}`);
    if (!resp.ok) {
      container.innerHTML = `
        <div class="zone-mini-item">
          <div class="zone-mini-left">
            <span class="zone-mini-dot"></span>
            <span class="zone-mini-name">Zona Standard K3</span>
          </div>
          <span class="zone-mini-status">Aktif</span>
        </div>
      `;
      return;
    }
    const data = await resp.json();
    const zones = data.zones || {};
    const lines = data.lines || {};
    const zoneKeys = Object.keys(zones);
    const lineKeys = Object.keys(lines);

    if (zoneKeys.length === 0 && lineKeys.length === 0) {
      container.innerHTML = `
        <div class="zone-mini-item">
          <div class="zone-mini-left">
            <span class="zone-mini-dot"></span>
            <span class="zone-mini-name">Full Frame Clear Area</span>
          </div>
          <span class="zone-mini-status">Aktif</span>
        </div>
      `;
      return;
    }

    let html = "";
    zoneKeys.slice(0, 3).forEach((zk) => {
      const zCfg = (data.zone_configs && data.zone_configs[zk]) || {};
      const zName = zCfg.name || zk.replace(/_/g, " ").toUpperCase();
      html += `
        <div class="zone-mini-item">
          <div class="zone-mini-left">
            <span class="zone-mini-dot"></span>
            <span class="zone-mini-name">${zName}</span>
          </div>
          <span class="zone-mini-status">Aman</span>
        </div>
      `;
    });
    lineKeys.slice(0, 2).forEach((lk) => {
      html += `
        <div class="zone-mini-item">
          <div class="zone-mini-left">
            <span class="zone-mini-dot" style="background:#3b82f6;"></span>
            <span class="zone-mini-name">Tripwire: ${lk}</span>
          </div>
          <span class="zone-mini-status" style="color:#3b82f6;">Siaga</span>
        </div>
      `;
    });
    container.innerHTML = html;
  } catch (e) {
    // Non-blocking
  }
}

/**
 * Fetch and render latest 2-3 incident events for active camera
 */
async function updateQuickEventTicker(camId) {
  const container = document.getElementById("quickEventTicker");
  if (!container) return;

  try {
    const resp = await fetch(`/api/events?cam=${camId}&limit=3`);
    if (!resp.ok) return;
    const data = await resp.json();
    const events = data.events || [];

    if (events.length === 0) {
      container.innerHTML = `<div class="ticker-empty">Belum ada insiden pelanggaran tercatat.</div>`;
      return;
    }

    container.innerHTML = events.slice(0, 3).map((ev) => {
      const timeStr = ev.timestamp ? ev.timestamp.split(" ")[1] || ev.timestamp : "--:--";
      const eventType = ev.event_type || "Pelanggaran";
      const icon = eventType.toLowerCase().includes("clear") ? "⚠️" : "⚡";
      return `
        <div class="ticker-item">
          <div class="ticker-item-left">
            <span class="ticker-icon">${icon}</span>
            <span class="ticker-desc">${ev.zone_id || eventType} (${ev.object_type || "objek"})</span>
          </div>
          <span class="ticker-time">${timeStr}</span>
        </div>
      `;
    }).join("");
  } catch (e) {
    // Non-blocking
  }
}

// Initial triggers when script loads
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    updateActiveZonesSummary(activeCamera);
    updateQuickEventTicker(activeCamera);
  }, 400);
});
