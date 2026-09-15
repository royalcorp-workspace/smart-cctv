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
    }
  } catch (err) {
    console.warn("Gagal memuat daftar kamera:", err);
  }
}

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

  const metaCam = document.getElementById("metaCamName");
  if (metaCam && select.selectedOptions && select.selectedOptions[0]) {
    const optText = select.selectedOptions[0].textContent || "";
    const match = optText.match(/-\s*([^(]+)/) || optText.match(/\(([^)]+)\)/);
    if (match && match[1]) {
      metaCam.textContent = match[1].trim();
    }
  }

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
    if (typeof fetchCameraZones === "function") {
      fetchCameraZones().then(() => {
        if (typeof renderZoneCanvas === "function") {
          renderZoneCanvas();
        }
      });
    }
  }

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
  } catch (err) {
    // Silent fail on transient poll errors
  } finally {
    _isTelemetryPolling = false;
  }
}
