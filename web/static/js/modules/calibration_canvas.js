/**
 * Smart CCTV 2.0 - Interactive Calibration Canvas & Vector Rendering Engine
 */

let zoneEditorActive = false;
let activeEditorCategory = "polygons"; // "polygons" | "tripwires"
let zoneData = {};
let originalZoneData = {};
let lineData = {};
let originalLineData = {};
let zoneConfigs = {};
let baseResolution = [1920, 1080];
let historyStack = [];

let selectedZone = "zone_1";
let selectedLine = "";
let zoneMode = "drag"; // "drag" | "add"
let selectedVertexIndex = -1; // polygon: 0..N-1 | tripwire: 0 (p1/A), 1 (p2/B)
let draggedPointIndex = -1;
let hoverPointIndex = -1;
let ghostMidpoint = null;
let isMouseDown = false;
let hasMovedDuringDrag = false;

// Enterprise Color Palettes
const ZONE_COLOR_SCHEMES = {
  zone_2_transit: {
    name: "Zona 2: Transit Steril",
    stroke: "#10b981",
    fill: "rgba(16, 185, 129, 0.15)",
    handle: "#34d399",
    glow: "rgba(16, 185, 129, 0.50)",
    dot: "#10b981",
  },
  zone_1_koridor: {
    name: "Zona 1: Koridor Akses",
    stroke: "#f59e0b",
    fill: "rgba(245, 158, 11, 0.10)",
    handle: "#fbbf24",
    glow: "rgba(245, 158, 11, 0.50)",
    dot: "#f59e0b",
  },
  invalid: {
    stroke: "#ef4444",
    fill: "rgba(239, 68, 68, 0.25)",
    handle: "#f87171",
    glow: "rgba(239, 68, 68, 0.65)",
    dot: "#ef4444",
  },
  tripwire: {
    name: "Tripwire Line",
    stroke: "#06b6d4",
    fill: "rgba(6, 182, 212, 0.20)",
    handle: "#38bdf8",
    glow: "rgba(6, 182, 212, 0.60)",
    dot: "#06b6d4",
  },
  default: {
    name: "Area Khusus",
    stroke: "#3b82f6",
    fill: "rgba(59, 130, 246, 0.15)",
    handle: "#60a5fa",
    glow: "rgba(59, 130, 246, 0.50)",
    dot: "#3b82f6",
  },
};

function getActiveZoneColors(zKey, isInvalid = false) {
  if (isInvalid) return ZONE_COLOR_SCHEMES.invalid;
  return ZONE_COLOR_SCHEMES[zKey] || ZONE_COLOR_SCHEMES.default;
}

/**
 * Dynamically align and anchor zoneEditorCanvas directly over visible video pixels,
 * eliminating letterbox/pillarbox offsets (e.g. 4:3 streams in 16:9 containers or fullscreen).
 */
function syncCanvasWithVideoFeed() {
  const canvas = document.getElementById("zoneEditorCanvas");
  const img = document.getElementById("streamFeed");
  const container = document.getElementById("videoContainer");
  if (!canvas || !container) return;

  const baseW = baseResolution[0] || 1920;
  const baseH = baseResolution[1] || 1080;

  // Set internal coordinate buffer matching camera's native base resolution
  canvas.width = baseW;
  canvas.height = baseH;

  // Compute rendered video area inside container (taking object-fit: contain into account)
  const containerRect = container.getBoundingClientRect();
  if (containerRect.width === 0 || containerRect.height === 0) return;

  const containerRatio = containerRect.width / containerRect.height;
  const imgRatio = (img && img.naturalWidth && img.naturalHeight)
    ? (img.naturalWidth / img.naturalHeight)
    : (baseW / baseH);

  let renderW, renderH, renderLeft, renderTop;

  if (containerRatio > imgRatio) {
    // Pillarbox: video is constrained by container height, black bars on left and right
    renderH = containerRect.height;
    renderW = renderH * imgRatio;
    renderLeft = (containerRect.width - renderW) / 2;
    renderTop = 0;
  } else {
    // Letterbox: video is constrained by container width, black bars on top and bottom
    renderW = containerRect.width;
    renderH = renderW / imgRatio;
    renderLeft = 0;
    renderTop = (containerRect.height - renderH) / 2;
  }

  // Anchor the canvas directly over visible video pixels with zero pixel offset
  canvas.style.position = "absolute";
  canvas.style.left = `${Math.round(renderLeft)}px`;
  canvas.style.top = `${Math.round(renderTop)}px`;
  canvas.style.width = `${Math.round(renderW)}px`;
  canvas.style.height = `${Math.round(renderH)}px`;
}

function getNativeCoords(e) {
  const canvas = document.getElementById("zoneEditorCanvas");
  if (!canvas) return { x: 0, y: 0 };
  const rect = canvas.getBoundingClientRect();
  const baseW = baseResolution[0] || canvas.width || 1920;
  const baseH = baseResolution[1] || canvas.height || 1080;
  const scaleX = baseW / (rect.width || 1);
  const scaleY = baseH / (rect.height || 1);
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);
  return {
    x: Math.max(0, Math.min(baseW, x)),
    y: Math.max(0, Math.min(baseH, y)),
  };
}

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
    syncCanvasWithVideoFeed();
    setupCanvasEvents();
    renderZoneCanvas();
    showToast(
      "Mode Kalibrasi Aktif",
      "Geser titik poligon dengan mouse untuk mengatur area. Klik (?) di toolbar untuk panduan lengkap.",
      "success",
      4000
    );
  } else {
    canvas.classList.remove("active");
    toolbar.style.display = "none";
    if (btn) btn.classList.remove("active");
    if (typeof togglePropertiesDrawer === "function") {
      togglePropertiesDrawer(false);
    }
    selectedVertexIndex = -1;
    draggedPointIndex = -1;
    hoverPointIndex = -1;
    ghostMidpoint = null;
  }
}

async function fetchCameraZones() {
  try {
    const resp = await fetch(`/api/zones/${activeCamera}`);
    if (!resp.ok) throw new Error("Gagal mengambil data zona kamera.");
    const data = await resp.json();

    zoneData = JSON.parse(JSON.stringify(data.zones || {}));
    originalZoneData = JSON.parse(JSON.stringify(zoneData));

    lineData = JSON.parse(JSON.stringify(data.lines || {}));
    originalLineData = JSON.parse(JSON.stringify(lineData));

    zoneConfigs = JSON.parse(JSON.stringify(data.zone_configs || {}));
    baseResolution = data.base_resolution || [1920, 1080];
    historyStack = [];

    // Auto-select valid element key from data (e.g. zone_1_koridor)
    const polyKeys = Object.keys(zoneData);
    if (polyKeys.length > 0 && (!selectedZone || !zoneData[selectedZone])) {
      selectedZone = polyKeys[0];
    }
    const lineKeys = Object.keys(lineData);
    if (lineKeys.length > 0 && (!selectedLine || !lineData[selectedLine])) {
      selectedLine = lineKeys[0];
    }

    syncCanvasWithVideoFeed();

    const badgePoly = document.getElementById("badgePolygonCount");
    if (badgePoly) badgePoly.textContent = Object.keys(zoneData).length;
    const badgeLine = document.getElementById("badgeTripwireCount");
    if (badgeLine) badgeLine.textContent = Object.keys(lineData).length;

    if (typeof populateElementSelector === "function") {
      populateElementSelector();
    }
  } catch (err) {
    console.error("Error fetching zones:", err);
    showToast("Gagal Memuat Zona", "Tidak dapat terhubung ke server kamera.", "error");
  }
}

function switchEditorCategory(category) {
  activeEditorCategory = category;
  const tabPoly = document.getElementById("tabPolygons");
  const tabLine = document.getElementById("tabTripwires");

  if (category === "polygons") {
    tabPoly?.classList.add("active");
    tabLine?.classList.remove("active");
  } else {
    tabLine?.classList.add("active");
    tabPoly?.classList.remove("active");
  }

  selectedVertexIndex = -1;
  draggedPointIndex = -1;
  hoverPointIndex = -1;
  ghostMidpoint = null;

  if (typeof populateElementSelector === "function") {
    populateElementSelector();
  }
  renderZoneCanvas();
}

function setZoneMode(mode) {
  zoneMode = mode;
  document.getElementById("btnModeDrag")?.classList.toggle("active", mode === "drag");
  document.getElementById("btnModeAdd")?.classList.toggle("active", mode === "add");
  renderZoneCanvas();
}

/* ==========================================================================
   GEOMETRY HELPER FUNCTIONS
   ========================================================================== */

function findVertexNear(nativeX, nativeY, pts, thresholdPx = 32) {
  for (let i = 0; i < pts.length; i++) {
    const d = Math.hypot(pts[i][0] - nativeX, pts[i][1] - nativeY);
    if (d <= thresholdPx) {
      return i;
    }
  }
  return -1;
}

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
    t: t,
  };
}

function ccw(A, B, C) {
  return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0]);
}

function segmentsIntersect(A, B, C, D) {
  if (
    (A[0] === C[0] && A[1] === C[1]) ||
    (A[0] === D[0] && A[1] === D[1]) ||
    (B[0] === C[0] && B[1] === C[1]) ||
    (B[0] === D[0] && B[1] === D[1])
  ) {
    return false;
  }
  return ccw(A, C, D) !== ccw(B, C, D) && ccw(A, B, C) !== ccw(A, B, D);
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

function drawArrowHead(ctx, fromX, fromY, toX, toY, headLength = 16) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

/* ==========================================================================
   CANVAS EVENT LISTENERS
   ========================================================================== */

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

    if (activeEditorCategory === "polygons") {
      const pts = zoneData[selectedZone] || [];
      if (isMouseDown && draggedPointIndex >= 0 && draggedPointIndex < pts.length) {
        hasMovedDuringDrag = true;
        pts[draggedPointIndex] = [x, y];
        ghostMidpoint = null;
        renderZoneCanvas();
      } else {
        const prevHover = hoverPointIndex;
        hoverPointIndex = findVertexNear(x, y, pts, 32);

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
    } else {
      // Tripwires Category
      const ldata = lineData[selectedLine];
      if (!ldata) return;
      const pts = [ldata.p1, ldata.p2];

      if (isMouseDown && draggedPointIndex >= 0) {
        hasMovedDuringDrag = true;
        if (draggedPointIndex === 0) ldata.p1 = [x, y];
        else if (draggedPointIndex === 1) ldata.p2 = [x, y];
        renderZoneCanvas();
      } else {
        hoverPointIndex = findVertexNear(x, y, pts, 32);
        renderZoneCanvas();
      }
    }
  });

  canvas.addEventListener("mousedown", (e) => {
    if (!zoneEditorActive) return;
    if (e.button !== 0) return; // Only process Left Click for dragging/selection
    isMouseDown = true;
    hasMovedDuringDrag = false;
    const { x, y } = getNativeCoords(e);

    if (activeEditorCategory === "polygons") {
      const keys = Object.keys(zoneData || {});
      if (keys.length === 0 || !selectedZone || !zoneData[selectedZone]) {
        if (typeof pushHistory === "function") pushHistory();
        const newKey = "zone_1";
        zoneData[newKey] = [[x, y]];
        zoneConfigs[newKey] = {
          name: "Zona Pemantauan 1",
          dwell_threshold_sec: 60,
          detect_unattended: false,
        };
        selectedZone = newKey;
        selectedVertexIndex = 0;
        window.selectedVertexIndex = 0;
        draggedPointIndex = 0;
        if (typeof populateElementSelector === "function") populateElementSelector();
        renderZoneCanvas();
        showToast("Zona Baru Dimulai", "Titik P1 dibuat. Klik area lain untuk menambahkan titik sudut berikutnya.", "info", 3500);
        return;
      }

      const pts = zoneData[selectedZone] || [];
      const hitVertex = findVertexNear(x, y, pts, 36);

      if (hitVertex >= 0) {
        selectedVertexIndex = hitVertex;
        window.selectedVertexIndex = hitVertex;
        draggedPointIndex = hitVertex;
        if (typeof pushHistory === "function") pushHistory();
        renderZoneCanvas();
      } else if (ghostMidpoint !== null) {
        if (typeof pushHistory === "function") pushHistory();
        pts.splice(ghostMidpoint.edgeIdx + 1, 0, [ghostMidpoint.x, ghostMidpoint.y]);
        selectedVertexIndex = ghostMidpoint.edgeIdx + 1;
        window.selectedVertexIndex = selectedVertexIndex;
        draggedPointIndex = selectedVertexIndex;
        ghostMidpoint = null;
        renderZoneCanvas();
        showToast("Titik Baru Disisipkan", "Titik poligon baru ditambahkan. Geser untuk memposisikan.", "success", 2000);
      } else if (zoneMode === "add" || pts.length < 3) {
        if (typeof pushHistory === "function") pushHistory();
        pts.push([x, y]);
        selectedVertexIndex = pts.length - 1;
        window.selectedVertexIndex = selectedVertexIndex;
        draggedPointIndex = selectedVertexIndex;
        renderZoneCanvas();
      } else {
        selectedVertexIndex = -1;
        window.selectedVertexIndex = -1;
        renderZoneCanvas();
      }
    } else {
      // Tripwires Category
      const ldata = lineData[selectedLine];
      if (!ldata) return;
      const pts = [ldata.p1, ldata.p2];
      const hit = findVertexNear(x, y, pts, 36);
      if (hit >= 0) {
        selectedVertexIndex = hit;
        window.selectedVertexIndex = hit;
        draggedPointIndex = hit;
        if (typeof pushHistory === "function") pushHistory();
        renderZoneCanvas();
      } else {
        for (const [lk, ld] of Object.entries(lineData)) {
          const proj = projectOnSegment(x, y, ld.p1[0], ld.p1[1], ld.p2[0], ld.p2[1]);
          if (proj.d <= 20) {
            selectedLine = lk;
            const selector = document.getElementById("zoneSelector");
            if (selector) selector.value = lk;
            if (typeof syncPropertiesDrawerValues === "function") syncPropertiesDrawerValues();
            renderZoneCanvas();
            break;
          }
        }
      }
    }
  });

  // RIGHT-CLICK (CONTEXT MENU): INSTANT VERTEX DELETION
  canvas.addEventListener("contextmenu", (e) => {
    if (!zoneEditorActive) return;
    e.preventDefault();
    const { x, y } = getNativeCoords(e);

    if (activeEditorCategory === "polygons") {
      const pts = zoneData[selectedZone] || [];
      const hitVertex = findVertexNear(x, y, pts, 36);

      if (hitVertex >= 0) {
        if (pts.length <= 3) {
          if (typeof promptDeleteZoneConfirm === "function") {
            promptDeleteZoneConfirm();
          } else {
            showToast("Batas Minimum Titik", "Zona poligon harus memiliki minimal 3 titik koordinat!", "warning", 3000);
          }
          return;
        }
        if (typeof pushHistory === "function") pushHistory();
        pts.splice(hitVertex, 1);
        selectedVertexIndex = -1;
        window.selectedVertexIndex = -1;
        hoverPointIndex = -1;
        renderZoneCanvas();
        showToast("Titik Dihapus", "Titik koordinat berhasil dihapus (Klik Kanan). Tekan Ctrl+Z jika ingin membatalkan.", "success", 2500);
      }
    }
  });

  window.addEventListener("mouseup", () => {
    if (!zoneEditorActive) return;
    isMouseDown = false;
    draggedPointIndex = -1;
    renderZoneCanvas();
  });

  window.addEventListener("keydown", (e) => {
    if (!zoneEditorActive) return;

    if (e.key === "Escape") {
      toggleZoneEditor(false);
    } else if (e.key === "Delete" || e.key === "Backspace") {
      if (document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        return;
      }
      if (selectedVertexIndex >= 0 && activeEditorCategory === "polygons") {
        e.preventDefault();
        if (typeof deleteSelectedPoint === "function") deleteSelectedPoint();
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (typeof undoZoneAction === "function") undoZoneAction();
    }
  });

  canvasEventsInitialized = true;
}

/* ==========================================================================
   CANVAS RENDERING ENGINE (POLYGONS & DIRECTIONAL TRIPWIRES)
   ========================================================================== */

function renderZoneCanvas() {
  const canvas = document.getElementById("zoneEditorCanvas");
  if (!canvas || !zoneEditorActive) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const baseW = canvas.width;
  const baseH = canvas.height;

  // 1. RENDER INACTIVE POLYGONS
  Object.keys(zoneData).forEach((zKey) => {
    if (activeEditorCategory === "polygons" && zKey === selectedZone) return;
    const pts = zoneData[zKey];
    if (!pts || pts.length < 3) return;

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

  // 2. RENDER INACTIVE TRIPWIRES
  Object.keys(lineData).forEach((lKey) => {
    if (activeEditorCategory === "tripwires" && lKey === selectedLine) return;
    const ld = lineData[lKey];
    if (!ld || !ld.p1 || !ld.p2) return;

    ctx.beginPath();
    ctx.moveTo(ld.p1[0], ld.p1[1]);
    ctx.lineTo(ld.p2[0], ld.p2[1]);
    ctx.strokeStyle = "rgba(6, 182, 212, 0.35)";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // 3. RENDER ACTIVE CATEGORY
  if (activeEditorCategory === "polygons") {
    const activePts = zoneData[selectedZone];
    if (!activePts || activePts.length === 0) {
      const valPill = document.getElementById("zoneValidationPill");
      const saveBtn = document.getElementById("btnSaveZones");
      if (valPill) {
        valPill.textContent = "0 Zona (Kosong)";
        valPill.className = "glass-validation-pill valid";
      }
      if (saveBtn) saveBtn.disabled = false;
      return;
    }

    const selfIntersect = isPolygonSelfIntersecting(activePts);
    const colors = getActiveZoneColors(selectedZone, selfIntersect);

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
    }

    // Polygon Body
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

    // Smart Midpoint (+) Ghost Handle
    if (ghostMidpoint !== null && !isMouseDown) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(ghostMidpoint.x, ghostMidpoint.y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = colors.stroke;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(ghostMidpoint.x - 4, ghostMidpoint.y);
      ctx.lineTo(ghostMidpoint.x + 4, ghostMidpoint.y);
      ctx.moveTo(ghostMidpoint.x, ghostMidpoint.y - 4);
      ctx.lineTo(ghostMidpoint.x, ghostMidpoint.y + 4);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    }

    // Vertex Handles
    activePts.forEach((pt, idx) => {
      const isHover = idx === hoverPointIndex;
      const isSelected = idx === selectedVertexIndex || idx === window.selectedVertexIndex;
      const isDragged = idx === draggedPointIndex;
      const radius = isSelected ? 11 : isDragged ? 10 : isHover ? 8 : 6;

      if (isSelected || isDragged || isHover) {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], radius + 6, 0, 2 * Math.PI);
        ctx.fillStyle = isSelected ? "rgba(239, 68, 68, 0.45)" : colors.glow;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(pt[0], pt[1], radius, 0, 2 * Math.PI);
      ctx.fillStyle = isSelected ? "#ef4444" : isDragged ? "#ffffff" : isHover ? colors.handle : colors.stroke;
      ctx.fill();
      ctx.strokeStyle = isSelected ? "#ffffff" : "#000000";
      ctx.lineWidth = isSelected ? 3 : 2.5;
      ctx.stroke();

      // Show Point Index Label on Hover or Select
      if (isSelected || isHover) {
        ctx.save();
        ctx.font = "bold 12px -apple-system, sans-serif";
        const labelText = `P${idx + 1}${isSelected ? " (Terpilih)" : ""}`;
        const tw = ctx.measureText(labelText).width;
        ctx.fillStyle = isSelected ? "rgba(239, 68, 68, 0.9)" : "rgba(15, 23, 42, 0.85)";
        ctx.beginPath();
        ctx.roundRect(pt[0] - tw / 2 - 6, pt[1] - radius - 24, tw + 12, 18, 4);
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.fillText(labelText, pt[0] - tw / 2, pt[1] - radius - 11);
        ctx.restore();
      }
    });

    if (selfIntersect) {
      ctx.save();
      ctx.font = "bold 16px -apple-system, sans-serif";
      const warnMsg = "⚠ Poligon Bersilangan! Garis tidak boleh memotong satu sama lain.";
      const wText = ctx.measureText(warnMsg).width;
      const bx = (baseW - wText) / 2 - 16;
      const by = baseH - 70;
      ctx.fillStyle = "rgba(239, 68, 68, 0.92)";
      ctx.beginPath();
      ctx.roundRect(bx, by, wText + 32, 38, 8);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(warnMsg, bx + 16, by + 24);
      ctx.restore();
    }
  } else {
    // ACTIVE TRIPWIRE RENDERING
    const ldata = lineData[selectedLine];
    const valPill = document.getElementById("zoneValidationPill");
    if (valPill) {
      valPill.textContent = "Tripwire Aktif";
      valPill.className = "glass-validation-pill valid";
    }

    if (ldata && ldata.p1 && ldata.p2) {
      const p1 = ldata.p1;
      const p2 = ldata.p2;
      const dir = ldata.direction || "both";

      // Glow backing
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
      ctx.lineWidth = 8;
      ctx.stroke();

      // Main line
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 4;
      ctx.stroke();

      // Draw directional arrows
      ctx.fillStyle = "#38bdf8";
      if (dir === "a_to_b" || dir === "both") {
        drawArrowHead(ctx, p1[0], p1[1], p2[0], p2[1], 18);
      }
      if (dir === "b_to_a" || dir === "both") {
        drawArrowHead(ctx, p2[0], p2[1], p1[0], p1[1], 18);
      }

      // Midpoint badge
      const midX = (p1[0] + p2[0]) / 2;
      const midY = (p1[1] + p2[1]) / 2;
      const dirText = dir === "both" ? "A ⇄ B" : dir === "a_to_b" ? "A ➔ B" : "B ➔ A";
      const labelText = `${ldata.name || selectedLine} (${dirText})`;

      ctx.save();
      ctx.font = "bold 13px ui-monospace, monospace";
      const textW = ctx.measureText(labelText).width;
      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(midX - textW / 2 - 8, midY - 14, textW + 16, 26, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, midX - textW / 2, midY + 4);
      ctx.restore();

      // Endpoints Handles (A = P1, B = P2)
      [
        { pt: p1, label: "A", idx: 0 },
        { pt: p2, label: "B", idx: 1 },
      ].forEach(({ pt, label, idx }) => {
        const isHover = idx === hoverPointIndex;
        const isSelected = idx === selectedVertexIndex;
        const isDragged = idx === draggedPointIndex;
        const radius = isSelected || isDragged ? 12 : isHover ? 10 : 7;

        ctx.beginPath();
        ctx.arc(pt[0], pt[1], radius + 5, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(6, 182, 212, 0.4)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(pt[0], pt[1], radius, 0, 2 * Math.PI);
        ctx.fillStyle = isSelected || isDragged ? "#ffffff" : isHover ? "#38bdf8" : "#06b6d4";
        ctx.fill();
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 2.5;
        ctx.stroke();

        ctx.font = "bold 11px sans-serif";
        ctx.fillStyle = isSelected || isDragged ? "#000000" : "#ffffff";
        ctx.fillText(label, pt[0] - 4, pt[1] + 4);
      });
    }
  }
}

// Window resize & Fullscreen responsive canvas resync
window.addEventListener("resize", () => {
  if (zoneEditorActive) {
    syncCanvasWithVideoFeed();
    renderZoneCanvas();
  }
});

document.addEventListener("fullscreenchange", () => {
  setTimeout(() => {
    if (zoneEditorActive) {
      syncCanvasWithVideoFeed();
      renderZoneCanvas();
    }
  }, 100);
});

// Export to window for global access across modules
window.syncCanvasWithVideoFeed = syncCanvasWithVideoFeed;
