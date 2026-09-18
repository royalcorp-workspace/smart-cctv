function populateElementSelector() {
  const selector = document.getElementById("zoneSelector");
  const dot = document.getElementById("zoneColorDot");
  if (!selector) return;

  selector.innerHTML = "";

  if (activeEditorCategory === "polygons") {
    const keys = Object.keys(zoneData || {});
    if (keys.length === 0) {
      selector.innerHTML = `<option value="">(Belum Ada Zona)</option>`;
      if (dot) dot.style.backgroundColor = "#94a3b8";
      return;
    }

    if (!selectedZone || !zoneData[selectedZone]) {
      selectedZone = keys[0];
    }

    keys.forEach((k) => {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = zoneConfigs[k]?.name || k;
      if (k === selectedZone) opt.selected = true;
      selector.appendChild(opt);
    });

    selector.value = selectedZone;
    if (dot && typeof getActiveZoneColors === "function") {
      dot.style.backgroundColor = getActiveZoneColors(selectedZone).stroke;
    }
  } else {
    const keys = Object.keys(lineData || {});
    if (keys.length === 0) {
      selector.innerHTML = `<option value="">(Belum Ada Tripwire)</option>`;
      if (dot) dot.style.backgroundColor = "#94a3b8";
      return;
    }

    if (!selectedLine || !lineData[selectedLine]) {
      selectedLine = keys[0];
    }

    keys.forEach((k) => {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = lineData[k]?.name || k;
      if (k === selectedLine) opt.selected = true;
      selector.appendChild(opt);
    });

    selector.value = selectedLine;
    if (dot) {
      dot.style.backgroundColor = "#06b6d4";
    }
  }

  if (typeof syncPropertiesDrawerValues === "function") {
    syncPropertiesDrawerValues();
  }
}

function onZoneSelectChange() {
  const selector = document.getElementById("zoneSelector");
  const dot = document.getElementById("zoneColorDot");
  if (!selector) return;

  const val = selector.value;
  if (!val) return;

  if (activeEditorCategory === "polygons") {
    selectedZone = val;
    if (dot && typeof getActiveZoneColors === "function") {
      dot.style.backgroundColor = getActiveZoneColors(selectedZone).stroke;
    }
  } else {
    selectedLine = val;
    if (dot) {
      dot.style.backgroundColor = "#06b6d4";
    }
  }

  selectedVertexIndex = -1;
  draggedPointIndex = -1;
  hoverPointIndex = -1;
  ghostMidpoint = null;

  if (typeof syncPropertiesDrawerValues === "function") {
    syncPropertiesDrawerValues();
  }
  if (typeof renderZoneCanvas === "function") {
    renderZoneCanvas();
  }
}

function confirmAddElement() {
  submitAddElement();
}

function openCalibrationGuide() {
  const modal = document.getElementById("calibrationGuideModal");
  if (modal) modal.style.display = "flex";
}

function closeCalibrationGuide() {
  const modal = document.getElementById("calibrationGuideModal");
  if (modal) modal.style.display = "none";
}

function togglePropertiesDrawer(forceState) {
  const drawer = document.getElementById("zonePropertiesDrawer");
  if (!drawer) return;

  const willShow = typeof forceState === "boolean" ? forceState : drawer.style.display === "none";
  drawer.style.display = willShow ? "flex" : "none";

  if (willShow) {
    syncPropertiesDrawerValues();
  }
}

function syncPropertiesDrawerValues() {
  const drawer = document.getElementById("zonePropertiesDrawer");
  if (!drawer || drawer.style.display === "none") return;

  const polySection = document.getElementById("propsPolygonSection");
  const lineSection = document.getElementById("propsTripwireSection");
  const title = document.getElementById("drawerElemTitle");
  const dot = document.getElementById("drawerElemDot");

  if (activeEditorCategory === "polygons") {
    if (polySection) polySection.style.display = "block";
    if (lineSection) lineSection.style.display = "none";

    const cfg = zoneConfigs[selectedZone] || {};
    const nameInput = document.getElementById("propZoneName");
    if (nameInput) nameInput.value = cfg.name || selectedZone;

    const dwellInput = document.getElementById("propZoneDwell");
    const dwellVal = cfg.dwell_threshold_sec || 60;
    if (dwellInput) dwellInput.value = dwellVal;

    const dwellHelp = document.getElementById("propZoneDwellHelp");
    if (dwellHelp) {
      dwellHelp.textContent = `${dwellVal} detik = ${(dwellVal / 60).toFixed(1)} menit. Melebihi batas ini memicu alarm & Telegram.`;
    }

    // Sync target classes checkboxes for polygons
    const targetClasses = Array.isArray(cfg.target_classes) ? cfg.target_classes : ["truck", "bus", "car"];
    const chkTruck = document.getElementById("propZoneClassTruck");
    const chkBus = document.getElementById("propZoneClassBus");
    const chkCar = document.getElementById("propZoneClassCar");
    const chkPerson = document.getElementById("propZoneClassPerson");
    if (chkTruck) chkTruck.checked = targetClasses.includes("truck");
    if (chkBus) chkBus.checked = targetClasses.includes("bus");
    if (chkCar) chkCar.checked = targetClasses.includes("car");
    if (chkPerson) chkPerson.checked = targetClasses.includes("person");

    if (title) title.textContent = `Properti: ${cfg.name || selectedZone}`;
    if (dot && typeof getActiveZoneColors === "function") {
      dot.style.backgroundColor = getActiveZoneColors(selectedZone).stroke;
    }
  } else {
    if (polySection) polySection.style.display = "none";
    if (lineSection) lineSection.style.display = "block";

    const ldata = lineData[selectedLine] || {};
    const nameInput = document.getElementById("propLineName");
    if (nameInput) nameInput.value = ldata.name || selectedLine;

    const dirSelect = document.getElementById("propLineDirection");
    if (dirSelect) dirSelect.value = ldata.direction || "both";

    // Sync target classes checkboxes for tripwires
    const lineTargetClasses = Array.isArray(ldata.target_classes) ? ldata.target_classes : ["person"];
    const chkLinePerson = document.getElementById("propLineClassPerson");
    const chkLineTruck = document.getElementById("propLineClassTruck");
    const chkLineBus = document.getElementById("propLineClassBus");
    const chkLineCar = document.getElementById("propLineClassCar");
    if (chkLinePerson) chkLinePerson.checked = lineTargetClasses.includes("person");
    if (chkLineTruck) chkLineTruck.checked = lineTargetClasses.includes("truck");
    if (chkLineBus) chkLineBus.checked = lineTargetClasses.includes("bus");
    if (chkLineCar) chkLineCar.checked = lineTargetClasses.includes("car");

    if (title) title.textContent = `Properti: ${ldata.name || selectedLine}`;
    if (dot) dot.style.backgroundColor = "#06b6d4";
  }
}

function updateCurrentElemMeta() {
  if (activeEditorCategory === "polygons") {
    if (!zoneConfigs[selectedZone]) {
      zoneConfigs[selectedZone] = {};
    }
    const nameVal = document.getElementById("propZoneName")?.value.trim() || selectedZone;
    const dwellVal = parseFloat(document.getElementById("propZoneDwell")?.value || "60");

    zoneConfigs[selectedZone].name = nameVal;
    zoneConfigs[selectedZone].dwell_threshold_sec = dwellVal;

    // Collect target classes for zone
    const classes = [];
    if (document.getElementById("propZoneClassTruck")?.checked) classes.push("truck");
    if (document.getElementById("propZoneClassBus")?.checked) classes.push("bus");
    if (document.getElementById("propZoneClassCar")?.checked) classes.push("car");
    if (document.getElementById("propZoneClassPerson")?.checked) classes.push("person");
    zoneConfigs[selectedZone].target_classes = classes;

    const dwellHelp = document.getElementById("propZoneDwellHelp");
    if (dwellHelp) {
      dwellHelp.textContent = `${dwellVal} detik = ${(dwellVal / 60).toFixed(1)} menit toleransi.`;
    }

    const opt = document.querySelector(`#zoneSelector option[value="${selectedZone}"]`);
    if (opt) opt.textContent = nameVal;
  } else {
    if (!lineData[selectedLine]) return;
    const nameVal = document.getElementById("propLineName")?.value.trim() || selectedLine;
    const dirVal = document.getElementById("propLineDirection")?.value || "both";

    lineData[selectedLine].name = nameVal;
    lineData[selectedLine].direction = dirVal;

    // Collect target classes for tripwire
    const lineClasses = [];
    if (document.getElementById("propLineClassPerson")?.checked) lineClasses.push("person");
    if (document.getElementById("propLineClassTruck")?.checked) lineClasses.push("truck");
    if (document.getElementById("propLineClassBus")?.checked) lineClasses.push("bus");
    if (document.getElementById("propLineClassCar")?.checked) lineClasses.push("car");
    lineData[selectedLine].target_classes = lineClasses;

    const opt = document.querySelector(`#zoneSelector option[value="${selectedLine}"]`);
    if (opt) opt.textContent = nameVal;
  }

  if (typeof renderZoneCanvas === "function") {
    renderZoneCanvas();
  }
}

/* ==========================================================================
   ADD & DELETE ELEMENT MODAL
   ========================================================================== */

function openAddElementPrompt() {
  const modal = document.getElementById("addElementModal");
  const title = document.getElementById("addElemModalTitle");
  const nameInput = document.getElementById("newElemName");
  const zoneFields = document.getElementById("newZoneFields");
  const lineFields = document.getElementById("newLineFields");

  if (activeEditorCategory === "polygons") {
    const nextIdx = Object.keys(zoneData).length + 1;
    if (title) title.textContent = "Tambah Zona Poligon Baru";
    if (nameInput) {
      nameInput.value = `Area Khusus ${nextIdx}`;
      nameInput.placeholder = "Zebra Cross / Area Antrean";
    }
    if (zoneFields) zoneFields.style.display = "block";
    if (lineFields) lineFields.style.display = "none";
  } else {
    const nextIdx = Object.keys(lineData).length + 1;
    if (title) title.textContent = "Tambah Garis Virtual (Tripwire)";
    if (nameInput) {
      nameInput.value = `Tripwire ${nextIdx}`;
      nameInput.placeholder = "Garis Pintu Masuk / Koridor";
    }
    if (zoneFields) zoneFields.style.display = "none";
    if (lineFields) lineFields.style.display = "block";
  }

  if (modal) modal.style.display = "flex";
}

function closeAddElementModal() {
  const modal = document.getElementById("addElementModal");
  if (modal) modal.style.display = "none";
}

function submitAddElement() {
  const nameVal = document.getElementById("newElemName")?.value.trim();
  if (!nameVal) {
    showToast("Nama Elemen Wajib", "Masukkan nama untuk elemen baru.", "warning");
    return;
  }

  pushHistory();
  const baseW = baseResolution[0] || 1920;
  const baseH = baseResolution[1] || 1080;

  if (activeEditorCategory === "polygons") {
    let newKey = `zone_${Object.keys(zoneData).length + 1}`;
    while (zoneData[newKey]) {
      newKey = `zone_${Math.floor(Math.random() * 1000)}`;
    }

    const dwellVal = parseFloat(document.getElementById("newZoneDwell")?.value || "60");

    const cx = Math.round(baseW / 2);
    const cy = Math.round(baseH / 2);
    const halfW = Math.round(baseW * 0.18);
    const halfH = Math.round(baseH * 0.16);

    zoneData[newKey] = [
      [cx - halfW, cy - halfH],
      [cx + halfW, cy - halfH],
      [cx + halfW, cy + halfH],
      [cx - halfW, cy + halfH],
    ];

    zoneConfigs[newKey] = {
      name: nameVal,
      dwell_threshold_sec: dwellVal,
      detect_unattended: false,
      target_classes: ["truck", "bus", "car"],
    };

    selectedZone = newKey;
    showToast("Zona Ditambahkan", `Zona '${nameVal}' berhasil dibuat. Silakan atur titik koordinat.`, "success");
  } else {
    let newKey = `line_${Object.keys(lineData).length + 1}`;
    while (lineData[newKey]) {
      newKey = `line_${Math.floor(Math.random() * 1000)}`;
    }

    const dirVal = document.getElementById("newLineDir")?.value || "both";
    const cy = Math.round(baseH / 2);
    const p1x = Math.round(baseW * 0.25);
    const p2x = Math.round(baseW * 0.75);

    lineData[newKey] = {
      name: nameVal,
      p1: [p1x, cy],
      p2: [p2x, cy],
      direction: dirVal,
      target_classes: ["person"],
    };

    selectedLine = newKey;
    showToast("Tripwire Ditambahkan", `Garis '${nameVal}' berhasil dibuat. Geser titik A dan B.`, "success");
  }

  closeAddElementModal();
  populateElementSelector();
  togglePropertiesDrawer(true);
  if (typeof renderZoneCanvas === "function") {
    renderZoneCanvas();
  }
}

function deleteCurrentElement() {
  if (activeEditorCategory === "polygons") {
    const keys = Object.keys(zoneData || {});
    if (keys.length === 0 || !selectedZone || !zoneData[selectedZone]) {
      showToast("Tidak Ada Zona", "Tidak ada zona yang dipilih untuk dihapus.", "warning");
      return;
    }
    pushHistory();
    delete zoneData[selectedZone];
    delete zoneConfigs[selectedZone];

    const remainingKeys = Object.keys(zoneData || {});
    selectedZone = remainingKeys.length > 0 ? remainingKeys[0] : "";
    selectedVertexIndex = -1;
    if (typeof window !== "undefined") window.selectedVertexIndex = -1;
    hoverPointIndex = -1;
    draggedPointIndex = -1;
    ghostMidpoint = null;

    if (remainingKeys.length === 0) {
      showToast(
        "Semua Zona Dihapus",
        "Kanvas sekarang kosong (0 zona). Klik tombol 'Simpan' untuk menerapkan ke CCTV atau '+' untuk membuat zona baru.",
        "info",
        5000
      );
    } else {
      showToast("Zona Dihapus", "Zona berhasil dihapus.", "success");
    }
  } else {
    const keys = Object.keys(lineData || {});
    if (keys.length === 0 || !selectedLine || !lineData[selectedLine]) {
      showToast("Tidak Ada Elemen", "Tidak ada tripwire yang dipilih untuk dihapus.", "warning");
      return;
    }
    pushHistory();
    delete lineData[selectedLine];
    const remainingLines = Object.keys(lineData || {});
    selectedLine = remainingLines.length > 0 ? remainingLines[0] : "";
    selectedVertexIndex = -1;
    if (typeof window !== "undefined") window.selectedVertexIndex = -1;
    hoverPointIndex = -1;
    draggedPointIndex = -1;
    showToast("Tripwire Dihapus", "Garis virtual berhasil dihapus.", "success");
  }

  populateElementSelector();
  if (typeof renderZoneCanvas === "function") {
    renderZoneCanvas();
  }
}

function promptDeleteZoneConfirm() {
  const modal = document.getElementById("deleteZoneConfirmModal");
  const nameDisplay = document.getElementById("delZoneNameDisplay");
  const zoneName = zoneConfigs[selectedZone]?.name || selectedZone;
  if (nameDisplay) nameDisplay.textContent = `'${zoneName}'`;
  if (modal) modal.style.display = "flex";
}

function closeDeleteZoneConfirmModal() {
  const modal = document.getElementById("deleteZoneConfirmModal");
  if (modal) modal.style.display = "none";
}

function confirmDeleteZoneAction() {
  closeDeleteZoneConfirmModal();
  deleteCurrentElement();
}

function resetCurrentZone() {
  pushHistory();
  if (activeEditorCategory === "polygons") {
    if (originalZoneData[selectedZone]) {
      zoneData[selectedZone] = JSON.parse(JSON.stringify(originalZoneData[selectedZone]));
      showToast("Reset Selesai", `Zona ${selectedZone} dikembalikan ke konfigurasi tersimpan.`, "warning");
    }
  } else {
    if (originalLineData[selectedLine]) {
      lineData[selectedLine] = JSON.parse(JSON.stringify(originalLineData[selectedLine]));
      showToast("Reset Selesai", `Tripwire ${selectedLine} dikembalikan ke konfigurasi tersimpan.`, "warning");
    }
  }
  selectedVertexIndex = -1;
  draggedPointIndex = -1;
  hoverPointIndex = -1;
  ghostMidpoint = null;
  if (typeof renderZoneCanvas === "function") {
    renderZoneCanvas();
  }
}

function deleteSelectedPoint() {
  if (activeEditorCategory !== "polygons") return;
  const pts = zoneData[selectedZone] || [];

  // Robust target resolution: check selectedVertexIndex, window.selectedVertexIndex, or hoverPointIndex
  let targetIdx = -1;
  if (typeof selectedVertexIndex !== "undefined" && selectedVertexIndex >= 0) {
    targetIdx = selectedVertexIndex;
  } else if (typeof window.selectedVertexIndex !== "undefined" && window.selectedVertexIndex >= 0) {
    targetIdx = window.selectedVertexIndex;
  } else if (typeof hoverPointIndex !== "undefined" && hoverPointIndex >= 0) {
    targetIdx = hoverPointIndex;
  }

  if (targetIdx < 0 || targetIdx >= pts.length) {
    showToast("Pilih Titik", "Klik titik sudut terlebih dahulu, atau langsung Klik Kanan pada titik di video.", "warning", 3000);
    return;
  }
  if (pts.length <= 3) {
    promptDeleteZoneConfirm();
    return;
  }

  pushHistory();
  pts.splice(targetIdx, 1);
  selectedVertexIndex = -1;
  window.selectedVertexIndex = -1;
  hoverPointIndex = -1;
  if (typeof renderZoneCanvas === "function") {
    renderZoneCanvas();
  }
  showToast("Titik Dihapus", `Titik P${targetIdx + 1} berhasil dihapus. Tekan Ctrl+Z untuk membatalkan.`, "success", 2500);
}

/* ==========================================================================
   ATOMIC ZONE & TRIPWIRE SAVE
   ========================================================================== */

async function saveZonesToServer() {
  for (const [zKey, pts] of Object.entries(zoneData)) {
    if (!pts || pts.length === 0) {
      // Zona kosong (0 titik) diperbolehkan sebagai zona cadangan / nonaktif
      continue;
    }
    if (pts.length < 3) {
      showToast("Gagal Menyimpan", `Zona '${zKey}' belum selesai dibuat (hanya ${pts.length} titik, minimal 3 titik)!`, "warning");
      return;
    }
    if (isPolygonSelfIntersecting(pts)) {
      showToast("Geometri Bersilangan", `Garis pada zona '${zKey}' bersilangan! Harap rapikan.`, "error");
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
        lines: lineData,
        zone_configs: zoneConfigs,
        base_resolution: baseResolution,
      }),
    });

    const res = await resp.json();
    if (!resp.ok) {
      throw new Error(res.detail || "Gagal menyimpan konfigurasi zona & tripwire.");
    }

    originalZoneData = JSON.parse(JSON.stringify(zoneData));
    originalLineData = JSON.parse(JSON.stringify(lineData));

    showToast(
      "Kalibrasi Berhasil Diterapkan",
      "Zona dan tripwire telah dimuat ulang secara instan ke pipeline AI.",
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

function pushHistory() {
  historyStack.push({
    zones: JSON.parse(JSON.stringify(zoneData)),
    lines: JSON.parse(JSON.stringify(lineData)),
    configs: JSON.parse(JSON.stringify(zoneConfigs)),
  });
  if (historyStack.length > 25) {
    historyStack.shift();
  }
}

function undoZoneAction() {
  if (historyStack.length === 0) {
    showToast("Undo", "Tidak ada perubahan yang dapat diurungkan.", "warning", 2000);
    return;
  }
  const prev = historyStack.pop();
  zoneData = prev.zones;
  lineData = prev.lines;
  zoneConfigs = prev.configs;
  selectedVertexIndex = -1;
  draggedPointIndex = -1;
  hoverPointIndex = -1;
  ghostMidpoint = null;
  populateElementSelector();
  if (typeof renderZoneCanvas === "function") {
    renderZoneCanvas();
  }
  showToast("Undo", "Perubahan sebelumnya berhasil diurungkan.", "warning", 2000);
}

// Window global exports
window.populateElementSelector = populateElementSelector;
window.onZoneSelectChange = onZoneSelectChange;
window.confirmAddElement = confirmAddElement;
window.openCalibrationGuide = openCalibrationGuide;
window.closeCalibrationGuide = closeCalibrationGuide;
