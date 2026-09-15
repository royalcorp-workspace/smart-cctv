/**
 * Smart CCTV 2.0 - Camera Onboarding & Safe Deletion Module
 */

function openAddCameraModal() {
  const modal = document.getElementById("addCameraModal");
  const form = document.getElementById("formAddCamera");
  if (form) form.reset();

  // Auto suggest next camera ID based on existing cameras
  const select = document.getElementById("cameraSelect");
  let nextNum = 5;
  if (select) {
    const existingIds = Array.from(select.options).map((o) => o.value);
    for (let i = 1; i <= 99; i++) {
      const idStr = `cam_${String(i).padStart(2, "0")}`;
      if (!existingIds.includes(idStr)) {
        nextNum = i;
        break;
      }
    }
  }
  const camIdInput = document.getElementById("newCamId");
  if (camIdInput) {
    camIdInput.value = `cam_${String(nextNum).padStart(2, "0")}`;
  }

  const portInput = document.getElementById("newCamPort");
  if (portInput) portInput.value = "554";
  const channelInput = document.getElementById("newCamChannel");
  if (channelInput) channelInput.value = "102";
  const userInput = document.getElementById("newCamUser");
  if (userInput) userInput.value = "user";

  const banner = document.getElementById("rtspTestBanner");
  if (banner) banner.style.display = "none";

  updateRtspPreview();
  if (modal) modal.style.display = "flex";
}

function closeAddCameraModal() {
  const modal = document.getElementById("addCameraModal");
  if (modal) modal.style.display = "none";
}

function onCamNameInput() {
  updateRtspPreview();
}

function updateRtspPreview() {
  const ip = document.getElementById("newCamIp")?.value.trim() || "192.212.160.99";
  const port = document.getElementById("newCamPort")?.value.trim() || "554";
  const user = document.getElementById("newCamUser")?.value.trim() || "user";
  const pass = document.getElementById("newCamPass")?.value ? "••••" : "pass";
  const channel = document.getElementById("newCamChannel")?.value.trim() || "102";
  const box = document.getElementById("rtspPreviewBox");
  if (box) {
    box.textContent = `rtsp://${user}:${pass}@${ip}:${port}/Streaming/Channels/${channel}`;
  }
}

async function testRtspConnection() {
  const ip = document.getElementById("newCamIp")?.value.trim();
  const port = parseInt(document.getElementById("newCamPort")?.value || "554", 10);
  const user = document.getElementById("newCamUser")?.value.trim() || "";
  const pass = document.getElementById("newCamPass")?.value || "";
  const channel = parseInt(document.getElementById("newCamChannel")?.value || "102", 10);

  if (!ip) {
    showToast("Input Belum Lengkap", "Silakan masukkan IP address kamera terlebih dahulu.", "warning");
    return;
  }

  const banner = document.getElementById("rtspTestBanner");
  const icon = document.getElementById("rtspTestIcon");
  const msg = document.getElementById("rtspTestMsg");
  const btn = document.getElementById("btnTestRtsp");

  if (banner) {
    banner.style.display = "flex";
    banner.className = "test-banner";
  }
  if (icon) icon.textContent = "⏳";
  if (msg) msg.textContent = "Sedang menguji koneksi RTSP ke " + ip + "...";
  if (btn) btn.disabled = true;

  try {
    const resp = await fetch("/api/cameras/test_rtsp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ip, port, user, pass, channel }),
    });
    const result = await resp.json();
    if (result.success) {
      if (banner) banner.className = "test-banner success";
      if (icon) icon.textContent = "✅";
      if (msg) msg.textContent = result.message || "Koneksi RTSP berhasil terhubung!";
      showToast("Koneksi RTSP Berhasil", result.message, "success", 3000);
    } else {
      if (banner) banner.className = "test-banner error";
      if (icon) icon.textContent = "❌";
      if (msg) msg.textContent = result.message || "Gagal membuka stream RTSP.";
      showToast("Tes RTSP Gagal", result.message, "error", 4000);
    }
  } catch (err) {
    if (banner) banner.className = "test-banner error";
    if (icon) icon.textContent = "❌";
    if (msg) msg.textContent = "Error komunikasi: " + err.message;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function submitAddCamera(event) {
  if (event) event.preventDefault();

  const name = document.getElementById("newCamName")?.value.trim();
  const camera_id = document.getElementById("newCamId")?.value.trim().toLowerCase();
  const ip = document.getElementById("newCamIp")?.value.trim();
  const port = parseInt(document.getElementById("newCamPort")?.value || "554", 10);
  const user = document.getElementById("newCamUser")?.value.trim() || "";
  const pass = document.getElementById("newCamPass")?.value || "";
  const channel = parseInt(document.getElementById("newCamChannel")?.value || "102", 10);

  if (!name || !camera_id || !ip) {
    showToast("Input Wajib", "Nama, ID, dan IP Kamera wajib diisi.", "warning");
    return;
  }

  const btnSubmit = document.getElementById("btnSubmitCam");
  if (btnSubmit) {
    btnSubmit.disabled = true;
    btnSubmit.innerHTML = `<span>Mendaftarkan Kamera...</span>`;
  }

  try {
    const resp = await fetch("/api/cameras/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, camera_id, ip, port, user, pass, channel }),
    });
    const result = await resp.json();
    if (!resp.ok) {
      throw new Error(result.detail || result.message || "Gagal menambahkan kamera.");
    }

    showToast("Kamera Ditambahkan", result.message, "success", 4000);
    closeAddCameraModal();
    await loadCameraList();

    // Automatically switch to the newly registered camera
    const select = document.getElementById("cameraSelect");
    if (select) {
      select.value = camera_id;
      onCameraChange();
    }
  } catch (err) {
    console.error("Add camera error:", err);
    showToast("Gagal Menambahkan Kamera", err.message, "error", 5000);
  } finally {
    if (btnSubmit) {
      btnSubmit.disabled = false;
      btnSubmit.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
        <span>Simpan & Aktifkan Kamera</span>
      `;
    }
  }
}

function openDeleteCameraModal() {
  const modal = document.getElementById("deleteCameraModal");
  const display = document.getElementById("delCamNameDisplay");
  if (display) {
    display.textContent = activeCamera;
  }
  if (modal) modal.style.display = "flex";
}

function closeDeleteCameraModal() {
  const modal = document.getElementById("deleteCameraModal");
  if (modal) modal.style.display = "none";
}

async function confirmDeleteCamera() {
  const btn = document.getElementById("btnConfirmDeleteCam");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span>Menonaktifkan...</span>`;
  }

  try {
    const resp = await fetch("/api/cameras/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: activeCamera }),
    });
    const result = await resp.json();
    if (!resp.ok) {
      throw new Error(result.detail || result.message || "Gagal menghapus kamera.");
    }

    showToast("Kamera Dinonaktifkan", result.message, "success", 4000);
    closeDeleteCameraModal();
    await loadCameraList();
    const select = document.getElementById("cameraSelect");
    if (select && select.options.length > 0) {
      select.value = select.options[0].value;
      onCameraChange();
    }
  } catch (err) {
    console.error("Delete camera error:", err);
    showToast("Gagal Menonaktifkan", err.message, "error", 5000);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
        <span>Ya, Nonaktifkan Kamera</span>
      `;
    }
  }
}
