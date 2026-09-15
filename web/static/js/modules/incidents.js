/**
 * Smart CCTV 2.0 - Incident Logs & DVR HTML5 Video Player Module
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

    tbody.innerHTML = events
      .map((ev) => {
        const dt = ev.trigger_time ? new Date(ev.trigger_time).toLocaleString("id-ID") : "--";
        const dwell = ev.dwell_duration
          ? `${(ev.dwell_duration / 60).toFixed(1)} menit (${ev.dwell_duration.toFixed(0)}s)`
          : "--";
        const statusText = ev.is_resolved
          ? `<span class="tag-badge authorized">Selesai</span>`
          : `<span class="tag-badge unknown">Aktif</span>`;
        const owner =
          ev.owner_name && ev.owner_name.toLowerCase() !== "unknown"
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
      })
      .join("");
  } catch (err) {
    console.warn("Gagal memuat log insiden:", err);
  }
}

function openVideoPlayer(clipPath, camId, timestamp, dwell) {
  const modal = document.getElementById("videoModal");
  const player = document.getElementById("incidentVideoPlayer");
  const title = document.getElementById("modalClipTitle");
  const meta = document.getElementById("modalClipMeta");
  const dlBtn = document.getElementById("modalDownloadBtn");
  if (!modal || !player) return;

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
