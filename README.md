# Smart CCTV 2.0 — Real-Time Spatial Obstruction & Dwell Detection System

Sistem analitik video cerdas berbasis Computer Vision untuk mendeteksi barang tergeletak dan objek diam di koridor kantor serta area transit secara otomatis. Menggunakan logika Dual-MOG2 Subtractor, penapisan poligon spasial (ROI), persistensi database SQLite mode WAL, perekaman snapshot bukti pelanggaran, dan sistem alarm audio bertahap.

---

## Fitur Utama

- **Threaded Frame Grabber & Auto-Reconnect**: Buffer frame tunggal non-blocking (buffer size = 1) untuk mengeliminasi akumulasi latensi stream RTSP (kompatibel penuh dengan sub-stream & main-stream Hikvision/Dahua/RTSP IP Cam) serta fitur auto-reconnect tanpa crash.
- **Dual MOG2 Static Object Extractor**: Kombinasi `Slow_MOG2` (history panjang) dan `Fast_MOG2` (history responsif) dengan formula vektor:
  $$\text{Static\_Mask} = \text{Slow\_MOG2} \land \neg(\text{Fast\_MOG2})$$
  dilengkapi eliminasi bayangan abu-abu dan pembersihan morfologi (`OPEN` + `CLOSE`).
- **Interactive Multi-Zone ROI Calibrator**: Penentuan batas koordinat poligon zona visual melalui klik mouse langsung pada layar feed kamera.
- **Multi-Zone Dwell Timing**:
  - **Zona 1 (Koridor Utama)**: Batas waktu diam > 15 detik, ambang kontur min. 400 px.
  - **Zona 2 (Area Transit Depan)**: Batas waktu diam > 30 menit (1800 detik), ambang kontur min. 2500 px.
- **Hardened Security & Isolation**: Kredensial kamera tersimpan aman di `.env` (tidak ter-commit ke Git) dengan substitusi variabel `${CAM_USER}` & `${CAM_PASS}` dinamis.
- **Auto-Purge & Retention Maintenance**: Pembersihan otomatis snapshot gambar dan log SQLite berumur > 30 hari saat startup untuk menghemat kapasitas disk.
- **Laporan & Diagnostik Lengkap**: Diagnostik socket TCP/RTSP mandiri dan ekspor laporan CSV satu klik.

---

## Struktur Direktori

```
smart-cctv/
├── cameras/                     # Workspace konfigurasi per kamera
│   └── cam_01/
│       ├── config.json          # Parameter kamera, target resolusi, threshold zona
│       ├── roi_zones.json       # Koordinat poligon ROI Zona 1 & Zona 2
│       └── snapshots/           # Folder bukti snapshot kejadian pelanggaran
│
├── engine/                      # Core pipeline logic (shared)
│   ├── __init__.py
│   ├── config_loader.py         # Parser config & ekspansi variabel .env
│   ├── dual_subtractor.py       # Dual MOG2 background subtractor
│   ├── retention.py             # Policy pembersihan data kadaluarsa & VACUUM SQLite
│   ├── rtsp_stream.py           # Threaded capture & anti-lag buffer
│   ├── tracker.py               # Centroid tracker, dwell calculator, anti-leak purge
│   └── zone_filter.py           # Multi-zone polygon test & area filter
│
├── storage/                     # Persistensi data lokal
│   ├── __init__.py
│   └── db.py                    # SQLite engine (WAL mode, index, event_logs schema)
│
├── notification/                # Decoupled notification & rendering
│   ├── __init__.py
│   ├── alert.wav                # Audio nada peringatan bawaan
│   └── local_alert.py           # Audio cooldown worker & Visual HUD overlay
│
├── tools/                       # Alat bantu operasional
│   ├── check_camera.py          # Diagnostik koneksi TCP & handshake RTSP
│   ├── export_incidents.py      # Ekspor log database ke format CSV
│   ├── purge_old_data.py        # Eksekusi manual pembersihan data retensi
│   └── roi_calibrator.py        # GUI kalibrasi poligon zona interaktif
│
├── reports/                     # Folder penyimpanan laporan ekspor CSV
├── calibrate_cam01.bat          # Runner 1-klik: Kalibrasi ROI cam_01
├── export_report.bat            # Runner 1-klik: Ekspor laporan insiden CSV
├── purge_data.bat               # Runner 1-klik: Bersihkan data lama (>30 hari)
├── start_cctv.bat               # Runner 1-klik: Menjalankan sistem CCTV live
├── test_camera.bat              # Runner 1-klik: Uji diagnostik koneksi kamera
├── requirements.txt             # Dependensi Python terkunci
├── .env.example                 # Template environment variables aman
├── .gitignore                   # Proteksi git untuk venv, .env, DB, & snapshots
└── main.py                      # Entry point orchestrator multi-kamera
```

---

## Panduan Instalasi & Setup Awal

### 1. Prasyarat Sistem
- **Sistem Operasi**: Windows 10 / 11 / Server
- **Python**: Python 3.12 (64-bit)
- **Konektivitas**: Terhubung ke LAN/Jaringan yang dapat mengakses IP kamera CCTV.

### 2. Setup Virtual Environment
Buka terminal PowerShell atau CMD di root folder `smart-cctv/`:
```cmd
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 3. Konfigurasi Kredensial `.env`
Salin template `.env.example` menjadi `.env`:
```cmd
copy .env.example .env
```
Buka file `.env` dan masukkan username serta password kamera:
```env
CAM01_USER=admin
CAM01_PASS=KataSandiAsliKamera
```

---

## SOP Penggunaan Lapangan (1-Click Runners)

Jalankan skrip `.bat` sesuai tahapan berikut:

### Langkah 1: Uji Koneksi Kamera (`test_camera.bat`)
Klik ganda `test_camera.bat` untuk memvalidasi ketersediaan stream RTSP:
- Melakukan ping TCP socket ke Port 554.
- Menguji otentikasi RTSP dan decode 1 frame.
- Menampilkan resolusi native dan FPS kamera jika berhasil.

### Langkah 2: Kalibrasi Batas Zona (`calibrate_cam01.bat`)
Klik ganda `calibrate_cam01.bat` untuk mengatur area pantau visual:
1. Tekan tombol `1` untuk mengaktifkan **Zona 1 (Koridor Utama)**:
   - Klik mouse pada canvas video untuk membuat titik-titik sudut poligon.
2. Tekan tombol `2` untuk beralih ke **Zona 2 (Area Transit Depan)**:
   - Klik mouse untuk membuat batas poligon zona transit.
3. Tekan tombol `S` untuk menyimpan koordinat ke `cameras/cam_01/roi_zones.json`.
4. Tekan tombol `Q` untuk keluar dari calibrator.

### Langkah 3: Menjalankan Sistem Pemantauan Live (`start_cctv.bat`)
Klik ganda `start_cctv.bat`:
- Sistem otomatis memuat seluruh konfigurasi kamera dari folder `cameras/`.
- Menjalankan pembersihan retensi data otomatis (>30 hari).
- Menampilkan jendela display video beranotasi:
  - **Kotak Hijau**: Orang / objek bergerak aktif (Moving).
  - **Kotak Kuning**: Objek mulai diam (Dwell timer berjalan).
  - **Kotak Merah & Border Kedip**: Terjadi pelanggaran batas waktu diam (Violation).
  - **Snapshot Otomatis**: Gambar raw disimpan ke `cameras/cam_01/snapshots/`.
  - **Audio Alarm**: Berbunyi dalam siklus 3 detik aktif dan 15 detik jeda hening.
  - **Database Logging**: Tercatat otomatis di `storage/events.db`.
  - **Auto-Resolve**: Saat objek diangkat/hilang, status insiden otomatis diperbarui menjadi resolved.

### Langkah 4: Ekspor Laporan Rekap Insiden (`export_report.bat`)
Klik ganda `export_report.bat`:
- Seluruh riwayat pelanggaran diekstrak ke file CSV di folder `reports/incident_report_YYYYMMDD_HHMMSS.csv`.
- Memuat kolom: `ID`, `Camera_ID`, `Zone_ID`, `Start_Time`, `End_Time`, `Duration_Seconds`, `Snapshot_Path`, dan `Status_Resolved`.

### Langkah 5: Pemeliharaan Retensi Manual (`purge_data.bat`)
Klik ganda `purge_data.bat`:
- Menghapus snapshot fisik dan record insiden yang berumur lebih dari 30 hari secara manual.
- Mengklaim ulang ruang penyimpanan harddisk via perintah `VACUUM`.

---

## Pintasan Keyboard & Kontrol

### 1. ROI Calibrator (`tools/roi_calibrator.py`)
| Tombol | Fungsi |
| :---: | :--- |
| `1` | Beralih edit Zona 1 (`zone_1_koridor`) |
| `2` | Beralih edit Zona 2 (`zone_2_transit`) |
| `U` / `Z` | Undo (menghapus titik sudut poligon terakhir) |
| `C` | Clear (menghapus seluruh titik pada zona aktif) |
| `S` | Simpan seluruh konfigurasi poligon ke `roi_zones.json` |
| `Q` / `ESC` | Keluar dari aplikasi kalibrasi |

### 2. Live Monitoring Window (`main.py`)
| Tombol | Fungsi |
| :---: | :--- |
| `Q` / `ESC` | Menghentikan seluruh pipeline kamera dan mematikan sistem secara aman |

---

## Lisensi & Hak Cipta
Internal Enterprise CCTV Analytical System — Versi 2.0.
