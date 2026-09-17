# Smart CCTV — Edge Video Analytics & Hybrid Surveillance System

Platform analitik video cerdas multi-kamera berbasis Computer Vision dan Deep Learning untuk monitoring operasional, deteksi perlintasan batas (tripwire line-crossing), serta pengawasan zona terlarang dan objek tertinggal secara real-time.

Sistem memproses stream RTSP dari IP Camera standar (*hardware-agnostic*) menggunakan arsitektur hybrid terakselerasi CPU, dilengkapi Web Dashboard manajemen visual, persistensi database SQLite (mode WAL), perekaman bukti pelanggaran (snapshot & klip video MP4), serta notifikasi instan Telegram.

---

## Arsitektur Deteksi: Hybrid AI + Classical CV

Sistem menggunakan pendekatan **Hybrid Engine** yang menggabungkan keunggulan Deep Learning dan Classical Computer Vision:

```
RTSP Stream (IP Cam) ──► ThreadedCapture (Frame Ingestion)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   [ Jalur A: Real-Time ]                [ Jalur B: Verifier ]
    YOLO11n via OpenVINO                   DualSubtractor MOG2
   (Person, Car, Truck, dll.)           (Slow H=5000 & Fast H=40)
            │                                     │
     CentroidTracker                              │
   (Track ID & History)                    Static Anomaly Blob
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
               [ Confirmation Gate Window (5.0s) ]
           (Blob MOG2 wajib dikonfirmasi bounding box YOLO)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
    Zone Dwell Monitor                   TripwireEngine
 (Area Koridor / Ruangan)             (Garis Batas Timbangan)
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
              Event Trigger, SQLite Log, MP4 Clip,
                 Telegram Alert & Visual HUD
```

### Mengapa Hybrid?
- **YOLO11n (OpenVINO)**: Bertindak sebagai detektor utama real-time untuk mendeteksi dan mengklasifikasikan objek bergerak dinamis (`person`, `car`, `bus`, `truck`, `backpack`, `handbag`, `suitcase`) dengan akselerasi CPU Intel (INT8/FP16).
- **DualSubtractor (Dual MOG2)**: Menggunakan kombinasi *Slow MOG2* ($History=5000$) dan *Fast MOG2* ($History=40$) khusus untuk menangkap anomali statis (barang tergeletak atau kendaraan berhenti lama) yang rawan terlewat oleh inferensi reguler.
- **Confirmation Gate (5.0 Detik)**: Blob anomali statis dari MOG2 **wajib** dikonfirmasi oleh bounding box YOLO dalam jendela waktu $\le 5$ detik sebelum dinyatakan sebagai event valid. Pendekatan ini secara drastis mengeliminasi *false alarm* akibat perubahan intensitas cahaya mendadak, bayangan awan, atau pantulan kaca.

---

## Metode Deteksi Spesifik per Kamera

Metode deteksi diatur secara independen sesuai karakteristik spasial dan fungsi operasional masing-masing kamera:

| Kamera | Lokasi / Peruntukan | Metode Deteksi | Deskripsi & Rasionalisasi |
|---|---|---|---|
| **cam_01** | Koridor LT.2 Gedung A | **Zone-based Dwell Detection** | Memantau poligon area transit/koridor. Mengukur durasi diam (*dwell time*) untuk mendeteksi orang nongkrong (*loitering*) atau barang tertinggal. **Tidak menggunakan garis virtual.** |
| **cam_04** | Area Timbangan Truk | **Tripwire Line-Crossing** | Memantau 3 garis virtual melintang. Menggunakan `TripwireEngine` (*Segment Intersection*, Liang-Barsky occlusion guard, dan 4-layer anti-ghost filter) untuk menghitung arus kendaraan/personil masuk dan keluar. |
| **cam_02 & cam_03** | Gerbang & Perimeter | **Zone Monitoring** | Pengawasan area terlarang (*restricted perimeter*) dan deteksi keberadaan personil di luar jam operasional. |

---

## Struktur Folder Repositori

```
smart-cctv/
├── cameras/                 # Workspace konfigurasi per kamera
│   ├── cam_01/              # Kamera 1 (Koridor LT.2)
│   │   ├── config.json      # Pengaturan RTSP, resolusi, alert, & target class
│   │   └── roi_zones.json   # Koordinat poligon zona pantau
│   ├── cam_02/ .. cam_04/   # Workspace kamera lainnya (cam_04 memuat konfigurasi garis tripwire)
├── engine/                  # Core pipeline & analytics engine
│   ├── config_loader.py     # Parser JSON & ekspansi variabel .env
│   ├── dual_subtractor.py   # Dual MOG2 static anomaly verifier
│   ├── line_crossing.py     # TripwireEngine (vektor crossing & anti-ghost gate)
│   ├── retention.py         # DiskGuardWorker & auto-purge retensi storage
│   ├── rtsp_stream.py       # Threaded capture non-blocking anti-lag RTSP
│   ├── tracker.py           # CentroidTracker & track lifecycle management
│   ├── yolo_detector.py     # YOLO11n detector terakselerasi OpenVINO
│   └── zone_filter.py       # Multi-zone polygon point-in-polygon tester
├── notification/            # Subsistem alerting & rendering
│   ├── buzzer_alert.py      # Notifikasi hardware buzzer
│   ├── local_alert.py       # Visual HUD overlay & audio chime alert
│   └── telegram_alert.py    # Telegram Bot alert (kirim foto snapshot + caption)
├── storage/                 # Manajemen persistensi data
│   └── db.py                # Engine SQLite (mode WAL, event logs, indexing)
├── web/                     # Web Dashboard & Streaming Server
│   ├── server.py            # REST API backend berbasis FastAPI & Uvicorn
│   ├── buffer.py            # MultiCameraBuffer untuk multi-stream MJPEG
│   └── templates/ & static/ # UI antarmuka web interaktif
├── tools/                   # Utilitas diagnostik & kalibrasi
│   ├── check_camera.py      # Diagnostik koneksi TCP & handshake RTSP
│   └── roi_calibrator.py    # GUI kalibrasi koordinat zona visual
├── models/                  # Bobot model AI (YOLO OpenVINO IR / ONNX)
├── data/                    # Database SQLite runtime, snapshot, & klip rekaman MP4
└── docs/                    # Dokumentasi arsitektur, SOP, & roadmap multi-proses
```

---

## Panduan Instalasi & Menjalankan Sistem

### 1. Prasyarat Sistem
- **OS**: Windows 10/11 atau Linux (Ubuntu 20.04+)
- **Python**: Versi 3.10 s/d 3.12 (64-bit)
- **Akselerasi AI**: CPU Intel dengan dukungan AVX-512/VNNI direkomendasikan untuk OpenVINO Toolkit.

### 2. Pemasangan Dependensi
Buka terminal di root direktori proyek:
```bash
# Buat virtual environment
python -m venv .venv

# Aktivasi virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux:
source .venv/bin/activate

# Install package dependensi
pip install -r requirements.txt
```

### 3. Konfigurasi Kredensial `.env`
Salin template `.env.example` ke `.env`:
```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux
```
Sesuaikan kredensial RTSP kamera dan bot Telegram:
```env
CAM_DEFAULT_USER=admin
CAM_DEFAULT_PASS=PasswordKamera123

# Konfigurasi Notifikasi Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRstuVWXyz
TELEGRAM_CHAT_ID=-1001234567890
```

### 4. Menjalankan Aplikasi

Aplikasi dapat dijalankan melalui beberapa mode CLI:

#### Mode Headless (Default / Server Mode)
Menjalankan seluruh pipeline kamera di latar belakang dan mengaktifkan Web Dashboard tanpa membuka jendela GUI desktop:
```bash
python main.py
# atau eksplisit:
python main.py --headless
```

#### Mode GUI Desktop
Menjalankan pipeline sekaligus membuka jendela pemantauan visual OpenCV (`cv2.imshow`) pada monitor lokal:
```bash
python main.py --gui
```

#### Opsi Custom Port Dashboard
```bash
python main.py --host 0.0.0.0 --port 8080
```

Setelah aplikasi berjalan, buka Web Dashboard pada peramban web:
👉 **`http://localhost:8000`**

---

## Manajemen Kamera & Kalibrasi Zona (Tanpa Edit Kode)

Sistem telah dirancang agar penambahan titik kamera dan kalibrasi zona dapat dilakukan secara visual:

### 1. Menambah Kamera Baru
1. Buat direktori baru di dalam folder `cameras/` (contoh: `cameras/cam_05/`).
2. Buat file `config.json` (bisa menyalin dari template kamera yang sudah ada) dan tentukan URL RTSP atau kredensial kamera.
3. Buat file `roi_zones.json` awal:
   ```json
   {
     "zones": {},
     "lines": {}
   }
   ```
4. Sistem otomatis mendeteksi kamera baru saat startup atau melalui menu penambahan kamera di Web Dashboard.

### 2. Kalibrasi Visual Zona / Tripwire via Web Dashboard
1. Buka Web Dashboard di `http://localhost:8000`.
2. Pilih kamera yang ingin dikalibrasi dari daftar panel kamera.
3. Masuk ke menu **Zone Editor / Calibration**:
   - Untuk **Zone (Area)**: Klik beberapa titik pada feed video untuk membentuk poligon batas pantau.
   - Untuk **Tripwire (Garis)**: Klik dua titik ($P_1$ dan $P_2$) untuk mendefinisikan garis perlintasan virtual dan tentukan arahnya (*in*, *out*, atau *both*).
4. Klik **Save Zones**. Koordinat akan langsung tersimpan ke `cameras/cam_XX/roi_zones.json` tanpa perlu me-restart kode program secara manual.

---

## Known Limitations (Batasan yang Diketahui)

1. **Parameter Tripwire Engine Masih Bersifat Global / Hardcoded**:
   Parameter internal tripwire (seperti `cooldown_sec`, `min_track_frames`, `max_step_px`, dan toleransi displacement) saat ini masih menggunakan nilai default yang didefinisikan di `main.py`, dan belum dapat dikustomisasi per-kamera melalui `config.json`.
2. **Konkurensi Single-Process**:
   Saat ini seluruh pipeline kamera (4 kamera) berjalan di dalam satu proses Python (*multi-threaded runtime*). Untuk ekspansi skala besar (>8 kamera), sistem telah memiliki cetak biru migrasi ke arsitektur *multi-process* dengan `SharedMemory` IPC (lihat detail di [docs/ROADMAP_MULTIPROCESS_VMS.md](file:///d:/Project/smart-cctv/docs/ROADMAP_MULTIPROCESS_VMS.md)).
3. **Modul Pengenalan Wajah (Face Recognition)**:
   Modul pendeteksi wajah (YuNet) dan verifikasi identitas (SFace ONNX) sudah tersedia di `engine/`, namun secara default dinonaktifkan (`"enabled": false`) pada seluruh kamera untuk mengoptimalkan alokasi resource CPU bagi deteksi objek utama.

---

## Dokumentasi Terkait
- Dokumen Desain Teknis Lengkap: `docs/Rancangan_Smart_CCTV_Hybrid.md`
- Roadmap Arsitektur Multi-Process VMS: [docs/ROADMAP_MULTIPROCESS_VMS.md](file:///d:/Project/smart-cctv/docs/ROADMAP_MULTIPROCESS_VMS.md)
