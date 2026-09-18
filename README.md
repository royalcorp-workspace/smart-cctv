# Smart CCTV — Edge Video Analytics & Hybrid Surveillance System

Platform analitik video cerdas multi-kamera berbasis Computer Vision dan Deep Learning untuk monitoring operasional, deteksi perlintasan batas (*tripwire line-crossing*), pengawasan zona kepatuhan K3, batas waktu parkir kendaraan, serta pemantauan zona steril dan barang tertinggal secara real-time.

Sistem memproses stream RTSP dari IP Camera standar (*hardware-agnostic*) menggunakan arsitektur hybrid terakselerasi CPU, dilengkapi Web Dashboard manajemen visual interaktif, persistensi database SQLite (mode WAL), perekaman bukti pelanggaran (snapshot & klip video MP4), serta notifikasi instan Telegram.

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
 (Area Koridor / Parkir)              (Garis Batas Timbangan)
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
              Event Trigger, SQLite Log, MP4 Clip,
                 Telegram Alert & Visual HUD
```

### Keunggulan Arsitektur Hybrid:
- **YOLO11n (OpenVINO)**: Bertindak sebagai detektor utama real-time untuk mendeteksi dan mengklasifikasikan objek dinamis (`person`, `car`, `bus`, `truck`, `backpack`, `handbag`, `suitcase`) dengan akselerasi CPU Intel (INT8/FP16).
- **DualSubtractor (Dual MOG2)**: Menggunakan kombinasi *Slow MOG2* ($History=5000$) dan *Fast MOG2* ($History=40$) khusus untuk menangkap anomali statis (barang tergeletak atau kendaraan berhenti lama) yang rawan terlewat oleh inferensi reguler.
- **Confirmation Gate (5.0 Detik)**: Blob anomali statis dari MOG2 **wajib** dikonfirmasi oleh bounding box YOLO dalam jendela waktu $\le 5$ detik sebelum dinyatakan sebagai event valid. Pendekatan ini mengeliminasi *false alarm* akibat perubahan cahaya mendadak, bayangan awan, atau pantulan kaca.
- **In-Zone Only Person Filtering**: Khusus kamera indoor (`cam_01`, `cam_05`), sistem hanya menampilkan kotak deteksi orang yang berada di dalam zona pantau aktif (`zone_1_koridor`, `zone_2_transit`). Objek di luar zona seperti kursi kantor, jaket, dan staf di kubikel meja kerja otomatis diabaikan sehingga tampilan feed bebas dari *false positive*.

---

## Karakteristik & Konfigurasi per Kamera

Setiap kamera dikonfigurasi secara mandiri sesuai karakteristik spasial dan SOP operasional:

| Kamera | Nama & Lokasi | Metode Deteksi Utama | Target Kelas & Aturan Pelanggaran |
|---|---|---|---|
| **cam_01** | Koridor Utama LT.2 Gedung A | **Zone Dwell & Corridor Monitoring** | Memantau zona koridor (`zone_1_koridor`) dan area meja rapat (`zone_2_transit`). Deteksi orang hanya aktif di dalam zona pantau (menyaring kursi/meja kantor). |
| **cam_02** | Area Parkir POS-2 | **Vehicle Dwell & Obstruction** | Memantau antrean/parkir kendaraan (`truck`, `bus`, `car`). Zebra cross toleransi 60 detik; area parkir toleransi 30 menit. |
| **cam_03** | Jalur Logistik Arah POS-1 | **Logistics Traffic & Perimeter** | Pengawasan jalur manuver logistik (toleransi 20 menit). Dilengkapi sensitivitas tinggi untuk mendeteksi pejalan kaki berjarak jauh di pintu gerbang. |
| **cam_04** | Area Timbangan Truk | **Tripwire & K3 Walkway Compliance** | Memantau 3 garis tripwire arah masuk/keluar timbangan, clearance area zebra cross, serta kepatuhan jalur pejalan kaki K3 (*Walkway Compliance*). |
| **cam_05** | Area Uji Coba | **Indoor Testbed Workspace** | Workspace eksperimen live monitoring indoor dengan konfigurasi terisolasi untuk pengujian zona dan analitik baru. |

---

## Struktur Folder Repositori

```
smart-cctv/
├── cameras/                 # Workspace konfigurasi mandiri per kamera
│   ├── cam_01/              # Kamera 1 (Koridor Utama LT.2)
│   ├── cam_02/              # Kamera 2 (Area Parkir POS-2)
│   ├── cam_03/              # Kamera 3 (Jalur Logistik POS-1)
│   ├── cam_04/              # Kamera 4 (Area Timbangan Truk & Tripwire)
│   └── cam_05/              # Kamera 5 (Area Uji Coba)
│       ├── config.json      # Pengaturan RTSP, resolusi, alert, & target class
│       └── roi_zones.json   # Koordinat poligon zona dan garis tripwire
├── engine/                  # Core pipeline & analytics engine
│   ├── config_loader.py     # Parser JSON & ekspansi variabel .env
│   ├── dual_subtractor.py   # Dual MOG2 static anomaly verifier
│   ├── line_crossing.py     # TripwireEngine (vektor crossing & anti-ghost gate)
│   ├── retention.py         # DiskGuardWorker & auto-purge retensi storage
│   ├── rtsp_stream.py       # Threaded capture non-blocking anti-lag RTSP
│   ├── tracker.py           # CentroidTracker & track lifecycle management
│   ├── yolo_detector.py     # YOLO11n detector terakselerasi OpenVINO
│   └── zone_filter.py       # Multi-zone polygon point-in-polygon tester
├── notification/            # Subsistem alerting & visual overlay
│   ├── buzzer_alert.py      # Notifikasi webhook hardware buzzer
│   ├── local_alert.py       # VisualHUD overlay, status badge, & audio chime
│   └── telegram_alert.py    # Telegram Bot alert (kirim foto snapshot + caption)
├── storage/                 # Manajemen persistensi data
│   └── db.py                # Engine SQLite (mode WAL, event logs, indexing)
├── web/                     # Web Dashboard & Streaming Server
│   ├── server.py            # REST API backend berbasis FastAPI & Uvicorn
│   ├── buffer.py            # MultiCameraBuffer untuk multi-stream MJPEG
│   └── templates/ & static/ # UI antarmuka web interaktif (Single/Grid View)
├── tools/                   # Utilitas diagnostik, kalibrasi, & benchmark
│   ├── check_camera.py      # Diagnostik koneksi TCP & handshake RTSP
│   ├── roi_calibrator.py    # GUI kalibrasi koordinat zona visual
│   ├── test_pose_comparison.py # Evaluasi komparasi model Standar vs Pose
│   └── test_person_box_and_zone_classes.py # Pengujian unit test zona & kelas
├── models/                  # Bobot model AI (YOLO OpenVINO IR / ONNX / PT)
├── data/                    # Database SQLite runtime, snapshot, & klip MP4
└── docs/                    # Dokumentasi teknis, SOP, & roadmap arsitektur
```

---

## Panduan Instalasi & Menjalankan Sistem

### 1. Prasyarat Sistem
- **OS**: Windows 10/11 atau Linux (Ubuntu 20.04+)
- **Python**: Versi 3.10 s/d 3.12 (64-bit)
- **Akselerasi AI**: CPU Intel dengan instruksi AVX-512/VNNI direkomendasikan untuk OpenVINO Toolkit.

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

# Install dependensi
pip install -r requirements.txt
```

### 3. Konfigurasi Kredensial `.env`
Salin template `.env.example` ke `.env`:
```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux
```
Sesuaikan kredensial IP kamera dan notifikasi Telegram:
```env
# Kredensial Kamera per Kamera
CAM01_IP=172.16.2.101
CAM01_PORT=554
CAM01_USER=admin
CAM01_PASS=PasswordKamera123
CAM01_CHANNEL=101

# Konfigurasi Notifikasi Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRstuVWXyz
TELEGRAM_CHAT_ID=-1001234567890
```

### 4. Menjalankan Aplikasi

Aplikasi dapat dijalankan melalui CLI:

#### Mode Headless (Default / Production Mode)
Menjalankan seluruh pipeline kamera di latar belakang dan mengaktifkan Web Dashboard:
```bash
python main.py
```

#### Opsi Custom Host & Port
```bash
python main.py --host 0.0.0.0 --port 8000
```

Setelah aplikasi berjalan, buka Web Dashboard pada peramban web:
- **Admin Console (Full Control & Kalibrasi)**:  
  👉 **`http://localhost:8000/dashboard_admin`**
- **Live Monitoring (View Only)**:  
  👉 **`http://localhost:8000/dashboard`**

---

## Manajemen Kamera & Kalibrasi Zona Interaktif

Sistem dilengkapi antarmuka kalibrasi zona visual tanpa perlu menulis kode atau me-restart aplikasi:

### 1. Kalibrasi Visual Zona & Garis (ROI Builder)
1. Buka dashboard di `http://localhost:8000/dashboard_admin`.
2. Pilih kamera pada mode **Single View**, lalu klik tombol **Kalibrasi Zona**.
3. Di kanvas interaktif:
   - **Zona Poligon**: Klik untuk menambah titik sudut area pantau. Geser titik sudut untuk menyesuaikan batas fisik.
   - **Garis Tripwire**: Tentukan titik $P_1$ dan $P_2$ untuk garis perlintasan virtual serta arah lintasan (*A ke B*, *B ke A*, atau *Dua Arah*).
4. Di panel drawer properti samping:
   - Atur batas **Dwell Time** (detik toleransi).
   - Pilih **Target Kelas Pelanggaran**: centang kombinasi `Truk`, `Bus`, `Mobil`, atau `Orang`.
5. Klik **Simpan**. Konfigurasi langsung diterapkan secara atomik (*hot-reload*) ke pipeline AI tanpa memutus stream.

### 2. Menambah Kamera Baru
1. Buat folder baru di `cameras/` (contoh: `cameras/cam_06/`).
2. Buat file `config.json` dan `roi_zones.json`.
3. Atau gunakan tombol **Tambah Kamera** pada Web Dashboard untuk menambahkan kamera secara instan.

---

## Pengujian & Verifikasi Otomatis

Proyek ini dilengkapi serangkaian pengujian unit test dan regresi:
```bash
# Menjalankan seluruh pengujian fungsional pipeline
.venv\Scripts\python.exe -m pytest tools/test_dynamic_camera_and_calibration.py tools/test_cam04_and_tripwire_foot.py -v

# Menjalankan verifikasi render bounding box orang & persistensi kelas zona
.venv\Scripts\python.exe -m pytest tools/test_person_box_and_zone_classes.py -v

# Menjalankan benchmark komparasi model deteksi orang
.venv\Scripts\python.exe tools/test_pose_comparison.py
```

---

## Dokumentasi Terkait
- Dokumen Desain Teknis Lengkap: `docs/Rancangan_Smart_CCTV_Hybrid.md`
- Roadmap Arsitektur Multi-Process VMS: [docs/ROADMAP_MULTIPROCESS_VMS.md](file:///d:/Project/smart-cctv/docs/ROADMAP_MULTIPROCESS_VMS.md)
