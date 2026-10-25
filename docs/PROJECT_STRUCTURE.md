# Project Structure & Implementation Guide
## Flood Detection Area Segmentation System (FloodDetec AI)

- **Document Version**: 1.0.0
- **Status**: Approved
- **Runtime Environment**: Python 3.10+ / 3.12 (Windows 11)

---

## 1. Tata Letak Direktori (Directory Layout)

Mengikuti prinsip minimalis dan efisiensi (`ponytail` pattern: *fewest files possible, zero unrequested abstractions*), tata letak proyek disusun tanpa lapisan yang berlebihan:

```
Flood_Area_Project/
├── backend/
│   ├── app.py                 # FastAPI application, CORS, static mounting, dan endpoints
│   ├── model_builder.py       # Definisi Attention U-Net Keras untuk memuat model_training.h5
│   ├── convert_onnx.py        # Skrip konversi satu kali dari Keras H5 ke ONNX
│   ├── inference.py           # Unified inference wrapper (ONNXRuntime prioritas, TF fallback)
│   ├── preprocessor.py        # Decoding citra, resize 256x256, dan normalisasi
│   └── postprocessor.py       # Thresholding, kalkulasi % luas, overlay kontur cyan, Base64 encode
├── frontend/
│   ├── index.html             # Antarmuka Single Page Application (Semantic HTML5)
│   ├── style.css              # Custom styling anti-slop (tema slate navy, aksen cyan & status)
│   └── app.js                 # Event handler drag-and-drop, slider interaktif, dan fetch API
├── samples/                   # Kumpulan foto sampel banjir untuk uji coba instan
│   ├── sample_01.jpg          # Sampel banjir pemukiman
│   ├── sample_02.jpg          # Sampel genangan kawasan pertanian
│   ├── sample_03.jpg          # Sampel luapan sungai dan jembatan
│   └── sample_04.jpg          # Sampel daratan kering / non-banjir (uji negatif)
├── models/                    # Direktori penyimpanan bobot model
│   ├── model_training.h5      # Tautan / salinan model pra-latih (122.6 MB)
│   └── model.onnx             # Model teroptimasi hasil konversi ONNX (122 MB)
├── docs/                      # Berkas dokumentasi arsitektur dan persyaratan
│   ├── PRD.md                 # Product Requirements Document
│   ├── SYSTEM_DESIGN.md       # System Design & Architecture Document
│   ├── MODEL_CARD.md          # Spesifikasi Teknis Model Attention U-Net
│   └── PROJECT_STRUCTURE.md   # Panduan Struktur Direktori & Eksekusi Proyek
├── DESIGN.md                  # Panduan Desain Visual, Tokens, dan Anti-Slop Dials
├── requirements.txt           # Daftar dependensi Python
└── README.md                  # Panduan Memulai Cepat (Quickstart Guide)
```

---

## 2. Dependensi Proyek (`requirements.txt`)

Dependensi dipilih secara selektif agar instalasi cepat dan hemat ruang disk:

```text
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
python-multipart>=0.0.9
numpy>=1.26.0
opencv-python>=4.9.0.80
pillow>=10.2.0
onnxruntime>=1.17.0
h5py>=3.10.0
# tensorflow>=2.15.0  (Opsional, hanya diperlukan saat menjalankan konversi awal dari .h5 ke .onnx)
```

---

## 3. Alur Kerja Implementasi dan Eksekusi

### Langkah 1: Persiapan Lingkungan Virtual (Virtual Environment)
Buka terminal PowerShell pada direktori proyek `Flood_Area_Project`:
```powershell
# 1. Buat virtual environment lokal
python -m venv venv

# 2. Aktifkan virtual environment
.\venv\Scripts\Activate.ps1

# 3. Perbarui pip dan pasang dependensi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Langkah 2: Menyiapkan Berkas Model
Salin atau buat symbolic link dari berkas `model_training.h5` ke dalam folder `models/`:
```powershell
# Salin berkas model dari direktori induk
Copy-Item "..\model_training.h5" ".\models\model_training.h5"
```

### Langkah 3: Ekspor Model ke Format ONNX (Satu Kali Eksekusi)
Jalankan skrip konverter untuk menghasilkan graf ONNX yang sangat cepat dan ringan:
```powershell
python backend/convert_onnx.py
```
*Catatan*: Setelah berkas `models/model.onnx` terbentuk, backend FastAPI dapat berjalan secara mandiri hanya dengan pustaka `onnxruntime` tanpa perlu memuat TensorFlow sama sekali (menghemat RAM hingga 700 MB dan memangkas waktu startup dari 15 detik menjadi di bawah 1 detik).

### Langkah 4: Menjalankan Server Web FastAPI
Mulai server pengembangan lokal:
```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Akses antarmuka web melalui browser:
`http://127.0.0.1:8000`

---

## 4. Daftar Periksa Verifikasi Sistem (Verification Checklist)

1. [ ] **Verifikasi Pemuatan Model**:
   - Jalankan `GET http://127.0.0.1:8000/api/health`.
   - Pastikan mengembalikan `{"status": "healthy", "model_loaded": true}`.
2. [ ] **Verifikasi Citra Sampel**:
   - Klik kartu sampel pada antarmuka.
   - Pastikan citra sampel langsung dimuat ke panel perbandingan tanpa jeda.
3. [ ] **Verifikasi Inferensi Citra Baru**:
   - Unggah foto banjir format PNG/JPG melalui drag-and-drop.
   - Periksa apakah masker banjir berwarna cyan muncul dengan garis kontur yang jelas.
4. [ ] **Verifikasi Slider Threshold**:
   - Geser slider ambang batas dari 0.20 ke 0.80.
   - Periksa pembaruan persentase luas banjir secara real-time.
5. [ ] **Verifikasi Aksesibilitas Keyboard**:
   - Gunakan tombol `Tab` untuk berpindah antarelemen interaktif.
   - Gunakan tombol panah kiri/kanan untuk menggeser pembatas split-slider.
