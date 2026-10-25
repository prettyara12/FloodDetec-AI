# Product Requirements Document (PRD)
## Flood Detection Area Segmentation System (FloodDetec AI)

- **Document Version**: 1.0.0
- **Status**: Approved for Architecture & Implementation
- **Target Audience**: AI Course Evaluators, Software Engineers, Disaster Mitigation Analysts
- **Model Reference**: Attention U-Net (`model_training.h5`, 122.6 MB, TensorFlow/Keras)

---

## 1. Latar Belakang dan Pernyataan Masalah

Bencana banjir memerlukan estimasi luasan area terdampak secara cepat dan akurat untuk memandu respon evakuasi, penyaluran logistik, dan penilaian kerugian infrastruktur. Metode inspeksi manual secara konvensional memakan waktu hitungan jam hingga hari.

Proyek ini memanfaatkan model Deep Learning berbasis arsitektur **Attention U-Net** yang telah dilatih pada dataset segmentasi banjir (Kaggle Flood Area Segmentation dataset). Sistem ini mengotomatisasi segmentasi biner area air/banjir beresolusi 256x256 piksel, menghitung rasio tutupan air secara kuantitatif, serta memetakan tingkat keparahan risiko banjir ke dalam antarmuka web interaktif yang intuitif dan responsif.

---

## 2. Tujuan Proyek dan Kriteria Sukses

### 2.1 Tujuan Utama
1. **Otomatisasi Segmentasi**: Mengubah foto masukan area banjir (format JPG/PNG) menjadi peta tutupan genangan banjir biner dalam waktu kurang dari 1.5 detik per citra pada CPU standar.
2. **Kalkulasi Metrik Kuantitatif**: Menghitung persentase luasan banjir relatif terhadap keseluruhan bidang pandang citra, lengkap dengan klasifikasi tingkat risiko bertingkat.
3. **Visualisasi Interaktif**: Menyediakan antarmuka web konsol dengan slider perbandingan visual (Original vs Mask Overlay) dan pengatur sensitivitas probabilitas (threshold slider).
4. **Efisiensi Sistem**: Memastikan runtime inferensi ringan, dapat berjalan lokal tanpa ketergantungan framework berat atau pipeline kompilasi Node.js/npm.

### 2.2 Kriteria Sukses (Success Metrics)
- **Akurasi Inferensi**: Menghasilkan segmentasi biner yang konsisten dengan bobot model Attention U-Net pra-latih (akurasi biner > 90% pada dataset validasi).
- **Latensi Respon**: Waktu pemrosesan inferensi backend (termasuk resize, inferensi tensor, dan masking kontur) <= 1200 ms pada CPU komputer desktop/laptop standar.
- **Kepatuhan Aksesibilitas**: Antarmuka web mematuhi standar WCAG AA (rasio kontras teks minimal 4.5:1, navigasi penuh via keyboard).
- **Stabilitas Pengujian**: 100% penanganan error yang anggun (graceful degradation) untuk file non-citra, citra berdimensi ekstrem, atau koneksi terputus.

---

## 3. Target Pengguna (User Personas)

1. **Analis Penanggulangan Bencana (BPBD / SAR)**
   - *Kebutuhan*: Memerlukan data cepat mengenai estimasi persentase wilayah yang tergenang untuk menentukan zona prioritas evakuasi.
   - *Pain Point*: Tidak memiliki latar belakang teknis pemrograman untuk menjalankan skrip Python secara manual di terminal.
2. **Dosen / Penilai Akademik Mata Kuliah AI**
   - *Kebutuhan*: Mengevaluasi pipeline arsitektur model, verifikasi bobot `.h5`, kesesuaian implementasi inferensi, dan fungsionalitas antarmuka sistem.
   - *Pain Point*: Seringkali menemukan proyek AI yang hanya berjalan di Jupyter Notebook tanpa antarmuka praktis yang siap diuji.
3. **Mahasiswa / Peneliti Machine Learning**
   - *Kebutuhan*: Menguji ketahanan model segmentasi pada berbagai ambang batas (threshold), mengamati pengaruh bobot perhatian (attention gates), dan mengekspor hasil segmentasi.

---

## 4. Kebutuhan Fungsional (Functional Requirements)

| ID | Modul | Deskripsi Kebutuhan | Prioritas |
| :--- | :--- | :--- | :--- |
| **FR-1** | Input Citra | Sistem menyediakan zona Drag-and-Drop serta pemilih file lokal untuk berkas format JPG, JPEG, dan PNG dengan batas ukuran maksimal 10 MB. | Tinggi |
| **FR-2** | Galeri Sampel | Sistem menyediakan setidaknya 4 sampel citra bawaan dari dataset validasi agar pengguna dapat langsung menguji sistem dengan 1 klik tanpa perlu menyiapkan berkas sendiri. | Tinggi |
| **FR-3** | Pipeline Inferensi | Sistem melakukan pra-pemrosesan citra ke dimensi tensor `(1, 256, 256, 3)`, menormalkan nilai piksel `[0, 1]`, dan mengeksekusi inferensi model Attention U-Net. | Kritis |
| **FR-4** | Kontrol Threshold | Sistem menyediakan slider interaktif dengan rentang 0.10 hingga 0.90 (default 0.50, step 0.05) untuk mengatur ambang batas probabilitas binerisasi piksel air. | Tinggi |
| **FR-5** | Visualisasi Overlay | Sistem menampilkan visualisasi hasil dengan 3 mode: (a) Split-slider interaktif (Before/After), (b) Mask kontur cyan semi-transparan di atas citra asli, dan (c) Mask biner murni hitam-putih. | Tinggi |
| **FR-6** | Kalkulasi Metrik | Sistem menghitung total piksel air, total piksel citra, dan persentase luas genangan: `(flood_pixels / total_pixels) * 100%`. | Kritis |
| **FR-7** | Klasifikasi Risiko | Sistem mengkategorikan status keparahan bencana berdasarkan luasan genangan: Aman (<5%), Waspada (5%-20%), Siaga (20%-50%), dan Awas (>50%). | Tinggi |
| **FR-8** | Ekspor Hasil | Pengguna dapat mengunduh citra hasil segmentasi beserta ringkasan laporan metrik dalam format PNG. | Sedang |

---

## 5. Kebutuhan Non-Fungsional (Non-Functional Requirements)

### 5.1 Performa dan Skalabilitas
- Waktu inferensi pada CPU i5/Ryzen 5 generasi modern tidak melebihi 1.5 detik per gambar.
- Ukuran bundle frontend web (HTML + CSS + JS) di bawah 300 KB tanpa ketergantungan library pihak ketiga yang besar.
- Penggunaan memori RAM backend dijaga di bawah 600 MB (dengan mode ONNX Runtime).

### 5.2 Keandalan dan Penanganan Error
- Berkas yang diunggah divalidasi tipe MIME dan ekstensinya sebelum masuk ke pipeline tensor.
- Citra berukuran lebih dari 256x256 secara otomatis di-resampling secara proporsional dan dikembalikan ke rasio aslinya pada tampilan overlay.
- Sistem menyediakan pesan kesalahan yang informatif saat file korup atau server tidak dapat memuat bobot model.

### 5.3 Aksesibilitas dan Desain (Anti-Slop Compliance)
- Mematuhi panduan `DESIGN.md`: tema konsol analitik slate navy yang terfokus pada konten.
- Tidak menggunakan elemen dekoratif berlebih yang tidak memiliki tujuan fungsional.
- Mendukung penuh navigasi keyboard (Tab, Shift+Tab, Enter, Space, Arrow keys untuk slider).
- Memiliki 3 status tampilan yang lengkap: Empty State, Loading State, dan Error State.

---

## 6. Batasan Sistem (Constraints & Assumptions)

### In-Scope (Termasuk dalam Cakupan):
- Foto atau citra masukan format RGB 3-channel (JPG/PNG).
- Segmentasi biner 1 kelas (Banjir / Tergenang Air vs Bukan Banjir).
- Backend berbasis Python FastAPI dengan opsi eksekusi ganda: TensorFlow dan ONNX Runtime.
- Frontend Single Page Application (SPA) berbasis Vanilla HTML5, Modern CSS, dan JavaScript murni.

### Out-of-Scope (Di Luar Cakupan):
- Segmentasi multi-kelas (misalnya memisahkan air sungai alami vs genangan perumahan vs sawah).
- Pemrosesan format raster geospasial bertingkat multispektral (seperti 12-band Sentinel GeoTIFF dengan koordinat CRS kompleks).
- Sistem autentikasi pengguna atau integrasi database multi-user (karena sistem dirancang sebagai alat inspeksi instan).

---

## 7. Skema Klasifikasi Tingkat Risiko (Severity Matrix)

Kategori risiko ditentukan berdasarkan proporsi luas genangan hasil segmentasi biner:

| Rentang Persentase Genangan | Level Status | Indikator Warna | Tindakan Rekomendasi Sistem |
| :--- | :--- | :--- | :--- |
| **0.0% - 4.9%** | **Aman (Normal)** | Emerald (`#10B981`) | Kondisi perairan normal / drainase terkendali. Tidak ada indikasi bahaya banjir signifikan. |
| **5.0% - 19.9%** | **Waspada (Low Risk)** | Cobalt (`#3B82F6`) | Genangan lokal terdeteksi di beberapa titik. Pemantauan berkala direkomendasikan. |
| **20.0% - 49.9%** | **Siaga (Medium Risk)** | Amber (`#F59E0B`) | Genangan signifikan menutupi sebagian area daratan. Persiapan mitigasi dan evakuasi dini. |
| **>= 50.0%** | **Awas (High Risk)** | Crimson (`#EF4444`) | Banjir besar meluas. Akses darat terputus. Tindakan darurat dan evakuasi segera diperlukan. |
