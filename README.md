# FloodDetec AI: Deteksi Banjir Berbasis AI

Sistem segmentasi semantik berbasis kecerdasan buatan (Artificial Intelligence) untuk mendeteksi batas genangan air dan menghitung luasan banjir secara otomatis. Sistem ini ditenagai oleh model deep learning **Attention U-Net** yang dilatih menggunakan dataset Kaggle Flood Area Segmentation (`model_training.h5`).

---

## 📚 Dokumentasi Proyek Lengkap

Seluruh dokumen analisis, spesifikasi kebutuhan, dan desain arsitektur sistem telah disusun secara rinci pada direktori `docs/`:

1. [**Product Requirements Document (PRD)**](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/docs/PRD.md):
   - Pernyataan masalah, user personas, tujuan proyek, dan metrik keberhasilan.
   - Kebutuhan fungsional (FR-1 s.d. FR-8) dan kebutuhan non-fungsional.
   - Matriks klasifikasi tingkat risiko banjir (Aman, Waspada, Siaga, Awas).
2. [**System Design & Architecture Document (SDD)**](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/docs/SYSTEM_DESIGN.md):
   - Arsitektur sistem client-server ringan (FastAPI + Vanilla HTML/CSS/JS).
   - Dekomposisi komponen, pipeline data pra/pasca-pemrosesan, dan spesifikasi REST API.
   - Analisis matematika arsitektur Attention Gate dan formulasi koefisien atensi $\psi$.
3. [**ML Model Card: Attention U-Net**](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/docs/MODEL_CARD.md):
   - Rincian lapisan jaringan hasil pembongkaran biner berkas `model_training.h5` (122.6 MB).
   - Struktur 4-tahap Encoder, Bottleneck 512 kanal, 4-tahap Attention Gate, dan Decoder.
   - Konfigurasi optimizer Adam (lr=0.001) dan metrik evaluasi IoU / Dice Coefficient.
4. [**Project Structure & Implementation Guide**](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/docs/PROJECT_STRUCTURE.md):
   - Tata letak direktori proyek minimalis (`ponytail` pattern).
   - Panduan instalasi dependensi, persiapan virtual environment, dan eksekusi server FastAPI.
5. [**Design Direction & Anti-Slop Specification**](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/DESIGN.md):
   - Panduan tema visual konsol analitik modern yang bersih dan profesional.
   - Standar aksesibilitas WCAG AA, penanganan status UI, dan pemenuhan filter anti-slop.

---

## ⚡ Ringkasan Cepat Arsitektur Model

- **Input Dimension**: `(1, 256, 256, 3)` RGB float32 `[0.0, 1.0]`
- **Backbone**: 4 Encoder Blocks (32 -> 64 -> 128 -> 256 filters) dengan MaxPooling2D
- **Bottleneck**: 512 filters dengan Dropout 0.3
- **Attention Mechanism**: 4 Attention Gates dengan aktivasi Sigmoid untuk menyaring skip connection
- **Decoder**: 4 Decoder Blocks dengan 2x2 Upsampling dan penggabungan kanal fitur
- **Output Head**: `Conv2D(1, 1x1, activation='sigmoid')` menghasilkan probability map `(256, 256, 1)`
- **Inference Strategy**: Dual-Support (Native TensorFlow/Keras & High-speed ONNX Runtime)
