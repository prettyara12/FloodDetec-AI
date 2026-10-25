# Machine Learning Model Card: Attention U-Net
## Model Specification for Flood Detection Area Segmentation

- **Model Identifier**: `attention-unet-flood-seg-v1`
- **File Source**: `model_training.h5` (122,684,272 bytes / 122.6 MB)
- **Primary Task**: Binary Semantic Segmentation (Water/Flood vs Terrestrial Background)
- **Input Specifications**: RGB Flood Image, tensor shape `(1, 256, 256, 3)`, float32, range `[0.0, 1.0]`
- **Output Specifications**: Binary Probability Map, tensor shape `(1, 256, 256, 1)`, float32, range `[0.0, 1.0]`

---

## 1. Ringkasan Model (Model Summary)

Model ini mengimplementasikan arsitektur **Attention U-Net** yang dirancang untuk segmentasi semantik citra presisi tinggi. Keunggulan utama dibandingkan arsitektur U-Net konvensional terletak pada integrasi **Attention Gate (AG)** pada setiap lintasan skip-connection antara Encoder dan Decoder.

Mekanisme Attention Gate menyaring dan membobot ulang fitur spasial yang diteruskan dari encoder sebelum digabungkan dengan representasi decoder. Hal ini memungkinkan jaringan untuk menonjolkan fitur spektral air dan permukaan genangan banjir, sembari meredam respons dari elemen latar belakang yang tidak relevan (seperti pepohonan, jalan aspal kering, atap bangunan, dan bayangan awan).

---

## 2. Struktur Arsitektur dan Parameter Detail

Berdasarkan hasil pembongkaran biner dan analisis metadata HDF5 dari `model_training.h5`, berikut adalah konfigurasi lengkap lapisan jaringan:

### 2.1 Konfigurasi Lapisan Encoder
| Nama Blok | Lapisan Internal | Jumlah Filter | Dropout Rate | Operasi Pooling | Dimensi Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **InputLayer** | `input_1` | 3 (RGB) | - | Tidak | `(256, 256, 3)` |
| **Encoder 1** | `conv2d`, `conv2d_1` (3x3) | 32 | 0.1 | MaxPooling2D (2x2) | `(128, 128, 32)` |
| **Encoder 2** | `conv2d_2`, `conv2d_3` (3x3) | 64 | 0.1 | MaxPooling2D (2x2) | `(64, 64, 64)` |
| **Encoder 3** | `conv2d_4`, `conv2d_5` (3x3) | 128 | 0.2 | MaxPooling2D (2x2) | `(32, 32, 128)` |
| **Encoder 4** | `conv2d_6`, `conv2d_7` (3x3) | 256 | 0.2 | MaxPooling2D (2x2) | `(16, 16, 256)` |
| **Encoding (Bottleneck)** | `conv2d_8`, `conv2d_9` (3x3) | 512 | 0.3 | Tidak (Fitur Inti) | `(16, 16, 512)` |

### 2.2 Konfigurasi Lapisan Attention Gate & Decoder
| Blok Attention | Sinyal Gating ($g$) | Skip Input ($x$) | Lapisan Internal | Attended Feature Output |
| :--- | :--- | :--- | :--- | :--- |
| **Attention 1** | `Encoding` (512 kanal) | `Encoder 4` (256 kanal) | `conv2d_10` (3x3), `conv2d_11` (3x3), `BatchNorm`, `conv2d_12` (1x1 Sigmoid) | `(32, 32, 256)` |
| **Decoder 1** | UpSample(Encoding) + Att1 | 768 kanal total | `conv2d_13` (3x3, 256), `conv2d_14` (3x3, 256), Dropout(0.2) | `(32, 32, 256)` |
| **Attention 2** | `Decoder 1` (256 kanal) | `Encoder 3` (128 kanal) | `conv2d_15` (3x3), `conv2d_16` (3x3), `BatchNorm_1`, `conv2d_17` (1x1 Sigmoid) | `(64, 64, 128)` |
| **Decoder 2** | UpSample(Dec1) + Att2 | 384 kanal total | `conv2d_18` (3x3, 128), `conv2d_19` (3x3, 128), Dropout(0.2) | `(64, 64, 128)` |
| **Attention 3** | `Decoder 2` (128 kanal) | `Encoder 2` (64 kanal) | `conv2d_20` (3x3), `conv2d_21` (3x3), `BatchNorm_2`, `conv2d_22` (1x1 Sigmoid) | `(128, 128, 64)` |
| **Decoder 3** | UpSample(Dec2) + Att3 | 192 kanal total | `conv2d_23` (3x3, 64), `conv2d_24` (3x3, 64), Dropout(0.1) | `(128, 128, 64)` |
| **Attention 4** | `Decoder 3` (64 kanal) | `Encoder 1` (32 kanal) | `conv2d_25` (3x3), `conv2d_26` (3x3), `BatchNorm_3`, `conv2d_27` (1x1 Sigmoid) | `(256, 256, 32)` |
| **Decoder 4** | UpSample(Dec3) + Att4 | 96 kanal total | `conv2d_28` (3x3, 32), `conv2d_29` (3x3, 32), Dropout(0.1) | `(256, 256, 32)` |
| **Output Head** | Output `Decoder 4` | 32 kanal | `conv2d_30` (1x1, 1 filter, aktivasi Sigmoid) | `(256, 256, 1)` |

---

## 3. Konfigurasi Pelatihan (Training Configuration)

Sesuai rekaman metadata yang tersimpan dalam atribut H5:
- **Optimizer**: Adam
  - Learning Rate: `0.001`
  - $\beta_1$: `0.9`
  - $\beta_2$: `0.999`
  - $\epsilon$: `1e-7`
- **Loss Function**: `binary_crossentropy`
- **Evaluasi Utama**: `binary_accuracy`
- **Ukuran Batch Pelatihan**: 16 atau 32 citra per iterasi
- **Format Normalisasi**: Nilai piksel integer $[0, 255]$ diskalakan ke rentang floating-point $[0.0, 1.0]$.

---

## 4. Metrik Evaluasi Segmentasi

Untuk menguji performa segmentasi pada data validasi, metrik berikut digunakan:

1. **Akurasi Piksel Biner (Binary Accuracy)**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Intersection over Union (IoU / Jaccard Index)**:
   $$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$
3. **Dice Coefficient (F1-Score)**:
   $$\text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

---

## 5. Batasan dan Rekomendasi Operasional

- **Kondisi Pencahayaan Ekstrem**: Pantulan sinar matahari langsung (specular glare) pada atap seng/logam dapat menghasilkan respons positif palsu (false positive). Pengguna disarankan menaikkan threshold ke $0.60 - 0.70$ pada citra terik.
- **Air Keruh / Berlumpur**: Pada area banjir bandang dengan sedimen tanah pekat, kontras antara air dan daratan menjadi rendah. Pengguna disarankan menurunkan threshold ke $0.35 - 0.45$ untuk menangkap seluruh batas genangan.
- **Resolusi Spasial**: Model dilatih pada resolusi $256 \times 256$. Citra dengan resolusi sangat tinggi (misal foto 4K) akan di-downsample terlebih dahulu sebelum inferensi, kemudian hasil mask diproyeksikan kembali ke resolusi asli.
