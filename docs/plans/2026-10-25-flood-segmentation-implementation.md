# Flood Detection Area Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun aplikasi web interaktif segmentasi area banjir menggunakan model deep learning Attention U-Net berbasis FastAPI dan Vanilla JavaScript modern.

**Architecture:** Arsitektur client-server monolitik ringan (decoupled single-page web app). Backend FastAPI menjalankan pra-pemrosesan citra, inferensi tensor Attention U-Net (prioritas ONNX Runtime dengan fallback TensorFlow), kalkulasi metrik luasan genangan air, dan penyajian antarmuka statis. Frontend menyajikan konsol analitik interaktif dengan split-slider perbandingan, slider sensitivitas threshold, dan kartu klasifikasi keparahan bencana.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, NumPy, OpenCV, ONNX Runtime, Keras / TensorFlow, Vanilla HTML5, Modern CSS, Vanilla ES6 JavaScript.

**Spec:** [System Design Document](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/docs/SYSTEM_DESIGN.md), [PRD](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/docs/PRD.md), [Model Card](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/docs/MODEL_CARD.md), dan [Design Direction](file:///d:/Mata%20Kuliah/Artificial%20Intellegence/Flood_Area_Project/DESIGN.md).

## Global Constraints

- Tensor input harus berdimensi tepat `(1, 256, 256, 3)` dengan normalisasi nilai piksel `[0.0, 1.0]`.
- Tensor output model menghasilkan probabilitas sigmoid biner berdimensi `(1, 256, 256, 1)`.
- Batas ambang default binerisasi adalah `0.50`, dapat disesuaikan pada rentang `0.10` hingga `0.90`.
- Klasifikasi keparahan banjir: Aman (<5.0%), Waspada (5.0%-19.9%), Siaga (20.0%-49.9%), Awas (>=50.0%).
- Frontend tidak boleh menggunakan dependensi Node.js/npm; wajib Vanilla HTML5, CSS murni, dan ES6 JavaScript.
- Tampilan harus memenuhi standar `DESIGN.md` (Slate Navy `#0B132B`, Surface `#1C2541`, Cyan `#00F0FF`, WCAG AA rasio kontras >= 4.5:1).
- Larangan aturan R-02 anti-slop: tidak boleh memuat karakter em dash (`—`) pada teks aplikasi dan antarmuka.

## Review Focus

1. **Format Citra Masukan**: Pengunggahan citra RGBA (4 channel dengan transparansi) atau Grayscale (1 channel) harus dikonversi ke RGB 3-channel tanpa error `ValueError: shape mismatch`.
2. **Pemuatan Bobot HDF5**: Rekonstruksi model Attention U-Net harus memetakan bobot dari `model_training.h5` secara presisi tanpa ketergantungan class custom layer Keras lama.
3. **Kinerja Latensi CPU**: Inferensi ONNX Runtime pada CPU harus selesai dalam waktu kurang dari 1200 ms untuk citra 256x256.
4. **Respon Threshold Real-Time**: Perubahan slider threshold tidak boleh memicu inferensi ulang neural network; sistem harus mengolah ulang matriks probabilitas yang telah dicache.
5. **Ketahanan UI**: Seluruh status antarmuka (Empty State, Loading Spinner, Error Alert, Success Comparison) harus teruji dan dapat diakses penuh via keyboard.

---

### Task 1: Environment Dependencies & Model Reconstruction (`model_builder.py`)

**Files:**
- Create: `requirements.txt`
- Create: `backend/model_builder.py`
- Create: `tests/test_model_builder.py`

**Interfaces:**
- Produces: `build_attention_unet(input_shape=(256, 256, 3)) -> keras.Model`
- Produces: `load_trained_weights(model: keras.Model, h5_path: str) -> keras.Model`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_builder.py
import pytest
import os
import numpy as np

def test_build_attention_unet_shape():
    from backend.model_builder import build_attention_unet
    model = build_attention_unet(input_shape=(256, 256, 3))
    assert model.input_shape == (None, 256, 256, 3)
    assert model.output_shape == (None, 256, 256, 1)

def test_load_weights_from_h5():
    from backend.model_builder import build_attention_unet, load_trained_weights
    h5_path = r"D:\Mata Kuliah\Artificial Intellegence\model_training.h5"
    if os.path.exists(h5_path):
        model = build_attention_unet(input_shape=(256, 256, 3))
        loaded_model = load_trained_weights(model, h5_path)
        dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
        out = loaded_model.predict(dummy_input)
        assert out.shape == (1, 256, 256, 1)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend'`

- [ ] **Step 3: Implement dependencies and `model_builder.py`**

Buat `requirements.txt` dan implementasikan rekonstruksi Attention U-Net menggunakan Functional API Keras standar (Conv2D, BatchNormalization, ReLU, Add, Multiply, UpSampling2D, Concatenate, Sigmoid) serta pemuatan bobot layer-by-layer dari HDF5 menggunakan `h5py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_builder.py -v`
Expected: PASS

---

### Task 2: ONNX Model Export & Unified Inference Engine (`inference.py`)

**Files:**
- Create: `backend/convert_onnx.py`
- Create: `backend/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `build_attention_unet` dan `load_trained_weights` dari `backend.model_builder`
- Produces: `class FloodInferenceEngine` dengan metode `predict(image_tensor: np.ndarray) -> np.ndarray` (mengembalikan matriks probabilitas `(256, 256)` float32)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
import pytest
import numpy as np

def test_inference_engine_prediction():
    from backend.inference import FloodInferenceEngine
    engine = FloodInferenceEngine()
    dummy_tensor = np.random.uniform(0.0, 1.0, (1, 256, 256, 3)).astype(np.float32)
    prob_map = engine.predict(dummy_tensor)
    assert prob_map.shape == (256, 256)
    assert prob_map.dtype == np.float32
    assert float(np.min(prob_map)) >= 0.0
    assert float(np.max(prob_map)) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.inference'`

- [ ] **Step 3: Implement `backend/convert_onnx.py` & `backend/inference.py`**

Implementasikan `convert_onnx.py` untuk mengonversi model Attention U-Net ke `models/model.onnx`. Implementasikan `FloodInferenceEngine` yang memuat `models/model.onnx` via `onnxruntime.InferenceSession` (dengan fallback otomatis ke Keras jika berkas ONNX belum ada).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference.py -v`
Expected: PASS

---

### Task 3: Image Preprocessor & Postprocessor (`preprocessor.py`, `postprocessor.py`)

**Files:**
- Create: `backend/preprocessor.py`
- Create: `backend/postprocessor.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces `preprocess_image_bytes(image_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]` (mengembalikan tensor `(1, 256, 256, 3)`, ukuran asli `(height, width)`, dan citra RGB uint8 berukuran 256x256).
- Produces `postprocess_flood_mask(prob_map: np.ndarray, orig_rgb: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]` (menghasilkan persentase luas, piksel tergenang, status risiko, dan Base64 Data URL untuk masker biner serta overlay cyan).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import pytest
import numpy as np
import cv2

def test_preprocessing_flow():
    from backend.preprocessor import preprocess_image_bytes
    # Create fake RGB image
    img = np.full((500, 700, 3), 120, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    tensor, orig_size, rgb_256 = preprocess_image_bytes(buf.tobytes())
    assert tensor.shape == (1, 256, 256, 3)
    assert tensor.dtype == np.float32
    assert orig_size == (500, 700)
    assert rgb_256.shape == (256, 256, 3)

def test_postprocessing_metrics_and_severity():
    from backend.postprocessor import postprocess_flood_mask
    prob_map = np.zeros((256, 256), dtype=np.float32)
    # Simulate 30% flood coverage
    prob_map[:77, :] = 0.85
    orig_rgb = np.full((256, 256, 3), 200, dtype=np.uint8)
    res = postprocess_flood_mask(prob_map, orig_rgb, threshold=0.5)
    assert 29.0 <= res["flood_percentage"] <= 31.0
    assert res["risk_level"] == "Siaga (Medium Risk)"
    assert res["risk_color"] == "#F59E0B"
    assert "binary_mask_url" in res["images"]
    assert "overlay_url" in res["images"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.preprocessor'`

- [ ] **Step 3: Implement `backend/preprocessor.py` & `backend/postprocessor.py`**

Implementasikan pembacaan citra yang mendukung 1/3/4 channel menjadi RGB 3-channel, resize bilinear 256x256, normalisasi `[0, 1]`. Implementasikan thresholding biner, kalkulasi persentase luas, klasifikasi status risiko sesuai PRD, penggambaran overlay Electric Cyan (`#00F0FF`, alpha 40%), serta encoding Base64 PNG.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

---

### Task 4: FastAPI Server & REST API Endpoints (`app.py`)

**Files:**
- Create: `backend/app.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `FloodInferenceEngine` dari `backend.inference`
- Consumes: `preprocess_image_bytes` dari `backend.preprocessor`
- Consumes: `postprocess_flood_mask` dari `backend.postprocessor`
- Produces: REST endpoints `GET /api/health`, `GET /api/samples`, `POST /api/segment`, `POST /api/recalculate`, serta static file mounting untuk folder `frontend` dan `samples`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2

def test_api_health_endpoint():
    from backend.app import app
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "engine" in data

def test_api_segment_endpoint():
    from backend.app import app
    client = TestClient(app)
    img = np.full((256, 256, 3), 150, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    files = {"file": ("test.png", buf.tobytes(), "image/png")}
    response = client.post("/api/segment", files=files, data={"threshold": "0.5"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "flood_percentage" in data["data"]
    assert "risk_level" in data["data"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.app'`

- [ ] **Step 3: Implement `backend/app.py`**

Bangun aplikasi FastAPI dengan CORS middleware, mounting static frontend, penanganan multipart upload dengan validasi tipe berkas dan batas 10 MB, endpoint segmentasi citra, dan endpoint rekalkulasi ambang batas instan.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS

---

### Task 5: Sample Dataset Preparation (`samples/`)

**Files:**
- Create: `samples/sample_01.jpg`
- Create: `samples/sample_02.jpg`
- Create: `samples/sample_03.jpg`
- Create: `samples/sample_04.jpg`
- Create: `samples/metadata.json`

**Interfaces:**
- Produces: 4 citra sampel uji coba validasi (pemukiman banjir, genangan pertanian, bantaran sungai, dan daratan kering) beserta metadata deskriptifnya.

- [ ] **Step 1: Extract or synthesize sample test images**

Sediakan 4 citra sampel representatif berdimensi 256x256 atau lebih dengan format JPG di dalam direktori `samples/` dan daftarkan pada `samples/metadata.json`.

- [ ] **Step 2: Verify sample files accessibility via API**

Run: `pytest tests/test_api.py -k test_api_samples -v`
Expected: PASS mengembalikan 4 item sampel dengan URL yang valid.

---

### Task 6: Frontend Interface & Anti-Slop UI Implementation (`frontend/`)

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/style.css`
- Create: `frontend/app.js`

**Interfaces:**
- Consumes: REST API `/api/segment`, `/api/samples`, `/api/recalculate`, `/api/health`.
- Implements: Standar visual `DESIGN.md` (Slate Navy, Electric Cyan, Split-slider perbandingan, Keyboard accessibility, dan 3 mandatory states: Empty, Loading, Error).

- [ ] **Step 1: Implement semantic HTML5 layout (`index.html`)**

Bangun struktur halaman yang memuat Header navigasi, Dropzone area dengan input file, Gallery sampel cepat, Panel hasil perbandingan citra dengan elemen split-canvas/slider, Panel metrik keparahan bencana, dan Slider threshold interaktif.

- [ ] **Step 2: Implement styling anti-slop modern (`style.css`)**

Terapkan design token dari `DESIGN.md`: palet slate navy (`#0B132B`, `#1C2541`, `#3A506B`), teks kontras tinggi WCAG AA (`#F1F5F9`), aksen cyan (`#00F0FF`), indikator fokus `:focus-visible` 2px yang jelas, tap targets >= 44px, serta layout responsif dari mobile hingga desktop tanpa overflow horizontal.

- [ ] **Step 3: Implement client-side logic & interaction (`app.js`)**

Tulis kode JavaScript murni untuk:
1. Drag-and-drop file upload dengan feedback visual.
2. Penanganan klik pada kartu galeri sampel.
3. Split-slider interaktif (mouse drag & tombol keyboard Left/Right untuk menggeser pembatas Before/After).
4. Penanganan perubahan slider threshold secara responsif.
5. Manajemen ketiga status UI (Empty State, Loading state dengan status indikator, dan Error message alert).

- [ ] **Step 4: Verify UI compliance and interaction**

Pastikan seluruh elemen interaktif berfungsi normal, konsol browser bersih dari error, dan tema visual konsisten.

---

### Task 7: End-to-End System Integration & Verification

**Files:**
- Modify: `tests/test_e2e.py`

**Interfaces:**
- Runs full system pipeline from server boot to image upload, inference, metric rendering, and threshold recalculation.

- [ ] **Step 1: Write and run end-to-end integration test**

```python
# tests/test_e2e.py
import pytest
from fastapi.testclient import TestClient
import os

def test_full_pipeline_with_sample():
    from backend.app import app
    client = TestClient(app)
    sample_path = "samples/sample_01.jpg"
    assert os.path.exists(sample_path)
    with open(sample_path, "rb") as f:
        response = client.post("/api/segment", files={"file": ("sample_01.jpg", f.read(), "image/jpeg")})
    assert response.status_code == 200
    res_data = response.json()["data"]
    assert "flood_percentage" in res_data
    assert "risk_level" in res_data
    assert len(res_data["images"]["overlay_url"]) > 100
```

- [ ] **Step 2: Execute all unit and integration tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 3: Verification report**

Verifikasi jalannya server lokal dan antarmuka web pada `http://127.0.0.1:8000`.
