# Model Weights Directory

Direktori ini digunakan untuk menyimpan bobot model machine learning:

1. `model_training.h5` (122.6 MB): Model pra-latih Attention U-Net Keras dari dataset Kaggle Flood Area Segmentation.
2. `model.onnx` (~40.8 MB): Model teroptimasi hasil konversi ONNX untuk inferensi CPU berkecepatan tinggi.

### Petunjuk Penyiapan Model:
1. Letakkan berkas `model_training.h5` di dalam direktori `models/` ini.
2. Jalankan konversi ke format ONNX:
   ```bash
   python backend/convert_onnx.py
   ```
3. Berkas `model.onnx` akan terbentuk secara otomatis dan siap digunakan oleh backend FastAPI.
