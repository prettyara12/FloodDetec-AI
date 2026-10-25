"""
Unified Inference Engine for Flood Area Segmentation
Uses ONNX Runtime for ultra-fast, lightweight CPU inference.
"""

import os
from typing import Optional
import numpy as np
import onnxruntime as ort
from backend.convert_onnx import export_h5_to_onnx


class FloodInferenceEngine:
    def __init__(self, onnx_model_path: str = "models/model.onnx", h5_fallback_path: str = "models/model_training.h5"):
        self.onnx_model_path = onnx_model_path
        self.h5_fallback_path = h5_fallback_path
        self.session: Optional[ort.InferenceSession] = None
        self._initialize_session()

    def _initialize_session(self):
        """Ensures the ONNX model is available and loads it into an InferenceSession."""
        if not os.path.exists(self.onnx_model_path):
            # Attempt export if H5 file exists
            candidate_h5 = self.h5_fallback_path
            if not os.path.exists(candidate_h5):
                candidate_h5 = r"D:\Mata Kuliah\Artificial Intellegence\model_training.h5"

            if os.path.exists(candidate_h5):
                export_h5_to_onnx(candidate_h5, self.onnx_model_path)
            else:
                raise FileNotFoundError(
                    f"Neither ONNX model ({self.onnx_model_path}) nor HDF5 source ({self.h5_fallback_path}) exists."
                )

        # Initialize ONNX Runtime session on CPU
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_tensor: np.ndarray) -> np.ndarray:
        """
        Runs segmentation inference on preprocessed tensor.
        Input: shape (1, 256, 256, 3) or (256, 256, 3) float32 [0.0, 1.0].
        Output: probability map shape (256, 256) float32 in range [0.0, 1.0].
        """
        if image_tensor.ndim == 3:
            image_tensor = np.expand_dims(image_tensor, axis=0)

        if image_tensor.shape != (1, 256, 256, 3):
            raise ValueError(f"Expected input shape (1, 256, 256, 3), got {image_tensor.shape}")

        if image_tensor.dtype != np.float32:
            image_tensor = image_tensor.astype(np.float32)

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: image_tensor})
        raw_output = outputs[0]  # shape (1, 256, 256, 1)

        # Squeeze to 2D probability map (256, 256)
        prob_map = np.squeeze(raw_output).astype(np.float32)
        # Ensure values are strictly bounded [0.0, 1.0]
        return np.clip(prob_map, 0.0, 1.0)
