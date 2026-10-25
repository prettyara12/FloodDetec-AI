import pytest
import numpy as np

def test_inference_engine_initialization_and_prediction():
    from backend.inference import FloodInferenceEngine
    engine = FloodInferenceEngine()
    dummy_tensor = np.random.uniform(0.0, 1.0, (1, 256, 256, 3)).astype(np.float32)
    prob_map = engine.predict(dummy_tensor)
    assert prob_map.shape == (256, 256)
    assert prob_map.dtype == np.float32
    assert float(np.min(prob_map)) >= 0.0
    assert float(np.max(prob_map)) <= 1.0
