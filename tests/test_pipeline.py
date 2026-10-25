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
    assert np.min(tensor) >= 0.0 and np.max(tensor) <= 1.0

def test_preprocessing_handles_rgba():
    from backend.preprocessor import preprocess_image_bytes
    # 4 channel RGBA image
    rgba = np.full((200, 200, 4), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".png", rgba)
    tensor, orig_size, rgb_256 = preprocess_image_bytes(buf.tobytes())
    assert tensor.shape == (1, 256, 256, 3)
    assert rgb_256.shape == (256, 256, 3)

def test_postprocessing_metrics_and_severity():
    from backend.postprocessor import postprocess_flood_mask
    prob_map = np.zeros((256, 256), dtype=np.float32)
    # Simulate ~30% flood coverage (77 rows out of 256)
    prob_map[:77, :] = 0.85
    orig_rgb = np.full((256, 256, 3), 200, dtype=np.uint8)
    res = postprocess_flood_mask(prob_map, orig_rgb, threshold=0.5)
    assert 29.0 <= res["flood_percentage"] <= 31.0
    assert res["risk_level"] == "Siaga (Medium Risk)"
    assert res["risk_color"] == "#F59E0B"
    assert "binary_mask_url" in res["images"]
    assert "overlay_url" in res["images"]
    assert res["images"]["binary_mask_url"].startswith("data:image/png;base64,")
    assert res["images"]["overlay_url"].startswith("data:image/png;base64,")
