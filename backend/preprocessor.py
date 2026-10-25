"""
Image Preprocessor for Attention U-Net
Decodes input image bytes, handles color spaces, resizes to 256x256, and normalizes pixel values.
"""

from typing import Tuple
import cv2
import numpy as np


def preprocess_image_bytes(image_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Decodes raw image bytes into:
    1. Model input tensor: float32 shape (1, 256, 256, 3), range [0.0, 1.0]
    2. Original image dimensions: (height, width)
    3. Resized RGB image: uint8 shape (256, 256, 3) for overlay rendering
    """
    if not image_bytes:
        raise ValueError("Image bytes buffer is empty.")

    np_buf = np.frombuffer(image_bytes, np.uint8)
    # Read with IMREAD_UNCHANGED to detect channels
    img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image bytes. Unsupported or corrupted image format.")

    orig_height, orig_width = img.shape[:2]

    # Convert to RGB (3-channel)
    if img.ndim == 2:
        # Grayscale -> RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] == 4:
            # RGBA -> RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            # BGR -> RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported number of image channels: {img.shape[2]}")
    else:
        raise ValueError(f"Invalid image array dimension: {img.ndim}")

    # Resize to model input dimensions (256, 256) using bilinear interpolation
    rgb_256 = cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0.0, 1.0] float32
    norm_img = rgb_256.astype(np.float32) / 255.0

    # Expand to batch tensor (1, 256, 256, 3)
    tensor = np.expand_dims(norm_img, axis=0)

    return tensor, (orig_height, orig_width), rgb_256
