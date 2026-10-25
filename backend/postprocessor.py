"""
Postprocessor for Attention U-Net Flood Segmentation
Handles thresholding, contour extraction, overlay blending, risk metrics, and Base64 encoding.
"""

from typing import Dict, Any, Tuple
import base64
import cv2
import numpy as np


def _encode_png_base64(img_rgb_or_gray: np.ndarray) -> str:
    """Encodes a uint8 RGB or grayscale image into a Base64 data URL."""
    if img_rgb_or_gray.ndim == 3:
        # OpenCV imencode expects BGR order
        bgr = cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR)
    else:
        bgr = img_rgb_or_gray

    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format.")

    b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def classify_risk_level(flood_percentage: float) -> Tuple[str, str]:
    """
    Returns (risk_level, risk_color) according to PRD severity standards:
    - Aman: < 5.0% (#10B981)
    - Waspada: 5.0% - 19.99% (#3B82F6)
    - Siaga: 20.0% - 49.99% (#F59E0B)
    - Awas: >= 50.0% (#EF4444)
    """
    if flood_percentage < 5.0:
        return "Aman (Normal)", "#10B981"
    elif flood_percentage < 20.0:
        return "Waspada (Low Risk)", "#3B82F6"
    elif flood_percentage < 50.0:
        return "Siaga (Medium Risk)", "#F59E0B"
    else:
        return "Awas (High Risk)", "#EF4444"


def postprocess_flood_mask(
    prob_map: np.ndarray,
    orig_rgb: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes flood metrics, risk status, binary mask, and electric cyan overlay.
    """
    threshold = float(np.clip(threshold, 0.05, 0.95))
    total_pixels = int(prob_map.size)

    # Binary mask: uint8 0 or 255
    mask_bool = prob_map >= threshold
    flood_pixels = int(np.sum(mask_bool))
    flood_percentage = round((flood_pixels / total_pixels) * 100.0, 2)

    risk_level, risk_color = classify_risk_level(flood_percentage)

    binary_mask = (mask_bool.astype(np.uint8)) * 255

    # Create visual overlay: Electric Cyan (0, 240, 255) in RGB
    overlay = orig_rgb.copy()
    cyan_color = np.array([0, 240, 255], dtype=np.float32)

    if flood_pixels > 0:
        # Alpha blending for flood region (alpha = 0.40)
        alpha = 0.40
        overlay_region = overlay[mask_bool].astype(np.float32)
        blended_region = (1.0 - alpha) * overlay_region + alpha * cyan_color
        overlay[mask_bool] = blended_region.astype(np.uint8)

        # Draw contour boundary lines (cyan, 1px)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 240, 255), 1)

    # Encode images to Base64 URLs
    original_url = _encode_png_base64(orig_rgb)
    binary_mask_url = _encode_png_base64(binary_mask)
    overlay_url = _encode_png_base64(overlay)

    return {
        "flood_percentage": flood_percentage,
        "flood_pixel_count": flood_pixels,
        "total_pixel_count": total_pixels,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "threshold_used": threshold,
        "images": {
            "original_url": original_url,
            "binary_mask_url": binary_mask_url,
            "overlay_url": overlay_url
        }
    }
