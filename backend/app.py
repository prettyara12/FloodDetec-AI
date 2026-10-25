"""
FastAPI Server for Flood Detection Area Segmentation
Serves the web application and REST endpoints for Attention U-Net inference.
"""

from typing import Optional, Dict, Any
import os
import time
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.inference import FloodInferenceEngine
from backend.preprocessor import preprocess_image_bytes
from backend.postprocessor import postprocess_flood_mask

app = FastAPI(
    title="Flood Area Segmentation API",
    description="High-precision AI flood detection and segmentation powered by Attention U-Net",
    version="1.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine singleton and memory cache
_engine: Optional[FloodInferenceEngine] = None
_cache: Dict[str, Any] = {
    "prob_map": None,
    "rgb_256": None
}


def get_engine() -> FloodInferenceEngine:
    global _engine
    if _engine is None:
        _engine = FloodInferenceEngine()
    return _engine


class RecalculateRequest(BaseModel):
    threshold: float = Field(0.5, ge=0.05, le=0.95, description="Probability threshold for water segmentation")


@app.get("/api/health")
def health_check():
    """Returns engine health status and device specifications."""
    try:
        engine = get_engine()
        return {
            "status": "healthy",
            "model_loaded": True,
            "engine": "ONNXRuntime",
            "device": "CPU",
            "input_shape": [1, 256, 256, 3],
            "output_shape": [1, 256, 256, 1]
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": str(e), "model_loaded": False}
        )


@app.get("/api/samples")
def get_sample_images():
    """Returns available sample flood images for model evaluation."""
    metadata_path = "samples/metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback to direct directory listing if metadata not found
    samples = []
    samples_dir = "samples"
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append({
                    "id": os.path.splitext(f)[0],
                    "title": f.replace("_", " ").title(),
                    "filename": f,
                    "url": f"/samples/{f}"
                })
    return {"samples": samples}


@app.post("/api/segment")
async def segment_flood_area(
    file: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    """
    Executes Attention U-Net binary segmentation on an uploaded flood image.
    Calculates flood area percentage, hazard risk level, and visual overlays.
    """
    # 1. Validate file format and size
    filename = file.filename or "unknown"
    content_type = file.content_type or ""
    valid_extensions = (".jpg", ".jpeg", ".png")

    if not (filename.lower().endswith(valid_extensions) or content_type.startswith("image/")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Tipe file tidak didukung. Harap unggah citra dengan format JPG, JPEG, atau PNG."
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Ukuran berkas melebihi batas maksimal 10 MB."
        )

    # 2. Preprocess image
    try:
        tensor, orig_size, rgb_256 = preprocess_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gagal memproses berkas citra: {str(e)}"
        )

    # 3. Execute inference
    engine = get_engine()
    start_time = time.perf_counter()
    prob_map = engine.predict(tensor)
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 4. Cache probability map for instant slider recalculation
    _cache["prob_map"] = prob_map
    _cache["rgb_256"] = rgb_256

    # 5. Postprocess metrics and visual overlay
    result = postprocess_flood_mask(prob_map, rgb_256, threshold=threshold)
    result["inference_time_ms"] = inference_time_ms
    result["original_dimensions"] = {"height": orig_size[0], "width": orig_size[1]}

    return {
        "status": "success",
        "data": result
    }


@app.post("/api/recalculate")
def recalculate_threshold(req: RecalculateRequest):
    """
    Recalculates flood mask and coverage percentage using the cached probability map.
    Runs instantaneously without re-executing neural network inference.
    """
    if _cache["prob_map"] is None or _cache["rgb_256"] is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tidak ada citra aktif di cache. Silakan unggah citra terlebih dahulu."
        )

    result = postprocess_flood_mask(_cache["prob_map"], _cache["rgb_256"], threshold=req.threshold)
    return {
        "status": "success",
        "data": result
    }


# Mount static assets if directories exist
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

if os.path.exists("samples"):
    app.mount("/samples", StaticFiles(directory="samples"), name="samples")


@app.api_route("/", methods=["GET", "HEAD"])
def serve_home():
    """Serves the FloodDetec AI landing homepage."""
    home_path = os.path.join("frontend", "home.html")
    if os.path.exists(home_path):
        return FileResponse(home_path)
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "FloodDetec AI API is running. Access /docs for API documentation."}


@app.api_route("/studio", methods=["GET", "HEAD"])
def serve_studio():
    """Serves the interactive Studio Workbench application."""
    studio_path = os.path.join("frontend", "studio.html")
    if os.path.exists(studio_path):
        return FileResponse(studio_path)
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Studio page not found.")


@app.api_route("/model", methods=["GET", "HEAD"])
def serve_model_docs():
    """Serves the AI Model & Architecture Documentation page."""
    model_path = os.path.join("frontend", "model.html")
    if os.path.exists(model_path):
        return FileResponse(model_path)
    raise HTTPException(status_code=404, detail="Model documentation page not found.")
