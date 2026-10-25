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
    assert data["model_loaded"] is True
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
    assert "images" in data["data"]

def test_api_recalculate_endpoint():
    from backend.app import app
    client = TestClient(app)
    # First upload an image to populate cache
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    client.post("/api/segment", files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")}, data={"threshold": "0.5"})

    # Now recalculate with threshold 0.8
    response = client.post("/api/recalculate", json={"threshold": 0.8})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["threshold_used"] == 0.8

def test_api_rejects_non_image():
    from backend.app import app
    client = TestClient(app)
    files = {"file": ("document.txt", b"This is not an image file.", "text/plain")}
    response = client.post("/api/segment", files=files)
    assert response.status_code in [400, 415]

def test_api_samples_endpoint():
    from backend.app import app
    client = TestClient(app)
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) == 4
    first = data["samples"][0]
    assert "id" in first
    assert "url" in first

def test_frontend_static_serving():
    from backend.app import app
    client = TestClient(app)
    
    # Test Home page
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert "FloodDetec AI" in res_home.text
    assert "Deteksi Banjir Berbasis AI" in res_home.text

    # Test Studio page
    res_studio = client.get("/studio")
    assert res_studio.status_code == 200
    assert "dropzone" in res_studio.text
    assert "Studio Analisis" in res_studio.text

    # Test Model Documentation page
    res_model = client.get("/model")
    assert res_model.status_code == 200
    assert "Attention U-Net" in res_model.text
    assert "Spesifikasi REST API" in res_model.text


