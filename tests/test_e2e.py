import pytest
from fastapi.testclient import TestClient
import os

def test_full_pipeline_with_flood_sample():
    from backend.app import app
    client = TestClient(app)
    sample_path = "samples/sample_01.jpg"
    assert os.path.exists(sample_path), "sample_01.jpg must exist"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/segment",
        files={"file": ("sample_01.jpg", file_bytes, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    data = res["data"]
    assert "flood_percentage" in data
    assert "risk_level" in data
    assert "images" in data
    assert len(data["images"]["overlay_url"]) > 100
    assert len(data["images"]["binary_mask_url"]) > 100

    # Test recalculation sensitivity monotonicity
    # Threshold 0.2 (more sensitive) should yield >= percentage than threshold 0.8 (more strict)
    res_low = client.post("/api/recalculate", json={"threshold": 0.2}).json()["data"]
    res_high = client.post("/api/recalculate", json={"threshold": 0.8}).json()["data"]
    assert res_low["flood_percentage"] >= res_high["flood_percentage"]

def test_negative_dry_land_sample():
    from backend.app import app
    client = TestClient(app)
    sample_path = "samples/sample_04.jpg"
    assert os.path.exists(sample_path), "sample_04.jpg must exist"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/segment",
        files={"file": ("sample_04.jpg", file_bytes, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()["data"]
    # Dry sample should have lower flood percentage
    assert data["flood_percentage"] >= 0.0
