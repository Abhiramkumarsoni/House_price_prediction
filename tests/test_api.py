# ============================================================
# test_api.py — Tests for the Flask API
# ============================================================
# What this file does:
#   Checks that all API endpoints work correctly.
#
# How to run:
#   pytest tests/test_api.py -v
# ============================================================

import json
import sys
from pathlib import Path

import pytest

# Tell Python where to find our api/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Setup: create a test version of the Flask app ────────────
@pytest.fixture
def client():
    from api.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── Test 1: Health check returns OK ──────────────────────────
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"


# ── Test 2: Metrics endpoint returns numbers ──────────────────
def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "r2"   in data
    assert "rmse" in data


# ── Test 3: Predict returns a price ──────────────────────────
def test_predict(client):
    house = {
        "MedInc":     8.3,
        "HouseAge":   41.0,
        "AveRooms":   7.0,
        "AveBedrms":  1.0,
        "Population": 322.0,
        "AveOccup":   2.5,
        "Latitude":   37.88,
        "Longitude":  -122.23,
    }
    response = client.post(
        "/predict",
        data=json.dumps(house),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0


# ── Test 4: Predict with no data returns an error ────────────
def test_predict_no_data(client):
    response = client.post("/predict", content_type="application/json")
    assert response.status_code == 400
