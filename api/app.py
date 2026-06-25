# ============================================================
# Flask API — House Price Prediction
# ============================================================
# What this file does:
#   Starts a small web server that receives house data
#   and returns a predicted price from our trained model.
#
# How to run:
#   python api/app.py
#
# How to test in browser:
#   http://localhost:5000/health
# ============================================================

import json
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Where are our saved model files? ─────────────────────────
ARTIFACTS = Path(__file__).parent.parent / "models" / "artifacts"

# ── Create the web app ────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Let Streamlit call this API

# ── Load the model when the server starts ────────────────────
model    = joblib.load(ARTIFACTS / "model.pkl")
scaler   = joblib.load(ARTIFACTS / "scaler.pkl")
imputer  = joblib.load(ARTIFACTS / "imputer.pkl")

with open(ARTIFACTS / "feature_names.json") as f:
    feature_names = json.load(f)

with open(ARTIFACTS / "metrics.json") as f:
    metrics = json.load(f)

print(f"Model loaded! Features: {feature_names}")


# ── ROUTE 1: Health Check ─────────────────────────────────────
# Visit: http://localhost:5000/health
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})


# ── ROUTE 2: Model Metrics ────────────────────────────────────
# Visit: http://localhost:5000/metrics
@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)


# ── ROUTE 3: Predict Price ────────────────────────────────────
# Send: POST http://localhost:5000/predict
# Body: {"MedInc": 8.3, "HouseAge": 41, "AveRooms": 7, ...}
@app.route("/predict", methods=["POST"])
def predict():
    # Step 1: Read the JSON data sent by the user
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Send JSON data!"}), 400

    # Step 2: Build a list of values in the right order
    values = []
    for feature in feature_names:
        value = data.get(feature, 0.0)  # use 0 if feature is missing
        values.append(value)

    # Step 3: Convert to numpy array and reshape to 1 row
    X = np.array(values).reshape(1, -1)

    # Step 4: Apply the same preprocessing as training
    X = imputer.transform(X)   # fill missing values
    X = scaler.transform(X)    # scale the numbers

    # Step 5: Predict!
    price = model.predict(X)[0]

    return jsonify({
        "predicted_price": round(float(price), 2),
        "formatted":       f"${price:,.0f}"
    })


# ── Start the server ──────────────────────────────────────────
if __name__ == "__main__":
    print("API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
