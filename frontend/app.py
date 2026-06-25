# ============================================================
# Streamlit Frontend — House Price Predictor
# ============================================================
# What this file does:
#   Shows a web page where users can input house details
#   and click a button to get a predicted price.
#   It calls the Flask API to get the prediction.
#
# How to run:
#   streamlit run frontend/app.py
#
# Make sure the Flask API is running first!
#   python api/app.py
# ============================================================

import os
import requests
import streamlit as st

# ── API URL ───────────────────────────────────────────────────
# When running locally:  http://localhost:5000
# When running Docker:   http://api:5000  (set via docker-compose env)
API_URL = os.environ.get("API_URL", "http://localhost:5000")

# ── Page Setup ───────────────────────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")
st.write("Fill in the house details below and click **Predict Price**.")
st.markdown("---")

# ── Check if API is running ───────────────────────────────────
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    st.success("Flask API is running!")
except Exception:
    st.error("Flask API is not running. Start it with: python api/app.py")
    st.stop()  # Stop here if API is not available

# ── Show Model Performance ────────────────────────────────────
try:
    m = requests.get(f"{API_URL}/metrics").json()
    col1, col2, col3 = st.columns(3)
    col1.metric("R2 Score",  m.get("r2", "N/A"))
    col2.metric("RMSE",      m.get("rmse_formatted", "N/A"))
    col3.metric("MAE",       m.get("mae_formatted", "N/A"))
except Exception:
    pass

st.markdown("---")

# ── Input Fields ──────────────────────────────────────────────
st.subheader("Enter House Details")

col_a, col_b = st.columns(2)

with col_a:
    medinc     = st.slider("Median Income (x$10K)",  0.5,  15.0,  5.0)
    house_age  = st.slider("House Age (years)",        1.0,  52.0, 20.0)
    ave_rooms  = st.slider("Average Rooms",            1.0,  20.0,  6.0)
    ave_bedrms = st.slider("Average Bedrooms",         0.5,   6.0,  1.1)

with col_b:
    population = st.slider("Block Population",         5.0, 5000.0, 1200.0)
    ave_occup  = st.slider("Average Occupants",        1.0,  10.0,  3.0)
    latitude   = st.slider("Latitude",                32.5,  42.0, 37.5)
    longitude  = st.slider("Longitude",             -124.5, -114.3, -122.0)

# ── Predict Button ────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):

    # Build the data to send to the API
    house_data = {
        "MedInc":     medinc,
        "HouseAge":   house_age,
        "AveRooms":   ave_rooms,
        "AveBedrms":  ave_bedrms,
        "Population": population,
        "AveOccup":   ave_occup,
        "Latitude":   latitude,
        "Longitude":  longitude,
    }

    # Call the Flask API
    try:
        response = requests.post(f"{API_URL}/predict", json=house_data)
        result   = response.json()

        # Show the predicted price
        price = result["formatted"]
        st.markdown(f"## Predicted Price: **{price}**")
        st.balloons()

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Random Forest | Dataset: California Housing (sklearn)")
