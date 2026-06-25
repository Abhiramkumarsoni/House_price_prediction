# House Price Prediction MLOps

A production-grade, end-to-end Machine Learning pipeline for predicting house prices.

## 🏗️ Architecture

```
                    ┌─────────────────────────────┐
                    │   GitHub Actions CI/CD       │
                    │  Test → Train → Docker Push  │
                    └────────────┬────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │            docker-compose            │
              │  ┌─────────────┐  ┌───────────────┐ │
              │  │  Flask API  │  │  Streamlit UI │ │
              │  │  :5000      │  │  :8501        │ │
              │  └──────┬──────┘  └───────┬───────┘ │
              │         │                 │          │
              │  ┌──────▼─────────────────▼──────┐  │
              │  │     models/artifacts/          │  │
              │  │  model.pkl / scaler.pkl / ...  │  │
              │  └───────────────────────────────┘  │
              └─────────────────────────────────────┘
```

## 📁 Project Structure

```
House_price_prediction_mlops/
├── data/
│   ├── raw/                  # Downloaded CSV (Kaggle or sklearn)
│   └── processed/            # Train/test splits
├── models/
│   └── artifacts/
│       ├── model.pkl         # Trained Random Forest
│       ├── scaler.pkl        # StandardScaler
│       ├── imputer.pkl       # SimpleImputer
│       ├── encoders.pkl      # LabelEncoders
│       ├── feature_names.json
│       ├── metrics.json      # RMSE, MAE, R²
│       └── model_metadata.json
├── src/
│   ├── data_ingestion.py     # Kaggle / sklearn fallback
│   ├── preprocessing.py      # Feature engineering & scaling
│   ├── train.py              # Random Forest training
│   └── evaluate.py           # Metrics computation
├── api/
│   └── app.py                # Flask REST API
├── frontend/
│   └── app.py                # Streamlit UI
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
├── .github/
│   └── workflows/
│       └── ci_cd.yml         # GitHub Actions pipeline
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── train_pipeline.py
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_pipeline.py
```

This will:
- Download data (Kaggle if credentials found, else California Housing from sklearn)
- Preprocess and split data
- Train a Random Forest Regressor
- Save all artifacts to `models/artifacts/`

### 3. Start the Flask API

```bash
python api/app.py
```

API available at `http://localhost:5000`

#### Test the API

```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/info

# Predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.9, "AveBedrms": 1.02, "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}'
```

### 4. Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Frontend available at `http://localhost:8501`

## 🐳 Docker

### Build & Run with Docker Compose

```bash
# Build and start both services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

Services:
- **API**: `http://localhost:5000`
- **Frontend**: `http://localhost:8501`

### Individual Docker builds

```bash
# API
docker build -f Dockerfile.api -t house-price-api .
docker run -p 5000:5000 house-price-api

# Frontend
docker build -f Dockerfile.frontend -t house-price-frontend .
docker run -p 8501:8501 house-price-frontend
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing
```

## ⚙️ GitHub Actions CI/CD

The pipeline at `.github/workflows/ci_cd.yml` runs on every push to `main`:

| Stage | What it does |
|-------|-------------|
| 🧪 **Test** | Runs pytest unit tests |
| 🤖 **Train** | Runs full training pipeline, uploads artifacts |
| 🐳 **Build & Push** | Builds Docker images and pushes to DockerHub |
| 🚀 **Deploy** | Deploy notification (customize for your host) |

### Setup GitHub Secrets

In your GitHub repo → Settings → Secrets → Actions:

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Your DockerHub username |
| `DOCKER_PASSWORD` | Your DockerHub password or access token |

### Setup Kaggle (Optional)

1. Go to [kaggle.com](https://kaggle.com) → Account → API → Create New Token
2. Place `kaggle.json` at `C:\Users\<YourName>\.kaggle\kaggle.json`

If not configured, the pipeline uses **California Housing dataset** from sklearn automatically.

## 📊 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Trees | 200 |
| Max Depth | 15 |
| CV Folds | 5 |
| Features | 8 (California) / 79 (Kaggle) |
| Target | House Price (USD) |

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/info` | Model metadata & metrics |
| `GET` | `/features` | Feature names & sample input |
| `POST` | `/predict` | Predict house price |

### Predict Request Example

```json
{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.023810,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}
```

### Predict Response Example

```json
{
    "predicted_price": 452300.0,
    "predicted_price_formatted": "$452,300",
    "currency": "USD"
}
```
