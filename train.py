# ============================================================
# train.py — Train and Save the Model
# ============================================================
# What this file does:
#   1. Loads the California Housing dataset (built into sklearn)
#   2. Splits into training and testing data
#   3. Trains a Random Forest model
#   4. Checks accuracy
#   5. Saves the model and tools to disk (models/artifacts/)
#
# How to run:
#   python train.py
# ============================================================

import json
import numpy as np
import joblib
from pathlib import Path

from sklearn.datasets       import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.impute         import SimpleImputer
from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score

# ── Folder to save model files ────────────────────────────────
ARTIFACTS = Path("models/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load the dataset
# ─────────────────────────────────────────────────────────────
print("Step 1: Loading California Housing dataset...")

dataset = fetch_california_housing(as_frame=True)
df = dataset.frame.copy()

# Rename target column
df.rename(columns={"MedHouseVal": "SalePrice"}, inplace=True)

# Convert from $100k units to full USD
df["SalePrice"] = df["SalePrice"] * 100_000

# Save a copy as CSV
df.to_csv(DATA_DIR / "train.csv", index=False)
print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────
# STEP 2: Split into features (X) and target (y)
# ─────────────────────────────────────────────────────────────
print("\nStep 2: Preparing features and target...")

X = df.drop(columns=["SalePrice"])  # everything except price
y = df["SalePrice"]                 # just the price column

feature_names = list(X.columns)
print(f"  Features: {feature_names}")


# ─────────────────────────────────────────────────────────────
# STEP 3: Fill missing values and scale
# ─────────────────────────────────────────────────────────────
print("\nStep 3: Preprocessing (impute + scale)...")

imputer = SimpleImputer(strategy="median")  # fill missing with median
X = imputer.fit_transform(X)

scaler = StandardScaler()                   # make all values similar scale
X = scaler.fit_transform(X)


# ─────────────────────────────────────────────────────────────
# STEP 4: Split into train and test sets
# ─────────────────────────────────────────────────────────────
print("\nStep 4: Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train size: {len(X_train)}")
print(f"  Test size:  {len(X_test)}")


# ─────────────────────────────────────────────────────────────
# STEP 5: Train the model
# ─────────────────────────────────────────────────────────────
print("\nStep 5: Training Random Forest model...")

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("  Training done!")


# ─────────────────────────────────────────────────────────────
# STEP 6: Evaluate accuracy
# ─────────────────────────────────────────────────────────────
print("\nStep 6: Evaluating on test data...")

y_pred = model.predict(X_test)
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae    = float(mean_absolute_error(y_test, y_pred))
r2     = float(r2_score(y_test, y_pred))

print(f"  RMSE : ${rmse:,.0f}  (average error)")
print(f"  MAE  : ${mae:,.0f}  (average absolute error)")
print(f"  R2   : {r2:.4f}  (1.0 = perfect)")


# ─────────────────────────────────────────────────────────────
# STEP 7: Save everything to disk
# ─────────────────────────────────────────────────────────────
print("\nStep 7: Saving model and tools...")

joblib.dump(model,   ARTIFACTS / "model.pkl")
joblib.dump(scaler,  ARTIFACTS / "scaler.pkl")
joblib.dump(imputer, ARTIFACTS / "imputer.pkl")

with open(ARTIFACTS / "feature_names.json", "w") as f:
    json.dump(feature_names, f)

with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump({
        "rmse":           round(rmse, 2),
        "mae":            round(mae, 2),
        "r2":             round(r2, 4),
        "rmse_formatted": f"${rmse:,.0f}",
        "mae_formatted":  f"${mae:,.0f}",
    }, f, indent=2)

print("  Saved: model.pkl, scaler.pkl, imputer.pkl")
print("  Saved: feature_names.json, metrics.json")

print("\n[DONE] Now run the API:")
print("  python api/app.py")
