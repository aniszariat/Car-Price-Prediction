# Car Price Prediction

A compact guide to this project: data exploration, preprocessing, model training, model persistence, FastAPI service, and frontend integration.

## Project Structure

- `API/` — FastAPI app and `index.html` frontend.
- `data/` — raw dataset (`car_price_prediction.csv`).
- `models/` — saved models and preprocessing artifacts (scaler, encoders, model .pkl files).
- `notebooks/` — exploratory analysis and training notebooks.
- `scripts/` — reusable preprocessing utilities.

---

## Initial Data Understanding

- Open the dataset at `data/car_price_prediction.csv` to inspect columns and sample rows.
- Typical columns: `Price`, `Levy`, `Manufacturer`, `Model`, `Prod. year`, `Category`, `Leather interior`, `Fuel type`, `Engine volume`, `Mileage`, `Cylinders`, `Gear box type`, `Drive wheels`, `Doors`, `Wheel`, `Color`, `Airbags`.
- Key checks:
  - Missing values and placeholder tokens (e.g., `-`).
  - Unusual formats (e.g., `Mileage` stored as `"186005 km"`).
  - Cardinality on categorical columns (`Model`, `Manufacturer`).
  - Target distribution (`Price`) and outliers.

Tips: use the notebooks in `notebooks/` (for example `EDA.ipynb`) for visuals and summary statistics.

---

## Data Cleaning

Steps typically applied in `scripts/preprocessing.py` and notebooks:

- Normalize numeric columns: strip non-numeric characters from `Mileage`, convert to integers.
- Replace placeholders (`-`) with `NaN` and decide imputation or removal.
- Convert `Prod. year` into a derived `Age` feature: `Age = current_year - Prod. year`.
- Drop irrelevant or leaky columns (e.g., `ID`, `Doors`, original `Prod. year` after `Age`).
- Standardize categorical values (trim/upper-case), and group rare categories if needed.
- Encode categories:
  - One-hot for small cardinality (e.g., `Leather interior`, `Gear box type`, `Drive wheels`, `Wheel`).
  - Label encoding or target/mean encoding for high-cardinality columns (`Manufacturer`, `Model`).
- Scale numerical features with a fitted `scaler` (e.g., `StandardScaler` or `MinMaxScaler`).

Save preprocessing artifacts (scaler, one-hot encoder, label encoders) to `models/` for reproducible inference.

---

## Training a Model

Typical training flow (see `notebooks/training.ipynb`):

1. Load cleaned dataset and split into train/validation/test sets.
2. Select and train a model (e.g., `RandomForestRegressor`, `XGBoost`, or `LightGBM`).
3. Evaluate with relevant metrics (MAE, RMSE, R^2) and inspect residuals.
4. Tune hyperparameters (GridSearch / RandomizedSearch / Optuna).

Keep experiments and random seeds documented in notebooks for reproducibility.

---

## Saving the Model

- Persist the trained model and preprocessing objects using `pickle` (or `joblib`) into `models/`, for example:

```
models/randomForestModel.pkl
models/scaler.pkl
models/label_encoders.pkl
models/one_hot_encoder.pkl
```

- Ensure the inference pipeline expects the same feature order and names as saved artifacts.

---

## Creating an API with FastAPI

- The `API/main.py` file implements a small FastAPI app exposing endpoints:
  - `POST /predict/` — returns predicted price for a provided car spec JSON.
  - Several `GET` endpoints for populating selects (`/manufacturers/`, `/categories/`, etc.).

- Run locally during development with:

```bash
pip install -r requirements.txt    # include fastapi, uvicorn, pandas, scikit-learn, etc.
uvicorn API.main:app --reload --host 0.0.0.0 --port 8000
```

- The API loads preprocessing artifacts and the saved model from `models/` and applies the same transforms before predicting.

---

## Integrating the API in the Frontend

- The frontend is a single-page UI at `API/index.html` that posts the form payload to the `POST /predict/` endpoint.
- Dynamic selects are populated from the API `GET` endpoints so the UI reflects the dataset vocabulary.
- Typical flow:
  1. User fills the form in `index.html` and clicks predict.
  2. Frontend builds a JSON payload matching the API `CarInput` schema.
  3. Frontend displays loading state, then renders the returned predicted price.

---

## Quick Notes & Troubleshooting

- If you change preprocessing or model training, re-save artifacts in `models/` and restart the FastAPI server.
- Confirm feature names expected by the API match the pipeline; mismatches cause inference errors.
- Check `models/` permissions and relative paths; API uses `../models/` relative to `API/`.

---

