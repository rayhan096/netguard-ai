import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "ml" / "models"))

model = joblib.load(MODEL_DIR / "rf_model.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
raw_columns = joblib.load(MODEL_DIR / "raw_columns.joblib")

app = FastAPI(title="NetGuard AI")


class PredictRequest(BaseModel):
    records: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(status_code=422, detail="records is empty")

    df = pd.DataFrame(req.records)
    missing = [c for c in raw_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing columns: {missing}")

    X = pd.get_dummies(df[raw_columns], dtype=int)
    X = X.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    return {
        "results": [
            {"label": "attack" if p >= 0.5 else "normal", "attack_probability": round(float(p), 4)}
            for p in proba
        ]
    }