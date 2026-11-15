import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("netguard")

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "ml" / "models"))
MAX_RECORDS = 5000
CATEGORICAL = ["proto", "service", "state"]

model = joblib.load(MODEL_DIR / "rf_model.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
raw_columns = joblib.load(MODEL_DIR / "raw_columns.joblib")
numeric_columns = [c for c in raw_columns if c not in CATEGORICAL]
log.info("model loaded: %d input columns", len(feature_columns))

app = FastAPI(title="NetGuard AI")


class PredictRequest(BaseModel):
    records: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    n = len(req.records)
    if n == 0:
        raise HTTPException(status_code=422, detail="records is empty")
    if n > MAX_RECORDS:
        raise HTTPException(status_code=413, detail=f"too many records: {n} (max {MAX_RECORDS})")

    df = pd.DataFrame(req.records)
    missing = [c for c in raw_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing columns: {missing}")

    df = df[raw_columns].copy()
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    bad_rows = int(df[numeric_columns].isna().any(axis=1).sum())
    df[numeric_columns] = df[numeric_columns].fillna(0)

    X = pd.get_dummies(df, dtype=int)
    X = X.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    flagged = int((proba >= 0.5).sum())
    log.info("predict rows=%d flagged=%d bad_rows=%d ms=%d", n, flagged, bad_rows, (time.time() - start) * 1000)

    return {
        "results": [
            {"label": "attack" if p >= 0.5 else "normal", "attack_probability": round(float(p), 4)}
            for p in proba
        ]
    }


FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")