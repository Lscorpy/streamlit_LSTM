#main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import onnxruntime as ort
import numpy as np
import pandas as pd
import json
import os

app = FastAPI()

# n8n / Streamlit will call this service from a different origin, so open it up.
# Tighten allow_origins to your actual n8n + Streamlit hosts once you know them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TIME_STEP = int(os.environ.get("TIME_STEP", 31))  # must match app.py / n8n and the training window length
MODEL_PATH = os.environ.get("MODEL_PATH", "lstm_tesla_model.onnx")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler_params.json")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open(SCALER_PATH) as f:
    scaler_params = json.load(f)

# MinMaxScaler(feature_range=(0,1)) params, matching training:
# x_scaled = (x - data_min) / (data_max - data_min) * (feature_max - feature_min) + feature_min
DATA_MIN = scaler_params["data_min"]
DATA_MAX = scaler_params["data_max"]
FEATURE_MIN = scaler_params["feature_min"]
FEATURE_MAX = scaler_params["feature_max"]


def scale(x: np.ndarray) -> np.ndarray:
    return (x - DATA_MIN) / (DATA_MAX - DATA_MIN) * (FEATURE_MAX - FEATURE_MIN) + FEATURE_MIN


def unscale(x: float) -> float:
    return (x - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN) * (DATA_MAX - DATA_MIN) + DATA_MIN


class PriceWindow(BaseModel):
    closing_prices: List[float] = Field(
        ..., description=f"Chronological closing prices (oldest first), needs >= {TIME_STEP} values"
    )


class MultiDayRequest(BaseModel):
    closing_prices: List[float] = Field(
        ..., description=f"Chronological closing prices (oldest first), needs >= {TIME_STEP} values"
    )
    days_ahead: int = Field(5, ge=1, le=60, description="How many future days to forecast recursively")
    last_date: Optional[str] = Field(
        None,
        description="ISO date (YYYY-MM-DD) of the most recent closing_prices entry. "
                    "If provided, each forecast day is returned with a real (business-day) calendar date, "
                    "so callers like n8n don't need to compute future trading dates themselves.",
    )


def _predict_one(window: np.ndarray) -> float:
    """Single 1-step prediction from a window of exactly TIME_STEP raw prices."""
    scaled = scale(window).astype(np.float32).reshape(1, TIME_STEP, 1)
    onnx_out = session.run([output_name], {input_name: scaled})[0]
    pred_scaled = float(onnx_out.flatten()[0])
    return unscale(pred_scaled)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: PriceWindow):
    if len(c.closing_prices) < TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {TIME_STEP} closing prices, got {len(c.closing_prices)}.",
        )

    # model predicts the next scaled price level directly from a window of
    # TIME_STEP scaled prices -- no log returns involved (see create_dataset
    # in the training notebook)
    prices = np.array(c.closing_prices[-TIME_STEP:], dtype=np.float64)
    pred_price = _predict_one(prices)
    last_price = float(prices[-1])

    return {
        "last_price": round(last_price, 2),
        "predicted_next_price": round(pred_price, 2),
    }


@app.post("/predict_multi")
def predict_multi(c: MultiDayRequest):
    """
    Recursive multi-day forecast: predicts day+1, appends it to the window,
    drops the oldest price, predicts day+2, and so on. Each step beyond the
    first is forecasting from the model's own prior output rather than real
    data, so error compounds -- treat later days as much less reliable than
    the first. Swap in real closes as they become available and re-call this
    endpoint fresh rather than trusting a long unattended forecast.

    If `last_date` is supplied, each forecast entry also gets a `date` field
    (next business days after last_date), which saves n8n from having to
    compute a business-day calendar with an expression/Code node.
    """
    if len(c.closing_prices) < TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {TIME_STEP} closing prices, got {len(c.closing_prices)}.",
        )

    window = list(c.closing_prices[-TIME_STEP:])
    last_price = float(window[-1])

    forecasts = []
    for day in range(1, c.days_ahead + 1):
        pred_price = _predict_one(np.array(window, dtype=np.float64))
        forecasts.append({"day": day, "predicted_price": round(pred_price, 2)})
        window.append(pred_price)  # feed prediction back in
        window = window[-TIME_STEP:]  # slide the window

    if c.last_date:
        try:
            start = pd.Timestamp(c.last_date) + pd.Timedelta(days=1)
            future_dates = pd.bdate_range(start=start, periods=c.days_ahead)
            for f, d in zip(forecasts, future_dates):
                f["date"] = d.date().isoformat()
        except (ValueError, TypeError):
            # bad/unparseable last_date -- just skip the date annotation rather than fail the request
            pass

    return {
        "last_known_price": round(last_price, 2),
        "days_ahead": c.days_ahead,
        "forecasts": forecasts,
    }
