#main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import onnxruntime as ort
import numpy as np
import json
import os

app = FastAPI()

TIME_STEP = int(os.environ.get("TIME_STEP", 31))  # must match app.py and training window length
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

    return {
        "last_known_price": round(last_price, 2),
        "days_ahead": c.days_ahead,
        "forecasts": forecasts,
    }

