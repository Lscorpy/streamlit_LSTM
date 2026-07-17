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
    scaled = scale(prices).astype(np.float32).reshape(1, TIME_STEP, 1)

    onnx_out = session.run([output_name], {input_name: scaled})[0]
    pred_scaled = float(onnx_out.flatten()[0])
    pred_price = unscale(pred_scaled)

    last_price = float(prices[-1])

    return {
        "last_price": round(last_price, 2),
        "predicted_next_price": round(pred_price, 2),
    }
