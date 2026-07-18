from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import onnxruntime as ort
import numpy as np
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # calls are server-to-server (n8n / Streamlit backend), tighten if you expose this publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

TIME_STEP = int(os.environ.get("TIME_STEP", 31))
MODEL_PATH = os.environ.get("MODEL_PATH", "lstm_tesla_model.onnx")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler_params.json")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")  # set this in Render; leave unset to disable auth

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open(SCALER_PATH) as f:
    scaler_params = json.load(f)

DATA_MIN = scaler_params["data_min"]
DATA_MAX = scaler_params["data_max"]
FEATURE_MIN = scaler_params["feature_min"]
FEATURE_MAX = scaler_params["feature_max"]

# --- GLOBAL IN-MEMORY STORAGE ---
# Holds the latest compiled workflow run from n8n
LATEST_DASHBOARD_DATA: Dict[str, Any] = {}


def scale(x: np.ndarray) -> np.ndarray:
    return (x - DATA_MIN) / (DATA_MAX - DATA_MIN) * (FEATURE_MAX - FEATURE_MIN) + FEATURE_MIN


def unscale(x: float) -> float:
    return (x - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN) * (DATA_MAX - DATA_MIN) + DATA_MIN


def next_business_days(start_date: datetime, n: int) -> List[datetime]:
    """Skip weekends so forecast dates land on plausible trading days."""
    dates = []
    current = start_date
    while len(dates) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
    return dates


# --- PYDANTIC SCHEMAS ---
class PriceWindow(BaseModel):
    closing_prices: List[float] = Field(..., description=f"Chronological closing prices, >= {TIME_STEP} values")

class MultiDayRequest(BaseModel):
    closing_prices: List[float] = Field(..., description=f"Chronological closing prices, >= {TIME_STEP} values")
    days_ahead: int = Field(5, ge=1, le=60, description="How many future days to forecast")
    last_date: Optional[str] = Field(
        None, description="ISO date (YYYY-MM-DD) of the most recent closing price. If provided, "
                           "each forecast row includes a 'date' so this can be passed straight into /update_latest."
    )

class DashboardPayload(BaseModel):
    ticker: str = "TSLA"
    actual_history: List[Dict[str, Any]] = Field(..., description="List of dicts containing {'date': ..., 'close': ...} from Marketstack")
    forecasts: List[Dict[str, Any]] = Field(..., description="List of dicts containing {'date': ..., 'predicted_price': ...}")
    news_headlines: List[Dict[str, Any]] = Field(..., description="List of articles from Mediastack")
    ai_insight: str = Field(..., description="Generated analysis text from Gemini")


def _predict_one(window: np.ndarray) -> float:
    scaled = scale(window).astype(np.float32).reshape(1, TIME_STEP, 1)
    onnx_out = session.run([output_name], {input_name: scaled})[0]
    return unscale(float(onnx_out.flatten()[0]))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_multi")
def predict_multi(c: MultiDayRequest):
    """Called by n8n during the automated pipeline execution."""
    if len(c.closing_prices) < TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {TIME_STEP} closing prices, got {len(c.closing_prices)}.",
        )

    future_dates = None
    if c.last_date:
        try:
            start = datetime.strptime(c.last_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="last_date must be in YYYY-MM-DD format.")
        future_dates = next_business_days(start, c.days_ahead)

    window = list(c.closing_prices[-TIME_STEP:])
    last_price = float(window[-1])

    forecasts = []
    for day in range(1, c.days_ahead + 1):
        pred_price = _predict_one(np.array(window, dtype=np.float64))
        entry = {"day": day, "predicted_price": round(pred_price, 2)}
        if future_dates:
            entry["date"] = future_dates[day - 1].strftime("%Y-%m-%d")
        forecasts.append(entry)
        window.append(pred_price)
        window = window[-TIME_STEP:]

    return {
        "last_known_price": round(last_price, 2),
        "last_date": c.last_date,
        "days_ahead": c.days_ahead,
        "forecasts": forecasts,
    }


# --- WORKFLOW INTEGRATION ENDPOINTS ---

@app.post("/update_latest")
def update_latest(payload: DashboardPayload, x_webhook_secret: Optional[str] = Header(None)):
    """Endpoint for n8n to send the finalized workflow data bundle."""
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing webhook secret.")

    global LATEST_DASHBOARD_DATA
    LATEST_DASHBOARD_DATA = payload.model_dump()
    LATEST_DASHBOARD_DATA["_updated_at"] = datetime.utcnow().isoformat()
    return {"status": "success", "message": "Dashboard data updated successfully."}


@app.get("/latest")
def get_latest():
    """Endpoint for Streamlit to pull the most recent data bundle instantly."""
    if not LATEST_DASHBOARD_DATA:
        raise HTTPException(status_code=404, detail="No workflow data available yet. Please execute n8n workflow.")
    return LATEST_DASHBOARD_DATA
