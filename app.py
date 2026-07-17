#app.py
import streamlit as st, requests
import yfinance as yf
from datetime import date, timedelta

API_URL = "http://localhost:8000/predict"  # replace with your deployed Render backend URL

TIME_STEP = 31  # must match main.py

st.title("TSLA Smart Price Predictor")

lookback_days = st.slider(
    "Lookback buffer (days)",
    30, 180, 90,
    help="Extra calendar days to fetch so weekends/holidays don't leave too few trading days.",
)

if st.button("Fetch latest data & predict next close"):
    end = date.today()
    start = end - timedelta(days=lookback_days)

    df = yf.download("TSLA", start=start.isoformat(), end=end.isoformat(), auto_adjust=False)

    if len(df) < TIME_STEP:
        st.error(
            f"Only {len(df)} trading days returned, need at least {TIME_STEP}. "
            f"Increase the lookback buffer."
        )
    else:
        closes = df["Close"].tail(TIME_STEP).values.flatten().tolist()
        payload = {"closing_prices": closes}
        d = requests.post(API_URL, json=payload, timeout=60).json()

        col1, col2 = st.columns(2)
        col1.metric("Last known close (USD)", f"${d['last_price']:.2f}")
        col2.metric("Predicted next close (USD)", f"${d['predicted_next_price']:.2f}")

        st.subheader("Recent closing price history")
        st.line_chart(df["Close"])
