#app.py
import streamlit as st, requests
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

API_BASE = "https://streamlit-lstm.onrender.com"  # replace with your deployed Render backend URL

TIME_STEP = 31  # must match main.py

st.title("TSLA Smart Price Predictor")

lookback_days = st.slider(
    "Lookback buffer (days)",
    30, 180, 90,
    help="Extra calendar days to fetch so weekends/holidays don't leave too few trading days.",
)

days_ahead = st.slider(
    "Forecast horizon (trading days)",
    1, 30, 1,
    help="Day 1 is a real 1-step prediction from actual data. Beyond that, each day is "
         "forecast from the model's own prior prediction, so accuracy drops the further out you go.",
)

if st.button("Fetch latest data & predict"):
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

        if days_ahead == 1:
            d = requests.post(f"{API_BASE}/predict", json={"closing_prices": closes}, timeout=60).json()
            col1, col2 = st.columns(2)
            col1.metric("Last known close (USD)", f"${d['last_price']:.2f}")
            col2.metric("Predicted next close (USD)", f"${d['predicted_next_price']:.2f}")
            forecast_prices = [d["predicted_next_price"]]
        else:
            payload = {"closing_prices": closes, "days_ahead": days_ahead}
            d = requests.post(f"{API_BASE}/predict_multi", json=payload, timeout=60).json()
            st.metric("Last known close (USD)", f"${d['last_known_price']:.2f}")
            st.caption(
                f"Showing a {days_ahead}-day recursive forecast. Only day 1 uses real data as "
                f"input -- later days compound on the model's own prior guesses, so treat them "
                f"as directional, not precise."
            )
            forecast_prices = [f["predicted_price"] for f in d["forecasts"]]

        # build a combined real + forecast chart, forecast plotted as its own series
        history = df["Close"].copy()
        history.name = "Actual"
        last_date = history.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_prices))
        forecast_series = pd.Series(forecast_prices, index=future_dates, name="Forecast")

        combined = pd.concat([history, forecast_series], axis=1)
        st.subheader("Price history + forecast")
        st.line_chart(combined)

        if days_ahead > 1:
            st.dataframe(
                pd.DataFrame({"Date": future_dates.date, "Predicted Price": forecast_prices})
            )

