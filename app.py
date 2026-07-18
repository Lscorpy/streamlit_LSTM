#app.py
import streamlit as st, requests
import pandas as pd

# n8n production webhook URL (the "Webhook" node's path is "stock-forecast" in My workflow.json)
N8N_WEBHOOK_URL = "https://jjypqr.app.n8n.cloud/webhook/stock-forecast"  # replace with your n8n webhook URL

def to_naive_datetimeindex(values) -> pd.DatetimeIndex:
    """
    Parse a list of date/timestamp strings into a guaranteed tz-naive DatetimeIndex.
 
    pd.to_datetime() on a plain list can silently return a mixed tz-aware/tz-naive
    index (object dtype) if even one string in the batch carries a timezone offset
    while the rest don't -- e.g. one bar arriving as a full ISO timestamp while the
    others are plain "YYYY-MM-DD". That later blows up pd.concat with
    "Cannot join tz-naive with tz-aware DatetimeIndex". Forcing utc=True first
    normalizes everything to a single tz-aware dtype, then tz_localize(None)
    strips the tz so it's safe to combine with any other naive index.
    """
    idx = pd.to_datetime(values, format="mixed", utc=True)
    return idx.tz_localize(None)
 
 
st.title("TSLA Smart Price Predictor")
 
lookback_days = st.slider(
    "Lookback buffer (days)",
    30, 180, 90,
    help="Extra calendar days to fetch so weekends/holidays don't leave too few trading days. "
         "n8n / Marketstack use this to fetch enough history for the model's 31-day window.",
)
 
days_ahead = st.slider(
    "Forecast horizon (trading days)",
    1, 30, 1,
    help="Day 1 is a real 1-step prediction from actual data. Beyond that, each day is "
         "forecast from the model's own prior prediction, so accuracy drops the further out you go.",
)
 
if st.button("Fetch latest data & predict"):
    with st.spinner("Fetching prices, running the model, pulling news, and generating insight..."):
        try:
            resp = requests.post(
                N8N_WEBHOOK_URL,
                json={"lookback_days": lookback_days, "days_ahead": days_ahead},
                timeout=120,  # chain of Marketstack -> FastAPI -> Mediastack -> Gemini can take a while
            )
            resp.raise_for_status()
            d = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the n8n workflow: {e}")
            st.stop()
 
    # ---- Expected response shape from the n8n workflow ----
    # {
    #   "actual": {"dates": [...ISO dates...], "closing_prices": [...]},
    #   "last_known_price": float,
    #   "forecasts": [{"day": 1, "predicted_price": float, "date": "YYYY-MM-DD"}, ...],
    #   "insight": {"news_highlight": str, "ai_insight": str}
    # }
    try:
        actual_dates = to_naive_datetimeindex(d["actual"]["dates"])
        actual_prices = d["actual"]["closing_prices"]
        forecasts = d["forecasts"]
        last_known_price = d["last_known_price"]
        insight = d.get("insight", {})
    except (KeyError, TypeError) as e:
        st.error(f"Unexpected response shape from n8n workflow: missing {e}")
        st.json(d)
        st.stop()
 
    # defensive guard: dates/prices should always be the same length, but if an
    # upstream step (Marketstack gaps, aggregation, etc.) ever desyncs them,
    # trim to the shorter one instead of crashing the whole app.
    if len(actual_dates) != len(actual_prices):
        st.warning(
            f"Actual dates ({len(actual_dates)}) and prices ({len(actual_prices)}) "
            f"came back mismatched from the workflow -- trimming to the shorter length."
        )
        n = min(len(actual_dates), len(actual_prices))
        actual_dates, actual_prices = actual_dates[:n], actual_prices[:n]
 
    forecast_prices = [f["predicted_price"] for f in forecasts]
    if forecasts and forecasts[0].get("date"):
        forecast_dates = to_naive_datetimeindex([f.get("date") for f in forecasts])
    else:
        forecast_dates = pd.bdate_range(start=actual_dates[-1] + pd.Timedelta(days=1), periods=len(forecast_prices))
 
    if days_ahead == 1:
        col1, col2 = st.columns(2)
        col1.metric("Last known close (USD)", f"${last_known_price:.2f}")
        col2.metric("Predicted next close (USD)", f"${forecast_prices[0]:.2f}")
    else:
        st.metric("Last known close (USD)", f"${last_known_price:.2f}")
        st.caption(
            f"Showing a {days_ahead}-day recursive forecast. Only day 1 uses real data as "
            f"input -- later days compound on the model's own prior guesses, so treat them "
            f"as directional, not precise."
        )
 
    # build a combined real + forecast chart, forecast plotted as its own series
    history = pd.Series(actual_prices, index=actual_dates, name="Actual")
    forecast_series = pd.Series(forecast_prices, index=forecast_dates, name="Forecast")
    combined = pd.concat([history, forecast_series], axis=1)
 
    st.subheader("Price history + forecast")
    st.line_chart(combined)
 
    if days_ahead > 1:
        st.dataframe(
            pd.DataFrame({"Date": [dt.date() for dt in forecast_dates], "Predicted Price": forecast_prices})
        )
 
    # ---- News + AI insight, generated by the n8n AI Agent node ----
    if insight:
        st.subheader("Market news & AI insight")
        if insight.get("news_highlight"):
            st.markdown(f"**News highlight:** {insight['news_highlight']}")
        if insight.get("ai_insight"):
            st.info(insight["ai_insight"])
 
