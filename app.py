#app.py
import streamlit as st, requests
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

API_BASE = "https://streamlit-lstm.onrender.com"  # replace with your deployed Render backend URL
import streamlit as st
import requests
import pandas as pd
import os

# Set this via environment variable (preferred for Render/Streamlit Cloud) or .streamlit/secrets.toml
API_BASE = os.environ.get("API_BASE") or st.secrets.get("API_BASE", "https://streamlit-lstm.onrender.com")

st.set_page_config(page_title="TSLA Intelligent Forecast Dashboard", layout="wide")
st.title("📈 TSLA Smart Price Predictor & Market Insights")

if st.sidebar.button("🔄 Refresh Dashboard Data"):
    st.cache_data.clear()
st.sidebar.caption(f"Backend: {API_BASE}")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest():
    response = requests.get(f"{API_BASE}/latest", timeout=30)
    response.raise_for_status()
    return response.json()


with st.spinner("Fetching latest automated workflow results..."):
    try:
        data = fetch_latest()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            st.warning("⚠️ No workflow data found. Run your n8n workflow pipeline first!")
        else:
            st.error(f"Backend returned an error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to backend api at {API_BASE}. Error: {e}")
        st.stop()

# Parse payload
actual_hist = pd.DataFrame(data["actual_history"])   # Expected: keys 'date' and 'close'
forecasts_list = pd.DataFrame(data["forecasts"])     # Expected: keys 'date' and 'predicted_price'
news_list = data.get("news_headlines", [])
ai_insight = data.get("ai_insight", "No insight available.")
updated_at = data.get("_updated_at")

# Structure dates for plotting
actual_hist["date"] = pd.to_datetime(actual_hist["date"])
actual_hist = actual_hist.sort_values("date").set_index("date").rename(columns={"close": "Actual"})

forecasts_list["date"] = pd.to_datetime(forecasts_list["date"])
forecasts_list = forecasts_list.sort_values("date").set_index("date").rename(columns={"predicted_price": "Forecast"})

# Bridge the visual gap: prepend the last actual price so the forecast line connects to history
if not actual_hist.empty:
    bridge = pd.DataFrame({"Forecast": [actual_hist["Actual"].iloc[-1]]}, index=[actual_hist.index[-1]])
    forecast_for_chart = pd.concat([bridge, forecasts_list[["Forecast"]]])
else:
    forecast_for_chart = forecasts_list[["Forecast"]]

combined_df = pd.concat([actual_hist["Actual"], forecast_for_chart["Forecast"]], axis=1)

if updated_at:
    st.caption(f"Last updated: {updated_at} UTC")

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price History + Multi-Day Forecast Horizon")
    st.line_chart(combined_df)

    st.subheader("🔮 Multi-Day Forecast Ledger")
    st.dataframe(forecasts_list.rename(columns={"Forecast": "predicted_price"}), use_container_width=True)

with col2:
    st.subheader("🤖 AI Support & Market Insight")
    st.info(ai_insight)

    st.subheader("📰 Relevant Context Headlines")
    if not news_list:
        st.write("No corresponding contextual news fetched for this run.")
    for article in news_list[:5]:  # Display top 5 headlines
        title = article.get("title", "No Title Available")
        source = article.get("source", "Unknown Source")
        url = article.get("url", "#")
        st.markdown(f"- **[{title}]({url})** *(Source: {source})*")
