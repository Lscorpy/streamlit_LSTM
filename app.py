import streamlit as st
import requests
import pandas as pd
import os

# Set this via environment variable (preferred for Render/Streamlit Cloud) or .streamlit/secrets.toml
# This is your n8n WEBHOOK trigger URL (the "Webhook - Run Pipeline" node), e.g.:
#   https://your-n8n-instance.app.n8n.cloud/webhook/run-pipeline
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL") or st.secrets.get(
    "N8N_WEBHOOK_URL", "https://jjypqr.app.n8n.cloud/webhook-test/run-pipeline"
)

st.set_page_config(page_title="TSLA Intelligent Forecast Dashboard", layout="wide")
st.title("📈 TSLA Smart Price Predictor & Market Insights")
st.sidebar.caption(f"Pipeline webhook: {N8N_WEBHOOK_URL}")

if "data" not in st.session_state:
    st.session_state.data = None

run_clicked = st.sidebar.button("🚀 Get Forecast")

if run_clicked:
    with st.spinner(
        "Running full pipeline (Marketstack → LSTM forecast → Gemini insight → "
        "news) — this can take up to a minute..."
    ):
        try:
            # Empty JSON body: n8n's flow doesn't need any input, it fetches
            # everything itself. A long timeout is important since this call
            # doesn't return until the entire n8n workflow has finished.
            response = requests.post(N8N_WEBHOOK_URL, json={}, timeout=120)
            response.raise_for_status()
            st.session_state.data = response.json()
        except requests.exceptions.Timeout:
            st.error("The pipeline took too long to respond. Please try again.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"n8n returned an error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Failed to reach n8n webhook at {N8N_WEBHOOK_URL}. Error: {e}")
            st.stop()

data = st.session_state.data

if data is None:
    st.info("Click **🚀 Get Forecast** in the sidebar to run the pipeline.")
    st.stop()

# Parse payload (same shape n8n's "Build Dashboard Payload" node assembles)
actual_hist = pd.DataFrame(data["actual_history"])   # keys: 'date', 'close'
forecasts_list = pd.DataFrame(data["forecasts"])     # keys: 'date', 'predicted_price'
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
    st.caption(f"Last run: {updated_at}")

st.subheader("Price History + Forecast")
st.line_chart(combined_df)

st.subheader("🔮 Forecast Ledger")
st.dataframe(forecasts_list.rename(columns={"Forecast": "predicted_price"}), use_container_width=True)

st.subheader("🤖 AI Market Insight")
st.info(ai_insight)

st.subheader("📰 Relevant Headlines")
if not news_list:
    st.write("No corresponding contextual news fetched for this run.")
for article in news_list[:5]:
    title = article.get("title", "No Title Available")
    source = article.get("source", "Unknown Source")
    url = article.get("url", "#")
    st.markdown(f"- **[{title}]({url})** *(Source: {source})*")
