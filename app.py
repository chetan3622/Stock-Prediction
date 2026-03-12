import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Reliance Forecast", layout="wide")

# ---------------- BACKGROUND STYLE ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #1e293b, #0f172a);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    color: white;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

.title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1 !important;
    margin-bottom: 40px;
}

.metric-box {
    text-align: center;
    padding: 20px;
    border-radius: 14px;
    background: rgba(255,255,255,0.05);
    transition: 0.3s ease;
}

.metric-box:hover {
    transform: translateY(-6px);
    background: rgba(0,245,160,0.08);
}

.stButton>button {
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    color: black;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">📈 Reliance Industries Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered SARIMA Stock Prediction Dashboard</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("sarima_model.joblib")
        return model
    except:
        st.error("Model file not found. Please place 'sarima_model.joblib' in this folder.")
        return None

model = load_model()

# ---------------- USER INPUT ----------------
days = st.slider("Select Forecast Days", 7, 365, 30)
generate = st.button("Generate Forecast")

# ---------------- FORECAST ----------------
if generate and model is not None:

    forecast = model.forecast(steps=days)

    future_dates = pd.date_range(
        start=pd.Timestamp.today(),
        periods=days,
        freq="B"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast Price": forecast.values
    })

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
        <h4>Days Forecasted</h4>
        <h2>{days}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
        <h4>Average Price</h4>
        <h2>{round(forecast_df['Forecast Price'].mean(),2)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
        <h4>Maximum Price</h4>
        <h2>{round(forecast_df['Forecast Price'].max(),2)}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- CHART ----------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast Price"],
        mode="lines",
        name="Forecast Price",
        line=dict(color="#00f5a0", width=3)
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Reliance Stock Forecast",
        xaxis_title="Date",
        yaxis_title="Predicted Price (₹)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- DATA TABLE ----------------
    st.subheader("Forecast Data")
    st.dataframe(forecast_df, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<center style='color:#94a3b8;'>Developed by Developer</center>",
    unsafe_allow_html=True
)