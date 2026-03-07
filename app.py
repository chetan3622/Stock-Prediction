import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Reliance Forecast", layout="wide")

# ---------- CLEAN ANIMATED BACKGROUND ----------
st.markdown("""
<style>

/* Animated Gradient Background */
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

/* Force readable text */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

/* Header */
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

/* Metric Cards */
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

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    color: black;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* Remove default streamlit padding gap */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">📈 Reliance Industries Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered SARIMA Stock Prediction Dashboard</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open("final_sarima_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------- CONTROLS ----------
days = st.slider("Select Forecast Days", 7, 365, 30)
generate = st.button("Generate Forecast")

# ---------- FORECAST ----------
if generate:

    forecast = model.forecast(steps=days)

    future_dates = pd.date_range(
        start=datetime.today(),
        periods=days,
        freq='B'
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast.values
    })

    st.markdown("<br>", unsafe_allow_html=True)

    # METRICS
    col1, col2, col3 = st.columns(3)

    metrics = [
        ("Days Forecasted", days),
        ("Average Price", round(forecast_df['Forecast'].mean(), 2)),
        ("Maximum Price", round(forecast_df['Forecast'].max(), 2))
    ]

    for col, (title, value) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <h4>{title}</h4>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CHART
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        mode='lines',
        line=dict(color='#00f5a0', width=3)
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(forecast_df, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("<center style='color:#94a3b8;'>Developed by Devloper</center>", unsafe_allow_html=True)
