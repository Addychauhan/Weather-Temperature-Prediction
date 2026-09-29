import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib


def find_model_path():
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "best_weather_model.joblib"),        # same folder as app.py
        os.path.join(here, "..", "best_weather_model.joblib"),  # one folder up (repo-root layout)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "best_weather_model.joblib not found next to app.py or in the parent folder."
    )

MODEL_PATH = find_model_path()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
saved=load_model()
model = saved["model"]
scaler = saved["scaler"]
needs_scaling = saved["needs_scaling"]
needed = saved["features"]

st.caption(f"Model used: **{saved['model_name']}**")

# ---------- Inputs ----------
st.subheader("Weather details")

col1, col2 = st.columns(2)

with col1:
    humidity = st.slider("Humidity", 0.0, 1.0, 0.75, 0.01)
    wind_speed = st.slider("Wind speed (km/h)", 0.0, 65.0, 10.0, 0.5)
    wind_bearing = st.slider("Wind direction (degrees)", 0, 359, 180)
    visibility = st.slider("Visibility (km)", 0.0, 17.0, 10.0, 0.1)

with col2:
    pressure = st.slider("Pressure (millibars)", 950.0, 1050.0, 1015.0, 0.5)
    precip = st.selectbox("Precipitation type", ["rain", "snow"])
    month = st.slider("Month", 1, 12, 6)
    hour = st.slider("Hour of day (0-23)", 0, 23, 12)

st.subheader("Recent temperature history")
last1 = st.number_input("Temperature 1 hour ago (°C)", value=12.0, step=0.5) \
    if "Temp_lag1" in needed or "Temp_last_hour" in needed else None
last3 = st.number_input("Temperature 3 hours ago (°C)", value=12.0, step=0.5) \
    if "Temp_lag3" in needed else None
hum_last1 = st.slider("Humidity 1 hour ago", 0.0, 1.0, 0.75, 0.01) \
    if "Humidity_lag1" in needed else None
roll3 = st.number_input("Average temperature, last 3 hours (°C)", value=12.0, step=0.5) \
    if "Temp_roll3" in needed else None

# ---------- Predict ----------
if st.button("Predict Temperature"):
    available = {
        "Humidity": humidity, "Wind Speed (km/h)": wind_speed, "Wind Bearing (degrees)": wind_bearing,
        "Visibility (km)": visibility, "Pressure (millibars)": pressure,
        "Precip_rain": 1 if precip == "rain" else 0, "Month": month, "Hour": hour,
        "Temp_last_hour": last1, "Temp_lag1": last1, "Temp_lag3": last3,
        "Humidity_lag1": hum_last1, "Temp_roll3": roll3,
    }
    row = np.array([[available[f] for f in needed]])
    if needs_scaling:
        row = scaler.transform(row)

    prediction = model.predict(row)[0]

    st.success(f"Predicted temperature: **{prediction:.1f} °C**")
    if last1 is not None:
        st.metric("Change from last hour", f"{prediction:.1f} °C", delta=f"{prediction - last1:+.1f} °C")

st.divider()
with st.expander("Model performance"):
    st.dataframe(pd.DataFrame({
        "Model": ["Deep Learning", "Random Forest", "Gradient Boosting", "Decision Tree", "Linear Regression"],
        "RMSE (°C)": [0.642, 0.752, 0.720, 0.888, 0.703],
        "R² Score": [0.9950, 0.9932, 0.9938, 0.9905, 0.9940],
    }), hide_index=True, use_container_width=True)

st.caption("Typical error is under 1 °C on the test set.")