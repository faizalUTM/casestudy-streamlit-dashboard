import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ==========================
# Page Setup
# ==========================
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction Dashboard")
st.markdown("Fill in the details below to get the estimated **car price in USD**.")

# ==========================
# Load Model
# ==========================
MODEL_PATH = "models/car_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please train the model and place it in the 'models/' folder.")
    st.stop()

model_artifact = joblib.load(MODEL_PATH)
model = model_artifact["model"]
features = model_artifact["features"]

# ==========================
# User Inputs
# ==========================
brand_options = [
    "Toyota", "Honda", "BMW", "Ford", "Hyundai", "Kia",
    "Mercedes", "Proton", "Perodua", "Other"
]
fuel_options = ["Petrol", "Diesel", "Electric", "Hybrid"]
service_options = ["Yes", "No"]
owner_options = ["First", "Few", "Many"]

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brand_options)
    fuel_type = st.selectbox("Fuel Type", fuel_options)
    service_history = st.selectbox("Service History", service_options)

with col2:
    car_age = st.slider("Car Age (years)", 0, 30, 5)
    engine_cc = st.slider("Engine Size (cc)", 800, 5000, 1500, step=100)
    mileage_kmpl = st.slider("Mileage (km/l)", 5.0, 30.0, 15.0, step=0.5)
    owner_type = st.selectbox("Owner Type", owner_options)

# ==========================
# Prediction
# ==========================
full_input_data = pd.DataFrame([{
    "car_age": car_age,
    "engine_cc": engine_cc,
    "mileage_kmpl": mileage_kmpl,
    "brand": brand,
    "fuel_type": fuel_type,
    "service_history": service_history,
    "owner_type": owner_type
}])

try:
    prediction = model.predict(full_input_data)[0]
    formatted_price = "${:,.2f}".format(prediction)

    # Show prediction result clearly
    st.subheader("💵 Estimated Car Price:")
    st.success(f"**{formatted_price}**")

    # Show input summary (without predicted price)
    display_columns = [
        "car_age", "engine_cc", "mileage_kmpl", "brand",
        "fuel_type", "service_history", "owner_type"
    ]
    display_df = full_input_data[display_columns]

    st.markdown("### 🔍 Input Summary")
    st.dataframe(display_df, use_container_width=True)

except Exception as e:
    st.error(f"Prediction failed: {e}")
