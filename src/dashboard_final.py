import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==========================
# Page Setup
# ==========================
st.set_page_config(page_title="🚗 Used Car Price Predictor", layout="wide")
st.title("🚗 Used Car Price Predictor Dashboard")

# ==========================
# Load Data
# ==========================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/processed_car_dataset.csv")
    except FileNotFoundError:
        st.error("❌ Data file not found at: data/processed_car_dataset.csv")
        return pd.DataFrame()

df = load_data()

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    path = "models/car_price_model.pkl"
    if not os.path.exists(path):
        st.error("❌ Model not found at: models/car_price_model.pkl")
        return None
    artifact = joblib.load(path)
    return artifact["model"]

model = load_model()

# ==========================
# Sidebar Filters
# ==========================
st.sidebar.header("🔎 Filter Dataset")

# 1. 🔍 Search bar (any field)
search_query = st.sidebar.text_input("Search (Brand, Color, etc.):").strip().lower()

# 2. 📆 Year range
if "make_year" in df.columns:
    year_min, year_max = int(df["make_year"].min()), int(df["make_year"].max())
    year_range = st.sidebar.slider("Select Year Range", year_min, year_max, (year_min, year_max))
else:
    year_range = (1990, 2025)

# 3. 🚘 Brand dropdown
brand_options = ["All"] + sorted(df["brand"].dropna().unique().tolist())
selected_brand = st.sidebar.selectbox("Select Brand", brand_options)

# 4. ⛽ Fuel type dropdown
fuel_options = ["All"] + sorted(df["fuel_type"].dropna().unique().tolist())
selected_fuel = st.sidebar.selectbox("Select Fuel Type", fuel_options)

# ==========================
# Apply Filters
# ==========================
filtered_df = df.copy()

if selected_brand != "All":
    filtered_df = filtered_df[filtered_df["brand"] == selected_brand]

if selected_fuel != "All":
    filtered_df = filtered_df[filtered_df["fuel_type"] == selected_fuel]

filtered_df = filtered_df[
    (filtered_df["make_year"] >= year_range[0]) & (filtered_df["make_year"] <= year_range[1])
]

if search_query:
    mask = filtered_df.apply(lambda row: row.astype(str).str.lower().str.contains(search_query).any(), axis=1)
    filtered_df = filtered_df[mask]

# ==========================
# Summary Stats
# ==========================
st.markdown("### 📊 Summary Statistics")

col1, col2, col3 = st.columns(3)
if not filtered_df.empty:
    col1.metric("📈 Average Price", "${:,.2f}".format(filtered_df["price_usd"].mean()))
    col2.metric("🔺 Max Price", "${:,.2f}".format(filtered_df["price_usd"].max()))
    col3.metric("🔻 Min Price", "${:,.2f}".format(filtered_df["price_usd"].min()))
else:
    col1.metric("📈 Average Price", "-")
    col2.metric("🔺 Max Price", "-")
    col3.metric("🔻 Min Price", "-")

# ==========================
# Show Filtered Data
# ==========================
st.markdown("### 🚘 Filtered Car Listings")
st.dataframe(filtered_df, use_container_width=True)

# ==========================
# Line Chart (Price by Year)
# ==========================
if not filtered_df.empty:
    st.markdown("### 📉 Price Trend by Make Year")
    trend_df = filtered_df.groupby("make_year")["price_usd"].mean().reset_index()
    st.line_chart(trend_df.rename(columns={"make_year": "Year", "price_usd": "Average Price"}))

# ==========================
# Prediction Panel
# ==========================
st.markdown("---")
st.header("🔮 Predict Used Car Price")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(df["brand"].dropna().unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].dropna().unique()))
    service_history = st.selectbox("Service History", ["Yes", "No"])
    owner_type = st.selectbox("Owner Type", ["First", "Few", "Many"])

with col2:
    car_age = st.slider("Car Age (Years)", 0, 30, 5)
    engine_cc = st.slider("Engine Size (cc)", 800, 5000, 1500, step=100)
    mileage_kmpl = st.slider("Mileage (km/l)", 5.0, 30.0, 15.0, step=0.5)

# ==========================
# Run Prediction
# ==========================
if model:
    input_df = pd.DataFrame([{
        "car_age": car_age,
        "engine_cc": engine_cc,
        "mileage_kmpl": mileage_kmpl,
        "brand": brand,
        "fuel_type": fuel_type,
        "service_history": service_history,
        "owner_type": owner_type
    }])

    try:
        prediction = model.predict(input_df)[0]
        formatted_price = "${:,.2f}".format(prediction)

        st.subheader("💵 Estimated Car Price:")
        st.success(f"**{formatted_price}**")

        st.markdown("### 📋 Input Summary")
        st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.error("Prediction model not loaded.")
