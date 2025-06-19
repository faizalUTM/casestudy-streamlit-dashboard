import streamlit as st
import pandas as pd

st.title("Used Car Price Dashboard")

# Load and clean the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/used_car_dataset.csv")
    
    # Replace 'None' string with actual NaN
    df['service_history'].replace("None", pd.NA, inplace=True)
    
    # Optional: Convert to appropriate types or categories if needed
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

brand = st.sidebar.selectbox("Select Brand", options=["All"] + sorted(df["brand"].unique().tolist()))
if brand != "All":
    df = df[df["brand"] == brand]

fuel_type = st.sidebar.selectbox("Select Fuel Type", options=["All"] + sorted(df["fuel_type"].unique().tolist()))
if fuel_type != "All":
    df = df[df["fuel_type"] == fuel_type]

# --- Summary Metrics ---
st.subheader("Summary Metrics")
st.metric("Total Cars", len(df))
st.metric("Average Price (USD)", f"${df['price_usd'].mean():,.2f}")

# --- Histogram: Price Distribution ---
st.subheader("Price Distribution")
st.bar_chart(df["price_usd"].value_counts().head(20).sort_index())

# --- Data Table ---
st.subheader("Filtered Car Listings")
st.dataframe(df)
