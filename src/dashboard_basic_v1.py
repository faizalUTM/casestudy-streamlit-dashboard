import streamlit as st
import pandas as pd

st.title("Used Car Price Dashboard")

# Load and clean the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/used_car_dataset.csv")
    
    # Replace 'None' string with actual NaN
    df['service_history'].replace("None", pd.NA, inplace=True)
    
    return df

df = load_data()

# --- Show Clean Data Table ---
st.subheader("Cleaned Used Car Data")
st.dataframe(df)


