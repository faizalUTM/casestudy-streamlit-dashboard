import pandas as pd
import streamlit as st

# Function to load the processed data
@st.cache_data
def load_data(filepath="data/processed_car_dataset.csv"):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        st.error("Processed dataset not found. Please run the processing script first.")
        return pd.DataFrame()

# Main function for Iteration 1
def main():
    st.title("Used Car Price Dashboard_Simple")
    st.write("Explore used car listings. Search by any field.")

    data = load_data()
    if data.empty:
        st.warning("No data to display. Please ensure the dataset is processed and available.")
        return

    # Display full dataset
    st.subheader("Full Dataset")
    st.dataframe(data)

    # General search
    st.subheader("Search Records")
    search_term = st.text_input("Search any value (e.g., brand, fuel_type, transmission):")
    if search_term:
        filtered_data = data[data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
        st.write(f"Results for '{search_term}':")
        st.dataframe(filtered_data)

if __name__ == "__main__":
    main()
