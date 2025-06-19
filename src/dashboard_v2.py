import pandas as pd
import streamlit as st

# Load the processed dataset
@st.cache_data
def load_data(filepath="data/processed_car_dataset.csv"):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error("Processed dataset not found.")
        return pd.DataFrame()

# Main app function
def main():
    st.title("Used Car Price Dashboard")
    st.write("Explore, search, and filter used car listings.")

    data = load_data()
    if data.empty:
        st.warning("No data to display.")
        return

    # Sidebar filter section
    st.sidebar.header("Filter Options")

    # Search bar (in sidebar)
    search_term = st.sidebar.text_input("Search by any field (brand, color, etc.):")

    if search_term:
        data = data[data.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

    # Brand filter
    selected_brand = st.sidebar.selectbox("Select Brand", options=sorted(data["brand"].dropna().unique()))
    data = data[data["brand"] == selected_brand]

    # Make year filter
    year_range = st.sidebar.slider("Select Make Year Range",
                                   min_value=int(data["make_year"].min()),
                                   max_value=int(data["make_year"].max()),
                                   value=(2010, 2023))
    data = data[(data["make_year"] >= year_range[0]) & (data["make_year"] <= year_range[1])]

    # Fuel type filter
    selected_fuels = st.sidebar.multiselect("Select Fuel Type", options=sorted(data["fuel_type"].dropna().unique()))
    if selected_fuels:
        data = data[data["fuel_type"].isin(selected_fuels)]

    # Main section: display results
    st.subheader(f"Filtered Results ({len(data)} records)")
    st.dataframe(data)

if __name__ == "__main__":
    main()
