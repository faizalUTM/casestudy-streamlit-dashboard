import pandas as pd
import os

def load_and_process_data(filepath="data/used_car_dataset.csv", output_path="data/processed_car_dataset.csv"):
    """
    Loads the dataset, removes duplicate rows, treats specific string values as NaN,
    and saves the processed dataset.

    Args:
        filepath (str): Path to the input dataset.
        output_path (str): Path to save the processed dataset.

    Returns:
        pd.DataFrame: Processed DataFrame with no duplicates and correctly parsed NaNs.
    """
    # Load the dataset with specific values treated as NaN
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath, na_values=["None", "", "none", "NA", "N/A"])
    print(f"Original dataset shape: {df.shape}")

    # Optional: Check null count in service_history
    if 'service_history' in df.columns:
        null_count = df['service_history'].isnull().sum()
        print(f"Missing values in 'service_history': {null_count}")

    # Remove duplicate rows
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

    # Save the processed dataset
    output_path = os.path.abspath(output_path)
    try:
        print(f"Attempting to save to: {output_path}")
        df.to_csv(output_path, index=False)
        print("Processed dataset saved successfully!")
    except Exception as e:
        print(f"Failed to save file. Error: {e}")

    return df

if __name__ == "__main__":
    load_and_process_data(filepath="data/used_car_dataset.csv", output_path="data/processed_car_dataset.csv")
