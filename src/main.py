import pandas as pd
def load_and_process_data(filepath="data/used_car_dataset.csv",
output_path="data/processed_dataset.csv"):
"""
Loads the dataset, removes duplicate rows, and saves the processed
dataset.
Args:
filepath (str): Path to the input dataset.
output_path (str): Path to save the processed dataset.
Returns:
pd.DataFrame: Processed DataFrame with no duplicates.
"""
# Load the dataset
df = pd.read_csv(filepath)
print(f"Original dataset shape: {df.shape}")
# Remove duplicate rows
df = df.drop_duplicates()
print(f"Dataset shape after removing duplicates: {df.shape}")
# Save the processed dataset
df.to_csv(output_path, index=False)
print(f"Processed dataset saved to {output_path}")
return df