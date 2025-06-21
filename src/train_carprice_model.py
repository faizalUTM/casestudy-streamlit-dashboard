# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load data
df = pd.read_csv("data/processed_car_dataset.csv")

# Define features and target
X = df[[
    "make_year", "mileage_kmpl", "engine_cc", "fuel_type", "owner_count",
    "brand", "transmission", "color", "service_history", "insurance_valid"
]]
y = df["price_usd"]

# Preprocessing
categorical_features = ["fuel_type", "brand", "transmission", "color", "service_history", "insurance_valid"]
numerical_features = ["make_year", "mileage_kmpl", "engine_cc", "owner_count"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "car_price_model.pkl")
print("Model saved as car_price_model.pkl")
