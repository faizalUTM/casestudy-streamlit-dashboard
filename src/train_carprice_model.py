# train_best_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("data/processed_car_dataset.csv")

# ==============================
# 2. FEATURE ENGINEERING
# ==============================
df['car_age'] = pd.Timestamp.now().year - df['make_year']
df['owner_type'] = pd.cut(df['owner_count'], bins=[0, 1, 3, float('inf')],
                          labels=['First', 'Few', 'Many'])
brand_counts = df['brand'].value_counts()
df['brand'] = np.where(df['brand'].isin(brand_counts[brand_counts < 10].index),
                       'Other', df['brand'])

# ==============================
# 3. FEATURES AND TARGET
# ==============================
features = [
    'car_age', 'engine_cc', 'mileage_kmpl',
    'brand', 'fuel_type', 'service_history', 'owner_type'
]
X = df[features]
y = df["price_usd"]

# ==============================
# 4. PREPROCESSING
# ==============================
numerical_features = ['car_age', 'engine_cc', 'mileage_kmpl']
categorical_features = ['brand', 'fuel_type', 'service_history', 'owner_type']

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

# ==============================
# 5. RANDOM FOREST MODEL
# ==============================
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42
    ))
])

# ==============================
# 6. TRAINING & VALIDATION
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Performance:")
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.3f}")

# ==============================
# 7. SAVE MODEL + METADATA
# ==============================
artifact = {
    "model": model,
    "features": features,
    "metadata": {
        "model_type": "RandomForestRegressor",
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "performance": {
            "mae": mae,
            "r2": r2
        }
    }
}

os.makedirs("models", exist_ok=True)
joblib.dump(artifact, "models/car_price_model.pkl")
print("\nModel saved as models/car_price_model.pkl")
