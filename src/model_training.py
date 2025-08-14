import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load cleaned data
df = pd.read_csv(os.path.join("data", "cleaned_bikes.csv"))

# Features & target
X = df.drop(columns=["price", "model_name"])
y = df["price"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Scale only numeric columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Save model, scaler, and feature names
os.makedirs("models", exist_ok=True)
joblib.dump(model, os.path.join("models", "best_model.joblib"))
joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
joblib.dump(list(X.columns), os.path.join("models", "feature_names.joblib"))

print("✅ Model training completed. Model & scaler saved.")

