import pandas as pd
import numpy as np
import re
import os

# Load dataset
df = pd.read_csv(os.path.join("data", "bikes.csv"))

# --- BASIC CLEANING ---
df.drop_duplicates(inplace=True)
df.columns = df.columns.str.strip()

# Extract CC from model_name
cc = []
for veh in df.model_name:
    match = re.search(r"(\d+)\s?cc", str(veh).lower())
    cc.append(int(match.group(1)) if match else np.nan)
df["cc"] = cc

# Extract brand
df["brand"] = df["model_name"].astype(str).str.split().str[0]

# Clean mileage
df["mileage"] = df["mileage"].astype(str).str.lower()
df["mileage"] = df["mileage"].str.replace("kmpl", "").str.replace("kms", "").str.strip()
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

# Fill missing mileage with brand mean
for brand in df["brand"].unique():
    mean_val = df.loc[df["brand"] == brand, "mileage"].mean()
    df.loc[(df["brand"] == brand) & (df["mileage"].isna()), "mileage"] = mean_val

# Clean power
df["power"] = df["power"].astype(str).str.lower()
df["power"] = df["power"].str.replace("bhp", "").str.replace("hp", "")
df["power"] = pd.to_numeric(df["power"], errors="coerce")
df["power"].fillna(df["power"].mean(), inplace=True)

# Clean kms_driven
df["kms_driven"] = df["kms_driven"].astype(str).str.replace("km", "").str.replace(",", "")
df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")
df["kms_driven"].fillna(df["kms_driven"].mean(), inplace=True)

# Feature: bike_age
current_year = 2023
df["bike_age"] = current_year - df["model_year"]

# Remove rows with missing price
df.dropna(subset=["price"], inplace=True)

# Save cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv(os.path.join("data", "cleaned_bikes.csv"), index=False)

print("✅ Data preprocessing completed. Saved to data/cleaned_bikes.csv")

