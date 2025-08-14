import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt  # ✅ Added

# Page config
st.set_page_config(page_title="Used Bike Price Prediction", page_icon="🏍️", layout="centered")

# Background image + styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1518655048521-f130df041f66");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

.css-1d391kg, .css-1v3fvcr, .st-bx {
    background: rgba(255, 255, 255, 0.85);
    padding: 20px;
    border-radius: 15px;
}

h1, h2, h3, h4, h5, h6, label {
    color: #1a1a1a !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load model artifacts & data
model = joblib.load(os.path.join("models", "best_model.joblib"))
scaler = joblib.load(os.path.join("models", "scaler.joblib"))
feature_names = joblib.load(os.path.join("models", "feature_names.joblib"))
cleaned_data = pd.read_csv(os.path.join("data", "cleaned_bikes.csv"))

# Title
st.markdown("<h1 style='text-align:center;'>🏍️ Used Bike Price Prediction</h1>", unsafe_allow_html=True)

# Input fields
model_year = st.number_input("Model Year", min_value=1990, max_value=2023, value=2018)
kms_driven = st.number_input("Kms Driven", min_value=0, value=20000)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=40.0)
power = st.number_input("Power (bhp)", min_value=0.0, value=20.0)
cc = st.number_input("Engine CC", min_value=50, value=150)
bike_age = 2023 - model_year

brand = st.text_input("Brand", "Bajaj")
location = st.text_input("Location", "Mumbai")
owner = st.selectbox("Owner Type", ["first", "second", "third"])

# Predict button
if st.button("🚀 Predict Price"):
    # Create input dict
    input_dict = {col: 0 for col in feature_names}

    # Fill numeric values
    input_dict["model_year"] = model_year
    input_dict["kms_driven"] = kms_driven
    input_dict["mileage"] = mileage
    input_dict["power"] = power
    input_dict["cc"] = cc
    input_dict["bike_age"] = bike_age

    # One-hot encoding
    if "brand_" + brand in input_dict:
        input_dict["brand_" + brand] = 1
    if "location_" + location in input_dict:
        input_dict["location_" + location] = 1
    if "owner_" + owner in input_dict:
        input_dict["owner_" + owner] = 1

    # DataFrame & scaling
    input_df = pd.DataFrame([input_dict])
    numeric_cols = ["model_year", "kms_driven", "mileage", "power", "cc", "bike_age"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    price_pred = model.predict(input_df)[0]

    # Show result
    st.markdown(
        f"<h2 style='text-align:center; color:#ffffff; background-color:rgba(0,0,0,0.7); padding:10px; border-radius:10px;'>💰 Estimated Price: ₹{price_pred:,.2f}</h2>",
        unsafe_allow_html=True
    )

    # Find similar bikes
    filtered = cleaned_data.copy()
    filtered = filtered[filtered["brand"].str.lower() == brand.lower()]
    filtered["year_diff"] = abs(filtered["model_year"] - model_year)
    filtered["kms_diff"] = abs(filtered["kms_driven"] - kms_driven)
    filtered["cc_diff"] = abs(filtered["cc"] - cc)
    similar_bikes = filtered.sort_values(by=["year_diff", "kms_diff", "cc_diff"]).head(5)
    similar_bikes = similar_bikes.drop(columns=["year_diff", "kms_diff", "cc_diff"])

    st.markdown("### 📊 Top 5 Similar Bikes in Dataset")
    st.dataframe(similar_bikes[["model_year", "brand", "model_name", "kms_driven", "mileage", "cc", "price"]])

    # Chart comparison
    st.markdown("### 📈 Price Comparison Chart")
    fig, ax = plt.subplots()
    labels = list(similar_bikes["model_name"]) + ["Predicted Bike"]
    prices = list(similar_bikes["price"]) + [price_pred]
    colors = ["#4682B4"] * len(similar_bikes) + ["#FF4500"]

    bars = ax.bar(labels, prices, color=colors)
    ax.set_ylabel("Price (₹)")
    ax.set_title("Predicted Price vs Similar Bikes")
    plt.xticks(rotation=45, ha="right")

    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f"₹{int(height):,}", 
                ha='center', va='bottom', fontsize=8, rotation=0)

    st.pyplot(fig)
