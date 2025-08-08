import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("models/best_model.pkl")
print(type(model))

# Custom styling
st.set_page_config(page_title="Bike Price Predictor", layout="centered")

st.markdown("<h1 style='text-align: center; color: teal;'>🛵 Used Bike Price Predictor</h1>", unsafe_allow_html=True)

st.sidebar.header("Enter Bike Details 👇")

# User Inputs
brand = st.sidebar.selectbox("Brand", ['Bajaj', 'Honda', 'Royal Enfield', 'Yamaha', 'TVS', 'Hero'])
owner = st.sidebar.selectbox("Number of Owners", ['First Owner', 'Second Owner', 'Third Owner'])
year = st.sidebar.slider("Year of Purchase", 1990, 2023, 2018)
kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 200000, 20000)
power = st.sidebar.number_input("Engine Power (bhp)", 5.0, 50.0, 12.0)
mileage = st.sidebar.number_input("Mileage (kmpl)", 20.0, 90.0, 45.0)
engine_capacity = st.sidebar.number_input("Engine Capacity (cc)", 50, 1000, 150)
bike_type = st.sidebar.selectbox("Bike Segment", ['Commuter', 'Cruiser', 'Sport', 'Off-road', 'Scooter'])

# DataFrame for prediction
input_df = pd.DataFrame({
    'brand': [brand],
    'owner': [owner],
    'year': [year],
    'kms_driven': [kms_driven],
    'power': [power],
    'mileage': [mileage],
    'engine': [engine_capacity],
    'segment': [bike_type]
})

# Predict button
if st.sidebar.button("Predict Price 💰"):
    try:
        price = model.predict(input_df)[0]
        st.success(f"Estimated Resale Price: ₹ {np.round(price, 2):,.2f}")

        # Confidence range (±10% as approximation)
        lower = price * 0.9
        upper = price * 1.1
        st.info(f"Confidence Range: ₹ {np.round(lower):,.0f} - ₹ {np.round(upper):,.0f}")

        with st.expander("🔍 Advanced Insights"):
            st.write("**Bike Summary**")
            st.table(input_df.T)

            st.write("**Note:** This model is based on historical patterns and may vary due to local market demand.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built with ❤️ for Internship Excellence</div>", unsafe_allow_html=True)
