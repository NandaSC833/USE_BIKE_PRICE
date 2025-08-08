import joblib
import pandas as pd

# ✅ Load the trained model
model = joblib.load("models/best_model.pkl")

# ✅ Create test input as a DataFrame (not a list)
test_input = pd.DataFrame([{
    'brand': 'Royal Enfield',
    'owner': 'First Owner',
    'year': 2018,
    'kms_driven': 30000,
    'power': 20.0,
    'mileage': 45.0,
    'engine': 346,
    'segment': 'Cruiser'
}])

# ✅ Make prediction
try:
    prediction = model.predict(test_input)
    print(f"✅ Predicted Price: ₹{prediction[0]:,.2f}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

