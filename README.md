# Used Bike Price Prediction

A machine learning web app that predicts the **price of a used bike** based on features like model year, mileage, power, engine capacity, brand, location, and ownership type.  
Built with **Python, scikit-learn, and Streamlit**.

---

##  Features
- Interactive **Streamlit** web app for easy predictions.
- Predicts **used bike prices** in Indian Rupees (₹).
- Dropdowns for **Brand**, **Location**, and **Owner Type** to avoid typos.
- Shows **similar bikes** in the dataset based on price range and brand.
- Price distribution chart with predicted price highlighted.
- Model trained using **Random Forest Regressor**.
- Clean UI with a custom background color.

---

##  Tech Stack
- **Python**: Pandas, NumPy, scikit-learn
- **Streamlit**: Web UI
- **Matplotlib**: Data visualization
- **Joblib**: Model saving/loading
- **GitHub**: Version control

---

## Project Structure
usedbikeprice/

│

├── data/

│ ├── bikes.csv # Original dataset

│ ├── cleaned_bikes.csv # Cleaned dataset (generated after training)


│

├── models/

│ ├── best_model.joblib # Trained model

│ ├── scaler.joblib # Scaler for numeric features

│ ├── feature_names.joblib # Saved feature names

│

├── src/

│ ├── model_training.py # Training and preprocessing script

│ ├── app.py # Streamlit web application

│

├── requirements.txt # Python dependencies

└── README.md # Project documentation

# Example Prediction

## Inputs:
Model Year: 2018
Kms Driven: 20,000
Mileage: 40 kmpl
Power: 20 bhp
CC: 150
Brand: Bajaj
Location: Mumbai
Owner: First
## Output:
Predicted Price: ₹65,000 (example)
Model Performance
Algorithm: Random Forest Regressor
R² Score: ~0.92
Mean Absolute Error: ~₹8,500

