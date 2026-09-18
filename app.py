import streamlit as st
import joblib
from sklearn.preprocessing import PolynomialFeatures

model = joblib.load("Ac_Electricity_Bill_Prediction.pkl")
st.title("Electricity Bill prediction Based on AC Units")
ac_units = st.number_input("Enter AC Units: ", min_value = 0.0, value = 100.0)
fan_units = st.number_input("Enter Fan Units: ", min_value = 0.0, value = 100.0)
if st.button("Predict"):
  poly = PolynomialFeatures()
  X_poly = poly.fit_transform([[ac_units, fan_units]])
  prediction = model.predict(X_poly)
  st.success(f"Predicted Electricity Bill: {prediction[0]:.2f}")
  
