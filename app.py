import streamlit as st
import joblib
from sklearn.preprocessing import PolynomialFeatures

model = joblib.load("AC_Electricity_Bill_Prediction.pkl")
st.title("Electricity Bill prediction Based on AC Units")
ac_units = st.number_input("Enter AC Units: ", min_value = 0.0, value = 100.0)
if st.button("Predict"):
  poly = PolynomialFeatures()
  ac_units_poly = poly.fit_transform([[ac_units]])
  prediction = model.predict(ac_units_poly)
  st.success(f"Predicted Electricity Bill: {prediction[0]:.2f}")
  
