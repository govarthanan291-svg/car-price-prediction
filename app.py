import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# ----------------------------------
# Load Model
# ----------------------------------
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------------
# Title
# ----------------------------------
st.markdown("<h1 style='text-align:center;'>🚗 Car Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Random Forest Regression</h4>", unsafe_allow_html=True)
st.divider()

# ----------------------------------
# User Input
# ----------------------------------
st.subheader("📋 Enter Car Details")

year = st.number_input("📅 Car Manufacturing Year", min_value=1990, max_value=2025, value=2015)
present_price = st.number_input("💵 Showroom Price (in lakhs)", min_value=0.0, value=5.0)
kms_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, value=50000)
owner = st.selectbox("👤 Number of Previous Owners", [0, 1, 2, 3])
fuel_type = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("⚙️ Transmission Type", ["Manual", "Automatic"])

# ----------------------------------
# Encoding (MUST match training)
# ----------------------------------
fuel_petrol = 1 if fuel_type == "Petrol" else 0
fuel_diesel = 1 if fuel_type == "Diesel" else 0

transmission_manual = 1 if transmission == "Manual" else 0

# ----------------------------------
# Prediction (Ensuring Correct Number of Features)
# ----------------------------------
if st.button("🔮 Predict Car Price"):

    input_data = np.array([[year,
                             present_price,
                             kms_driven,
                             owner,
                             fuel_diesel,
                             fuel_petrol,
                             transmission_manual]])

    # Check the shape of the input data (must match the number of features used during training)
    st.write(f"Input data shape: {input_data.shape}")

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Car Price: ₹ {round(prediction, 2)} Lakhs")
