import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("credit_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("Credit Score Prediction")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

income = st.number_input(
    "Income",
    min_value=0
)

education = st.selectbox(
    "Education",
    [
        "Associate's Degree",
        "Bachelor's Degree",
        "Doctorate",
        "High School Diploma",
        "Master's Degree"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

children = st.number_input(
    "Number of Children",
    min_value=0
)

home = st.selectbox(
    "Home Ownership",
    ["Owned", "Rented"]
)

# Predict Button
if st.button("Predict Credit Score"):

    # Create dataframe
    data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Income": [income],
        "Education": [education],
        "Marital Status": [marital_status],
        "Number of Children": [children],
        "Home Ownership": [home]
    })

    # Encode categorical columns
    categorical_cols = [
        "Gender",
        "Education",
        "Marital Status",
        "Home Ownership"
    ]

    for col in categorical_cols:
        data[col] = encoders[col].transform(data[col])

    # Prediction
    prediction = model.predict(data)

    # Convert back to original label
    result = target_encoder.inverse_transform(prediction)

    st.success(f"Predicted Credit Score: {result[0]}")