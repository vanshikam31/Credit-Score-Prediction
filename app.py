
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered"
)

# ---------------- LOAD FILES ----------------
model = joblib.load("credit_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About This Project")

st.sidebar.info(
    """
    This application predicts an individual's credit score using
    a Machine Learning model trained on demographic and financial data.

    **Model Used:** Random Forest Classifier
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### Technologies Used
    - Python
    - Scikit-learn
    - Streamlit
    - Pandas
    - Joblib
    """
)

# ---------------- MAIN TITLE ----------------
st.title("💳 Credit Score Prediction")

st.markdown(
    """
    Predict whether a customer's credit score is
    **High, Average, or Low** using Machine Learning.
    """
)

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    income = st.number_input(
        "Income",
        min_value=0,
        value=50000
    )

with col2:
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
        min_value=0,
        value=0
    )

home = st.selectbox(
    "Home Ownership",
    ["Owned", "Rented"]
)

st.markdown("")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Credit Score"):

    data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Income": [income],
        "Education": [education],
        "Marital Status": [marital_status],
        "Number of Children": [children],
        "Home Ownership": [home]
    })

    categorical_cols = [
        "Gender",
        "Education",
        "Marital Status",
        "Home Ownership"
    ]

    for col in categorical_cols:
        data[col] = encoders[col].transform(data[col])

    prediction = model.predict(data)

    result = target_encoder.inverse_transform(prediction)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if result == "High":
        st.success("🟢 Predicted Credit Score: HIGH")
        

    elif result == "Average":
        st.warning("🟡 Predicted Credit Score: AVERAGE")

    else:
        st.error("🔴 Predicted Credit Score: LOW")

# ---------------- MODEL INFO ----------------
with st.expander("📊 Model Information"):
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Prediction Classes:** High, Average, Low")
    st.write("**Input Features:**")
    st.write(
        [
            "Age",
            "Gender",
            "Income",
            "Education",
            "Marital Status",
            "Number of Children",
            "Home Ownership"
        ]
    )

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "Developed by Varsha Mehra • Machine Learning Portfolio Project"
)