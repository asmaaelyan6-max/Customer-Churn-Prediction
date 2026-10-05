import streamlit as st
import pandas as pd
import joblib
import numpy as np


# =========================
# Load Model
# =========================

model = joblib.load("customer_churn_model.pkl")


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================
# Title
# =========================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn "
    "based on their demographic and service information."
)

st.divider()


# =========================
# Customer Information
# =========================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# =========================
# Services
# =========================

st.subheader("🌐 Services")

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# =========================
# Billing Information
# =========================

st.subheader("💳 Billing Information")

col1, col2 = st.columns(2)

with col1:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )


st.divider()


# =========================
# Prediction
# =========================

if st.button("🔮 Predict Churn", use_container_width=True):

    # -------------------------
    # Create Input DataFrame
    # -------------------------

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # =========================
    # Feature Engineering
    # =========================

    input_data["TenureType"] = pd.cut(
        input_data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "New",
            "Short-term",
            "Medium-term",
            "Long-term"
        ]
    )


    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    input_data["NumServices"] = (
        input_data[service_cols]
        .apply(
            lambda row: (row == "Yes").sum(),
            axis=1
        )
    )


    input_data["AvgMonthlyCharges"] = np.where(
        input_data["tenure"] > 0,
        input_data["TotalCharges"] / input_data["tenure"],
        input_data["MonthlyCharges"]
    )


    # =========================
    # Prediction
    # =========================

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # =========================
    # Display Result
    # =========================

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🔴 Customer is likely to churn.")

    else:

        st.success("🟢 Customer is likely to stay.")


    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )