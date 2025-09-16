# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")  # hide future warnings

# Load model, scaler, and feature names
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Load dataset for EDA
df = pd.read_csv("C:/Users/mohan/Downloads/projects/ML projects/customer churn pred/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

# Tabs: EDA and Prediction
tab1, tab2 = st.tabs(["📈 EDA Dashboard", "🔮 Prediction"])

# ---------------------------
# TAB 1: EDA Dashboard
# ---------------------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Insights from the churn dataset:")

    # Churn distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

    # Churn by contract type
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    # Distribution of Monthly Charges
    st.subheader("Distribution of Monthly Charges")
    fig, ax = plt.subplots()
    sns.histplot(df["MonthlyCharges"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------
# TAB 2: Prediction
# ---------------------------
with tab2:
    st.header("Customer Churn Prediction")
    st.sidebar.header("Customer Information")

    # Upload CSV for batch prediction
    uploaded_file = st.sidebar.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)

        # Preprocess batch
        df_batch["TotalCharges"] = pd.to_numeric(df_batch["TotalCharges"], errors="coerce").fillna(0)
        df_batch["gender"] = df_batch["gender"].map({"Female": 0, "Male": 1})
        df_batch["Partner"] = df_batch["Partner"].map({"Yes": 1, "No": 0})
        df_batch["Dependents"] = df_batch["Dependents"].map({"Yes": 1, "No": 0})
        df_batch["Contract"] = df_batch["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

        # Fill missing features
        for col in features:
            if col not in df_batch.columns:
                df_batch[col] = 0

        # Reorder and scale
        df_batch = df_batch[features]
        X_batch_scaled = scaler.transform(df_batch)

        # Predict
        df_batch["Churn_Prediction"] = model.predict(X_batch_scaled)
        df_batch["Churn_Probability"] = model.predict_proba(X_batch_scaled)[:, 1]

        st.write("✅ Batch predictions:")
        st.dataframe(df_batch.head())

        # Download CSV
        csv = df_batch.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    else:
        # Manual single-customer prediction
        gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
        senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
        partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
        dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
        tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=800.0)

        # Prepare input
        input_data = pd.DataFrame({
            "gender": [0 if gender == "Female" else 1],
            "SeniorCitizen": [senior],
            "Partner": [1 if partner == "Yes" else 0],
            "Dependents": [1 if dependents == "Yes" else 0],
            "tenure": [tenure],
            "Contract": [0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[features]

        input_scaled = scaler.transform(input_data)

        if st.sidebar.button("🔍 Predict Churn"):
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ Customer is likely to churn. (Probability: {probability:.2f})")
            else:
                st.success(f"✅ Customer is not likely to churn. (Probability: {probability:.2f})")

            # Prepare single-row CSV
            input_data["Churn_Prediction"] = prediction
            input_data["Churn_Probability"] = probability
            csv_single = input_data.to_csv(index=False)
            st.download_button(
                label="Download Prediction as CSV",
                data=csv_single,
                file_name="single_churn_prediction.csv",
                mime="text/csv"
            )
