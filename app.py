# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")  # hide future warnings

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

# ---------------------------
# Step 1: Load ML model, scaler, features
# ---------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ---------------------------
# Step 2: Load dataset for EDA
# ---------------------------
uploaded_file_eda = st.file_uploader("Upload CSV for EDA (optional)", type=["csv"], key="eda_uploader")
if uploaded_file_eda:
    df = pd.read_csv(uploaded_file_eda)
else:
    try:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")  # local fallback
    except FileNotFoundError:
        st.warning("Please upload the CSV file for EDA to proceed.")
        st.stop()

# Preprocess
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ---------------------------
# Step 3: Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📈 EDA Dashboard", "🔮 Prediction"])

# ---------------------------
# TAB 1: EDA Dashboard (Cloud-safe)
# ---------------------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Insights from the churn dataset:")

    # Ensure dataset is not empty
    if df.empty:
        st.error("❌ Dataset is empty. Please upload a valid CSV for EDA.")
        st.stop()

    st.write("Columns in dataset:", df.columns.tolist())
    st.write("Preview of dataset:")
    st.dataframe(df.head())

    # Ensure 'Churn' column exists
    if "Churn" not in df.columns:
        st.error("❌ 'Churn' column not found. Please upload a CSV with this column.")
    else:
        df["Churn"] = df["Churn"].astype(str)

        # Churn distribution
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=df[df["Churn"].notna()], ax=ax)
        st.pyplot(fig)

    # Churn by Contract Type
    if "Contract" in df.columns and "Churn" in df.columns:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots()
        sns.countplot(x="Contract", hue="Churn", data=df.dropna(subset=["Contract", "Churn"]), palette="Set2", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ 'Contract' or 'Churn' column missing. Skipping this plot.")

    # Distribution of Monthly Charges
    if "MonthlyCharges" in df.columns:
        st.subheader("Distribution of Monthly Charges")
        fig, ax = plt.subplots()
        sns.histplot(df["MonthlyCharges"].dropna(), bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ 'MonthlyCharges' column missing. Skipping this plot.")

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 0:
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns found for correlation heatmap.")

# ---------------------------
# TAB 2: Prediction
# ---------------------------
with tab2:
    st.header("Customer Churn Prediction")
    st.sidebar.header("Customer Information")

    # --- Batch prediction via CSV ---
    uploaded_file_batch = st.sidebar.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_uploader")
    
    if uploaded_file_batch:
        df_batch = pd.read_csv(uploaded_file_batch)

        # Preprocess batch
        if "TotalCharges" in df_batch.columns:
            df_batch["TotalCharges"] = pd.to_numeric(df_batch["TotalCharges"], errors="coerce").fillna(0)
        if "gender" in df_batch.columns:
            df_batch["gender"] = df_batch["gender"].map({"Female": 0, "Male": 1})
        if "Partner" in df_batch.columns:
            df_batch["Partner"] = df_batch["Partner"].map({"Yes": 1, "No": 0})
        if "Dependents" in df_batch.columns:
            df_batch["Dependents"] = df_batch["Dependents"].map({"Yes": 1, "No": 0})
        if "Contract" in df_batch.columns:
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
        # --- Manual single-customer prediction ---
        gender = st.sidebar.selectbox("Gender", ["Female", "Male"], key="gender")
        senior = st.sidebar.selectbox("Senior Citizen", [0, 1], key="senior")
        partner = st.sidebar.selectbox("Partner", ["Yes", "No"], key="partner")
        dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"], key="dependents")
        tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12, key="tenure")
        contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0, key="monthly")
        total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=800.0, key="total")

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

        if st.sidebar.button("🔍 Predict Churn", key="predict_btn"):
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
