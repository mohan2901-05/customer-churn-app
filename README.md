
# Customer Churn Prediction App 📊

## Overview
This project is a **Machine Learning-based Customer Churn Prediction** system built using **Python**, **scikit-learn**, and **Streamlit**. The app predicts whether a customer is likely to churn (leave the service) based on their account and demographic details.  

It also provides **EDA visualizations** and allows **single or batch predictions** with downloadable CSV reports.

---

## Features
- **Exploratory Data Analysis (EDA):**  
  - Churn distribution  
  - Churn by contract type  
  - Monthly charges distribution  
  - Correlation heatmap  

- **Churn Prediction:**  
  - Predict for **single customer** via input form  
  - Predict for **multiple customers** via CSV upload  
  - **Downloadable predictions** as CSV  

- **Machine Learning Models:**  
  - Random Forest Classifier (trained with scikit-learn)  
  - Scaled input features for better accuracy  

---

## Dataset
- **Source:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Columns include:**  
  - Gender, SeniorCitizen, Partner, Dependents, Tenure, Contract, MonthlyCharges, TotalCharges, Churn  

---

## Installation
1. Clone the repository:
```bash
git clone <your_repo_url>
cd customer-churn-prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage
1. **Train the model** (optional if `churn_model.pkl` exists):
```bash
python churn_model.py
```

2. **Run the Streamlit app:**
```bash
streamlit run app.py
```

3. **App Features:**
- **EDA Tab:** Explore dataset insights  
- **Prediction Tab:**  
  - Enter single customer info or upload CSV for batch prediction  
  - Download prediction results as CSV  

---

## Sample CSV
You can use the provided sample CSV (`sample_customers.csv`) to test **batch predictions**.

---

## Screenshots
*(Add screenshots of your app running, EDA plots, and prediction results here)*

---

## Deployment
The app can be deployed for free using **Streamlit Community Cloud**:
1. Push your repository to GitHub  
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/)  
3. Create a **New App** from GitHub repository  
4. Select **`app.py`** as the main file  
5. Deploy and share the live URL  

---

## License
This project is for educational purposes.
