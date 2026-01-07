import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.markdown("<h1 style='text-align:center;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict whether a customer is likely to leave the company</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("telco.csv")

df = load_data()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "Churn"
]

df = df[features]

le = LabelEncoder()
for col in ["Contract", "PaymentMethod", "InternetService"]:
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown("### Model Accuracy")
st.success(f"Accuracy Score: {accuracy:.2%}")

st.markdown("---")
st.markdown("### Predict Customer Churn")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0, 150, 70)
    total = st.slider("Total Charges", 0, 10000, 2000)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

input_df = pd.DataFrame([[
    tenure,
    monthly,
    total,
    contract_map[contract],
    payment_map[payment],
    internet_map[internet]
]], columns=X.columns)

input_scaled = scaler.transform(input_df)

if st.button("Predict Churn"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"Likely to CHURN\n\nProbability: {prob:.2%}")
    else:
        st.success(f"Likely to STAY\n\nProbability: {prob:.2%}")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with Streamlit & Machine Learning</p>", unsafe_allow_html=True)
