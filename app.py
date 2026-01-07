import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

plt.style.use("dark_background")

st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide")

st.title("Telco Customer Churn Dashboard")
st.caption("Compact, clean dashboard to analyze customer churn and predict churn probability")

@st.cache_data
def load_data():
    return pd.read_csv("telco.csv")

df = load_data()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

total_customers = len(df)
churned = (df["Churn"] == "Yes").sum()
staying = (df["Churn"] == "No").sum()
churn_rate = churned / total_customers * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", total_customers)
c2.metric("Leaving", churned)
c3.metric("Staying", staying)
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.markdown("---")
st.subheader("Sample Customer Data")
st.dataframe(df.head(8), use_container_width=True)

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
for col in ["Contract", "PaymentMethod", "InternetService", "Churn"]:
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
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

st.markdown("---")
st.subheader("Customer Distribution")

fig1, ax1 = plt.subplots(facecolor="#0e1117")
sns.countplot(
    x=y.map({0: "Staying", 1: "Leaving"}),
    palette=["#2ecc71", "#e74c3c"],
    ax=ax1
)
ax1.set_facecolor("#0e1117")
ax1.set_xlabel("")
ax1.set_ylabel("Customers")
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_color("white")
st.pyplot(fig1)

st.markdown("---")
st.subheader("Model Performance")
st.metric("Accuracy", f"{accuracy:.2%}")

st.markdown("---")
st.subheader("Model Evaluation")

cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

col1, col2 = st.columns(2)

with col1:
    fig2, ax2 = plt.subplots(facecolor="#0e1117")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=["Stay", "Leave"],
        yticklabels=["Stay", "Leave"],
        cbar=False,
        ax=ax2
    )
    ax2.set_facecolor("#0e1117")
    ax2.set_xlabel("Predicted", color="white")
    ax2.set_ylabel("Actual", color="white")
    ax2.set_title("Confusion Matrix", color="white")
    ax2.tick_params(colors="white")
    st.pyplot(fig2)

with col2:
    fig3, ax3 = plt.subplots(facecolor="#0e1117")
    ax3.plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"AUC = {roc_auc:.2f}")
    ax3.plot([0, 1], [0, 1], linestyle="--", color="#e74c3c")
    ax3.set_facecolor("#0e1117")
    ax3.set_xlabel("False Positive Rate", color="white")
    ax3.set_ylabel("True Positive Rate", color="white")
    ax3.set_title("ROC Curve", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#0e1117", edgecolor="white", labelcolor="white")
    st.pyplot(fig3)

st.markdown("---")
st.subheader("Predict Customer Churn")

c1, c2, c3 = st.columns(3)

with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with c2:
    monthly = st.slider("Monthly Charges", 0, 150, 70)

with c3:
    total = st.slider("Total Charges", 0, 10000, 2000)

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
        st.error(f"Customer is likely to LEAVE (Probability: {prob:.2%})")
    else:
        st.success(f"Customer is likely to STAY (Probability: {prob:.2%})")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with Streamlit and Machine Learning</p>", unsafe_allow_html=True)
