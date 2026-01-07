import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction App")

@st.cache_data
def load_data():
    return pd.read_csv("telco.csv")

df = load_data()
df = df.dropna()

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
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
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write("**Confusion Matrix:**")
st.write(cm)
st.write(f"✔ Correctly identified churn customers: **{tp}**")
st.write(f"❌ Non-churn customers misclassified: **{fp}**")

st.subheader("🔮 Predict Churn for a New Customer")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(
        label=col,
        min_value=float(X[col].min()),
        max_value=float(X[col].max()),
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

if st.button("Predict Churn"):
    if prediction == 1:
        st.error(f"⚠ Likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Likely to stay (Probability: {probability:.2f})")
