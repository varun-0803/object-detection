import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("Diabetes Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\KML\Downloads\diabetes (1).csv")
    return df

@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    

    return model, scaler, report

model, scaler, report = train_model()

st.sidebar.header("Enter Patient Details")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

input_scaled = scaler.transform(input_data)

if st.sidebar.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    pred_prob = model.predict_proba(input_scaled)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High Risk of Diabetes (Probability: {pred_prob:.2f})")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {pred_prob:.2f})")

    st.subheader("Model Performance on Test Data")
    perf_df = pd.DataFrame(report).transpose()
    st.dataframe(perf_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

    st.markdown("Model: Logistic Regression | Scaled Input | 80-20 Split")
