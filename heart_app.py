import streamlit as st
import numpy as np
import joblib
import time

# Load model
model = joblib.load('heart_model.pkl')

# --- Page config ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="💖", layout="centered")

# --- App Title ---
st.markdown("""
    <h1 style='text-align: center; color: crimson;'>💖 Heart Disease Risk Predictor</h1>
    <p style='text-align: center; font-size:18px;'>Enter the details in the sidebar to predict heart disease risk.</p>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("🧪 Patient Information")

age = st.sidebar.slider("Age", 20, 80, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trtbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalachh = st.sidebar.slider("Max Heart Rate Achieved", 60, 200, 150)
exng = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slp = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
caa = st.sidebar.selectbox("Major Vessels Colored (0–4)", [0, 1, 2, 3, 4])
thall = st.sidebar.selectbox("Thalassemia", [1, 2, 3])

# --- Prediction ---
if st.sidebar.button("💡 Predict Now"):
    st.markdown("<h4 style='text-align: center;'>Running Prediction...</h4>", unsafe_allow_html=True)
    with st.spinner('Analyzing input data...'):
        time.sleep(1.5)
        input_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                                thalachh, exng, oldpeak, slp, caa, thall]])
        prediction = model.predict(input_data)

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("💔 **High Risk Detected:** Please consult a cardiologist as soon as possible.")
    else:
        st.success("❤️ **No Heart Disease Detected:** Keep up the healthy lifestyle!")

# --- Footer ---
st.markdown("""
    <hr>
    <p style='text-align: center; font-size:13px;'>Made with ❤️ by Abdul Rafay using Streamlit</p>
""", unsafe_allow_html=True)
