import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------- LOAD DATA ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "student-mat.csv")

data = pd.read_csv(file_path, sep=";")

# ---------- PREPROCESS ----------
y = data["G3"]
X = data.drop("G3", axis=1)
X = pd.get_dummies(X, drop_first=True)

# ---------- TRAIN MODEL ----------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

# ---------- TITLE ----------
st.title("🎓 Student Final Grade Predictor")
st.markdown(
    """
    Predict a student's final math grade (G3) 
    based on academic performance and study behavior.
    """
)

st.markdown("---")

# ---------- SIDEBAR INPUT ----------
st.sidebar.header("📥 Input Student Details")

G1 = st.sidebar.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.sidebar.slider("Second Period Grade (G2)", 0, 20, 10)
studytime = st.sidebar.selectbox("Weekly Study Time", [1, 2, 3, 4])
absences = st.sidebar.slider("Number of Absences", 0, 30, 5)

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Student Inputs")
    st.write(f"**G1:** {G1}")
    st.write(f"**G2:** {G2}")
    st.write(f"**Study Time:** Level {studytime}")
    st.write(f"**Absences:** {absences}")

# ---------- CREATE INPUT DATA ----------
input_dict = {
    "G1": G1,
    "G2": G2,
    "studytime": studytime,
    "absences": absences
}

input_df = pd.DataFrame([input_dict])

for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X.columns]

prediction = model.predict(input_df)[0]

# ---------- OUTPUT ----------
with col2:
    st.subheader("📈 Predicted Final Grade")

    if prediction >= 16:
        st.success(f"🎉 Excellent Performance Expected: {prediction:.2f}")
    elif prediction >= 12:
        st.info(f"👍 Good Performance Expected: {prediction:.2f}")
    elif prediction >= 8:
        st.warning(f"⚠️ Average Performance Expected: {prediction:.2f}")
    else:
        st.error(f"❗ Needs Improvement: {prediction:.2f}")

st.markdown("---")

# ---------- FOOTER ----------
st.markdown(
    """
    ### 📌 About This Model
    - Model: Random Forest Regressor
    - Dataset: UCI Student Performance
    - R² Score: ~0.81
    """
)
