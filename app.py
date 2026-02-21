import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Preprocess
y = data["G3"]
X = data.drop("G3", axis=1)
X = pd.get_dummies(X, drop_first=True)

# Train model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

# Streamlit UI
st.title("🎓 Student Final Grade Predictor")

st.write("Enter student details below:")

G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
studytime = st.slider("Study Time (1-4)", 1, 4, 2)
absences = st.slider("Number of Absences", 0, 30, 5)

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

prediction = model.predict(input_df)

st.subheader("📊 Predicted Final Grade (G3)")
st.success(f"{prediction[0]:.2f}")
