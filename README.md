# 🎓 Student Performance Prediction using Machine Learning

## 📌 Overview

This project predicts students' final math grade (G3) using demographic, academic, and behavioral attributes.

It demonstrates a complete supervised regression pipeline including:
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Feature importance analysis
- Interactive web deployment using Streamlit

---

## 📊 Dataset

- Source: UCI Machine Learning Repository
- Dataset: Student Performance (Math)
- File: student-mat.csv
- Records: 395 students
- Features: 30+ academic & demographic attributes
- Target: Final Grade (G3)

---

## ⚙️ Methodology

1. Data Cleaning
2. One-hot encoding of categorical variables
3. Train-Test split (80/20)
4. Model training using Random Forest Regressor
5. Evaluation using MAE, RMSE, and R²
6. Feature importance analysis

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| MAE | 1.17 |
| RMSE | 1.97 |
| R² Score | 0.81 |

The model explains approximately **81% of the variation** in student final grades.

---

## 📌 Key Insights

- Previous grades (G1, G2) are the strongest predictors
- Study time positively impacts final performance
- Absences negatively impact grades
- Academic history is a strong indicator of final outcome

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## 🚀 Running the Project Locally

### 1️⃣ Install dependencies
