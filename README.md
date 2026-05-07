# 🏦 Loan Approval Prediction ML Project

An end-to-end Machine Learning project to predict loan approval
using multiple classification algorithms with hyperparameter tuning.

## 🚀 Live Demo
- 🎯 Streamlit: 

## 🔍 Project Overview

This project compares 7 ML models (baseline + tuned):
* Logistic Regression
* Decision Tree Classifier (Baseline + Tuned)
* Random Forest Classifier (Baseline + Tuned)
* XGBoost Classifier (Baseline + Tuned)

## 📁 Project Structure

| File | Description |
|------|-------------|
| `loan_approved_pipeline.ipynb` | ML notebook with EDA, pipeline and model training |
| `app.py` | Streamlit web application |
| `loan_approval_model.pkl` | Trained ML model bundle |
| `requirements.txt` | Required libraries |

## 🛠️ Libraries Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib, Seaborn
* ydata-profiling

## 🚀 How to Run Locally

```bash
git clone https://github.com/Mythili-Pandiyarajan/loan-approval-app.git
cd loan-approval-app
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.7967 | 0.8043 | 0.9245 | 0.8602 |
| Decision Tree (Base) | 0.7642 | 0.8065 | 0.8679 | 0.8361 |
| Decision Tree (Tuned) | 0.7967 | 0.8246 | 0.8868 | 0.8545 |
| Random Forest (Base) | 0.8130 | 0.8333 | 0.8868 | 0.8593 |
| Random Forest (Tuned) | 0.8293 | 0.8475 | 0.9057 | 0.8756 |
| XGBoost (Base) | 0.8130 | 0.8475 | 0.8868 | 0.8667 |
| XGBoost (Tuned) | 0.8537 | 0.8621 | 0.9245 | 0.8922 |

> ✅ Best Model: **XGBoost (Tuned)** with ~85% Accuracy and F1 of 0.8922

## 🔧 Pipeline Architecture

```
ColumnTransformer (preprocessor)
├── num_pipeline  → SimpleImputer(median) → StandardScaler
└── cat_pipeline  → SimpleImputer(most_frequent) → OneHotEncoder(drop='first')
```

## 📌 Dataset

Dataset sourced from Kaggle — Loan Prediction Dataset  
Features: Gender, Married, Dependents, Education, Self Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

> `Credit_History` is the strongest single predictor for loan approval.

## 👩‍💻 Author

Mythili Pandiyarajan __[GitHub Profile](https://github.com/Mythili-Pandiyarajan)__
