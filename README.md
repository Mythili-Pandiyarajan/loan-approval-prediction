# 🏦 Loan Approval Prediction ML Project

An end-to-end Machine Learning project to predict loan approval
using multiple classification algorithms with hyperparameter tuning.

## 🚀 Live Demo
- 🎯 Streamlit: https://loan-predictions-ml.streamlit.app/

## 🔍 Project Overview

This project compares 8 ML models (baseline + tuned):
* Logistic Regression
* Decision Tree Classifier (Baseline + Tuned)
* Random Forest Classifier (Baseline + Tuned)
* XGBoost Classifier (Baseline + Tuned)
* Artificial Neural Network(ANN)

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
* TensorFlow / Keras
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
| Logistic Regression | 0.82 | 0.86 | 0.88 | 0.87 |
| Decision Tree (Base) | 0.73 | 0.82 | 0.78 | 0.80 |
| Decision Tree (Tuned) | 0.85 | 0.83 | 0.98 | 0.90 |
| Random Forest (Base) | 0.82 | 0.84 | 0.92 | 0.88 |
| Random Forest (Tuned) | 0.85 | 0.83 | 0.98 | 0.90 |
| XGBoost (Base) | 0.80 | 0.85 | 0.87 | 0.86 |
| XGBoost (Tuned) | 0.85 | 0.83 | 0.98 | 0.90 |
| ANN | 0.85 | 0.83 | 0.97 | 0.90 |

* ✅ Best Model: **XGBoost (Tuned)** with ~85% Accuracy and F1 of 0.8922
* 📝 Fill in ANN metrics after training.

## 🔧 Pipeline Architecture

```
ColumnTransformer (preprocessor)
├── num_pipeline  → SimpleImputer(median) → StandardScaler
└── cat_pipeline  → SimpleImputer(most_frequent) → OneHotEncoder(drop='first')
```

## 🧠 ANN Architecture

* Input Layer     → [11 features after encoding]
* Dense(64)       → ReLU → Dropout(0.3)
* Dense(32)       → ReLU → Dropout(0.2)
* Output Layer    → Dense(1) → Sigmoid
* Optimizer: Adam | Loss: Binary Crossentropy

## 📌 Dataset

Dataset sourced from Kaggle — Loan Prediction Dataset  
Features: Gender, Married, Dependents, Education, Self Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

> `Credit_History` is the strongest single predictor for loan approval.

## 👩‍💻 Author

Mythili Pandiyarajan __[GitHub Profile](https://github.com/Mythili-Pandiyarajan)__
