import streamlit as st
import pickle
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Load model bundle ───────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("loan_approval_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()

# ── Header ──────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Approval Predictor")
st.caption(f"Model: **{bundle['model_name']}** · Fill in the applicant details below")
st.divider()

# ── Input form ──────────────────────────────────────────────────────────────────
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=0, value=120, step=10)
        loan_term = st.selectbox("Loan Amount Term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)")

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ── Prediction ──────────────────────────────────────────────────────────────────
if submitted:
    new_applicant = pd.DataFrame([{
        "Gender":             gender,
        "Married":            married,
        "Dependents":         dependents,
        "Education":          education,
        "Self_Employed":      self_employed,
        "ApplicantIncome":    float(applicant_income),
        "CoapplicantIncome":  float(coapplicant_income),
        "LoanAmount":         float(loan_amount),
        "Loan_Amount_Term":   float(loan_term),
        "Credit_History":     float(credit_history),
        "Property_Area":      property_area,
    }])

    try:
        transformed = bundle["preprocessor"].transform(new_applicant)
        transformed_df = pd.DataFrame(
            transformed,
            columns=bundle["preprocessor"].get_feature_names_out()
        )
        prediction = bundle["model"].predict(transformed_df)[0]
        decision   = bundle["label_map"][prediction]

        st.divider()
        if prediction == 1:
            st.success(f"## ✅ {decision}")
            st.markdown("The applicant is **likely to be approved** for the loan based on the provided details.")
        else:
            st.error(f"## ❌ {decision}")
            st.markdown("The applicant is **unlikely to be approved** based on the provided details.")

        with st.expander("📋 Input Summary"):
            st.dataframe(new_applicant.T.rename(columns={0: "Value"}), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Model trained on loan approval dataset")
