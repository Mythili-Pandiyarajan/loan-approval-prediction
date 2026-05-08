import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSense — AI Approval Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf0;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

/* ── Hero ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #6b7280;
    font-weight: 300;
    margin-bottom: 2.5rem;
}

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1.2rem;
}

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
label { color: #9ca3af !important; font-size: 0.85rem !important; }

/* ── Metrics row ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
}
.metric-box {
    flex: 1;
    background: #1f2937;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid #374151;
}
.metric-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #e8eaf0; }
.metric-sub   { font-size: 0.72rem; color: #6b7280; margin-top: 2px; }

/* ── Result banner ── */
.result-approved {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #16a34a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, #1c0a0a, #450a0a);
    border: 1px solid #dc2626;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-sub { font-size: 0.95rem; color: #9ca3af; }

/* ── Risk bar ── */
.risk-bar-bg {
    background: #1f2937;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    margin: 0.8rem 0;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Validation pills ── */
.pill-pass {
    display: inline-block;
    background: #052e16;
    border: 1px solid #16a34a;
    color: #4ade80;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px 4px 3px 0;
}
.pill-fail {
    display: inline-block;
    background: #1c0a0a;
    border: 1px solid #dc2626;
    color: #f87171;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px 4px 3px 0;
}
.pill-warn {
    display: inline-block;
    background: #1c1200;
    border: 1px solid #d97706;
    color: #fbbf24;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px 4px 3px 0;
}

/* ── Submit button ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Divider ── */
hr { border-color: #1f2937 !important; margin: 1.5rem 0 !important; }

/* ── Disclaimer ── */
.disclaimer {
    font-size: 0.75rem;
    color: #374151;
    text-align: center;
    margin-top: 3rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model Bundle ───────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    pkl_path = "loan_approval_model.pkl"
    if not os.path.exists(pkl_path):
        st.error("❌ `loan_approval_model.pkl` not found. Place it in the same folder as app.py.")
        st.stop()
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

bundle = load_bundle()

# Pull emi_config from bundle if available, else use defaults
cfg = bundle.get("emi_config", {
    "annual_interest_rate": 0.09,
    "emi_ratio_threshold":  0.40,
    "loan_to_income_limit": 20,
    "loan_amount_unit":     1000,
})

# ─── Helper: compute EMI ─────────────────────────────────────────────────────
def compute_emi(loan_amount, loan_term_months):
    monthly_rate   = cfg["annual_interest_rate"] / 12
    principal      = loan_amount * cfg["loan_amount_unit"]
    if loan_term_months <= 0 or monthly_rate == 0:
        return principal / max(loan_term_months, 1)
    emi = (principal * monthly_rate * (1 + monthly_rate) ** loan_term_months) / \
          ((1 + monthly_rate) ** loan_term_months - 1)
    return emi


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">LoanSense</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Loan Approval Engine · EMI Affordability · Debt-to-Income Risk Validation</div>', unsafe_allow_html=True)

col_badge1, col_badge2, col_badge3, _ = st.columns([1, 1, 1, 3])
with col_badge1:
    st.markdown('<span class="pill-pass">⚡ XGBoost Model</span>', unsafe_allow_html=True)
with col_badge2:
    st.markdown('<span class="pill-pass">🔒 EMI Check</span>', unsafe_allow_html=True)
with col_badge3:
    st.markdown('<span class="pill-pass">📊 Risk Score</span>', unsafe_allow_html=True)

st.markdown("---")

# ─── Input Form ──────────────────────────────────────────────────────────────
with st.form("loan_form"):

    # ── Personal Details ──
    st.markdown('<div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        married = st.selectbox("Married", ["Yes", "No"])
    with c3:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    with c4:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    c5, c6 = st.columns(2)
    with c5:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    with c6:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.markdown("---")

    # ── Financial Details ──
    st.markdown('<div class="card-title">💰 Financial Details</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        applicant_income = st.number_input(
            "Applicant Income (₹/month)", min_value=0, value=5000, step=500
        )
    with f2:
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹/month)", min_value=0, value=0, step=500
        )

    f3, f4, f5 = st.columns(3)
    with f3:
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)", min_value=1.0, value=120.0, step=10.0
        )
    with f4:
        loan_term = st.selectbox(
            "Loan Term (months)",
            [360, 180, 480, 300, 240, 120, 84, 60, 36],
            index=0
        )
    with f5:
        credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                      format_func=lambda x: "Good (1)" if x == 1.0 else "Poor (0)")

    st.markdown("---")
    submitted = st.form_submit_button("🔍 Analyse & Predict")


# ─── On Submit ───────────────────────────────────────────────────────────────
if submitted:

    # ── Pre-flight validations ──────────────────────────────────────────────
    total_income = applicant_income + coapplicant_income
    checks = []
    hard_stop = False
    hard_stop_reason = ""

    # 1. Income zero check
    if total_income == 0:
        hard_stop = True
        hard_stop_reason = "Total income cannot be zero. Please enter valid income details."
        checks.append(("❌ Income > 0", "fail", "Income is zero"))
    else:
        checks.append(("✅ Income > 0", "pass", f"₹{total_income:,.0f}/month"))

    # Initialize so metrics block never crashes on early hard stop
    loan_to_income_ratio = 0.0
    emi       = 0.0
    emi_ratio = 0.0

    if not hard_stop:
        # 2. Loan-to-income ratio
        loan_to_income_ratio = loan_amount / total_income
        limit = cfg["loan_to_income_limit"]
        if loan_to_income_ratio > limit:
            hard_stop = True
            hard_stop_reason = f"Loan amount is too high relative to income (ratio: {loan_to_income_ratio:.1f}x, limit: {limit}x)."
            checks.append(("❌ Loan-to-Income", "fail", f"{loan_to_income_ratio:.1f}x > {limit}x limit"))
        else:
            checks.append(("✅ Loan-to-Income", "pass", f"{loan_to_income_ratio:.2f}x (limit {limit}x)"))

        # 3. EMI affordability — only run if loan-to-income passed
        emi       = compute_emi(loan_amount, loan_term)
        emi_ratio = emi / total_income
        threshold = cfg["emi_ratio_threshold"]

        if hard_stop:
            # Loan-to-income already failed — show EMI pill as skipped, don't overwrite reason
            checks.append(("— EMI Affordability", "warn", "Skipped (loan-to-income failed first)"))
        elif emi_ratio > threshold:
            hard_stop = True
            hard_stop_reason = f"EMI (₹{emi:,.0f}/month) exceeds {int(threshold*100)}% of income. High financial risk."
            checks.append(("❌ EMI Affordability", "fail",
                           f"EMI ratio {emi_ratio:.1%} > {threshold:.0%} threshold"))
        elif emi_ratio > threshold * 0.8:
            checks.append(("⚠️ EMI Affordability", "warn",
                           f"EMI ratio {emi_ratio:.1%} — approaching limit"))
        else:
            checks.append(("✅ EMI Affordability", "pass",
                           f"EMI ratio {emi_ratio:.1%} (threshold {threshold:.0%})"))

    # ── Display validation pills ────────────────────────────────────────────
    st.markdown("### Validation Checks")
    pills_html = ""
    for label, status, detail in checks:
        cls = {"pass": "pill-pass", "fail": "pill-fail", "warn": "pill-warn"}[status]
        pills_html += f'<span class="{cls}" title="{detail}">{label}</span>'
    st.markdown(pills_html, unsafe_allow_html=True)

    if hard_stop:
        st.markdown(f"""
        <div class="result-rejected" style="margin-top:1.5rem;">
            <div class="result-title" style="color:#f87171;">❌ Not Eligible</div>
            <div class="result-sub">{hard_stop_reason}</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── ML Prediction ───────────────────────────────────────────────────────
    dep_val = str(dependents).replace("3+", "3")

    # Check if preprocessor expects engineered features
    preprocessor    = bundle["preprocessor"]
    expected_cols   = [t[2] for t in preprocessor.transformers]
    num_cols_needed = expected_cols[0] if expected_cols else []
    has_engineered  = "Total_Income" in num_cols_needed

    input_data = {
        "Gender":            gender,
        "Married":           married,
        "Dependents":        float(dep_val),
        "Education":         education,
        "Self_Employed":     self_employed,
        "ApplicantIncome":   float(applicant_income),
        "CoapplicantIncome": float(coapplicant_income),
        "LoanAmount":        float(loan_amount),
        "Loan_Amount_Term":  float(loan_term),
        "Credit_History":    float(credit_history),
        "Property_Area":     property_area,
    }

    if has_engineered:
        input_data["Total_Income"]   = float(total_income)
        input_data["Loan_to_Income"] = float(loan_to_income_ratio)
        input_data["EMI_Ratio"]      = float(emi_ratio)

    input_df    = pd.DataFrame([input_data])
    transformed = preprocessor.transform(input_df)
    trans_df    = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())

    ml_prediction = bundle["model"].predict(trans_df)[0]
    prob          = bundle["model"].predict_proba(trans_df)[0][1]
    risk_score    = round(float(prob) * 100, 2)

    # Final decision: ML + EMI gate
    final_decision = (
        "Approved" if (ml_prediction == 1 and emi_ratio <= cfg["emi_ratio_threshold"])
        else "Rejected"
    )

    st.markdown("---")

    # ── Result Banner ───────────────────────────────────────────────────────
    if final_decision == "Approved":
        st.markdown(f"""
        <div class="result-approved">
            <div class="result-title" style="color:#4ade80;">✅ Loan Approved</div>
            <div class="result-sub">Application meets all credit and affordability criteria</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <div class="result-title" style="color:#f87171;">❌ Loan Rejected</div>
            <div class="result-sub">Application does not meet the required approval criteria</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Score Bar ──────────────────────────────────────────────────────
    bar_color = "#4ade80" if risk_score >= 60 else "#fbbf24" if risk_score >= 40 else "#f87171"
    st.markdown(f"""
    <div style="margin-bottom:0.3rem;">
        <span style="font-family:'Syne',sans-serif;font-size:0.8rem;color:#6b7280;
                     letter-spacing:1.5px;text-transform:uppercase;">Approval Probability</span>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                     color:{bar_color};float:right;">{risk_score:.2f}%</span>
    </div>
    <div class="risk-bar-bg">
        <div class="risk-bar-fill" style="width:{risk_score}%;background:{bar_color};"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Row ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-label">Total Income</div>
            <div class="metric-value">₹{total_income:,.0f}</div>
            <div class="metric-sub">per month</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Monthly EMI</div>
            <div class="metric-value">₹{emi:,.0f}</div>
            <div class="metric-sub">{cfg['annual_interest_rate']*100:.0f}% annual rate</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">EMI / Income</div>
            <div class="metric-value">{emi_ratio:.1%}</div>
            <div class="metric-sub">threshold {cfg['emi_ratio_threshold']:.0%}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Loan-to-Income</div>
            <div class="metric-value">{loan_to_income_ratio:.2f}x</div>
            <div class="metric-sub">limit {cfg['loan_to_income_limit']}x</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detail Breakdown ────────────────────────────────────────────────────
    with st.expander("📋 Full Application Summary"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Personal**")
            st.write(f"- Gender: {gender}")
            st.write(f"- Married: {married}")
            st.write(f"- Dependents: {dependents}")
            st.write(f"- Education: {education}")
            st.write(f"- Self Employed: {self_employed}")
            st.write(f"- Property Area: {property_area}")
        with col_b:
            st.markdown("**Financial**")
            st.write(f"- Applicant Income: ₹{applicant_income:,}")
            st.write(f"- Co-applicant Income: ₹{coapplicant_income:,}")
            st.write(f"- Loan Amount: ₹{loan_amount*1000:,.0f}")
            st.write(f"- Loan Term: {loan_term} months ({loan_term/12:.1f} yrs)")
            st.write(f"- Credit History: {'Good' if credit_history == 1.0 else 'Poor'}")
            st.write(f"- Model Used: {bundle.get('model_name','XGBoost_Tuned')}")

# ─── Disclaimer ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    LoanSense is a demonstration tool powered by a machine learning model trained on historical data.<br>
    Predictions are for illustrative purposes only and do not constitute financial advice.<br>
   © 2025 LoanSense · Built by Mythili · Streamlit & XGBoost</div>
""", unsafe_allow_html=True)
