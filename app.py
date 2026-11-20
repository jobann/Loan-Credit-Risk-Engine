import streamlit as st
from src.predict import CreditRiskPredictor

# ─── Page Configuration ───
st.set_page_config(
    page_title="Loan Credit Risk Underwriting Engine",
    page_icon="🏦",
    layout="wide"
)


# Cache the model so it loads once into memory instantaneously
@st.cache_resource
def get_predictor():
    return CreditRiskPredictor()


predictor = get_predictor()

# ─── Title & Header ───
st.title("🏦 Loan Credit Risk Underwriting Engine")
st.markdown("Automated Probability of Default (PD) scoring, FICO calibration, and FCRA adverse action decisioning.")
st.divider()

# ─── Quick-Fill Scenario Buttons ───
st.subheader("⚡ Quick-Load Test Profiles")
col_s1, col_s2, col_s3 = st.columns(3)

# Default values
default_income = 120000.0
default_credit = 250000.0
default_annuity = 15000.0
default_age = 45
default_employed = 8.0
default_ext = 0.65
default_active_loans = 1
default_utilization = 15.0
default_refusal_rate = 0.0
default_late_days = 0

if col_s1.button("🟢 Profile A: Prime Safe Borrower (Score ~840)"):
    st.session_state.income = 180000.0
    st.session_state.credit = 250000.0
    st.session_state.annuity = 12000.0
    st.session_state.age = 50
    st.session_state.employed = 12.0
    st.session_state.ext = 0.75
    st.session_state.active = 1
    st.session_state.util = 10.0
    st.session_state.refusal = 0.0
    st.session_state.late = 0

if col_s2.button("🟡 Profile B: Borderline Applicant (Manual Review)"):
    st.session_state.income = 60000.0
    st.session_state.credit = 280000.0
    st.session_state.annuity = 18000.0
    st.session_state.age = 32
    st.session_state.employed = 1.5
    st.session_state.ext = 0.42
    st.session_state.active = 3
    st.session_state.util = 55.0
    st.session_state.refusal = 0.25
    st.session_state.late = 12

if col_s3.button("🔴 Profile C: High-Risk Defaulter (Reject)"):
    st.session_state.income = 35000.0
    st.session_state.credit = 220000.0
    st.session_state.annuity = 19000.0
    st.session_state.age = 24
    st.session_state.employed = 0.5
    st.session_state.ext = 0.22
    st.session_state.active = 6
    st.session_state.util = 95.0
    st.session_state.refusal = 0.75
    st.session_state.late = 45

# ─── Sidebar: Underwriter Sliders ───
st.sidebar.header("📋 Applicant Profile Controls")

st.sidebar.subheader("💵 Financials")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000.0, max_value=1000000.0,
                                 value=st.session_state.get('income', default_income), step=5000.0)
credit = st.sidebar.number_input("Requested Loan Amount ($)", min_value=10000.0, max_value=2000000.0,
                                 value=st.session_state.get('credit', default_credit), step=10000.0)
annuity = st.sidebar.number_input("Annual Loan Payment ($)", min_value=1000.0, max_value=200000.0,
                                  value=st.session_state.get('annuity', default_annuity), step=1000.0)

st.sidebar.subheader("👤 Demographics")
age_years = st.sidebar.slider("Applicant Age (Years)", 20, 70, int(st.session_state.get('age', default_age)))
employed_years = st.sidebar.slider("Years at Current Job", 0.0, 40.0,
                                   float(st.session_state.get('employed', default_employed)), step=0.5)

st.sidebar.subheader("📊 Multi-Bureau & Behavioral Signals")
ext_score = st.sidebar.slider("Composite External Credit Bureau Score", 0.0, 1.0,
                              float(st.session_state.get('ext', default_ext)), step=0.01)
active_loans = st.sidebar.slider("Active Loans at Other Banks (bureau.csv)", 0, 15,
                                 int(st.session_state.get('active', default_active_loans)))
cc_utilization = st.sidebar.slider("Credit Card Utilization (credit_card.csv)", 0.0, 100.0,
                                   float(st.session_state.get('util', default_utilization)), step=1.0)
refusal_rate = st.sidebar.slider("Past Internal Bank Refusal Rate (previous_app.csv)", 0.0, 1.0,
                                 float(st.session_state.get('refusal', default_refusal_rate)), step=0.05)
max_late_days = st.sidebar.slider("Worst Past Payment Delay (installments.csv)", 0, 90,
                                  int(st.session_state.get('late', default_late_days)))

# ─── Construct Applicant Dictionary ───
applicant_dict = {
    'AMT_INCOME_TOTAL': income,
    'AMT_CREDIT': credit,
    'AMT_ANNUITY': annuity,
    'DAYS_BIRTH': -int(age_years * 365.25),
    'DAYS_EMPLOYED': -int(employed_years * 365.25),
    'EXT_SOURCE_1': ext_score,
    'EXT_SOURCE_2': ext_score,
    'EXT_SOURCE_3': ext_score,
    'BUREAU_ACTIVE_LOANS': active_loans,
    'BUREAU_TOTAL_OVERDUE': 1000.0 if max_late_days > 30 else 0.0,
    'PREV_REFUSAL_RATE': refusal_rate,
    'CC_UTILIZATION_MEAN': cc_utilization / 100.0,
    'INSTAL_MAX_DAYS_LATE': max_late_days,
    'POS_MAX_DPD': max_late_days
}

# ─── Execute Real-Time Underwriting ───
result = predictor.predict(applicant_dict)
score = result['credit_score']
pd_rate = result['probability_of_default']
decision = result['decision']
tier = result['risk_tier']

# ─── Main Panel: Decision Dashboard ───
st.divider()
st.subheader("🎯 Underwriting Verdict")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric("Estimated Credit Score", f"{score}", help="Calibrated FICO scale from 300 to 850")
metric_col2.metric("Default Risk (PD)", f"{pd_rate:.1%}", help="Estimated Probability of Default")

if decision == "APPROVE":
    metric_col3.success(f"### ✅ {decision}\n**{tier}**")
elif decision == "MANUAL REVIEW":
    metric_col3.warning(f"### ⚠️ {decision}\n**{tier}**")
else:
    metric_col3.error(f"### ❌ {decision}\n**{tier}**")

# ─── Tabs: Deep-Dive Details ───
tab1, tab2 = st.tabs(["📝 Adverse Action & Reason Codes", "📐 Financial Leverage Ratios"])

with tab1:
    st.markdown("#### Legal Underwriting Flags (FCRA / ECOA Compliance)")
    for reason in result['risk_reasons']:
        if decision == "APPROVE":
            st.info(f"✔️ {reason}")
        else:
            st.warning(f"⚠️ {reason}")

with tab2:
    dti = annuity / (income + 1e-5)
    leverage = credit / (income + 1e-5)
    payment_rate = annuity / (credit + 1e-5)

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Debt-to-Income (DTI)", f"{dti:.1%}", delta="- Safe" if dti < 0.20 else "+ High Burden",
               delta_color="inverse")
    rc2.metric("Loan-to-Income Leverage", f"{leverage:.1f}x", delta="- Moderate" if leverage < 4.0 else "+ Extreme",
               delta_color="inverse")
    rc3.metric("Annual Payment Rate", f"{payment_rate:.1%}")