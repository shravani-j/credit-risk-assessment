import streamlit as st
import joblib
import pandas as pd

# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("models/credit_risk_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

model, features_list = load_model()

# ── Risk evaluation (updated thresholds) ────────────────────────────────────
def evaluate_risk(p: float):
    """
    p = probability of GOOD credit (Class 1)
    Thresholds:
        0 – 40%  → High Risk
       40 – 70%  → Medium Risk
       70%+      → Low Risk
    """
    if p >= 0.70:
        return "Low Risk",    "success", "#4CAF50", "APPROVE"
    elif p >= 0.40:
        return "Medium Risk", "warning", "#FFA726", "MANUAL REVIEW"
    else:
        return "High Risk",   "error",   "#f44336", "REJECT"

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #2b3a42 0%, #3f5866 100%);
                padding: 25px 30px; border-radius: 12px; margin-bottom: 28px;
                border: 1px solid rgba(255,255,255,0.06);">
        <h1 style="color:#fff; margin:0; font-family:'Outfit',sans-serif;
                   font-size:30px; font-weight:700;">Credit Risk Assessor</h1>
        <p style="color:#a8dadc; margin:6px 0 0 0; font-size:13.5px;
                  font-family:'Inter',sans-serif;">
            Complete all sections below and click <strong>Run Assessment</strong>
            to receive an instant risk evaluation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Value maps ───────────────────────────────────────────────────────────────
checking_map = {
    "No checking account":                          4,
    "Negative balance (< 0 DM)":                   1,
    "Low balance (0 – 200 DM)":                    2,
    "Adequate balance (≥ 200 DM / salary-assigned)": 3,
}

savings_map = {
    "No savings account / unknown":  5,
    "Very low  (< 100 DM)":          1,
    "Low  (100 – 500 DM)":           2,
    "Medium  (500 – 1,000 DM)":      3,
    "High  (≥ 1,000 DM)":            4,
}

credit_history_map = {
    "All credits paid back on time":                              2,
    "All credits at this bank paid back duly":                    1,
    "No credits taken / all paid back":                           0,
    "Past delays in repayment":                                   3,
    "Critical account / credits exist elsewhere":                 4,
}

loan_purpose_map = {
    "Furniture / Equipment":   2,
    "Radio / Television":      3,
    "New car":                 0,
    "Used car":                1,
    "Business":                9,
    "Education":               6,
    "Repairs":                 5,
    "Domestic appliances":     4,
    "Retraining":              8,
    "Other":                   10,
}

employment_map = {
    "Unemployed":          1,
    "Less than 1 year":    2,
    "1 to 4 years":        3,
    "4 to 7 years":        4,
    "7 years or more":     5,
}

installment_rate_map = {
    "Less than 20% of income":   4,
    "20 – 25% of income":        3,
    "25 – 35% of income":        2,
    "More than 35% of income":   1,
}

personal_status_map = {
    "Male – Single":               3,
    "Male – Married / Widowed":    4,
    "Male – Divorced / Separated": 1,
    "Female – Divorced / Married": 2,
}

guarantor_map = {
    "None":           1,
    "Co-applicant":   2,
    "Guarantor":      3,
}

property_map = {
    "Real estate (house / land)":              1,
    "Building society / life insurance":       2,
    "Car or other movable assets":             3,
    "Unknown / no property":                   4,
}

installment_plans_map = {
    "None":     3,
    "Bank":     1,
    "Stores":   2,
}

housing_map = {
    "Owned":        2,
    "Rented":       1,
    "Provided free": 3,
}

job_map = {
    "Skilled employee / official":               3,
    "Management / self-employed / highly qualified": 4,
    "Unskilled – resident":                      2,
    "Unskilled – non-resident / unemployed":     1,
}

residence_map = {
    "4 years or more":  4,
    "2 to 3 years":     3,
    "1 to 2 years":     2,
    "Less than 1 year": 1,
}

people_liable_map = {
    "0 to 2 dependents":       1,
    "3 or more dependents":    2,
}

telephone_map = {
    "No registered telephone":              1,
    "Yes – registered under applicant":     2,
}

foreign_worker_map = {
    "Yes": 1,
    "No":  2,
}

# ── Section header helper ─────────────────────────────────────────────────────
def section(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="margin: 22px 0 10px 0; padding-bottom: 6px;
                    border-bottom: 1px solid rgba(255,255,255,0.08);">
            <span style="font-size:13px; font-weight:700; text-transform:uppercase;
                         letter-spacing:1px; color:#1E88E5;">{title}</span>
            {"<br><span style='font-size:12px; color:#888;'>" + subtitle + "</span>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FORM  –  two-column layout throughout
# ═══════════════════════════════════════════════════════════════════════════════

# ── A. Loan Details ───────────────────────────────────────────────────────────
section("Loan Details", "Information about the credit being applied for")

col_a, col_b = st.columns(2)
with col_a:
    credit_amount = st.number_input(
        "Credit Amount (€)",
        min_value=250, max_value=20_000, value=3_000, step=250,
        help="Total principal amount the applicant is requesting. "
             "Dataset range: €250 – €18,000.",
    )
    loan_purpose_label = st.selectbox(
        "Purpose of Loan",
        options=list(loan_purpose_map.keys()),
        index=0,
        help="Intended use of the credit funds.",
    )
    installment_rate_label = st.selectbox(
        "Instalment Rate",
        options=list(installment_rate_map.keys()),
        index=0,
        help="Monthly repayment as a percentage of the applicant's net disposable income.",
    )

with col_b:
    loan_duration = st.number_input(
        "Loan Duration (months)",
        min_value=4, max_value=72, value=24, step=1,
        help="Agreed repayment period. Longer durations are associated with higher default rates.",
    )
    guarantor_label = st.selectbox(
        "Guarantor / Co-debtor",
        options=list(guarantor_map.keys()),
        index=0,
        help="Whether another party is jointly liable for repayment.",
    )
    installment_plans_label = st.selectbox(
        "Other Active Instalment Plans",
        options=list(installment_plans_map.keys()),
        index=0,
        help="Ongoing repayment obligations to other lenders or retail stores.",
    )

# ── B. Financial Profile ──────────────────────────────────────────────────────
section("Financial Profile", "Applicant's current banking and credit standing")

col_c, col_d = st.columns(2)
with col_c:
    checking_label = st.selectbox(
        "Checking Account Status",
        options=list(checking_map.keys()),
        index=0,
        help="Current balance category of the applicant's primary checking account. "
             "A healthy balance significantly reduces predicted risk.",
    )
    credit_history_label = st.selectbox(
        "Credit History",
        options=list(credit_history_map.keys()),
        index=0,
        help="The applicant's track record of repaying past credit obligations.",
    )
    existing_credits = st.selectbox(
        "Number of Existing Credits at this Bank",
        options=[1, 2, 3, 4],
        index=0,
        help="Count of active loan or credit accounts the applicant already holds "
             "with this institution.",
    )

with col_d:
    savings_label = st.selectbox(
        "Savings Account / Bonds",
        options=list(savings_map.keys()),
        index=0,
        help="Total value of savings or bond holdings. Higher reserves reduce risk.",
    )
    property_label = st.selectbox(
        "Most Valuable Property / Asset",
        options=list(property_map.keys()),
        index=0,
        help="Primary collateral or asset class owned by the applicant.",
    )

# ── C. Personal & Employment ───────────────────────────────────────────────────
section("Personal & Employment Profile", "Demographic and employment information")

col_e, col_f = st.columns(2)
with col_e:
    age = st.number_input(
        "Age (years)",
        min_value=18, max_value=100, value=35,
        help="Applicant's age in whole years. Older applicants tend to show lower default rates.",
    )
    employment_label = st.selectbox(
        "Duration at Current Employer",
        options=list(employment_map.keys()),
        index=2,
        help="Length of continuous employment at the applicant's current job. "
             "Longer tenure correlates with lower credit risk.",
    )
    job_label = st.selectbox(
        "Job Classification",
        options=list(job_map.keys()),
        index=0,
        help="Occupational category reflecting skill level and income stability.",
    )
    personal_status_label = st.selectbox(
        "Personal Status & Gender",
        options=list(personal_status_map.keys()),
        index=0,
        help="Marital status and gender as encoded in the original German Credit dataset.",
    )

with col_f:
    housing_label = st.selectbox(
        "Housing Situation",
        options=list(housing_map.keys()),
        index=0,
        help="Whether the applicant owns, rents, or occupies housing at no cost.",
    )
    residence_label = st.selectbox(
        "Time at Current Address",
        options=list(residence_map.keys()),
        index=0,
        help="Duration of residence at the applicant's present address.",
    )
    people_liable_label = st.selectbox(
        "Number of Financial Dependents",
        options=list(people_liable_map.keys()),
        index=0,
        help="People the applicant is financially responsible for (e.g., children, partner).",
    )
    telephone_label = st.selectbox(
        "Telephone Registration",
        options=list(telephone_map.keys()),
        index=0,
        help="Whether a telephone is registered in the applicant's name — "
             "used as a proxy for stability.",
    )
    foreign_worker_label = st.selectbox(
        "Foreign Worker",
        options=list(foreign_worker_map.keys()),
        index=0,
        help="Whether the applicant is a foreign national working in Germany.",
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT BUTTON
# ═══════════════════════════════════════════════════════════════════════════════
run = st.button("Run Assessment", use_container_width=True, type="primary")

if run:
    # Build the feature dictionary and align column order to features.pkl
    record = {
        "checking_account_status": checking_map[checking_label],
        "loan_duration_months":    int(loan_duration),
        "credit_history":          credit_history_map[credit_history_label],
        "loan_purpose":            loan_purpose_map[loan_purpose_label],
        "credit_amount":           float(credit_amount),
        "savings_account_status":  savings_map[savings_label],
        "employment_duration":     employment_map[employment_label],
        "installment_rate":        installment_rate_map[installment_rate_label],
        "personal_status_sex":     personal_status_map[personal_status_label],
        "guarantors":              guarantor_map[guarantor_label],
        "residence_duration":      residence_map[residence_label],
        "property":                property_map[property_label],
        "age":                     int(age),
        "other_installment_plans": installment_plans_map[installment_plans_label],
        "housing":                 housing_map[housing_label],
        "existing_credits":        int(existing_credits),
        "job":                     job_map[job_label],
        "people_liable":           people_liable_map[people_liable_label],
        "telephone":               telephone_map[telephone_label],
        "foreign_worker":          foreign_worker_map[foreign_worker_label],
    }

    sample = pd.DataFrame([record])[features_list]

    prob_good = float(model.predict_proba(sample)[0][1])
    prob_bad  = 1.0 - prob_good

    risk_label, status_type, risk_color, recommendation = evaluate_risk(prob_good)

    # ── Results header card ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background-color: rgba(255,255,255,0.02); padding: 24px 28px;
                    border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);
                    margin-bottom: 24px;">
            <div style="display:flex; justify-content:space-between;
                        align-items:center; flex-wrap:wrap; gap:16px;">
                <div>
                    <p style="margin:0; font-size:12px; font-weight:700;
                               text-transform:uppercase; letter-spacing:1px;
                               color:#888;">Risk Category</p>
                    <p style="margin:4px 0 0 0; font-size:36px; font-weight:700;
                               font-family:'Outfit',sans-serif;
                               color:{risk_color};">{risk_label}</p>
                </div>
                <div style="text-align:right;">
                    <p style="margin:0; font-size:12px; font-weight:700;
                               text-transform:uppercase; letter-spacing:1px;
                               color:#888;">Decision</p>
                    <p style="margin:4px 0 0 0; font-size:36px; font-weight:700;
                               font-family:'Outfit',sans-serif;
                               color:{risk_color};">{recommendation}</p>
                </div>
            </div>
            <!-- Probability bar -->
            <div style="margin-top:20px; height:10px; border-radius:5px;
                        background:rgba(255,255,255,0.06); overflow:hidden;
                        display:flex;">
                <div style="width:{prob_good*100:.1f}%; background:{risk_color};
                             height:100%; transition:width .4s;"></div>
                <div style="width:{prob_bad*100:.1f}%; background:rgba(244,67,54,.35);
                             height:100%;"></div>
            </div>
            <p style="margin:6px 0 0 0; font-size:11px; color:#666;">
                Good-credit probability bar &nbsp;|&nbsp;
                Green = good credit &nbsp;|&nbsp; Red = bad credit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── st.metric row ───────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(
        label="Good Credit Probability",
        value=f"{prob_good:.1%}",
        help="Model-estimated probability that this applicant will repay the loan.",
    )
    mc2.metric(
        label="Default Probability",
        value=f"{prob_bad:.1%}",
        help="Complementary probability of non-repayment.",
    )
    mc3.metric(
        label="Risk Category",
        value=risk_label,
        help="0 – 40% = High Risk  |  40 – 70% = Medium Risk  |  70%+ = Low Risk",
    )
    mc4.metric(
        label="Recommendation",
        value=recommendation,
        help="Suggested lending decision based on the model output.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Narrative callout ───────────────────────────────────────────────────
    msg = {
        "success": (
            f"**APPROVED — Low Risk.** "
            f"The model assigns a good-credit probability of **{prob_good:.1%}**, "
            f"which exceeds the 70% approval threshold. "
            f"Standard lending terms are recommended."
        ),
        "warning": (
            f"**MANUAL REVIEW — Medium Risk.** "
            f"The good-credit probability of **{prob_good:.1%}** falls within the "
            f"moderate-risk band (40 – 70%). "
            f"Recommended actions: request additional collateral, obtain a co-signer, "
            f"or reduce the loan amount before approval."
        ),
        "error": (
            f"**REJECTED — High Risk.** "
            f"The good-credit probability of **{prob_good:.1%}** is below the 40% "
            f"minimum threshold. The default probability stands at **{prob_bad:.1%}**. "
            f"Approval is not recommended under current terms."
        ),
    }
    getattr(st, status_type)(msg[status_type])

    # ── Contextual comparison ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color:rgba(30,136,229,0.04); padding:18px 22px;
                    border-radius:8px; border:1px solid rgba(30,136,229,0.15);
                    font-family:'Inter',sans-serif;">
            <p style="margin:0 0 10px 0; font-size:13px; font-weight:700;
                      color:#1E88E5; text-transform:uppercase; letter-spacing:.8px;">
                Statistical Context
            </p>
            <p style="font-size:13.5px; color:#ccc; margin:0; line-height:1.7;">
                This applicant requested <strong>€{credit_amount:,.0f}</strong>
                over <strong>{loan_duration} months</strong>.<br>
                Historical dataset averages:<br>
                &nbsp;&nbsp;• Good-credit borrowers — avg. <strong>€2,856</strong>
                  over <strong>19.2 months</strong><br>
                &nbsp;&nbsp;• Bad-credit borrowers &nbsp;&nbsp;— avg.
                  <strong>€4,045</strong> over <strong>24.9 months</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Risk legend ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex; gap:20px; flex-wrap:wrap; font-size:12.5px;
                    color:#888; font-family:'Inter',sans-serif;">
            <span><span style="color:#4CAF50; font-weight:700;">&#9632;</span>
                  Low Risk — probability ≥ 70%</span>
            <span><span style="color:#FFA726; font-weight:700;">&#9632;</span>
                  Medium Risk — 40% ≤ probability &lt; 70%</span>
            <span><span style="color:#f44336; font-weight:700;">&#9632;</span>
                  High Risk — probability &lt; 40%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
