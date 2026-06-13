import streamlit as st
import pandas as pd

# Load dataset to display dynamic statistics
@st.cache_data
def load_summary_data():
    try:
        df = pd.read_csv("data/raw/german_credit_data.csv")
        return df
    except Exception:
        return None

df = load_summary_data()

# Page title with custom gradient background (Hero Section)
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 30px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 30px;">
        <h1 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 38px; margin: 0; font-weight: 700;">Credit Risk Assessment System</h1>
        <p style="color: #00b4d8; font-family: 'Inter', sans-serif; font-size: 16px; margin: 10px 0 0 0; font-weight: 500; letter-spacing: 0.5px;">
            Decision Engine for Credit Risk Evaluation & Default Mitigation
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Columns for Overview and Business Problem
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.02); padding: 25px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); height: 100%; min-height: 220px;">
            <h3 style="color: #1E88E5; margin-top: 0; font-family: 'Outfit', sans-serif; font-size: 20px;">Project Overview</h3>
            <p style="color: #e0e0e0; font-size: 14.5px; line-height: 1.6; font-family: 'Inter', sans-serif;">
                This system utilizes advanced machine learning classifiers trained on historical credit data to assess the creditworthiness of applicants. By predicting the probability of default, the application serves as a key tool for credit analysts and underwriters to make quick, objective, and data-driven lending decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.02); padding: 25px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); height: 100%; min-height: 220px;">
            <h3 style="color: #1E88E5; margin-top: 0; font-family: 'Outfit', sans-serif; font-size: 20px;">Business Problem</h3>
            <p style="color: #e0e0e0; font-size: 14.5px; line-height: 1.6; font-family: 'Inter', sans-serif;">
                <b>The Challenge:</b> Commercial banks and lending companies must balance risk and reward. Over-restrictive lending leads to lost revenue, while reckless lending results in high loan default rates. <br>
                <b>The Solution:</b> A machine learning model that accurately classifies borrowers into risk categories, minimizing credit defaults while maintaining an optimized approval rate.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Dataset Summary Section
st.subheader("Dataset Summary")
st.markdown("Metrics computed dynamically from the **German Credit Dataset**:")

# Calculate stats dynamically if dataframe loaded, else use defaults
if df is not None:
    total_records = f"{len(df):,}"
    features_count = str(len(df.columns) - 1)
    good_pct = f"{(df['kredit'] == 1).mean():.1%}"
    bad_pct = f"{(df['kredit'] == 0).mean():.1%}"
else:
    total_records = "1,000"
    features_count = "20"
    good_pct = "70.0%"
    bad_pct = "30.0%"

# Metric Display Cards
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric(label="Total Records", value=total_records, help="Number of observations in the dataset")
with m_col2:
    st.metric(label="Model Features", value=features_count, help="Predictor variables used for training")
with m_col3:
    st.metric(label="Good Credit (Class 1)", value=good_pct, help="Percentage of low-risk borrowers")
with m_col4:
    st.metric(label="Bad Credit (Class 0)", value=bad_pct, help="Percentage of high-risk / defaulted borrowers")

st.markdown("<br>", unsafe_allow_html=True)

# Feature list inside structured column layouts
st.markdown(
    """
    <div style="background-color: rgba(255, 255, 255, 0.01); padding: 20px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); font-family: 'Inter', sans-serif;">
        <h4 style="margin-top:0; color: #fff; font-size:16px;">Key Predictive Features:</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin-top: 15px;">
            <div style="background-color: rgba(255,255,255,0.02); padding: 12px; border-radius: 6px; border-left: 3px solid #1e3c72;">
                <b>Financial Status</b><br>
                <span style="font-size:12.5px; color:#aaa;">Checking and savings account statuses, existing credit accounts, other installment plans.</span>
            </div>
            <div style="background-color: rgba(255,255,255,0.02); padding: 12px; border-radius: 6px; border-left: 3px solid #1e3c72;">
                <b>Loan Attributes</b><br>
                <span style="font-size:12.5px; color:#aaa;">Requested credit amount, repayment duration, loan purpose, installment rate.</span>
            </div>
            <div style="background-color: rgba(255,255,255,0.02); padding: 12px; border-radius: 6px; border-left: 3px solid #1e3c72;">
                <b>Applicant Profile</b><br>
                <span style="font-size:12.5px; color:#aaa;">Age, gender, employment duration, job type, housing status, property ownership.</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Model Summary Section
st.subheader("Model Performance Summary")
st.markdown("Overview of the trained candidate models on the test split:")

mod_col1, mod_col2, mod_col3 = st.columns(3)

with mod_col1:
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); text-align: center;">
            <span style="background-color: rgba(30, 136, 229, 0.1); color: #1E88E5; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase;">Baseline</span>
            <h4 style="margin: 10px 0 5px 0; color:#fff; font-size:18px;">Logistic Regression</h4>
            <h3 style="color:#1E88E5; margin:0 0 10px 0; font-size:28px;">74.2% <span style="font-size:14px; color:#888;">Acc.</span></h3>
            <p style="font-size: 13px; color:#aaa; margin: 0; text-align:left; line-height: 1.4;">
                • Highly interpretable parameters<br>
                • Serves as baseline model<br>
                • Underperforms on non-linear boundaries
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_col2:
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); text-align: center;">
            <span style="background-color: rgba(255, 167, 38, 0.1); color: #FFA726; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase;">Ensemble</span>
            <h4 style="margin: 10px 0 5px 0; color:#fff; font-size:18px;">Random Forest</h4>
            <h3 style="color:#FFA726; margin:0 0 10px 0; font-size:28px;">76.3% <span style="font-size:14px; color:#888;">Acc.</span></h3>
            <p style="font-size: 13px; color:#aaa; margin: 0; text-align:left; line-height: 1.4;">
                • Excellent feature significance ranking<br>
                • Resilient to data outliers<br>
                • Slightly prone to overfitting on sparse data
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with mod_col3:
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; text-align: center; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.1);">
            <span style="background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase;">Selected Model</span>
            <h4 style="margin: 10px 0 5px 0; color:#fff; font-size:18px;">XGBoost Classifier</h4>
            <h3 style="color:#4CAF50; margin:0 0 10px 0; font-size:28px;">77.5% <span style="font-size:14px; color:#888;">Acc.</span></h3>
            <p style="font-size: 13px; color:#aaa; margin: 0; text-align:left; line-height: 1.4;">
                • Optimized gradient boosting framework<br>
                • Best-in-class predictive accuracy<br>
                • Integrates complex interactions automatically
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #777;">
        Credit Risk Assessment System | Powered by Streamlit & XGBoost
    </div>
    """,
    unsafe_allow_html=True
)

