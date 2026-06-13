import streamlit as st

# 1. Page Config
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Pages setup
home_page = st.Page("pages/1_home.py", title="Home", default=True)
prediction_page = st.Page("pages/2_prediction.py", title="Risk Assessor")
performance_page = st.Page("pages/3_performance.py", title="Model Performance")
eda_page = st.Page("pages/4_eda.py", title="Exploratory Data Analysis")
feature_importance_page = st.Page("pages/5_feature_importance.py", title="Feature Importance")

# 3. Create Navigation
pg = st.navigation({
    "Dashboard": [home_page],
    "Analytics": [eda_page, feature_importance_page, performance_page],
    "Application": [prediction_page]
})

# 4. Custom Sidebar Branding
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 15px 10px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <h2 style="margin: 0; font-size: 22px; color: #ffffff; font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: 0.5px;">Credit Risk AI</h2>
        <p style="margin: 5px 0 0 0; font-size: 13px; color: #e0e0e0; font-family: 'Inter', sans-serif;">XGBoost Decision Engine</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Information details
st.sidebar.markdown(
    """
    <div style="padding: 10px; background-color: rgba(255, 255, 255, 0.03); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); font-family: 'Inter', sans-serif;">
        <span style="font-weight: 600; color: #1E88E5; font-size: 13px;">SYSTEM STATUS</span>
        <table style="width:100%; margin-top:8px; font-size:11px; color:#aaa; border-collapse:collapse;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding:4px 0;">Model:</td>
                <td style="text-align:right; font-weight:600; color:#fff;">XGBoost Classifier</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding:4px 0;">Accuracy:</td>
                <td style="text-align:right; font-weight:600; color:#4CAF50;">81.0%</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding:4px 0;">ROC-AUC:</td>
                <td style="text-align:right; font-weight:600; color:#FFA726;">81.9%</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding:4px 0;">Dataset:</td>
                <td style="text-align:right; color:#fff;">German Credit</td>
            </tr>
            <tr>
                <td style="padding:4px 0;">Records:</td>
                <td style="text-align:right; color:#fff;">1000</td>
            </tr>
        </table>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 11px; color: #777;">
        Powered by Streamlit & XGBoost<br>
        v1.0.0 &copy; 2026
    </div>
    """,
    unsafe_allow_html=True
)

# 5. Run Navigation
pg.run()