import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    """Load the trained XGBoost model and expected features list."""
    model = joblib.load("models/credit_risk_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

def evaluate_risk(p: float):
    """
    Evaluates the risk category based on the probability of good credit.
    p = probability of GOOD credit (Class 1)
    Thresholds:
        0 - 40%  -> High Risk
       40 - 70%  -> Medium Risk
       70%+      -> Low Risk
    """
    if p >= 0.70:
        return "Low Risk",    "success", "#4CAF50", "APPROVE"
    elif p >= 0.40:
        return "Medium Risk", "warning", "#FFA726", "MANUAL REVIEW"
    else:
        return "High Risk",   "error",   "#f44336", "REJECT"

def page_header(title: str, description: str, gradient: str = "linear-gradient(135deg, #1f4068 0%, #162447 100%)"):
    """Render a consistent, professional page header."""
    st.markdown(
        f"""
        <div style="background: {gradient};
                    padding: 25px 30px; border-radius: 12px; margin-bottom: 28px;
                    border: 1px solid rgba(255,255,255,0.06);">
            <h1 style="color:#fff; margin:0; font-family:'Outfit',sans-serif;
                       font-size:30px; font-weight:700;">{title}</h1>
            <p style="color:#a8dadc; margin:6px 0 0 0; font-size:13.5px;
                      font-family:'Inter',sans-serif;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_header(title: str, subtitle: str = ""):
    """Render a consistent section header."""
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

def render_card(title: str, content_html: str, border_color: str = "rgba(255,255,255,0.06)", bg_color: str = "rgba(255,255,255,0.02)"):
    """Render a consistent stylized card."""
    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: 20px;
                    border-radius: 12px; border: 1px solid {border_color}; height: 100%;">
            <h4 style="margin: 0 0 10px 0; color: #fff; font-size: 16px;">{title}</h4>
            <div style="font-size: 14px; color: #ccc; line-height: 1.5;">
                {content_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
