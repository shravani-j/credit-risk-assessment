import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #2b3a42 0%, #3f5866 100%);
                padding: 25px 30px; border-radius: 12px; margin-bottom: 28px;
                border: 1px solid rgba(255,255,255,0.06);">
        <h1 style="color:#fff; margin:0; font-family:'Outfit',sans-serif;
                   font-size:30px; font-weight:700;">Feature Importance</h1>
        <p style="color:#a8dadc; margin:6px 0 0 0; font-size:13.5px;
                  font-family:'Inter',sans-serif;">
            Analyze which factors most strongly drive the model's credit risk decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    model = joblib.load("models/credit_risk_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

try:
    model, features_list = load_model()
    
    # Extract importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': features_list,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Top 10 features
    top_10 = importance_df.head(10)
    
    st.subheader("Top 10 Most Influential Features")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            top_10.style.format({"Importance": "{:.4f}"}).background_gradient(cmap="Blues"),
            use_container_width=True,
            hide_index=True
        )
        
    with col2:
        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=top_10,
            x='Importance',
            y='Feature',
            palette='viridis',
            ax=ax,
            hue='Feature',
            legend=False
        )
        ax.set_title("XGBoost Feature Importances (Top 10)", pad=15)
        ax.set_xlabel("Relative Importance")
        ax.set_ylabel("")
        sns.despine()
        
        st.pyplot(fig)
        
    st.divider()
    
    st.markdown("### Business Interpretation")
    st.markdown("""
    The feature importance metrics indicate the relative contribution of each applicant characteristic to the final credit risk prediction. 
    
    - **Top Drivers**: Features at the top of the chart (such as checking account status or loan duration) are the most critical factors the model uses to differentiate between good and bad credit risks. Changes in these variables have the largest impact on the final decision.
    - **Secondary Factors**: Lower-ranking features still provide nuanced predictive value but are generally less decisive on their own.
    - **Actionable Insights**: By understanding these drivers, loan officers can prioritize which information to verify most thoroughly and guide applicants on how to potentially improve their risk profile (e.g., by adjusting the loan duration or increasing their savings).
    """)

except Exception as e:
    st.error(f"Error loading model or generating feature importances: {str(e)}")
