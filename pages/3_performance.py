import streamlit as st
import pandas as pd
import plotly.express as px

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1f4068 0%, #162447 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.05);">
        <h1 style="color:#fff; margin:0; font-family:'Outfit', sans-serif; font-size:32px;">Model Performance Analytics</h1>
        <p style="color:#e4e6eb; margin:5px 0 0 0; font-size:14px; font-family:'Inter', sans-serif;">
            Detailed evaluation and comparisons for candidate classifiers.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Model Comparison Table")

# Data based on the provided metrics
data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.770, 0.795, 0.810],
    "ROC-AUC": [0.809, 0.824, 0.819]
}
df = pd.DataFrame(data)

# Display styled table
st.dataframe(
    df.style.format({
        "Accuracy": "{:.3f}",
        "ROC-AUC": "{:.3f}"
    }).highlight_max(subset=["Accuracy", "ROC-AUC"], color='rgba(76, 175, 80, 0.2)'),
    use_container_width=True,
    hide_index=True
)

st.markdown("<br><hr>", unsafe_allow_html=True)

st.subheader("Performance Visualizations")

# Professional color scheme matching the app's aesthetic
color_map = {
    "Logistic Regression": "#1E88E5",
    "Random Forest": "#FFA726",
    "XGBoost": "#4CAF50"
}

col1, col2 = st.columns(2)

with col1:
    fig_acc = px.bar(
        df, 
        x="Model", 
        y="Accuracy", 
        text="Accuracy",
        title="Model Accuracy Comparison",
        color="Model",
        color_discrete_map=color_map
    )
    fig_acc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_acc.update_layout(
        yaxis_range=[0.70, 0.85], 
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"),
        title_font=dict(size=18, color="#fff"),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    fig_acc.update_xaxes(showgrid=False, title="")
    fig_acc.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="")
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    fig_roc = px.bar(
        df, 
        x="Model", 
        y="ROC-AUC", 
        text="ROC-AUC",
        title="Model ROC-AUC Comparison",
        color="Model",
        color_discrete_map=color_map
    )
    fig_roc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_roc.update_layout(
        yaxis_range=[0.75, 0.85], 
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ddd"),
        title_font=dict(size=18, color="#fff"),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    fig_roc.update_xaxes(showgrid=False, title="")
    fig_roc.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="")
    st.plotly_chart(fig_roc, use_container_width=True)
