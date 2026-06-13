import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_eda_dataset():
    try:
        return pd.read_csv("data/raw/german_credit_data.csv")
    except Exception:
        return None

df = load_eda_dataset()

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.05);">
        <h1 style="color:#fff; margin:0; font-family:'Outfit', sans-serif; font-size:32px;">Exploratory Data Analysis</h1>
        <p style="color:#f1faee; margin:5px 0 0 0; font-size:14px; font-family:'Inter', sans-serif;">
            Visualizing statistical distributions, correlations, and key predictors from the German Credit Dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if df is None:
    st.error("Failed to load dataset. Please ensure 'data/raw/german_credit_data.csv' is present.")
else:
    # Map raw numbers to human-readable strings for visualization
    df_viz = df.copy()
    df_viz['Checking Status'] = df_viz['laufkont'].map({
        1: "1: < 0 DM (Debt)",
        2: "2: 0 - 200 DM (Low)",
        3: "3: >= 200 DM (Med)",
        4: "4: No checking account"
    })
    df_viz['Savings Status'] = df_viz['sparkont'].map({
        1: "1: < 100 DM (Low)",
        2: "2: 100 - 500 DM",
        3: "3: 500 - 1000 DM",
        4: "4: >= 1000 DM (High)",
        5: "5: Unknown / No savings"
    })
    df_viz['Housing Status'] = df_viz['wohn'].map({
        1: "Rent",
        2: "Own",
        3: "For Free"
    })
    df_viz['Employment Duration'] = df_viz['beszeit'].map({
        1: "1: Unemployed",
        2: "2: < 1 year",
        3: "3: 1 - 4 years",
        4: "4: 4 - 7 years",
        5: "5: >= 7 years"
    })
    df_viz['Credit Risk'] = df_viz['kredit'].map({
        0: "Bad Credit (Class 0)",
        1: "Good Credit (Class 1)"
    })

    # --- BUSINESS INSIGHTS CALLOUT CARDS ---
    st.subheader("Core Business Insights")
    st.markdown("Dynamic calculations based on actual historical borrower data:")
    
    avg_amt_good = df[df['kredit'] == 1]['hoehe'].mean()
    avg_amt_bad = df[df['kredit'] == 0]['hoehe'].mean()
    avg_dur_good = df[df['kredit'] == 1]['laufzeit'].mean()
    avg_dur_bad = df[df['kredit'] == 0]['laufzeit'].mean()

    ins_col1, ins_col2 = st.columns(2)
    
    with ins_col1:
        st.markdown(
            f"""
            <div style="background-color: rgba(244, 67, 54, 0.05); padding: 20px; border-radius: 8px; border: 1px solid rgba(244, 67, 54, 0.2); height: 100%;">
                <h4 style="margin: 0 0 10px 0; color: #f44336;">Insight 1: Higher Loan Amount vs Risk</h4>
                <p style="font-size: 14px; color: #ddd; margin: 0; line-height: 1.5;">
                    Borrowers who default requested significantly larger loans on average than borrowers who successfully repaid:<br>
                    • <b>Average Defaulted Loan:</b> €{avg_amt_bad:,.2f}<br>
                    • <b>Average Approved Loan:</b> €{avg_amt_good:,.2f}<br>
                    <span style="color: #f44336; font-weight:600; display:block; margin-top:8px;">
                        Risk increase: +{(avg_amt_bad - avg_amt_good)/avg_amt_good:.1%} loan amount requested.
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with ins_col2:
        st.markdown(
            f"""
            <div style="background-color: rgba(244, 67, 54, 0.05); padding: 20px; border-radius: 8px; border: 1px solid rgba(244, 67, 54, 0.2); height: 100%;">
                <h4 style="margin: 0 0 10px 0; color: #f44336;">Insight 2: Longer Repayment Duration vs Risk</h4>
                <p style="font-size: 14px; color: #ddd; margin: 0; line-height: 1.5;">
                    Borrowers requesting extended repayment periods exhibit higher default risk:<br>
                    • <b>Average Defaulted Duration:</b> {avg_dur_bad:.2f} months<br>
                    • <b>Average Approved Duration:</b> {avg_dur_good:.2f} months<br>
                    <span style="color: #f44336; font-weight:600; display:block; margin-top:8px;">
                        Risk increase: +{(avg_dur_bad - avg_dur_good)/avg_dur_good:.1%} length of loan.
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- TARGET DISTRIBUTION ---
    st.subheader("Target Variable Distribution")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        target_counts = df['kredit'].value_counts().reset_index()
        target_counts.columns = ['Credit Risk', 'Count']
        target_counts['Percentage'] = target_counts['Count'] / len(df)
        target_counts['Credit Risk'] = target_counts['Credit Risk'].map({0: "Bad (0)", 1: "Good (1)"})
        st.dataframe(target_counts, use_container_width=True, hide_index=True)
        st.info("The dataset exhibits moderate class imbalance, with 70% of borrowers classed as Good Credit risk.")

    with col2:
        target_chart = alt.Chart(df_viz).mark_bar(cornerRadius=4).encode(
            x=alt.X('count()', title='Number of Borrowers'),
            y=alt.Y('Credit Risk:N', title='Class Label'),
            color=alt.Color('Credit Risk:N', scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), legend=None)
        ).properties(height=160)
        st.altair_chart(target_chart, use_container_width=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- NUMERICAL DISTRIBUTIONS ---
    st.subheader("Numerical Features Analysis")
    st.markdown("Explore borrower distributions segmented by credit risk class:")

    num_col1, num_col2, num_col3 = st.columns(3)
    
    with num_col1:
        st.markdown("**Age Distribution**")
        age_chart = alt.Chart(df_viz).mark_bar(opacity=0.7).encode(
            alt.X("alter:Q", bin=alt.Bin(maxbins=20), title="Age (years)"),
            alt.Y("count()", title="Count"),
            alt.Color("Credit Risk:N", scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), title="Risk")
        ).properties(height=230)
        st.altair_chart(age_chart, use_container_width=True)
        st.markdown("<span style='font-size:12px; color:#aaa;'>Min: 19 | Max: 75 | Median: 33 years</span>", unsafe_allow_html=True)
        
    with num_col2:
        st.markdown("**Credit Amount Distribution**")
        amt_chart = alt.Chart(df_viz).mark_bar(opacity=0.7).encode(
            alt.X("hoehe:Q", bin=alt.Bin(maxbins=20), title="Credit Amount (€)"),
            alt.Y("count()", title="Count"),
            alt.Color("Credit Risk:N", scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), legend=None)
        ).properties(height=230)
        st.altair_chart(amt_chart, use_container_width=True)
        st.markdown("<span style='font-size:12px; color:#aaa;'>Min: €250 | Max: €18,424 | Median: €2,319</span>", unsafe_allow_html=True)

    with num_col3:
        st.markdown("**Loan Duration Distribution**")
        dur_chart = alt.Chart(df_viz).mark_bar(opacity=0.7).encode(
            alt.X("laufzeit:Q", bin=alt.Bin(maxbins=20), title="Duration (months)"),
            alt.Y("count()", title="Count"),
            alt.Color("Credit Risk:N", scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), legend=None)
        ).properties(height=230)
        st.altair_chart(dur_chart, use_container_width=True)
        st.markdown("<span style='font-size:12px; color:#aaa;'>Min: 4 | Max: 72 | Median: 18 months</span>", unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- CATEGORICAL DISTRIBUTIONS ---
    st.subheader("Categorical Features Analysis")
    cat_col1, cat_col2 = st.columns(2)

    with cat_col1:
        st.markdown("**Checking Account Status**")
        chk_chart = alt.Chart(df_viz).mark_bar(cornerRadius=3).encode(
            y=alt.Y('Checking Status:N', title='Status'),
            x=alt.X('count()', title='Count'),
            color=alt.Color('Credit Risk:N', scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), title="Risk")
        ).properties(height=200)
        st.altair_chart(chk_chart, use_container_width=True)
        
        st.markdown("**Employment Duration**")
        emp_chart = alt.Chart(df_viz).mark_bar(cornerRadius=3).encode(
            y=alt.Y('Employment Duration:N', title='Duration'),
            x=alt.X('count()', title='Count'),
            color=alt.Color('Credit Risk:N', scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), legend=None)
        ).properties(height=200)
        st.altair_chart(emp_chart, use_container_width=True)

    with cat_col2:
        st.markdown("**Savings Account Status**")
        sav_chart = alt.Chart(df_viz).mark_bar(cornerRadius=3).encode(
            y=alt.Y('Savings Status:N', title='Status'),
            x=alt.X('count()', title='Count'),
            color=alt.Color('Credit Risk:N', scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), legend=None)
        ).properties(height=200)
        st.altair_chart(sav_chart, use_container_width=True)

        st.markdown("**Housing Status**")
        house_chart = alt.Chart(df_viz).mark_bar(cornerRadius=3).encode(
            y=alt.Y('Housing Status:N', title='Status'),
            x=alt.X('count()', title='Count'),
            color=alt.Color('Credit Risk:N', scale=alt.Scale(domain=["Bad Credit (Class 0)", "Good Credit (Class 1)"], range=['#f44336', '#4CAF50']), legend=None)
        ).properties(height=200)
        st.altair_chart(house_chart, use_container_width=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # --- CORRELATION MATRIX ---
    st.subheader("Feature Correlations Heatmap")
    st.markdown("Correlation matrix of key numerical descriptors and target variable:")
    
    corr_cols = ['alter', 'hoehe', 'laufzeit', 'rate', 'wohnzeit', 'bishkred', 'pers', 'kredit']
    corr_matrix = df[corr_cols].corr()
    corr_matrix.columns = ['Age', 'Credit Amount', 'Duration', 'Installment Rate', 'Residence Duration', 'Existing Credits', 'Dependents', 'Credit Risk']
    corr_matrix.index = corr_matrix.columns

    # Render proper seaborn heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0E1117') # Streamlit dark theme background match
    ax.set_facecolor('#0E1117')
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        ax=ax,
        linewidths=0.5,
        linecolor='#1e1e1e',
        cbar_kws={'label': 'Correlation'}
    )
    
    plt.title("Feature Correlation Matrix", color='white', fontsize=16, pad=20)
    plt.xticks(color='#ddd', rotation=45, ha='right')
    plt.yticks(color='#ddd', rotation=0)
    
    # Style colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='#ddd')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ddd')
    cbar.ax.yaxis.label.set_color('#ddd')

    st.pyplot(fig)
    
    st.markdown("<span style='font-size:12px; color:#aaa;'>*Red indicates positive correlation, blue indicates negative correlation. Note the negative correlation between Duration/Amount and Credit Risk target (lower target class 0 = Bad Credit).*</span>", unsafe_allow_html=True)

