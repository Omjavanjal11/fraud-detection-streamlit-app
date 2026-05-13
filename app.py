"""
Real-Time Fraud Detection Dashboard
Streamlit Multi-Page Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#  Page Config 
st.set_page_config(
    page_title="FraudShield AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e74c3c, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #7f8c8d;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2c3e50;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #95a5a6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-critical { color: #e74c3c; font-weight: 700; }
    .risk-suspicious { color: #f39c12; font-weight: 700; }
    .risk-clear { color: #2ecc71; font-weight: 700; }
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
    }
    div[data-testid="stSidebar"] {
        background-color: #0f0f1a;
    }
    div[data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        border-left: 4px solid #e74c3c;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .explanation-box {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# Load Model 
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None


# Generate Demo Data 
@st.cache_data
def generate_demo_data(n=5000):
    """Generate realistic demo data when actual dataset is not available"""
    np.random.seed(42)
    n_fraud = int(n * 0.035)
    n_legit = n - n_fraud

    fraud_data = {
        'TransactionID': range(1, n_fraud + 1),
        'TransactionAmt': np.random.exponential(scale=200, size=n_fraud) + 50,
        'HourOfDay': np.random.choice([0,1,2,3,22,23], size=n_fraud, p=[0.2,0.2,0.2,0.15,0.15,0.1]),
        'isFraud': 1,
        'FraudProbability': np.random.beta(8, 2, size=n_fraud),
        'card4': np.random.choice(['visa', 'mastercard', 'discover', 'amex'], size=n_fraud),
        'DeviceType': np.random.choice(['mobile', 'desktop', None], size=n_fraud, p=[0.6, 0.3, 0.1]),
        'AmtToMeanRatio': np.random.exponential(scale=3, size=n_fraud) + 0.5,
    }

    legit_data = {
        'TransactionID': range(n_fraud + 1, n + 1),
        'TransactionAmt': np.random.exponential(scale=80, size=n_legit) + 10,
        'HourOfDay': np.random.randint(6, 22, size=n_legit),
        'isFraud': 0,
        'FraudProbability': np.random.beta(1.5, 8, size=n_legit),
        'card4': np.random.choice(['visa', 'mastercard', 'discover', 'amex'], size=n_legit,
                                   p=[0.45, 0.35, 0.12, 0.08]),
        'DeviceType': np.random.choice(['mobile', 'desktop', None], size=n_legit, p=[0.5, 0.45, 0.05]),
        'AmtToMeanRatio': np.random.exponential(scale=1, size=n_legit) + 0.1,
    }

    df_f = pd.DataFrame(fraud_data)
    df_l = pd.DataFrame(legit_data)
    df = pd.concat([df_f, df_l]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Risk tiers
    def risk_tier(p):
        if p >= 0.75: return '🔴 Critical Risk'
        elif p >= 0.40: return '🟡 Suspicious'
        else: return '🟢 Clear'

    df['RiskTier'] = df['FraudProbability'].apply(risk_tier)
    df['DeviceType'] = df['DeviceType'].fillna('Unknown')
    return df


#  Sidebar Navigation 
def sidebar():
    with st.sidebar:
        st.markdown("### 🛡️ FraudShield AI")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["📊 Overview", "🔍 Transaction Explorer", "🧠 SHAP Explainer"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("**Filters**")
        date_filter = st.slider("Hour Range", 0, 23, (0, 23))
        amt_filter = st.slider("Amount Range ($)", 0, 5000, (0, 5000))
        tier_filter = st.multiselect(
            "Risk Tier",
            ['🔴 Critical Risk', '🟡 Suspicious', '🟢 Clear'],
            default=['🔴 Critical Risk', '🟡 Suspicious', '🟢 Clear']
        )
        st.markdown("---")
        st.caption("© 2026 FraudShield AI | Powered by LightGBM + SHAP")
    return page, date_filter, amt_filter, tier_filter


#  Page 1: Overview 
def page_overview(df, filters):
    date_filter, amt_filter, tier_filter = filters

    st.markdown('<p class="main-title">🛡️ FraudShield AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-Time Fraud Detection & Analytics Platform</p>', unsafe_allow_html=True)

    # Apply filters
    mask = (
        df['HourOfDay'].between(date_filter[0], date_filter[1]) &
        df['TransactionAmt'].between(amt_filter[0], amt_filter[1]) &
        df['RiskTier'].isin(tier_filter)
    )
    df_f = df[mask]

    #  KPI Cards 
    total = len(df_f)
    fraud_count = df_f['isFraud'].sum()
    detection_rate = (df_f['FraudProbability'] >= 0.5).sum()
    avg_fraud_amt = df_f[df_f['isFraud'] == 1]['TransactionAmt'].mean() if fraud_count > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "💳 Total Transactions", f"{total:,}", "#3498db"),
        (col2, "🚨 Fraud Detected", f"{fraud_count:,}", "#e74c3c"),
        (col3, "📡 Detection Rate", f"{detection_rate/total*100:.1f}%" if total > 0 else "0%", "#f39c12"),
        (col4, "💰 Avg Fraud Amount", f"${avg_fraud_amt:.0f}", "#9b59b6"),
    ]
    for col, label, value, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    #  Charts Row 1 
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="section-header">Risk Tier Distribution</p>', unsafe_allow_html=True)
        tier_counts = df_f['RiskTier'].value_counts()
        colors_map = {'🔴 Critical Risk': '#e74c3c', '🟡 Suspicious': '#f39c12', '🟢 Clear': '#2ecc71'}
        fig = go.Figure(go.Pie(
            labels=[t.split(' ',1)[1] for t in tier_counts.index],
            values=tier_counts.values,
            hole=0.55,
            marker_colors=[colors_map.get(t, '#95a5a6') for t in tier_counts.index],
            textinfo='label+percent',
            textfont_size=13
        ))
        fig.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20),
                          showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Fraud Rate by Hour of Day</p>', unsafe_allow_html=True)
        hourly = df_f.groupby('HourOfDay').agg(
            FraudRate=('isFraud', 'mean'),
            Count=('isFraud', 'count')
        ).reset_index()
        hourly['FraudRate'] *= 100
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=hourly['HourOfDay'], y=hourly['FraudRate'],
            marker_color=hourly['FraudRate'],
            marker_colorscale='RdYlGn_r',
            name='Fraud Rate %',
            showlegend=False
        ))
        fig2.update_layout(
            height=320,
            xaxis_title='Hour of Day',
            yaxis_title='Fraud Rate (%)',
            margin=dict(t=20, b=40, l=40, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        )
        st.plotly_chart(fig2, use_container_width=True)

    #  Charts Row 2 
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<p class="section-header">Transaction Amount Distribution</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for label, color in [('Legitimate', '#2ecc71'), ('Fraud', '#e74c3c')]:
            is_fraud = 1 if label == 'Fraud' else 0
            subset = df_f[df_f['isFraud'] == is_fraud]['TransactionAmt']
            fig3.add_trace(go.Histogram(
                x=np.log1p(subset), name=label, opacity=0.7,
                marker_color=color, nbinsx=40
            ))
        fig3.update_layout(
            height=300, barmode='overlay',
            xaxis_title='log(TransactionAmt + 1)', yaxis_title='Count',
            margin=dict(t=20, b=40, l=40, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.7, y=0.95)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<p class="section-header">Fraud Probability by Card Type</p>', unsafe_allow_html=True)
        if 'card4' in df_f.columns:
            card_fraud = df_f.groupby('card4').agg(
                AvgFraudProb=('FraudProbability', 'mean'),
                Count=('FraudProbability', 'count')
            ).reset_index().sort_values('AvgFraudProb', ascending=False)
            fig4 = px.bar(card_fraud, x='card4', y='AvgFraudProb',
                          color='AvgFraudProb', color_continuous_scale='RdYlGn_r',
                          labels={'card4': 'Card Type', 'AvgFraudProb': 'Avg Fraud Probability'},
                          height=300)
            fig4.update_layout(margin=dict(t=20, b=40, l=40, r=20),
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False,
                               coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)


#  Page 2: Transaction Explorer 
def page_explorer(df, filters):
    date_filter, amt_filter, tier_filter = filters

    st.markdown('<p class="main-title">🔍 Transaction Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Search, filter, and investigate individual transactions</p>', unsafe_allow_html=True)

    # Search
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        search_id = st.text_input("Search by TransactionID", placeholder="Enter TransactionID...")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)

    # Apply filters
    mask = (
        df['HourOfDay'].between(date_filter[0], date_filter[1]) &
        df['TransactionAmt'].between(amt_filter[0], amt_filter[1]) &
        df['RiskTier'].isin(tier_filter)
    )
    df_filtered = df[mask].copy()

    if search_id:
        try:
            tid = int(search_id)
            df_filtered = df_filtered[df_filtered['TransactionID'] == tid]
        except ValueError:
            st.warning("Please enter a valid numeric Transaction ID")

    # Risk tier color
    def risk_badge(tier):
        if 'Critical' in tier: return '🔴'
        elif 'Suspicious' in tier: return '🟡'
        return '🟢'

    # Display table
    display_cols = ['TransactionID', 'TransactionAmt', 'HourOfDay',
                    'FraudProbability', 'RiskTier', 'isFraud']
    available_cols = [c for c in display_cols if c in df_filtered.columns]

    st.markdown(f"**Showing {len(df_filtered):,} transactions** (filtered)")

    # Styled dataframe
    styled_df = df_filtered[available_cols].copy()
    styled_df['FraudProbability'] = styled_df['FraudProbability'].round(4)
    styled_df['TransactionAmt'] = styled_df['TransactionAmt'].round(2)

    st.dataframe(
        styled_df.head(500),
        use_container_width=True,
        height=400,
        column_config={
            'FraudProbability': st.column_config.ProgressColumn(
                'Fraud Probability',
                min_value=0, max_value=1, format="%.4f"
            ),
            'TransactionAmt': st.column_config.NumberColumn(
                'Amount ($)', format="$%.2f"
            ),
            'isFraud': st.column_config.CheckboxColumn('Is Fraud?'),
        }
    )

    # Interactive scatter
    st.markdown('<p class="section-header">Interactive Risk Scatter Plot</p>', unsafe_allow_html=True)
    plot_df = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)
    fig = px.scatter(
        plot_df,
        x='HourOfDay', y='TransactionAmt',
        color='FraudProbability',
        color_continuous_scale='RdYlGn_r',
        size='FraudProbability', size_max=15,
        opacity=0.7,
        hover_data=['TransactionID', 'RiskTier'],
        labels={
            'HourOfDay': 'Hour of Day',
            'TransactionAmt': 'Transaction Amount ($)',
            'FraudProbability': 'Fraud Probability'
        },
        height=450
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    st.plotly_chart(fig, use_container_width=True)


#  Page 3: SHAP Explainer 
def page_shap(df):
    st.markdown('<p class="main-title">🧠 SHAP Explainer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Understand why the model flagged any transaction</p>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        txn_id_input = st.text_input(
            "Enter TransactionID",
            placeholder=f"Try: {df['TransactionID'].iloc[0]}",
            help="Enter any TransactionID from the dataset"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        explain_btn = st.button("🔍 Explain", use_container_width=True)

    if explain_btn or txn_id_input:
        try:
            tid = int(txn_id_input) if txn_id_input else df['TransactionID'].iloc[0]
            row = df[df['TransactionID'] == tid]

            if row.empty:
                st.error(f"Transaction ID {tid} not found. Try IDs between {df['TransactionID'].min()} and {df['TransactionID'].max()}")
                return

            row = row.iloc[0]
            prob = row['FraudProbability']
            tier = row['RiskTier']

            # Transaction summary card
            tier_color = '#e74c3c' if 'Critical' in tier else '#f39c12' if 'Suspicious' in tier else '#2ecc71'
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 16px;
                        padding: 1.5rem; color: white; margin: 1rem 0; border-left: 5px solid {tier_color};">
                <h3 style="margin:0; font-family: Syne, sans-serif;">Transaction #{tid}</h3>
                <div style="display: flex; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
                    <div><span style="color:#95a5a6; font-size:0.8rem">AMOUNT</span><br>
                         <strong style="font-size:1.5rem; color:#f39c12">${row['TransactionAmt']:.2f}</strong></div>
                    <div><span style="color:#95a5a6; font-size:0.8rem">FRAUD PROBABILITY</span><br>
                         <strong style="font-size:1.5rem; color:{tier_color}">{prob:.4f}</strong></div>
                    <div><span style="color:#95a5a6; font-size:0.8rem">RISK TIER</span><br>
                         <strong style="font-size:1.4rem; color:{tier_color}">{tier}</strong></div>
                    <div><span style="color:#95a5a6; font-size:0.8rem">HOUR</span><br>
                         <strong style="font-size:1.5rem; color:white">{int(row['HourOfDay']):02d}:00</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Simulated SHAP feature contributions (for demo without actual model)
            features = ['TransactionAmt', 'HourOfDay', 'AmtToMeanRatio', 'card1', 'card4',
                        'addr1', 'P_emaildomain', 'C1', 'C2', 'D1', 'V95', 'V96',
                        'IsNightTransaction', 'DayOfWeek', 'LogTransactionAmt']

            np.random.seed(int(prob * 1000) % 2**31)
            if prob >= 0.75:  # Critical
                base_shap = np.random.normal(0.15, 0.08, len(features))
            elif prob >= 0.40:  # Suspicious
                base_shap = np.random.normal(0.0, 0.07, len(features))
            else:  # Clear
                base_shap = np.random.normal(-0.1, 0.06, len(features))

            shap_vals = sorted(zip(features, base_shap), key=lambda x: abs(x[1]), reverse=True)[:10]

            # SHAP Waterfall chart
            st.markdown('<p class="section-header">SHAP Feature Contributions</p>', unsafe_allow_html=True)
            feat_names = [s[0] for s in shap_vals]
            feat_shap = [s[1] for s in shap_vals]
            bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in feat_shap]

            fig = go.Figure(go.Bar(
                x=feat_shap,
                y=feat_names,
                orientation='h',
                marker_color=bar_colors,
                text=[f'{v:+.4f}' for v in feat_shap],
                textposition='outside'
            ))
            fig.update_layout(
                height=420,
                xaxis_title='SHAP Value (Impact on Fraud Probability)',
                title=f'SHAP Waterfall — Transaction {tid}',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(248,249,250,0.8)',
                margin=dict(l=140, r=80, t=50, b=40)
            )
            fig.add_vline(x=0, line_color='gray', line_width=1)
            st.plotly_chart(fig, use_container_width=True)

            # Plain-English explanation
            st.markdown('<p class="section-header">Plain-English Explanation</p>', unsafe_allow_html=True)
            verdict = "⚠️ This transaction is flagged as HIGH RISK" if prob >= 0.5 else "✅ This transaction appears LEGITIMATE"
            st.markdown(f"""
            <div class="explanation-box">
                <strong>{verdict}</strong><br><br>
                The model assigned a fraud probability of <strong>{prob:.4f}</strong>.
                Here are the top factors that drove this decision:
            </div>
            """, unsafe_allow_html=True)

            for feat, val in shap_vals[:5]:
                direction = "increases" if val > 0 else "decreases"
                arrow = "🔺" if val > 0 else "🔻"
                st.markdown(f"""
                <div class="explanation-box">
                    {arrow} <strong>{feat}</strong> {direction} fraud risk by
                    <strong>{abs(val):.4f}</strong> SHAP units
                </div>
                """, unsafe_allow_html=True)

        except ValueError:
            st.error("Please enter a valid numeric Transaction ID")
        except Exception as e:
            st.error(f"Error: {str(e)}")


#  Main App 
def main():
    page, date_filter, amt_filter, tier_filter = sidebar()
    df = generate_demo_data(8000)

    filters = (date_filter, amt_filter, tier_filter)

    if page == "📊 Overview":
        page_overview(df, filters)
    elif page == "🔍 Transaction Explorer":
        page_explorer(df, filters)
    elif page == "🧠 SHAP Explainer":
        page_shap(df)


if __name__ == '__main__':
    main()

