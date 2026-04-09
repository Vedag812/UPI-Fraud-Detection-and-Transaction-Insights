"""
Streamlit dashboard. Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1F4E79; margin-bottom: 0.5rem; }
    .sub-header  { font-size: 1.1rem; color: #6b7280; margin-bottom: 1.5rem; }
    .real-badge  { background: #dcfce7; color: #166534; padding: 2px 10px; border-radius: 999px;
                   font-size: 0.75rem; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    flagged = os.path.join(base_dir, 'data', 'processed', 'upi_transactions_flagged.csv')
    processed = os.path.join(base_dir, 'data', 'processed', 'upi_transactions_processed.csv')
    sample = os.path.join(base_dir, 'data', 'sample', 'sample_flagged.csv')
    if os.path.exists(flagged):
        return pd.read_csv(flagged, parse_dates=['timestamp'])
    elif os.path.exists(processed):
        return pd.read_csv(processed, parse_dates=['timestamp'])
    elif os.path.exists(sample):
        # fallback for Streamlit Cloud (50K row sample)
        return pd.read_csv(sample, parse_dates=['timestamp'])
    else:
        st.error("No data found. Run the pipeline first.")
        st.stop()


@st.cache_data
def load_comparison():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'data', 'processed', 'method_comparison.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_real_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    npci_path = os.path.join(base_dir, 'data', 'real_npci_stats.csv')
    rbi_path = os.path.join(base_dir, 'data', 'real_rbi_fraud_data.csv')
    npci = pd.read_csv(npci_path) if os.path.exists(npci_path) else None
    rbi = pd.read_csv(rbi_path) if os.path.exists(rbi_path) else None
    return npci, rbi


def render_overview(df):
    st.markdown('<p class="main-header">Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Synthetic simulation of 1M UPI transactions across Jan-Dec 2024</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Total Value", f"Rs {df['amount'].sum()/1e7:.1f} Cr")
    c3.metric("Fraud Rate", f"{df['is_fraud'].mean()*100:.2f}%")
    c4.metric("Avg Transaction", f"Rs {df['amount'].mean():,.0f}")
    c5.metric("Fraud Value", f"Rs {df[df['is_fraud']==1]['amount'].sum()/1e5:.1f} L")

    st.divider()
    left, right = st.columns([2, 1])

    with left:
        daily = df.groupby(df['timestamp'].dt.date).agg(
            count=('transaction_id', 'count'), value=('amount', 'sum')
        ).reset_index()
        daily.columns = ['date', 'count', 'value']
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['count'],
                      name='Transaction Count', line=dict(color='#667eea', width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['value'],
                      name='Volume (Rs)', line=dict(color='#f093fb', width=2)), secondary_y=True)
        fig.update_layout(title='Daily Volume & Value Trend', height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with right:
        type_counts = df['transaction_type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                     title='Transaction Mix', color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if 'hour' in df.columns and 'day_of_week' in df.columns:
        hourly = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        pivot = hourly.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot.index = [days[i] for i in pivot.index]
        fig = px.imshow(pivot, color_continuous_scale='YlOrRd',
                       title='Transaction Heatmap (Day x Hour)',
                       labels=dict(x='Hour of Day', y='Day', color='Count'))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_fraud_intelligence(df):
    st.markdown('<p class="main-header">Fraud Intelligence</p>', unsafe_allow_html=True)

    fraud_df = df[df['is_fraud'] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Fraud Cases", f"{len(fraud_df):,}")
    c2.metric("Total Fraud Value", f"Rs {fraud_df['amount'].sum():,.0f}")
    c3.metric("Avg Fraud Amount", f"Rs {fraud_df['amount'].mean():,.0f}")
    if 'ensemble_flag' in df.columns:
        c4.metric("Ensemble Flagged", f"{df['ensemble_flag'].sum():,}")

    st.divider()
    left, right = st.columns(2)

    with left:
        if 'fraud_type' in df.columns:
            ft = fraud_df['fraud_type'].value_counts().reset_index()
            ft.columns = ['type', 'count']
            ft['type'] = ft['type'].str.replace('_', ' ').str.title()
            fig = px.bar(ft, x='count', y='type', orientation='h',
                        title='Fraud Cases by Pattern Type',
                        color='count', color_continuous_scale='Reds')
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if 'hour' in df.columns:
            hf = df.groupby('hour').agg(total=('is_fraud', 'count'), fraud=('is_fraud', 'sum')).reset_index()
            hf['fraud_rate'] = hf['fraud'] / hf['total'] * 100
            fig = px.bar(hf, x='hour', y='fraud_rate', title='Fraud Rate by Hour of Day (%)',
                        color='fraud_rate', color_continuous_scale='RdYlGn_r')
            fig.add_vrect(x0=0.5, x1=5.5, fillcolor='red', opacity=0.08,
                         annotation_text='High-risk 1-5 AM', annotation_position='top left')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detection Method Performance")
    comp = load_comparison()
    if comp is not None:
        fig = go.Figure()
        colors = ['#667eea', '#f093fb', '#f5576c']
        for i, m in enumerate(['precision', 'recall', 'f1_score']):
            fig.add_trace(go.Bar(name=m.replace('_', ' ').title(),
                                 x=comp['method'], y=comp[m],
                                 marker_color=colors[i % len(colors)]))
        fig.update_layout(barmode='group', title='Precision / Recall / F1 by Method',
                         height=400, template='plotly_white', yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Suspicious Accounts")
    suspicious = fraud_df.groupby('sender_upi_id').agg(
        fraud_count=('is_fraud', 'sum'), total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean')
    ).sort_values('fraud_count', ascending=False).head(15).reset_index()
    st.dataframe(suspicious, use_container_width=True)


def render_bank_analytics(df):
    st.markdown('<p class="main-header">Bank & City Analytics</p>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        bc = df['sender_bank'].value_counts().head(10)
        fig = px.treemap(names=bc.index, parents=[''] * len(bc), values=bc.values,
                        title='Bank Market Share (by Transaction Count)',
                        color=bc.values, color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        bf = df.groupby('sender_bank').agg(
            total=('status', 'count'),
            fraud=('is_fraud', 'sum')
        ).reset_index()
        bf['fraud_rate'] = (bf['fraud'] / bf['total'] * 100).round(2)
        bf = bf.sort_values('fraud_rate', ascending=True)
        fig = px.bar(bf, x='fraud_rate', y='sender_bank', orientation='h',
                    title='Fraud Rate by Sender Bank (%)',
                    color='fraud_rate', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("City-wise Analysis")
    cs = df[df['city'] != 'Unknown'].groupby('city').agg(
        txn_count=('transaction_id', 'count'),
        total_value=('amount', 'sum'),
        fraud_count=('is_fraud', 'sum')).reset_index()
    cs['fraud_rate'] = (cs['fraud_count'] / cs['txn_count'] * 100).round(2)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cs.sort_values('txn_count', ascending=False), x='city', y='txn_count',
                    title='Transaction Volume by City',
                    color='txn_count', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(cs, x='txn_count', y='fraud_rate', size='total_value',
                        color='fraud_rate', hover_name='city',
                        title='Fraud Rate vs Transaction Volume (bubble = total value)',
                        color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)


def render_real_market(df):
    st.markdown('<p class="main-header">Real Market Context</p>', unsafe_allow_html=True)
    st.info("This page shows REAL data from official NPCI and RBI sources to put our simulation in context.")

    npci, rbi = load_real_data()

    if npci is not None:
        st.subheader("NPCI Monthly UPI Statistics (Apr 2023 - Dec 2024)")
        st.caption("Source: NPCI UPI Ecosystem Statistics — official monthly reports")

        npci['MonthLabel'] = npci['month'] + ' ' + npci['year'].astype(str)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=npci['MonthLabel'], y=npci['volume_millions'],
                             name='Volume (Mn transactions)', marker_color='#667eea'), secondary_y=False)
        fig.add_trace(go.Scatter(x=npci['MonthLabel'], y=npci['value_crores'],
                                 name='Value (Rs Crore)', line=dict(color='#f093fb', width=3),
                                 mode='lines+markers'), secondary_y=True)
        fig.update_layout(title='UPI Monthly Volume & Value — Real India Market Data',
                          height=420, template='plotly_white')
        fig.update_yaxes(title_text="Volume (Millions)", secondary_y=False)
        fig.update_yaxes(title_text="Value (Rs Crore)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        peak_idx = npci['volume_millions'].idxmax()
        c1.metric("Peak Month Volume", f"{npci['volume_millions'].max():.0f}M txns",
                  delta=f"{npci['month'].iloc[peak_idx]} {npci['year'].iloc[peak_idx]}")
        c2.metric("Peak Month Value", f"Rs {npci['value_crores'].max()/100:.1f}K Cr")
        c3.metric("Avg Ticket Size (2024)", f"Rs {npci[npci['year']==2024]['avg_ticket_size_rs'].mean():.0f}")

    if rbi is not None:
        st.divider()
        st.subheader("RBI Annual Digital Payment Fraud Statistics")
        st.caption("Source: RBI Annual Report on Trends & Progress of Banking in India")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(rbi, x='fiscal_year', y='total_fraud_cases',
                        title='Total Digital Payment Fraud Cases (India)',
                        color='total_fraud_cases', color_continuous_scale='Oranges')
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(rbi, x='fiscal_year', y='total_fraud_amount_crores',
                        title='Total Fraud Amount (Rs Crore)',
                        color='total_fraud_amount_crores', color_continuous_scale='Reds')
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("How Our Simulation Compares to Real Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Our Simulated Fraud Rate", f"{df['is_fraud'].mean()*100:.2f}%")
    col2.metric("Real Reported Fraud Rate", "~0.002-0.005%",
                help="Official reported numbers are lower due to underreporting")
    col3.metric("Fraud Pattern Types Simulated", "6 types")
    st.info(
        "**Note on fraud rate:** Real NPCI reported fraud rates appear very low because many victims "
        "don't file complaints. Our simulation uses a deliberate 2.83% rate to create enough labelled "
        "fraud samples for meaningful statistical analysis and ML model training — a standard practice "
        "for building fraud research datasets when real labelled data is unavailable."
    )


def render_benfords(df):
    st.markdown('<p class="main-header">Benford\'s Law Analysis</p>', unsafe_allow_html=True)
    st.info("**Benford's Law:** In natural datasets, digit '1' appears as the first digit ~30% of the time, "
            "while '9' appears only ~4.6%. Fraudsters creating fake amounts unknowingly violate this pattern.")

    if 'first_digit' not in df.columns:
        df['first_digit'] = df['amount'].astype(str).str[0].astype(int)

    observed = df['first_digit'].value_counts(normalize=True).sort_index()
    expected = pd.Series({d: np.log10(1 + 1/d) for d in range(1, 10)})

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, 10)),
                         y=[observed.get(d, 0) for d in range(1, 10)],
                         name='Observed (All Txns)', marker_color='#667eea'))
    fig.add_trace(go.Scatter(x=list(range(1, 10)), y=expected.values,
                             name="Benford's Law (Expected)",
                             mode='lines+markers',
                             line=dict(color='#f093fb', width=3, dash='dot')))
    fig.update_layout(title="First Digit Distribution vs Benford's Law",
                     xaxis_title='First Digit', yaxis_title='Frequency Ratio',
                     height=450, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fo = df[df['is_fraud']==1]['first_digit'].value_counts(normalize=True).sort_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,10)), y=[fo.get(d,0) for d in range(1,10)],
                             name='Fraud Transactions', marker_color='#ef4444'))
        fig.add_trace(go.Scatter(x=list(range(1,10)), y=expected.values, name="Expected",
                                 mode='lines+markers', line=dict(color='#94a3b8', width=2, dash='dot')))
        fig.update_layout(title='Fraudulent Transactions vs Benford',
                          height=350, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        no = df[df['is_fraud']==0]['first_digit'].value_counts(normalize=True).sort_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,10)), y=[no.get(d,0) for d in range(1,10)],
                             name='Normal Transactions', marker_color='#22c55e'))
        fig.add_trace(go.Scatter(x=list(range(1,10)), y=expected.values, name="Expected",
                                 mode='lines+markers', line=dict(color='#94a3b8', width=2, dash='dot')))
        fig.update_layout(title='Normal Transactions vs Benford',
                          height=350, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    from scipy.stats import chisquare
    obs_counts = df['first_digit'].value_counts().sort_index()
    exp_counts = expected * obs_counts.sum()
    stat, p = chisquare([obs_counts.get(d, 0) for d in range(1, 10)], exp_counts.values)
    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-squared Statistic", f"{stat:.2f}")
    c2.metric("P-value", f"{p:.6f}")
    c3.metric("Verdict", "Conforms" if p > 0.05 else "Deviates from Benford")


def render_deep_dive(df):
    st.markdown('<p class="main-header">Deep Dive</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        search = st.text_input("Search UPI ID")
    with c2:
        dates = st.date_input("Date Range",
                              value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
                              min_value=df['timestamp'].min().date(),
                              max_value=df['timestamp'].max().date())
    with c3:
        fraud_filter = st.selectbox("Show", ['All', 'Fraud Only', 'Normal Only'])

    filtered = df.copy()
    if search:
        filtered = filtered[filtered['sender_upi_id'].str.contains(search, case=False, na=False)]
    if len(dates) == 2:
        filtered = filtered[(filtered['timestamp'].dt.date >= dates[0]) &
                           (filtered['timestamp'].dt.date <= dates[1])]
    if fraud_filter == 'Fraud Only':
        filtered = filtered[filtered['is_fraud'] == 1]
    elif fraud_filter == 'Normal Only':
        filtered = filtered[filtered['is_fraud'] == 0]

    st.write(f"**{len(filtered):,}** transactions matched")
    cols = [c for c in ['transaction_id', 'timestamp', 'sender_upi_id', 'receiver_upi_id',
                        'amount', 'transaction_type', 'city', 'status', 'is_fraud',
                        'fraud_type', 'ensemble_flag'] if c in filtered.columns]
    st.dataframe(filtered[cols].head(500), use_container_width=True)
    st.download_button("Download Filtered CSV", filtered.to_csv(index=False),
                      "filtered_transactions.csv", "text/csv")


def render_fraud_checker(df):
    """interactive page where anyone can enter a transaction and get a fraud verdict"""
    st.markdown('<p class="main-header">🔍 Live Fraud Checker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter transaction details below to check if it looks fraudulent</p>',
                unsafe_allow_html=True)

    st.divider()

    # --- input form ---
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("💰 Transaction Amount (₹)", min_value=1.0, max_value=500000.0,
                                  value=2500.0, step=100.0)
        city = st.selectbox("📍 City", ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
                                         'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
                                         'Chandigarh', 'Indore', 'Bhopal', 'Patna', 'Kochi'])
        sender_bank = st.selectbox("🏦 Sender Bank", ['PhonePe', 'Google Pay', 'Paytm',
                                                        'SBI', 'HDFC', 'ICICI', 'Axis',
                                                        'BOB', 'PNB', 'Kotak'])

    with col2:
        hour = st.slider("🕐 Hour of Transaction", 0, 23, 14)
        txn_type = st.selectbox("📋 Transaction Type", ['P2P', 'P2M', 'Bill Payment',
                                                          'Recharge', 'Investment'])
        is_weekend = st.selectbox("📅 Is Weekend?", ['No', 'Yes'])

    with col3:
        txn_velocity = st.number_input("⚡ Transactions in Last Hour", min_value=0, max_value=50,
                                        value=2, step=1)
        device_os = st.selectbox("📱 Device", ['Android', 'iOS'])
        st.write("")  # spacer
        check_btn = st.button("🔎 Check for Fraud", type="primary", use_container_width=True)

    if check_btn:
        st.divider()

        # --- run detection logic ---
        is_night = 1 if 1 <= hour <= 5 else 0
        is_wknd = 1 if is_weekend == 'Yes' else 0

        # compute z-score against the loaded dataset
        user_avg = df['amount'].mean()
        user_std = df['amount'].std()
        zscore = (amount - user_avg) / user_std if user_std > 0 else 0

        # rule-based checks
        rules_triggered = []
        risk_score = 0

        # rule 1: unusual hour (1-5 AM)
        if is_night:
            rules_triggered.append("🌙 Late Night Transaction (1-5 AM) — fraud rate 3x higher at night")
            risk_score += 25

        # rule 2: round number structuring (amounts like 9900, 9950, 9999)
        if 9000 <= amount <= 10000:
            rules_triggered.append("💵 Near ₹10K Threshold — possible structuring to avoid reporting limits")
            risk_score += 20

        # rule 3: high velocity
        if txn_velocity >= 5:
            rules_triggered.append(f"⚡ High Velocity ({txn_velocity} txns/hr) — possible account takeover")
            risk_score += 20
        if txn_velocity >= 10:
            rules_triggered.append("🚨 Extreme Velocity (10+ txns/hr) — rapid-fire fraud pattern")
            risk_score += 15

        # rule 4: high amount zscore
        if abs(zscore) > 3:
            rules_triggered.append(f"📊 Amount Anomaly (Z-Score: {zscore:.2f}) — far from average ₹{user_avg:,.0f}")
            risk_score += 20

        # rule 5: very high amount
        if amount > 50000:
            rules_triggered.append("💰 High Value Transaction (>₹50K) — higher scrutiny required")
            risk_score += 15

        # rule 6: late night + high amount
        if is_night and amount > 10000:
            rules_triggered.append("🔴 Late Night + High Amount — classic fraud combination")
            risk_score += 15

        # rule 7: high velocity + unusual hour
        if txn_velocity >= 5 and is_night:
            rules_triggered.append("🔴 Rapid Transactions at Night — strong fraud indicator")
            risk_score += 15

        risk_score = min(risk_score, 100)

        # --- display results ---
        if risk_score >= 60:
            verdict_color = "🔴"
            verdict = "HIGH RISK — Likely Fraudulent"
            st.error(f"## {verdict_color} {verdict}")
        elif risk_score >= 30:
            verdict_color = "🟡"
            verdict = "MEDIUM RISK — Suspicious"
            st.warning(f"## {verdict_color} {verdict}")
        else:
            verdict_color = "🟢"
            verdict = "LOW RISK — Appears Normal"
            st.success(f"## {verdict_color} {verdict}")

        # score cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{risk_score}/100")
        c2.metric("Amount Z-Score", f"{zscore:.2f}")
        c3.metric("Rules Triggered", f"{len(rules_triggered)}")
        c4.metric("Night Transaction", "Yes ⚠️" if is_night else "No ✅")

        # rules explanation
        if rules_triggered:
            st.divider()
            st.subheader("🚩 Rules Triggered")
            for rule in rules_triggered:
                st.markdown(f"- {rule}")
        else:
            st.divider()
            st.success("✅ No fraud rules triggered — transaction appears normal")

        # comparison with dataset
        st.divider()
        st.subheader("📊 How This Compares to Our Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Your Amount", f"₹{amount:,.0f}")
        c2.metric("Dataset Average", f"₹{df['amount'].mean():,.0f}")
        c3.metric("Dataset Fraud Avg", f"₹{df[df['is_fraud']==1]['amount'].mean():,.0f}")
        c4.metric("Your Percentile", f"{(df['amount'] < amount).mean()*100:.1f}%")

        # amount distribution chart
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df['is_fraud']==0]['amount'].clip(0, 50000),
                                    name='Normal Txns', marker_color='#22c55e', opacity=0.6, nbinsx=50))
        fig.add_trace(go.Histogram(x=df[df['is_fraud']==1]['amount'].clip(0, 50000),
                                    name='Fraud Txns', marker_color='#ef4444', opacity=0.6, nbinsx=50))
        fig.add_vline(x=amount, line_dash="dash", line_color="#3b82f6", line_width=3,
                      annotation_text=f"Your: ₹{amount:,.0f}", annotation_position="top")
        fig.update_layout(title="Where Your Transaction Falls in the Distribution",
                          xaxis_title="Amount (₹)", yaxis_title="Count",
                          barmode='overlay', height=350, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

        # causal insight
        st.divider()
        st.subheader("📈 Causal Insight: Effect of 2FA")
        causal_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'data', 'sample', 'causal_results.json')
        if not os.path.exists(causal_path):
            causal_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                        'data', 'processed', 'causal_results.json')

        if os.path.exists(causal_path):
            import json
            with open(causal_path) as f:
                causal = json.load(f)
            c1, c2, c3 = st.columns(3)
            c1.metric("2FA Effect on Fraud",
                      f"{causal['ate_pct_points']}",
                      delta=f"p = {causal['p_value']:.4f}")
            c2.metric("Parallel Trends", "Valid ✅" if causal['parallel_trends_valid'] else "Invalid ❌")
            c3.metric("Placebo Test", "Passed ✅" if causal['placebo_test_passed'] else "Failed ❌")
            st.info("💡 Our Difference-in-Differences analysis shows that 2FA significantly reduces "
                    "fraud probability. If this transaction had 2FA enabled, the fraud risk would be lower.")
        else:
            st.info("Run the full pipeline to see causal inference results here.")


def main():
    st.sidebar.title("UPI Fraud Detection")
    st.sidebar.caption("India | 2024 | 1M Transaction Simulation")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigate", [
        '🔍 Live Fraud Checker',
        'Overview',
        'Fraud Intelligence',
        'Bank & City Analytics',
        'Real Market Context',
        "Benford's Law",
        'Deep Dive'
    ])

    df = load_data()

    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    cities = ['All'] + sorted(df['city'].dropna().unique().tolist())
    sel_city = st.sidebar.selectbox("City", cities)
    if sel_city != 'All':
        df = df[df['city'] == sel_city]

    banks = ['All'] + sorted(df['sender_bank'].dropna().unique().tolist())
    sel_bank = st.sidebar.selectbox("Bank", banks)
    if sel_bank != 'All':
        df = df[df['sender_bank'] == sel_bank]

    st.sidebar.divider()
    st.sidebar.caption(f"**{len(df):,}** transactions loaded")
    st.sidebar.caption(f"Fraud: **{df['is_fraud'].sum():,}** ({df['is_fraud'].mean()*100:.2f}%)")

    if page == '🔍 Live Fraud Checker':   render_fraud_checker(df)
    elif page == 'Overview':              render_overview(df)
    elif page == 'Fraud Intelligence':    render_fraud_intelligence(df)
    elif page == 'Bank & City Analytics': render_bank_analytics(df)
    elif page == 'Real Market Context':   render_real_market(df)
    elif page == "Benford's Law":         render_benfords(df)
    elif page == 'Deep Dive':            render_deep_dive(df)


if __name__ == '__main__':
    main()

