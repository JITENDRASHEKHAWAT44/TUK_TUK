"""
TenantEase AI — Intelligent Property Management Console
Cynbit Technologies &bull; Assessment Deliverable

Design Standard:
  - Clean Pure White Background (#ffffff)
  - Harmonious Slate & Deep Navy Palette (Zero Clashing Colors)
  - Integrated Top Navigation Bar
  - High-Efficiency Operative Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# ─── 1. Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="TenantEase AI • Command Center",
    page_icon="https://cynbit.in/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 2. Pure White Harmonious Design System (CSS) ───────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Pure White Reset */
    html, body, [class*="st-"], .stApp {
        background-color: #ffffff !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #0f172a !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Remove default top margins & sidebar gap */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        max-width: 1360px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Browser Surfaces */
    ::selection {
        background: #0f172a !important;
        color: #ffffff !important;
    }
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #ffffff;
    }
    ::-webkit-scrollbar-thumb {
        background: #e2e8f0;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #cbd5e1;
    }

    /* Top Brand & Status Strip */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.85rem;
        margin-bottom: 0.85rem;
        border-bottom: 1px solid #f1f5f9;
        background: #ffffff;
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .brand-logo {
        width: 28px;
        height: 28px;
        background: #0f172a;
        color: #ffffff;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .brand-title-wrap h1 {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin: 0 !important;
        color: #0f172a !important;
        line-height: 1.2 !important;
    }
    .brand-title-wrap p {
        font-size: 0.75rem !important;
        color: #64748b !important;
        margin: 0 !important;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        color: #475569;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #059669;
    }

    /* Custom Nav Bar Styling via st.radio */
    div[data-testid="stRadio"] {
        margin-bottom: 1.25rem !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.35rem !important;
        background: #ffffff !important;
        padding: 0.35rem !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        width: 100% !important;
    }
    div[data-testid="stRadio"] label {
        flex: 1 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0.45rem 0.85rem !important;
        border-radius: 6px !important;
        font-size: 0.825rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
        cursor: pointer !important;
        margin: 0 !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease !important;
        background: transparent !important;
    }
    div[data-testid="stRadio"] label:hover {
        color: #0f172a !important;
        background: #f8fafc !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #0f172a !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }

    /* Clean KPI Cards on Pure White */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        position: relative;
    }
    .kpi-tag {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #0f172a;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .text-crimson { color: #b91c1c; font-weight: 600; }
    .text-emerald { color: #047857; font-weight: 600; }

    /* Clean Section Containers */
    .clean-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .clean-box-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .clean-heading {
        font-size: 0.925rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .clean-subheading {
        font-size: 0.75rem;
        color: #64748b;
        margin: 0;
    }

    /* Badges (Subtle, Non-conflicting) */
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .pill-critical { background: #fef2f2; color: #b91c1c; border: 1px solid #fee2e2; }
    .pill-warning  { background: #fffbeb; color: #b45309; border: 1px solid #fef3c7; }
    .pill-stable   { background: #f0fdf4; color: #047857; border: 1px solid #dcfce7; }
    .pill-neutral  { background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; }

    /* Action Callout */
    .action-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-top: 0.75rem;
    }
    .action-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    .action-text {
        font-size: 0.825rem;
        color: #0f172a;
        line-height: 1.4;
    }

    /* SLA Banner */
    .sla-alert {
        background: #fef2f2;
        border: 1px solid #fee2e2;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        color: #991b1b;
        font-size: 0.825rem;
        line-height: 1.4;
        margin-top: 0.75rem;
    }
    .sla-standard {
        background: #f0fdf4;
        border: 1px solid #dcfce7;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        color: #166534;
        font-size: 0.825rem;
        line-height: 1.4;
        margin-top: 0.75rem;
    }

    /* Buttons */
    div.stButton > button {
        background: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.825rem !important;
        padding: 0.45rem 1rem !important;
        transition: opacity 0.15s ease !important;
    }
    div.stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Inputs, Selectboxes, Textareas on White */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover,
    textarea:hover {
        border-color: #94a3b8 !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    textarea:focus {
        border-color: #0f172a !important;
        box-shadow: 0 0 0 1px #0f172a !important;
    }

    /* Code blocks */
    code, pre {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] > div {
        color: #0f172a !important;
    }

    /* Hide Streamlit Default UI elements */
    #MainMenu, footer, header {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── 3. Data & Model Ingestion ───────────────────────────────────
@st.cache_data(show_spinner=False)
def get_all_ecosystem_data():
    from data.generate_data import generate_all_data
    from models.payment_predictor import train_payment_model, predict_late_payment
    from models.churn_predictor import train_churn_model, predict_churn
    from models.maintenance_classifier import train_ml_classifier

    properties, tenants, payments, maintenance, features = generate_all_data()

    payment_model, pay_results = train_payment_model(features)
    churn_model, churn_results = train_churn_model(features)
    maint_model, maint_results = train_ml_classifier(maintenance)

    payment_preds = predict_late_payment(payment_model, features)
    churn_preds = predict_churn(churn_model, features)

    unified_df = features.copy()
    unified_df["late_probability"] = payment_preds["late_probability"]
    unified_df["late_risk_level"] = payment_preds["risk_level"]
    unified_df["churn_probability"] = churn_preds["churn_probability"]
    unified_df["churn_risk_level"] = churn_preds["risk_level"]
    unified_df["recommended_actions"] = churn_preds["recommended_actions"]

    prop_meta = properties.set_index("property_id")
    unified_df["property_name"] = unified_df["property_id"].map(prop_meta["property_name"])
    unified_df["area"] = unified_df["property_id"].map(prop_meta["area"])
    unified_df["property_type"] = unified_df["property_id"].map(prop_meta["property_type"])
    unified_df["deposit"] = unified_df["property_id"].map(prop_meta["deposit"])

    return {
        "properties": properties,
        "tenants": tenants,
        "payments": payments,
        "maintenance": maintenance,
        "features": unified_df,
        "payment_model": payment_model,
        "pay_results": pay_results,
        "churn_model": churn_model,
        "churn_results": churn_results,
        "maint_model": maint_model,
        "maint_results": maint_results,
    }


eco = get_all_ecosystem_data()
df = eco["features"]
properties = eco["properties"]
payments = eco["payments"]
maintenance = eco["maintenance"]
maint_model = eco["maint_model"]


# ─── 4. Top Header & Brand Bar ───────────────────────────────────
st.markdown("""
<div class="brand-bar">
    <div class="brand-left">
        <div class="brand-logo">TE</div>
        <div class="brand-title-wrap">
            <h1>TenantEase AI</h1>
            <p>Cynbit Technologies &bull; Property Intelligence Console</p>
        </div>
    </div>
    <div class="status-indicator">
        <span class="status-dot"></span>
        <span>Operational &bull; 25 Jaipur Properties Active</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── 5. Top Navigation Bar (Integrated Horizontal Segmented Control) ───
nav_tabs = [
    "Portfolio Overview",
    "Tenant Workbench",
    "Maintenance Triage Desk",
    "Financial Simulator",
    "Model Governance",
]

selected_view = st.radio(
    "Navigation",
    nav_tabs,
    horizontal=True,
    label_visibility="collapsed",
)


# ─── 6. View Routing ─────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════
# VIEW 1: Portfolio Overview
# ══════════════════════════════════════════════════════════════════
if selected_view == "Portfolio Overview":

    # KPI Strip
    total_units = len(df)
    total_revenue = df["monthly_rent"].sum()
    high_late = df[df["late_risk_level"] == "High Risk"]
    high_churn = df[df["churn_risk_level"] == "High Risk"]
    rent_at_risk = high_late["monthly_rent"].sum()
    churn_loss = high_churn["monthly_rent"].sum()

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-tag">Managed Units</div>
            <div class="kpi-value">{total_units}</div>
            <div class="kpi-sub">Across 15 Jaipur Localities</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-tag">Monthly Rent Roll</div>
            <div class="kpi-value">&#8377;{total_revenue:,.0f}</div>
            <div class="kpi-sub text-emerald">&#8377;{total_revenue/total_units:,.0f} average rent/unit</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-tag">Rent at Imminent Risk</div>
            <div class="kpi-value">&#8377;{rent_at_risk:,.0f}</div>
            <div class="kpi-sub text-crimson">{len(high_late)} tenants predicted late (>3d)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-tag">60-Day Churn Exposure</div>
            <div class="kpi-value">&#8377;{churn_loss:,.0f}</div>
            <div class="kpi-sub text-crimson">{len(high_churn)} units flagged for non-renewal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual Insights (Charts on White)
    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        st.markdown("""
        <div class="clean-box-header">
            <div>
                <div class="clean-heading">Bivariate Risk Quadrant (Late Payment vs. Churn)</div>
                <div class="clean-subheading">Bubble diameter indicates rent size. Top-right accounts exhibit compound risk.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_quad = px.scatter(
            df,
            x="late_probability",
            y="churn_probability",
            size="monthly_rent",
            color="churn_risk_level",
            color_discrete_map={
                "High Risk": "#b91c1c",
                "Medium Risk": "#b45309",
                "Low Risk": "#047857",
            },
            hover_name="tenant_name",
            hover_data={
                "area": True,
                "monthly_rent": ":,d",
                "late_probability": ":.1%",
                "churn_probability": ":.1%",
                "churn_risk_level": False,
            },
            labels={
                "late_probability": "Predicted Late Payment Probability",
                "churn_probability": "Predicted 60-Day Churn Risk",
            },
        )
        fig_quad.add_hline(y=0.60, line_dash="dash", line_color="#cbd5e1", opacity=0.8)
        fig_quad.add_vline(x=0.50, line_dash="dash", line_color="#cbd5e1", opacity=0.8)
        fig_quad.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickformat=".0%"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
            font=dict(family="Inter", size=11, color="#0f172a"),
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    with col_right:
        st.markdown("""
        <div class="clean-box-header">
            <div>
                <div class="clean-heading">12-Month Rent Collection Trajectory</div>
                <div class="clean-subheading">Total realized rent collected vs. percentage of delayed payments.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        monthly = payments.groupby("month").agg(
            total_due=("amount_due", "sum"),
            total_paid=("amount_paid", "sum"),
            late_count=("is_late", "sum"),
            total_count=("is_paid", "count"),
        ).reset_index()
        monthly["late_rate"] = (monthly["late_count"] / monthly["total_count"] * 100).round(1)

        fig_cf = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cf.add_trace(
            go.Bar(
                x=monthly["month"],
                y=monthly["total_paid"],
                name="Collected (INR)",
                marker_color="#0f172a",
                opacity=0.9,
            ),
            secondary_y=False,
        )
        fig_cf.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["late_rate"],
                name="Late Settle %",
                line=dict(color="#b91c1c", width=2),
                mode="lines+markers",
            ),
            secondary_y=True,
        )
        fig_cf.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter", size=11, color="#0f172a"),
            xaxis=dict(showgrid=False),
        )
        fig_cf.update_yaxes(title_text="Amount (INR)", secondary_y=False, gridcolor="#f1f5f9")
        fig_cf.update_yaxes(title_text="Late %", secondary_y=True, gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cf, use_container_width=True)

    # Priority Action Queue
    st.markdown("""
    <div class="clean-box-header" style="margin-top: 1.25rem;">
        <div>
            <div class="clean-heading">Priority Escalation Queue (Immediate Interventions)</div>
            <div class="clean-subheading">Accounts exhibiting simultaneous churn signals and payment instability.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    dual_risk = df.sort_values(by=["churn_probability", "late_probability"], ascending=[False, False]).head(5)

    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns([2.0, 1.8, 1.2, 1.2, 3.8])
    q_col1.markdown("<span style='font-size: 0.72rem; font-weight: 600; color: #64748b;'>TENANT / UNIT</span>", unsafe_allow_html=True)
    q_col2.markdown("<span style='font-size: 0.72rem; font-weight: 600; color: #64748b;'>LOCALITY / RENT</span>", unsafe_allow_html=True)
    q_col3.markdown("<span style='font-size: 0.72rem; font-weight: 600; color: #64748b;'>PAYMENT RISK</span>", unsafe_allow_html=True)
    q_col4.markdown("<span style='font-size: 0.72rem; font-weight: 600; color: #64748b;'>CHURN RISK</span>", unsafe_allow_html=True)
    q_col5.markdown("<span style='font-size: 0.72rem; font-weight: 600; color: #64748b;'>AI RECOMMENDED MITIGATION</span>", unsafe_allow_html=True)

    for _, r in dual_risk.iterrows():
        c1, c2, c3, c4, c5 = st.columns([2.0, 1.8, 1.2, 1.2, 3.8])
        c1.markdown(f"**{r['tenant_name']}**<br><span style='font-size: 0.75rem; color: #64748b;'>{r['property_name']}</span>", unsafe_allow_html=True)
        c2.markdown(f"{r['area']}<br><span style='font-size: 0.75rem; font-weight: 600; color: #0f172a;'>&#8377;{r['monthly_rent']:,}/mo</span>", unsafe_allow_html=True)

        late_pill = "pill-critical" if r['late_risk_level'] == "High Risk" else ("pill-warning" if r['late_risk_level'] == "Medium Risk" else "pill-stable")
        c3.markdown(f"<span class='pill {late_pill}'>{r['late_probability']:.1%}</span>", unsafe_allow_html=True)

        churn_pill = "pill-critical" if r['churn_risk_level'] == "High Risk" else ("pill-warning" if r['churn_risk_level'] == "Medium Risk" else "pill-stable")
        c4.markdown(f"<span class='pill {churn_pill}'>{r['churn_probability']:.1%}</span>", unsafe_allow_html=True)

        c5.markdown(f"<span style='font-size: 0.8rem; color: #334155;'>{r['recommended_actions']}</span>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.4rem 0; border: none; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# VIEW 2: Tenant Workbench (Master-Detail)
# ══════════════════════════════════════════════════════════════════
elif selected_view == "Tenant Workbench":

    st.markdown("""
    <div class="clean-box-header">
        <div>
            <div class="clean-heading">Tenant Account Explorer & Intervention Studio</div>
            <div class="clean-subheading">Inspect individual behavioral parameters, payment histories, and execute mitigation workflows.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    filter_c1, filter_c2, filter_c3 = st.columns([1.5, 1.5, 2.0])
    with filter_c1:
        selected_area = st.selectbox("Locality Filter", ["All Areas"] + sorted(list(df["area"].unique())))
    with filter_c2:
        selected_risk = st.selectbox("Risk Filter", ["All Risk Tiers", "High Risk", "Medium Risk", "Low Risk"])
    with filter_c3:
        search_query = st.text_input("Search Account", placeholder="Search tenant by name...")

    filtered_df = df.copy()
    if selected_area != "All Areas":
        filtered_df = filtered_df[filtered_df["area"] == selected_area]
    if selected_risk != "All Risk Tiers":
        filtered_df = filtered_df[filtered_df["churn_risk_level"] == selected_risk]
    if search_query.strip():
        filtered_df = filtered_df[filtered_df["tenant_name"].str.contains(search_query.strip(), case=False, na=False)]

    if len(filtered_df) == 0:
        st.info("No accounts match the current filters.")
    else:
        col_list, col_profile = st.columns([1.3, 2.7])

        with col_list:
            tenant_selection = st.radio(
                "Tenant Selection",
                options=filtered_df["tenant_id"].tolist(),
                format_func=lambda tid: f"{filtered_df[filtered_df['tenant_id']==tid].iloc[0]['tenant_name']} (₹{filtered_df[filtered_df['tenant_id']==tid].iloc[0]['monthly_rent']:,})",
                label_visibility="collapsed",
            )

        with col_profile:
            t = df[df["tenant_id"] == tenant_selection].iloc[0]

            # Account Header on Pure White
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">{t['tenant_name']}</div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.15rem;">
                            ID: <code>{t['tenant_id']}</code> &bull; {t['occupation']} ({t['age']} yrs) &bull; {t['property_name']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">&#8377;{t['monthly_rent']:,}<span style="font-size: 0.8rem; font-weight: 400; color: #64748b;"> /mo</span></div>
                        <div style="font-size: 0.75rem; color: #64748b;">Deposit: &#8377;{t['deposit']:,}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metric Strip
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Pay Delay", f"{t['avg_delay_days']} days", delta=f"{t['delay_trend']:+.1f} trend", delta_color="inverse")
            m2.metric("Payment Rate", f"{t['payment_rate']*100:.0f}%", delta=f"{t['late_payment_rate']*100:.0f}% late", delta_color="inverse")
            m3.metric("Open Tickets", f"{int(t['unresolved_complaints'])}", delta=f"{t['total_complaints']} total", delta_color="inverse")
            m4.metric("Lease Remaining", f"{t['days_until_lease_end']} days", delta=f"{t['tenure_months']} mos tenure", delta_color="off")

            # Risk Factor Breakdown
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            rf1, rf2 = st.columns(2)
            with rf1:
                st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748b;'>Late Payment Risk Index</span>", unsafe_allow_html=True)
                st.progress(float(t['late_probability']))
                st.caption(f"Score: {t['late_probability']*100:.1f}% &bull; Category: {t['late_risk_level']}")
            with rf2:
                st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748b;'>60-Day Churn Risk Index</span>", unsafe_allow_html=True)
                st.progress(float(t['churn_probability']))
                st.caption(f"Score: {t['churn_probability']*100:.1f}% &bull; Category: {t['churn_risk_level']}")

            # Payment Timeline
            t_payments = payments[payments["tenant_id"] == tenant_selection].sort_values(by="month", ascending=True)
            if len(t_payments) > 0:
                fig_t_pay = px.bar(
                    t_payments,
                    x="month",
                    y="delay_days",
                    color="is_late",
                    color_discrete_map={True: "#b91c1c", False: "#0f172a"},
                    labels={"delay_days": "Delay (Days)", "month": "Period", "is_late": "Delayed"},
                    title="12-Month Payment Delay History",
                )
                fig_t_pay.update_layout(
                    height=190,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(family="Inter", size=10, color="#0f172a"),
                    yaxis=dict(gridcolor="#f1f5f9"),
                    xaxis=dict(showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig_t_pay, use_container_width=True)

            # Action Directives Box
            st.markdown(f"""
            <div class="action-card">
                <div class="action-title">Prescriptive Management Protocol</div>
                <div class="action-text">{t['recommended_actions']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Fast Actions
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("Generate WhatsApp Nudge", key=f"btn_wa_{t['tenant_id']}", use_container_width=True):
                    st.code(
                        f"Namaste {t['tenant_name'].split()[0]} ji, this is regarding your unit at {t['property_name']}. "
                        f"We noted {int(t['unresolved_complaints'])} open maintenance ticket(s). "
                        f"Our technician is scheduled to inspect tomorrow. Please reply if convenient. — TenantEase",
                        language="text"
                    )
            with ac2:
                if st.button("Draft Early Renewal Offer", key=f"btn_ren_{t['tenant_id']}", use_container_width=True):
                    st.code(
                        f"Dear {t['tenant_name']},\n"
                        f"Your lease at {t['property_name']} concludes in {t['days_until_lease_end']} days. "
                        f"We are pleased to extend a 12-month renewal at ₹{t['monthly_rent']:,} with priority maintenance SLA. "
                        f"Confirm your renewal via TenantEase portal.",
                        language="text"
                    )
            with ac3:
                if st.button("Dispatch Property Audit", key=f"btn_aud_{t['tenant_id']}", use_container_width=True):
                    st.success(f"Audit work-order issued for {t['property_name']}.")


# ══════════════════════════════════════════════════════════════════
# VIEW 3: Maintenance Triage Desk (Bilingual NLP)
# ══════════════════════════════════════════════════════════════════
elif selected_view == "Maintenance Triage Desk":
    from models.maintenance_classifier import classify_complaint

    st.markdown("""
    <div class="clean-box-header">
        <div>
            <div class="clean-heading">Autonomous Natural Language Complaint Triage</div>
            <div class="clean-subheading">Evaluates unstructured tenant requests in Indian regional vernacular (Hinglish/English) and routes SLAs.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t_col1, t_col2 = st.columns([1.1, 1.0])

    with t_col1:
        st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #0f172a;'>Test Ticket Intake</span>", unsafe_allow_html=True)

        presets = [
            "Gas leak ho rahi hai kitchen me bahut smell aa rahi hai please urgent help!",
            "Switch board me spark aa rahi hai very dangerous situation",
            "AC not cooling at all room is very hot and water is dripping",
            "Bathroom me paani nahi aa raha hai geyser is not filling",
            "Need rent receipt for last 3 months for IT declaration",
            "Cockroach and pest infestation in kitchen cupboard",
            "Ceiling se plaster gir raha hai moisture dampness hai",
            "Custom text input...",
        ]

        preset_choice = st.selectbox("Select Test Scenario or Custom", presets, label_visibility="collapsed")

        if preset_choice == "Custom text input...":
            input_text = st.text_area("Inbound Message Text", placeholder="Type complaint in English or Hindi-English mix...", height=90)
        else:
            input_text = preset_choice
            st.text_area("Inbound Message Text", input_text, height=90, disabled=True)

        eval_btn = st.button("Classify & Triage Ticket", type="primary", use_container_width=True)

    with t_col2:
        st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #0f172a;'>Triage Outcome & Protocol</span>", unsafe_allow_html=True)

        if eval_btn or preset_choice != "Custom text input...":
            res = classify_complaint(input_text, ml_model=maint_model)

            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("Trade Category", res["category"])
            res_c2.metric("Confidence", f"{res['confidence']*100:.1f}%")
            res_c3.metric("Urgency", f"{res['urgency_score']}/10", delta=res["urgency_label"], delta_color="off")

            st.progress(res["urgency_score"] / 10.0)

            if res["is_emergency"]:
                st.markdown(f"""
                <div class="sla-alert">
                    <strong>EMERGENCY HAZARD PROTOCOL ACTIVATED</strong><br>
                    Life-safety hazard identified. Field technician dispatched within 30-minute SLA window.
                    Immediate safety containment directives issued to occupant.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="sla-standard">
                    <strong>Standard Operational Work Order Logged</strong><br>
                    Ticket routed to {res['category']} contractor desk. Target completion: 24 to 48 hours.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
            st.markdown("<span style='font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase;'>Automated Tenant Response</span>", unsafe_allow_html=True)
            st.info(res["auto_response"])

    # Historical Trade Volume
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 0.825rem; font-weight: 600; color: #0f172a;'>Portfolio Historical Ticket Distribution by Trade</span>", unsafe_allow_html=True)

    cat_counts = maintenance["category"].value_counts().reset_index()
    cat_counts.columns = ["Trade", "Volume"]

    fig_cat_bar = px.bar(
        cat_counts,
        x="Trade",
        y="Volume",
        color_discrete_sequence=["#0f172a"],
    )
    fig_cat_bar.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter", size=10, color="#0f172a"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f1f5f9"),
        showlegend=False,
    )
    st.plotly_chart(fig_cat_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# VIEW 4: Financial Simulator
# ══════════════════════════════════════════════════════════════════
elif selected_view == "Financial Simulator":

    st.markdown("""
    <div class="clean-box-header">
        <div>
            <div class="clean-heading">Executive Retention & Revenue What-If Simulator</div>
            <div class="clean-subheading">Model operational adjustments to forecast net annual rental capital recovery across Cynbit properties.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sim_c1, sim_c2 = st.columns([1.2, 1.8])

    with sim_c1:
        st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #0f172a;'>Operational Levers</span>", unsafe_allow_html=True)

        lever_complaints = st.slider(
            "Resolve Open Maintenance Tickets (<24h)",
            min_value=0,
            max_value=100,
            value=60,
            step=10,
            format="%d%%",
        )

        lever_discount = st.slider(
            "Targeted Early Lease Renewal Incentive",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            format="%.1f%%",
        )

        lever_nudges = st.slider(
            "Automated WhatsApp Payment Reminder Efficacy",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            format="%d%%",
        )

    with sim_c2:
        total_monthly_rent = df["monthly_rent"].sum()
        baseline_churn = len(df[df["churn_probability"] > 0.50])
        baseline_loss = baseline_churn * (df["monthly_rent"].mean() * 1.5 + 25000)

        churn_reduction_pct = (lever_complaints * 0.45 + lever_discount * 4.0) / 100.0
        recovered_tenants = int(round(baseline_churn * min(0.75, churn_reduction_pct)))
        net_saved_revenue = recovered_tenants * (df["monthly_rent"].mean() * 1.5 + 25000)

        st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #0f172a;'>Projected Value Realization</span>", unsafe_allow_html=True)

        kpi_s1, kpi_s2 = st.columns(2)
        kpi_s1.metric("Retained Leases", f"+{recovered_tenants} units", delta=f"{recovered_tenants}/{baseline_churn} saved")
        kpi_s2.metric("Turnover Capital Saved", f"₹{net_saved_revenue:,.0f}", delta="Annual recovery")

        fig_sim = go.Figure(go.Waterfall(
            name="Capital Retention",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Baseline Turnover Loss", "Ticket Speedup", "Renewal Discount", "Net Retained Portfolio Value"],
            y=[
                -baseline_loss,
                net_saved_revenue * 0.60,
                net_saved_revenue * 0.40,
                -baseline_loss + net_saved_revenue,
            ],
            connector={"line": {"color": "#cbd5e1"}},
            decreasing={"marker": {"color": "#b91c1c"}},
            increasing={"marker": {"color": "#047857"}},
            totals={"marker": {"color": "#0f172a"}},
        ))
        fig_sim.update_layout(
            title="Annual Capital Flow Breakdown (INR)",
            height=260,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter", size=10, color="#0f172a"),
            yaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# VIEW 5: Model Governance
# ══════════════════════════════════════════════════════════════════
elif selected_view == "Model Governance":

    st.markdown("""
    <div class="clean-box-header">
        <div>
            <div class="clean-heading">Technical Model Governance, Calibration & Validation</div>
            <div class="clean-subheading">Model metrics, confusion matrices, and feature coefficients.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    gov_tabs = st.tabs(["Late Payment Classifier", "Tenant Churn Ensemble", "Maintenance NLP"])

    with gov_tabs[0]:
        pr = eco["pay_results"]
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Accuracy", f"{pr['accuracy']:.1%}")
        g2.metric("Precision (Late)", f"{pr['precision']:.1%}")
        g3.metric("Recall (Late)", f"{pr['recall']:.1%}")
        g4.metric("ROC-AUC", f"{pr['auc_roc']:.3f}")

        c_cm1, c_imp1 = st.columns(2)
        with c_cm1:
            st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748b;'>Confusion Matrix</span>", unsafe_allow_html=True)
            cm1 = pr["confusion_matrix"]
            fig_cm1 = px.imshow(
                cm1,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["On-Time", "Delayed (>3d)"],
                y=["On-Time", "Delayed (>3d)"],
                color_continuous_scale=["#f8fafc", "#0f172a"],
                text_auto=True,
            )
            fig_cm1.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Inter", size=10))
            st.plotly_chart(fig_cm1, use_container_width=True)

        with c_imp1:
            st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748b;'>Feature Importance Weights</span>", unsafe_allow_html=True)
            imp_df1 = pd.DataFrame(list(pr["feature_importance"].items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_imp1 = px.bar(imp_df1, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#0f172a"])
            fig_imp1.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Inter", size=9), xaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_imp1, use_container_width=True)

    with gov_tabs[1]:
        cr = eco["churn_results"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{cr['accuracy']:.1%}")
        c2.metric("Precision (Churn)", f"{cr['precision']:.1%}")
        c3.metric("Recall (Churn)", f"{cr['recall']:.1%}")
        c4.metric("ROC-AUC", f"{cr['auc_roc']:.3f}")

        c_cm2, c_imp2 = st.columns(2)
        with c_cm2:
            st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748b;'>Confusion Matrix</span>", unsafe_allow_html=True)
            cm2 = cr["confusion_matrix"]
            fig_cm2 = px.imshow(
                cm2,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Renew", "Churn"],
                y=["Renew", "Churn"],
                color_continuous_scale=["#f8fafc", "#0f172a"],
                text_auto=True,
            )
            fig_cm2.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Inter", size=10))
            st.plotly_chart(fig_cm2, use_container_width=True)

        with c_imp2:
            st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748b;'>Feature Importance Weights</span>", unsafe_allow_html=True)
            imp_df2 = pd.DataFrame(list(cr["feature_importance"].items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_imp2 = px.bar(imp_df2, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#0f172a"])
            fig_imp2.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), font=dict(family="Inter", size=9), xaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig_imp2, use_container_width=True)

    with gov_tabs[2]:
        mr = eco["maint_results"]
        st.metric("NLP Pipeline Accuracy", f"{mr['accuracy']:.1%}")

        st.markdown("<span style='font-size: 0.75rem; font-weight: 600; color: #64748b;'>Trade Category Validation Breakdown</span>", unsafe_allow_html=True)
        audit_rows = []
        for cat in ["Plumbing", "Electrical", "HVAC", "Structural", "Gas", "Pest", "General"]:
            if cat in mr:
                audit_rows.append({
                    "Trade Category": cat,
                    "Precision": f"{mr[cat]['precision']:.1%}",
                    "Recall": f"{mr[cat]['recall']:.1%}",
                    "F1 Score": f"{mr[cat]['f1-score']:.1%}",
                    "Sample Support": int(mr[cat]["support"]),
                })
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
