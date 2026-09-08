import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Add src to python path to load helper scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
# pyrefly: ignore [missing-import]
from predict import (
    predict_single, 
    predict_batch, 
    calculate_customer_clv,
    explain_customer_prediction,
    get_global_feature_importance,
    simulate_retention_scenario,
    generate_sample_batch_csv,
    generate_executive_report_text
)

def apply_cotton_candy_theme(fig, title_text):
    """Applies clean pastel Cotton Candy theme to Plotly figures with Dark Slate Blue typography."""
    fig.update_layout(
        title={'text': title_text, 'font': {'color': '#2C3E50', 'family': 'Outfit', 'size': 18, 'weight': 800}},
        xaxis=dict(tickfont=dict(color="#2C3E50", family='Outfit'), title=dict(font=dict(color="#2C3E50", family='Outfit')), gridcolor="rgba(44, 62, 80, 0.08)"),
        yaxis=dict(tickfont=dict(color="#2C3E50", family='Outfit'), title=dict(font=dict(color="#2C3E50", family='Outfit')), gridcolor="rgba(44, 62, 80, 0.08)"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color="#2C3E50", family='Outfit')),
        font=dict(family="Outfit", color="#2C3E50")
    )
    return fig

# Page Configuration
st.set_page_config(
    page_title="Bank Customer Churn AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Cotton Candy Theme, 3D Animations & Mobile Responsiveness)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    /* 🎨 2. Cotton Candy (Soft & Pastel) Mesh Background */
    .stApp {
        background: 
            radial-gradient(circle at 10% 20%, rgba(255, 222, 233, 0.8) 0%, transparent 45%),
            radial-gradient(circle at 90% 15%, rgba(181, 255, 252, 0.8) 0%, transparent 45%),
            radial-gradient(circle at 50% 80%, rgba(255, 255, 255, 0.9) 0%, transparent 50%),
            linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%) fixed !important;
        color: #2C3E50 !important;
        font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
    }

    /* Streamlit Global Text Overrides */
    p, label, span, div, h1, h2, h3, h4, h5, h6 {
        color: #2C3E50 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Sidebar Clean Pastel Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        box-shadow: 5px 0 25px rgba(44, 62, 80, 0.05) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] label {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }
    
    /* 3D Floating Main App Header with Cotton Candy Gradient */
    .main-header {
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 900;
        background: linear-gradient(135deg, #2C3E50 0%, #7971ea 50%, #ff758c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
        text-align: center;
        letter-spacing: -0.03em;
        animation: float3D 6s ease-in-out infinite;
        transform-style: preserve-3d;
    }
    .sub-header {
        font-size: clamp(0.95rem, 1.5vw, 1.2rem);
        color: #4a5568 !important;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 500;
    }
    .sidebar-title {
        font-size: 1.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #2C3E50 0%, #7971ea 60%, #ff758c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* 3D Interactive Card Containers (Glassmorphism & Tilt on Hover) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.78) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        border-radius: 1.25rem !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 30px rgba(44, 62, 80, 0.07), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        transform: perspective(1000px) rotateX(0deg) rotateY(0deg);
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: perspective(1000px) translateY(-6px) rotateX(1deg) rotateY(-1deg) scale(1.008) !important;
        border-color: rgba(121, 113, 234, 0.4) !important;
        box-shadow: 0 20px 45px rgba(44, 62, 80, 0.12), inset 0 1px 2px rgba(255, 255, 255, 1) !important;
    }

    /* Card Section Titles with Gradient Typography */
    .card-title {
        font-weight: 800;
        font-size: 1.3rem;
        background: linear-gradient(135deg, #2C3E50 0%, #7971ea 60%, #ff758c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.1rem;
        border-bottom: 2px solid rgba(121, 113, 234, 0.15);
        padding-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }

    /* 3D KPI Metric Badges */
    .kpi-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 1);
        border-radius: 1rem;
        padding: 1.15rem;
        text-align: center;
        margin-bottom: 0.85rem;
        box-shadow: 0 8px 24px rgba(44, 62, 80, 0.06);
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform: perspective(600px) translateZ(0);
    }
    .kpi-card:hover {
        transform: perspective(600px) translateY(-5px) translateZ(12px);
        box-shadow: 0 15px 35px rgba(121, 113, 234, 0.18);
        border-color: rgba(121, 113, 234, 0.3);
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #64748b !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 900;
        background: linear-gradient(135deg, #7971ea 0%, #ff758c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.3rem;
    }

    /* 3D Risk Indicator Banners */
    .risk-banner {
        padding: 1rem;
        border-radius: 1rem;
        font-weight: 900;
        font-size: 1.55rem;
        text-align: center;
        margin-bottom: 1.1rem;
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 0.02em;
        transform: perspective(800px) translateZ(10px);
        transition: transform 0.3s ease;
    }
    .risk-low {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.35);
    }
    .risk-medium {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 50%, #fbbf24 100%);
        box-shadow: 0 10px 25px rgba(245, 158, 11, 0.35);
    }
    .risk-high {
        background: linear-gradient(135deg, #e11d48 0%, #f43f5e 50%, #ff758c 100%);
        box-shadow: 0 10px 30px rgba(244, 63, 94, 0.4);
        animation: pulse3D 2.5s infinite;
    }

    /* Soft Pastel Recommendation Box */
    .recommendation-box {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(121, 113, 234, 0.25);
        border-radius: 1rem;
        padding: 1.25rem;
        font-size: 1rem;
        color: #2C3E50 !important;
        margin-top: 0.95rem;
        border-left: 6px solid #7971ea;
        box-shadow: 0 6px 20px rgba(44, 62, 80, 0.05);
        line-height: 1.55;
    }

    /* Explainable AI Pills */
    .factor-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        margin-bottom: 0.65rem;
        border-radius: 0.75rem;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .factor-pill:hover {
        transform: translateX(4px);
    }
    .factor-positive {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.3);
        color: #9f1239 !important;
    }
    .factor-positive * {
        color: #9f1239 !important;
    }
    .factor-negative {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #065f46 !important;
    }
    .factor-negative * {
        color: #065f46 !important;
    }

    /* 3D Interactive Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #ff758c 0%, #ff7eb3 40%, #7971ea 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 0.8rem 1.6rem !important;
        box-shadow: 0 8px 25px rgba(255, 117, 140, 0.4) !important;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        transform: perspective(500px) translateZ(0);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #ff5778 0%, #ff68a5 40%, #685fe8 100%) !important;
        transform: perspective(500px) translateY(-4px) translateZ(15px) scale(1.02) !important;
        box-shadow: 0 14px 35px rgba(121, 113, 234, 0.45) !important;
    }
    div.stButton > button:active {
        transform: perspective(500px) translateY(2px) translateZ(-5px) scale(0.98) !important;
    }

    div.stDownloadButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #7971ea 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 0.8rem 1.6rem !important;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.35) !important;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    div.stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 14px 35px rgba(59, 130, 246, 0.45) !important;
    }

    /* Clean Pastel Form Inputs */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 0.75rem !important;
        border: 1.5px solid rgba(44, 62, 80, 0.15) !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }

    /* 3D Animation Keyframes */
    @keyframes float3D {
        0% { transform: translateY(0px) rotateX(0deg) rotateY(0deg); }
        50% { transform: translateY(-8px) rotateX(2deg) rotateY(-2deg); }
        100% { transform: translateY(0px) rotateX(0deg) rotateY(0deg); }
    }

    @keyframes pulse3D {
        0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4), 0 10px 30px rgba(244, 63, 94, 0.3); }
        70% { box-shadow: 0 0 0 18px rgba(244, 63, 94, 0), 0 14px 35px rgba(244, 63, 94, 0.4); }
        100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0), 0 10px 30px rgba(244, 63, 94, 0.3); }
    }

    /* 📱 Mobile & Tablet Responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
        }
        .sub-header {
            font-size: 0.95rem !important;
            margin-bottom: 1.25rem !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 1rem !important;
            border-radius: 1rem !important;
        }
        .kpi-value {
            font-size: 1.4rem !important;
        }
        .risk-banner {
            font-size: 1.25rem !important;
            padding: 0.75rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_comparison_metrics():
    path = "models/model_comparison.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# Navigation Sidebar
st.sidebar.markdown("<div class='sidebar-title'>🏦 Churn AI SaaS</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Single Churn Predictor", "📂 Batch Predictions & Portfolio Risk", "📊 Model Performance & Explainability"]
)

# Model Selection
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Configuration")
selected_model = st.sidebar.selectbox(
    "Classifier",
    ["ANN", "Random Forest", "XGBoost", "Logistic Regression"],
    index=0
)

# Decision Threshold Slider
st.sidebar.markdown("---")
st.sidebar.subheader(r"🎯 Decision Threshold ($\Theta$)")
decision_threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Default is 0.50. Lower thresholds (e.g. 0.30 - 0.40) increase Recall to aggressively catch more at-risk customers."
)
if decision_threshold < 0.50:
    st.sidebar.caption(f"⚡ **High-Recall Mode**: Catching potential churners earlier (Threshold: {decision_threshold:.2f}).")
elif decision_threshold > 0.50:
    st.sidebar.caption(f"🎯 **High-Precision Mode**: Flagging only high-confidence churners (Threshold: {decision_threshold:.2f}).")

# Load Raw Dataset for EDA
@st.cache_data
def load_raw_dataset():
    if os.path.exists("data/Churn_Modelling.csv"):
        return pd.read_csv("data/Churn_Modelling.csv")
    return None

df_raw = load_raw_dataset()

# ----------------- PAGE 1: CHURN PREDICTOR -----------------
if page == "🔮 Single Churn Predictor":
    st.markdown("<h1 class='main-header'>🏦 BANK CUSTOMER CHURN AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Real-time risk scoring, Explainable AI factor breakdown, CLV estimation & What-If retention simulation.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.8, 1.2])
    
    with col1:
        with st.container(border=True):
            st.markdown("<div class='card-title'>Customer Demographics & Account Information</div>", unsafe_allow_html=True)
            
            grid1_1, grid1_2 = st.columns(2)
            with grid1_1:
                credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650, step=1)
                geography = st.selectbox("Geography / Country", ["India", "United States", "United Kingdom", "Canada", "Australia", "France", "Germany", "Spain"])
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.slider("Age (Years)", min_value=18, max_value=100, value=42, step=1)
                tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=4, step=1)
                
            with grid1_2:
                balance = st.number_input("Account Balance ($)", min_value=0.0, value=85000.0, step=2500.0, format="%.2f")
                num_products = st.selectbox("Number of Products Held", [1, 2, 3, 4], index=0)
                has_cr_card = st.radio("Credit Card Holder?", ["Yes", "No"], index=0, horizontal=True)
                is_active = st.radio("Is Active Banking Member?", ["Yes", "No"], index=1, horizontal=True)
                estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, value=95000.0, step=5000.0, format="%.2f")
                
            predict_btn = st.button("🔮 Calculate Churn Risk & Intelligence", use_container_width=True)

    customer_dict = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": estimated_salary
    }
    
    # Run prediction by default or upon button click
    result = predict_single(customer_dict, model_name=selected_model, threshold=decision_threshold)
    clv_metrics = calculate_customer_clv(customer_dict)
    explanations = explain_customer_prediction(customer_dict, model_name=selected_model)
    prob_pct = result["probability"] * 100
    risk_cat = result["risk_category"]
    
    with col2:
        with st.container(border=True):
            st.markdown("<div class='card-title'>AI Assessment Output</div>", unsafe_allow_html=True)
            
            # Risk Banner Styling
            if risk_cat == "Low Risk":
                banner_class = "risk-banner risk-low"
            elif risk_cat == "Medium Risk":
                banner_class = "risk-banner risk-medium"
            else:
                banner_class = "risk-banner risk-high"
                
            st.markdown(f"<div class='{banner_class}'>{risk_cat.upper()} ({prob_pct:.1f}%)</div>", unsafe_allow_html=True)
            
            # Gauge chart in Cotton Candy theme
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'suffix': "%", 'font': {'color': '#2C3E50', 'family': 'Outfit', 'size': 34, 'weight': 800}},
                title = {'text': f"Model: {selected_model} (Threshold: {decision_threshold:.2f})", 'font': {'size': 14, 'color': '#64748b', 'family': 'Outfit'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1.5, 'tickcolor': "#2C3E50", 'tickfont': {'color': '#2C3E50', 'family': 'Outfit'}},
                    'bar': {'color': "#7971ea"},
                    'bgcolor': "rgba(44, 62, 80, 0.05)",
                    'borderwidth': 1.5,
                    'bordercolor': "rgba(44, 62, 80, 0.15)",
                    'steps': [
                        {'range': [0, decision_threshold * 60], 'color': 'rgba(16, 185, 129, 0.25)'},
                        {'range': [decision_threshold * 60, decision_threshold * 100], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [decision_threshold * 100, 100], 'color': 'rgba(244, 63, 94, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ff758c", 'width': 4},
                        'thickness': 0.75,
                        'value': prob_pct
                    }
                }
            ))
            fig_gauge.update_layout(
                height=220, 
                margin=dict(l=15, r=15, t=30, b=15),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Financial Impact Cards
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Est. Customer Lifetime Value</div>
                    <div class='kpi-value'>${clv_metrics['clv']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with f_col2:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Annual Banking Revenue</div>
                    <div class='kpi-value'>${clv_metrics['annual_revenue']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Strategic Retention Recommendation")
            st.markdown(f"<div class='recommendation-box'>{result['recommendation']}</div>", unsafe_allow_html=True)

    # ----------------- SECTION: EXPLAINABLE AI (XAI) & WHAT-IF SIMULATOR -----------------
    st.markdown("---")
    res_col1, res_col2 = st.columns([1.3, 1.7])

    with res_col1:
        with st.container(border=True):
            st.markdown("<div class='card-title'>🔍 Explainable AI: Key Risk Drivers & Protective Factors</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.92rem; color: #64748b;'>Breakdown of customer attributes influencing the churn score:</p>", unsafe_allow_html=True)
            
            # Factors Horizontal Bar chart
            factors_df = pd.DataFrame(explanations)
            fig_factors = px.bar(
                factors_df,
                x="score",
                y="feature",
                orientation="h",
                color="impact",
                color_discrete_map={"Increases Risk": "#f43f5e", "Reduces Risk": "#10b981", "Neutral / Slight Risk": "#f59e0b"},
                labels={"score": "Risk Contribution Impact", "feature": "Customer Attribute"}
            )
            fig_factors.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(color="#2C3E50", size=11, family='Outfit')),
                xaxis=dict(tickfont=dict(color="#2C3E50", family='Outfit'), gridcolor="rgba(44, 62, 80, 0.08)"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(font=dict(color="#2C3E50", size=10, family='Outfit'), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_factors, use_container_width=True)

            for exp in explanations[:3]:
                is_danger = exp["impact"] == "Increases Risk"
                cls_name = "factor-positive" if is_danger else "factor-negative"
                icon = "⚠️" if is_danger else "🛡️"
                st.markdown(f"""
                <div class='factor-pill {cls_name}'>
                    <div><strong>{icon} {exp['feature']}</strong>: {exp['description']}</div>
                </div>
                """, unsafe_allow_html=True)

    with res_col2:
        with st.container(border=True):
            st.markdown("<div class='card-title'>💡 Interactive 'What-If' Retention Simulator</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.92rem; color: #64748b;'>Simulate the impact of proposed retention interventions on this customer's churn risk:</p>", unsafe_allow_html=True)

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                sim_active = st.selectbox("Simulate Membership Status", ["Active Member (1)", "Inactive Member (0)"], index=0 if is_active == "Yes" else 0)
            with sim_c2:
                sim_products = st.selectbox("Simulate Products Held", [1, 2, 3, 4], index=min(3, max(0, num_products)))
            with sim_c3:
                sim_has_card = st.selectbox("Simulate Credit Card", ["Has Card (1)", "No Card (0)"], index=0 if has_cr_card == "Yes" else 0)

            adjustments = {
                "IsActiveMember": 1 if "Active" in sim_active else 0,
                "NumOfProducts": sim_products,
                "HasCrCard": 1 if "Has Card" in sim_has_card else 0
            }

            sim_result = simulate_retention_scenario(
                customer_dict, 
                adjustments, 
                model_name=selected_model, 
                threshold=decision_threshold
            )

            # Simulation Metrics Row
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Current Churn Prob</div>
                    <div class='kpi-value' style='color: #f43f5e;'>{sim_result['base_probability']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with s_col2:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Simulated Churn Prob</div>
                    <div class='kpi-value' style='color: #10b981;'>{sim_result['simulated_probability']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with s_col3:
                delta_sign = "+" if sim_result['delta_pct'] > 0 else ""
                delta_color = "#10b981" if sim_result['delta_pct'] <= 0 else "#f43f5e"
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Risk Change (&Delta;)</div>
                    <div class='kpi-value' style='color: {delta_color};'>{delta_sign}{sim_result['delta_pct']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            if sim_result["delta_prob"] < 0:
                st.success(f"🎉 **Retention Opportunity**: Applying these actions reduces churn probability by **{abs(sim_result['delta_pct']):.1f}%** (moving risk from {sim_result['base_risk']} to **{sim_result['simulated_risk']}**).")
            else:
                st.warning(f"⚠️ Scenario projects a **{sim_result['delta_pct']:+.1f}%** shift in churn risk.")

            # Download Executive Memo Button
            report_text = generate_executive_report_text(customer_dict, result, clv_metrics, explanations)
            st.download_button(
                label="📄 Download Executive Customer Churn Report (.TXT)",
                data=report_text,
                file_name=f"churn_memo_customer_{credit_score}_{age}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ----------------- PAGE 2: BATCH PREDICTIONS & PORTFOLIO RISK -----------------
elif page == "📂 Batch Predictions & Portfolio Risk":
    st.markdown("<h1 class='main-header'>📂 BATCH PREDICTIONS & REVENUE RISK</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Bulk customer risk evaluation, portfolio financial exposure, and automated test dataset generator.</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='card-title'>Upload Dataset or Generate Synthetic Sample</div>", unsafe_allow_html=True)
        
        b_col1, b_col2 = st.columns([1.5, 1])
        with b_col1:
            uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
            st.caption("Required columns: `CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary`")
            
        with b_col2:
            st.markdown("<div style='margin-top: 0.5rem;'><strong>Need Sample Data to Test?</strong></div>", unsafe_allow_html=True)
            sample_df = generate_sample_batch_csv(50)
            sample_csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample Batch CSV (50 Customers)",
                data=sample_csv_bytes,
                file_name="sample_bank_churn_test_batch.csv",
                mime="text/csv",
                use_container_width=True
            )
            use_sample = st.button("🚀 Load Sample Data Directly into View", use_container_width=True)

    df_to_process = None
    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` containing {len(df_to_process)} records.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif use_sample:
        df_to_process = sample_df.copy()
        st.info("Loaded 50 synthetic customer records for immediate evaluation.")

    if df_to_process is not None:
        with st.spinner("Executing Batch Intelligence & Risk Valuation..."):
            df_results = predict_batch(df_to_process, model_name=selected_model, threshold=decision_threshold)
            
            st.markdown("---")
            st.subheader("💰 Portfolio Financial & Churn Risk Metrics")
            
            total_customers = len(df_results)
            high_risk_df = df_results[df_results["Risk Category"] == "High Risk"]
            high_risk_count = len(high_risk_df)
            churn_rate = (high_risk_count / total_customers) * 100 if total_customers > 0 else 0
            
            total_balance_at_risk = high_risk_df["Balance"].sum() if "Balance" in high_risk_df.columns else 0
            total_annual_rev_at_risk = df_results["Revenue At Risk ($)"].sum()
            projected_retention_savings = total_annual_rev_at_risk * 0.20 # 20% rescue rate
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Total Customers</div>
                    <div class='kpi-value'>{total_customers:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>High Risk Customers</div>
                    <div class='kpi-value' style='color: #f43f5e;'>{high_risk_count:,} ({churn_rate:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Total Balance at Risk</div>
                    <div class='kpi-value' style='color: #f59e0b;'>${total_balance_at_risk:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>Annual Revenue at Risk</div>
                    <div class='kpi-value' style='color: #7971ea;'>${total_annual_rev_at_risk:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Visualizations Row
            c1, c2 = st.columns([1, 1.5])
            with c1:
                fig_pie = px.pie(
                    df_results, 
                    names='Risk Category', 
                    color='Risk Category',
                    color_discrete_map={
                        'Low Risk': '#10B981',
                        'Medium Risk': '#F59E0B',
                        'High Risk': '#F43F5E'
                    },
                    hole=0.45
                )
                apply_cotton_candy_theme(fig_pie, 'Portfolio Churn Segmentation')
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                if "Geography" in df_results.columns:
                    geo_rev = df_results.groupby("Geography")["Revenue At Risk ($)"].sum().reset_index()
                    fig_geo = px.bar(
                        geo_rev, 
                        x="Geography", 
                        y="Revenue At Risk ($)", 
                        color="Geography",
                        labels={"Revenue At Risk ($)": "Revenue at Risk ($)"},
                        color_discrete_sequence=['#ff758c', '#7971ea', '#06b6d4']
                    )
                    apply_cotton_candy_theme(fig_geo, 'Revenue at Risk by Geography ($)')
                    st.plotly_chart(fig_geo, use_container_width=True)

            # Filterable Table
            st.subheader("📋 Enriched Batch Predictions Data Table")
            filter_risk = st.multiselect("Filter by Risk Category:", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk", "Low Risk"])
            df_filtered = df_results[df_results["Risk Category"].isin(filter_risk)]
            
            st.dataframe(df_filtered, use_container_width=True)
            
            # Download full results
            full_csv_bytes = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Export Enriched Predictions & Valuation CSV",
                data=full_csv_bytes,
                file_name="churn_predictions_with_clv_output.csv",
                mime="text/csv",
                use_container_width=True
            )

# ----------------- PAGE 3: MODEL PERFORMANCE & EXPLAINABILITY -----------------
elif page == "📊 Model Performance & Explainability":
    st.markdown("<h1 class='main-header'>📊 SYSTEM BENCHMARKS & EXPLAINABILITY</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Global Feature Importance rankings, performance benchmarks, and decision boundary analysis.</p>", unsafe_allow_html=True)
    
    tab_perf, tab_feat, tab_eda = st.tabs([
        "🏆 Model Comparisons", 
        "🔍 Global Feature Importance", 
        "🌌 3D Space & Data Exploration"
    ])
    
    with tab_perf:
        st.subheader("🤖 Multi-Model Benchmark Comparison")
        metrics = load_comparison_metrics()
        
        if metrics is not None:
            perf_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Classifier"})
            
            fig = go.Figure()
            colors = ["#ff758c", "#ff7eb3", "#7971ea", "#06b6d4", "#10b981"]
            for idx, metric_name in enumerate(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]):
                fig.add_trace(go.Bar(
                    x=perf_df["Classifier"],
                    y=perf_df[metric_name],
                    name=metric_name,
                    marker_color=colors[idx % len(colors)]
                ))
            fig.update_layout(barmode='group')
            apply_cotton_candy_theme(fig, 'Multi-Metric Performance Across Trained Classifiers')
            fig.update_layout(yaxis=dict(range=[0, 1.05]))
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(perf_df, use_container_width=True)
            
            st.info("""
            💡 **Banking Context**: **Recall** is prioritized in churn prediction because identifying an at-risk customer early allows proactive retention at a minor incentive cost, whereas missing a churning customer results in a permanent loss of account balances and recurring lifetime value.
            """)
        else:
            st.warning("Performance benchmark metrics not found. Run 'python src/train_model.py' first.")

    with tab_feat:
        st.subheader("🔍 Global Feature Importance Analysis")
        st.markdown("Feature importance rankings identify the core statistical factors that drive customer churn across the bank's entire customer portfolio.")
        
        feat_imp_dict = get_global_feature_importance()
        
        fi_c1, fi_c2 = st.columns(2)
        if "Random Forest" in feat_imp_dict:
            with fi_c1:
                fig_rf = px.bar(
                    feat_imp_dict["Random Forest"].head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Random Forest Feature Importance (MDI)",
                    color="Importance",
                    color_continuous_scale=["#B5FFFC", "#7971ea", "#ff758c"]
                )
                fig_rf.update_layout(yaxis=dict(autorange="reversed"))
                apply_cotton_candy_theme(fig_rf, 'Random Forest Feature Importance')
                st.plotly_chart(fig_rf, use_container_width=True)

        if "XGBoost" in feat_imp_dict:
            with fi_c2:
                fig_xgb = px.bar(
                    feat_imp_dict["XGBoost"].head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="XGBoost Feature Importance (Gain)",
                    color="Importance",
                    color_continuous_scale=["#B5FFFC", "#ff7eb3", "#ff758c"]
                )
                fig_xgb.update_layout(yaxis=dict(autorange="reversed"))
                apply_cotton_candy_theme(fig_xgb, 'XGBoost Feature Importance')
                st.plotly_chart(fig_xgb, use_container_width=True)

    with tab_eda:
        if df_raw is not None:
            st.markdown("### 🌌 3D Customer Demographics Space")
            st.markdown("Interact with the 3D customer distribution: **Age** (Y-axis), **Account Balance** (Z-axis), and **Credit Score** (X-axis). Rotate, tilt, and zoom in 3D space.")
            
            df_3d = df_raw.sample(n=min(1500, len(df_raw)), random_state=42)
            df_3d['Status'] = df_3d['Exited'].map({0: 'Retained', 1: 'Churned'})
            
            fig_3d = px.scatter_3d(
                df_3d,
                x='CreditScore',
                y='Age',
                z='Balance',
                color='Status',
                color_discrete_map={'Retained': '#10B981', 'Churned': '#F43F5E'},
                opacity=0.75,
                size_max=8,
                hover_data=['Geography', 'Gender', 'NumOfProducts', 'EstimatedSalary']
            )
            
            fig_3d.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis_title='Credit Score',
                    yaxis_title='Age',
                    zaxis_title='Account Balance ($)',
                    xaxis=dict(backgroundcolor="rgba(255, 255, 255, 0.4)", gridcolor="rgba(44, 62, 80, 0.1)", showbackground=True, tickfont=dict(color="#2C3E50"), titlefont=dict(color="#2C3E50")),
                    yaxis=dict(backgroundcolor="rgba(255, 255, 255, 0.4)", gridcolor="rgba(44, 62, 80, 0.1)", showbackground=True, tickfont=dict(color="#2C3E50"), titlefont=dict(color="#2C3E50")),
                    zaxis=dict(backgroundcolor="rgba(255, 255, 255, 0.4)", gridcolor="rgba(44, 62, 80, 0.1)", showbackground=True, tickfont=dict(color="#2C3E50"), titlefont=dict(color="#2C3E50"))
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    font=dict(color="#2C3E50")
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # 2D EDA tabs
            eda_col1, eda_col2 = st.columns(2)
            with eda_col1:
                geo_churn = df_raw.groupby('Geography')['Exited'].mean().reset_index()
                geo_churn['Exited'] = geo_churn['Exited'] * 100
                fig_geo = px.bar(geo_churn, x='Geography', y='Exited', labels={'Exited': 'Churn Rate (%)'}, color='Geography', color_discrete_sequence=['#ff758c', '#7971ea', '#06b6d4'])
                apply_cotton_candy_theme(fig_geo, 'Average Churn Rate by Geography (%)')
                st.plotly_chart(fig_geo, use_container_width=True)
                
            with eda_col2:
                prod_churn = df_raw.groupby('NumOfProducts')['Exited'].mean().reset_index()
                prod_churn['Exited'] = prod_churn['Exited'] * 100
                fig_prod = px.bar(prod_churn, x='NumOfProducts', y='Exited', labels={'Exited': 'Churn Rate (%)'}, color='NumOfProducts', color_discrete_sequence=['#ff758c', '#7971ea', '#ff7eb3', '#06b6d4'])
                apply_cotton_candy_theme(fig_prod, 'Churn Rate by Products Held (%)')
                st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.warning("Customer dataset not found.")
