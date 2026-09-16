"""
app.py - Malaysia Overtourism Risk Classifier & Forecasting Lab (DOSM Datathon 2026)

A dual-feature Streamlit web application providing:
1. K-Means clustering (k=3) for state overtourism tier classification ('Overcrowded', 'Balanced', 'Under-visited')
   and hypothetical area scenario risk prediction.
2. ARIMA Time Series Forecasting with automated AIC model selection for multi-year visitor growth projections (2024-2026+).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import modular helper functions
from data_processing import load_tourism_data, prepare_features
from model import train_kmeans_model, predict_tier
from forecast import (
    load_historical_tourism_data,
    get_state_time_series,
    find_best_arima_model,
    forecast_future_visitors,
    forecast_future_metric
)
from socio_data import load_and_clean_socio_data

# ==============================================================================
# 1. Page Configuration & Custom CSS
# ==============================================================================
st.set_page_config(
    page_title="Malaysia Overtourism Risk Classifier & Forecasting | DOSM Datathon 2026",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling, clean metric cards, and colored badges
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 4px;
        color: #f8fafc;
    }
    
    /* Result Badges */
    .badge-overcrowded {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1.5px solid #ef4444;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-balanced {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1.5px solid #f59e0b;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-undervisited {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1.5px solid #10b981;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Disagreement / Callout boxes */
    .disagree-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .forecast-explanation-box {
        background-color: rgba(99, 102, 241, 0.12);
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 0.92rem;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. Sidebar: Project Info, Settings & Data Source Credits
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/malaysia.png", width=64)
    st.title("🇲🇾 Overtourism Lab")
    st.caption("**DOSM Datathon 2026 Track: Sustainable Tourism**")
    
    st.markdown("---")
    st.markdown("### 📌 Project Description")
    st.markdown("""
    This application integrates **two complementary AI/ML analytical engines**:
    
    1. **K-Means Clustering ($k=3$):** Unsupervised classification of states into *Overcrowded*, *Balanced*, and *Under-visited* tiers based on **Density** (visitors/resident) and **Spend per Visitor** (RM).
    2. **ARIMA Time Series Forecasting:** Automated $(p,d,q)$ parameter optimization via **AIC minimization** to project future visitor volumes (2024–2026+).
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Configurations")
    scaler_choice = st.selectbox(
        "K-Means Feature Scaler",
        options=["standard", "minmax"],
        format_func=lambda x: "StandardScaler (Z-score)" if x == "standard" else "MinMaxScaler [0, 1]",
        help="Choose how Density and Spend per Visitor are normalized before clustering."
    )
    
    random_seed = st.number_input("K-Means Random Seed", value=42, min_value=0, max_value=9999, step=1)
    
    st.markdown("---")
    st.markdown("### 📊 Data Source Credits")
    st.markdown("""
    - **Portal:** [OpenDOSM (open.dosm.gov.my)](https://open.dosm.gov.my)
    - **Surveys:** Domestic Tourism by State (2018–2023)
    - **Demographics:** Population Table by State (2023)
    """)
    st.info("Primary Dataset: `Sustainable_Tourism_V1.xlsx`\nTime Series: `historical_tourism.csv`")


# ==============================================================================
# 3. Data Pipeline & Caching
# ==============================================================================
@st.cache_data
def get_kmeans_pipeline(scaler_type: str, seed: int):
    df_raw = load_tourism_data("Sustainable_Tourism_V1.xlsx", "Cleaned_Data_V3")
    X_scaled, scaler, feature_cols = prepare_features(df_raw, scaler_type=scaler_type)
    model, df_results, label_map, cluster_summary = train_kmeans_model(X_scaled, df_raw, random_state=seed)
    return df_results, scaler, model, label_map, cluster_summary, feature_cols

@st.cache_data
def get_historical_dataset():
    return load_historical_tourism_data("historical_tourism.csv")

@st.cache_data
def get_socio_dataset():
    return load_and_clean_socio_data()

try:
    df_results, scaler, model, label_map, cluster_summary, feature_cols = get_kmeans_pipeline(scaler_choice, random_seed)
    df_history = get_historical_dataset()
    df_socio = get_socio_dataset()
except Exception as e:
    st.error(f"Error loading datasets or training models: {e}")
    st.stop()


# ==============================================================================
# 4. Main App Header & Navigation Tabs
# ==============================================================================
st.title("🇲🇾 Malaysia Overtourism Risk Classifier")
st.markdown("**A Machine Learning & Time Series Forecasting Framework for Sustainable Tourism Planning**")

# Top KPI Summary row across the 16 states
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
total_states = len(df_results)
overcrowded_cnt = (df_results["KMeans_Predicted_Tier"] == "Overcrowded").sum()
balanced_cnt = (df_results["KMeans_Predicted_Tier"] == "Balanced").sum()
undervisited_cnt = (df_results["KMeans_Predicted_Tier"] == "Under-visited").sum()

with col_kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total States Analyzed</div>
        <div class="metric-value">{total_states}</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="metric-card" style="border-color: rgba(239, 68, 68, 0.4);">
        <div class="metric-title" style="color: #f87171;">🔴 Overcrowded Tier</div>
        <div class="metric-value" style="color: #f87171;">{overcrowded_cnt}</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
    <div class="metric-card" style="border-color: rgba(245, 158, 11, 0.4);">
        <div class="metric-title" style="color: #fbbf24;">🟡 Balanced Tier</div>
        <div class="metric-value" style="color: #fbbf24;">{balanced_cnt}</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    st.markdown(f"""
    <div class="metric-card" style="border-color: rgba(16, 185, 129, 0.4);">
        <div class="metric-title" style="color: #34d399;">🟢 Under-visited Tier</div>
        <div class="metric-value" style="color: #34d399;">{undervisited_cnt}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Dual Main Tabs
tab_kmeans, tab_arima, tab_socio = st.tabs([
    "🎯 1. Overtourism Risk Classifier (K-Means)",
    "📈 2. Tourism Forecasting (ARIMA)",
    "📊 3. Socio-Economic Forecasting (ARIMA)"
])


# ==============================================================================
# TAB 1: K-Means Overtourism Tier Classification & Scenario Predictor
# ==============================================================================
with tab_kmeans:
    # --------------------------------------------------------------------------
    # Section 1: 16 States Comparison Table (Rule-Based vs K-Means)
    # --------------------------------------------------------------------------
    st.markdown("### 📋 State-by-State Classification Comparison")
    st.caption("Compare existing DOSM rule-based tiers against the machine-learned K-Means clustering tiers. Disagreements are highlighted.")

    display_df = df_results[[
        "State", 
        "Domestic_Visitors", 
        "Population", 
        "Density", 
        "Spend_Per_Visitor_RM", 
        "Rule_Based_Tier", 
        "KMeans_Predicted_Tier", 
        "Agreement"
    ]].copy()

    def highlight_disagreements(row):
        if not row["Agreement"]:
            return ["background-color: rgba(239, 68, 68, 0.18); color: #ffffff; font-weight: bold;"] * len(row)
        return [""] * len(row)

    styled_table = display_df.style.apply(highlight_disagreements, axis=1).format({
        "Domestic_Visitors": "{:,.0f}",
        "Population": "{:,.0f}",
        "Density": "{:.2f}",
        "Spend_Per_Visitor_RM": "RM {:.2f}",
        "Agreement": lambda x: "✅ Match" if x else "⚠️ Disagreement"
    })

    st.dataframe(styled_table, use_container_width=True, height=460)

    # Disagreement Explanation box
    disagreed_states = df_results[~df_results["Agreement"]]
    if len(disagreed_states) > 0:
        st.markdown(f"""
        <div class="disagree-box">
            <strong>🔍 Model Insights on Classification Disagreements ({len(disagreed_states)} States):</strong><br>
            {', '.join([f"<strong>{r['State']}</strong> (Rule: <em>{r['Rule_Based_Tier']}</em> → K-Means: <em>{r['KMeans_Predicted_Tier']}</em>, Density: {r['Density']:.2f}, Spend: RM {r['Spend_Per_Visitor_RM']:.2f})" for _, r in disagreed_states.iterrows()])}
            <br><br>
            <em>Rationale: K-Means clusters simultaneously on both Density and Spend per Visitor, uncovering multidimensional patterns that one-dimensional density thresholds miss!</em>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Section 2: Interactive Scenario Simulator / Area Prediction
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧪 Hypothetical Area / State Risk Predictor")
    st.caption("Enter tourism and demographic parameters for any destination or administrative district to predict its overtourism tier.")

    pred_col1, pred_col2 = st.columns([1.1, 0.9])

    with pred_col1:
        st.markdown("#### Input Parameters")
        
        st.write("💡 *Prefill with actual state profiles:*")
        preset_cols = st.columns(4)
        preset_vals = None
        if preset_cols[0].button("🏙️ Melaka"):
            preset_vals = (15558661, 1028300, 6438411057)
        if preset_cols[1].button("🌳 Selangor"):
            preset_vals = (27579478, 7209700, 11102713152)
        if preset_cols[2].button("🏝️ Labuan"):
            preset_vals = (331360, 99000, 194064160)
        if preset_cols[3].button("🏛️ KL"):
            preset_vals = (22232643, 2005700, 10995324143)

        default_visitors = preset_vals[0] if preset_vals else 10000000
        default_pop = preset_vals[1] if preset_vals else 1500000
        default_receipts = preset_vals[2] if preset_vals else 4000000000

        input_visitors = st.number_input(
            "Domestic Visitors (Number of tourists)",
            min_value=1000,
            max_value=100000000,
            value=default_visitors,
            step=100000,
            help="Total annual domestic visitors to the area."
        )
        
        input_population = st.number_input(
            "Resident Population (Number of people)",
            min_value=1000,
            max_value=50000000,
            value=default_pop,
            step=50000,
            help="Total local resident population of the area."
        )
        
        input_receipts = st.number_input(
            "Total Tourism Receipts in RM (Revenue)",
            min_value=1000,
            max_value=100000000000,
            value=default_receipts,
            step=10000000,
            help="Total annual expenditure by visitors in Ringgit Malaysia."
        )

        calc_density = input_visitors / input_population if input_population > 0 else 0
        calc_spend_per_visitor = input_receipts / input_visitors if input_visitors > 0 else 0

    with pred_col2:
        st.markdown("#### Prediction Result")
        
        pred_result = predict_tier(calc_density, calc_spend_per_visitor, scaler, model, label_map)
        pred_tier = pred_result["predicted_tier"]
        
        if pred_tier == "Overcrowded":
            badge_html = '<div class="badge-overcrowded">🔴 OVERCROWDED TIER</div>'
            tier_msg = "High risk of carrying capacity stress, traffic congestion, and infrastructure strain relative to local residents."
        elif pred_tier == "Balanced":
            badge_html = '<div class="badge-balanced">🟡 BALANCED TIER</div>'
            tier_msg = "Healthy visitor-to-resident ratio with steady economic contribution."
        else:
            badge_html = '<div class="badge-undervisited">🟢 UNDER-VISITED TIER</div>'
            tier_msg = "Significant growth capacity; low visitor density relative to resident population."

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">K-Means Predicted Classification</div>
            {badge_html}
            <div style="margin-top: 14px; font-size: 0.88rem; color: #cbd5e1;">{tier_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        st.markdown(f"""
        <div style="display: flex; gap: 10px;">
            <div class="metric-card" style="flex: 1;">
                <div class="metric-title">Calculated Density</div>
                <div class="metric-value" style="font-size: 1.3rem; color: #38bdf8;">{calc_density:.2f}</div>
                <div style="font-size: 0.7rem; color: #94a3b8;">visitors per resident</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div class="metric-title">Spend per Visitor</div>
                <div class="metric-value" style="font-size: 1.3rem; color: #38bdf8;">RM {calc_spend_per_visitor:.2f}</div>
                <div style="font-size: 0.7rem; color: #94a3b8;">average spend in RM</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Section 3: Visual Analytics & Cluster Benchmark Charts
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Visual Benchmarks & Cluster Diagnostics")

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("#### Comparison with Cluster Averages")
        
        comp_data = []
        for _, row in cluster_summary.iterrows():
            comp_data.append({
                "Category": f"Cluster: {row['KMeans_Predicted_Tier']}",
                "Density": row["Avg_Density"],
                "Spend_Per_Visitor": row["Avg_Spend"],
                "Type": "Cluster Average"
            })
        comp_data.append({
            "Category": "Your Input (Hypothetical)",
            "Density": calc_density,
            "Spend_Per_Visitor": calc_spend_per_visitor,
            "Type": "Your Scenario"
        })
        comp_df = pd.DataFrame(comp_data)
        
        metric_to_plot = st.radio(
            "Select Metric to Compare:",
            ["Density (visitors/resident)", "Spend Per Visitor (RM)"],
            horizontal=True
        )
        
        y_col = "Density" if "Density" in metric_to_plot else "Spend_Per_Visitor"
        
        fig_bar = px.bar(
            comp_df,
            x="Category",
            y=y_col,
            color="Type",
            color_discrete_map={
                "Cluster Average": "#6366f1",
                "Your Scenario": "#06b6d4"
            },
            text_auto=".2f",
            title=f"Benchmark: {metric_to_plot}"
        )
        fig_bar.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title=metric_to_plot,
            legend_title="",
            height=380
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.markdown("#### 2D Cluster Space & Scenario Placement")
        
        color_map = {
            "Overcrowded": "#ef4444",
            "Balanced": "#f59e0b",
            "Under-visited": "#10b981"
        }
        
        fig_scatter = px.scatter(
            df_results,
            x="Density",
            y="Spend_Per_Visitor_RM",
            color="KMeans_Predicted_Tier",
            text="State",
            color_discrete_map=color_map,
            hover_data=["Domestic_Visitors", "Population", "Rule_Based_Tier"],
            title="Malaysian States in Feature Space (Density vs Spend)"
        )
        
        fig_scatter.add_trace(go.Scatter(
            x=[calc_density],
            y=[calc_spend_per_visitor],
            mode="markers+text",
            name="Your Scenario",
            text=["📍 Your Input"],
            textposition="top center",
            marker=dict(size=16, color="#06b6d4", symbol="star", line=dict(color="#ffffff", width=2))
        ))
        
        fig_scatter.update_traces(textposition="top right")
        fig_scatter.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Visitor Density (visitors per resident)",
            yaxis_title="Spend per Visitor (RM)",
            legend_title="Predicted Tier",
            height=380
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# TAB 2: Time Series Visitor Forecasting (ARIMA)
# ==============================================================================
with tab_arima:
    st.markdown("### 📈 Time Series Visitor Growth & Carrying Capacity Forecasting")
    
    # Required short explanation text
    st.markdown("""
    <div class="forecast-explanation-box">
        💡 <strong>Analytical Context:</strong> This forecast estimates how visitor numbers may change if current trends continue, which can indicate rising overtourism risk.
    </div>
    """, unsafe_allow_html=True)

    # State Selection and Horizon Controls
    f_col1, f_col2, f_col3 = st.columns([1.2, 0.8, 1.0])
    
    state_list = sorted(df_history["State"].unique().tolist())
    
    with f_col1:
        selected_state = st.selectbox(
            "Select Malaysian State / Territory",
            options=state_list,
            index=state_list.index("Melaka") if "Melaka" in state_list else 0,
            help="Choose a state to fit its historical time series (2018-2023)."
        )
        
    with f_col2:
        forecast_horizon = st.slider(
            "Forecast Horizon (Years ahead)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Number of future years to forecast (e.g. 2024 to 2026+)."
        )

    with f_col3:
        st.markdown("**Active Time Series Data:**")
        st.caption(f"Historical span: **2018 – 2023** (DOSM Domestic Tourism Survey)")

    # Extract state series and fit ARIMA
    state_series = get_state_time_series(df_history, selected_state)
    
    with st.spinner(f"Optimizing ARIMA model for {selected_state}..."):
        best_fit, best_order, best_aic, grid_results_df = find_best_arima_model(state_series)
        forecast_df, combined_df, overall_growth = forecast_future_visitors(best_fit, state_series, forecast_years=forecast_horizon)

    # Summary Metrics Row for Forecasting
    last_year_actual = int(state_series.values[-1])
    target_year_pred = int(forecast_df.iloc[-1]["Forecasted_Visitors"])
    target_year = int(forecast_df.iloc[-1]["Year"])

    st.write("")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Optimal Model Order</div>
            <div class="metric-value" style="font-size: 1.3rem; color: #818cf8;">ARIMA{best_order}</div>
            <div style="font-size: 0.72rem; color: #94a3b8;">Lowest AIC: {best_aic:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">2023 Actual Visitors</div>
            <div class="metric-value" style="font-size: 1.3rem;">{last_year_actual:,.0f}</div>
            <div style="font-size: 0.72rem; color: #94a3b8;">Baseline volume</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        growth_color = "#f87171" if overall_growth > 20 else "#fbbf24" if overall_growth > 0 else "#34d399"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Projected {target_year} Volume</div>
            <div class="metric-value" style="font-size: 1.3rem; color: {growth_color};">{target_year_pred:,.0f}</div>
            <div style="font-size: 0.72rem; color: {growth_color};">{overall_growth:+.1f}% vs 2023</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        risk_tag = "🔴 Escalating Pressure" if overall_growth > 15 else "🟡 Moderate / Stable" if overall_growth >= -10 else "🟢 Growth Buffer"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Trajectory Risk Status</div>
            <div class="metric-value" style="font-size: 1.1rem; margin-top: 8px;">{risk_tag}</div>
            <div style="font-size: 0.72rem; color: #94a3b8;">Multi-year trend signal</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Interactive Forecast Plotly Chart
    fig_forecast = go.Figure()

    # 1. Historical Actual Line
    hist_df = combined_df[combined_df["Type"] == "Historical (Actual)"]
    fig_forecast.add_trace(go.Scatter(
        x=hist_df["Year"],
        y=hist_df["Visitors"],
        mode="lines+markers",
        name="Historical Actual (2018–2023)",
        line=dict(color="#38bdf8", width=3.5),
        marker=dict(size=8, color="#38bdf8")
    ))

    # 2. Forecast Line (Connecting seamlessly from last historical point)
    last_hist_x = [hist_df["Year"].iloc[-1]]
    last_hist_y = [hist_df["Visitors"].iloc[-1]]
    
    pred_x = last_hist_x + forecast_df["Year"].tolist()
    pred_y = last_hist_y + forecast_df["Forecasted_Visitors"].tolist()

    fig_forecast.add_trace(go.Scatter(
        x=pred_x,
        y=pred_y,
        mode="lines+markers",
        name=f"ARIMA{best_order} Forecast",
        line=dict(color="#f43f5e", width=3.5, dash="dash"),
        marker=dict(size=9, color="#f43f5e", symbol="diamond")
    ))

    # 3. 95% Confidence Interval Shaded Area
    ci_x = forecast_df["Year"].tolist() + forecast_df["Year"].tolist()[::-1]
    ci_y = forecast_df["Upper_CI_95"].tolist() + forecast_df["Lower_CI_95"].tolist()[::-1]

    fig_forecast.add_trace(go.Scatter(
        x=ci_x,
        y=ci_y,
        fill="toself",
        fillcolor="rgba(244, 63, 94, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval"
    ))

    fig_forecast.update_layout(
        title=f"<b>Domestic Visitor Trajectory & Forecast: {selected_state}</b>",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Year", tickmode="linear", dtick=1),
        yaxis=dict(title="Domestic Visitors (Number of tourists)", tickformat=","),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Forecast Numbers Table and Model Evaluation Grid
    tab_f_table, tab_f_grid = st.columns([1.1, 0.9])

    with tab_f_table:
        st.markdown(f"#### 📑 Projected Figures for {selected_state}")
        
        display_forecast_df = forecast_df[[
            "Year", 
            "Forecasted_Visitors", 
            "Lower_CI_95", 
            "Upper_CI_95", 
            "YoY_Growth_Pct"
        ]].copy()
        
        display_forecast_df.rename(columns={
            "Forecasted_Visitors": "Projected Visitors",
            "Lower_CI_95": "95% CI Lower Bound",
            "Upper_CI_95": "95% CI Upper Bound",
            "YoY_Growth_Pct": "YoY Change (%)"
        }, inplace=True)

        st.dataframe(
            display_forecast_df.style.format({
                "Projected Visitors": "{:,.0f}",
                "95% CI Lower Bound": "{:,.0f}",
                "95% CI Upper Bound": "{:,.0f}",
                "YoY Change (%)": lambda x: f"{x:+.2f}%"
            }),
            use_container_width=True
        )

    with tab_f_grid:
        st.markdown("#### 🔬 ARIMA Parameter Search (AIC Rankings)")
        st.caption("Lower Akaike Information Criterion (AIC) indicates better goodness-of-fit with parsimony penalty.")
        
        st.dataframe(
            grid_results_df.head(6).style.format({
                "AIC": "{:.2f}",
                "BIC": "{:.2f}"
            }),
            use_container_width=True
        )

# ==============================================================================
# TAB 3: Socio-Economic Forecasting (ARIMA)
# ==============================================================================
with tab_socio:
    st.markdown("### 📊 Socio-Economic Metric Forecasting")
    
    st.markdown("""
    <div class="forecast-explanation-box">
        💡 <strong>Analytical Context:</strong> Forecast future trends for key state socio-economic metrics (Population, Labour Force, Poverty Rate, etc.) to aid in comprehensive sustainable tourism planning.
    </div>
    """, unsafe_allow_html=True)

    socio_col1, socio_col2, socio_col3 = st.columns([1.2, 1.0, 1.0])
    
    socio_state_list = sorted(df_socio["State"].unique().tolist())
    socio_metrics = [
        "Population", "Labour_Force_Size", "Employed", "Unemployed",
        "Unemployment_Rate", "Employment_Population_Ratio",
        "Mean_Income", "Gini", "Poverty_Rate"
    ]
    
    with socio_col1:
        sel_socio_state = st.selectbox(
            "Select State",
            options=socio_state_list,
            index=socio_state_list.index("Melaka") if "Melaka" in socio_state_list else 0,
            key="socio_state_sel"
        )
        
    with socio_col2:
        sel_socio_metric = st.selectbox(
            "Select Metric to Forecast",
            options=socio_metrics,
            index=0,
            key="socio_metric_sel"
        )
        
    with socio_col3:
        socio_horizon = st.slider(
            "Forecast Horizon (Years)",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key="socio_horizon"
        )

    # Extract series
    socio_series = get_state_time_series(df_socio, sel_socio_state, metric=sel_socio_metric)
    
    if len(socio_series) < 3:
        st.error("Not enough historical data points to fit an ARIMA model for this metric.")
    else:
        with st.spinner(f"Optimizing ARIMA model for {sel_socio_metric} in {sel_socio_state}..."):
            best_socio_fit, best_socio_order, best_socio_aic, _ = find_best_arima_model(socio_series)
            s_forecast_df, s_combined_df, s_overall_growth = forecast_future_metric(
                best_socio_fit, socio_series, forecast_years=socio_horizon, metric_name=sel_socio_metric
            )

        st.write("")
        sm_col1, sm_col2, sm_col3, sm_col4 = st.columns(4)

        last_s_actual = socio_series.values[-1]
        target_s_pred = s_forecast_df.iloc[-1][f"Forecasted_{sel_socio_metric}"]
        target_s_year = int(s_forecast_df.iloc[-1]["Year"])
        last_s_year = int(socio_series.index[-1])

        with sm_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Optimal Model Order</div>
                <div class="metric-value" style="font-size: 1.3rem; color: #818cf8;">ARIMA{best_socio_order}</div>
            </div>
            """, unsafe_allow_html=True)

        with sm_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{last_s_year} Actual ({sel_socio_metric})</div>
                <div class="metric-value" style="font-size: 1.3rem;">{last_s_actual:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with sm_col3:
            s_color = "#34d399" if s_overall_growth > 0 else "#f87171"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Projected {target_s_year}</div>
                <div class="metric-value" style="font-size: 1.3rem; color: {s_color};">{target_s_pred:,.2f}</div>
                <div style="font-size: 0.72rem; color: {s_color};">{s_overall_growth:+.1f}% vs {last_s_year}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with sm_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Trend</div>
                <div class="metric-value" style="font-size: 1.1rem; margin-top: 8px;">{"📈 Upward" if s_overall_growth > 0 else "📉 Downward" if s_overall_growth < 0 else "➖ Stable"}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Interactive Forecast Plotly Chart
        fig_s_forecast = go.Figure()

        # Historical Actual Line
        s_hist_df = s_combined_df[s_combined_df["Type"] == "Historical (Actual)"]
        fig_s_forecast.add_trace(go.Scatter(
            x=s_hist_df["Year"],
            y=s_hist_df["Value"],
            mode="lines+markers",
            name="Historical Actual",
            line=dict(color="#10b981", width=3.5),
            marker=dict(size=8, color="#10b981")
        ))

        # Forecast Line
        last_sh_x = [s_hist_df["Year"].iloc[-1]]
        last_sh_y = [s_hist_df["Value"].iloc[-1]]
        
        s_pred_x = last_sh_x + s_forecast_df["Year"].tolist()
        s_pred_y = last_sh_y + s_forecast_df[f"Forecasted_{sel_socio_metric}"].tolist()

        fig_s_forecast.add_trace(go.Scatter(
            x=s_pred_x,
            y=s_pred_y,
            mode="lines+markers",
            name=f"ARIMA{best_socio_order} Forecast",
            line=dict(color="#f59e0b", width=3.5, dash="dash"),
            marker=dict(size=9, color="#f59e0b", symbol="diamond")
        ))

        # 95% CI
        s_ci_x = s_forecast_df["Year"].tolist() + s_forecast_df["Year"].tolist()[::-1]
        s_ci_y = s_forecast_df["Upper_CI_95"].tolist() + s_forecast_df["Lower_CI_95"].tolist()[::-1]

        fig_s_forecast.add_trace(go.Scatter(
            x=s_ci_x,
            y=s_ci_y,
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Interval"
        ))

        fig_s_forecast.update_layout(
            title=f"<b>{sel_socio_metric} Trajectory & Forecast: {sel_socio_state}</b>",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Year", tickmode="linear", dtick=1),
            yaxis=dict(title=sel_socio_metric, tickformat=","),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )

        st.plotly_chart(fig_s_forecast, use_container_width=True)

