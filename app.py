"""Medium-Term Swing Trading Ecosystem: Research Station Dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
from datetime import datetime
import pandas_ta as ta
import logging

from src.feed.data import FastDataEngine
from src.feed.enrichment import CompanyEnrichment
from src.engine.cognition import CognitionEngine
from src.engine.strategies import StrategyRunner
from src.engine.assistant import TradeAssistant
from src.engine.analyst import IntelligenceEngine
from src.engine.calibration import CalibrationEngine
from src.models.schemas import TradeSignal
import config

logger = logging.getLogger(__name__)

# Page config - Mobile optimized
st.set_page_config(
    page_title="Swing Trading Dashboard - Nifty 500",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",  # Will be collapsed on mobile by default, but accessible via menu
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Swing Trading Dashboard for Nifty 500 - Accessible on mobile devices"
    }
)

# Material Design CSS - Light Theme Only
st.markdown("""
    <style>
    /* Material Design Light Theme */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .stApp {
        background: #fafafa;
        color: #212121;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Material Design Cards with Elevation */
    .material-card {
        background: #ffffff;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
        padding: 24px;
        margin-bottom: 16px;
        transition: box-shadow 0.3s ease;
    }
    
    .material-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.08);
    }
    
    .material-card-elevated {
        box-shadow: 0 8px 16px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.08);
    }
    
    /* KPI Cards - Material Design */
    .kpi-card {
        background: #ffffff;
        border-radius: 4px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left: 4px solid #2196F3;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .kpi-value {
        font-size: 36px;
        font-weight: 500;
        color: #1976D2;
        margin: 12px 0;
        font-family: 'Roboto', sans-serif;
    }
    
    .kpi-label {
        font-size: 14px;
        color: #757575;
        text-transform: uppercase;
        letter-spacing: 1.25px;
        font-weight: 500;
    }
    
    /* Market Status Bar - Material Design */
    .market-status-card {
        background: #ffffff;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2196F3;
    }
    
    .status-chip {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-aggressive {
        background: #E8F5E9;
        color: #2E7D32;
    }
    
    .status-defensive {
        background: #FFF3E0;
        color: #F57C00;
    }
    
    .status-cash {
        background: #FFEBEE;
        color: #C62828;
    }
    
    /* Opportunity List - Material Design */
    .opportunity-item {
        background: #ffffff;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        border-left: 4px solid transparent;
    }
    
    .opportunity-item:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateX(4px);
        border-left-color: #2196F3;
    }
    
    .opportunity-item.selected {
        background: #E3F2FD;
        border-left-color: #2196F3;
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.2);
    }
    
    /* Inspector Panel - Material Design */
    .inspector-panel {
        background: #ffffff;
        border-radius: 4px;
        padding: 24px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-height: 400px;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #E0E0E0;
    }
    
    .metric-row:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        color: #757575;
        font-size: 14px;
        font-weight: 400;
    }
    
    .metric-value {
        color: #212121;
        font-size: 16px;
        font-weight: 500;
    }
    
    .metric-value.positive {
        color: #2E7D32;
    }
    
    .metric-value.negative {
        color: #C62828;
    }
    
    /* Material Design Badges */
    .material-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 8px;
    }
    
    .badge-primary {
        background: #E3F2FD;
        color: #1976D2;
    }
    
    .badge-success {
        background: #E8F5E9;
        color: #2E7D32;
    }
    
    .badge-warning {
        background: #FFF3E0;
        color: #F57C00;
    }
    
    .badge-danger {
        background: #FFEBEE;
        color: #C62828;
    }
    
    /* Thesis Box - Material Design */
    .thesis-box {
        background: #F5F5F5;
        border-left: 4px solid #2196F3;
        border-radius: 4px;
        padding: 20px;
        margin: 16px 0;
        font-size: 15px;
        line-height: 1.75;
        color: #424242;
    }
    
    /* Material Design Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #212121;
        font-weight: 500;
    }
    
    /* Responsive Grid - Mobile Optimizations */
    @media (max-width: 768px) {
        .kpi-card {
            margin-bottom: 16px;
            padding: 16px;
        }
        
        .kpi-value {
            font-size: 28px;
        }
        
        .kpi-label {
            font-size: 12px;
        }
        
        .market-status-card {
            padding: 12px;
            margin-bottom: 12px;
        }
        
        .material-card {
            padding: 16px;
        }
        
        .opportunity-item {
            padding: 12px;
            font-size: 14px;
        }
        
        .inspector-panel {
            padding: 16px;
            min-height: auto;
        }
        
        /* Stack columns on mobile */
        .stColumn {
            width: 100% !important;
        }
        
        /* Make buttons full width on mobile */
        .stButton > button {
            width: 100%;
        }
        
        /* Adjust chart height for mobile */
        .js-plotly-plot {
            height: 400px !important;
        }
    }
    
    /* Extra small devices */
    @media (max-width: 480px) {
        .kpi-value {
            font-size: 24px;
        }
        
        .market-status-card {
            padding: 10px;
            font-size: 12px;
        }
        
        h1 {
            font-size: 24px;
        }
        
        h2 {
            font-size: 20px;
        }
        
        h3 {
            font-size: 18px;
        }
    }
    
    /* Material Design Buttons */
    .stButton > button {
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Material Design Selectbox */
    .stSelectbox > div > div {
        border-radius: 4px;
    }
    
    /* Hide Streamlit branding on desktop */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Show header on mobile for sidebar access */
    @media (max-width: 768px) {
        header {visibility: visible !important; display: block !important;}
        header[data-testid="stHeader"] {
            background: #ffffff !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            z-index: 999 !important;
        }
        /* Ensure hamburger menu is visible and styled */
        button[data-testid="baseButton-header"],
        button[kind="header"],
        [data-testid="stHeader"] button {
            visibility: visible !important;
            display: block !important;
            background: #2196F3 !important;
            color: white !important;
            border-radius: 4px !important;
            padding: 8px 12px !important;
            margin: 8px !important;
        }
        /* Make sidebar accessible */
        section[data-testid="stSidebar"] {
            z-index: 1000 !important;
        }
        /* Add padding to main content to account for fixed header */
        .main .block-container {
            padding-top: 80px !important;
        }
    }
    
    /* Floating menu button for mobile */
    .mobile-menu-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 998;
        display: none;
        background: #2196F3;
        color: white;
        border: none;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        cursor: pointer;
        font-size: 24px;
        transition: all 0.3s ease;
    }
    
    .mobile-menu-button:hover {
        background: #1976D2;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        transform: scale(1.1);
    }
    
    @media (max-width: 768px) {
        .mobile-menu-button {
            display: block;
        }
    }
    
    /* Material Design Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 12px 24px;
    }
    </style>
""", unsafe_allow_html=True)


def calculate_kpis(signals: List[TradeSignal]) -> Dict:
    """Calculate aggregate KPIs from signals."""
    if not signals:
        return {
            "total_opportunities": 0,
            "total_capital_at_risk": 0,
            "total_potential_profit": 0,
            "avg_risk_reward": 0,
            "avg_holding_period": 0,
            "avg_swing_score": 0,
        }
    
    total_risk = 0
    total_profit = 0
    total_rr = 0
    total_days = 0
    total_score = 0
    
    for signal in signals:
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        capital_at_risk = risk_per_share * signal.quantity
        profit_per_share = abs(signal.target - signal.entry_price)
        potential_profit = profit_per_share * signal.quantity
        
        total_risk += capital_at_risk
        total_profit += potential_profit
        total_rr += signal.risk_reward_ratio
        total_days += signal.holding_period_days
        total_score += signal.quality_score
    
    n = len(signals)
    return {
        "total_opportunities": n,
        "total_capital_at_risk": total_risk,
        "total_potential_profit": total_profit,
        "avg_risk_reward": total_rr / n if n > 0 else 0,
        "avg_holding_period": total_days / n if n > 0 else 0,
        "avg_swing_score": total_score / n if n > 0 else 0,
    }


def calculate_market_health_score(intelligence: Dict) -> float:
    """Calculate overall market health score (0-100)."""
    regime = intelligence.get("market_regime", {}).get("regime", "CASH_PROTECTION")
    vix = intelligence.get("volatility_regime", {}).get("vix", 20)
    
    regime_scores = {
        "AGGRESSIVE": 80,
        "DEFENSIVE": 50,
        "CASH_PROTECTION": 20,
    }
    
    base_score = regime_scores.get(regime, 50)
    
    # Adjust for VIX
    if vix < 15:
        vix_adjustment = 10
    elif vix < 20:
        vix_adjustment = 0
    else:
        vix_adjustment = -20
    
    final_score = max(0, min(100, base_score + vix_adjustment))
    return final_score


def render_market_status_bar(intelligence: Dict):
    """Render Material Design market status bar."""
    regime = intelligence.get("market_regime", {}).get("regime", "CASH_PROTECTION")
    vix = intelligence.get("volatility_regime", {}).get("vix", 20)
    nifty_50 = intelligence.get("market_regime", {}).get("nifty_50_change", 0)
    top_sector = intelligence.get("sector_intelligence", {}).get("top_sector", "N/A")
    top_sector_change = intelligence.get("sector_intelligence", {}).get("top_sector_change", 0)
    health_score = calculate_market_health_score(intelligence)
    
    regime_labels = {
        "AGGRESSIVE": ("AGGRESSIVE", "status-aggressive"),
        "DEFENSIVE": ("DEFENSIVE", "status-defensive"),
        "CASH_PROTECTION": ("CASH PROTECTION", "status-cash"),
    }
    
    regime_label, regime_class = regime_labels.get(regime, ("UNKNOWN", ""))
    
    # Responsive columns: 6 columns (will stack on mobile via CSS)
    cols = st.columns(6)
    
    with cols[0]:
        st.markdown(f'''
            <div class="market-status-card">
                <div style="font-size: 12px; color: #757575; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Market Regime</div>
                <div class="{regime_class} status-chip">{regime_label}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with cols[1]:
        nifty_color = "#2E7D32" if nifty_50 >= 0 else "#C62828"
        st.markdown(f'''
            <div class="market-status-card">
                <div style="font-size: 12px; color: #757575; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Nifty 50</div>
                <div style="font-size: 24px; font-weight: 500; color: {nifty_color};">{nifty_50:+.2f}%</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with cols[2]:
        vix_color = "#2E7D32" if vix < 20 else "#C62828" if vix > 25 else "#F57C00"
        st.markdown(f'''
            <div class="market-status-card">
                <div style="font-size: 12px; color: #757575; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">VIX</div>
                <div style="font-size: 24px; font-weight: 500; color: {vix_color};">{vix:.2f}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with cols[3]:
        sector_color = "#2E7D32" if top_sector_change >= 0 else "#C62828"
        st.markdown(f'''
            <div class="market-status-card">
                <div style="font-size: 12px; color: #757575; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Top Sector</div>
                <div style="font-size: 16px; font-weight: 500; color: {sector_color};">{top_sector[:15]} {top_sector_change:+.2f}%</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with cols[4]:
        health_color = "#2E7D32" if health_score >= 70 else "#F57C00" if health_score >= 40 else "#C62828"
        st.markdown(f'''
            <div class="market-status-card">
                <div style="font-size: 12px; color: #757575; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Health Score</div>
                <div style="font-size: 24px; font-weight: 500; color: {health_color};">{health_score:.0f}/100</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with cols[5]:
        st.markdown(f'''
            <div class="market-status-card">
                <div style="font-size: 12px; color: #757575; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Opportunities</div>
                <div style="font-size: 24px; font-weight: 500; color: #1976D2;">{len(st.session_state.signals) if st.session_state.signals else 0}</div>
            </div>
        ''', unsafe_allow_html=True)


def render_kpi_dashboard(signals: List[TradeSignal]):
    """Render Material Design KPI dashboard."""
    kpis = calculate_kpis(signals)
    
    st.markdown("### 📊 Portfolio Overview")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-label">Total Opportunities</div>
                <div class="kpi-value">{kpis["total_opportunities"]}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-label">Capital at Risk</div>
                <div class="kpi-value">₹{kpis["total_capital_at_risk"]:,.0f}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
            <div class="kpi-card" style="border-left-color: #2E7D32;">
                <div class="kpi-label">Potential Profit</div>
                <div class="kpi-value" style="color: #2E7D32;">₹{kpis["total_potential_profit"]:,.0f}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-label">Avg Risk:Reward</div>
                <div class="kpi-value">1:{kpis["avg_risk_reward"]:.2f}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col5:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-label">Avg Holding Period</div>
                <div class="kpi-value">{kpis["avg_holding_period"]:.0f} Days</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col6:
        score_color = "#2E7D32" if kpis["avg_swing_score"] >= 70 else "#F57C00" if kpis["avg_swing_score"] >= 50 else "#C62828"
        st.markdown(f'''
            <div class="kpi-card" style="border-left-color: {score_color};">
                <div class="kpi-label">Avg Swing Score</div>
                <div class="kpi-value" style="color: {score_color};">{kpis["avg_swing_score"]:.1f}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def create_strategy_chart(df: pd.DataFrame, signal: TradeSignal, strategy_name: str) -> go.Figure:
    """Create strategy-specific chart with overlays."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{signal.ticker} - {strategy_name}", "Volume"),
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    
    # Entry, Stop, Targets
    # Convert Timestamp to datetime for Plotly compatibility (pandas 2.0+ issue)
    entry_date = pd.Timestamp(df.index[-1]).to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else df.index[-1]
    try:
        fig.add_vline(
            x=entry_date,
            line_dash="dash",
            line_color="#2196F3",
            annotation_text="Entry",
            row=1,
            col=1,
        )
    except (TypeError, ValueError):
        # Fallback: skip vline if there's a timestamp issue
        pass
    fig.add_hline(y=signal.entry_price, line_dash="dash", line_color="#2E7D32", annotation_text="Entry", row=1, col=1)
    fig.add_hline(y=signal.stop_loss, line_dash="dash", line_color="#C62828", annotation_text="Stop Loss", row=1, col=1)
    fig.add_hline(y=signal.target, line_dash="dash", line_color="#2E7D32", annotation_text="Target", row=1, col=1)
    
    # Strategy-specific overlays
    if "Institutional Wave" in strategy_name:
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["ema_200"] = ta.ema(df["close"], length=200)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df["ema_50"], name="50 EMA", line=dict(color="#FF9800", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["ema_200"], name="200 EMA", line=dict(color="#2196F3", width=2)),
            row=1,
            col=1,
        )
        
        if not df["ema_50"].empty:
            ema_50_value = df["ema_50"].iloc[-1]
            fig.add_hrect(
                y0=ema_50_value * 0.99,
                y1=ema_50_value * 1.01,
                fillcolor="rgba(255, 152, 0, 0.2)",
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )
    
    elif "Golden Pocket" in strategy_name:
        metadata = signal.metadata
        fib_levels = metadata.get("fib_levels", {})
        
        if fib_levels:
            for level_name, level_value in fib_levels.items():
                if level_value > 0:
                    fig.add_hline(
                        y=level_value,
                        line_dash="dash",
                        line_color="#FFC107" if "0_618" in level_name else "#2196F3",
                        annotation_text=level_name.replace("_", " ").title(),
                        row=1,
                        col=1,
                    )
    
    elif "Darvas Box" in strategy_name:
        metadata = signal.metadata
        box_high = metadata.get("resistance_level", df["high"].max())
        box_low = metadata.get("support_level", df["low"].min())
        
        fig.add_hrect(
            y0=box_low,
            y1=box_high,
            fillcolor="rgba(33, 150, 243, 0.2)",
            layer="below",
            line_width=2,
            line_color="#2196F3",
            row=1,
            col=1,
            annotation_text="Darvas Box",
        )
    
    # Volume
    colors = ["#C62828" if df["close"].iloc[i] < df["open"].iloc[i] else "#2E7D32" for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.7),
        row=2,
        col=1,
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def render_inspector_panel(signal: TradeSignal):
    """Render Material Design inspector panel with tabs."""
    if not signal:
        st.info("👈 Select an opportunity from the list to view details")
        return
    
    risk_per_share = abs(signal.entry_price - signal.stop_loss)
    capital_at_risk = risk_per_share * signal.quantity
    profit_per_share = abs(signal.target - signal.entry_price)
    potential_profit = profit_per_share * signal.quantity
    target_1 = signal.metadata.get("target_1", signal.target)
    target_2 = signal.metadata.get("target_2", signal.target)
    sector_perf = signal.metadata.get("sector_performance", {})
    sector_perf_pct = sector_perf.get("change_pct", 0.0) if isinstance(sector_perf, dict) else 0.0
    strategy_group = signal.metadata.get("strategy_group", "N/A")
    holding_period = signal.holding_period_days
    sector = signal.metadata.get("sector", "Unknown")
    thesis = signal.metadata.get("thesis", "No thesis available.")
    
    st.markdown(f'<div class="inspector-panel">', unsafe_allow_html=True)
    
    # Header
    company_name = signal.company_name if signal.company_name else signal.ticker.replace('.NS', '')
    st.markdown(f"### {company_name} ({signal.ticker.replace('.NS', '')})")
    st.markdown(f"**{signal.strategy_name}** | Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if signal.timestamp else 'N/A'}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Trade Details", "💡 Analysis", "📈 Chart"])
    
    with tab1:
        st.markdown("#### Entry & Exit Levels")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">Entry Price</span>
                    <span class="metric-value">₹{signal.entry_price:,.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Stop Loss</span>
                    <span class="metric-value negative">₹{signal.stop_loss:,.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Target 1 (Swing High)</span>
                    <span class="metric-value positive">₹{target_1:,.2f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Target 2 (2x Risk)</span>
                    <span class="metric-value positive">₹{target_2:,.2f}</span>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">Quantity</span>
                    <span class="metric-value">{signal.quantity} shares</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Capital at Risk</span>
                    <span class="metric-value negative">₹{capital_at_risk:,.0f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Potential Profit</span>
                    <span class="metric-value positive">₹{potential_profit:,.0f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Risk:Reward Ratio</span>
                    <span class="metric-value">1:{signal.risk_reward_ratio:.2f}</span>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Trade Metrics")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">Swing Score</span>
                    <span class="metric-value">{signal.quality_score:.1f}/100</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Holding Period</span>
                    <span class="metric-value">{holding_period} Days</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Strategy Group</span>
                    <span class="metric-value">{strategy_group}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">ADX (Trend Strength)</span>
                    <span class="metric-value">{signal.adx_value:.1f}</span>
                </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            quality_tag = signal.metadata.get("quality_tag", "STANDARD")
            confidence = signal.metadata.get("confidence", 0.0) * 100 if isinstance(signal.metadata.get("confidence"), float) else 0.0
            
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">Quality Tag</span>
                    <span class="metric-value">{quality_tag}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Pattern Clarity</span>
                    <span class="metric-value">{confidence:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">RSI (Momentum)</span>
                    <span class="metric-value {'positive' if signal.rsi_value < 70 else 'negative' if signal.rsi_value > 30 else ''}">{signal.rsi_value:.1f}</span>
                </div>
            ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### Investment Thesis")
        st.markdown(f'<div class="thesis-box">{thesis}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Technical Indicators")
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            # ADX interpretation
            adx_status = "Strong Trend" if signal.adx_value >= 25 else "Weak Trend" if signal.adx_value < 20 else "Moderate Trend"
            adx_color = "positive" if signal.adx_value >= 25 else "negative" if signal.adx_value < 20 else ""
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">ADX (Trend Strength)</span>
                    <span class="metric-value {adx_color}">{signal.adx_value:.1f} - {adx_status}</span>
                </div>
            ''', unsafe_allow_html=True)
        
        with col_tech2:
            # RSI interpretation
            rsi_status = "Overbought" if signal.rsi_value >= 70 else "Oversold" if signal.rsi_value <= 30 else "Neutral"
            rsi_color = "negative" if signal.rsi_value >= 70 else "positive" if signal.rsi_value <= 30 else ""
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">RSI (Momentum)</span>
                    <span class="metric-value {rsi_color}">{signal.rsi_value:.1f} - {rsi_status}</span>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Market Context")
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">Sector</span>
                    <span class="metric-value">{sector}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Sector Performance</span>
                    <span class="metric-value {'positive' if sector_perf_pct >= 0 else 'negative'}">{sector_perf_pct:+.2f}%</span>
                </div>
            ''', unsafe_allow_html=True)
        
        with col6:
            sector_status = sector_perf.get("status", "neutral") if isinstance(sector_perf, dict) else "neutral"
            signal_time = signal.timestamp.strftime('%Y-%m-%d %H:%M:%S') if signal.timestamp else 'N/A'
            
            st.markdown(f'''
                <div class="metric-row">
                    <span class="metric-label">Sector Status</span>
                    <span class="metric-value">{sector_status.title()}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Signal Generated</span>
                    <span class="metric-value" style="font-size: 13px;">{signal_time}</span>
                </div>
            ''', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### Price Chart with Strategy Overlays")
        # Use cache for chart (no force refresh needed for display)
        df = st.session_state.data_engine.fetch_single(signal.ticker, period="1y", use_cache=True, force_refresh=False)
        if not df.empty:
            fig = create_strategy_chart(df, signal, signal.strategy_name)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Chart data not available")
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    # Initialize session state
    if "data_engine" not in st.session_state:
        st.session_state.data_engine = FastDataEngine()
    
    if "enrichment" not in st.session_state:
        st.session_state.enrichment = CompanyEnrichment()
    
    if "intelligence" not in st.session_state:
        st.session_state.intelligence = None
    
    if "signals" not in st.session_state:
        st.session_state.signals = []
    
    if "selected_signal" not in st.session_state:
        st.session_state.selected_signal = None
    
    if "force_refresh" not in st.session_state:
        st.session_state.force_refresh = False
    
    if "strategy_runner" not in st.session_state:
        trade_assistant = TradeAssistant(account_equity=config.ACCOUNT_EQUITY, data_engine=st.session_state.data_engine)
        st.session_state.strategy_runner = StrategyRunner(
            data_engine=st.session_state.data_engine,
            trade_assistant=trade_assistant
        )
    
    # Get last update date from database
    last_update_date = st.session_state.data_engine.get_last_update_date()
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Swing Trading Dashboard")
        st.markdown("---")
        
        # Data Freshness Info
        st.subheader("📊 Data Status")
        if last_update_date:
            from datetime import datetime, timedelta
            age = datetime.now() - last_update_date
            hours_ago = age.total_seconds() / 3600
            
            if hours_ago < 1:
                freshness_status = "🟢 Fresh"
                freshness_color = "#2E7D32"
                age_text = f"{int(age.total_seconds() / 60)} minutes ago"
            elif hours_ago < 24:
                freshness_status = "🟡 Recent"
                freshness_color = "#F57C00"
                age_text = f"{int(hours_ago)} hours ago"
            else:
                freshness_status = "🔴 Stale"
                freshness_color = "#C62828"
                days_ago = int(hours_ago / 24)
                age_text = f"{days_ago} day{'s' if days_ago > 1 else ''} ago"
            
            st.markdown(f"""
                <div style="background: #ffffff; padding: 12px; border-radius: 4px; border-left: 4px solid {freshness_color}; margin-bottom: 12px;">
                    <div style="font-size: 12px; color: #757575; margin-bottom: 4px;">Last Data Update</div>
                    <div style="font-size: 14px; font-weight: 500; color: #212121; margin-bottom: 4px;">
                        {last_update_date.strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    <div style="font-size: 12px; color: {freshness_color}; font-weight: 500;">
                        {freshness_status} • {age_text}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: #FFF3E0; padding: 12px; border-radius: 4px; border-left: 4px solid #F57C00; margin-bottom: 12px;">
                    <div style="font-size: 12px; color: #757575; margin-bottom: 4px;">Last Data Update</div>
                    <div style="font-size: 14px; font-weight: 500; color: #212121;">
                        No data in database yet
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("⚙️ Configuration")
        regime = st.selectbox(
            "Market Regime",
            ["AGGRESSIVE", "DEFENSIVE", "CASH_PROTECTION"],
            index=0,
        )
        
        st.markdown("---")
        
        # Hard Refresh button
        if st.button("🔄 Hard Refresh Data", use_container_width=True, help="Force fetch fresh data from yfinance (ignores cache)"):
            st.session_state.force_refresh = True
            st.session_state.signals = []  # Clear existing signals
            st.session_state.intelligence = None  # Clear intelligence
            st.success("🔄 Hard refresh enabled! Click 'Scan Nifty 500' to fetch fresh data.")
        
        st.markdown("---")
        
        # Auto-load data if not already loaded
        should_auto_load = (
            "signals" not in st.session_state or 
            len(st.session_state.signals) == 0 or 
            st.session_state.intelligence is None
        )
        
        scan_button_label = "🔍 Scan Nifty 500" if not should_auto_load else "🔍 Scan Nifty 500 (Auto-load from DB)"
        
        if st.button(scan_button_label, type="primary", use_container_width=True):
            with st.spinner("Analyzing market..."):
                # Get complete market intelligence (regime + volatility + sectors)
                cognition = CognitionEngine(st.session_state.data_engine)
                intelligence = cognition.get_all_intelligence(force_refresh=st.session_state.force_refresh)
                st.session_state.intelligence = intelligence
                
                # Run strategies with parallel data fetching
                all_tickets = []
                ticker_list = st.session_state.data_engine.get_nifty500_tickers()
                
                # Step 1: Fetch all data in parallel
                status_text = st.empty()
                if st.session_state.force_refresh:
                    status_text.text("🔄 Hard Refresh: Fetching fresh data from yfinance...")
                else:
                    status_text.text("📥 Loading data from cache (use Hard Refresh for fresh data)...")
                progress_bar = st.progress(0)
                
                def update_progress(completed, total):
                    progress_bar.progress(completed / total)
                    if st.session_state.force_refresh:
                        status_text.text(f"🔄 Hard Refresh: Fetching fresh data... ({completed}/{total} tickers)")
                    else:
                        status_text.text(f"📥 Loading data from cache... ({completed}/{total} tickers)")
                
                ticker_data = st.session_state.data_engine.fetch_batch_data(
                    ticker_list,
                    period="1y",
                    use_cache=True,
                    force_refresh=st.session_state.force_refresh,
                    progress_callback=update_progress
                )
                
                # Reset force_refresh after use
                if st.session_state.force_refresh:
                    st.session_state.force_refresh = False
                
                # Step 2: Scan all tickers with strategies (with caching)
                status_text.text("🔍 Scanning with strategies...")
                progress_bar.progress(0)
                
                # Initialize engines
                calibration = CalibrationEngine(st.session_state.strategy_runner)
                trade_assistant = TradeAssistant(account_equity=config.ACCOUNT_EQUITY, data_engine=st.session_state.data_engine)
                analyst = IntelligenceEngine(st.session_state.data_engine, trade_assistant)
                enrichment = st.session_state.enrichment
                volatility_regime = intelligence.get("volatility_regime", {})
                
                all_signals = []
                cached_count = 0
                computed_count = 0
                
                for i, (ticker, df) in enumerate(ticker_data.items()):
                    if df.empty:
                        progress_bar.progress((i + 1) / len(ticker_data))
                        continue
                    
                    # Calculate data hash to check if we have valid cache
                    data_hash = st.session_state.data_engine._calculate_data_hash(ticker, df)
                    
                    # Try to load from cache
                    cached_signals = st.session_state.data_engine.load_strategy_results(
                        ticker, regime, data_hash, max_age_hours=24
                    )
                    
                    if cached_signals and not st.session_state.force_refresh:
                        # Load from cache - convert dicts back to TradeSignal objects
                        from src.models.schemas import TradeSignal
                        ticker_signals = [TradeSignal(**sig_dict) for sig_dict in cached_signals]
                        all_signals.extend(ticker_signals)
                        cached_count += 1
                    else:
                        # Compute strategies
                        tickets = st.session_state.strategy_runner.scan_ticker(ticker, df, regime=regime)
                        
                        if tickets:
                            # Calibrate
                            calibrated_tickets = calibration.calibrate(tickets, regime=regime)
                            
                            # Convert to signals
                            ticker_signals = []
                            for ticket in calibrated_tickets:
                                # Get sector information
                                sector = enrichment.get_sector_fast(ticket.ticker) or "Unknown"
                                sector_perf = enrichment.get_sector_performance(sector, st.session_state.data_engine) if sector != "Unknown" else {"change_pct": 0.0, "status": "neutral"}
                                ticket.metadata["sector"] = sector
                                ticket.metadata["sector_performance"] = sector_perf
                                
                                # Calculate swing score
                                swing_score = analyst.calculate_swing_score(ticket, sector_perf)
                                
                                # Generate thesis
                                thesis = analyst.generate_investment_thesis(
                                    ticket, sector, sector_perf, volatility_regime
                                )
                                ticket.metadata["thesis"] = thesis
                                
                                # Analyze signal and create trade signal
                                signal = analyst.analyze_signal(ticket, run_backtest=False)
                                if signal:
                                    # Update quality score with calculated swing score
                                    signal.quality_score = swing_score
                                    ticker_signals.append(signal)
                            
                            # Save to cache
                            if ticker_signals:
                                # Convert signals to dict with datetime serialization
                                signals_dict = [serialize_signal_for_json(sig) for sig in ticker_signals]
                                st.session_state.data_engine.save_strategy_results(
                                    ticker, regime, signals_dict, data_hash
                                )
                                all_signals.extend(ticker_signals)
                                computed_count += 1
                    
                    progress_bar.progress((i + 1) / len(ticker_data))
                
                signals = all_signals
                
                # Log cache statistics
                if cached_count > 0 or computed_count > 0:
                    logger.info(f"Strategy scan: {cached_count} from cache, {computed_count} computed")
                
                st.session_state.signals = signals
                st.success(f"✅ Found {len(signals)} opportunities!")
    
    # Auto-load data from database on first load if data exists and not already loaded
    if (
        last_update_date and 
        ("signals" not in st.session_state or len(st.session_state.signals) == 0) and
        st.session_state.intelligence is None and
        "auto_load_attempted" not in st.session_state
    ):
        st.session_state.auto_load_attempted = True
        with st.spinner("🔄 Auto-loading data from database..."):
            # Get complete market intelligence (regime + volatility + sectors)
            cognition = CognitionEngine(st.session_state.data_engine)
            intelligence = cognition.get_all_intelligence(force_refresh=False)
            st.session_state.intelligence = intelligence
            
            # Run strategies with parallel data fetching
            all_tickets = []
            ticker_list = st.session_state.data_engine.get_nifty500_tickers()
            
            # Fetch all data in parallel from cache
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("📥 Loading data from database...")
            
            def update_progress(completed, total):
                progress_bar.progress(completed / total)
                status_text.text(f"📥 Loading data from database... ({completed}/{total} tickers)")
            
            ticker_data = st.session_state.data_engine.fetch_batch_data(
                ticker_list,
                period="1y",
                use_cache=True,
                force_refresh=False,  # Always use cache for auto-load
                progress_callback=update_progress
            )
            
            # Scan all tickers with strategies (with caching)
            status_text.text("🔍 Scanning with strategies...")
            progress_bar.progress(0)
            
            # Initialize engines
            calibration = CalibrationEngine(st.session_state.strategy_runner)
            trade_assistant = TradeAssistant(account_equity=config.ACCOUNT_EQUITY, data_engine=st.session_state.data_engine)
            analyst = IntelligenceEngine(st.session_state.data_engine, trade_assistant)
            enrichment = st.session_state.enrichment
            volatility_regime = intelligence.get("volatility_regime", {})
            
            all_signals = []
            
            for i, (ticker, df) in enumerate(ticker_data.items()):
                if df.empty:
                    progress_bar.progress((i + 1) / len(ticker_data))
                    continue
                
                # Calculate data hash to check if we have valid cache
                data_hash = st.session_state.data_engine._calculate_data_hash(ticker, df)
                
                # Try to load from cache
                cached_signals = st.session_state.data_engine.load_strategy_results(
                    ticker, regime, data_hash, max_age_hours=24
                )
                
                if cached_signals:
                    # Load from cache - convert dicts back to TradeSignal objects
                    from src.models.schemas import TradeSignal
                    ticker_signals = [TradeSignal(**sig_dict) for sig_dict in cached_signals]
                    all_signals.extend(ticker_signals)
                else:
                    # Compute strategies
                    tickets = st.session_state.strategy_runner.scan_ticker(ticker, df, regime=regime)
                    
                    if tickets:
                        # Calibrate
                        calibrated_tickets = calibration.calibrate(tickets, regime=regime)
                        
                        # Convert to signals
                        ticker_signals = []
                        for ticket in calibrated_tickets:
                            # Get sector information
                            sector = enrichment.get_sector_fast(ticket.ticker) or "Unknown"
                            sector_perf = enrichment.get_sector_performance(sector, st.session_state.data_engine) if sector != "Unknown" else {"change_pct": 0.0, "status": "neutral"}
                            ticket.metadata["sector"] = sector
                            ticket.metadata["sector_performance"] = sector_perf
                            
                            # Calculate swing score
                            swing_score = analyst.calculate_swing_score(ticket, sector_perf)
                            
                            # Generate thesis
                            thesis = analyst.generate_investment_thesis(
                                ticket, sector, sector_perf, volatility_regime
                            )
                            ticket.metadata["thesis"] = thesis
                            
                            # Analyze signal (includes backtest and signal creation)
                            signal = analyst.analyze_signal(ticket, run_backtest=True)
                            if signal:
                                # Update quality score with calculated swing score
                                signal.quality_score = swing_score
                                ticker_signals.append(signal)
                        
                        # Save to cache
                        if ticker_signals:
                            # Convert signals to dict with datetime serialization
                            signals_dict = [serialize_signal_for_json(sig) for sig in ticker_signals]
                            st.session_state.data_engine.save_strategy_results(
                                ticker, regime, signals_dict, data_hash
                            )
                            all_signals.extend(ticker_signals)
                
                progress_bar.progress((i + 1) / len(ticker_data))
            
            st.session_state.signals = all_signals
            status_text.empty()
            progress_bar.empty()
            st.rerun()  # Rerun to display the loaded data
    
    # Mobile menu button (floating button for easy sidebar access on mobile)
    st.markdown("""
        <button class="mobile-menu-button" id="mobileMenuBtn" onclick="openMobileSidebar()">
            ☰
        </button>
        <script>
            function openMobileSidebar() {
                // Try multiple methods to open sidebar
                const sidebarToggle = document.querySelector('button[kind="header"]') || 
                                     document.querySelector('[data-testid="baseButton-header"]') ||
                                     document.querySelector('button[aria-label*="menu" i]') ||
                                     document.querySelector('[data-testid="stHeader"] button');
                if (sidebarToggle) {
                    sidebarToggle.click();
                } else {
                    // Fallback: show sidebar directly via CSS
                    const sidebar = document.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {
                        sidebar.style.visibility = 'visible';
                        sidebar.style.display = 'block';
                        sidebar.style.transform = 'translateX(0)';
                    }
                }
            }
        </script>
    """, unsafe_allow_html=True)
    
    # Main content
    st.title("📈 Medium-Term Swing Trading Dashboard")
    st.markdown("Nifty 500 Analysis | 10-20 Day Holding Period")
    
    # Display last update date in main area
    if last_update_date:
        from datetime import datetime
        age = datetime.now() - last_update_date
        hours_ago = age.total_seconds() / 3600
        
        if hours_ago < 1:
            freshness_badge = "🟢 Fresh"
            freshness_color = "#2E7D32"
        elif hours_ago < 24:
            freshness_badge = "🟡 Recent"
            freshness_color = "#F57C00"
        else:
            freshness_badge = "🔴 Stale"
            freshness_color = "#C62828"
        
        st.markdown(f"""
            <div style="background: #E3F2FD; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; border-left: 4px solid #2196F3;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 13px; color: #757575;">Last Data Retrieval:</span>
                        <span style="font-size: 14px; font-weight: 500; color: #212121; margin-left: 8px;">
                            {last_update_date.strftime('%Y-%m-%d %H:%M:%S')}
                        </span>
                    </div>
                    <div style="font-size: 13px; font-weight: 500; color: {freshness_color};">
                        {freshness_badge}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Market Status Bar
    if st.session_state.intelligence:
        render_market_status_bar(st.session_state.intelligence)
    
    # KPI Dashboard
    if st.session_state.signals:
        render_kpi_dashboard(st.session_state.signals)
    
    # Main Content Area
    if st.session_state.signals:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("### 📋 Opportunities")
            
            # Strategy filter
            all_strategies = list(set([s.strategy_name for s in st.session_state.signals]))
            selected_strategy = st.selectbox("Filter by Strategy", ["All"] + all_strategies)
            
            filtered_signals = st.session_state.signals
            if selected_strategy != "All":
                filtered_signals = [s for s in st.session_state.signals if s.strategy_name == selected_strategy]
            
            # Sort options
            sort_by = st.selectbox("Sort by", ["Score (High to Low)", "Risk:Reward (High to Low)", "Potential Profit (High to Low)"])
            if sort_by == "Score (High to Low)":
                filtered_signals = sorted(filtered_signals, key=lambda x: x.quality_score, reverse=True)
            elif sort_by == "Risk:Reward (High to Low)":
                filtered_signals = sorted(filtered_signals, key=lambda x: x.risk_reward_ratio, reverse=True)
            elif sort_by == "Potential Profit (High to Low)":
                filtered_signals = sorted(filtered_signals, key=lambda x: abs(x.target - x.entry_price) * x.quantity, reverse=True)
            
            # Display opportunities
            for i, signal in enumerate(filtered_signals):
                is_selected = (st.session_state.selected_signal and st.session_state.selected_signal.ticker == signal.ticker)
                
                risk = abs(signal.entry_price - signal.stop_loss) * signal.quantity
                profit = abs(signal.target - signal.entry_price) * signal.quantity
                
                company_name = signal.company_name if signal.company_name else signal.ticker.replace('.NS', '')
                if st.button(
                    f"{company_name[:30]} - {signal.strategy_name}",
                    key=f"opp_{i}",
                    use_container_width=True,
                ):
                    st.session_state.selected_signal = signal
                    st.rerun()
                
                # Show key info in Material Design card
                card_class = "opportunity-item selected" if is_selected else "opportunity-item"
                company_name = signal.company_name if signal.company_name else signal.ticker.replace('.NS', '')
                st.markdown(f'''
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div>
                                <strong style="font-size: 16px; color: #212121;">{company_name[:25]}</strong>
                                <span style="font-size: 12px; color: #757575; margin-left: 8px;">({signal.ticker.replace('.NS', '')})</span>
                                <span class="material-badge badge-primary">{signal.strategy_name[:20]}</span>
                            </div>
                            <span class="material-badge {'badge-success' if signal.quality_score >= 70 else 'badge-warning' if signal.quality_score >= 50 else 'badge-danger'}">{signal.quality_score:.0f}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; font-size: 13px; color: #757575;">
                            <div><strong>Entry:</strong> ₹{signal.entry_price:,.2f}</div>
                            <div><strong>Risk:</strong> ₹{risk:,.0f}</div>
                            <div><strong>Profit:</strong> ₹{profit:,.0f}</div>
                        </div>
                        <div style="margin-top: 8px; font-size: 12px; color: #757575;">
                            R:R 1:{signal.risk_reward_ratio:.2f} | {signal.holding_period_days} Days | {signal.metadata.get("sector", "Unknown")[:15]} | ADX: {signal.adx_value:.1f} | RSI: {signal.rsi_value:.1f}
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
        
        with col_right:
            render_inspector_panel(st.session_state.selected_signal)
    else:
        st.info("👆 Click 'Scan Nifty 500' in the sidebar to find opportunities!")


if __name__ == "__main__":
    main()
