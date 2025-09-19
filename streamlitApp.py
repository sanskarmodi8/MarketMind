import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from MarketMind.pipeline.prediction import prediction_pipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    st.warning("⚠️ MarketMind pipeline not available. Running in demo mode.")

# Page config
st.set_page_config(
    page_title="MarketMind - AI Bitcoin Trading Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with modern glassmorphism design
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #64748b;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .prediction-card-buy {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.9) 0%, rgba(22, 163, 74, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 20px 40px rgba(34, 197, 94, 0.3);
        animation: pulse-green 2s infinite;
    }
    
    .prediction-card-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.9) 0%, rgba(220, 38, 38, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 20px 40px rgba(239, 68, 68, 0.3);
        animation: pulse-red 2s infinite;
    }
    
    .prediction-card-hold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.9) 0%, rgba(217, 119, 6, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 20px 40px rgba(245, 158, 11, 0.3);
        animation: pulse-yellow 2s infinite;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 20px 40px rgba(34, 197, 94, 0.3); }
        50% { box-shadow: 0 25px 50px rgba(34, 197, 94, 0.5); }
        100% { box-shadow: 0 20px 40px rgba(34, 197, 94, 0.3); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 20px 40px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 25px 50px rgba(239, 68, 68, 0.5); }
        100% { box-shadow: 0 20px 40px rgba(239, 68, 68, 0.3); }
    }
    
    @keyframes pulse-yellow {
        0% { box-shadow: 0 20px 40px rgba(245, 158, 11, 0.3); }
        50% { box-shadow: 0 25px 50px rgba(245, 158, 11, 0.5); }
        100% { box-shadow: 0 20px 40px rgba(245, 158, 11, 0.3); }
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-online {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        padding: 3rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "live_data" not in st.session_state:
    st.session_state.live_data = None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_bitcoin_data(period="180d"):
    """Fetch real Bitcoin data from Yahoo Finance."""
    try:
        btc = yf.Ticker("BTC-USD")
        data = btc.history(period=period)
        info = btc.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None


def load_model_once():
    """Load the model once and cache in session state."""
    if not PIPELINE_AVAILABLE:
        return False

    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI model... This may take a moment."):
            try:
                success = prediction_pipeline.load_model()
                st.session_state.model_loaded = success
            except Exception as e:
                st.error(f"❌ Model loading error: {str(e)}")
                return False
    return st.session_state.model_loaded


def display_prediction_card(prediction_result):
    """Display modern prediction card with enhanced styling."""
    if not prediction_result["success"]:
        st.error(f"🚨 Error: {prediction_result.get('error', 'Unknown error')}")
        return

    prediction = prediction_result["prediction"]
    current_price = prediction_result["current_price"]
    price_change = prediction_result["price_change_24h"]

    # Choose styling based on prediction
    if prediction == "BUY":
        card_class = "prediction-card-buy"
        emoji = "🚀"
        action_text = "STRONG BUY SIGNAL"
    elif prediction == "SELL":
        card_class = "prediction-card-sell"
        emoji = "📉"
        action_text = "SELL RECOMMENDATION"
    else:
        card_class = "prediction-card-hold"
        emoji = "⏸️"
        action_text = "HOLD POSITION"

    change_color = "🟢" if price_change >= 0 else "🔴"

    st.markdown(
        f"""
    <div class="{card_class}">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <h1 style="margin: 0; font-size: 4rem;">{emoji}</h1>
        </div>
        <h2 style="margin: 0; font-size: 1.8rem; margin-bottom: 0.5rem;">Bitcoin (BTC-USD)</h2>
        <h1 style="margin: 0; font-size: 2.5rem; margin-bottom: 1rem;">{action_text}</h1>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">
            <div>
                <h3 style="margin: 0; font-size: 1rem; opacity: 0.8;">Current Price</h3>
                <h2 style="margin: 0; font-size: 2rem;">${current_price:,.2f}</h2>
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1rem; opacity: 0.8;">24h Change</h3>
                <h2 style="margin: 0; font-size: 2rem;">{change_color} {price_change:+.2f}%</h2>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_performance_metrics(data):
    """Display performance metrics in a grid."""
    if data is None or data.empty:
        return

    current_price = data["Close"].iloc[-1]
    week_ago_price = data["Close"].iloc[-7] if len(data) > 7 else data["Close"].iloc[0]
    month_ago_price = (
        data["Close"].iloc[-30] if len(data) > 30 else data["Close"].iloc[0]
    )

    week_change = ((current_price - week_ago_price) / week_ago_price) * 100
    month_change = ((current_price - month_ago_price) / month_ago_price) * 100

    volatility = (
        data["Close"].pct_change().std() * np.sqrt(252) * 100
    )  # Annualized volatility
    volume_avg = data["Volume"].tail(7).mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="color: #64748b; margin: 0;">7-Day Performance</h3>
            <h2 style="color: {'#10b981' if week_change >= 0 else '#ef4444'}; margin: 0.5rem 0;">{week_change:+.2f}%</h2>
            <p style="margin: 0; font-size: 0.9rem;">vs last week</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="color: #64748b; margin: 0;">30-Day Performance</h3>
            <h2 style="color: {'#10b981' if month_change >= 0 else '#ef4444'}; margin: 0.5rem 0;">{month_change:+.2f}%</h2>
            <p style="margin: 0; font-size: 0.9rem;">vs last month</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="color: #64748b; margin: 0;">Volatility</h3>
            <h2 style="color: #f59e0b; margin: 0.5rem 0;">{volatility:.1f}%</h2>
            <p style="margin: 0; font-size: 0.9rem;">annualized</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="color: #64748b; margin: 0;">Avg Volume</h3>
            <h2 style="color: #8b5cf6; margin: 0.5rem 0;">{volume_avg/1e9:.1f}B</h2>
            <p style="margin: 0; font-size: 0.9rem;">7-day average</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# Main App Layout
def main():
    # Header
    st.markdown('<h1 class="main-header">MarketMind 🧠</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-Powered Bitcoin Trading Agent with Reinforcement Learning</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown("### 🤖 AI Trading Agent")

    # Model status
    model_loaded = load_model_once()
    status_badge = "status-online" if model_loaded else "status-offline"
    status_text = "AI Model: Ready" if model_loaded else "AI Model: Not Available"
    status_emoji = "🟢" if model_loaded else "🔴"

    st.sidebar.markdown(
        f'<span class="status-badge {status_badge}">{status_emoji} {status_text}</span>',
        unsafe_allow_html=True,
    )

    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    data_period = st.sidebar.selectbox(
        "Historical data period:", ["180d", "1y", "2y"], index=2
    )

    # Prediction button
    predict_btn = st.sidebar.button(
        "🔮 Get AI Prediction",
        type="primary",
        disabled=not model_loaded,
        use_container_width=True,
    )

    # Fetch live data
    with st.spinner("📡 Fetching live Bitcoin data..."):
        data, info = fetch_bitcoin_data(data_period)
        st.session_state.live_data = data

    if data is not None and not data.empty:
        # Current market status
        current_price = data["Close"].iloc[-1]
        prev_price = data["Close"].iloc[-2] if len(data) > 1 else current_price
        price_change_24h = ((current_price - prev_price) / prev_price) * 100

        # Market status indicator
        market_status = (
            "🟢 Market Open" if datetime.now().weekday() < 5 else "🔴 Market Closed"
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### 📈 Market Status")
        st.sidebar.markdown(f"**{market_status}**")
        st.sidebar.markdown(f"**Current Price:** ${current_price:,.2f}")
        st.sidebar.markdown(f"**24h Change:** {price_change_24h:+.2f}%")

        # Handle prediction request
        if predict_btn:
            with st.spinner("🤖 AI Agent analyzing market conditions..."):
                if PIPELINE_AVAILABLE:
                    try:
                        result = prediction_pipeline.get_prediction(
                            "BTC-USD", int(data_period[:-1])
                        )
                        result["timestamp_local"] = datetime.now()
                        st.session_state.prediction_history.append(result)
                        st.session_state["last_result"] = result
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.session_state["last_result"] = {
                            "success": False,
                            "error": str(e),
                        }
                else:
                    # Demo prediction for when pipeline is not available
                    demo_result = {
                        "success": True,
                        "prediction": np.random.choice(
                            ["BUY", "SELL", "HOLD"], p=[0.4, 0.3, 0.3]
                        ),
                        "current_price": current_price,
                        "price_change_24h": price_change_24h,
                        "technical_indicators": {
                            "sma_short": current_price * np.random.uniform(0.98, 1.02),
                            "sma_long": current_price * np.random.uniform(0.95, 1.05),
                            "volatility": np.random.uniform(0.02, 0.08),
                            "trend": np.random.choice(
                                ["Bullish", "Bearish", "Neutral"]
                            ),
                        },
                        "data_points_used": len(data),
                        "timestamp_local": datetime.now(),
                    }
                    st.session_state.prediction_history.append(demo_result)
                    st.session_state["last_result"] = demo_result
                    st.info("🔄 Running in demo mode - this is a simulated prediction")

        # Main content area
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            if result["success"]:
                # Display prediction
                display_prediction_card(result)

                # Performance metrics
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Market Performance")
                st.markdown("<br>", unsafe_allow_html=True)
                display_performance_metrics(data)

                # Technical analysis
                if result.get("technical_indicators"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Technical Analysis")
                    st.markdown("<br>", unsafe_allow_html=True)
                    tech_indicators = result["technical_indicators"]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Short SMA", f"${tech_indicators['sma_short']:.2f}")
                    with col2:
                        st.metric("Long SMA", f"${tech_indicators['sma_long']:.2f}")
                    with col3:
                        st.metric("Volatility", f"{tech_indicators['volatility']:.4f}")
                    with col4:
                        st.metric("Trend", tech_indicators["trend"])

        # Simple price chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="BTC-USD",
                line=dict(color="#667eea", width=3),
            )
        )
        fig.update_layout(
            title="Bitcoin Price Movement",
            template="plotly_dark",
            height=400,
            xaxis_title="Date",
            yaxis_title="Price ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prediction history
        if st.session_state.prediction_history:
            st.markdown("### 🕐 Recent AI Predictions")

            # Create history dataframe
            history_data = []
            for pred in reversed(
                st.session_state.prediction_history[-10:]
            ):  # Last 10 predictions
                emoji = (
                    "🚀"
                    if pred["prediction"] == "BUY"
                    else "📉" if pred["prediction"] == "SELL" else "⏸️"
                )
                history_data.append(
                    {
                        "Timestamp": pred["timestamp_local"].strftime("%m/%d %H:%M"),
                        "Signal": f"{emoji} {pred['prediction']}",
                        "Price": f"${pred['current_price']:.2f}",
                        "24h Change": f"{pred['price_change_24h']:+.2f}%",
                        "Trend": (
                            pred["technical_indicators"]["trend"]
                            if pred.get("technical_indicators")
                            else "N/A"
                        ),
                    }
                )

            if history_data:
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True, hide_index=True)

    else:
        st.error(
            "❌ Unable to fetch Bitcoin data. Please check your internet connection."
        )

    # Footer
    st.markdown(
        """
    <div class="footer">
        <h3>MarketMind - AI-Powered Bitcoin Trading Agent</h3>
        <p>Powered by Reinforcement Learning & Advanced Technical Analysis</p>
        <p>⚠️ <strong>Disclaimer:</strong> This is for educational and research purposes only. 
        Always conduct your own research and consult with financial advisors before making investment decisions.</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
            Built with Streamlit • Powered by PPO Algorithm • Real-time Yahoo Finance Data
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
