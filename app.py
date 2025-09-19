import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from MarketMind.config.configuration import ConfigurationManager

warnings.filterwarnings("ignore")

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from MarketMind.pipeline.prediction import prediction_pipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="MarketMind - AI Bitcoin Trading Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - Modern design
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
    
    .prediction-card-error {
        background: linear-gradient(135deg, rgba(107, 114, 128, 0.9) 0%, rgba(75, 85, 99, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 20px 40px rgba(107, 114, 128, 0.3);
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
    
    .warning-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        padding: 3rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Session state initialization
def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "last_data_fetch_time" not in st.session_state:
        st.session_state.last_data_fetch_time = None
    if "cached_data" not in st.session_state:
        st.session_state.cached_data = {}


init_session_state()


@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def fetch_bitcoin_data_robust(
    period: str,
) -> Tuple[Optional[pd.DataFrame], Optional[dict], str]:
    """
    Robust Bitcoin data fetching with retry logic and proper error handling.
    Returns: (data, info, status_message)
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period=period)

            if data is None or data.empty:
                raise Exception(f"No data returned for period {period}")

            # Try to get info (optional)
            try:
                info = btc.info
            except:
                info = {"symbol": "BTC-USD", "shortName": "Bitcoin USD"}

            return data, info, f"Success: Fetched {len(data)} data points"

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                return None, None, f"Error after {max_retries} attempts: {error_msg}"

    return None, None, "Failed to fetch data"


def convert_period_to_days(period: str) -> int:
    """Convert period string to days."""
    if period.endswith("d"):
        return int(period[:-1])
    elif period.endswith("y"):
        return int(period[:-1]) * 365
    elif period == "6mo":
        return 180
    elif period == "3mo":
        return 90
    elif period == "1mo":
        return 30
    else:
        return 180  # Default


def load_model_with_status():
    """Load model with proper status handling."""
    if not PIPELINE_AVAILABLE:
        return False, "Pipeline not available - running in demo mode"

    if not st.session_state.model_loaded:
        try:
            with st.spinner("Loading AI model... This may take a moment."):
                success = prediction_pipeline.load_model()
                st.session_state.model_loaded = success

                if success:
                    return True, "Model loaded successfully"
                else:
                    return False, "Model failed to load"
        except Exception as e:
            return False, f"Model loading error: {str(e)}"

    return True, "Model already loaded"


def display_prediction_card(prediction_result: dict):
    """Display prediction card with proper error handling."""
    if not prediction_result.get("success", False):
        error_message = prediction_result.get("error", "Unknown error")
        st.markdown(
            f"""
            <div class="prediction-card-error">
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                    <h1 style="margin: 0; font-size: 4rem;">⚠️</h1>
                </div>
                <h2 style="margin: 0; font-size: 1.8rem; margin-bottom: 0.5rem;">Bitcoin (BTC-USD)</h2>
                <h1 style="margin: 0; font-size: 2rem; margin-bottom: 1rem;">PREDICTION ERROR</h1>
                <div style="margin-top: 1rem;">
                    <p style="margin: 0; font-size: 1rem; opacity: 0.9;">{error_message}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    prediction = prediction_result.get("prediction", "UNKNOWN")
    current_price = prediction_result.get("current_price", 0)
    price_change = prediction_result.get("price_change_24h", 0)

    # Choose styling based on prediction
    card_configs = {
        "BUY": {
            "class": "prediction-card-buy",
            "emoji": "🚀",
            "text": "STRONG BUY SIGNAL",
        },
        "SELL": {
            "class": "prediction-card-sell",
            "emoji": "📉",
            "text": "SELL RECOMMENDATION",
        },
        "HOLD": {
            "class": "prediction-card-hold",
            "emoji": "⏸️",
            "text": "HOLD POSITION",
        },
    }

    config = card_configs.get(prediction, card_configs["HOLD"])
    change_color = "🟢" if price_change >= 0 else "🔴"

    st.markdown(
        f"""
        <div class="{config['class']}">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <h1 style="margin: 0; font-size: 4rem;">{config['emoji']}</h1>
            </div>
            <h2 style="margin: 0; font-size: 1.8rem; margin-bottom: 0.5rem;">Bitcoin (BTC-USD)</h2>
            <h1 style="margin: 0; font-size: 2.5rem; margin-bottom: 1rem;">{config['text']}</h1>
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


def display_performance_metrics(data: pd.DataFrame):
    """Display performance metrics with error handling."""
    if data is None or data.empty:
        st.warning("No data available for performance metrics")
        return

    try:
        current_price = data["Close"].iloc[-1]
        week_ago_price = (
            data["Close"].iloc[-7] if len(data) > 7 else data["Close"].iloc[0]
        )
        month_ago_price = (
            data["Close"].iloc[-30] if len(data) > 30 else data["Close"].iloc[0]
        )

        week_change = ((current_price - week_ago_price) / week_ago_price) * 100
        month_change = ((current_price - month_ago_price) / month_ago_price) * 100

        volatility = data["Close"].pct_change().std() * np.sqrt(252) * 100  # Annualized
        volume_avg = data["Volume"].tail(7).mean()

        col1, col2, col3, col4 = st.columns(4)

        metrics_data = [
            (
                "7-Day Performance",
                f"{week_change:+.2f}%",
                "#10b981" if week_change >= 0 else "#ef4444",
                "vs last week",
            ),
            (
                "30-Day Performance",
                f"{month_change:+.2f}%",
                "#10b981" if month_change >= 0 else "#ef4444",
                "vs last month",
            ),
            ("Volatility", f"{volatility:.1f}%", "#f59e0b", "annualized"),
            ("Avg Volume", f"{volume_avg/1e9:.1f}B", "#8b5cf6", "7-day average"),
        ]

        for col, (title, value, color, subtitle) in zip(
            [col1, col2, col3, col4], metrics_data
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3 style="color: #64748b; margin: 0;">{title}</h3>
                        <h2 style="color: {color}; margin: 0.5rem 0;">{value}</h2>
                        <p style="margin: 0; font-size: 0.9rem;">{subtitle}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")


def create_price_chart(data: pd.DataFrame) -> go.Figure:
    """Create price chart with error handling."""
    try:
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
            showlegend=False,
        )
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None


def display_prediction_history():
    """Display prediction history with proper filtering."""
    if not st.session_state.prediction_history:
        return

    st.markdown("### 🕐 Recent AI Predictions")

    # Filter successful predictions for main table
    successful_predictions = [
        pred
        for pred in st.session_state.prediction_history
        if pred.get("success", False)
    ]

    if successful_predictions:
        history_data = []
        for pred in reversed(successful_predictions[-10:]):  # Last 10 successful
            prediction = pred.get("prediction", "UNKNOWN")
            emoji_map = {"BUY": "🚀", "SELL": "📉", "HOLD": "⏸️"}
            emoji = emoji_map.get(prediction, "❓")

            history_data.append(
                {
                    "Timestamp": pred.get("timestamp_local", datetime.now()).strftime(
                        "%m/%d %H:%M"
                    ),
                    "Signal": f"{emoji} {prediction}",
                    "Price": f"${pred.get('current_price', 0):.2f}",
                    "24h Change": f"{pred.get('price_change_24h', 0):+.2f}%",
                    "Trend": pred.get("technical_indicators", {}).get("trend", "N/A"),
                }
            )

        if history_data:
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.info("No successful predictions in history yet.")

    # Show failed predictions in expandable section
    failed_predictions = [
        pred
        for pred in st.session_state.prediction_history
        if not pred.get("success", False)
    ]
    if failed_predictions:
        with st.expander("View Failed Predictions", expanded=False):
            for i, pred in enumerate(reversed(failed_predictions[-5:]), 1):
                st.error(
                    f"**{i}.** {pred.get('timestamp_local', datetime.now()).strftime('%m/%d %H:%M')} - {pred.get('error', 'Unknown error')}"
                )


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">MarketMind 🧠</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI-Powered Bitcoin Trading Agent with Reinforcement Learning</p>',
        unsafe_allow_html=True,
    )

    # Display pipeline status warning if needed
    if not PIPELINE_AVAILABLE:
        st.markdown(
            """
            <div class="warning-box">
                <strong>⚠️ Notice:</strong> MarketMind prediction pipeline is not available. 
                Running in demo mode with simulated predictions for demonstration purposes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar
    st.sidebar.markdown("### 🤖 AI Trading Agent")

    # Model status
    model_loaded, model_status = load_model_with_status()
    status_badge = "status-online" if model_loaded else "status-offline"
    status_text = "AI Model: Ready" if model_loaded else "AI Model: Not Available"
    status_emoji = "🟢" if model_loaded else "🔴"

    st.sidebar.markdown(
        f'<span class="status-badge {status_badge}">{status_emoji} {status_text}</span>',
        unsafe_allow_html=True,
    )

    if not model_loaded and PIPELINE_AVAILABLE:
        st.sidebar.warning(f"Model Status: {model_status}")

    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    data_period = st.sidebar.selectbox(
        "Historical data period:",
        [
            str(ConfigurationManager().get_model_training_config().window_size + 50)
            + "d",
            "180d",
            "1y",
            "2y",
        ],
        index=0,
        help="Select the time period for historical data analysis",
    )

    days = convert_period_to_days(data_period)

    # Prediction button
    predict_btn = st.sidebar.button(
        "🔮 Get AI Prediction",
        type="primary",
        disabled=not model_loaded,
        use_container_width=True,
    )

    # Fetch live data with proper error handling
    with st.spinner("📡 Fetching live Bitcoin data..."):
        data, info, status_msg = fetch_bitcoin_data_robust(data_period)

    if "Success" not in status_msg:
        st.error(f"Data fetch failed: {status_msg}")
        st.info("Please check your internet connection and try again.")
        return

    if data is None or data.empty:
        st.error("Unable to fetch Bitcoin data. Please try again later.")
        return

    # Display data fetch status
    st.sidebar.success(f"✅ {status_msg}")

    # Current market status
    try:
        current_price = data["Close"].iloc[-1]
        prev_price = data["Close"].iloc[-2] if len(data) > 1 else current_price
        price_change_24h = ((current_price - prev_price) / prev_price) * 100

        # Market status indicator
        market_status = (
            "🟢 Market Open" if datetime.now().weekday() < 5 else "🔴 Market Closed"
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 Market Status")
        st.sidebar.markdown(f"**{market_status}**")
        st.sidebar.markdown(f"**Current Price:** ${current_price:,.2f}")
        st.sidebar.markdown(f"**24h Change:** {price_change_24h:+.2f}%")
    except Exception as e:
        st.sidebar.error(f"Error calculating market status: {str(e)}")

    # Handle prediction request
    if predict_btn:
        with st.spinner("🤖 AI Agent analyzing market conditions..."):
            if PIPELINE_AVAILABLE and model_loaded:
                try:
                    result = prediction_pipeline.get_prediction("BTC-USD", days)
                    result["timestamp_local"] = datetime.now()
                    st.session_state.prediction_history.append(result)
                    st.session_state["last_result"] = result

                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "symbol": "BTC-USD",
                        "timestamp_local": datetime.now(),
                    }
                    st.session_state.prediction_history.append(error_result)
                    st.session_state["last_result"] = error_result
                    st.error(f"❌ Prediction error: {str(e)}")
            else:
                # Demo prediction
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
                        "trend": np.random.choice(["BULLISH", "BEARISH", "NEUTRAL"]),
                    },
                    "data_points_used": len(data),
                    "timestamp_local": datetime.now(),
                }
                st.session_state.prediction_history.append(demo_result)
                st.session_state["last_result"] = demo_result
                st.info(
                    "🔄 Demo mode: This is a simulated prediction for demonstration purposes"
                )

    # Display prediction results
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        display_prediction_card(result)

        if result.get("success"):
            # Performance metrics
            st.write("")
            st.write("")
            st.write("")
            st.markdown("### 📊 Market Performance")
            st.write("")
            display_performance_metrics(data)

            # Technical analysis
            if result.get("technical_indicators"):
                st.write("")
                st.write("")
                st.write("")
                st.markdown("### 🔍 Technical Analysis")
                st.write("")
                tech_indicators = result["technical_indicators"]

                col1, col2, col3, col4 = st.columns(4)

                try:
                    with col1:
                        st.metric(
                            "Short SMA", f"${tech_indicators.get('sma_short', 0):.2f}"
                        )
                    with col2:
                        st.metric(
                            "Long SMA", f"${tech_indicators.get('sma_long', 0):.2f}"
                        )
                    with col3:
                        st.metric(
                            "Volatility", f"{tech_indicators.get('volatility', 0):.4f}"
                        )
                    with col4:
                        st.metric("Trend", tech_indicators.get("trend", "N/A"))
                except Exception as e:
                    st.error(f"Error displaying technical indicators: {str(e)}")

    # Price chart
    fig = create_price_chart(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Prediction history
    st.write("")
    st.write("")
    display_prediction_history()

    # Important disclaimer
    st.write("")
    st.write("")
    st.markdown(
        """
        <div class="warning-box">
            <strong>⚠️ Important Disclaimer:</strong> This application is for educational and research purposes only. 
            The AI predictions are based on historical data and technical analysis, but cryptocurrency markets are highly volatile and unpredictable. 
            Never make investment decisions based solely on AI predictions. Always conduct your own research and consult with qualified financial advisors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown(
        """
        <div class="footer">
            <h3>MarketMind - AI-Powered Bitcoin Trading Agent</h3>
            <p>Powered by Reinforcement Learning & Advanced Technical Analysis</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">
                Built with Streamlit • Powered by PPO Algorithm • Real-time Yahoo Finance Data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
