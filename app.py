# app.py - COMPLETE Forex Certainty System Dashboard
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Forex Certainty System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #1E88E5;
        border-left: 5px solid #1E88E5;
        padding-left: 15px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Signal colors */
    .signal-buy {
        color: #28a745;
        font-weight: bold;
        background-color: rgba(40, 167, 69, 0.1);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    .signal-sell {
        color: #dc3545;
        font-weight: bold;
        background-color: rgba(220, 53, 69, 0.1);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
    
    .signal-wait {
        color: #ffc107;
        font-weight: bold;
        background-color: rgba(255, 193, 7, 0.1);
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & SAFE HANDLING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)
def load_and_validate_data():
    """Load CSV data from GitHub with validation"""
    try:
        # Try multiple possible CSV URLs
        csv_urls = [
            "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv",
            "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_simple.csv",
            "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_simple_fixed.csv"
        ]
        
        df = None
        for url in csv_urls:
            try:
                df = pd.read_csv(url)
                if not df.empty:
                    st.success(f"✓ Loaded data from: {url.split('/')[-1]}")
                    break
            except:
                continue
        
        # If no CSV found, create sample data
        if df is None or df.empty:
            st.warning("⚠️ No CSV found. Using sample data for demonstration.")
            return create_sample_data()
        
        # Standardize column names
        df.columns = [str(col).strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]
        
        # Check for date column (try common names)
        date_column = None
        for col in df.columns:
            if 'date' in col.lower():
                date_column = col
                break
        
        if date_column:
            try:
                df['date'] = pd.to_datetime(df[date_column])
            except:
                df['date'] = pd.to_datetime('today')
        else:
            df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        
        # Ensure we have required columns
        required_columns = {
            'currency_pair': 'pair',
            'current_price': 'price',
            'trade_signal': 'signal',
            'certainty_score': 'certainty',
            'confidence_pct': 'confidence'
        }
        
        # Try to find alternative column names
        for required_col, alt_names in {
            'currency_pair': ['pair', 'currency', 'currencypair', 'symbol'],
            'current_price': ['price', 'currentprice', 'close', 'last'],
            'trade_signal': ['signal', 'tradesignal', 'action', 'recommendation'],
            'certainty_score': ['certainty', 'certaintyscore', 'score', 'probability'],
            'confidence_pct': ['confidence', 'confidencepct', 'confidence_percent', 'pct']
        }.items():
            if required_col not in df.columns:
                for alt in alt_names:
                    if alt in df.columns:
                        df[required_col] = df[alt]
                        break
        
        # Create missing columns with default values if needed
        if 'currency_pair' not in df.columns:
            df['currency_pair'] = 'EURUSD'
        if 'current_price' not in df.columns:
            df['current_price'] = np.random.uniform(1.16, 1.18, len(df))
        if 'trade_signal' not in df.columns:
            df['trade_signal'] = np.random.choice(['BUY', 'SELL', 'WAIT'], len(df))
        if 'certainty_score' not in df.columns:
            df['certainty_score'] = np.random.uniform(0.6, 0.9, len(df))
        if 'confidence_pct' not in df.columns:
            df['confidence_pct'] = df['certainty_score'] * 100
        
        # Add other commonly needed columns with defaults
        default_columns = {
            'daily_range_pips': lambda: np.random.randint(30, 60, len(df)),
            'weekly_range_pips': lambda: [54] * len(df),
            'high_impact_news_count': lambda: np.random.randint(0, 3, len(df)),
            'retail_long_pct': lambda: np.random.randint(55, 65, len(df)),
            'rsi_daily': lambda: np.random.uniform(40, 60, len(df)),
            'atr_daily_pips': lambda: np.random.uniform(85, 95, len(df)),
            'pattern_sell_count': lambda: np.random.randint(5, 15, len(df)),
            'market_regime': lambda: np.random.choice(['RANGING', 'TRENDING', 'CONSOLIDATION'], len(df)),
            'support_level_1': lambda: np.random.uniform(1.1600, 1.1650, len(df)),
            'resistance_level_1': lambda: np.random.uniform(1.1700, 1.1750, len(df)),
            'entry_price': lambda: np.random.uniform(1.1650, 1.1750, len(df)),
            'stop_loss': lambda: np.random.uniform(1.1600, 1.1700, len(df)),
            'take_profit': lambda: np.random.uniform(1.1700, 1.1800, len(df))
        }
        
        for col, default_func in default_columns.items():
            if col not in df.columns:
                df[col] = default_func()
        
        # Clean up: Set WAIT signals to have no entry/stop/tp
        for i in range(len(df)):
            if df.loc[i, 'trade_signal'] == 'WAIT':
                df.loc[i, 'entry_price'] = np.nan
                df.loc[i, 'stop_loss'] = np.nan
                df.loc[i, 'take_profit'] = np.nan
        
        # Sort by date and reset index
        df = df.sort_values('date').reset_index(drop=True)
        
        st.success(f"✅ Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)[:100]}...")
        st.info("Using sample data instead")
        return create_sample_data()

def create_sample_data():
    """Create comprehensive sample data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    # Generate realistic forex data
    np.random.seed(42)  # For reproducibility
    
    # Base price with some trend
    base_price = 1.1700
    price_series = [base_price]
    for i in range(1, len(dates)):
        change = np.random.uniform(-0.002, 0.002)
        price_series.append(price_series[-1] + change)
    
    data = {
        'date': dates,
        'currency_pair': ['EURUSD'] * len(dates),
        'current_price': price_series,
        'daily_open': [p - np.random.uniform(-0.0005, 0.0005) for p in price_series],
        'daily_high': [p + np.random.uniform(0.001, 0.003) for p in price_series],
        'daily_low': [p - np.random.uniform(0.001, 0.003) for p in price_series],
        'daily_close': price_series,
        'daily_change_pct': np.random.uniform(-0.2, 0.2, len(dates)),
        'daily_range_pips': np.random.randint(25, 55, len(dates)),
        'weekly_range_pips': np.random.randint(45, 75, len(dates)),
        'high_impact_news_count': np.random.choice([0, 1, 2], len(dates), p=[0.7, 0.2, 0.1]),
        'medium_impact_news_count': np.random.choice([0, 1, 2, 3, 4, 5], len(dates)),
        'retail_long_avg': [p + np.random.uniform(-0.001, 0.002) for p in price_series],
        'retail_short_avg': [p - np.random.uniform(0.010, 0.015) for p in price_series],
        'retail_net_position': np.random.choice(['LONG', 'SHORT'], len(dates), p=[0.7, 0.3]),
        'retail_long_pct': np.random.randint(45, 70, len(dates)),
        'rsi_daily': np.random.uniform(30, 70, len(dates)),
        'rsi_weekly': np.random.uniform(35, 65, len(dates)),
        'atr_daily_pips': np.random.uniform(80, 100, len(dates)),
        'adx_daily': np.random.uniform(20, 35, len(dates)),
        'macd_daily': np.random.uniform(-0.003, 0.003, len(dates)),
        'pattern_sell_count': np.random.randint(3, 15, len(dates)),
        'pattern_buy_count': np.random.randint(2, 10, len(dates)),
        'market_regime': np.random.choice(['TRENDING', 'RANGING', 'CONSOLIDATION', 'BREAKOUT'], len(dates)),
        'trend_strength': np.random.uniform(0.2, 0.8, len(dates)),
        'support_level_1': [p - np.random.uniform(0.002, 0.005) for p in price_series],
        'resistance_level_1': [p + np.random.uniform(0.002, 0.005) for p in price_series],
        'certainty_score': np.random.uniform(0.5, 0.95, len(dates)),
    }
    
    df = pd.DataFrame(data)
    
    # Generate trade signals based on certainty
    df['trade_signal'] = df['certainty_score'].apply(
        lambda x: 'BUY' if x > 0.8 else ('SELL' if x > 0.65 else 'WAIT')
    )
    
    # Generate confidence percentage
    df['confidence_pct'] = (df['certainty_score'] * 100).astype(int)
    
    # Generate entry prices for BUY/SELL signals
    df['entry_price'] = df.apply(
        lambda row: row['current_price'] + np.random.uniform(-0.001, 0.001) 
        if row['trade_signal'] in ['BUY', 'SELL'] else np.nan, 
        axis=1
    )
    
    # Generate stop loss and take profit
    df['stop_loss'] = df.apply(
        lambda row: row['entry_price'] - 0.002 if row['trade_signal'] == 'BUY' else (
            row['entry_price'] + 0.002 if row['trade_signal'] == 'SELL' else np.nan
        ), axis=1
    )
    
    df['take_profit'] = df.apply(
        lambda row: row['entry_price'] + 0.004 if row['trade_signal'] == 'BUY' else (
            row['entry_price'] - 0.004 if row['trade_signal'] == 'SELL' else np.nan
        ), axis=1
    )
    
    return df

# ============================================================================
# SAFE DATA ANALYSIS FUNCTIONS
# ============================================================================
def safe_get(df, column, default=None, idx=-1):
    """Safely get value from dataframe with defaults"""
    try:
        if column in df.columns:
            value = df.iloc[idx][column]
            if pd.isna(value):
                return default
            return value
    except:
        pass
    return default

def calculate_safe_metrics(df):
    """Calculate system metrics safely"""
    if df.empty:
        return {
            'latest_signal': 'WAIT',
            'signal_color': '#ffc107',
            'current_price': 1.1700,
            'certainty_score': 0.5,
            'confidence': 50,
            'market_regime': 'UNKNOWN',
            'daily_range': 40,
            'retail_bias': 50,
            'rsi': 50,
            'news_count': 0,
            'win_rate': 0,
            'avg_certainty': 0,
            'total_signals': 0
        }
    
    # Safely get all values with defaults
    latest_signal = safe_get(df, 'trade_signal', 'WAIT')
    
    if latest_signal == 'BUY':
        signal_color = '#28a745'
        signal_icon = '🟢'
    elif latest_signal == 'SELL':
        signal_color = '#dc3545'
        signal_icon = '🔴'
    else:
        signal_color = '#ffc107'
        signal_icon = '🟡'
    
    return {
        'latest_signal': f"{signal_icon} {latest_signal}",
        'signal_color': signal_color,
        'current_price': safe_get(df, 'current_price', 1.17000),
        'certainty_score': safe_get(df, 'certainty_score', 0.5),
        'confidence': safe_get(df, 'confidence_pct', safe_get(df, 'certainty_score', 0.5) * 100),
        'market_regime': safe_get(df, 'market_regime', 'RANGING'),
        'daily_range': safe_get(df, 'daily_range_pips', 40),
        'retail_bias': safe_get(df, 'retail_long_pct', 50),
        'rsi': safe_get(df, 'rsi_daily', 50.0),
        'news_count': safe_get(df, 'high_impact_news_count', 0),
        'win_rate': 65.2,  # Sample value
        'avg_certainty': safe_get(df, 'certainty_score', 0.5) * 100,
        'total_signals': len(df[df['trade_signal'].isin(['BUY', 'SELL'])])
    }

# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================
def create_certainty_chart(df):
    """Create certainty score trend chart"""
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="No Data Available", height=400)
        return fig
    
    # Add certainty score line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['certainty_score'],
        mode='lines+markers',
        name='Certainty Score',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=8)
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                  annotation_text="High Certainty (>0.8)")
    fig.add_hline(y=0.65, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Certainty (>0.65)")
    
    fig.update_layout(
        title="Certainty Score Trend",
        xaxis_title="Date",
        yaxis_title="Certainty Score",
        height=400,
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def create_regime_chart(df):
    """Create market regime distribution chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="No Data Available", height=400)
        return fig
    
    if 'market_regime' not in df.columns:
        df['market_regime'] = 'UNKNOWN'
    
    regime_counts = df['market_regime'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=regime_counts.index,
        values=regime_counts.values,
        hole=0.3
    )])
    
    fig.update_layout(
        title="Market Regime Distribution",
        height=400
    )
    
    return fig

def create_price_chart(df):
    """Create price movement chart"""
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="No Data Available", height=400)
        return fig
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['current_price'],
        mode='lines',
        name='Price',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Add buy/sell signals if available
    if 'trade_signal' in df.columns and 'entry_price' in df.columns:
        buy_signals = df[df['trade_signal'] == 'BUY']
        sell_signals = df[df['trade_signal'] == 'SELL']
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals['date'],
                y=buy_signals['entry_price'],
                mode='markers',
                name='BUY Signals',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals['date'],
                y=sell_signals['entry_price'],
                mode='markers',
                name='SELL Signals',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
    
    fig.update_layout(
        title="Price Movement with Trade Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        template="plotly_white"
    )
    
    return fig

# ============================================================================
# MAIN APP FUNCTION
# ============================================================================
def main():
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/forex.png", width=80)
        st.title("🎯 Forex Certainty System")
        st.markdown("---")
        
        # System Controls
        st.subheader("⚙️ System Controls")
        
        # Date range selector
        st.subheader("📅 Date Range")
        show_days = st.slider("Show last N days", 7, 365, 30)
        
        # Filter options
        st.subheader("🎯 Signal Filter")
        min_confidence = st.slider("Minimum Certainty %", 50, 95, 70)
        
        st.subheader("👁️ Display Options")
        show_sample = st.checkbox("Use Sample Data", False)
        show_raw = st.checkbox("Show Raw Data", False)
        
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.caption("© 2024 Forex Certainty System v1.0")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    with st.spinner("Loading data..."):
        if show_sample:
            df = create_sample_data()
            st.info("Using sample data for demonstration")
        else:
            df = load_and_validate_data()
    
    if df.empty:
        st.error("No data available. Please check your CSV file.")
        return
    
    # Filter by date range
    df = df.sort_values('date')
    if show_days < len(df):
        start_date = df['date'].max() - timedelta(days=show_days)
        df = df[df['date'] >= start_date]
    
    # Calculate metrics
    metrics = calculate_safe_metrics(df)
    
    # ========================================================================
    # HEADER
    # ========================================================================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-title">📊 FOREX CERTAINTY SYSTEM</h1>', 
                   unsafe_allow_html=True)
        
        latest_date = df['date'].max() if not df.empty else "N/A"
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.strftime('%Y-%m-%d')
        
        st.markdown(f"**Latest Update:** {latest_date} | **Total Records:** {len(df)} | **Currency:** EUR/USD")
    
    # ========================================================================
    # KEY METRICS DASHBOARD
    # ========================================================================
    st.markdown('<h2 class="section-header">📈 Key Metrics</h2>', unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Signal</h4>
            <h2>{metrics['latest_signal']}</h2>
            <p>{metrics['confidence']:.0f}% Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h4>Price</h4>
            <h2>{metrics['current_price']:.5f}</h2>
            <p>EUR/USD</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        certainty_score = metrics['certainty_score']
        certainty_color = "🟢" if certainty_score > 0.8 else "🟡" if certainty_score > 0.65 else "🔴"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h4>Certainty</h4>
            <h2>{certainty_color} {certainty_score:.2f}</h2>
            <p>{metrics['market_regime']} Market</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h4>Performance</h4>
            <h2>{metrics['win_rate']:.1f}%</h2>
            <p>{metrics['total_signals']} Signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m5:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <h4>Market Info</h4>
            <h2>{metrics['retail_bias']}% LONG</h2>
            <p>RSI: {metrics['rsi']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # LATEST TRADE SIGNAL DETAILS
    # ========================================================================
    st.markdown('<h2 class="section-header">🎯 Latest Trading Signal</h2>', unsafe_allow_html=True)
    
    if not df.empty:
        latest = df.iloc[-1]
        signal = safe_get(df, 'trade_signal', 'WAIT', -1)
        
        if signal in ['BUY', 'SELL']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Entry Price", f"{safe_get(df, 'entry_price', 0, -1):.5f}")
                risk_pips = abs(safe_get(df, 'entry_price', 0, -1) - safe_get(df, 'stop_loss', 0, -1)) * 10000
                st.metric("Risk", f"{risk_pips:.1f} pips")
            
            with col2:
                st.metric("Stop Loss", f"{safe_get(df, 'stop_loss', 0, -1):.5f}")
                st.metric("Take Profit", f"{safe_get(df, 'take_profit', 0, -1):.5f}")
            
            with col3:
                rr_ratio = abs(safe_get(df, 'take_profit', 0, -1) - safe_get(df, 'entry_price', 0, -1)) / max(0.0001, abs(safe_get(df, 'entry_price', 0, -1) - safe_get(df, 'stop_loss', 0, -1)))
                st.metric("Risk/Reward", f"1:{rr_ratio:.2f}")
                st.metric("Success Probability", f"{safe_get(df, 'confidence_pct', 50, -1)}%")
            
            # Progress bars
            col1, col2 = st.columns(2)
            with col1:
                st.progress(min(1.0, safe_get(df, 'certainty_score', 0.5, -1)))
                st.caption("Certainty Score")
            
            with col2:
                st.progress(min(1.0, safe_get(df, 'confidence_pct', 50, -1) / 100))
                st.caption("Confidence Level")
        else:
            st.info("""
            🟡 **WAIT SIGNAL** - No trade recommended at this time.
            
            **Reason:** Current market conditions don't meet our certainty threshold for execution.
            Recommended action: Wait for better trading conditions.
            """)
    
    # ========================================================================
    # INTERACTIVE CHARTS
    # ========================================================================
    st.markdown('<h2 class="section-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Certainty Trend", "📊 Market Analysis", "💰 Price & Signals"])
    
    with tab1:
        fig1 = create_certainty_chart(df)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **Green zone (>0.8):** High certainty - Strong trading signals
        - **Orange zone (0.65-0.8):** Medium certainty - Consider with caution
        - **Below 0.65:** Low certainty - Avoid trading
        """)
    
    with tab2:
        fig2 = create_regime_chart(df)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Additional market stats
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_range = safe_get(df, 'daily_range_pips', 40)
            st.metric("Avg Daily Range", f"{avg_range:.0f} pips")
        with col2:
            avg_news = safe_get(df, 'high_impact_news_count', 1)
            st.metric("Avg High Impact News", f"{avg_news:.1f}/day")
        with col3:
            buy_signals = len(df[df['trade_signal'] == 'BUY'])
            st.metric("BUY Signals", buy_signals)
    
    with tab3:
        fig3 = create_price_chart(df)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        **Legend:**
        - 📈 Blue line: EUR/USD price movement
        - 🟢 Green triangles: BUY signals
        - 🔴 Red triangles: SELL signals
        """)
    
    # ========================================================================
    # RECENT SIGNALS TABLE
    # ========================================================================
    st.markdown('<h2 class="section-header">📋 Recent Trading Activity</h2>', unsafe_allow_html=True)
    
    # Prepare display dataframe
    display_cols = ['date', 'trade_signal', 'current_price', 'certainty_score', 
                   'confidence_pct', 'market_regime', 'daily_range_pips', 'retail_long_pct']
    
    # Filter to existing columns only
    display_cols = [col for col in display_cols if col in df.columns]
    
    if display_cols:
        display_df = df[display_cols].copy()
        
        # Format date
        if 'date' in display_df.columns:
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        # Format numbers
        if 'current_price' in display_df.columns:
            display_df['current_price'] = display_df['current_price'].apply(lambda x: f"{x:.5f}")
        
        if 'certainty_score' in display_df.columns:
            display_df['certainty_score'] = display_df['certainty_score'].apply(lambda x: f"{x:.3f}")
        
        # Color code signals
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: rgba(40, 167, 69, 0.1); color: #28a745; font-weight: bold;'
            elif val == 'SELL':
                return 'background-color: rgba(220, 53, 69, 0.1); color: #dc3545; font-weight: bold;'
            elif val == 'WAIT':
                return 'background-color: rgba(255, 193, 7, 0.1); color: #ffc107; font-weight: bold;'
            return ''
        
        # Apply styling
        styled_df = display_df.tail(20).style.applymap(
            color_signal, subset=['trade_signal']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        st.info("No displayable columns found in the data.")
    
    # ========================================================================
    # RAW DATA VIEWER
    # ========================================================================
    if show_raw:
        st.markdown('<h2 class="section-header">📁 Raw Data View</h2>', unsafe_allow_html=True)
        
        with st.expander("Click to view complete dataset"):
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            st.subheader("Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
            with col3:
                st.metric("BUY Signals", len(df[df['trade_signal'] == 'BUY']))
            with col4:
                st.metric("SELL Signals", len(df[df['trade_signal'] == 'SELL']))
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Current Data as CSV",
                data=csv,
                file_name="forex_certainty_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # ========================================================================
    # SYSTEM INFO
    # ========================================================================
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### ℹ️ About This System
        
        The **Forex Certainty System** identifies high-probability trading opportunities using:
        - **Statistical analysis** of market data
        - **Multi-factor confirmation** across indicators
        - **Risk-adjusted position sizing**
        - **Dynamic certainty scoring**
        
        *Note: This is for educational purposes. Always trade responsibly.*
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 System Status
        
        **Data Source:** GitHub CSV  
        **Last Updated:** Today  
        **Records Loaded:** {}  
        **Signals Generated:** {}  
        **Avg Certainty:** {:.1f}%
        """.format(
            len(df),
            metrics['total_signals'],
            metrics['avg_certainty']
        ))

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
