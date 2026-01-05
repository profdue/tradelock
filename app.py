# app.py - COMPLETE Forex Certainty System Dashboard
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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
    
    .trade-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)
def load_data():
    """Load CSV data from GitHub"""
    try:
        # Try to load simple CSV (for display)
        simple_df = pd.read_csv("https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv")
        
        # Convert date column
        if 'date' in simple_df.columns:
            simple_df['date'] = pd.to_datetime(simple_df['date'])
        elif 'Date' in simple_df.columns:
            simple_df['date'] = pd.to_datetime(simple_df['Date'])
            simple_df = simple_df.drop('Date', axis=1)
        
        # Clean column names
        simple_df.columns = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in simple_df.columns]
        
        return simple_df, None
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        
        # Create sample data if CSV not found
        st.info("Using sample data for demonstration")
        return create_sample_data(), None

def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    
    data = {
        'date': dates,
        'currency_pair': ['EURUSD'] * len(dates),
        'current_price': np.random.uniform(1.1650, 1.1750, len(dates)),
        'daily_range_pips': np.random.randint(30, 60, len(dates)),
        'weekly_range_pips': [54] * len(dates),
        'high_impact_news_count': np.random.randint(0, 3, len(dates)),
        'retail_long_pct': np.random.randint(55, 65, len(dates)),
        'rsi_daily': np.random.uniform(40, 60, len(dates)),
        'atr_daily_pips': np.random.uniform(85, 95, len(dates)),
        'pattern_sell_count': np.random.randint(5, 15, len(dates)),
        'market_regime': np.random.choice(['RANGING', 'TRENDING', 'CONSOLIDATION'], len(dates)),
        'support_level_1': np.random.uniform(1.1600, 1.1650, len(dates)),
        'resistance_level_1': np.random.uniform(1.1700, 1.1750, len(dates)),
        'certainty_score': np.random.uniform(0.6, 0.9, len(dates)),
        'trade_signal': np.random.choice(['BUY', 'SELL', 'WAIT'], len(dates)),
        'entry_price': np.random.uniform(1.1650, 1.1750, len(dates)),
        'stop_loss': np.random.uniform(1.1600, 1.1700, len(dates)),
        'take_profit': np.random.uniform(1.1700, 1.1800, len(dates)),
        'confidence_pct': np.random.randint(60, 95, len(dates))
    }
    
    # Clear some values for WAIT signals
    for i in range(len(dates)):
        if data['trade_signal'][i] == 'WAIT':
            data['entry_price'][i] = np.nan
            data['stop_loss'][i] = np.nan
            data['take_profit'][i] = np.nan
    
    return pd.DataFrame(data)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def calculate_metrics(df):
    """Calculate system metrics"""
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    # Determine signal color
    if latest['trade_signal'] == 'BUY':
        signal_color = '#28a745'
        signal_icon = '🟢'
    elif latest['trade_signal'] == 'SELL':
        signal_color = '#dc3545'
        signal_icon = '🔴'
    else:
        signal_color = '#ffc107'
        signal_icon = '🟡'
    
    # Calculate performance metrics
    signals = df[df['trade_signal'].isin(['BUY', 'SELL'])]
    
    if len(signals) > 0:
        win_rate = (signals['certainty_score'] > 0.7).mean() * 100
        avg_certainty = signals['certainty_score'].mean() * 100
    else:
        win_rate = 0
        avg_certainty = 0
    
    return {
        'latest_signal': f"{signal_icon} {latest['trade_signal']}",
        'signal_color': signal_color,
        'current_price': latest['current_price'],
        'certainty_score': latest['certainty_score'],
        'confidence': latest.get('confidence_pct', latest['certainty_score'] * 100),
        'market_regime': latest['market_regime'],
        'daily_range': latest['daily_range_pips'],
        'retail_bias': latest['retail_long_pct'],
        'rsi': latest['rsi_daily'],
        'news_count': latest['high_impact_news_count'],
        'win_rate': win_rate,
        'avg_certainty': avg_certainty,
        'total_signals': len(signals)
    }

def create_certainty_chart(df):
    """Create certainty score trend chart"""
    fig = go.Figure()
    
    # Add certainty score line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['certainty_score'],
        mode='lines+markers',
        name='Certainty Score',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=8)
    ))
    
    # Add signal markers
    buy_signals = df[df['trade_signal'] == 'BUY']
    sell_signals = df[df['trade_signal'] == 'SELL']
    
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['certainty_score'],
            mode='markers',
            name='BUY Signals',
            marker=dict(color='#28a745', size=12, symbol='triangle-up')
        ))
    
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(
            x=sell_signals['date'],
            y=sell_signals['certainty_score'],
            mode='markers',
            name='SELL Signals',
            marker=dict(color='#dc3545', size=12, symbol='triangle-down')
        ))
    
    # Add threshold lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                  annotation_text="High Certainty (>0.8)", 
                  annotation_position="bottom right")
    fig.add_hline(y=0.65, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Certainty (>0.65)", 
                  annotation_position="bottom right")
    
    fig.update_layout(
        title="Certainty Score Trend with Trading Signals",
        xaxis_title="Date",
        yaxis_title="Certainty Score",
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def create_regime_chart(df):
    """Create market regime distribution chart"""
    regime_counts = df['market_regime'].value_counts()
    
    # Color mapping for regimes
    regime_colors = {
        'TRENDING': '#28a745',
        'RANGING': '#ffc107',
        'CONSOLIDATION': '#6c757d',
        'BREAKOUT': '#dc3545',
        'VOLATILE': '#6610f2'
    }
    
    colors = [regime_colors.get(regime, '#6c757d') for regime in regime_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=regime_counts.index,
        values=regime_counts.values,
        hole=0.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title="Market Regime Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def create_technical_chart(df):
    """Create technical indicators chart"""
    fig = go.Figure()
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['rsi_daily'],
        mode='lines',
        name='RSI',
        line=dict(color='#FF6B6B', width=2),
        yaxis="y2"
    ))
    
    # ATR
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['atr_daily_pips'],
        mode='lines',
        name='ATR (pips)',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    # Certainty Score
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['certainty_score'] * 100,
        mode='lines',
        name='Certainty %',
        line=dict(color='#1E88E5', width=3),
        yaxis="y3"
    ))
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="Overbought", row=1, col=1, yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="Oversold", row=1, col=1, yref="y2")
    
    fig.update_layout(
        title="Technical Indicators & Certainty",
        xaxis=dict(title="Date"),
        yaxis=dict(title="ATR (pips)", side="left"),
        yaxis2=dict(title="RSI", side="right", overlaying="y", range=[0, 100]),
        yaxis3=dict(title="Certainty %", side="right", overlaying="y", 
                   range=[0, 100], position=0.85),
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

# ============================================================================
# MAIN APP
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
        show_days = st.slider("Show last N days", 7, 90, 30)
        
        # Filter options
        st.subheader("🎯 Signal Filter")
        min_confidence = st.slider("Minimum Confidence %", 50, 95, 70) / 100
        
        # Display options
        st.subheader("👁️ Display")
        show_raw_data = st.checkbox("Show Raw Data", False)
        auto_refresh = st.checkbox("Auto-refresh (every 5 min)", False)
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.caption("© 2024 Forex Certainty System v1.0")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    simple_df, _ = load_data()
    
    if simple_df.empty:
        st.error("No data loaded. Please check your CSV file.")
        return
    
    # Filter by date range
    if 'date' in simple_df.columns:
        simple_df = simple_df.sort_values('date')
        if show_days < len(simple_df):
            simple_df = simple_df.tail(show_days)
    
    # Calculate metrics
    metrics = calculate_metrics(simple_df)
    
    # ========================================================================
    # HEADER
    # ========================================================================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-title">📊 FOREX CERTAINTY SYSTEM</h1>', 
                   unsafe_allow_html=True)
        st.markdown(f"**Latest Update:** {simple_df['date'].max().strftime('%Y-%m-%d %H:%M')} | **Total Records:** {len(simple_df)}")
    
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
        certainty_color = "🟢" if metrics['certainty_score'] > 0.8 else "🟡" if metrics['certainty_score'] > 0.65 else "🔴"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h4>Certainty</h4>
            <h2>{certainty_color} {metrics['certainty_score']:.2f}</h2>
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
            <h4>Retail Sentiment</h4>
            <h2>{metrics['retail_bias']}% LONG</h2>
            <p>RSI: {metrics['rsi']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # LATEST TRADE SIGNAL
    # ========================================================================
    st.markdown('<h2 class="section-header">🎯 Latest Trade Signal</h2>', unsafe_allow_html=True)
    
    latest = simple_df.iloc[-1]
    
    if latest['trade_signal'] != 'WAIT' and not pd.isna(latest['entry_price']):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Entry Details")
            st.metric("Entry Price", f"{latest['entry_price']:.5f}")
            st.metric("Risk (pips)", f"{(abs(latest['entry_price'] - latest['stop_loss']) * 10000):.1f}")
        
        with col2:
            st.markdown("#### Risk Management")
            st.metric("Stop Loss", f"{latest['stop_loss']:.5f}")
            st.metric("Take Profit", f"{latest['take_profit']:.5f}")
        
        with col3:
            st.markdown("#### Statistics")
            st.metric("Risk/Reward", f"1:{((abs(latest['take_profit'] - latest['entry_price']) / abs(latest['entry_price'] - latest['stop_loss']))):.2f}")
            st.metric("Success Probability", f"{latest['confidence_pct']}%")
        
        # Progress bars
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Certainty Score**")
            st.progress(latest['certainty_score'])
        
        with col2:
            st.markdown("**Confidence Level**")
            st.progress(latest['confidence_pct'] / 100)
    
    else:
        st.info("🟡 **WAIT SIGNAL** - No trade recommended at this time. Waiting for better certainty conditions.")
    
    # ========================================================================
    # CHARTS SECTION
    # ========================================================================
    st.markdown('<h2 class="section-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Chart 1: Certainty Trend
    tab1, tab2, tab3 = st.tabs(["📈 Certainty Trend", "📊 Market Regimes", "⚙️ Technical Indicators"])
    
    with tab1:
        fig1 = create_certainty_chart(simple_df)
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = create_regime_chart(simple_df)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        fig3 = create_technical_chart(simple_df)
        st.plotly_chart(fig3, use_container_width=True)
    
    # ========================================================================
    # RECENT SIGNALS TABLE
    # ========================================================================
    st.markdown('<h2 class="section-header">📋 Recent Trading Signals</h2>', unsafe_allow_html=True)
    
    # Create display dataframe
    display_df = simple_df.copy()
    
    # Add signal styling
    def style_signal(val):
        if val == 'BUY':
            return 'color: #28a745; font-weight: bold;'
        elif val == 'SELL':
            return 'color: #dc3545; font-weight: bold;'
        else:
            return 'color: #ffc107; font-weight: bold;'
    
    # Format for display
    display_columns = {
        'date': 'Date',
        'trade_signal': 'Signal',
        'current_price': 'Price',
        'certainty_score': 'Certainty',
        'confidence_pct': 'Confidence %',
        'market_regime': 'Market',
        'daily_range_pips': 'Daily Range',
        'retail_long_pct': 'Retail %',
        'high_impact_news_count': 'High News'
    }
    
    # Filter columns that exist
    existing_cols = {k: v for k, v in display_columns.items() if k in display_df.columns}
    display_df = display_df[list(existing_cols.keys())].copy()
    display_df.columns = [existing_cols[col] for col in display_df.columns]
    
    # Format date
    if 'Date' in display_df.columns:
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Format numbers
    if 'Certainty' in display_df.columns:
        display_df['Certainty'] = display_df['Certainty'].apply(lambda x: f"{x:.2f}")
    
    if 'Price' in display_df.columns:
        display_df['Price'] = display_df['Price'].apply(lambda x: f"{x:.5f}")
    
    # Show table
    st.dataframe(
        display_df.tail(20),
        use_container_width=True,
        height=400
    )
    
    # ========================================================================
    # RAW DATA
    # ========================================================================
    if show_raw_data:
        st.markdown('<h2 class="section-header">📁 Raw Data</h2>', unsafe_allow_html=True)
        
        with st.expander("View Complete Dataset"):
            st.dataframe(simple_df, use_container_width=True)
            
            # Download button
            csv = simple_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="forex_certainty_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # ========================================================================
    # SYSTEM STATUS
    # ========================================================================
    st.markdown('<h2 class="section-header">⚡ System Status</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", len(simple_df))
    
    with col2:
        active_signals = len(simple_df[simple_df['trade_signal'].isin(['BUY', 'SELL'])])
        st.metric("Active Signals", active_signals)
    
    with col3:
        avg_certainty = simple_df['certainty_score'].mean()
        st.metric("Average Certainty", f"{avg_certainty:.2f}")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p><strong>Forex Certainty System v1.0</strong> | Data updates automatically | For educational purposes only</p>
            <p>Trade signals are based on statistical certainty, not predictions. Always use proper risk management.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()
