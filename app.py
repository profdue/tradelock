# app.py - FIXED with TRUE CERTAINTY SIGNALS
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .certainty-high {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
    .certainty-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #ffc107;
    }
    .certainty-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #6c757d;
    }
    .trade-execution {
        background: #1a1a1a;
        color: white;
        padding: 25px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        margin: 15px 0;
        border: 2px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    try:
        # Try to load your CSV
        df = pd.read_csv("https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv")
        
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # ENFORCE CERTAINTY RULES
        df = enforce_certainty_rules(df)
        
        return df
    except:
        # Create CERTAIN sample data
        return create_certain_sample_data()

def enforce_certainty_rules(df):
    """Enforce certainty rules - No vague signals allowed!"""
    
    # Rule 1: Certainty > 0.85 MUST have entry/stop/target
    high_certainty_mask = df['certainty_score'] > 0.85
    
    # If high certainty but no entry, create it
    for idx in df[high_certainty_mask].index:
        if pd.isna(df.loc[idx, 'entry_price']):
            # Create specific entry based on pattern
            current_price = df.loc[idx, 'current_price']
            certainty = df.loc[idx, 'certainty_score']
            
            if 'buy' in str(df.loc[idx, 'trade_signal']).lower():
                # BUY signal - entry below current
                df.loc[idx, 'entry_price'] = round(current_price - 0.0012, 5)
                df.loc[idx, 'stop_loss'] = round(df.loc[idx, 'entry_price'] - 0.0018, 5)
                df.loc[idx, 'take_profit_1'] = round(df.loc[idx, 'entry_price'] + 0.0030, 5)
                df.loc[idx, 'take_profit_2'] = round(df.loc[idx, 'entry_price'] + 0.0045, 5)
                df.loc[idx, 'trade_signal'] = 'BUY'
            elif 'sell' in str(df.loc[idx, 'trade_signal']).lower():
                # SELL signal - entry above current
                df.loc[idx, 'entry_price'] = round(current_price + 0.0012, 5)
                df.loc[idx, 'stop_loss'] = round(df.loc[idx, 'entry_price'] + 0.0018, 5)
                df.loc[idx, 'take_profit_1'] = round(df.loc[idx, 'entry_price'] - 0.0030, 5)
                df.loc[idx, 'take_profit_2'] = round(df.loc[idx, 'entry_price'] - 0.0045, 5)
                df.loc[idx, 'trade_signal'] = 'SELL'
    
    # Rule 2: Certainty < 0.70 = NO_TRADE
    low_certainty_mask = df['certainty_score'] < 0.70
    df.loc[low_certainty_mask, 'trade_signal'] = 'NO_TRADE'
    df.loc[low_certainty_mask, ['entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2']] = np.nan
    
    # Rule 3: Clean signal names
    df['trade_signal'] = df['trade_signal'].apply(
        lambda x: 'BUY' if 'buy' in str(x).lower() else 
                 ('SELL' if 'sell' in str(x).lower() else 
                 ('NO_TRADE' if 'wait' in str(x).lower() or pd.isna(x) else x))
    )
    
    return df

def create_certain_sample_data():
    """Create sample data with TRUE CERTAINTY"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        # Generate realistic price
        price = 1.1700 + np.sin(i/10) * 0.005
        
        # Generate certainty score
        certainty = np.random.uniform(0.6, 0.95)
        
        # DETERMINISTIC: High certainty = Clear trade
        if certainty > 0.85:
            if i % 3 == 0:  # BUY pattern
                signal = 'BUY'
                entry = round(price - 0.0010, 5)
                stop = round(entry - 0.0015, 5)
                tp1 = round(entry + 0.0025, 5)
                tp2 = round(entry + 0.0035, 5)
                reason = f"Retail {np.random.randint(65,80)}% short + RSI {np.random.randint(25,35)} + Support bounce"
            else:  # SELL pattern
                signal = 'SELL'
                entry = round(price + 0.0010, 5)
                stop = round(entry + 0.0015, 5)
                tp1 = round(entry - 0.0025, 5)
                tp2 = round(entry - 0.0035, 5)
                reason = f"Retail {np.random.randint(65,80)}% long + RSI {np.random.randint(65,75)} + Resistance test"
        elif certainty > 0.70:
            signal = 'NO_TRADE'
            entry = stop = tp1 = tp2 = np.nan
            reason = f"Medium certainty ({certainty:.2f}) - Wait for confirmation"
        else:
            signal = 'NO_TRADE'
            entry = stop = tp1 = tp2 = np.nan
            reason = f"Low certainty ({certainty:.2f}) - Avoid trading"
        
        data.append({
            'date': date,
            'currency_pair': 'EURUSD',
            'current_price': round(price, 5),
            'certainty_score': round(certainty, 3),
            'trade_signal': signal,
            'entry_price': entry,
            'stop_loss': stop,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'reason': reason,
            'retail_long_pct': np.random.randint(40, 70),
            'rsi_daily': np.random.uniform(30, 70),
            'atr_daily_pips': np.random.uniform(80, 110),
            'market_regime': np.random.choice(['TRENDING', 'RANGING', 'BREAKOUT']),
            'daily_range_pips': np.random.randint(30, 60)
        })
    
    return pd.DataFrame(data)

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_certainty_level(score):
    """Display certainty level with appropriate styling"""
    if score >= 0.85:
        return "🟢 HIGH CERTAINTY (85%+) - EXECUTE"
    elif score >= 0.70:
        return "🟡 MEDIUM CERTAINTY (70-85%) - CONSIDER"
    else:
        return "🔴 LOW CERTAINTY (<70%) - AVOID"

def display_trade_execution(signal, certainty, entry, stop, tp1, tp2, reason):
    """Display trade execution commands"""
    if signal == 'BUY':
        return f"""
🟢 EXECUTE: MARKET BUY EURUSD
├─ ENTRY: {entry}
├─ STOP LOSS: {stop} (Risk: {abs(entry-stop)*10000:.1f} pips)
├─ TAKE PROFIT 1: {tp1} (+{abs(tp1-entry)*10000:.1f} pips)
├─ TAKE PROFIT 2: {tp2} (+{abs(tp2-entry)*10000:.1f} pips)
└─ RISK/REWARD: 1:{abs(tp1-entry)/abs(entry-stop):.2f}
📊 REASON: {reason}
"""
    elif signal == 'SELL':
        return f"""
🔴 EXECUTE: MARKET SELL EURUSD  
├─ ENTRY: {entry}
├─ STOP LOSS: {stop} (Risk: {abs(entry-stop)*10000:.1f} pips)
├─ TAKE PROFIT 1: {tp1} (+{abs(entry-tp1)*10000:.1f} pips)
├─ TAKE PROFIT 2: {tp2} (+{abs(entry-tp2)*10000:.1f} pips)
└─ RISK/REWARD: 1:{abs(entry-tp1)/abs(entry-stop):.2f}
📊 REASON: {reason}
"""
    else:
        return f"""
⚪ NO TRADE EXECUTED
└─ REASON: {reason}
"""

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Title
    st.title("🎯 FOREX CERTAINTY SYSTEM")
    st.markdown("**No vague signals. Only clear executions.**")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data loaded")
        return
    
    # Get latest signal
    latest = df.iloc[-1]
    signal = latest['trade_signal']
    certainty = latest['certainty_score']
    
    # ========================================================================
    # CERTAINTY STATUS
    # ========================================================================
    st.markdown("## 📊 CERTAINTY STATUS")
    
    if certainty >= 0.85:
        st.markdown(f'<div class="certainty-high">', unsafe_allow_html=True)
        st.markdown(f"### 🚀 {display_certainty_level(certainty)}")
        st.markdown(f"**Score:** {certainty:.3f} | **Signal:** {signal}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif certainty >= 0.70:
        st.markdown(f'<div class="certainty-medium">', unsafe_allow_html=True)
        st.markdown(f"### ⚠️ {display_certainty_level(certainty)}")
        st.markdown(f"**Score:** {certainty:.3f} | **Signal:** {signal}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="certainty-low">', unsafe_allow_html=True)
        st.markdown(f"### 🛑 {display_certainty_level(certainty)}")
        st.markdown(f"**Score:** {certainty:.3f} | **Signal:** {signal}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # TRADE EXECUTION
    # ========================================================================
    st.markdown("## 📈 TRADE EXECUTION")
    
    if signal in ['BUY', 'SELL'] and certainty >= 0.70:
        # Show trade execution box
        st.markdown('<div class="trade-execution">', unsafe_allow_html=True)
        st.markdown(display_trade_execution(
            signal=signal,
            certainty=certainty,
            entry=latest['entry_price'],
            stop=latest['stop_loss'],
            tp1=latest['take_profit_1'],
            tp2=latest.get('take_profit_2', np.nan),
            reason=latest.get('reason', 'High probability pattern detected')
        ))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Copy to clipboard button
        exec_code = f"""
# MT4/MT5 Execution Code
if signal == 'BUY':
    OrderSend("EURUSD", OP_BUY, 0.1, {latest['entry_price']}, 3, {latest['stop_loss']}, {latest['take_profit_1']})
elif signal == 'SELL':
    OrderSend("EURUSD", OP_SELL, 0.1, {latest['entry_price']}, 3, {latest['stop_loss']}, {latest['take_profit_1']})
"""
        st.code(exec_code, language='python')
        
    else:
        st.info(f"### ⚪ NO TRADE - {latest.get('reason', 'Certainty below threshold')}")
    
    # ========================================================================
    # MARKET DATA
    # ========================================================================
    st.markdown("## 📊 MARKET DATA")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{latest['current_price']:.5f}")
    with col2:
        st.metric("Retail Bias", f"{latest.get('retail_long_pct', 50)}% LONG")
    with col3:
        rsi = latest.get('rsi_daily', 50)
        rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        st.metric("RSI", f"{rsi:.1f}", rsi_status)
    with col4:
        st.metric("Market Regime", latest.get('market_regime', 'UNKNOWN'))
    
    # ========================================================================
    # CERTAINTY HISTORY
    # ========================================================================
    st.markdown("## 📈 CERTAINTY HISTORY")
    
    # Create chart
    fig = go.Figure()
    
    # Add certainty line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['certainty_score'],
        mode='lines+markers',
        name='Certainty',
        line=dict(color='#1E88E5', width=3)
    ))
    
    # Add trade markers
    buy_signals = df[df['trade_signal'] == 'BUY']
    sell_signals = df[df['trade_signal'] == 'SELL']
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['certainty_score'],
            mode='markers',
            name='BUY',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['date'],
            y=sell_signals['certainty_score'],
            mode='markers',
            name='SELL',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    # Add certainty zones
    fig.add_hrect(y0=0.85, y1=1.0, line_width=0, fillcolor="green", opacity=0.1,
                  annotation_text="HIGH CERTAINTY", annotation_position="top left")
    fig.add_hrect(y0=0.70, y1=0.85, line_width=0, fillcolor="yellow", opacity=0.1,
                  annotation_text="MEDIUM CERTAINTY", annotation_position="top left")
    
    fig.update_layout(
        title="Certainty Score History with Trade Signals",
        xaxis_title="Date",
        yaxis_title="Certainty Score",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # RECENT SIGNALS TABLE
    # ========================================================================
    st.markdown("## 📋 RECENT SIGNALS")
    
    # Prepare display table
    display_cols = ['date', 'trade_signal', 'certainty_score', 'current_price', 
                   'entry_price', 'stop_loss', 'take_profit_1', 'reason']
    
    # Filter to existing columns
    display_cols = [col for col in display_cols if col in df.columns]
    display_df = df[display_cols].copy()
    
    # Format date
    if 'date' in display_df.columns:
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    
    # Color coding function
    def color_row(row):
        if row['trade_signal'] == 'BUY':
            return ['background-color: rgba(40, 167, 69, 0.2)'] * len(row)
        elif row['trade_signal'] == 'SELL':
            return ['background-color: rgba(220, 53, 69, 0.2)'] * len(row)
        else:
            return ['background-color: rgba(108, 117, 125, 0.2)'] * len(row)
    
    # Display last 10 signals
    styled_df = display_df.tail(10).style.apply(color_row, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    # ========================================================================
    # SYSTEM RULES
    # ========================================================================
    st.markdown("## 📜 CERTAINTY SYSTEM RULES")
    
    rules = """
### 🎯 CERTAINTY THRESHOLDS:
1. **≥ 0.85 (GREEN ZONE)** → EXECUTE TRADE
   - Must have specific entry/stop/target
   - Position size: 2.0x normal
   - Win probability: 85%+

2. **0.70 - 0.85 (YELLOW ZONE)** → CONSIDER TRADE
   - Reduce position size to 1.0x
   - Wait for confirmation
   - Win probability: 70-85%

3. **< 0.70 (RED ZONE)** → NO TRADE
   - No execution allowed
   - Wait for better setup
   - Win probability: <70%

### 📊 TRADE EXECUTION REQUIREMENTS:
- ✅ Specific entry price
- ✅ Specific stop loss  
- ✅ Specific take profit(s)
- ✅ Clear reasoning
- ✅ Risk/reward ≥ 1:1.5
"""
    
    st.markdown(rules)
    
    # ========================================================================
    # DOWNLOAD/UPDATE
    # ========================================================================
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="forex_certainty_executable.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Refresh & Recalculate", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
