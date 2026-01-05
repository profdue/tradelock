# app.py - FIXED Forex Certainty System
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .trade-buy {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #28a745;
    }
    .trade-sell {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #dc3545;
    }
    .no-trade {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #6c757d;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .pattern-item {
        background: #e9ecef;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA PROCESSING - FIXED
# ============================================================================
@st.cache_data(ttl=300)
def load_and_validate_data():
    """Load and validate CSV data with proper error handling"""
    try:
        # Load your CSV from GitHub
        csv_url = "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv"
        df = pd.read_csv(csv_url)
        
        # DEBUG: Show what we loaded
        st.info(f"📊 CSV Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Clean column names
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
        
        # Convert date - handle different formats
        date_column = None
        for col in df.columns:
            if 'date' in col.lower():
                date_column = col
                break
        
        if date_column:
            df['date'] = pd.to_datetime(df[date_column], errors='coerce')
            # If conversion failed, use today's date
            df['date'] = df['date'].fillna(pd.Timestamp('today').normalize())
        else:
            df['date'] = pd.Timestamp('today').normalize()
        
        # Sort by date and get latest
        df = df.sort_values('date', ascending=True)
        
        # Ensure we have current_price column
        if 'current_price' not in df.columns:
            # Try to find price column
            for col in ['price', 'close', 'last']:
                if col in df.columns:
                    df['current_price'] = df[col]
                    break
        
        # Convert numeric columns
        numeric_columns = [
            'current_price', 'daily_range_pips', 'weekly_range_pips',
            'retail_long_pct', 'rsi_daily', 'atr_daily_pips',
            'certainty_score', 'confidence_pct', 'support_level_1',
            'resistance_level_1', 'daily_change_pct'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        # Get latest data
        latest = df.iloc[-1].to_dict()
        
        # Ensure we have required fields
        defaults = {
            'current_price': 1.1700,
            'retail_long_pct': 50,
            'rsi_daily': 50,
            'atr_daily_pips': 80,
            'daily_range_pips': 40,
            'support_level_1': 1.1650,
            'resistance_level_1': 1.1750,
            'market_regime': 'RANGING',
            'certainty_score': 0.5,
            'currency_pair': 'EURUSD'
        }
        
        for key, default_value in defaults.items():
            if key not in latest or pd.isna(latest[key]):
                latest[key] = default_value
        
        st.success("✅ Data loaded and validated successfully")
        return df, latest
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)[:100]}...")
        
        # Create emergency data
        today = datetime.now().date()
        emergency_data = {
            'date': today,
            'current_price': 1.1700,
            'retail_long_pct': 50,
            'rsi_daily': 50,
            'atr_daily_pips': 80,
            'daily_range_pips': 40,
            'support_level_1': 1.1650,
            'resistance_level_1': 1.1750,
            'market_regime': 'RANGING',
            'certainty_score': 0.5,
            'currency_pair': 'EURUSD',
            'daily_change_pct': 0.0,
            'weekly_range_pips': 60
        }
        
        df = pd.DataFrame([emergency_data])
        return df, emergency_data

# ============================================================================
# PATTERN DETECTION - FIXED
# ============================================================================
def detect_patterns(data):
    """Detect trading patterns from data"""
    patterns = {
        'retail_patterns': [],
        'rsi_patterns': [],
        'volatility_patterns': [],
        'market_regime': {}
    }
    
    # 1. Retail Patterns
    retail_pct = float(data.get('retail_long_pct', 50))
    retail_distance = float(data.get('retail_distance_pips', 0))
    
    if retail_pct > 70:
        patterns['retail_patterns'].append({
            'name': 'retail_crowded_long',
            'signal': 'SELL',
            'strength': min((retail_pct - 70) / 30, 1.0),
            'reason': f'Retail {retail_pct:.0f}% long (crowded)',
            'certainty': 0.85
        })
    elif retail_pct < 30:
        patterns['retail_patterns'].append({
            'name': 'retail_crowded_short',
            'signal': 'BUY',
            'strength': min((30 - retail_pct) / 30, 1.0),
            'reason': f'Retail {retail_pct:.0f}% long (crowded short)',
            'certainty': 0.85
        })
    
    if abs(retail_distance) > 30:
        patterns['retail_patterns'].append({
            'name': 'retail_pain_point',
            'signal': 'SELL' if retail_distance < 0 else 'BUY',
            'strength': min(abs(retail_distance) / 100, 1.0),
            'reason': f'Retail {abs(retail_distance):.0f} pips underwater',
            'certainty': 0.80
        })
    
    # 2. RSI Patterns
    rsi = float(data.get('rsi_daily', 50))
    
    if rsi < 30:
        patterns['rsi_patterns'].append({
            'name': 'rsi_oversold',
            'signal': 'BUY',
            'strength': (30 - rsi) / 30,
            'reason': f'RSI oversold: {rsi:.1f}',
            'certainty': 0.80
        })
    elif rsi > 70:
        patterns['rsi_patterns'].append({
            'name': 'rsi_overbought',
            'signal': 'SELL',
            'strength': (rsi - 70) / 30,
            'reason': f'RSI overbought: {rsi:.1f}',
            'certainty': 0.80
        })
    
    # 3. Volatility Patterns
    daily_range = float(data.get('daily_range_pips', 40))
    atr = float(data.get('atr_daily_pips', 80))
    
    if atr > 0:
        volatility_ratio = daily_range / atr
        
        if volatility_ratio < 0.5:
            patterns['volatility_patterns'].append({
                'name': 'low_volatility',
                'signal': 'RANGE_BOUND',
                'strength': (0.5 - volatility_ratio) * 2,
                'reason': f'Low volatility: {daily_range:.0f}pips ({volatility_ratio:.2f}x ATR)',
                'certainty': 0.75
            })
        elif volatility_ratio > 1.5:
            patterns['volatility_patterns'].append({
                'name': 'high_volatility',
                'signal': 'TRENDING',
                'strength': min((volatility_ratio - 1.5) / 2, 1.0),
                'reason': f'High volatility: {daily_range:.0f}pips ({volatility_ratio:.2f}x ATR)',
                'certainty': 0.70
            })
    
    # 4. Market Regime
    regime = str(data.get('market_regime', 'RANGING')).upper()
    
    if 'TREND' in regime:
        patterns['market_regime'] = {
            'regime': 'TRENDING',
            'signal': 'BUY' if 'UP' in regime else 'SELL' if 'DOWN' in regime else 'NEUTRAL',
            'strength': float(data.get('trend_strength', 0.5)),
            'certainty': 0.75
        }
    elif 'RANG' in regime:
        patterns['market_regime'] = {
            'regime': 'RANGING',
            'signal': 'RANGE_BOUND',
            'strength': 0.7,
            'certainty': 0.70
        }
    else:
        patterns['market_regime'] = {
            'regime': regime,
            'signal': 'NEUTRAL',
            'strength': 0.5,
            'certainty': 0.65
        }
    
    return patterns

# ============================================================================
# CERTAINTY CALCULATION - FIXED
# ============================================================================
def calculate_certainty(patterns):
    """Calculate overall certainty score"""
    
    weights = {
        'retail_patterns': 0.25,
        'rsi_patterns': 0.20,
        'volatility_patterns': 0.20,
        'market_regime': 0.35
    }
    
    category_scores = {}
    category_signals = {}
    
    # Calculate scores for each category
    for category, pattern_data in patterns.items():
        if category == 'market_regime':
            # Handle market regime specially
            if pattern_data:
                score = pattern_data.get('strength', 0.5) * pattern_data.get('certainty', 0.5)
                signal = pattern_data.get('signal', 'NEUTRAL')
                category_scores[category] = score
                category_signals[category] = signal
            continue
        
        # For pattern lists
        if isinstance(pattern_data, list) and pattern_data:
            # Find strongest pattern in category
            strongest = max(pattern_data, key=lambda x: x.get('strength', 0))
            score = strongest.get('strength', 0) * strongest.get('certainty', 0.5)
            signal = strongest.get('signal', 'NEUTRAL')
            category_scores[category] = score
            category_signals[category] = signal
        else:
            category_scores[category] = 0.5
            category_signals[category] = 'NEUTRAL'
    
    # Apply weights
    weighted_score = 0
    for category, score in category_scores.items():
        weight = weights.get(category, 0.10)
        weighted_score += score * weight
    
    # Determine primary signal
    signal_counts = {}
    for signal in category_signals.values():
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    primary_signal = max(signal_counts, key=signal_counts.get) if signal_counts else 'NEUTRAL'
    
    return {
        'overall_certainty': min(weighted_score, 1.0),
        'primary_signal': primary_signal,
        'category_scores': category_scores,
        'category_signals': category_signals
    }

# ============================================================================
# TRADE GENERATION - FIXED
# ============================================================================
def generate_trade(data, patterns, certainty_data):
    """Generate trade based on patterns and certainty"""
    
    certainty = certainty_data['overall_certainty']
    signal = certainty_data['primary_signal']
    
    if certainty < 0.60:
        return {
            'signal': 'NO_TRADE',
            'certainty': certainty,
            'reason': f'Insufficient certainty ({certainty:.1%})',
            'recommendation': 'Wait for better setup'
        }
    
    # Get price data
    current_price = float(data.get('current_price', 1.1700))
    atr = float(data.get('atr_daily_pips', 80))
    support = float(data.get('support_level_1', current_price - 0.0020))
    resistance = float(data.get('resistance_level_1', current_price + 0.0020))
    
    if signal in ['BUY', 'BULLISH']:
        # Generate BUY trade
        entry_distance = atr * 0.0001  # 10% of ATR
        entry_price = max(support + 0.0001, current_price - entry_distance)
        stop_distance = atr * 0.00015  # 1.5x ATR
        stop_loss = entry_price - stop_distance
        
        # Calculate take profit based on risk/reward
        risk = entry_price - stop_loss
        if certainty > 0.85:
            risk_reward = 2.5
        elif certainty > 0.75:
            risk_reward = 2.0
        else:
            risk_reward = 1.5
        
        take_profit = entry_price + (risk * risk_reward)
        
        return {
            'signal': 'BUY',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': round(100 / (abs(entry_price - stop_loss) * 10000 * 10), 2),
            'reason': _generate_reason(patterns, 'BUY'),
            'conditions': ['Price reaches entry level', 'Stop loss on breach']
        }
    
    elif signal in ['SELL', 'BEARISH']:
        # Generate SELL trade
        entry_distance = atr * 0.0001
        entry_price = min(resistance - 0.0001, current_price + entry_distance)
        stop_distance = atr * 0.00015
        stop_loss = entry_price + stop_distance
        
        risk = stop_loss - entry_price
        if certainty > 0.85:
            risk_reward = 2.5
        elif certainty > 0.75:
            risk_reward = 2.0
        else:
            risk_reward = 1.5
        
        take_profit = entry_price - (risk * risk_reward)
        
        return {
            'signal': 'SELL',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': round(100 / (abs(entry_price - stop_loss) * 10000 * 10), 2),
            'reason': _generate_reason(patterns, 'SELL'),
            'conditions': ['Price reaches entry level', 'Stop loss on breach']
        }
    
    else:
        return {
            'signal': 'NO_TRADE',
            'certainty': certainty,
            'reason': 'No clear directional signal',
            'recommendation': 'Wait for pattern confirmation'
        }

def _generate_reason(patterns, signal):
    """Generate trade reason from patterns"""
    reasons = []
    
    for category, pattern_data in patterns.items():
        if category == 'market_regime':
            if pattern_data and pattern_data.get('signal') == signal:
                reasons.append(f"Market in {pattern_data.get('regime', '')} regime")
            continue
        
        if isinstance(pattern_data, list):
            for pattern in pattern_data:
                if pattern.get('signal') == signal:
                    reasons.append(pattern.get('reason', ''))
    
    # Return top 2 reasons
    return " | ".join(reasons[:2]) if reasons else "Multiple factors aligned"

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Title
    st.title("🎯 FOREX CERTAINTY SYSTEM")
    st.markdown("**Real-time analysis from CSV data**")
    
    # Load data
    df, latest_data = load_and_validate_data()
    
    # Show current date
    current_date = latest_data.get('date', datetime.now().date())
    if hasattr(current_date, 'strftime'):
        display_date = current_date.strftime('%Y-%m-%d')
    else:
        display_date = str(current_date)
    
    st.markdown(f"### 📅 Current Analysis Date: **{display_date}**")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price", 
            f"{latest_data.get('current_price', 0):.5f}",
            f"{latest_data.get('daily_change_pct', 0):.2f}%"
        )
    
    with col2:
        retail = latest_data.get('retail_long_pct', 50)
        st.metric("Retail Bias", f"{retail:.0f}% LONG")
    
    with col3:
        rsi = latest_data.get('rsi_daily', 50)
        rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        st.metric("RSI", f"{rsi:.1f}", rsi_status)
    
    with col4:
        regime = latest_data.get('market_regime', 'RANGING')
        st.metric("Market Regime", regime)
    
    # Pattern detection
    patterns = detect_patterns(latest_data)
    
    # Calculate certainty
    certainty_data = calculate_certainty(patterns)
    certainty_score = certainty_data['overall_certainty']
    
    # Display certainty
    st.markdown(f"### 🎯 Certainty Score: **{certainty_score:.1%}**")
    
    # Progress bar with color coding
    if certainty_score >= 0.80:
        color = "green"
    elif certainty_score >= 0.65:
        color = "orange"
    else:
        color = "red"
    
    st.progress(certainty_score)
    
    # Generate trade
    trade = generate_trade(latest_data, patterns, certainty_data)
    
    # Display trade signal
    st.markdown("### 📊 TRADE SIGNAL")
    
    if trade['signal'] == 'NO_TRADE':
        st.markdown(f"""
        <div class="no-trade">
            <h3>🚫 NO TRADE RECOMMENDED</h3>
            <p><strong>Reason:</strong> {trade['reason']}</p>
            <p><strong>Certainty:</strong> {trade['certainty']:.1%}</p>
            <p><strong>Recommendation:</strong> {trade.get('recommendation', 'Wait')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        trade_class = "trade-buy" if trade['signal'] == 'BUY' else "trade-sell"
        
        st.markdown(f"""
        <div class="{trade_class}">
            <h3>📈 {'🟢 BUY' if trade['signal'] == 'BUY' else '🔴 SELL'} EURUSD</h3>
            <p><strong>Entry Price:</strong> {trade['entry_price']}</p>
            <p><strong>Stop Loss:</strong> {trade['stop_loss']}</p>
            <p><strong>Take Profit:</strong> {trade['take_profit']}</p>
            <p><strong>Risk/Reward:</strong> 1:{trade['risk_reward']:.1f}</p>
            <p><strong>Position Size:</strong> {trade['position_size']} lots</p>
            <p><strong>Certainty:</strong> {trade['certainty']:.1%}</p>
            <p><strong>Reason:</strong> {trade['reason']}</p>
            <p><strong>Conditions:</strong> {', '.join(trade['conditions'])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pattern Analysis
    st.markdown("### 🔍 PATTERN ANALYSIS")
    
    pattern_cols = st.columns(4)
    
    with pattern_cols[0]:
        st.markdown("**Retail Patterns**")
        if patterns['retail_patterns']:
            for pattern in patterns['retail_patterns']:
                st.markdown(f"<div class='pattern-item'>{pattern.get('reason', '')}</div>", unsafe_allow_html=True)
        else:
            st.info("No retail patterns detected")
    
    with pattern_cols[1]:
        st.markdown("**RSI Patterns**")
        if patterns['rsi_patterns']:
            for pattern in patterns['rsi_patterns']:
                st.markdown(f"<div class='pattern-item'>{pattern.get('reason', '')}</div>", unsafe_allow_html=True)
        else:
            st.info("No RSI patterns detected")
    
    with pattern_cols[2]:
        st.markdown("**Volatility Patterns**")
        if patterns['volatility_patterns']:
            for pattern in patterns['volatility_patterns']:
                st.markdown(f"<div class='pattern-item'>{pattern.get('reason', '')}</div>", unsafe_allow_html=True)
        else:
            st.info("No volatility patterns")
    
    with pattern_cols[3]:
        st.markdown("**Market Regime**")
        if patterns['market_regime']:
            regime_info = patterns['market_regime']
            st.markdown(f"<div class='pattern-item'>{regime_info.get('regime', 'Unknown')} (Strength: {regime_info.get('strength', 0):.1f})</div>", unsafe_allow_html=True)
    
    # Debug Information
    with st.expander("🔧 Debug Information"):
        tab1, tab2, tab3 = st.tabs(["CSV Data", "Patterns", "Calculations"])
        
        with tab1:
            st.write("Latest Data Row:")
            st.json(latest_data)
            
            st.write("Full DataFrame:")
            st.dataframe(df)
        
        with tab2:
            st.json(patterns)
        
        with tab3:
            st.json(certainty_data)
            st.json(trade)
    
    # Execution Code for BUY/SELL
    if trade['signal'] in ['BUY', 'SELL']:
        st.markdown("### 💻 EXECUTION CODE")
        
        code = f"""
# MT4/MT5 Execution Code
symbol = "EURUSD"
entry = {trade['entry_price']}
stoploss = {trade['stop_loss']}
takeprofit = {trade['take_profit']}
lots = {trade['position_size']}

# {'Buy' if trade['signal'] == 'BUY' else 'Sell'} order
OrderSend(
    Symbol=symbol,
    Cmd={'OP_BUY' if trade['signal'] == 'BUY' else 'OP_SELL'},
    Volume=lots,
    Price=entry,
    Slippage=3,
    StopLoss=stoploss,
    TakeProfit=takeprofit
)
"""
        
        st.code(code, language='python')
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **System Status:** ✅ Operational | **Data Source:** GitHub CSV  
    **Last Analysis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Records:** {len(df)} rows
    """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
