# app.py - Forex Certainty System: RAW DATA → SYSTEM DECISIONS
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .trade-buy {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #28a745;
    }
    .trade-sell {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #dc3545;
    }
    .no-trade {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #6c757d;
    }
    .system-decision {
        background: #1a1a1a;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 2px solid #ffcc00;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD RAW DATA
# ============================================================================
@st.cache_data(ttl=300)
def load_raw_data():
    """Load ONLY raw data from CSV"""
    try:
        csv_url = "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv"
        df = pd.read_csv(csv_url)
        
        # Clean column names
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Get latest raw data
        df = df.sort_values('date', ascending=True)
        latest_raw = df.iloc[-1].to_dict()
        
        st.success(f"✅ RAW DATA LOADED: {len(df)} rows, {len(df.columns)} columns")
        return df, latest_raw
        
    except Exception as e:
        st.error(f"❌ Error loading raw data: {e}")
        return None, None

# ============================================================================
# PATTERN DETECTION FROM RAW DATA
# ============================================================================
def detect_patterns_from_raw(raw_data):
    """System detects patterns from RAW data"""
    
    # Extract raw values
    current_price = float(raw_data.get('current_price', 1.1700))
    retail_long = float(raw_data.get('retail_long_percentage', 50))
    retail_short = float(raw_data.get('retail_short_percentage', 50))
    retail_long_avg = float(raw_data.get('retail_long_avg_price', current_price))
    retail_short_avg = float(raw_data.get('retail_short_avg_price', current_price))
    
    rsi_daily = float(raw_data.get('rsi_daily', 50))
    rsi_weekly = float(raw_data.get('rsi_weekly', 50))
    
    daily_range = abs(float(raw_data.get('daily_high', current_price)) - float(raw_data.get('daily_low', current_price))) * 10000
    atr = float(raw_data.get('atr_daily_pips', 80))
    
    support_1 = float(raw_data.get('support_level_1', current_price - 0.0020))
    resistance_1 = float(raw_data.get('resistance_level_1', current_price + 0.0020))
    
    market_condition = raw_data.get('market_condition', 'RANGING')
    trend_strength = float(raw_data.get('trend_strength', 0.5))
    
    # Initialize patterns
    patterns = {
        'retail_patterns': [],
        'rsi_patterns': [],
        'volatility_patterns': [],
        'market_regime': {},
        'support_resistance': {},
        'candlestick_patterns': []
    }
    
    # 1. RETAIL SENTIMENT PATTERNS
    retail_distance = (current_price - retail_long_avg) * 10000
    
    # Pattern 1: Retail crowded long (>70%)
    if retail_long > 70:
        patterns['retail_patterns'].append({
            'name': 'retail_crowded_long',
            'signal': 'SELL',
            'strength': min((retail_long - 70) / 30, 1.0),
            'reason': f'Retail {retail_long}% long (crowded trade)',
            'certainty': 0.85
        })
    
    # Pattern 2: Retail pain point (underwater)
    if abs(retail_distance) > 15:
        patterns['retail_patterns'].append({
            'name': 'retail_pain_point',
            'signal': 'SELL' if retail_distance < 0 else 'BUY',
            'strength': min(abs(retail_distance) / 50, 1.0),
            'reason': f'Retail avg position {abs(retail_distance):.1f} pips underwater',
            'certainty': 0.80
        })
    
    # 2. RSI PATTERNS
    # Daily RSI oversold/overbought
    if rsi_daily < 30:
        patterns['rsi_patterns'].append({
            'name': 'rsi_oversold_daily',
            'signal': 'BUY',
            'strength': (30 - rsi_daily) / 30,
            'reason': f'Daily RSI oversold: {rsi_daily:.1f}',
            'certainty': 0.80
        })
    elif rsi_daily > 70:
        patterns['rsi_patterns'].append({
            'name': 'rsi_overbought_daily',
            'signal': 'SELL',
            'strength': (rsi_daily - 70) / 30,
            'reason': f'Daily RSI overbought: {rsi_daily:.1f}',
            'certainty': 0.80
        })
    
    # Weekly RSI extremes
    if rsi_weekly < 35:
        patterns['rsi_patterns'].append({
            'name': 'rsi_oversold_weekly',
            'signal': 'BUY',
            'strength': (35 - rsi_weekly) / 35,
            'reason': f'Weekly RSI oversold: {rsi_weekly:.1f}',
            'certainty': 0.75
        })
    
    # 3. VOLATILITY PATTERNS
    if atr > 0:
        volatility_ratio = daily_range / atr
        
        # Low volatility (range < 0.5x ATR)
        if volatility_ratio < 0.5:
            patterns['volatility_patterns'].append({
                'name': 'low_volatility',
                'signal': 'RANGE_BOUND',
                'strength': (0.5 - volatility_ratio) * 2,
                'reason': f'Low volatility: range {daily_range:.0f}pips ({volatility_ratio:.2f}x ATR)',
                'certainty': 0.75
            })
        # High volatility (range > 1.5x ATR)
        elif volatility_ratio > 1.5:
            patterns['volatility_patterns'].append({
                'name': 'high_volatility',
                'signal': 'TRENDING',
                'strength': min((volatility_ratio - 1.5) / 2, 1.0),
                'reason': f'High volatility: range {daily_range:.0f}pips ({volatility_ratio:.2f}x ATR)',
                'certainty': 0.70
            })
    
    # 4. MARKET REGIME
    patterns['market_regime'] = {
        'regime': market_condition,
        'signal': 'BUY' if trend_strength > 0.6 and 'UP' in str(market_condition).upper() else 
                  'SELL' if trend_strength > 0.6 and 'DOWN' in str(market_condition).upper() else 
                  'RANGE_BOUND',
        'strength': trend_strength,
        'certainty': 0.70 if trend_strength > 0.6 else 0.60
    }
    
    # 5. SUPPORT/RESISTANCE
    price_to_support = (current_price - support_1) * 10000
    price_to_resistance = (resistance_1 - current_price) * 10000
    
    patterns['support_resistance'] = {
        'support_1': support_1,
        'resistance_1': resistance_1,
        'price_to_support': price_to_support,
        'price_to_resistance': price_to_resistance,
        'near_support': price_to_support < 20,  # Within 20 pips
        'near_resistance': price_to_resistance < 20,
        'signal': 'BUY' if price_to_support < 20 else 'SELL' if price_to_resistance < 20 else 'NEUTRAL'
    }
    
    # 6. NEWS IMPACT
    high_news = int(raw_data.get('high_impact_news', 0))
    if high_news > 0:
        patterns['news_impact'] = {
            'high_impact_count': high_news,
            'signal': 'AVOID' if high_news >= 2 else 'CAUTION',
            'certainty': 0.90 if high_news >= 2 else 0.75,
            'reason': f'{high_news} high-impact news events'
        }
    
    return patterns

# ============================================================================
# CERTAINTY CALCULATION
# ============================================================================
def calculate_certainty_from_patterns(patterns):
    """SYSTEM calculates certainty from detected patterns"""
    
    weights = {
        'retail_patterns': 0.25,
        'rsi_patterns': 0.20,
        'volatility_patterns': 0.15,
        'market_regime': 0.15,
        'support_resistance': 0.15,
        'news_impact': 0.10
    }
    
    category_scores = {}
    category_signals = {}
    
    # Calculate scores for each pattern category
    for category, pattern_data in patterns.items():
        if not pattern_data:
            continue
            
        if category in ['market_regime', 'support_resistance', 'news_impact']:
            # Single object categories
            if isinstance(pattern_data, dict):
                certainty = pattern_data.get('certainty', 0.5)
                strength = pattern_data.get('strength', 0.5)
                score = certainty * strength
                signal = pattern_data.get('signal', 'NEUTRAL')
                
                category_scores[category] = score
                category_signals[category] = signal
        
        elif isinstance(pattern_data, list) and pattern_data:
            # List of patterns (find strongest)
            strongest = max(pattern_data, key=lambda x: x.get('strength', 0))
            score = strongest.get('strength', 0) * strongest.get('certainty', 0.5)
            signal = strongest.get('signal', 'NEUTRAL')
            
            category_scores[category] = score
            category_signals[category] = signal
    
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
    
    # Calculate signal alignment bonus
    alignment = max(signal_counts.values()) / len(category_signals) if category_signals else 0
    if alignment >= 0.75:
        weighted_score *= 1.15  # 15% boost for good alignment
    
    return {
        'overall_certainty': min(weighted_score, 1.0),
        'primary_signal': primary_signal,
        'category_scores': category_scores,
        'category_signals': category_signals,
        'signal_alignment': alignment
    }

# ============================================================================
# TRADE GENERATION
# ============================================================================
def generate_trade_from_analysis(raw_data, patterns, certainty_data):
    """SYSTEM generates trade from analysis"""
    
    certainty = certainty_data['overall_certainty']
    signal = certainty_data['primary_signal']
    
    # NO TRADE conditions
    if certainty < 0.65:
        return {
            'signal': 'NO_TRADE',
            'type': 'SYSTEM_DECISION',
            'certainty': certainty,
            'reason': f'Insufficient certainty: {certainty:.1%}',
            'recommendation': 'Wait for better setup'
        }
    
    # Get raw data values
    current_price = float(raw_data.get('current_price', 1.1700))
    atr = float(raw_data.get('atr_daily_pips', 80))
    support_1 = float(raw_data.get('support_level_1', current_price - 0.0020))
    resistance_1 = float(raw_data.get('resistance_level_1', current_price + 0.0020))
    
    # Avoid trading if too many news events
    high_news = int(raw_data.get('high_impact_news', 0))
    if high_news >= 2:
        return {
            'signal': 'NO_TRADE',
            'type': 'SYSTEM_DECISION',
            'certainty': 0.90,
            'reason': f'{high_news} high-impact news events - Too risky',
            'recommendation': 'Avoid trading during major news'
        }
    
    # Generate trade parameters
    if signal in ['BUY', 'BULLISH']:
        # BUY trade logic
        entry_distance = atr * 0.00012  # 12% of ATR
        entry_price = max(support_1 + 0.0001, current_price - entry_distance)
        stop_loss = entry_price - (atr * 0.00018)  # 1.8x ATR
        
        risk = entry_price - stop_loss
        risk_reward = self._get_risk_reward_ratio(certainty)
        take_profit = entry_price + (risk * risk_reward)
        
        position_size = self._calculate_position_size(entry_price, stop_loss, certainty)
        
        return {
            'signal': 'BUY',
            'type': 'SYSTEM_DECISION',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': position_size,
            'stake_multiplier': self._get_stake_multiplier(certainty),
            'reason': self._generate_trade_reason(patterns, 'BUY'),
            'entry_conditions': [
                f'Price reaches {entry_price:.5f}',
                f'Stop loss at {stop_loss:.5f}',
                'No conflicting high-impact news'
            ]
        }
    
    elif signal in ['SELL', 'BEARISH']:
        # SELL trade logic
        entry_distance = atr * 0.00012
        entry_price = min(resistance_1 - 0.0001, current_price + entry_distance)
        stop_loss = entry_price + (atr * 0.00018)
        
        risk = stop_loss - entry_price
        risk_reward = self._get_risk_reward_ratio(certainty)
        take_profit = entry_price - (risk * risk_reward)
        
        position_size = self._calculate_position_size(entry_price, stop_loss, certainty)
        
        return {
            'signal': 'SELL',
            'type': 'SYSTEM_DECISION',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': position_size,
            'stake_multiplier': self._get_stake_multiplier(certainty),
            'reason': self._generate_trade_reason(patterns, 'SELL'),
            'entry_conditions': [
                f'Price reaches {entry_price:.5f}',
                f'Stop loss at {stop_loss:.5f}',
                'No conflicting high-impact news'
            ]
        }
    else:
        return {
            'signal': 'NO_TRADE',
            'type': 'SYSTEM_DECISION',
            'certainty': certainty,
            'reason': 'No clear directional signal from pattern analysis',
            'recommendation': 'Wait for pattern confirmation'
        }

def _get_risk_reward_ratio(self, certainty):
    """Get risk/reward ratio based on certainty"""
    if certainty > 0.85:
        return 2.5
    elif certainty > 0.75:
        return 2.0
    elif certainty > 0.65:
        return 1.5
    else:
        return 1.0

def _get_stake_multiplier(self, certainty):
    """Get stake multiplier based on certainty"""
    if certainty > 0.85:
        return 2.0
    elif certainty > 0.75:
        return 1.5
    elif certainty > 0.65:
        return 1.0
    else:
        return 0.5

def _calculate_position_size(self, entry, stop_loss, certainty):
    """Calculate position size"""
    base_risk = 100  # $100 for 1% risk on $10,000
    
    if certainty > 0.85:
        risk_multiplier = 2.0
    elif certainty > 0.75:
        risk_multiplier = 1.5
    elif certainty > 0.65:
        risk_multiplier = 1.0
    else:
        risk_multiplier = 0.5
    
    risk_amount = base_risk * risk_multiplier
    risk_pips = abs(entry - stop_loss) * 10000
    
    if risk_pips == 0:
        return 0
    
    position_size = risk_amount / (risk_pips * 10)  # $10 per pip for EURUSD
    return round(position_size, 2)

def _generate_trade_reason(self, patterns, signal):
    """Generate trade reason from patterns"""
    reasons = []
    
    for category, pattern_data in patterns.items():
        if not pattern_data:
            continue
            
        if isinstance(pattern_data, dict) and pattern_data.get('signal') == signal:
            reasons.append(pattern_data.get('reason', ''))
        elif isinstance(pattern_data, list):
            for pattern in pattern_data:
                if pattern.get('signal') == signal:
                    reasons.append(pattern.get('reason', ''))
    
    return " | ".join(reasons[:3]) if reasons else "Pattern analysis indicates opportunity"

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.set_page_config(layout="wide")
    
    # Title
    st.title("🎯 FOREX CERTAINTY SYSTEM")
    st.markdown("**RAW DATA → SYSTEM ANALYSIS → TRADE DECISIONS**")
    
    # Load raw data
    df, latest_raw = load_raw_data()
    
    if latest_raw is None:
        st.error("Failed to load raw data")
        return
    
    # Display current date
    current_date = latest_raw.get('date', datetime.now().date())
    if hasattr(current_date, 'strftime'):
        display_date = current_date.strftime('%Y-%m-%d')
    else:
        display_date = str(current_date)
    
    st.markdown(f"### 📅 Analysis Date: **{display_date}**")
    
    # Show raw data summary
    with st.expander("📊 RAW DATA SUMMARY"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Price", f"{latest_raw.get('current_price', 0):.5f}")
        with col2:
            st.metric("Retail Long", f"{latest_raw.get('retail_long_percentage', 0)}%")
        with col3:
            st.metric("RSI Daily", f"{latest_raw.get('rsi_daily', 0):.1f}")
        with col4:
            st.metric("High News", latest_raw.get('high_impact_news', 0))
    
    # SYSTEM ANALYSIS SECTION
    st.markdown("### 🔍 SYSTEM ANALYSIS (Processing Raw Data...)")
    
    with st.spinner("Detecting patterns from raw data..."):
        # 1. System detects patterns
        patterns = detect_patterns_from_raw(latest_raw)
        
        # 2. System calculates certainty
        certainty_data = calculate_certainty_from_patterns(patterns)
        
        # 3. System generates trade
        trade = generate_trade_from_analysis(latest_raw, patterns, certainty_data)
    
    # SYSTEM DECISION
    st.markdown("### 🎯 SYSTEM DECISION")
    
    certainty = certainty_data['overall_certainty']
    
    # Display certainty score
    col1, col2 = st.columns([3, 1])
    
    with col1:
        certainty_color = "🟢" if certainty > 0.80 else "🟡" if certainty > 0.65 else "🔴"
        st.markdown(f"**Certainty Score:** {certainty_color} **{certainty:.1%}**")
        st.progress(certainty)
    
    with col2:
        st.metric("Signal Alignment", f"{certainty_data['signal_alignment']:.0%}")
    
    # Display trade decision
    if trade['signal'] == 'NO_TRADE':
        st.markdown(f"""
        <div class="system-decision">
            <h3>🚫 SYSTEM DECISION: NO TRADE</h3>
            <p><strong>Reason:</strong> {trade['reason']}</p>
            <p><strong>Certainty:</strong> {trade['certainty']:.1%}</p>
            <p><strong>Recommendation:</strong> {trade.get('recommendation', 'Wait')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        trade_class = "trade-buy" if trade['signal'] == 'BUY' else "trade-sell"
        
        st.markdown(f"""
        <div class="{trade_class}">
            <h3>📈 SYSTEM DECISION: {trade['signal']} EURUSD</h3>
            <p><strong>Entry Price:</strong> {trade['entry_price']}</p>
            <p><strong>Stop Loss:</strong> {trade['stop_loss']}</p>
            <p><strong>Take Profit:</strong> {trade['take_profit']}</p>
            <p><strong>Risk/Reward:</strong> 1:{trade['risk_reward']:.1f}</p>
            <p><strong>Position Size:</strong> {trade['position_size']} lots</p>
            <p><strong>Stake Multiplier:</strong> {trade['stake_multiplier']}x</p>
            <p><strong>Certainty:</strong> {trade['certainty']:.1%}</p>
            <p><strong>Reason:</strong> {trade['reason']}</p>
            <p><strong>Entry Conditions:</strong></p>
            <ul>
            {''.join([f'<li>{cond}</li>' for cond in trade.get('entry_conditions', [])])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Execution code
        st.markdown("### 💻 EXECUTION CODE")
        exec_code = f"""
# MT4/MT5 Execution Code (SYSTEM GENERATED)
symbol = "EURUSD"
entry = {trade['entry_price']}
stoploss = {trade['stop_loss']}
takeprofit = {trade['take_profit']}
lots = {trade['position_size']}

# {'BUY' if trade['signal'] == 'BUY' else 'SELL'} order
OrderSend(
    Symbol=symbol,
    Cmd={'OP_BUY' if trade['signal'] == 'BUY' else 'OP_SELL'},
    Volume=lots,
    Price=entry,
    Slippage=3,
    StopLoss=stoploss,
    TakeProfit=takeprofit,
    Comment="Forex Certainty System"
)
"""
        st.code(exec_code, language='python')
    
    # PATTERN ANALYSIS
    st.markdown("### 🔍 PATTERN DETECTION RESULTS")
    
    pattern_cols = st.columns(4)
    
    pattern_categories = [
        ('Retail Patterns', 'retail_patterns'),
        ('RSI Patterns', 'rsi_patterns'),
        ('Volatility', 'volatility_patterns'),
        ('Market Regime', 'market_regime')
    ]
    
    for idx, (title, key) in enumerate(pattern_categories):
        with pattern_cols[idx]:
            st.markdown(f"**{title}**")
            if key in patterns:
                data = patterns[key]
                if isinstance(data, list) and data:
                    for pattern in data:
                        st.markdown(f"<div class='metric-card'>{pattern.get('reason', '')}</div>", unsafe_allow_html=True)
                elif isinstance(data, dict) and data:
                    st.markdown(f"<div class='metric-card'>{data.get('regime', 'Unknown')} (Strength: {data.get('strength', 0):.2f})</div>", unsafe_allow_html=True)
                else:
                    st.info("No patterns")
            else:
                st.info("No data")
    
    # RAW DATA VIEWER
    with st.expander("📁 VIEW RAW DATA"):
        st.write("**Latest Raw Data Row:**")
        st.json({k: v for k, v in latest_raw.items() if not pd.isna(v)})
        
        st.write("**Full DataFrame:**")
        st.dataframe(df)
    
    # SYSTEM LOG
    st.markdown("### 📝 SYSTEM LOG")
    
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_date': display_date,
        'patterns_detected': sum(len(v) if isinstance(v, list) else 1 for v in patterns.values() if v),
        'certainty_score': certainty,
        'system_decision': trade['signal'],
        'trade_reason': trade.get('reason', ''),
        'raw_data_points': len([v for v in latest_raw.values() if not pd.isna(v)])
    }
    
    st.json(log_data)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **SYSTEM STATUS:** ✅ Processing Raw Data | **DECISIONS:** System-Generated  
    **DATA SOURCE:** forex_certainty_data.csv | **RECORDS:** {len(df)} rows
    """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
