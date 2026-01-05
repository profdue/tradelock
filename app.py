# app.py - Forex Certainty System with Future Projections
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any

# ============================================================================
# PAGE CONFIG
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
    .projection-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #ffcc00;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .pattern-item {
        background: #e9ecef;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 5px;
        border-left: 3px solid #1E88E5;
    }
    .header-section {
        background: linear-gradient(90deg, #1E88E5, #4A00E0);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADER
# ============================================================================
@st.cache_data(ttl=300)
def load_and_validate_csv():
    """Load and validate CSV with dynamic column handling"""
    try:
        # Load your CSV
        csv_url = "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv"
        df = pd.read_csv(csv_url)
        
        # Show what we loaded
        st.info(f"📊 Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        
        # Clean column names (handle any format)
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
        
        # Convert date - find any date column
        date_column = None
        for col in df.columns:
            if 'date' in col.lower():
                date_column = col
                break
        
        if date_column:
            df['date'] = pd.to_datetime(df[date_column], errors='coerce')
        else:
            # If no date column, create one
            df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        
        # Convert numeric columns dynamically
        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Sort by date
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        # Get latest data
        latest_raw = df.iloc[-1].to_dict()
        
        # Show column mapping for debugging
        with st.expander("🔍 Column Mapping"):
            st.write("Available columns in your CSV:")
            for i, col in enumerate(df.columns, 1):
                st.write(f"{i:2}. {col}")
        
        return df, latest_raw
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None, None

# ============================================================================
# PATTERN DETECTOR
# ============================================================================
class PatternDetector:
    def __init__(self, df):
        self.df = df
    
    def detect_patterns(self, data: Dict) -> Dict:
        """Detect patterns from raw data using dynamic column access"""
        patterns = {
            'retail_patterns': [],
            'rsi_patterns': [],
            'volatility_patterns': [],
            'market_regime': {},
            'support_resistance': {},
            'candlestick_patterns': []
        }
        
        # Safely get values with defaults
        def get_value(key, default=0):
            return float(data.get(key, default)) if pd.notna(data.get(key)) else default
        
        # 1. RETAIL SENTIMENT PATTERNS
        retail_long = get_value('retail_long_percentage', 50)
        retail_long_avg = get_value('retail_long_avg_price', get_value('current_price', 1.1700))
        current_price = get_value('current_price', 1.1700)
        
        # Retail distance from average
        retail_distance_pips = (current_price - retail_long_avg) * 10000
        
        # Crowded long pattern
        if retail_long > 70:
            patterns['retail_patterns'].append({
                'name': 'retail_crowded_long',
                'signal': 'SELL',
                'strength': min((retail_long - 70) / 30, 1.0),
                'reason': f'Retail {retail_long:.0f}% long (crowded)',
                'certainty': 0.85
            })
        
        # Retail pain point
        if abs(retail_distance_pips) > 15:
            patterns['retail_patterns'].append({
                'name': 'retail_pain_point',
                'signal': 'SELL' if retail_distance_pips < 0 else 'BUY',
                'strength': min(abs(retail_distance_pips) / 50, 1.0),
                'reason': f'Retail avg position {abs(retail_distance_pips):.1f} pips underwater',
                'certainty': 0.80
            })
        
        # 2. RSI PATTERNS
        rsi_daily = get_value('rsi_daily', 50)
        
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
        
        # 3. VOLATILITY PATTERNS
        daily_high = get_value('daily_high', current_price)
        daily_low = get_value('daily_low', current_price)
        daily_range = (daily_high - daily_low) * 10000
        atr = get_value('atr_daily_pips', 80)
        
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
        
        # 4. MARKET REGIME
        market_condition = data.get('market_condition', 'RANGING')
        if isinstance(market_condition, str):
            market_condition = market_condition.upper()
        else:
            market_condition = 'RANGING'
        
        trend_strength = get_value('trend_strength', 0.5)
        
        patterns['market_regime'] = {
            'regime': market_condition,
            'signal': 'BUY' if trend_strength > 0.6 and 'UP' in str(market_condition) else 
                     'SELL' if trend_strength > 0.6 and 'DOWN' in str(market_condition) else 
                     'RANGE_BOUND',
            'strength': trend_strength,
            'certainty': 0.70 if trend_strength > 0.6 else 0.60
        }
        
        # 5. SUPPORT/RESISTANCE
        support_1 = get_value('support_level_1', current_price - 0.0020)
        resistance_1 = get_value('resistance_level_1', current_price + 0.0020)
        
        price_to_support = (current_price - support_1) * 10000
        price_to_resistance = (resistance_1 - current_price) * 10000
        
        patterns['support_resistance'] = {
            'support_1': support_1,
            'resistance_1': resistance_1,
            'price_to_support': price_to_support,
            'price_to_resistance': price_to_resistance,
            'near_support': price_to_support < 20,
            'near_resistance': price_to_resistance < 20,
            'signal': 'BUY' if price_to_support < 20 else 'SELL' if price_to_resistance < 20 else 'NEUTRAL'
        }
        
        # 6. NEWS IMPACT
        high_news = int(get_value('high_impact_news', 0))
        if high_news > 0:
            patterns['news_impact'] = {
                'high_impact_count': high_news,
                'signal': 'AVOID' if high_news >= 2 else 'CAUTION',
                'certainty': 0.90 if high_news >= 2 else 0.75,
                'reason': f'{high_news} high-impact news events'
            }
        
        # 7. CANDLESTICK PATTERNS
        weekly_patterns = data.get('weekly_patterns_raw', '')
        daily_patterns = data.get('daily_patterns_raw', '')
        
        if weekly_patterns:
            patterns['candlestick_patterns'].append({
                'timeframe': 'WEEKLY',
                'patterns': weekly_patterns,
                'bearish_bias': any(bearish in weekly_patterns.lower() for bearish in ['hanging', 'shooting', 'evening']),
                'bullish_bias': any(bullish in weekly_patterns.lower() for bullish in ['hammer', 'morning', 'engulfing'])
            })
        
        return patterns

# ============================================================================
# CERTAINTY CALCULATOR
# ============================================================================
class CertaintyCalculator:
    @staticmethod
    def calculate(patterns: Dict) -> Dict:
        """Calculate certainty scores from patterns"""
        
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
        
        # Calculate scores for each category
        for category, pattern_data in patterns.items():
            if not pattern_data:
                category_scores[category] = 0.5
                category_signals[category] = 'NEUTRAL'
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
                # List of patterns
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
        
        # Signal alignment bonus
        alignment = max(signal_counts.values()) / len(category_signals) if category_signals else 0
        if alignment >= 0.75:
            weighted_score = min(weighted_score * 1.15, 1.0)  # 15% boost
        
        return {
            'overall_certainty': weighted_score,
            'primary_signal': primary_signal,
            'category_scores': category_scores,
            'category_signals': category_signals,
            'signal_alignment': alignment
        }

# ============================================================================
# TRADE GENERATOR
# ============================================================================
class TradeGenerator:
    @staticmethod
    def generate(data: Dict, patterns: Dict, certainty_data: Dict) -> Dict:
        """Generate trade based on analysis"""
        
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
        
        # Get values safely
        def get_price(key, default):
            val = data.get(key)
            return float(val) if pd.notna(val) else default
        
        current_price = get_price('current_price', 1.1700)
        atr = get_price('atr_daily_pips', 80)
        support_1 = get_price('support_level_1', current_price - 0.0020)
        resistance_1 = get_price('resistance_level_1', current_price + 0.0020)
        
        # Check news
        high_news = int(get_price('high_impact_news', 0))
        if high_news >= 2:
            return {
                'signal': 'NO_TRADE',
                'type': 'SYSTEM_DECISION',
                'certainty': 0.90,
                'reason': f'{high_news} high-impact news events - Avoid trading',
                'recommendation': 'Wait for news to pass'
            }
        
        if signal in ['BUY', 'BULLISH']:
            return TradeGenerator._generate_buy_trade(
                current_price, atr, support_1, certainty, patterns
            )
        elif signal in ['SELL', 'BEARISH']:
            return TradeGenerator._generate_sell_trade(
                current_price, atr, resistance_1, certainty, patterns
            )
        else:
            return {
                'signal': 'NO_TRADE',
                'type': 'SYSTEM_DECISION',
                'certainty': certainty,
                'reason': 'No clear directional signal',
                'recommendation': 'Wait for pattern confirmation'
            }
    
    @staticmethod
    def _generate_buy_trade(current_price, atr, support, certainty, patterns):
        """Generate buy trade parameters"""
        entry_distance = atr * 0.00012
        entry_price = max(support + 0.0001, current_price - entry_distance)
        stop_loss = entry_price - (atr * 0.00018)
        
        risk = entry_price - stop_loss
        risk_reward = TradeGenerator._get_risk_reward(certainty)
        take_profit = entry_price + (risk * risk_reward)
        
        position_size = TradeGenerator._calculate_position_size(entry_price, stop_loss, certainty)
        
        return {
            'signal': 'BUY',
            'type': 'SYSTEM_DECISION',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': position_size,
            'stake_multiplier': TradeGenerator._get_stake_multiplier(certainty),
            'reason': TradeGenerator._generate_reason(patterns, 'BUY'),
            'entry_conditions': [
                f'Price reaches {entry_price:.5f}',
                f'Stop loss at {stop_loss:.5f}',
                'No conflicting high-impact news'
            ]
        }
    
    @staticmethod
    def _generate_sell_trade(current_price, atr, resistance, certainty, patterns):
        """Generate sell trade parameters"""
        entry_distance = atr * 0.00012
        entry_price = min(resistance - 0.0001, current_price + entry_distance)
        stop_loss = entry_price + (atr * 0.00018)
        
        risk = stop_loss - entry_price
        risk_reward = TradeGenerator._get_risk_reward(certainty)
        take_profit = entry_price - (risk * risk_reward)
        
        position_size = TradeGenerator._calculate_position_size(entry_price, stop_loss, certainty)
        
        return {
            'signal': 'SELL',
            'type': 'SYSTEM_DECISION',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': position_size,
            'stake_multiplier': TradeGenerator._get_stake_multiplier(certainty),
            'reason': TradeGenerator._generate_reason(patterns, 'SELL'),
            'entry_conditions': [
                f'Price reaches {entry_price:.5f}',
                f'Stop loss at {stop_loss:.5f}',
                'No conflicting high-impact news'
            ]
        }
    
    @staticmethod
    def _get_risk_reward(certainty):
        if certainty > 0.85: return 2.5
        elif certainty > 0.75: return 2.0
        elif certainty > 0.65: return 1.5
        else: return 1.0
    
    @staticmethod
    def _get_stake_multiplier(certainty):
        if certainty > 0.85: return 2.0
        elif certainty > 0.75: return 1.5
        elif certainty > 0.65: return 1.0
        else: return 0.5
    
    @staticmethod
    def _calculate_position_size(entry, stop_loss, certainty):
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
        
        position_size = risk_amount / (risk_pips * 10)
        return round(position_size, 2)
    
    @staticmethod
    def _generate_reason(patterns, signal):
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
# FUTURE PROJECTOR
# ============================================================================
class FutureProjector:
    def __init__(self, historical_df):
        self.df = historical_df
    
    def project_next_day(self, today_data: Dict, patterns: Dict) -> Dict:
        """Project tomorrow's likely scenarios"""
        projections = []
        
        # Safely get values
        def get_val(key, default):
            val = today_data.get(key)
            return float(val) if pd.notna(val) else default
        
        # 1. MARKET REGIME PROJECTION
        market_condition = today_data.get('market_condition', 'RANGING')
        if isinstance(market_condition, str) and 'RANG' in market_condition.upper():
            projections.append({
                'type': 'MARKET_REGIME',
                'prediction': 'Continue RANGING',
                'probability': 75,
                'reason': f'Current regime: {market_condition}, ADX: {get_val("adx_daily", 20):.1f} (<25)',
                'trading_implication': 'Trade range extremes, buy support/sell resistance',
                'expected_range': '30-50 pips'
            })
        
        # 2. VOLATILITY PROJECTION
        daily_high = get_val('daily_high', get_val('current_price', 1.1700))
        daily_low = get_val('daily_low', get_val('current_price', 1.1700))
        daily_range = (daily_high - daily_low) * 10000
        atr = get_val('atr_daily_pips', 80)
        
        if atr > 0:
            vol_ratio = daily_range / atr
            if vol_ratio < 0.5:
                # Count compression days
                compression_days = self._count_compression_days()
                if compression_days >= 2:
                    projections.append({
                        'type': 'VOLATILITY',
                        'prediction': 'Volatility expansion likely',
                        'probability': 80,
                        'reason': f'Compression: {daily_range:.0f}pips ({vol_ratio:.2f}x ATR) for {compression_days} days',
                        'trading_implication': 'Prepare for breakout, widen stops',
                        'expected_range': '60-80 pips if expansion occurs'
                    })
        
        # 3. RSI PROJECTION
        rsi_daily = get_val('rsi_daily', 50)
        if rsi_daily < 40:
            momentum = self._calculate_rsi_momentum()
            if momentum < 0:  # Declining
                days_to_30 = (rsi_daily - 30) / abs(momentum) if momentum != 0 else 99
                if days_to_30 < 5:
                    projections.append({
                        'type': 'RSI',
                        'prediction': 'Approaching oversold',
                        'probability': 70,
                        'reason': f'RSI: {rsi_daily:.1f}, momentum: {momentum:.1f} points/day',
                        'trading_implication': 'Prepare BUY setup near support',
                        'timing': f'Could reach oversold in {max(1, int(days_to_30))} days'
                    })
        
        # 4. RETAIL PROJECTION
        retail_long = get_val('retail_long_percentage', 50)
        if retail_long > 65:
            projections.append({
                'type': 'RETAIL',
                'prediction': 'Approaching crowded long',
                'probability': 85,
                'reason': f'Retail {retail_long:.0f}% long (>70% triggers reversal)',
                'trading_implication': 'Watch for reversal signals, prepare SELL setups',
                'threshold': 'Reversal likely if hits 70%+'
            })
        
        # 5. SESSION PROJECTION (Fixed schedule)
        projections.append({
            'type': 'SESSION',
            'prediction': 'London-NY overlap volatility',
            'probability': 100,
            'reason': 'Fixed market schedule',
            'trading_implication': 'Highest probability trades during 13:00-16:00 GMT',
            'timing': 'Daily 13:00-16:00 GMT'
        })
        
        return projections
    
    def _count_compression_days(self) -> int:
        """Count consecutive low volatility days"""
        if len(self.df) < 2:
            return 0
        
        count = 0
        for i in range(len(self.df)-1, max(0, len(self.df)-6)-1, -1):
            if 'current_price' in self.df.columns and 'atr_daily_pips' in self.df.columns:
                try:
                    price = float(self.df.iloc[i]['current_price'])
                    atr = float(self.df.iloc[i]['atr_daily_pips'])
                    if atr > 0:
                        # Estimate range (simplified)
                        range_est = atr * 0.3  # Assume 30% of ATR
                        vol_ratio = range_est / atr
                        if vol_ratio < 0.5:
                            count += 1
                        else:
                            break
                except:
                    break
        return count
    
    def _calculate_rsi_momentum(self) -> float:
        """Calculate RSI momentum from recent data"""
        if len(self.df) < 3 or 'rsi_daily' not in self.df.columns:
            return 0
        
        try:
            recent_rsi = []
            for i in range(max(0, len(self.df)-4), len(self.df)):
                val = self.df.iloc[i]['rsi_daily']
                if pd.notna(val):
                    recent_rsi.append(float(val))
            
            if len(recent_rsi) > 1:
                return (recent_rsi[-1] - recent_rsi[0]) / (len(recent_rsi) - 1)
        except:
            pass
        
        return 0

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="header-section">
        <h1>🎯 FOREX CERTAINTY SYSTEM</h1>
        <p>RAW DATA → SYSTEM ANALYSIS → TRADE DECISIONS → FUTURE PROJECTIONS</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("📊 Loading and analyzing CSV data..."):
        df, latest_raw = load_and_validate_csv()
        
        if df is None or latest_raw is None:
            st.error("Failed to load data. Please check your CSV.")
            return
        
        # Initialize components
        detector = PatternDetector(df)
        calculator = CertaintyCalculator()
        generator = TradeGenerator()
        projector = FutureProjector(df)
        
        # Process data
        patterns = detector.detect_patterns(latest_raw)
        certainty_data = calculator.calculate(patterns)
        trade = generator.generate(latest_raw, patterns, certainty_data)
        projections = projector.project_next_day(latest_raw, patterns)
    
    # Current Date
    current_date = latest_raw.get('date', datetime.now().date())
    if hasattr(current_date, 'strftime'):
        display_date = current_date.strftime('%Y-%m-%d')
    else:
        display_date = str(current_date)
    
    st.markdown(f"### 📅 Analysis Date: **{display_date}**")
    
    # Key Metrics
    st.markdown("### 📊 KEY METRICS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = latest_raw.get('current_price', 0)
        st.metric("Price", f"{price:.5f}")
    
    with col2:
        retail = latest_raw.get('retail_long_percentage', 50)
        st.metric("Retail Long", f"{retail:.0f}%")
    
    with col3:
        rsi = latest_raw.get('rsi_daily', 50)
        status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        st.metric("RSI Daily", f"{rsi:.1f}", status)
    
    with col4:
        regime = latest_raw.get('market_condition', 'RANGING')
        st.metric("Market Regime", regime)
    
    # Certainty Score
    certainty = certainty_data['overall_certainty']
    st.markdown(f"### 🎯 SYSTEM CERTAINTY: **{certainty:.1%}**")
    
    if certainty > 0.80:
        color = "green"
    elif certainty > 0.65:
        color = "orange"
    else:
        color = "red"
    
    st.progress(certainty)
    
    # Trade Decision
    st.markdown("### 📈 TRADE DECISION")
    
    if trade['signal'] == 'NO_TRADE':
        st.markdown(f"""
        <div class="no-trade">
            <h3>🚫 NO TRADE</h3>
            <p><strong>Reason:</strong> {trade['reason']}</p>
            <p><strong>Certainty:</strong> {trade['certainty']:.1%}</p>
            <p><strong>Recommendation:</strong> {trade.get('recommendation', 'Wait')}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        trade_class = "trade-buy" if trade['signal'] == 'BUY' else "trade-sell"
        
        st.markdown(f"""
        <div class="{trade_class}">
            <h3>📊 {trade['signal']} EURUSD</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 15px;">
                <div><strong>Entry:</strong> {trade['entry_price']}</div>
                <div><strong>Stop Loss:</strong> {trade['stop_loss']}</div>
                <div><strong>Take Profit:</strong> {trade['take_profit']}</div>
                <div><strong>Risk/Reward:</strong> 1:{trade['risk_reward']:.1f}</div>
                <div><strong>Position Size:</strong> {trade['position_size']} lots</div>
                <div><strong>Stake Multiplier:</strong> {trade['stake_multiplier']}x</div>
            </div>
            <p style="margin-top: 15px;"><strong>Reason:</strong> {trade['reason']}</p>
            <p><strong>Entry Conditions:</strong> {', '.join(trade.get('entry_conditions', []))}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Execution Code
        with st.expander("💻 Execution Code"):
            code = f"""
# MT4/MT5 Execution Code
symbol = "EURUSD"
entry = {trade['entry_price']}
stoploss = {trade['stop_loss']}
takeprofit = {trade['take_profit']}
lots = {trade['position_size']}

# {trade['signal']} Order
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
            st.code(code, language='python')
    
    # Future Projections
    st.markdown("### 🔮 TOMORROW'S PROJECTIONS")
    
    if projections:
        for proj in projections:
            prob_color = "🟢" if proj['probability'] >= 80 else "🟡" if proj['probability'] >= 60 else "🔴"
            
            st.markdown(f"""
            <div class="projection-card">
                <h4>{prob_color} {proj['type']}: {proj['prediction']} ({proj['probability']}%)</h4>
                <p><strong>Reason:</strong> {proj['reason']}</p>
                <p><strong>Trading Implication:</strong> {proj.get('trading_implication', 'N/A')}</p>
                {f"<p><strong>Expected Range/Timing:</strong> {proj.get('expected_range', proj.get('timing', 'N/A'))}</p>" if proj.get('expected_range') or proj.get('timing') else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No clear projections for tomorrow based on current data.")
    
    # Pattern Analysis
    st.markdown("### 🔍 PATTERN ANALYSIS")
    
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
                    for pattern in data[:3]:  # Show top 3
                        st.markdown(f"<div class='pattern-item'>{pattern.get('reason', '')}</div>", unsafe_allow_html=True)
                elif isinstance(data, dict) and data:
                    regime = data.get('regime', 'Unknown')
                    strength = data.get('strength', 0)
                    st.markdown(f"<div class='pattern-item'>{regime} (Strength: {strength:.2f})</div>", unsafe_allow_html=True)
            else:
                st.info("No patterns")
    
    # Debug View
    with st.expander("🔧 Debug View"):
        tab1, tab2, tab3 = st.tabs(["Raw Data", "Patterns", "Analysis"])
        
        with tab1:
            st.write("Latest Raw Data:")
            st.json({k: v for k, v in latest_raw.items() if pd.notna(v)})
        
        with tab2:
            st.write("Detected Patterns:")
            st.json(patterns)
        
        with tab3:
            st.write("Certainty Analysis:")
            st.json(certainty_data)
            st.write("Trade Decision:")
            st.json(trade)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    **System Status:** ✅ Operational | **Records:** {len(df)} rows | **Last Update:** {datetime.now().strftime('%H:%M:%S')}
    """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
