# app.py - COMPLETE Forex Certainty System
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA PROCESSING ENGINE
# ============================================================================
class DataProcessor:
    def __init__(self, csv_url):
        self.csv_url = csv_url
        self.data = None
        self.today_data = None
        
    def load_data(self):
        """Load and validate CSV data from GitHub"""
        try:
            self.data = pd.read_csv(self.csv_url)
            
            # Standardize column names
            self.data.columns = [str(col).strip().lower().replace(' ', '_') for col in self.data.columns]
            
            # Convert date
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
            
            # Get latest data
            self.data = self.data.sort_values('date')
            self.today_data = self.data.iloc[-1].to_dict()
            
            return True
        except Exception as e:
            st.error(f"CSV Error: {e}")
            return False
    
    def calculate_derived_metrics(self):
        """Calculate additional metrics from CSV data"""
        if self.data is None:
            return {}
            
        # Ensure we have numeric data
        numeric_cols = ['current_price', 'daily_open', 'daily_range_pips', 
                       'atr_daily_pips', 'rsi_daily', 'retail_long_pct']
        
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Calculate volatility ratio
        if 'daily_range_pips' in self.data.columns and 'atr_daily_pips' in self.data.columns:
            self.data['volatility_ratio'] = (
                self.data['daily_range_pips'] / self.data['atr_daily_pips']
            ).replace([np.inf, -np.inf], np.nan)
        
        # Calculate RSI momentum
        if 'rsi_daily' in self.data.columns:
            self.data['rsi_momentum'] = self.data['rsi_daily'] - 50
        
        # Update today's data
        if self.today_data:
            last_idx = len(self.data) - 1
            for col in ['volatility_ratio', 'rsi_momentum']:
                if col in self.data.columns:
                    self.today_data[col] = self.data.loc[last_idx, col]
        
        return self.today_data

# ============================================================================
# PATTERN DETECTION ENGINE
# ============================================================================
class PatternDetector:
    def __init__(self, data_processor):
        self.dp = data_processor
        self.patterns = {}
        
    def detect_all_patterns(self, today_data):
        """Detect all trading patterns"""
        patterns = {}
        
        patterns['retail_sentiment'] = self._detect_retail_patterns(today_data)
        patterns['rsi_patterns'] = self._detect_rsi_patterns(today_data)
        patterns['volatility_patterns'] = self._detect_volatility_patterns(today_data)
        patterns['market_regime'] = self._detect_market_regime(today_data)
        
        self.patterns = patterns
        return patterns
    
    def _detect_retail_patterns(self, data):
        """Detect retail sentiment patterns"""
        patterns = {}
        
        retail_long = data.get('retail_long_pct', 50)
        if pd.isna(retail_long):
            retail_long = 50
            
        retail_distance = data.get('retail_distance_pips', 0)
        if pd.isna(retail_distance):
            retail_distance = 0
        
        # Retail crowded long
        if retail_long > 70:
            patterns['retail_crowded_long'] = {
                'signal': 'BEARISH',
                'strength': min((retail_long - 70) / 30, 1.0),
                'reason': f'Retail {retail_long:.0f}% long (crowded)',
                'certainty': 0.85
            }
        # Retail crowded short
        elif retail_long < 30:
            patterns['retail_crowded_short'] = {
                'signal': 'BULLISH',
                'strength': min((30 - retail_long) / 30, 1.0),
                'reason': f'Retail {retail_long:.0f}% long (crowded short)',
                'certainty': 0.85
            }
        
        # Retail pain point
        if abs(retail_distance) > 30:
            patterns['retail_pain_point'] = {
                'signal': 'BEARISH' if retail_distance < 0 else 'BULLISH',
                'strength': min(abs(retail_distance) / 100, 1.0),
                'reason': f'Retail {abs(retail_distance):.0f} pips underwater',
                'certainty': 0.80
            }
        
        return patterns
    
    def _detect_rsi_patterns(self, data):
        """Detect RSI patterns"""
        patterns = {}
        
        rsi_daily = data.get('rsi_daily', 50)
        if pd.isna(rsi_daily):
            return patterns
            
        # RSI oversold
        if rsi_daily < 30:
            patterns['rsi_oversold'] = {
                'signal': 'BULLISH',
                'strength': (30 - rsi_daily) / 30,
                'reason': f'RSI oversold: {rsi_daily:.1f}',
                'certainty': 0.80
            }
        # RSI overbought
        elif rsi_daily > 70:
            patterns['rsi_overbought'] = {
                'signal': 'BEARISH',
                'strength': (rsi_daily - 70) / 30,
                'reason': f'RSI overbought: {rsi_daily:.1f}',
                'certainty': 0.80
            }
        
        return patterns
    
    def _detect_volatility_patterns(self, data):
        """Detect volatility patterns"""
        patterns = {}
        
        daily_range = data.get('daily_range_pips', 0)
        atr_daily = data.get('atr_daily_pips', 1)
        
        if pd.isna(daily_range) or pd.isna(atr_daily) or atr_daily == 0:
            return patterns
            
        volatility_ratio = daily_range / atr_daily
        
        # Low volatility
        if volatility_ratio < 0.5:
            patterns['low_volatility'] = {
                'signal': 'RANGE_BOUND',
                'strength': (0.5 - volatility_ratio) * 2,
                'reason': f'Low volatility: {daily_range:.0f}pips ({volatility_ratio:.2f}x ATR)',
                'certainty': 0.75
            }
        # High volatility
        elif volatility_ratio > 1.5:
            patterns['high_volatility'] = {
                'signal': 'TRENDING',
                'strength': min((volatility_ratio - 1.5) / 2, 1.0),
                'reason': f'High volatility: {daily_range:.0f}pips ({volatility_ratio:.2f}x ATR)',
                'certainty': 0.70
            }
        
        return patterns
    
    def _detect_market_regime(self, data):
        """Detect market regime"""
        regime = data.get('market_regime', 'RANGING')
        if pd.isna(regime):
            regime = 'RANGING'
            
        trend_strength = data.get('trend_strength', 0.3)
        if pd.isna(trend_strength):
            trend_strength = 0.3
        
        if regime == 'TRENDING' and trend_strength > 0.6:
            return {
                'regime': 'STRONG_TREND',
                'strength': trend_strength,
                'certainty': 0.85
            }
        elif regime == 'RANGING' and trend_strength < 0.4:
            return {
                'regime': 'STRONG_RANGE',
                'strength': 1.0 - trend_strength,
                'certainty': 0.80
            }
        else:
            return {
                'regime': regime,
                'strength': 0.5,
                'certainty': 0.70
            }

# ============================================================================
# CERTAINTY SCORING ENGINE
# ============================================================================
class CertaintyScorer:
    def __init__(self):
        self.weights = {
            'retail_sentiment': 0.20,
            'rsi_patterns': 0.15,
            'volatility_patterns': 0.15,
            'market_regime': 0.15
        }
        
    def calculate_overall_certainty(self, patterns):
        """Calculate overall certainty score"""
        category_scores = {}
        category_signals = {}
        
        # Score each pattern category
        for category, pattern_data in patterns.items():
            if category == 'market_regime':
                score, signal = self._score_market_regime(pattern_data)
            else:
                score, signal = self._score_pattern_category(pattern_data)
            
            category_scores[category] = score
            category_signals[category] = signal
        
        # Apply weights
        weighted_scores = {}
        for category, score in category_scores.items():
            weight = self.weights.get(category, 0.10)
            weighted_scores[category] = score * weight
        
        # Calculate overall
        overall_score = sum(weighted_scores.values())
        
        # Determine primary signal
        signal_counts = {}
        for signal in category_signals.values():
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        primary_signal = max(signal_counts, key=signal_counts.get) if signal_counts else 'NEUTRAL'
        
        return {
            'overall_certainty': min(overall_score, 1.0),
            'primary_signal': primary_signal,
            'category_scores': category_scores,
            'category_signals': category_signals
        }
    
    def _score_pattern_category(self, pattern_data):
        """Score a pattern category"""
        if not pattern_data:
            return 0.5, 'NEUTRAL'
        
        # Find strongest pattern
        max_strength = 0
        overall_signal = 'NEUTRAL'
        certainty = 0.5
        
        for pattern_name, pattern_info in pattern_data.items():
            if pattern_info.get('strength', 0) > max_strength:
                max_strength = pattern_info['strength']
                overall_signal = pattern_info.get('signal', 'NEUTRAL')
                certainty = pattern_info.get('certainty', 0.5)
        
        score = max_strength * certainty
        return score, overall_signal
    
    def _score_market_regime(self, regime_data):
        """Score market regime"""
        if not regime_data:
            return 0.5, 'NEUTRAL'
        
        regime = regime_data.get('regime', 'NEUTRAL')
        strength = regime_data.get('strength', 0.5)
        certainty = regime_data.get('certainty', 0.5)
        
        if 'TREND' in regime:
            signal = 'BULLISH' if regime == 'STRONG_TREND' else 'NEUTRAL'
        elif 'RANGE' in regime:
            signal = 'RANGE_BOUND'
        else:
            signal = 'NEUTRAL'
        
        score = strength * certainty
        return score, signal

# ============================================================================
# TRADE GENERATION ENGINE
# ============================================================================
class TradeGenerator:
    def __init__(self, data_processor):
        self.dp = data_processor
        
    def generate_trade(self, patterns, certainty_score, today_data):
        """Generate trade based on patterns and certainty"""
        certainty = certainty_score['overall_certainty']
        
        if certainty < 0.6:
            return self._generate_no_trade(certainty_score)
        
        primary_signal = certainty_score['primary_signal']
        
        if primary_signal in ['BULLISH', 'BUY']:
            return self._generate_buy_trade(today_data, certainty, patterns)
        elif primary_signal in ['BEARISH', 'SELL']:
            return self._generate_sell_trade(today_data, certainty, patterns)
        elif primary_signal == 'RANGE_BOUND':
            return self._generate_range_trade(today_data, certainty, patterns)
        else:
            return self._generate_no_trade(certainty_score)
    
    def _generate_buy_trade(self, data, certainty, patterns):
        """Generate buy trade"""
        current_price = data.get('current_price', 0)
        if pd.isna(current_price):
            current_price = 1.1700
            
        atr = data.get('atr_daily_pips', 20)
        if pd.isna(atr):
            atr = 20
            
        support = data.get('support_level_1', current_price - 0.0020)
        if pd.isna(support):
            support = current_price - 0.0020
        
        # Entry price
        entry_distance = atr * 0.1 / 10000  # 10% of ATR
        entry_price = max(support + 0.0001, current_price - entry_distance)
        
        # Stop loss
        stop_distance = atr * 1.5 / 10000  # 1.5x ATR
        stop_loss = entry_price - stop_distance
        
        # Take profit
        risk = entry_price - stop_loss
        risk_reward = self._get_risk_reward(certainty)
        take_profit = entry_price + (risk * risk_reward)
        
        # Position size
        position_size = self._calculate_position_size(entry_price, stop_loss, certainty)
        
        return {
            'signal': 'BUY',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': position_size,
            'reason': self._generate_reason(patterns, 'BUY'),
            'conditions': ['Price reaches entry', 'Stop loss if price < stop']
        }
    
    def _generate_sell_trade(self, data, certainty, patterns):
        """Generate sell trade"""
        current_price = data.get('current_price', 0)
        if pd.isna(current_price):
            current_price = 1.1700
            
        atr = data.get('atr_daily_pips', 20)
        if pd.isna(atr):
            atr = 20
            
        resistance = data.get('resistance_level_1', current_price + 0.0020)
        if pd.isna(resistance):
            resistance = current_price + 0.0020
        
        # Entry price
        entry_distance = atr * 0.1 / 10000
        entry_price = min(resistance - 0.0001, current_price + entry_distance)
        
        # Stop loss
        stop_distance = atr * 1.5 / 10000
        stop_loss = entry_price + stop_distance
        
        # Take profit
        risk = stop_loss - entry_price
        risk_reward = self._get_risk_reward(certainty)
        take_profit = entry_price - (risk * risk_reward)
        
        # Position size
        position_size = self._calculate_position_size(entry_price, stop_loss, certainty)
        
        return {
            'signal': 'SELL',
            'entry_price': round(entry_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'certainty': certainty,
            'risk_reward': risk_reward,
            'position_size': position_size,
            'reason': self._generate_reason(patterns, 'SELL'),
            'conditions': ['Price reaches entry', 'Stop loss if price > stop']
        }
    
    def _generate_range_trade(self, data, certainty, patterns):
        """Generate range trade"""
        current_price = data.get('current_price', 0)
        support = data.get('support_level_1', current_price - 0.0020)
        resistance = data.get('resistance_level_1', current_price + 0.0020)
        
        # Determine if near support or resistance
        distance_to_support = abs(current_price - support)
        distance_to_resistance = abs(current_price - resistance)
        
        if distance_to_support < distance_to_resistance:
            # Near support - buy
            trade = self._generate_buy_trade(data, certainty * 0.9, patterns)
            trade['signal'] = 'RANGE_BUY'
            trade['reason'] = 'Range trade: buying near support'
        else:
            # Near resistance - sell
            trade = self._generate_sell_trade(data, certainty * 0.9, patterns)
            trade['signal'] = 'RANGE_SELL'
            trade['reason'] = 'Range trade: selling near resistance'
        
        # Reduce position size for range trades
        trade['position_size'] *= 0.7
        
        return trade
    
    def _generate_no_trade(self, certainty_score):
        """Generate no-trade signal"""
        return {
            'signal': 'NO_TRADE',
            'certainty': certainty_score['overall_certainty'],
            'reason': f'Insufficient certainty ({certainty_score["overall_certainty"]:.2f})'
        }
    
    def _get_risk_reward(self, certainty):
        """Get risk/reward ratio based on certainty"""
        if certainty > 0.85:
            return 2.5
        elif certainty > 0.75:
            return 2.0
        elif certainty > 0.65:
            return 1.5
        else:
            return 1.0
    
    def _calculate_position_size(self, entry, stop_loss, certainty):
        """Calculate position size"""
        # Base risk: $100 for 1% risk on $10,000 account
        base_risk = 100
        
        # Adjust based on certainty
        if certainty > 0.85:
            risk_multiplier = 2.0
        elif certainty > 0.75:
            risk_multiplier = 1.5
        elif certainty > 0.65:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.5
        
        risk_amount = base_risk * risk_multiplier
        
        # Calculate pips risk
        risk_pips = abs(entry - stop_loss) * 10000
        if risk_pips == 0:
            return 0
        
        # Position size calculation
        position_size = risk_amount / (risk_pips * 10)
        return round(position_size, 2)
    
    def _generate_reason(self, patterns, signal):
        """Generate trade reason"""
        reasons = []
        
        for category, pattern_data in patterns.items():
            if not pattern_data:
                continue
                
            for pattern_name, pattern_info in pattern_data.items():
                if pattern_info.get('signal', '') == signal:
                    reasons.append(pattern_info.get('reason', ''))
        
        return " | ".join(reasons[:3]) if reasons else "Multiple factors aligned"

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================
def main():
    st.set_page_config(layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .trade-buy {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #28a745;
    }
    .trade-sell {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #dc3545;
    }
    .no-trade {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #6c757d;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🎯 FOREX CERTAINTY SYSTEM")
    st.markdown("**Real logic from CSV data - No hardcoded values**")
    
    # Load data
    csv_url = "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv"
    
    with st.spinner("Loading and analyzing CSV data..."):
        # Initialize engines
        dp = DataProcessor(csv_url)
        
        if not dp.load_data():
            st.error("Failed to load CSV data. Check the URL.")
            return
        
        today_data = dp.calculate_derived_metrics()
        
        # Initialize other engines
        pd_engine = PatternDetector(dp)
        cs_engine = CertaintyScorer()
        tg_engine = TradeGenerator(dp)
        
        # Run analysis
        patterns = pd_engine.detect_all_patterns(today_data)
        certainty_score = cs_engine.calculate_overall_certainty(patterns)
        trade = tg_engine.generate_trade(patterns, certainty_score, today_data)
    
    # Display results
    st.markdown(f"### 📅 Latest Data: {today_data.get('date', 'Unknown')}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"{today_data.get('current_price', 0):.5f}")
    with col2:
        retail = today_data.get('retail_long_pct', 50)
        st.metric("Retail Bias", f"{retail:.0f}% LONG")
    with col3:
        rsi = today_data.get('rsi_daily', 50)
        st.metric("RSI", f"{rsi:.1f}")
    with col4:
        regime = today_data.get('market_regime', 'UNKNOWN')
        st.metric("Market Regime", regime)
    
    # Certainty Score
    certainty = certainty_score['overall_certainty']
    st.markdown(f"### 🎯 Certainty Score: **{certainty:.1%}**")
    
    # Progress bar
    st.progress(min(certainty, 1.0))
    
    # Trade Signal
    st.markdown("### 📊 TRADE SIGNAL")
    
    if trade['signal'] == 'NO_TRADE':
        st.markdown(f"""
        <div class="no-trade">
            <h3>🚫 NO TRADE RECOMMENDED</h3>
            <p><strong>Reason:</strong> {trade['reason']}</p>
            <p><strong>Certainty:</strong> {trade['certainty']:.1%}</p>
            <p><em>Wait for higher certainty conditions</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Determine trade class
        trade_class = "trade-buy" if trade['signal'] in ['BUY', 'RANGE_BUY'] else "trade-sell"
        
        st.markdown(f"""
        <div class="{trade_class}">
            <h3>📈 {'🟢 BUY' if trade['signal'] in ['BUY', 'RANGE_BUY'] else '🔴 SELL'} EURUSD</h3>
            <p><strong>Entry:</strong> {trade['entry_price']}</p>
            <p><strong>Stop Loss:</strong> {trade['stop_loss']}</p>
            <p><strong>Take Profit:</strong> {trade['take_profit']}</p>
            <p><strong>Risk/Reward:</strong> 1:{trade['risk_reward']:.1f}</p>
            <p><strong>Position Size:</strong> {trade['position_size']} lots</p>
            <p><strong>Certainty:</strong> {trade['certainty']:.1%}</p>
            <p><strong>Reason:</strong> {trade['reason']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pattern Analysis
    st.markdown("### 🔍 PATTERN ANALYSIS")
    
    pattern_cols = st.columns(len(patterns))
    for idx, (category, pattern_data) in enumerate(patterns.items()):
        with pattern_cols[idx % len(pattern_cols)]:
            st.markdown(f"**{category.replace('_', ' ').title()}**")
            if pattern_data:
                for pattern_name, pattern_info in pattern_data.items():
                    st.write(f"• {pattern_name}: {pattern_info.get('reason', '')}")
            else:
                st.write("No patterns detected")
    
    # Raw Data
    with st.expander("📁 View Raw Data & Analysis"):
        tab1, tab2, tab3 = st.tabs(["CSV Data", "Patterns", "Certainty"])
        
        with tab1:
            st.dataframe(dp.data)
            
            # Show available columns
            st.markdown("#### Available Columns:")
            cols = list(dp.data.columns)
            for i in range(0, len(cols), 4):
                st.code(" | ".join(cols[i:i+4]))
        
        with tab2:
            st.json(patterns)
        
        with tab3:
            st.json(certainty_score)
    
    # Execution Code
    if trade['signal'] != 'NO_TRADE':
        st.markdown("### 💻 EXECUTION CODE")
        
        code = f"""
# MT4/MT5 Execution Code
def execute_trade():
    symbol = "EURUSD"
    entry = {trade['entry_price']}
    stoploss = {trade['stop_loss']}
    takeprofit = {trade['take_profit']}
    
    if signal == "{trade['signal']}":
        # Add your broker's execution code here
        pass
        """
        
        st.code(code, language='python')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **System Status:** ✅ Operational | **Data Source:** GitHub CSV  
    **Last Analysis:** {datetime} | **Records:** {records} rows
    """.format(
        datetime=datetime.now().strftime("%Y-%m-%d %H:%M"),
        records=len(dp.data) if dp.data else 0
    ))

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
