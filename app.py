# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Forex Certainty System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pattern-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
    }
    .trade-signal {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .high-confidence {
        color: #4CAF50;
        font-weight: bold;
    }
    .medium-confidence {
        color: #FF9800;
        font-weight: bold;
    }
    .low-confidence {
        color: #F44336;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# System Constants
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.90,
    'MEDIUM': 0.85,
    'LOW': 0.80
}

STAKE_MULTIPLIERS = {
    'HIGH': 2.0,
    'MEDIUM': 1.5,
    'LOW': 1.0
}

BASE_RISK_PCT = 0.005  # 0.5%
ACCOUNT_BALANCE = 10000  # Default account balance

class ForexCertaintySystem:
    def __init__(self, csv_path):
        """Initialize the system with CSV data"""
        # Load CSV with correct encoding and handle column names
        self.df = pd.read_csv(csv_path)
        
        # Clean column names: strip whitespace and convert to lowercase
        self.df.columns = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in self.df.columns]
        
        # Debug: Show column names
        print("Column names in CSV:", self.df.columns.tolist())
        
        # Rename columns to match expected names
        column_mapping = {
            'date': 'date',
            'currency_pair': 'currency_pair',
            'current_price': 'current_price',
            'weekly_open': 'weekly_open',
            'weekly_high': 'weekly_high',
            'weekly_low': 'weekly_low',
            'weekly_close': 'weekly_close',
            'daily_open': 'daily_open',
            'daily_high': 'daily_high',
            'daily_low': 'daily_low',
            'daily_close': 'daily_close',
            'daily_change_%': 'daily_change_pct',
            'high_impact_news': 'high_impact_news_count',
            'medium_impact_news': 'medium_impact_news_count',
            'next_major_news_time': 'next_major_news_time',
            'next_major_news_currency': 'next_major_news_currency',
            'retail_long_avg': 'retail_long_avg',
            'retail_short_avg': 'retail_short_avg',
            'retail_net_position': 'retail_net_position',
            'retail_distance_pips': 'retail_distance_pips',
            'days_to_holiday': 'days_to_next_holiday',
            'holiday_type': 'holiday_type',
            'expected_liquidity': 'expected_liquidity',
            'london_range': 'london_session_range',
            'ny_range': 'ny_session_range',
            'asian_range': 'asian_session_range',
            'weekly_range_pips': 'weekly_range_pips',
            'weekly_direction': 'weekly_direction',
            'daily_range_pips': 'daily_range_pips',
            'daily_direction': 'daily_direction',
            'support_1': 'support_level_1',
            'resistance_1': 'resistance_level_1',
            'price_to_support_pips': 'price_to_support_pips',
            'price_to_resistance_pips': 'price_to_resistance_pips',
            'volatility_score': 'volatility_score',
            'trend_strength': 'trend_strength',
            'consolidation_score': 'consolidation_score',
            'pattern_flags': 'pattern_flags'
        }
        
        # Apply column renaming
        self.df = self.df.rename(columns=column_mapping)
        
        # Convert date column to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort by date
        self.df = self.df.sort_values('date')
        
        # Get current data
        self.current_date = self.df['date'].max()
        self.today_data = self.df[self.df['date'] == self.current_date].iloc[0]
        
        # Debug: Show today's data
        print("Today's data:", self.today_data.to_dict())
        
    def get_pattern_confidence(self, pattern_id):
        """Return confidence level based on pattern ID"""
        if '85' in pattern_id:
            return CONFIDENCE_THRESHOLDS['MEDIUM']
        elif '90' in pattern_id:
            return CONFIDENCE_THRESHOLDS['HIGH']
        elif '95' in pattern_id:
            return CONFIDENCE_THRESHOLDS['HIGH']
        elif '80' in pattern_id:
            return CONFIDENCE_THRESHOLDS['LOW']
        elif '92' in pattern_id:
            return CONFIDENCE_THRESHOLDS['HIGH']
        else:
            return CONFIDENCE_THRESHOLDS['LOW']
    
    def check_retail_pattern(self):
        """Pattern 1: Retail Position Pain Points"""
        try:
            retail_distance = abs(float(self.today_data['retail_distance_pips']))
            if retail_distance > 10:
                pattern = {
                    'id': 'RETAIL_RESISTANCE_85',
                    'name': 'Retail Position Pain Points',
                    'description': 'Retail traders at breakeven points show predictable behavior',
                    'conditions_met': [
                        f"Retail distance: {self.today_data['retail_distance_pips']} pips",
                        f"Retail net position: {self.today_data['retail_net_position']}",
                        f"Current price: {self.today_data['current_price']}",
                        f"Retail avg: {self.today_data['retail_long_avg'] if str(self.today_data['retail_net_position']).upper() == 'LONG' else self.today_data['retail_short_avg']}"
                    ]
                }
                return pattern
        except Exception as e:
            print(f"Error in retail pattern: {e}")
        return None
    
    def check_news_pattern(self):
        """Pattern 2: News Volatility Expansion"""
        try:
            high_impact_news = int(float(self.today_data['high_impact_news_count']))
            if high_impact_news > 0:
                pattern = {
                    'id': 'NEWS_VOLATILITY_90',
                    'name': 'News Volatility Expansion',
                    'description': 'High-impact news creates predictable volatility expansion',
                    'conditions_met': [
                        f"High impact news: {high_impact_news} events",
                        f"Next major news: {self.today_data['next_major_news_time']}",
                        f"Currency: {self.today_data['next_major_news_currency']}",
                        f"Daily volatility: {self.today_data.get('volatility_score', 'N/A')}"
                    ]
                }
                return pattern
        except Exception as e:
            print(f"Error in news pattern: {e}")
        return None
    
    def check_holiday_pattern(self):
        """Pattern 3: Holiday Range Compression"""
        try:
            holiday_type = str(self.today_data['holiday_type']).upper()
            days_to_holiday = int(float(self.today_data['days_to_next_holiday']))
            
            if holiday_type != 'NONE' and days_to_holiday <= 2:
                pattern = {
                    'id': 'HOLIDAY_RANGE_95',
                    'name': 'Holiday Range Compression',
                    'description': 'Days before major holidays exhibit predictable range compression',
                    'conditions_met': [
                        f"Days to holiday: {days_to_holiday}",
                        f"Holiday type: {holiday_type}",
                        f"Expected liquidity: {self.today_data['expected_liquidity']}",
                        f"Consolidation score: {self.today_data.get('consolidation_score', 'N/A')}"
                    ]
                }
                return pattern
        except Exception as e:
            print(f"Error in holiday pattern: {e}")
        return None
    
    def check_overlap_pattern(self):
        """Pattern 4: Session Overlap Momentum"""
        try:
            current_hour = datetime.now().hour
            # Check if we're in overlap hours (13:00-16:00 GMT)
            if 13 <= current_hour <= 16:
                pattern = {
                    'id': 'OVERLAP_CONTINUATION_80',
                    'name': 'Session Overlap Momentum',
                    'description': 'Price momentum tends to continue during London-NY overlap',
                    'conditions_met': [
                        f"Current session: London-NY Overlap (13:00-16:00 GMT)",
                        f"Trend strength: {self.today_data.get('trend_strength', 'N/A')}",
                        f"Daily direction: {self.today_data['daily_direction']}",
                        f"London range: {self.today_data.get('london_session_range', 'N/A')} pips"
                    ]
                }
                return pattern
        except Exception as e:
            print(f"Error in overlap pattern: {e}")
        return None
    
    def check_gap_pattern(self):
        """Pattern 5: Gap Fill Probability"""
        try:
            weekly_open = float(self.today_data['weekly_open'])
            daily_open = float(self.today_data['daily_open'])
            gap_size = abs(weekly_open - daily_open) * 10000
            
            if gap_size > 25:
                pattern = {
                    'id': 'GAP_FILL_92',
                    'name': 'Gap Fill Probability',
                    'description': 'Opening gaps > 25 pips fill within 24 hours',
                    'conditions_met': [
                        f"Gap size: {gap_size:.1f} pips",
                        f"Weekly open: {weekly_open}",
                        f"Daily open: {daily_open}",
                        f"Current price: {self.today_data['current_price']}"
                    ]
                }
                return pattern
        except Exception as e:
            print(f"Error in gap pattern: {e}")
        return None
    
    def generate_trade_signals(self, patterns):
        """Convert patterns into trade signals"""
        trades = []
        
        for pattern in patterns:
            confidence = self.get_pattern_confidence(pattern['id'])
            
            # Calculate trade parameters based on pattern
            if 'RETAIL' in pattern['id']:
                trade = self._generate_retail_trade(pattern, confidence)
            elif 'NEWS' in pattern['id']:
                trade = self._generate_news_trade(pattern, confidence)
            elif 'HOLIDAY' in pattern['id']:
                trade = self._generate_holiday_trade(pattern, confidence)
            elif 'OVERLAP' in pattern['id']:
                trade = self._generate_overlap_trade(pattern, confidence)
            elif 'GAP' in pattern['id']:
                trade = self._generate_gap_trade(pattern, confidence)
            else:
                continue
            
            if trade:
                trades.append(trade)
        
        return trades
    
    def _generate_retail_trade(self, pattern, confidence):
        """Generate retail fade trade"""
        try:
            retail_position = str(self.today_data['retail_net_position']).upper()
            
            if retail_position == 'LONG':
                direction = 'SELL'
                retail_avg = float(self.today_data['retail_long_avg'])
                entry_price = retail_avg - 0.0002
                stop_loss = entry_price + 0.0017
                take_profit_1 = entry_price - 0.0015
                take_profit_2 = entry_price - 0.0028
            else:
                direction = 'BUY'
                retail_avg = float(self.today_data['retail_short_avg'])
                entry_price = retail_avg + 0.0002
                stop_loss = entry_price - 0.0017
                take_profit_1 = entry_price + 0.0015
                take_profit_2 = entry_price + 0.0028
            
            # Get stake multiplier based on confidence
            if confidence >= CONFIDENCE_THRESHOLDS['HIGH']:
                stake_mult = STAKE_MULTIPLIERS['HIGH']
            elif confidence >= CONFIDENCE_THRESHOLDS['MEDIUM']:
                stake_mult = STAKE_MULTIPLIERS['MEDIUM']
            else:
                stake_mult = STAKE_MULTIPLIERS['LOW']
            
            trade = {
                'pattern_id': pattern['id'],
                'pattern_name': pattern['name'],
                'currency_pair': 'EURUSD',
                'direction': direction,
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profit_1, 5),
                'take_profit_2': round(take_profit_2, 5),
                'confidence': confidence,
                'stake_multiplier': stake_mult,
                'risk_pips': round(abs(entry_price - stop_loss) * 10000, 1),
                'reward_pips': round(abs(entry_price - take_profit_1) * 10000, 1),
                'risk_reward_ratio': round(abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss), 2),
                'conditions_met': pattern['conditions_met']
            }
            
            # Add position sizing
            trade.update(self._calculate_position_size(trade))
            
            return trade
        except Exception as e:
            print(f"Error generating retail trade: {e}")
            return None
    
    def _generate_news_trade(self, pattern, confidence):
        """Generate news volatility trade"""
        try:
            # Straddle/Strangle strategy for news
            current_price = float(self.today_data['current_price'])
            expected_move = 40 * 0.0001  # 40 pips minimum
            
            trade = {
                'pattern_id': pattern['id'],
                'pattern_name': pattern['name'],
                'currency_pair': 'EURUSD',
                'direction': 'STRADDLE',
                'entry_type': 'MARKET',
                'entry_price': 'Straddle around current price',
                'stop_loss': round(current_price + expected_move * 5, 5),  # Wide stop
                'take_profit_1': round(current_price + expected_move * 0.4, 5),  # 40% move
                'take_profit_2': round(current_price + expected_move * 0.8, 5),  # 80% move
                'confidence': confidence,
                'stake_multiplier': 1.5,
                'strategy': 'Enter 30min before news, exit partial at 40%, runner at 80%',
                'conditions_met': pattern['conditions_met'],
                'news_time': self.today_data['next_major_news_time']
            }
            
            return trade
        except Exception as e:
            print(f"Error generating news trade: {e}")
            return None
    
    def _generate_holiday_trade(self, pattern, confidence):
        """Generate holiday range trade"""
        try:
            current_price = float(self.today_data['current_price'])
            predicted_range = float(self.today_data['daily_range_pips']) * 0.5  # 50% compression
            
            trade = {
                'pattern_id': pattern['id'],
                'pattern_name': pattern['name'],
                'currency_pair': 'EURUSD',
                'direction': 'RANGE_BOUND',
                'entry_price': round(current_price, 5),
                'stop_loss': round(current_price + predicted_range * 1.5 * 0.0001, 5),
                'take_profit_1': round(current_price - predicted_range * 0.5 * 0.0001, 5),
                'take_profit_2': round(current_price + predicted_range * 0.5 * 0.0001, 5),
                'confidence': confidence,
                'stake_multiplier': 2.0,
                'strategy': 'Range trade / volatility sell',
                'conditions_met': pattern['conditions_met'],
                'expiry': 'End of trading day'
            }
            
            return trade
        except Exception as e:
            print(f"Error generating holiday trade: {e}")
            return None
    
    def _generate_overlap_trade(self, pattern, confidence):
        """Generate overlap continuation trade"""
        try:
            direction = str(self.today_data['daily_direction']).upper()
            current_price = float(self.today_data['current_price'])
            
            if direction == 'BULLISH':
                entry_price = current_price - 0.0010  # Wait for retracement
                stop_loss = entry_price - 0.0015
                take_profit = entry_price + 0.0025
            else:
                entry_price = current_price + 0.0010
                stop_loss = entry_price + 0.0015
                take_profit = entry_price - 0.0025
            
            trade = {
                'pattern_id': pattern['id'],
                'pattern_name': pattern['name'],
                'currency_pair': 'EURUSD',
                'direction': direction,
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'confidence': confidence,
                'stake_multiplier': 1.0,
                'time_limit': 'Exit before overlap ends (16:00 GMT)',
                'conditions_met': pattern['conditions_met']
            }
            
            return trade
        except Exception as e:
            print(f"Error generating overlap trade: {e}")
            return None
    
    def _generate_gap_trade(self, pattern, confidence):
        """Generate gap fill trade"""
        try:
            weekly_open = float(self.today_data['weekly_open'])
            daily_open = float(self.today_data['daily_open'])
            gap_size = abs(weekly_open - daily_open) * 10000
            
            gap_direction = 'UP' if daily_open > weekly_open else 'DOWN'
            
            if gap_direction == 'UP':
                direction = 'SELL'  # Fade the gap up
                entry_price = daily_open + 0.0005
                stop_loss = entry_price + gap_size * 1.5 * 0.0001
                take_profit = weekly_open
            else:
                direction = 'BUY'  # Fade the gap down
                entry_price = daily_open - 0.0005
                stop_loss = entry_price - gap_size * 1.5 * 0.0001
                take_profit = weekly_open
            
            trade = {
                'pattern_id': pattern['id'],
                'pattern_name': pattern['name'],
                'currency_pair': 'EURUSD',
                'direction': direction,
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'confidence': confidence,
                'stake_multiplier': 1.8,
                'time_limit': '24 hours',
                'gap_size': f"{gap_size:.1f} pips",
                'conditions_met': pattern['conditions_met']
            }
            
            return trade
        except Exception as e:
            print(f"Error generating gap trade: {e}")
            return None
    
    def _calculate_position_size(self, trade):
        """Calculate position size based on risk parameters"""
        try:
            risk_amount = ACCOUNT_BALANCE * BASE_RISK_PCT * trade['stake_multiplier']
            risk_pips = trade['risk_pips']
            
            if risk_pips > 0:
                # Standard pip value for EURUSD
                pip_value = 0.0001
                position_size = risk_amount / (risk_pips * pip_value)
                
                return {
                    'position_size': round(position_size, 2),
                    'risk_amount': round(risk_amount, 2),
                    'reward_amount': round(risk_amount * trade['risk_reward_ratio'], 2)
                }
        except Exception as e:
            print(f"Error calculating position size: {e}")
        
        return {'position_size': 0, 'risk_amount': 0, 'reward_amount': 0}
    
    def get_market_metrics(self):
        """Calculate market metrics for dashboard"""
        try:
            return {
                'volatility_score': float(self.today_data.get('volatility_score', 0)),
                'trend_strength': float(self.today_data.get('trend_strength', 0)),
                'consolidation_score': float(self.today_data.get('consolidation_score', 0)),
                'daily_range': float(self.today_data.get('daily_range_pips', 0)),
                'weekly_range': float(self.today_data.get('weekly_range_pips', 0)),
                'retail_bias': str(self.today_data.get('retail_net_position', 'N/A')),
                'news_impact': 'HIGH' if int(float(self.today_data.get('high_impact_news_count', 0))) > 0 else 'LOW',
                'liquidity_status': str(self.today_data.get('expected_liquidity', 'N/A'))
            }
        except Exception as e:
            print(f"Error getting market metrics: {e}")
            return {}
    
    def get_historical_performance(self, days=30):
        """Get historical performance data"""
        try:
            historical = self.df.tail(days).copy()
            
            # Convert columns to numeric
            historical['retail_distance_pips'] = pd.to_numeric(historical['retail_distance_pips'], errors='coerce')
            historical['high_impact_news_count'] = pd.to_numeric(historical['high_impact_news_count'], errors='coerce')
            historical['weekly_open'] = pd.to_numeric(historical['weekly_open'], errors='coerce')
            historical['daily_open'] = pd.to_numeric(historical['daily_open'], errors='coerce')
            
            # Simulate pattern occurrences
            pattern_counts = {
                'RETAIL': len(historical[abs(historical['retail_distance_pips']) > 10]),
                'NEWS': len(historical[historical['high_impact_news_count'] > 0]),
                'HOLIDAY': len(historical[historical['holiday_type'] != 'NONE']),
                'GAP': len(historical[abs(historical['weekly_open'] - historical['daily_open']) * 10000 > 25])
            }
            
            return pattern_counts
        except Exception as e:
            print(f"Error getting historical performance: {e}")
            return {}

# Streamlit App
def main():
    # Header
    st.markdown("<h1 class='main-header'>🚀 Forex Certainty System v1.0</h1>", unsafe_allow_html=True)
    
    # Load data
    csv_url = "https://raw.githubusercontent.com/profdue/tradelock/main/forex_certainty_data.csv"
    
    try:
        system = ForexCertaintySystem(csv_url)
        
        # Sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/forex.png", width=100)
            st.title("System Controls")
            
            st.subheader("Account Settings")
            global ACCOUNT_BALANCE
            ACCOUNT_BALANCE = st.number_input("Account Balance ($)", value=10000, min_value=1000, step=1000)
            
            st.subheader("Risk Parameters")
            base_risk = st.slider("Base Risk %", 0.1, 2.0, 0.5) / 100
            global BASE_RISK_PCT
            BASE_RISK_PCT = base_risk
            
            st.subheader("Pattern Filters")
            min_confidence = st.select_slider(
                "Minimum Confidence",
                options=['80%', '85%', '90%', '95%'],
                value='80%'
            )
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            st.markdown("---")
            st.caption(f"Last Updated: {system.current_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Main Dashboard
        st.subheader(f"📅 Current Date: {system.current_date.strftime('%Y-%m-%d')}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = float(system.today_data['current_price'])
            daily_change = float(system.today_data['daily_change_pct'])
            st.metric("Current Price", f"{current_price:.5f}", 
                     f"{daily_change:.2f}%")
        with col2:
            daily_range = float(system.today_data['daily_range_pips'])
            daily_dir = str(system.today_data['daily_direction'])
            st.metric("Daily Range", f"{daily_range:.0f} pips", 
                     daily_dir)
        with col3:
            weekly_dir = str(system.today_data['weekly_direction'])
            weekly_range = float(system.today_data['weekly_range_pips'])
            st.metric("Weekly Direction", weekly_dir, 
                     f"{weekly_range:.0f} pips")
        with col4:
            news_count = int(float(system.today_data['high_impact_news_count']))
            next_news = str(system.today_data['next_major_news_time'])
            st.metric("High Impact News", news_count, 
                     f"Next: {next_news}" if news_count > 0 and next_news != 'nan' else "None")
        
        # Market Metrics Section
        st.markdown("---")
        st.subheader("📊 Market Metrics")
        
        metrics = system.get_market_metrics()
        if metrics:
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            
            with mcol1:
                vol_score = metrics.get('volatility_score', 0)
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Volatility</h4>
                    <h2>{vol_score:.2f}</h2>
                    <p>{'High' if vol_score > 0.7 else 'Medium' if vol_score > 0.4 else 'Low'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with mcol2:
                trend_score = metrics.get('trend_strength', 0)
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Trend Strength</h4>
                    <h2>{trend_score:.2f}</h2>
                    <p>{'Strong' if trend_score > 0.6 else 'Weak' if trend_score < 0.4 else 'Moderate'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with mcol3:
                cons_score = metrics.get('consolidation_score', 0)
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Consolidation</h4>
                    <h2>{cons_score:.2f}</h2>
                    <p>{'Range-bound' if cons_score > 0.6 else 'Trending'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with mcol4:
                retail_bias = metrics.get('retail_bias', 'N/A')
                retail_dist = float(system.today_data.get('retail_distance_pips', 0))
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Retail Bias</h4>
                    <h2>{retail_bias}</h2>
                    <p>{retail_dist:.1f} pips from avg</p>
                </div>
                """, unsafe_allow_html=True)
            
            with mcol5:
                liquidity = metrics.get('liquidity_status', 'N/A')
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Liquidity</h4>
                    <h2>{liquidity}</h2>
                    <p>{'Normal' if liquidity == 'NORMAL' else 'Reduced'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Pattern Detection
        st.markdown("---")
        st.subheader("🎯 Pattern Detection")
        
        patterns = []
        
        # Check each pattern
        retail_pattern = system.check_retail_pattern()
        if retail_pattern:
            patterns.append(retail_pattern)
            with st.expander(f"✅ {retail_pattern['name']} (85% Confidence)"):
                st.write(retail_pattern['description'])
                for condition in retail_pattern['conditions_met']:
                    st.write(f"• {condition}")
        
        news_pattern = system.check_news_pattern()
        if news_pattern:
            patterns.append(news_pattern)
            with st.expander(f"✅ {news_pattern['name']} (90% Confidence)"):
                st.write(news_pattern['description'])
                for condition in news_pattern['conditions_met']:
                    st.write(f"• {condition}")
        
        holiday_pattern = system.check_holiday_pattern()
        if holiday_pattern:
            patterns.append(holiday_pattern)
            with st.expander(f"✅ {holiday_pattern['name']} (95% Confidence)"):
                st.write(holiday_pattern['description'])
                for condition in holiday_pattern['conditions_met']:
                    st.write(f"• {condition}")
        
        overlap_pattern = system.check_overlap_pattern()
        if overlap_pattern:
            patterns.append(overlap_pattern)
            with st.expander(f"✅ {overlap_pattern['name']} (80% Confidence)"):
                st.write(overlap_pattern['description'])
                for condition in overlap_pattern['conditions_met']:
                    st.write(f"• {condition}")
        
        gap_pattern = system.check_gap_pattern()
        if gap_pattern:
            patterns.append(gap_pattern)
            with st.expander(f"✅ {gap_pattern['name']} (92% Confidence)"):
                st.write(gap_pattern['description'])
                for condition in gap_pattern['conditions_met']:
                    st.write(f"• {condition}")
        
        if not patterns:
            st.info("No high-probability patterns detected for today.")
        
        # Trade Generation
        if patterns:
            st.markdown("---")
            st.subheader("📈 Generated Trade Signals")
            
            trades = system.generate_trade_signals(patterns)
            
            for i, trade in enumerate(trades):
                if not trade:
                    continue
                    
                # Determine confidence color
                if trade['confidence'] >= CONFIDENCE_THRESHOLDS['HIGH']:
                    conf_class = "high-confidence"
                    conf_label = "HIGH"
                elif trade['confidence'] >= CONFIDENCE_THRESHOLDS['MEDIUM']:
                    conf_class = "medium-confidence"
                    conf_label = "MEDIUM"
                else:
                    conf_class = "low-confidence"
                    conf_label = "LOW"
                
                with st.container():
                    st.markdown(f"""
                    <div class='trade-signal'>
                        <h3>Trade #{i+1}: {trade['pattern_name']}</h3>
                        <p><strong>Currency:</strong> {trade['currency_pair']} | 
                        <strong>Direction:</strong> {trade['direction']} | 
                        <strong>Confidence:</strong> <span class='{conf_class}'>{conf_label} ({trade['confidence']*100:.0f}%)</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Trade details in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'entry_price' in trade and trade['entry_price'] != 'Straddle around current price':
                            st.metric("Entry Price", f"{trade['entry_price']:.5f}")
                        else:
                            st.metric("Entry", trade.get('entry_price', 'N/A'))
                        
                        if 'take_profit_1' in trade:
                            st.metric("Take Profit 1", f"{trade['take_profit_1']:.5f}")
                        if 'take_profit_2' in trade:
                            st.metric("Take Profit 2", f"{trade['take_profit_2']:.5f}")
                        elif 'take_profit' in trade:
                            st.metric("Take Profit", f"{trade['take_profit']:.5f}")
                    
                    with col2:
                        if 'stop_loss' in trade:
                            st.metric("Stop Loss", f"{trade['stop_loss']:.5f}")
                        if 'risk_pips' in trade:
                            st.metric("Risk", f"{trade['risk_pips']} pips")
                        if 'reward_pips' in trade:
                            st.metric("Reward", f"{trade['reward_pips']} pips")
                    
                    with col3:
                        if 'risk_reward_ratio' in trade:
                            st.metric("Risk/Reward", f"1:{trade['risk_reward_ratio']}")
                        if 'stake_multiplier' in trade:
                            st.metric("Stake Multiplier", f"{trade['stake_multiplier']}x")
                        if 'position_size' in trade:
                            st.metric("Position Size", f"{trade['position_size']:.2f}")
                    
                    # Conditions
                    with st.expander("Trade Conditions"):
                        for condition in trade['conditions_met']:
                            st.write(f"• {condition}")
                        
                        if 'strategy' in trade:
                            st.info(f"**Strategy:** {trade['strategy']}")
                        if 'time_limit' in trade:
                            st.info(f"**Time Limit:** {trade['time_limit']}")
                        if 'news_time' in trade:
                            st.warning(f"**News Time:** {trade['news_time']}")
                        if 'gap_size' in trade:
                            st.info(f"**Gap Size:** {trade['gap_size']}")
                    
                    st.markdown("---")
        
        # Historical Analysis
        st.markdown("---")
        st.subheader("📅 Historical Pattern Analysis")
        
        pattern_counts = system.get_historical_performance(30)
        
        if pattern_counts:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(pattern_counts.keys()),
                    y=list(pattern_counts.values()),
                    marker_color=['#1E88E5', '#43A047', '#FB8C00', '#E53935']
                )
            ])
            
            fig.update_layout(
                title="Pattern Occurrences (Last 30 Days)",
                xaxis_title="Pattern Type",
                yaxis_title="Occurrences",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw Data
        with st.expander("📋 View Raw CSV Data"):
            st.dataframe(system.df)
            
            # CSV download
            csv = system.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed CSV",
                data=csv,
                file_name="forex_certainty_data_processed.csv",
                mime="text/csv"
            )
        
        # Debug Info
        with st.expander("🔧 Debug Information"):
            st.write("Today's data sample:")
            st.json(system.today_data.to_dict())
            
            st.write("Column names in dataframe:")
            st.write(system.df.columns.tolist())
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Make sure the CSV is available at the specified URL and has the correct format.")
        
        # Debug information
        with st.expander("Debug Info"):
            st.write("Trying to load CSV from:", csv_url)
            try:
                # Try to load and show raw CSV
                test_df = pd.read_csv(csv_url)
                st.write("Raw CSV loaded successfully")
                st.write("Columns found:", test_df.columns.tolist())
                st.write("First few rows:")
                st.dataframe(test_df.head())
            except Exception as load_error:
                st.write(f"Failed to load CSV: {load_error}")

if __name__ == "__main__":
    main()
