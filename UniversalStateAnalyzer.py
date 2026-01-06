# UniversalStateAnalyzer.py
# Place in: C:\Users\davidug.SIFAXGROUP\AppData\Roaming\MetaQuotes\Terminal\BA78EA1631820D7AEF052166A20E7B1A\MQL5\Experts\Trading_Analysis

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8-darkgrid')

class StatePatternAnalyzer:
    def __init__(self, data_path=None):
        """Initialize analyzer with data path"""
        print("=== UNIVERSAL STATE PATTERN ANALYZER ===")
        print("Version 1.0 - For MQL5 Data Collection")
        
        # Set default path based on your structure
        if data_path is None:
            # Your MT5 files location
            data_path = r"C:\Users\davidug.SIFAXGROUP\AppData\Roaming\MetaQuotes\Terminal\BA78EA1631820D7AEF052166A20E7B1A\MQL5\Files\UniversalStateData.csv"
        
        self.data_path = data_path
        self.df = None
        self.state_stats = {}
        self.patterns = {}
        
    def load_data(self):
        """Load and preprocess the data"""
        print("\n[1/8] LOADING DATA...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(self.df)} records")
            print(f"✓ Columns: {len(self.df.columns)}")
            print(f"✓ Date range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
            
            # Convert timestamp to datetime
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
            
            # Sort by timestamp
            self.df = self.df.sort_values('Timestamp')
            
            # Display sample
            print("\nSample data:")
            print(self.df.head(3))
            
            return True
            
        except FileNotFoundError:
            print(f"✗ File not found: {self.data_path}")
            print("Make sure the MQL5 EA has been running to generate data")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def analyze_basic_stats(self):
        """Calculate basic statistics"""
        print("\n[2/8] BASIC STATISTICS...")
        
        # Market sessions distribution
        session_counts = self.df['Session'].value_counts()
        print("\nMarket Sessions Distribution:")
        for session, count in session_counts.items():
            print(f"  {session}: {count} records ({count/len(self.df)*100:.1f}%)")
        
        # State frequency
        state_counts = self.df['State_Category'].value_counts()
        print("\nState Category Distribution (Top 10):")
        for state, count in state_counts.head(10).items():
            print(f"  {state}: {count} records ({count/len(self.df)*100:.1f}%)")
        
        # Volatility distribution
        vol_counts = self.df['Volatility'].value_counts()
        print("\nVolatility Distribution:")
        for vol, count in vol_counts.items():
            print(f"  {vol}: {count} records")
        
        return session_counts, state_counts
    
    def calculate_returns_analysis(self):
        """Analyze returns by state"""
        print("\n[3/8] RETURNS ANALYSIS BY STATE...")
        
        # Group by state category
        returns_by_state = self.df.groupby('State_Category').agg({
            'Return_15m': ['mean', 'std', 'count'],
            'Return_1h': ['mean', 'std'],
            'Return_4h': ['mean', 'std'],
            'State_Confidence': 'mean'
        }).round(4)
        
        # Calculate Sharpe-like ratios (mean/std)
        returns_by_state['Sharpe_15m'] = returns_by_state['Return_15m']['mean'] / returns_by_state['Return_15m']['std'].replace(0, np.nan)
        returns_by_state['Sharpe_1h'] = returns_by_state['Return_1h']['mean'] / returns_by_state['Return_1h']['std'].replace(0, np.nan)
        
        # Win rates (positive returns)
        win_rates_15m = []
        win_rates_1h = []
        
        for state in returns_by_state.index:
            state_data = self.df[self.df['State_Category'] == state]
            win_rate_15m = (state_data['Return_15m'] > 0).mean() * 100
            win_rate_1h = (state_data['Return_1h'] > 0).mean() * 100
            win_rates_15m.append(win_rate_15m)
            win_rates_1h.append(win_rate_1h)
        
        returns_by_state['WinRate_15m'] = win_rates_15m
        returns_by_state['WinRate_1h'] = win_rates_1h
        
        # Clean up multi-index
        returns_by_state.columns = ['_'.join(col).strip() for col in returns_by_state.columns.values]
        
        print("\nReturns by State Category:")
        print(returns_by_state[['Return_15m_count', 'Return_15m_mean', 'WinRate_15m', 
                               'Return_1h_mean', 'WinRate_1h', 'Sharpe_15m']].sort_values('Sharpe_15m', ascending=False))
        
        # Identify best performing states
        min_samples = 20  # Minimum samples to consider
        best_states = returns_by_state[
            (returns_by_state['Return_15m_count'] >= min_samples) &
            (returns_by_state['WinRate_15m'] > 55) &
            (returns_by_state['Sharpe_15m'] > 0.5)
        ].sort_values('Sharpe_15m', ascending=False)
        
        print(f"\n✓ Best performing states (>{min_samples} samples, >55% win rate):")
        print(best_states[['Return_15m_count', 'Return_15m_mean', 'WinRate_15m', 'Sharpe_15m']])
        
        return returns_by_state, best_states
    
    def analyze_state_transitions(self):
        """Analyze Markov chain of state transitions"""
        print("\n[4/8] STATE TRANSITION ANALYSIS...")
        
        transitions = []
        for i in range(1, len(self.df)):
            prev_state = self.df.iloc[i-1]['State_Category']
            curr_state = self.df.iloc[i]['State_Category']
            if prev_state != curr_state:
                transitions.append((prev_state, curr_state))
        
        if not transitions:
            print("No state transitions found")
            return None
        
        # Create transition matrix
        transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
        transition_matrix = pd.crosstab(
            transition_df['From'],
            transition_df['To'],
            normalize='index'
        ).round(3)
        
        print(f"\n✓ Found {len(transitions)} state transitions")
        print(f"✓ Unique transition pairs: {len(transition_matrix)}")
        
        # Find absorbing states (high probability of staying)
        absorbing_states = []
        for state in transition_matrix.index:
            if state in transition_matrix.columns:
                stay_prob = transition_matrix.loc[state, state] if state in transition_matrix.loc[state].index else 0
                if stay_prob > 0.6:  # More than 60% chance of staying
                    absorbing_states.append((state, stay_prob))
        
        if absorbing_states:
            print("\nAbsorbing States (>60% chance of staying):")
            for state, prob in sorted(absorbing_states, key=lambda x: x[1], reverse=True):
                print(f"  {state}: {prob:.1%}")
        
        # Find most predictable transitions
        print("\nMost Predictable Transitions (>40% probability):")
        predictable_transitions = []
        for from_state in transition_matrix.index:
            for to_state in transition_matrix.columns:
                prob = transition_matrix.loc[from_state, to_state]
                if prob > 0.4 and from_state != to_state:
                    predictable_transitions.append((from_state, to_state, prob))
        
        for from_state, to_state, prob in sorted(predictable_transitions, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  {from_state} → {to_state}: {prob:.1%}")
        
        return transition_matrix
    
    def analyze_session_patterns(self):
        """Analyze patterns by trading session"""
        print("\n[5/8] SESSION-BASED PATTERNS...")
        
        session_results = {}
        
        for session in self.df['Session'].unique():
            session_data = self.df[self.df['Session'] == session]
            
            if len(session_data) < 10:
                continue
            
            avg_return_15m = session_data['Return_15m'].mean()
            avg_return_1h = session_data['Return_1h'].mean()
            win_rate_15m = (session_data['Return_15m'] > 0).mean() * 100
            volatility = session_data['Return_15m'].std()
            
            # Most common states in this session
            common_states = session_data['State_Category'].value_counts().head(3)
            
            session_results[session] = {
                'samples': len(session_data),
                'avg_return_15m': avg_return_15m,
                'avg_return_1h': avg_return_1h,
                'win_rate_15m': win_rate_15m,
                'volatility': volatility,
                'common_states': dict(common_states)
            }
        
        print("\nPerformance by Trading Session:")
        session_df = pd.DataFrame(session_results).T
        print(session_df[['samples', 'avg_return_15m', 'win_rate_15m', 'volatility']].sort_values('avg_return_15m', ascending=False))
        
        return session_results
    
    def find_high_probability_patterns(self):
        """Find specific high-probability patterns"""
        print("\n[6/8] HIGH-PROBABILITY PATTERN DETECTION...")
        
        patterns = []
        
        # Look for specific state combinations
        state_combinations = [
            # (Volatility, Control, Participation, TimePressure)
            ("HIGH", "BULL_CONTROL", "STRONG_BUY", "ACCELERATING"),
            ("HIGH", "BEAR_CONTROL", "STRONG_SELL", "ACCELERATING"),
            ("LOW", "CONTESTED", "WEAK", "STEADY"),
            ("NORMAL", "BULL_CONTROL", "MODERATE_BUY", "SLOWING_UP"),
            ("NORMAL", "BEAR_CONTROL", "MODERATE_SELL", "SLOWING_DOWN"),
        ]
        
        for vol, ctrl, part, time in state_combinations:
            pattern_data = self.df[
                (self.df['Volatility'] == vol) &
                (self.df['Control'] == ctrl) &
                (self.df['Participation'] == part) &
                (self.df['TimePressure'] == time)
            ]
            
            if len(pattern_data) >= 5:  # Minimum samples
                avg_return_15m = pattern_data['Return_15m'].mean()
                avg_return_1h = pattern_data['Return_1h'].mean()
                win_rate_15m = (pattern_data['Return_15m'] > 0).mean() * 100
                win_rate_1h = (pattern_data['Return_1h'] > 0).mean() * 100
                
                patterns.append({
                    'pattern': f"{vol}|{ctrl}|{part}|{time}",
                    'samples': len(pattern_data),
                    'avg_return_15m': avg_return_15m,
                    'avg_return_1h': avg_return_1h,
                    'win_rate_15m': win_rate_15m,
                    'win_rate_1h': win_rate_1h
                })
        
        if patterns:
            patterns_df = pd.DataFrame(patterns)
            print("\nHigh-Probability Patterns Found:")
            print(patterns_df.sort_values('win_rate_15m', ascending=False))
            
            # Save best patterns
            best_patterns = patterns_df[patterns_df['samples'] >= 10].sort_values('win_rate_15m', ascending=False)
            
            print(f"\n✓ Best tradeable patterns (>10 samples):")
            for idx, row in best_patterns.head(10).iterrows():
                print(f"  Pattern: {row['pattern']}")
                print(f"    Samples: {row['samples']}, Win Rate 15m: {row['win_rate_15m']:.1f}%, Avg Return: {row['avg_return_15m']:.2f}")
                print()
            
            return best_patterns
        else:
            print("No significant patterns found yet (need more data)")
            return None
    
    def create_trading_playbook(self, best_states, best_patterns):
        """Create a trading playbook from analysis"""
        print("\n[7/8] CREATING TRADING PLAYBOOK...")
        
        playbook = []
        
        # Add best states
        if best_states is not None and len(best_states) > 0:
            for state, row in best_states.iterrows():
                state_data = self.df[self.df['State_Category'] == state]
                
                # Get typical characteristics
                typical_vol = state_data['Volatility'].mode()[0]
                typical_ctrl = state_data['Control'].mode()[0]
                typical_part = state_data['Participation'].mode()[0]
                typical_session = state_data['Session'].mode()[0]
                
                playbook.append({
                    'type': 'STATE',
                    'name': state,
                    'samples': row['Return_15m_count'],
                    'avg_return_15m': row['Return_15m_mean'],
                    'win_rate_15m': row['WinRate_15m'],
                    'sharpe': row['Sharpe_15m'],
                    'typical_volatility': typical_vol,
                    'typical_control': typical_ctrl,
                    'typical_participation': typical_part,
                    'best_session': typical_session,
                    'recommendation': self._generate_recommendation(state, row),
                    'confidence': min(100, row['WinRate_15m'] * 1.5)  # Simple confidence score
                })
        
        # Add best patterns
        if best_patterns is not None and len(best_patterns) > 0:
            for idx, row in best_patterns.iterrows():
                playbook.append({
                    'type': 'PATTERN',
                    'name': row['pattern'],
                    'samples': row['samples'],
                    'avg_return_15m': row['avg_return_15m'],
                    'win_rate_15m': row['win_rate_15m'],
                    'recommendation': self._generate_pattern_recommendation(row['pattern']),
                    'confidence': min(100, row['win_rate_15m'] * 1.2)
                })
        
        # Convert to DataFrame and save
        if playbook:
            playbook_df = pd.DataFrame(playbook)
            
            # Save to CSV
            playbook_path = self.data_path.replace('UniversalStateData.csv', 'TradingPlaybook.csv')
            playbook_df.to_csv(playbook_path, index=False)
            
            print(f"\n✓ Trading Playbook created with {len(playbook)} entries")
            print(f"✓ Saved to: {playbook_path}")
            
            # Print summary
            print("\nPLAYBOOK SUMMARY:")
            print("="*80)
            for entry in playbook[:5]:  # Show top 5
                print(f"\n{entry['type']}: {entry['name']}")
                print(f"  Samples: {entry['samples']}")
                print(f"  Win Rate: {entry['win_rate_15m']:.1f}%")
                print(f"  Avg Return: {entry['avg_return_15m']:.2f} pips")
                print(f"  Recommendation: {entry['recommendation']}")
                print(f"  Confidence: {entry['confidence']:.0f}%")
            
            return playbook_df
        else:
            print("No playbook entries created (need more data)")
            return None
    
    def _generate_recommendation(self, state, stats):
        """Generate trading recommendation based on state"""
        if "TREND_UP" in state:
            return "Look for BUY opportunities, especially on pullbacks"
        elif "TREND_DOWN" in state:
            return "Look for SELL opportunities, especially on rallies"
        elif "COMPRESSION" in state:
            return "Prepare for breakout - wait for volume confirmation"
        elif "VOLATILE_RANGE" in state:
            return "Range trading with tight stops - buy low, sell high"
        else:
            if stats['Return_15m_mean'] > 0:
                return "Mild bullish bias - consider small BUY positions"
            else:
                return "Mild bearish bias - consider small SELL positions"
    
    def _generate_pattern_recommendation(self, pattern):
        """Generate recommendation for specific pattern"""
        if "BULL_CONTROL" in pattern and "STRONG_BUY" in pattern:
            return "STRONG BUY - trend continuation likely"
        elif "BEAR_CONTROL" in pattern and "STRONG_SELL" in pattern:
            return "STRONG SELL - trend continuation likely"
        elif "CONTESTED" in pattern and "WEAK" in pattern:
            return "BREAKOUT IMMINENT - wait for direction"
        else:
            return "NEUTRAL - wait for confirmation"
    
    def create_visualizations(self):
        """Create visualization charts"""
        print("\n[8/8] CREATING VISUALIZATIONS...")
        
        try:
            # 1. Returns distribution
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            self.df['Return_15m'].hist(bins=50, alpha=0.7)
            plt.title('Distribution of 15-Minute Returns')
            plt.xlabel('Return (pips)')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
            # 2. Returns by state
            plt.subplot(2, 2, 2)
            state_returns = self.df.groupby('State_Category')['Return_15m'].mean().sort_values()
            state_returns.tail(10).plot(kind='barh')  # Top 10 states
            plt.title('Average 15-Min Returns by State (Top 10)')
            plt.xlabel('Average Return (pips)')
            
            # 3. Win rate by session
            plt.subplot(2, 2, 3)
            session_winrates = []
            for session in self.df['Session'].unique():
                session_data = self.df[self.df['Session'] == session]
                win_rate = (session_data['Return_15m'] > 0).mean() * 100
                session_winrates.append((session, win_rate))
            
            sessions, winrates = zip(*sorted(session_winrates, key=lambda x: x[1]))
            plt.bar(range(len(sessions)), winrates)
            plt.title('Win Rate by Trading Session')
            plt.xticks(range(len(sessions)), sessions, rotation=45)
            plt.ylabel('Win Rate (%)')
            plt.ylim(0, 100)
            
            # 4. State confidence vs returns
            plt.subplot(2, 2, 4)
            plt.scatter(self.df['State_Confidence'], self.df['Return_1h'], alpha=0.5)
            plt.title('State Confidence vs 1-Hour Returns')
            plt.xlabel('State Confidence')
            plt.ylabel('1-Hour Return (pips)')
            
            plt.tight_layout()
            
            # Save the figure
            viz_path = self.data_path.replace('UniversalStateData.csv', 'StateAnalysis.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualizations saved to: {viz_path}")
            
            # Create correlation heatmap
            numeric_cols = ['Return_15m', 'Return_1h', 'Return_4h', 'State_Confidence', 'RSI', 'ATR']
            numeric_df = self.df[numeric_cols].copy()
            
            plt.figure(figsize=(10, 8))
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Heatmap')
            
            heatmap_path = self.data_path.replace('UniversalStateData.csv', 'CorrelationHeatmap.png')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Correlation heatmap saved to: {heatmap_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error creating visualizations: {e}")
            return False
    
    def run_complete_analysis(self):
        """Run all analysis steps"""
        print("="*80)
        print("STARTING COMPLETE STATE PATTERN ANALYSIS")
        print("="*80)
        
        # Step 1: Load data
        if not self.load_data():
            return
        
        # Step 2-7: Run analyses
        session_counts, state_counts = self.analyze_basic_stats()
        returns_by_state, best_states = self.calculate_returns_analysis()
        transition_matrix = self.analyze_state_transitions()
        session_results = self.analyze_session_patterns()
        best_patterns = self.find_high_probability_patterns()
        playbook = self.create_trading_playbook(best_states, best_patterns)
        
        # Step 8: Visualizations
        self.create_visualizations()
        
        # Final summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*80)
        print(f"Total data points analyzed: {len(self.df)}")
        print(f"Time period covered: {self.df['Timestamp'].min().date()} to {self.df['Timestamp'].max().date()}")
        print(f"Unique states identified: {len(state_counts)}")
        
        if best_states is not None and len(best_states) > 0:
            best_state = best_states.iloc[0]
            print(f"\n★ BEST STATE PATTERN:")
            print(f"   State: {best_states.index[0]}")
            print(f"   Samples: {int(best_state['Return_15m_count'])}")
            print(f"   Win Rate: {best_state['WinRate_15m']:.1f}%")
            print(f"   Avg Return: {best_state['Return_15m_mean']:.2f} pips")
            print(f"   Sharpe: {best_state['Sharpe_15m']:.2f}")
        
        if best_patterns is not None and len(best_patterns) > 0:
            best_pattern = best_patterns.iloc[0]
            print(f"\n★ BEST SPECIFIC PATTERN:")
            print(f"   Pattern: {best_pattern['pattern']}")
            print(f"   Samples: {int(best_pattern['samples'])}")
            print(f"   Win Rate: {best_pattern['win_rate_15m']:.1f}%")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("1. Review TradingPlaybook.csv for specific trading rules")
        print("2. Check the PNG files for visual patterns")
        print("3. Run MQL5 EA for more data collection")
        print("4. Re-run analysis with more data for better accuracy")
        print("="*80)
        
        return {
            'data': self.df,
            'best_states': best_states,
            'best_patterns': best_patterns,
            'playbook': playbook
        }

# Quick analysis function
def quick_analysis(data_path=None):
    """Run quick analysis on existing data"""
    analyzer = StatePatternAnalyzer(data_path)
    results = analyzer.run_complete_analysis()
    return results

# Command line interface
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("UNIVERSAL STATE PATTERN ANALYZER")
    print("="*80)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Use default path from your MT5 installation
        data_path = r"C:\Users\davidug.SIFAXGROUP\AppData\Roaming\MetaQuotes\Terminal\BA78EA1631820D7AEF052166A20E7B1A\MQL5\Files\UniversalStateData.csv"
    
    print(f"Data path: {data_path}")
    print("\nStarting analysis...")
    
    # Run analysis
    try:
        results = quick_analysis(data_path)
        print("\n✓ Analysis completed successfully!")
        
        # Wait for user input
        input("\nPress Enter to exit...")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure MQL5 EA has been running and generated data")
        print("2. Check file path is correct")
        print("3. Ensure Python has pandas, matplotlib, seaborn, scikit-learn installed")
        print("\nTo install dependencies: pip install pandas matplotlib seaborn scikit-learn")
        
        input("\nPress Enter to exit...")
