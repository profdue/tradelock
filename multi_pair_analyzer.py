import pandas as pd
import json
import os
from datetime import datetime

class MultiPairAnalyzer:
    def __init__(self):
        self.pairs = ['GOLD#', 'EURUSD', 'GBPUSD']
        self.results = {}
        
    def analyze_all(self):
        """Analyze all pairs and find cross-pair patterns"""
        print("="*60)
        print("MULTI-PAIR STATE ANALYZER")
        print("="*60)
        
        all_states = {}
        
        for pair in self.pairs:
            filename = f"{pair.replace('#', '')}_StateLog.csv"
            if os.path.exists(filename):
                print(f"\n📊 Analyzing {pair}...")
                df = pd.read_csv(filename)
                
                # Create state cluster
                df['State'] = df['Volatility'] + '|' + df['Control'] + '|' + df['Participation'] + '|' + df['TimePressure']
                
                # Store in dictionary
                all_states[pair] = {
                    'data': df,
                    'states': list(df['State'].unique()),
                    'current_state': df['State'].iloc[-1] if len(df) > 0 else None,
                    'count': len(df)
                }
                
                print(f"   Records: {len(df)}")
                print(f"   Current State: {all_states[pair]['current_state']}")
            else:
                print(f"\n⚠️  No data for {pair} - file not found: {filename}")
        
        return all_states
    
    def find_correlated_states(self, states_data):
        """Find states that occur simultaneously across pairs"""
        print("\n" + "="*60)
        print("CORRELATED STATES ANALYSIS")
        print("="*60)
        
        correlations = []
        
        # Get common timestamps
        common_times = None
        for pair, data in states_data.items():
            if 'data' in data:
                times = set(data['data']['Time'])
                if common_times is None:
                    common_times = times
                else:
                    common_times = common_times.intersection(times)
        
        if common_times and len(common_times) > 0:
            print(f"\nAnalyzing {len(common_times)} common timestamps...")
            
            # Create DataFrame for each pair's states at common times
            state_matrix = {}
            for pair, data in states_data.items():
                if 'data' in data:
                    df = data['data']
                    df_common = df[df['Time'].isin(common_times)]
                    state_matrix[pair] = dict(zip(df_common['Time'], df_common['State']))
            
            # Find correlations
            for time in sorted(common_times)[-10:]:  # Last 10 common times
                states_at_time = {}
                for pair in self.pairs:
                    if pair in state_matrix and time in state_matrix[pair]:
                        states_at_time[pair] = state_matrix[pair][time]
                
                if len(states_at_time) == len(self.pairs):
                    correlations.append({
                        'time': time,
                        'states': states_at_time
                    })
                    
                    print(f"\n{time}")
                    for pair, state in states_at_time.items():
                        print(f"  {pair:8}: {state}")
        
        return correlations
    
    def find_persistent_states(self, states_data, min_occurrences=5):
        """Find persistent states across all pairs"""
        print("\n" + "="*60)
        print("PERSISTENT STATES BY PAIR")
        print("="*60)
        
        persistent_states = {}
        
        for pair, data in states_data.items():
            if 'data' in data:
                df = data['data']
                
                if len(df) > 1:
                    df['StateChange'] = df['State'] != df['State'].shift()
                    df['StateID'] = df['StateChange'].cumsum()
                    
                    # Calculate persistence
                    state_stats = df.groupby('StateID').agg({
                        'State': 'first',
                        'Time': 'count'
                    }).reset_index()
                    
                    state_stats.columns = ['StateID', 'State', 'Duration']
                    
                    # Find states that persist
                    for state in state_stats['State'].unique():
                        durations = state_stats[state_stats['State'] == state]['Duration']
                        persist_count = sum(durations > 1)
                        total_count = len(durations)
                        
                        if total_count >= min_occurrences:
                            persistence_ratio = persist_count / total_count
                            
                            if persistence_ratio >= 0.7:  # 70% persistence
                                if pair not in persistent_states:
                                    persistent_states[pair] = []
                                
                                persistent_states[pair].append({
                                    'state': state,
                                    'persistence': persistence_ratio,
                                    'avg_duration': durations.mean(),
                                    'occurrences': total_count
                                })
        
        # Display results
        for pair, states in persistent_states.items():
            if states:
                print(f"\n✅ {pair} - Persistent States Found:")
                for state_info in sorted(states, key=lambda x: x['persistence'], reverse=True)[:3]:
                    print(f"   • {state_info['state']}")
                    print(f"     Persistence: {state_info['persistence']*100:.1f}%")
                    print(f"     Avg Duration: {state_info['avg_duration']:.1f} hours")
                    print(f"     Occurrences: {state_info['occurrences']} times")
        
        return persistent_states
    
    def save_results(self, states_data, correlations, persistent_states):
        """Save all analysis results"""
        output = {
            'analysis_time': datetime.now().isoformat(),
            'summary': {},
            'pair_details': {},
            'correlations': correlations,
            'persistent_states': persistent_states
        }
        
        for pair, data in states_data.items():
            if 'data' in data:
                output['pair_details'][pair] = {
                    'records': len(data['data']),
                    'unique_states': len(data['states']),
                    'current_state': data['current_state'],
                    'states_list': data['states'][-10:]  # Last 10 states
                }
        
        with open('multi_pair_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        # Also create Excel summary
        excel_data = []
        for pair, states in persistent_states.items():
            for state_info in states:
                excel_data.append({
                    'Pair': pair,
                    'State': state_info['state'],
                    'Persistence_%': state_info['persistence'] * 100,
                    'Avg_Duration_Hours': state_info['avg_duration'],
                    'Occurrences': state_info['occurrences']
                })
        
        if excel_data:
            df_excel = pd.DataFrame(excel_data)
            df_excel.to_csv('persistent_states_summary.csv', index=False)
            print(f"\n📁 Results saved to:")
            print(f"   • multi_pair_results.json")
            print(f"   • persistent_states_summary.csv")

def main():
    analyzer = MultiPairAnalyzer()
    
    # Analyze all pairs
    states_data = analyzer.analyze_all()
    
    if len(states_data) > 0:
        # Find correlations
        correlations = analyzer.find_correlated_states(states_data)
        
        # Find persistent states
        persistent_states = analyzer.find_persistent_states(states_data)
        
        # Save results
        analyzer.save_results(states_data, correlations, persistent_states)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        if persistent_states:
            print("\n🎯 TRADABLE INSIGHTS:")
            print("Look for these football-like states (other side doesn't fight):")
            for pair, states in persistent_states.items():
                if states:
                    best_state = max(states, key=lambda x: x['persistence'])
                    print(f"\n{pair}:")
                    print(f"  When you see: {best_state['state']}")
                    print(f"  It persists: {best_state['persistence']*100:.1f}% of the time")
                    print(f"  Average duration: {best_state['avg_duration']:.1f} hours")
        else:
            print("\n⚠️  No persistent states found yet.")
            print("   Need more data (minimum 7 days recommended)")
    else:
        print("\n❌ No data found for any pair!")
        print("   Make sure the MT5 EA is running and collecting data.")

if __name__ == "__main__":
    main()
