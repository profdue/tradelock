#!/usr/bin/env python3
"""
analyze_states.py
ONE FILE - Analyzes state log, finds persistent states
NO TRADING - Just discovery
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
import os

class StateAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.states = {}
        
    def run_full_analysis(self):
        """One function call does everything"""
        print("1. Loading data...")
        self.clean_data()
        
        print("2. Clustering states...")
        self.cluster_states()
        
        print("3. Analyzing persistence...")
        self.analyze_persistence()
        
        print("4. Finding non-reversal states...")
        results = self.find_non_reversal_states()
        
        print("5. Saving results...")
        self.save_results(results)
        
        return results
    
    def clean_data(self):
        """Basic cleanup"""
        self.df['Time'] = pd.to_datetime(self.df['Time'])
        self.df = self.df.sort_values('Time')
        self.df = self.df.drop_duplicates()
    
    def cluster_states(self):
        """Group by the 4 dimensions"""
        # Create cluster key
        self.df['Cluster'] = self.df[
            ['Volatility', 'Control', 'Participation', 'TimePressure']
        ].apply(lambda x: '|'.join(x), axis=1)
    
    def analyze_persistence(self):
        """How long does each state last?"""
        self.df['StateChange'] = self.df['Cluster'] != self.df['Cluster'].shift()
        self.df['StateID'] = self.df['StateChange'].cumsum()
        
        # Calculate duration for each state occurrence
        state_stats = self.df.groupby(['StateID', 'Cluster']).agg({
            'Time': ['min', 'max', 'count']
        }).reset_index()
        
        state_stats.columns = ['StateID', 'Cluster', 'Start', 'End', 'Bars']
        state_stats['Hours'] = state_stats['Bars']
        
        self.state_stats = state_stats
    
    def find_non_reversal_states(self, min_occurrences=15, min_persistence=0.7):
        """Find states where the other side doesn't fight back"""
        results = []
        
        for cluster, group in self.df.groupby('Cluster'):
            occurrences = len(group['StateID'].unique())
            
            if occurrences < min_occurrences:
                continue
            
            # Calculate persistence (how often it continues vs reverses)
            persistence_score = self.calculate_persistence_score(cluster)
            
            if persistence_score >= min_persistence:
                # Analyze what breaks this state
                break_conditions = self.analyze_break_conditions(cluster)
                
                results.append({
                    'State': cluster,
                    'Occurrences': occurrences,
                    'PersistenceScore': round(persistence_score, 3),
                    'AvgDurationHours': round(group.groupby('StateID')['Time'].count().mean(), 1),
                    'BreakConditions': break_conditions,
                    'ExampleTimestamp': group['Time'].iloc[0].strftime('%Y-%m-%d %H:%M')
                })
        
        return pd.DataFrame(results).sort_values('PersistenceScore', ascending=False)
    
    def calculate_persistence_score(self, cluster):
        """Simple persistence metric"""
        cluster_states = self.state_stats[self.state_stats['Cluster'] == cluster]
        
        if len(cluster_states) < 2:
            return 0.0
        
        # How often does this state last more than 1 bar?
        multi_bar = sum(cluster_states['Bars'] > 1)
        total = len(cluster_states)
        
        return multi_bar / total
    
    def analyze_break_conditions(self, cluster):
        """What breaks this state?"""
        cluster_df = self.df[self.df['Cluster'] == cluster]
        breaks = []
        
        # Find transitions FROM this state
        for i in range(len(cluster_df)-1):
            if cluster_df['Cluster'].iloc[i] != cluster_df['Cluster'].iloc[i+1]:
                next_state = cluster_df['Cluster'].iloc[i+1]
                breaks.append(next_state)
        
        # Most common breaks
        if breaks:
            from collections import Counter
            most_common = Counter(breaks).most_common(2)
            return [state for state, _ in most_common]
        
        return ["Unknown"]
    
    def save_results(self, results):
        """Save everything to one JSON file"""
        output = {
            'analysis_date': datetime.now().isoformat(),
            'total_observations': len(self.df),
            'unique_states': len(self.df['Cluster'].unique()),
            'states_found': results.to_dict('records'),
            'summary': {
                'high_persistence_states': len(results[results['PersistenceScore'] >= 0.8]),
                'avg_persistence': round(results['PersistenceScore'].mean(), 3),
                'most_common_state': results.iloc[0]['State'] if len(results) > 0 else None
            }
        }
        
        with open('state_discovery_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        # Also save as CSV for quick viewing
        results.to_csv('validated_states.csv', index=False)
        
        print(f"\n✅ Analysis complete!")
        print(f"   Observations: {len(self.df)}")
        print(f"   States found: {len(results)}")
        print(f"   Results saved to: state_discovery_results.json")

# ONE COMMAND TO RUN EVERYTHING
if __name__ == "__main__":
    # Change this to your CSV file path
    CSV_FILE = "StateLog.csv"  # From MT5
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ File not found: {CSV_FILE}")
        print("   First run the MT5 StateObserver EA to generate data")
    else:
        analyzer = StateAnalyzer(CSV_FILE)
        results = analyzer.run_full_analysis()
        
        # Print top states
        print("\n" + "="*60)
        print("TOP PERSISTENT STATES (Other side doesn't fight)")
        print("="*60)
        for i, row in results.head(5).iterrows():
            print(f"\n{i+1}. {row['State']}")
            print(f"   Persistence: {row['PersistenceScore']*100:.1f}%")
            print(f"   Avg Duration: {row['AvgDurationHours']} hours")
            print(f"   Breaks into: {', '.join(row['BreakConditions'])}")
