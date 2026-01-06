# CheckData.py - Quick data checker
# Place in same folder as analyzer

import pandas as pd
import os

def check_mt5_data():
    """Quick check of MT5 data files"""
    # Your file paths
    base_path = r"C:\Users\davidug.SIFAXGROUP\AppData\Roaming\MetaQuotes\Terminal\BA78EA1631820D7AEF052166A20E7B1A\MQL5\Files"
    
    files = [
        "UniversalStateData.csv",
        "TradePatterns.csv",
        "StateTransitions.csv"
    ]
    
    print("Checking MT5 Data Files...")
    print("="*50)
    
    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\n✓ {file}:")
                print(f"   Records: {len(df)}")
                print(f"   Columns: {len(df.columns)}")
                print(f"   Size: {os.path.getsize(file_path) / 1024:.1f} KB")
                
                if len(df) > 0:
                    print(f"   First record: {df.iloc[0]['Timestamp'] if 'Timestamp' in df.columns else 'N/A'}")
                    print(f"   Last record: {df.iloc[-1]['Timestamp'] if 'Timestamp' in df.columns else 'N/A'}")
                    
            except Exception as e:
                print(f"\n✗ {file}: Error reading - {e}")
        else:
            print(f"\n✗ {file}: File not found")
    
    print("\n" + "="*50)
    print("RECOMMENDED ACTIONS:")
    
    # Check if enough data exists
    main_file = os.path.join(base_path, "UniversalStateData.csv")
    if os.path.exists(main_file):
        df = pd.read_csv(main_file)
        if len(df) < 100:
            print("⚠ Need more data! Run MQL5 EA for at least 24 hours")
        else:
            print("✓ Enough data for analysis! Run UniversalStateAnalyzer.py")
    else:
        print("⚠ No data file found! Run MQL5 EA first")

if __name__ == "__main__":
    check_mt5_data()
    input("\nPress Enter to exit...")
