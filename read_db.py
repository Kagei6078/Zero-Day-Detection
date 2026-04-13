import sqlite3
import pandas as pd
import os

# --- 1. CONFIGURATION ---
DB_NAME = "security_logs.db"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_report():
    if not os.path.exists(DB_NAME):
        print(f"❌ Error: {DB_NAME} not found. Have you run live_detection.py yet?")
        return

    # Connect to SQLite
    conn = sqlite3.connect(DB_NAME)
    
    try:
        # 2. FETCH SUMMARY STATISTICS
        total_events = pd.read_sql_query("SELECT COUNT(*) as count FROM network_events", conn).iloc[0]['count']
        attack_events = pd.read_sql_query("SELECT COUNT(*) as count FROM network_events WHERE label LIKE '%ATTACK%'", conn).iloc[0]['count']
        
        # 3. FETCH LATEST 20 EVENTS
        # We sort by ID descending to see the most recent activity at the top
        df = pd.read_sql_query("SELECT * FROM network_events ORDER BY id DESC LIMIT 20", conn)

        # --- DISPLAY ---
        clear_screen()
        print("="*60)
        print("🛡️  AI NETWORK INTRUSION DETECTION SYSTEM - LOG REPORT")
        print("="*60)
        print(f"📊 SYSTEM SUMMARY:")
        print(f"   Total Packets Analyzed: {total_events}")
        print(f"   Threats Detected:       {attack_events}")
        print("-" * 60)
        
        if df.empty:
            print("\n ⏳ Database exists but no events logged yet.")
            print("    (Note: The sniffer must finish calibration to log events)")
        else:
            print("🕒 LATEST 20 EVENTS:")
            # Format the dataframe for pretty printing
            print(df.to_string(index=False))
        
        print("="*60)
        print("💡 Tip: Run 'live_detection.py' in another window to see live updates.")

    except Exception as e:
        print(f"❌ Error reading database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_report()