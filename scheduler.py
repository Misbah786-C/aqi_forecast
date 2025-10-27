import schedule
import time
import subprocess
import os

def run_backfill():
    backfill_path = os.path.join(os.getcwd(), "features", "features", "backfill.py")
    print(f"⏳ Running {backfill_path}...")
    
    try:
        subprocess.run(["python", backfill_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Backfill failed: {e}")
        return

    # Show current dataset size
    csv_path = os.path.join(os.getcwd(), "data", "features", "training_dataset.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"📊 Current dataset size: {len(df)} rows")
    else:
        print("⚠️ No dataset file found yet.")

# Schedule every 2 minutes for testing
schedule.every(15).minutes.do(run_backfill)

print("✅ Scheduler started. Running backfill every 15 minutes...")

while True:
    schedule.run_pending()
    time.sleep(10)
