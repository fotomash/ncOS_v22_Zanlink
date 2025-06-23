# core/startup_loading.py

import sys
import time
from datetime import datetime

VERSION = "5.2.0"

def animated_loading_screen():
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    captain_motto = (
        f"\n🧭 Captain Zanzibar: Chart the Unknown. Conquer the Impossible.\n"
        f"Version: {VERSION} | Launched at: {timestamp} UTC\n"
    )
    print(captain_motto)

    loading_phrases = [
        "Charting maps",
        "Scouting tides",
        "Aligning stars",
        "Ready for launch"
    ]

    for phrase in loading_phrases:
        for i in range(4):
            sys.stdout.write(f"\r{phrase}{'.' * i}   ")
            sys.stdout.flush()
            time.sleep(0.4)
        time.sleep(0.5)

    print("\n🚀 Systems Ready. Captain Zanzibar Awaiting Orders.\n")
    print("🔄 Fetching raw M1 and resampling to HTF via DataPipeline...")

if __name__ == "__main__":
    animated_loading_screen()
