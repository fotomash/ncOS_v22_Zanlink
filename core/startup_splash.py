# core/startup_splash.py

import time
import os
from datetime import datetime

VERSION = "5.2.0"

# Configuration
LOGO_PATH = "assets/branding/captain_zanzibar.gif"  # Adjust path as needed
ANIMATION_DISPLAY_SECONDS = 3  # How long to show the animation

# Updated splash text with version and dynamic timestamp
TEXT_SPLASH = f"""

    🌊🚀🧭
    WELCOME ABOARD, CAPTAIN ZANZIBAR
    Chart the Unknown. Conquer the Impossible.

    ZANALYTICS VERSION: {VERSION}
    INITIALIZED AT: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

"""

def display_splash():
    print("\nLoading Captain Zanzibar...")
    
    # Check if logo animation file exists
    if os.path.exists(LOGO_PATH):
        try:
            from PIL import Image
            img = Image.open(LOGO_PATH)
            img.show()
            time.sleep(ANIMATION_DISPLAY_SECONDS)
            img.close()
        except Exception as e:
            print(f"[Splash] Couldn't load animated logo. Reason: {e}")
            print(TEXT_SPLASH)
    else:
        # Fallback to text splash
        print(TEXT_SPLASH)

    print("🔄 Data will be loaded from 'tick_data/m1' and resampled to 'tick_data/htf' for processing.")

if __name__ == "__main__":
    display_splash()