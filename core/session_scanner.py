# Zanzibar v5.1 Core Module
# Version: 5.1.0
# Module: session_scanner.py
# Description: Scans designated trading pairs at scheduled times, logs session data, and triggers webhook alerts if entries or confluences are detected.

# session_scanner.py - ZANZIBAR Session Scanner with CLI, Logging, Alerts

import time
import schedule
import os
import csv
import json
import requests
import argparse
from datetime import datetime

from copilot_orchestrator import handle_price_check

# --- Webhook Alert Function ---
def send_webhook_alert(payload: dict):
    webhook_url = os.environ.get("SCANNER_WEBHOOK_URL")
    if not webhook_url:
        print("⚠️ No webhook URL found in environment (SCANNER_WEBHOOK_URL). Skipping alert.")
        return
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        if response.status_code == 200:
            print("📤 Alert sent successfully.")
        else:
            print(f"⚠️ Webhook failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Exception while sending webhook alert: {e}")

# --- Logging Utility ---
def log_session_result(result: dict, log_dir: str, format: str):
    os.makedirs(log_dir, exist_ok=True)
    log_date = datetime.utcnow().strftime("%Y-%m-%d")
    log_path = os.path.join(log_dir, f"session_scan_{log_date}.{format}")
    result_copy = result.copy()
    result_copy["log_timestamp"] = datetime.utcnow().isoformat()

    if format == "csv":
        header = ["log_timestamp", "pair", "timeframe", "target_time", "bos", "poi", "entry", "confluence"]
        row = [
            result_copy.get("log_timestamp"),
            result_copy.get("pair"),
            result_copy.get("timeframe"),
            result_copy.get("target_time"),
            result_copy.get("bos"),
            result_copy.get("poi"),
            result_copy.get("entry"),
            json.dumps(result_copy.get("confluence", {}))
        ]
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    elif format == "json":
        with open(log_path, "a") as f:
            f.write(json.dumps(result_copy) + "\n")
    else:
        print(f"❌ Unsupported log format: {format}")

# --- Scan Logic ---
def run_session_scan(pairs, tf, log_dir, log_format):
    print("\n🔍 Running Session Scan @", datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
    for pair in pairs:
        timestamp = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
        print(f"\n📈 Analyzing {pair} @ {timestamp_str} ({tf})...")
        result = handle_price_check(pair, timestamp_str, tf)

        if result.get("status") == "ok":
            print("✅ Result:")
            print(f"   BOS: {result['bos']}")
            print(f"   POI: {result['poi']}")
            print(f"   Entry: {result['entry']}")
            print(f"   Confluence: {result.get('confluence', {})}")
            log_session_result(result, log_dir, log_format)

            entry_triggered = "confirmed" in result.get("entry", "").lower()
            confluence_present = bool(result.get("confluence"))
            if entry_triggered or confluence_present:
                alert_payload = {
                    "pair": result.get("pair"),
                    "timeframe": result.get("timeframe"),
                    "timestamp": result.get("target_time"),
                    "bos": result.get("bos"),
                    "poi": result.get("poi"),
                    "entry": result.get("entry"),
                    "confluence": result.get("confluence", {})
                }
                send_webhook_alert(alert_payload)
        else:
            print("⚠️ Error:", result.get("message"))

# --- CLI Entrypoint ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZANZIBAR Session Scanner")
    parser.add_argument("--pairs", type=str, default="OANDA:GBP_USD,OANDA:EUR_USD",
                        help="Comma-separated list of symbols (default: GU,EU)")
    parser.add_argument("--tf", type=str, default="15m", help="Timeframe to scan (default: 15m)")
    parser.add_argument("--session-time", type=str, default="08:00", help="Time (UTC) to run scan daily")
    parser.add_argument("--run-now", action="store_true", help="Run scan once immediately and exit")
    parser.add_argument("--log-format", type=str, default="csv", choices=["csv", "json"], help="Log format")
    parser.add_argument("--log-dir", type=str, default="session_logs", help="Directory to store scan logs")
    args = parser.parse_args()

    pair_list = [p.strip() for p in args.pairs.split(",") if p.strip()]

    if args.run_now:
        print("🟢 Running session scanner in dry-run mode...")
        run_session_scan(pair_list, args.tf, args.log_dir, args.log_format)
    else:
        print(f"🟢 ZANZIBAR Scanner active — scheduled daily at {args.session_time} UTC")
        schedule.every().day.at(args.session_time).do(lambda: run_session_scan(pair_list, args.tf, args.log_dir, args.log_format))
        while True:
            schedule.run_pending()
            time.sleep(10)
