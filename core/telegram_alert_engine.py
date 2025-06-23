# telegram_alert_engine.py

import requests
import json

def send_telegram_alert(entry_data, config_path="config/webhook_settings.json"):
    """
    Sends a formatted message to a Telegram bot based on entry confirmation.
    Expects a config file with token and chat_id.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if not config.get("webhook_enabled", False):
            print("[TELEGRAM] Webhook disabled in config.")
            return

        token = config["bot_token"]
        chat_id = config["chat_id"]
        endpoint = f"https://api.telegram.org/bot{token}/sendMessage"

        message = format_trade_message(entry_data)
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

        response = requests.post(endpoint, data=payload)
        if response.status_code != 200:
            print(f"[TELEGRAM ERROR] {response.status_code}: {response.text}")
        else:
            print("[TELEGRAM] Alert sent.")
    except Exception as e:
        print(f"[TELEGRAM EXCEPTION] {e}")


def send_simple_summary_alert(summary_text, config_path="config/webhook_settings.json"):
    """
    Send a plain summary line to Telegram — e.g., from AI agents or journaling.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if not config.get("webhook_enabled", False):
            print("[TELEGRAM] Webhook disabled in config.")
            return

        token = config["bot_token"]
        chat_id = config["chat_id"]
        endpoint = f"https://api.telegram.org/bot{token}/sendMessage"

        payload = {"chat_id": chat_id, "text": summary_text, "parse_mode": "HTML"}

        response = requests.post(endpoint, data=payload)
        if response.status_code != 200:
            print(f"[TELEGRAM ERROR] {response.status_code}: {response.text}")
        else:
            print("[TELEGRAM] Summary alert sent.")
    except Exception as e:
        print(f"[TELEGRAM EXCEPTION] {e}")


def format_trade_message(entry_data):
    """
    Format a detailed message for any trade entry (scalp or macro).
    """
    return f"""
📘 <b>Trade Log – {entry_data.get('symbol')} {entry_data.get('entry_type', 'LIMIT').upper()} (v10 Structured Play)</b>

<b>Date:</b> {entry_data.get('date', '25 May 2025')}  
<b>Time Activated:</b> {entry_data.get('time', '02:30 UK')}  
<b>Strategy Profile:</b> v10.md + inducement_sweep_poi + Wyckoff Phase E + Z-bar  

────

🔧 <b>Trade Parameters</b>
<b>Asset:</b> {entry_data.get('symbol')}
<b>Entry:</b> {entry_data.get('entry_price')}
<b>Stop Loss:</b> {entry_data.get('sl')}
<b>Take Profit 1:</b> {entry_data.get('tp')}
<b>Take Profit 2:</b> {entry_data.get('tp2', '—')}
<b>R:R (TP1):</b> {entry_data.get('r_multiple', '—')}
<b>R:R (TP2):</b> {entry_data.get('r_multiple_tp2', '—')}
<b>Risk Type:</b> {entry_data.get('risk_type', '—')}
<b>Conviction:</b> {entry_data.get('conviction', '—')}

────

🧠 <b>Strategic Rationale (v10 logic stack)</b>
Structure: {entry_data.get('rationale_structure', '—')}
Z-bar Context: {entry_data.get('rationale_zbar', '—')}
Wyckoff Phase Analysis: {entry_data.get('rationale_wyckoff', '—')}
Volume + Momentum: {entry_data.get('rationale_volume', '—')}

────

🛑 <b>Invalidation Logic</b>
{entry_data.get('invalidation', '—')}

────

✅ <b>Execution Rules</b>
{entry_data.get('execution_rules', '—')}

────

📎 <b>Journal Notes</b>
{entry_data.get('journal_notes', '—')}

<b>Logged by:</b> {entry_data.get('logger', 'xanalytics_v10')}  
<b>Setup Classification:</b> {entry_data.get('setup_type', '—')}  
<b>Linked Play:</b> {entry_data.get('linked_play', '—')}
""".strip()
