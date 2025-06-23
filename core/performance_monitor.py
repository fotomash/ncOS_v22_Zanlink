# core/performance_monitor.py

import pandas as pd
import os
import datetime

# Configurable thresholds
STOP_TRADING_AFTER_N_LOSSES = 3
MIN_WINRATE_THRESHOLD = 0.45  # Example: if below 45% winrate, alert
MAX_ALLOWED_DRAWDOWN_PERCENT = 5  # Example: 5% daily max loss

TRADE_LOG_PATH = 'journal/trade_log.csv'
SESSION_LOG_PATH = 'journal/session_log.csv'
REPORTS_DIR = 'journal/reports/'

# Ensure the reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_trade_log():
    if not os.path.exists(TRADE_LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(TRADE_LOG_PATH, parse_dates=['date_time'])
    return df

def today_trades(df):
    today = datetime.datetime.now().date()
    return df[df['date_time'].dt.date == today]

def review_today_performance():
    df = load_trade_log()
    if df.empty:
        return "No trades today."

    df_today = today_trades(df)
    if df_today.empty:
        return "No trades recorded for today."

    wins = (df_today['outcome'] == 'Win').sum()
    losses = (df_today['outcome'] == 'Loss').sum()
    breakevens = (df_today['outcome'] == 'Breakeven').sum()

    total_trades = len(df_today)
    winrate = wins / total_trades if total_trades else 0
    net_r = df_today['risk_reward_tp1'].where(df_today['outcome'] == 'Win', -1).sum()

    alerts = []

    if losses >= STOP_TRADING_AFTER_N_LOSSES:
        alerts.append(f"🚨 ALERT: {losses} losses today. Recommend stopping trading.")
    if winrate < MIN_WINRATE_THRESHOLD:
        alerts.append(f"🚨 WARNING: Winrate low at {winrate:.2%}.")
    if df_today['percent_gain_loss'].sum() <= -MAX_ALLOWED_DRAWDOWN_PERCENT:
        alerts.append(f"🚨 CRITICAL: Daily loss exceeds {MAX_ALLOWED_DRAWDOWN_PERCENT}%.")

    summary = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'breakevens': breakevens,
        'winrate': f"{winrate:.2%}",
        'net_r': round(net_r, 2),
        'alerts': alerts
    }
    return summary

def print_performance_summary(summary):
    print("\n=== Zanzibar Daily Performance Summary ===")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']}")
    print(f"Losses: {summary['losses']}")
    print(f"Breakevens: {summary['breakevens']}")
    print(f"Winrate: {summary['winrate']}")
    print(f"Net R-Multiple: {summary['net_r']}")

    if summary['alerts']:
        print("\n--- Alerts ---")
        for alert in summary['alerts']:
            print(alert)
    else:
        print("\nAll systems nominal. 🌟")

def save_daily_report(summary):
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    report_path = os.path.join(REPORTS_DIR, f"performance_report_{today_str}.md")

    with open(report_path, 'w') as f:
        f.write(f"# Zanzibar Performance Report - {today_str}\n\n")
        f.write(f"**Total Trades:** {summary['total_trades']}\n\n")
        f.write(f"**Wins:** {summary['wins']}\n\n")
        f.write(f"**Losses:** {summary['losses']}\n\n")
        f.write(f"**Breakevens:** {summary['breakevens']}\n\n")
        f.write(f"**Winrate:** {summary['winrate']}\n\n")
        f.write(f"**Net R-Multiple:** {summary['net_r']}\n\n")
        if summary['alerts']:
            f.write("## Alerts\n")
            for alert in summary['alerts']:
                f.write(f"- {alert}\n")
        else:
            f.write("All systems nominal. \ud83c\udf1f\n")

# Example usage:
if __name__ == "__main__":
    summary = review_today_performance()
    if isinstance(summary, dict):
        print_performance_summary(summary)
        save_daily_report(summary)
    else:
        print(summary)