# core/equity_curve_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import os

TRADE_LOG_PATH = 'journal/trade_log.csv'

def load_trade_log():
    if not os.path.exists(TRADE_LOG_PATH):
        print("Trade log not found.")
        return pd.DataFrame()
    return pd.read_csv(TRADE_LOG_PATH, parse_dates=['date_time'])

def generate_equity_curve(df):
    if df.empty:
        print("No trades to plot.")
        return

    df = df.sort_values('date_time')

    # Assume each trade's "percent_gain_loss" is applied to a starting balance
    starting_balance = 100_000  # Example: 100k USD
    balance = [starting_balance]

    for pct in df['percent_gain_loss']:
        new_balance = balance[-1] * (1 + pct / 100)
        balance.append(new_balance)

    df['equity_curve'] = balance[1:]

    plt.figure(figsize=(10,6))
    plt.plot(df['date_time'], df['equity_curve'], marker='o', linestyle='-')
    plt.title("Zanzibar Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Balance (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_trade_log()
    generate_equity_curve(df)