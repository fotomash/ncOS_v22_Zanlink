# Zanzibar v5.1 Core Module
# Version: 5.1.0
# Module: risk_model.py
# Description: Provides trade risk calculations including adjusted SL/TP, risk amount, and position sizing.

# risk_model.py

def calculate_trade_risk(entry: float, sl: float, r_multiple: float = 3.0,
                          account_size: float = 100_000, risk_pct: float = 1.0,
                          spread_points: float = 0.0) -> dict:
    """Calculate TP, lot size, and adjusted SL/TP based on entry and R-multiple."""
    spread_buffer = spread_points / 10
    sl_adjusted = sl - spread_buffer
    points_risked = entry - sl_adjusted
    tp = entry + (r_multiple * points_risked)
    risk_usd = account_size * (risk_pct / 100)
    lot_size = round(risk_usd / (points_risked * 100), 2)  # XAUUSD pip value = $100/lot

    return {
        "entry": round(entry, 2),
        "sl": round(sl_adjusted, 2),
        "tp": round(tp, 2),
        "points_risked": round(points_risked, 2),
        "r_multiple": r_multiple,
        "risk_usd": round(risk_usd, 2),
        "lot_size": lot_size
    }
