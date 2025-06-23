
import pandas as pd

def get_final_signal_from_data(df: pd.DataFrame):
    """
    This function contains the core trading logic.
    It analyzes the provided DataFrame and returns a trading signal.

    For this example, we'll use a very simple logic:
    - If the last price is higher than the first price, signal BUY.
    - If the last price is lower than the first price, signal SELL.
    - Otherwise, HOLD.

    This can be replaced with your more complex SMC/Wyckoff logic.
    """
    if df.empty:
        return {
            "signal": "HOLD",
            "confidence": 0.1,
            "reason": "Logic Error: No data provided to analyze."
        }

    try:
        # Use 'last' column if available, otherwise 'bid'
        price_col = 'last' if 'last' in df.columns else 'bid'

        first_price = df[price_col].iloc[0]
        last_price = df[price_col].iloc[-1]

        if last_price > first_price:
            signal = "BUY"
            confidence = 0.65
            reason = f"Analysis of {len(df)} ticks shows upward momentum. Last price ({last_price}) > First price ({first_price})."
        elif last_price < first_price:
            signal = "SELL"
            confidence = 0.65
            reason = f"Analysis of {len(df)} ticks shows downward momentum. Last price ({last_price}) < First price ({first_price})."
        else:
            signal = "HOLD"
            confidence = 0.5
            reason = "Market is flat. No significant price change detected."

        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason
        }
    except Exception as e:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": f"An error occurred during analysis: {str(e)}"
        }
