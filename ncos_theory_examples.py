# ncOS Theory Integration Examples

## Example 1: XAUUSD Wyckoff + SMC Setup

```python
# Real setup from the chart data
setup = {
    "pair": "XAUUSD",
    "timeframe": "1m",
    "theory": "Wyckoff Accumulation + SMC",
    "entry": 3225.27,  # LPS retest + OB tap
    "stop_loss": 3224.53,  # Below SC
    "take_profit": 3228.24,  # Projected target
    "risk_reward": 4.08,
    "confluences": [
        "Wyckoff Spring completed",
        "CHoCH confirmed",
        "Order Block tapped",
        "Rising volume on test",
        "IMB fill"
    ]
}
```

## Example 2: MAZ Strategy - USDCAD

```python
# Unmitigated weak high setup
maz_setup = {
    "pair": "USDCAD",
    "timeframe": "H1",
    "entry_zone": [1.3850, 1.3860],
    "trigger": "M15 bearish engulfing",
    "stop_loss": 1.3885,
    "targets": [
        1.3820,  # TP1 (50%)
        1.3790,  # TP2 (30%)
        1.3750   # TP3 (20%)
    ],
    "notes": "Unmitigated H1 supply zone"
}
```

## Example 3: Day Trading Bottom - ES Futures

```python
# Hidden order + stop run setup
bottom_catch = {
    "instrument": "ES",
    "time": "09:42 EST",
    "signal": "Large iceberg buy order",
    "confirmation": "Stop run below support",
    "entry": 4518.50,
    "stop": 4516.00,
    "target": 4528.50,  # 10 points
    "order_flow": {
        "iceberg_size": "500+ contracts",
        "stop_volume": "High",
        "recovery_speed": "Immediate"
    }
}
```

## Integration Code Example

```python
from ncos_theory_integration import TheoryIntegrationEngine
import pandas as pd

# Initialize engine
engine = TheoryIntegrationEngine()

# Load your price data
price_data = pd.read_csv('XAUUSD_ticks.csv')

# Define chart configuration
chart_config = {
    "sc": 3224.53,
    "spring": 3225.78,
    "lps": 3225.27,
    "sos": 3227.01,
    "entry": {"price": 3225.27},
    "sl": 3224.53,
    "tp": 3228.24,
    "confluence": ["IMB", "OB", "LPS retest", "Rising Volume"]
}

# Analyze for setup
setup = engine.analyze_wyckoff_smc_setup(price_data, chart_config)

if setup:
    print(f"Valid setup found: {setup.theory_type}")
    print(f"Entry: {setup.entry_price}")
    print(f"Risk/Reward: {setup.risk_reward:.2f}")
```
