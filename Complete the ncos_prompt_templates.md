# Complete the ncos_prompt_templates.py file

# ... (previous content) ...

### Volume Analysis:
- **Climax Volume**: Extremely high volume at turning points
- **Test Volume**: Should be lower than climax volume
- **Breakout Volume**: Should expand on SOS

---

## 3. Smart Money Concepts

### Order Blocks (OB)
- Last opposing candle before impulsive move
- Represents institutional orders
- Entry: Within OB range
- Stop: Beyond OB

### Fair Value Gaps (FVG)
- Price inefficiencies between candles
- 3-candle pattern with gap
- High probability retracement zones

### Break of Structure (BOS)
- Price breaking previous swing high/low
- Confirms trend change
- Entry on retest

### Liquidity Pools
- Equal highs/lows
- Stop loss clusters
- Targets for smart money

---

## 4. MAZ Strategy Implementation

### Core Concept:
- Find unmitigated (untested) levels
- These act as magnets for price
- Higher timeframe = stronger level

### Rules:
1. Identify significant high/low
2. Confirm no retest occurred
3. Wait for price approach
4. Enter on confirmation

---

## 5. Hidden Order Detection

### Iceberg Orders:
- Large orders split into smaller pieces
- Detected by:
  - Consistent buying/selling at level
  - Price absorption
  - Volume clustering

### Detection Method:
1. Monitor tick data
2. Identify price levels with unusual volume
3. Calculate absorption rate
4. Confirm with price action

---

## 6. Integration and Confluence

### Confluence Factors:
1. Wyckoff phase alignment
2. SMC level present
3. Unmitigated MAZ level
4. Hidden order activity

### Signal Strength:
- 2 factors = Medium (60-70%)
- 3 factors = Strong (70-85%)
- 4 factors = Very Strong (85%+)

---

## 7. Risk Management

### Position Sizing:
- 1-2% risk per trade
- Scale in at confluence zones
- Partial profits at targets

### Stop Loss Placement:
- Beyond structure
- Outside order blocks
- Below/above liquidity

---

## 8. Live Trading Examples

### Example 1: Wyckoff Spring + SMC OB
- Wyckoff Phase C spring detected
- Bullish OB at spring level
- Entry: Within OB
- Stop: Below spring low
- Target: Previous range high

### Example 2: MAZ + Hidden Orders
- Unmitigated high from 4H
- Hidden sell orders detected
- Entry: Approach of level
- Stop: Above high
- Target: Previous support

---

## Implementation in ncOS

```python
# Quick setup
engine = AdvancedTheoryEngine()
signals = engine.generate_integrated_signals(ohlcv_data, tick_data, current_price)

# Process top signal
if signals:
    top_signal = signals[0]
    print(f"Entry: {top_signal['entry']}")
    print(f"Stop: {top_signal['stop_loss']}")
    print(f"Targets: {top_signal['take_profits']}")