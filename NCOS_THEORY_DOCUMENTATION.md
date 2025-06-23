# ncOS Trading Theory Documentation

## Overview
This document outlines the integrated trading theories implemented in ncOS v22, combining Wyckoff Method, Smart Money Concepts (SMC), and proprietary strategies from expert traders.

## 1. Wyckoff Method Integration

### Accumulation Schematic
- **Phase A**: Preliminary Support (PS) and Selling Climax (SC)
- **Phase B**: Automatic Rally (AR) and Secondary Test (ST)
- **Phase C**: Spring or Shakeout
- **Phase D**: Last Point of Support (LPS) and Sign of Strength (SOS)
- **Phase E**: Markup Phase begins

### Key Implementation from Chart Data
Based on XAUUSD analysis:
- SC Level: 3224.53
- Spring Level: 3225.78
- LPS Level: 3225.27
- SOS Level: 3227.01
- Target: 3228.24

### Volume Analysis
- SC Volume: 78 (High selling pressure)
- Test Volume: 69 (Decreasing - bullish)
- SOS Volume: 74 (Increasing on rally - bullish)

## 2. Smart Money Concepts (SMC)

### Order Blocks (OB)
- Bullish OB: Last bearish candle before impulsive bullish move
- Used as support/resistance levels
- Entry trigger: "CHoCH + OB tap"

### Imbalances
- Fair Value Gaps (FVG)
- Buy Side Imbalance (BISI)
- Sell Side Imbalance (SIBI)

### Change of Character (CHoCH)
- Indicates potential trend reversal
- Combined with OB for high-probability entries

## 3. MAZ Trading Strategy

### Core Principles
1. **Unmitigated Weak High/Low**
   - Identify levels that haven't been tested
   - Wait for price to return to these levels

2. **Multi-Timeframe Analysis**
   - H1 for zone identification
   - M1 for entry confirmation

3. **Risk Management**
   - 1% risk per trade
   - Minimum 1:2 RR
   - Partial profits at 50%, 30%, 20%

## 4. Day Trading Tops & Bottoms

### Key Indicators
- Hidden/Iceberg Orders
- Stop Runs
- Volume Profile Shifts
- Order Flow Imbalances

### Entry Criteria
1. Large iceberg order presence
2. Stop run completion
3. Volume confirmation
4. Price rejection from level

## 5. Integration Rules

### Confluence Requirements (Minimum 3)
- [ ] Wyckoff phase alignment
- [ ] SMC structure (OB/FVG/CHoCH)
- [ ] Volume confirmation
- [ ] Multi-timeframe alignment
- [ ] Order flow support

### Entry Checklist
1. Identify primary theory setup (Wyckoff/SMC/MAZ)
2. Check for additional confluences
3. Confirm with lower timeframe
4. Set risk parameters
5. Execute with discipline

## 6. Risk Management

### Position Sizing
```
Position Size = (Account Balance × Risk %) / (Entry - Stop Loss)
```

### Stop Loss Placement
- Wyckoff: Below Spring level
- SMC: Below Order Block
- MAZ: Below unmitigated level

### Take Profit Strategy
- TP1: 50% at 1:1 RR
- TP2: 30% at 1:2 RR
- TP3: 20% runner

## 7. Implementation in ncOS

### Automated Detection
```python
# Example usage
from ncos_theory_integration import TheoryIntegrationEngine

engine = TheoryIntegrationEngine()
setup = engine.analyze_wyckoff_smc_setup(price_data, chart_config)

if setup and setup.risk_reward >= 2.0:
    execute_trade(setup)
```

### Real-time Monitoring
- Continuous pattern scanning
- Alert generation on confluences
- Automated risk calculation
- Trade execution via API

## 8. Performance Metrics

### Key Indicators to Track
- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Average RR Achieved

### Optimization Parameters
- Timeframe selection
- Confluence weighting
- Risk per trade
- Partial profit levels

## Conclusion
The integration of these theories provides a robust framework for identifying high-probability trading opportunities. The combination of Wyckoff's market cycle analysis, SMC's institutional perspective, and proven strategies from successful traders creates a comprehensive trading system.
