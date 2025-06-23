# NCOS v24.1 - Phoenix Unified Architecture

## 🎯 System Overview

NCOS v24.1 "Phoenix-Unified" is a consolidated multi-agent trading framework that combines:
- Smart Money Concepts (SMC) analysis
- Wyckoff methodology 
- Institutional liquidity detection
- Multi-timeframe market structure analysis
- Advanced risk management and execution

## 🏗️ Core Architecture

### Orchestration Layer
- **Copilot Orchestrator**: Central routing and coordination hub
- **State Machine**: Manages system states and transitions
- **Data Pipeline**: Unified data flow from M1 to higher timeframes

### Analysis Engines
- **Market Structure Analyzer**: CHoCH, BOS, swing detection
- **Wyckoff Phase Detector**: Accumulation/distribution phases
- **Liquidity Engines**: Sweep detection, VWAP analysis, POI management
- **Impulse/Correction Detector**: Market phase identification

### Strategy Framework
- **Strategy Match Engine**: Rule-based strategy selection
- **Entry Executor**: Multi-variant execution logic (Inv, MAZ2, TMC, Mentfx)
- **Risk Calculator**: Position sizing and SL/TP calculation

### Enrichment Pipeline
- **SMC Enrichment**: Order blocks, FVGs, liquidity zones
- **Macro Sentiment**: Intermarket analysis and bias
- **Marker Enrichment**: Volume patterns and anomalies

## 🔄 Data Flow

1. **Ingestion**: M1 tick data → Multi-timeframe resampling
2. **Enrichment**: Apply SMC tags, indicators, Wyckoff analysis
3. **Analysis**: Structure detection, liquidity analysis, phase identification
4. **Strategy Matching**: Rule-based strategy selection
5. **Execution**: Entry confirmation and trade execution
6. **Monitoring**: Performance tracking and risk management

Typed Pydantic models define the expected inputs and outputs for each step. See
[pipeline_models.md](pipeline_models.md) for field-level details.

## 🎛️ Configuration

### Strategy Rules (strategy_rules.json)
- Rule-based conditions for each strategy variant
- Confluence requirements and scoring
- Risk parameters and filters

### Agent Instructions (ZANZIBAR_AGENT_INSTRUCTIONS.md)
- Master directive for all orchestrators
- Workflow definitions and feature flags
- Macro dashboard integration

## 🔌 Integrations

### MT5 Integration
- Real-time data handling
- Order execution and management
- Position monitoring

### CSV/Vector Processing
- Historical data ingestion
- Vector database integration
- Batch processing capabilities

### Monitoring & Alerting
- Real-time dashboard
- Performance analytics
- System health monitoring

## 🚀 Key Features

### Multi-Strategy Support
- **Inv**: Inversion setups with wick mitigation
- **MAZ2**: FVG re-test with refined entry rules
- **TMC**: BOS confirmation with confluence
- **Mentfx**: DSS/RSI with candlestick patterns

### Advanced Filtering
- Ultra-precision gating (VWAP, DSS, POI scoring)
- Wyckoff phase context
- Macro sentiment alignment

### Risk Management
- Dynamic position sizing
- Spread-adjusted SL/TP calculation
- Account equity protection

### Automation
- Scheduled scanning loops
- Automatic strategy selection
- Real-time market monitoring

## 📊 Performance Monitoring

- Trade journaling with markdown logs
- Strategy performance tracking
- System health monitoring
- Error handling and recovery

## 🔧 Technical Requirements

- Python 3.8+
- pandas, numpy for data processing
- plotly for chart generation
- MT5 terminal for live trading
- Redis/SQLite for state management

## 🎯 Getting Started

1. Configure strategy rules and parameters
2. Set up data connections (MT5/CSV)
3. Initialize the copilot orchestrator
4. Start monitoring and analysis loops
5. Review trade journals and performance

## 📈 Version History

- **v24.1.0**: Phoenix-Unified consolidation
- **v5.x**: Individual module versions
- **Legacy**: Previous NCOS iterations

---
*NCOS v24.1 - Built for institutional-grade trading performance*
