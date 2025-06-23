# ncOS v4 - Complete System Documentation

## 🚀 Overview

ncOS (NeuroCoreOS) v4 is a fully-featured trading intelligence system that combines:
- Smart Money Concepts (SMC)
- Wyckoff Method
- Advanced Pattern Recognition
- Order Block Detection
- Liquidity Zone Mapping
- Real-time Signal Generation

## 📁 System Architecture

```
ncOS v4
├── Core Components
│   ├── ncos_engine_v4.py         # Main API server with all endpoints
│   ├── ncos_theory_module_v2.py  # Intelligence engine with trading logic
│   └── start_ncos_v4.sh          # Startup script
│
├── Configuration
│   ├── .env                      # Environment variables
│   └── openapi_v3.yaml          # API schema for GPT integration
│
└── Data
    └── XAUUSD_TICKS_1days_20250623.csv  # Market data
```

## 🔧 Installation & Setup

### 1. Prerequisites
- Python 3.8+
- Virtual environment (outside project folder)
- Active ngrok tunnel

### 2. Setup Steps

```bash
# 1. Create virtual environment (if not exists)
python3 -m venv ~/virtualenvs/ncos_env

# 2. Activate environment
source ~/virtualenvs/ncos_env/bin/activate

# 3. Navigate to project
cd ~/path/to/ncOS_v22_Zanlink

# 4. Make startup script executable
chmod +x start_ncos_v4.sh

# 5. Launch the system
./start_ncos_v4.sh
```

## 🎯 Available API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/api/status` | GET | System health check |
| `/api/signal` | GET | Get trading signal with full analysis |
| `/api/structure` | GET | Market structure (HH, HL, LL, LH) |
| `/api/wyckoff` | GET | Wyckoff phase detection |
| `/api/orderblocks` | GET | Identify order blocks |
| `/api/liquidity` | GET | Map liquidity zones |
| `/api/analysis/full` | GET | Complete market analysis |
| `/api/menu` | GET | Interactive capabilities menu |

### Example Responses

#### Trading Signal (`/api/signal`)
```json
{
  "signal": "BUY",
  "confidence": 0.725,
  "reason": "Market Structure: BULLISH_STRUCTURE | Wyckoff Phase: ACCUMULATION | Momentum: BULLISH | Active Order Blocks: 2",
  "details": {
    "structure": {...},
    "wyckoff": {...},
    "order_blocks": [...],
    "liquidity": {...},
    "bull_score": 2.175,
    "bear_score": 0.825
  },
  "timestamp": "2025-06-23T12:00:00Z"
}
```

## 🤖 Custom GPT Integration

### 1. Create Custom GPT
1. Go to [chat.openai.com](https://chat.openai.com)
2. Navigate to "Explore GPTs" → "+ Create"
3. Switch to "Configure" tab
4. Click "Create new action"

### 2. Import API Schema
1. Copy contents of `openapi_v3.yaml`
2. Paste into the schema box
3. **IMPORTANT**: Update the server URL to your current ngrok URL

### 3. Sample Prompts for Your GPT
- "What's the current trading signal?"
- "Analyze the market structure"
- "Which Wyckoff phase are we in?"
- "Show me the order blocks"
- "Where are the liquidity zones?"
- "Give me a complete market analysis"

## 🧠 Intelligence Modules

### Market Structure Analysis
- Identifies swing highs and lows
- Determines trend structure (Bullish/Bearish/Ranging)
- Calculates structure strength

### Wyckoff Phase Detection
- Accumulation: Smart money buying
- Distribution: Smart money selling
- Markup: Trending up
- Markdown: Trending down

### Order Block Identification
- Detects institutional order zones
- Rates effectiveness (0-1)
- Tracks both bullish and bearish blocks

### Liquidity Zone Mapping
- Identifies price levels with high activity
- Separates buy-side and sell-side liquidity
- Helps predict price targets

## 🔄 System Flow

```
1. User Query (via GPT) → 
2. API Request (via ngrok) → 
3. Engine processes request → 
4. Theory module analyzes data → 
5. Signal generated → 
6. Response sent back → 
7. GPT presents to user
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Endpoint not found" error**
   - Ensure you're running `ncos_engine_v4.py` (not v3)
   - Check that ngrok URL is correct in GPT schema

2. **"Module not found" error**
   - Ensure `ncos_theory_module_v2.py` is in same directory
   - Check virtual environment is activated

3. **"Data file not found" error**
   - Ensure `XAUUSD_TICKS_1days_20250623.csv` is present
   - Check file permissions

### Debug Commands

```bash
# Check if server is running
curl http://localhost:8000/api/status

# Test ngrok connection
curl https://your-ngrok-url.ngrok-free.app/api/status

# View all capabilities
curl http://localhost:8000/api/menu
```

## 📈 Extending the System

### Adding New Analysis Methods

1. Add method to `NCOSTheoryEngine` class in `ncos_theory_module_v2.py`
2. Create new endpoint in `ncos_engine_v4.py`
3. Update `openapi_v3.yaml` with new endpoint
4. Re-import schema in Custom GPT

### Adding New Data Sources

1. Update `load_market_data()` function
2. Modify data parsing logic
3. Ensure compatibility with existing analysis methods

## 🔐 Security Notes

- Never commit `.env` files with real API keys
- Keep ngrok URL private
- Regularly rotate access tokens
- Monitor API usage

## 📞 Quick Reference

- **Start Server**: `./start_ncos_v4.sh`
- **Stop Server**: `Ctrl+C` in terminal
- **View Logs**: Check terminal output
- **Test Endpoint**: `curl http://localhost:8000/api/signal`

---

*ncOS v4 - Built with intelligence, powered by data*
