# ncOS v23 Quick Reference

## 🚀 For GPT Users

1. **Upload to GPT**: Upload `ncos_bootstrap.py` to your Custom GPT
2. **Auto-init**: The system initializes automatically when loaded
3. **Check status**: Call `_bootstrap.menu()` to see available options

## 💻 For Local Users

### Quick Start
```bash
# Method 1: Launch script
chmod +x launch.sh
./launch.sh

# Method 2: Docker
docker-compose up

# Method 3: Manual
pip install -r requirements.txt
python ncos_v23_api_server.py
```

### API Usage
```python
# Initialize
GET http://localhost:8000/api/bootstrap

# Get menu
GET http://localhost:8000/api/menu

# Analyze market
POST http://localhost:8000/api/analyze
{
    "pair": "XAUUSD",
    "timeframe": "M5",
    "data_file": "XAUUSD_ticks.csv"
}
```

## 📊 Data Format

### Tick Data (CSV)
```
timestamp,bid,ask,spread_points,volume
2025.06.23 05:02:32,3357.27,3357.58,31.0,0.00
```

### OHLCV Data (CSV)
```
timestamp,open,high,low,close,volume
2025-06-23 05:00:00,3357.20,3357.80,3357.10,3357.50,1250
```

## 🤖 Available Agents

1. **SMC Master** - Market structure & POIs
2. **Liquidity Sniper** - Sweep detection
3. **Risk Guardian** - Position sizing
4. **Structure Validator** - MTF analysis

## 🎯 Trading Strategies

1. **Prometheus** - ML pattern matching
2. **Wyckoff** - Phase analysis
3. **MAZ** - Unmitigated levels
4. **London/NY** - Session reversals

## 🔧 Configuration

Edit `ncos_v23_config.json`:
- Agent weights
- Confidence thresholds
- Risk parameters
- Vector dimensions

## 📁 Directory Structure

```
project/
├── ncos_bootstrap.py      # GPT loader
├── ncos_v23_*            # Core files
├── data/                 # External data
│   ├── XAUUSD_ticks.csv
│   └── EURUSD_ohlc.csv
└── configs/              # Custom configs
```

## 🌐 Environment Variables

```bash
export NCOS_DATA_PATH=/path/to/data
export NCOS_CONFIG_PATH=/path/to/config.json
```

## 📡 API Endpoints

### Core
- `GET /` - System info
- `GET /api/bootstrap` - Initialize
- `GET /api/menu` - Menu system

### Analysis
- `POST /api/analyze` - Analyze market
- `GET /api/agents/status` - Agent status
- `GET /api/strategy/{name}` - Strategy info

### Data
- `POST /api/data/load` - Load file
- `GET /api/data/status` - Data status

## 🐛 Troubleshooting

### System won't start
```bash
# Check Python version (need 3.8+)
python --version

# Install missing deps
pip install -r requirements.txt
```

### No data found
```bash
# Create data directory
mkdir -p data

# Check path
echo $NCOS_DATA_PATH
```

### API errors
```bash
# Check logs
tail -f ncos.log

# Test health
curl http://localhost:8000/health
```

## 📚 Further Reading

- Full docs: `README.md`
- GPT guide: `ncos_v23_gpt_instructions.md`
- API docs: http://localhost:8000/docs
