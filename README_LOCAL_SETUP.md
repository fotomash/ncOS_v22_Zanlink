# ncOS Local Development Setup (No API Key Required)

This setup allows you to run ncOS locally without any external API dependencies.

## Quick Start

1. **Start the server:**
   ```bash
   ./start_ncos_local.sh
   ```

2. **Test the connection:**
   ```bash
   python test_ncos_connection.py
   ```

## Configuration

- **Server**: Runs on `http://localhost:8000`
- **Ngrok URL**: `https://emerging-tiger-fair.ngrok-free.app`
- **No API keys required** - everything runs locally

## Endpoints

- `/` - Home/Status
- `/health` - Health check
- `/data` - Data management
- `/upload-trace` - Upload traces
- `/api/trading` - Trading operations (demo mode)
- `/api/market` - Market data
- `/api/signals` - Trading signals
- `/api/backtest` - Backtesting

## Features

1. **Demo Trading**: Paper trading without real money
2. **Local Data**: Uses local CSV/Parquet files
3. **No External Dependencies**: No API keys needed
4. **Ngrok Compatible**: Works with your ngrok tunnel

## File Structure

```
.
├── ncos_local_server.py    # Main server
├── ncos_config_local.json  # Configuration
├── .env.local              # Environment variables
├── start_ncos_local.sh     # Startup script
├── test_ncos_connection.py # Test client
├── data/
│   ├── ticks/             # Tick data (CSV)
│   └── historical/        # Historical data (Parquet)
├── logs/                  # Log files
└── uploads/               # Uploaded traces
```

## Troubleshooting

1. **502 Bad Gateway**: Make sure the local server is running
2. **Connection Refused**: Check if port 8000 is available
3. **Missing Data**: Place your CSV files in `data/ticks/`

## Next Steps

1. Place your tick data in `data/ticks/`
2. Configure your trading strategies
3. Start backtesting with demo data
4. Monitor logs in `logs/ncos.log`
