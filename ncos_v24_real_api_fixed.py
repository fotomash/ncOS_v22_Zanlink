#!/usr/bin/env python3
"""
ncOS v24 REAL DATA API Server - FIXED
Handles tab-separated tick data with proper timestamp parsing
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine_instance = None

class RealDataProcessor:
    """Process REAL tick data from TSV file"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.tick_data = None
        self.ohlc_data = {}
        self.load_tick_data()

    def load_tick_data(self):
        """Load actual tick data - FIXED for tab-separated format"""
        try:
            # Load with tab separator
            self.tick_data = pd.read_csv(self.csv_path, sep='\t')
            logger.info(f"Loaded {len(self.tick_data)} ticks from {self.csv_path}")

            # Parse timestamp with the actual format: "2025.06.23 19:50:28"
            self.tick_data['timestamp'] = pd.to_datetime(
                self.tick_data['timestamp'], 
                format='%Y.%m.%d %H:%M:%S'
            )

            # Sort by timestamp and set as index
            self.tick_data = self.tick_data.sort_values('timestamp')
            self.tick_data.set_index('timestamp', inplace=True)

            # Log data info
            logger.info(f"Date range: {self.tick_data.index[0]} to {self.tick_data.index[-1]}")
            logger.info(f"Columns: {list(self.tick_data.columns)}")

            # Generate OHLC data
            self.convert_to_ohlc()

        except Exception as e:
            logger.error(f"Error loading tick data: {e}")
            raise

    def convert_to_ohlc(self):
        """Convert ticks to OHLC"""
        timeframes = {
            'M1': '1T', 
            'M5': '5T', 
            'M15': '15T', 
            'M30': '30T', 
            'H1': '1H', 
            'H4': '4H'
        }

        for tf_name, freq in timeframes.items():
            try:
                # Use bid price for OHLC
                ohlc = self.tick_data['bid'].resample(freq).agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last'
                }).dropna()

                # Add volume (tick count) and average spread
                ohlc['volume'] = self.tick_data['bid'].resample(freq).count()
                ohlc['spread'] = self.tick_data['spread_price'].resample(freq).mean()

                self.ohlc_data[tf_name] = ohlc
                logger.info(f"{tf_name}: {len(ohlc)} candles created")

            except Exception as e:
                logger.error(f"Error creating {tf_name}: {e}")

class MarketStructureAnalyzer:
    """Real market structure analysis"""

    def __init__(self, data_processor):
        self.data = data_processor

    def find_swing_points(self, df, lookback=5):
        """Find swing highs/lows"""
        highs = []
        lows = []

        if len(df) < lookback * 2 + 1:
            return highs, lows

        for i in range(lookback, len(df) - lookback):
            # Check if it's a swing high
            if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
                highs.append({
                    'time': df.index[i].isoformat(), 
                    'price': float(df['high'].iloc[i])
                })
            # Check if it's a swing low
            if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
                lows.append({
                    'time': df.index[i].isoformat(), 
                    'price': float(df['low'].iloc[i])
                })

        return highs, lows

    def analyze(self, symbol='XAUUSD', timeframe='H4'):
        """Complete market analysis"""
        if timeframe not in self.data.ohlc_data:
            return {'error': f'Timeframe {timeframe} not available'}

        df = self.data.ohlc_data[timeframe].tail(100)

        if len(df) < 10:
            return {'error': f'Not enough data for {timeframe} analysis'}

        # Get current price from latest M1 candle
        if 'M1' in self.data.ohlc_data and len(self.data.ohlc_data['M1']) > 0:
            current_price = float(self.data.ohlc_data['M1']['close'].iloc[-1])
        else:
            current_price = float(df['close'].iloc[-1])

        # Get structure
        highs, lows = self.find_swing_points(df)

        # Determine bias
        if len(highs) >= 2 and len(lows) >= 2:
            # Compare last two lows for trend
            bias = 'BULLISH' if lows[-1]['price'] > lows[-2]['price'] else 'BEARISH'
        else:
            bias = 'NEUTRAL'

        # Find key levels
        key_levels = []

        # Add swing highs as resistance
        for high in highs[-5:]:
            key_levels.append({
                'level': high['price'], 
                'type': 'resistance',
                'source': 'swing_high'
            })

        # Add swing lows as support
        for low in lows[-5:]:
            key_levels.append({
                'level': low['price'], 
                'type': 'support',
                'source': 'swing_low'
            })

        # Sort by distance from current price
        key_levels.sort(key=lambda x: abs(x['level'] - current_price))

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'bias': bias,
            'key_levels': key_levels[:10],
            'recent_highs': highs[-3:] if highs else [],
            'recent_lows': lows[-3:] if lows else [],
            'candles_analyzed': len(df),
            'timestamp': datetime.now().isoformat()
        }

class TradingEngine:
    """Real trading logic"""

    def __init__(self, csv_path: str):
        self.processor = RealDataProcessor(csv_path)
        self.analyzer = MarketStructureAnalyzer(self.processor)

    def scan_market(self, symbol='XAUUSD'):
        """Multi-timeframe scan"""
        result = {'symbol': symbol, 'scan_time': datetime.now().isoformat()}

        # Analyze multiple timeframes
        for tf in ['H4', 'H1', 'M15', 'M5']:
            result[tf] = self.analyzer.analyze(symbol, tf)

        # Add current tick info
        if len(self.processor.tick_data) > 0:
            last_tick = self.processor.tick_data.iloc[-1]
            result['current'] = {
                'bid': float(last_tick['bid']),
                'ask': float(last_tick['ask']),
                'spread': float(last_tick['spread_price']),
                'time': self.processor.tick_data.index[-1].isoformat()
            }

        return result

    def find_setup(self, symbol='XAUUSD'):
        """Find trading setup based on real data"""
        scan = self.scan_market(symbol)

        # Get biases from different timeframes
        h4_bias = scan.get('H4', {}).get('bias', 'NEUTRAL')
        h1_bias = scan.get('H1', {}).get('bias', 'NEUTRAL')
        m15_bias = scan.get('M15', {}).get('bias', 'NEUTRAL')

        current_price = scan.get('current', {}).get('bid', 0)

        # Check for confluence
        if h4_bias == h1_bias and h4_bias != 'NEUTRAL':
            # Get key levels
            h4_levels = scan.get('H4', {}).get('key_levels', [])

            if h4_bias == 'BULLISH':
                # Find nearest support for stop loss
                supports = [l for l in h4_levels 
                          if l['type'] == 'support' and l['level'] < current_price]

                if supports:
                    sl = supports[0]['level'] - 1.0  # 1 point buffer
                    tp = current_price + (current_price - sl) * 3  # 3:1 RR

                    return {
                        'signal': 'BUY',
                        'entry': current_price,
                        'stop_loss': round(sl, 2),
                        'take_profit': round(tp, 2),
                        'risk_reward': 3.0,
                        'reason': f"H4 and H1 aligned {h4_bias}",
                        'support_level': supports[0]['level'],
                        'confluence': {
                            'H4': h4_bias,
                            'H1': h1_bias,
                            'M15': m15_bias
                        },
                        'timestamp': datetime.now().isoformat()
                    }
            else:  # BEARISH
                # Find nearest resistance for stop loss
                resistances = [l for l in h4_levels 
                             if l['type'] == 'resistance' and l['level'] > current_price]

                if resistances:
                    sl = resistances[0]['level'] + 1.0  # 1 point buffer
                    tp = current_price - (sl - current_price) * 3  # 3:1 RR

                    return {
                        'signal': 'SELL',
                        'entry': current_price,
                        'stop_loss': round(sl, 2),
                        'take_profit': round(tp, 2),
                        'risk_reward': 3.0,
                        'reason': f"H4 and H1 aligned {h4_bias}",
                        'resistance_level': resistances[0]['level'],
                        'confluence': {
                            'H4': h4_bias,
                            'H1': h1_bias,
                            'M15': m15_bias
                        },
                        'timestamp': datetime.now().isoformat()
                    }

        return {
            'signal': 'NO_TRADE',
            'reason': 'No confluence between timeframes',
            'biases': {
                'H4': h4_bias,
                'H1': h1_bias,
                'M15': m15_bias
            },
            'current_price': current_price
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance
    logger.info("Starting ncOS Real Data Engine...")

    # Try to find CSV file
    csv_paths = [
        'XAUUSD_TICKS_1days_20250623.csv',
        '/Users/tom/Documents/GitHub/ncOS_v22_Zanlink/XAUUSD_TICKS_1days_20250623.csv',
        './XAUUSD_TICKS_1days_20250623.csv'
    ]

    loaded = False
    for path in csv_paths:
        try:
            engine_instance = TradingEngine(path)
            logger.info(f"✅ Engine successfully loaded data from {path}")
            loaded = True
            break
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            continue

    if not loaded:
        logger.error("❌ Could not find or load tick data CSV!")

    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="ncOS Real Data API",
    version="24.1",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {
        "status": "ncOS Real Data API v24.1",
        "engine": "REAL_DATA",
        "loaded": engine_instance is not None
    }

@app.post("/scan")
def scan_endpoint():
    if not engine_instance:
        raise HTTPException(500, "Engine not loaded - check CSV file path")
    return engine_instance.scan_market()

@app.post("/setup")
def setup_endpoint():
    if not engine_instance:
        raise HTTPException(500, "Engine not loaded - check CSV file path")
    return engine_instance.find_setup()

@app.get("/data_status")
def data_status():
    if not engine_instance:
        return {"error": "No engine loaded"}

    try:
        tick_count = len(engine_instance.processor.tick_data)
        timeframe_info = {}

        for tf, data in engine_instance.processor.ohlc_data.items():
            if len(data) > 0:
                timeframe_info[tf] = {
                    'candles': len(data),
                    'first': data.index[0].isoformat(),
                    'last': data.index[-1].isoformat(),
                    'last_close': float(data['close'].iloc[-1])
                }

        return {
            'ticks_loaded': tick_count,
            'date_range': {
                'start': engine_instance.processor.tick_data.index[0].isoformat(),
                'end': engine_instance.processor.tick_data.index[-1].isoformat()
            },
            'timeframes': timeframe_info,
            'file_path': engine_instance.processor.csv_path
        }
    except Exception as e:
        return {"error": f"Error getting data status: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
