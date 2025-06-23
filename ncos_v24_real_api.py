#!/usr/bin/env python3
"""
ncOS v24 REAL DATA API Server
No mock data - uses actual CSV tick data
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
    """Process REAL tick data from CSV"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.tick_data = None
        self.ohlc_data = {}
        self.load_tick_data()

    def load_tick_data(self):
        """Load actual tick data"""
        try:
            self.tick_data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.tick_data)} ticks from {self.csv_path}")

            # Parse timestamp properly
            self.tick_data['timestamp'] = pd.to_datetime(self.tick_data['timestamp'], format='%Y.%m.%d %H:%M:%S')
            self.tick_data.set_index('timestamp', inplace=True)

            # Generate OHLC data
            self.convert_to_ohlc()
        except Exception as e:
            logger.error(f"Error loading tick data: {e}")
            raise

    def convert_to_ohlc(self):
        """Convert ticks to OHLC"""
        timeframes = {
            'M1': '1T', 'M5': '5T', 'M15': '15T', 
            'M30': '30T', 'H1': '1H', 'H4': '4H'
        }

        for tf_name, freq in timeframes.items():
            try:
                ohlc = self.tick_data['bid'].resample(freq).agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last'
                }).dropna()

                ohlc['volume'] = self.tick_data['bid'].resample(freq).count()
                ohlc['spread'] = self.tick_data['spread_price'].resample(freq).mean()

                self.ohlc_data[tf_name] = ohlc
                logger.info(f"{tf_name}: {len(ohlc)} candles")
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

        for i in range(lookback, len(df) - lookback):
            if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
                highs.append({'time': df.index[i], 'price': df['high'].iloc[i]})
            if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
                lows.append({'time': df.index[i], 'price': df['low'].iloc[i]})

        return highs, lows

    def detect_order_blocks(self, df):
        """Detect order blocks"""
        obs = []

        for i in range(10, len(df)):
            # Bullish OB
            if (df['close'].iloc[i] > df['open'].iloc[i] and 
                df['close'].iloc[i] > df['high'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5):
                obs.append({
                    'type': 'bullish_ob',
                    'high': df['high'].iloc[i],
                    'low': df['low'].iloc[i],
                    'time': df.index[i]
                })

        return obs

    def analyze(self, symbol='XAUUSD', timeframe='H4'):
        """Complete market analysis"""
        if timeframe not in self.data.ohlc_data:
            return {'error': 'Timeframe not available'}

        df = self.data.ohlc_data[timeframe].tail(100)
        current_price = float(self.data.ohlc_data['M1']['close'].iloc[-1])

        # Get structure
        highs, lows = self.find_swing_points(df)

        # Determine bias
        if len(highs) >= 2 and len(lows) >= 2:
            bias = 'BULLISH' if lows[-1]['price'] > lows[-2]['price'] else 'BEARISH'
        else:
            bias = 'NEUTRAL'

        # Find key levels
        key_levels = []
        for high in highs[-5:]:
            key_levels.append({'level': float(high['price']), 'type': 'resistance'})
        for low in lows[-5:]:
            key_levels.append({'level': float(low['price']), 'type': 'support'})

        # Sort by distance from current price
        key_levels.sort(key=lambda x: abs(x['level'] - current_price))

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'bias': bias,
            'key_levels': key_levels[:10],
            'recent_highs': highs[-3:],
            'recent_lows': lows[-3:],
            'timestamp': datetime.now().isoformat()
        }

class TradingEngine:
    """Real trading logic"""

    def __init__(self, csv_path: str):
        self.processor = RealDataProcessor(csv_path)
        self.analyzer = MarketStructureAnalyzer(self.processor)

    def scan_market(self, symbol='XAUUSD'):
        """Multi-timeframe scan"""
        htf_analysis = self.analyzer.analyze(symbol, 'H4')
        mtf_analysis = self.analyzer.analyze(symbol, 'M15')
        ltf_analysis = self.analyzer.analyze(symbol, 'M5')

        # Current tick info
        last_tick = self.processor.tick_data.iloc[-1]

        return {
            'symbol': symbol,
            'current': {
                'bid': float(last_tick['bid']),
                'ask': float(last_tick['ask']),
                'spread': float(last_tick['spread_price']),
                'time': self.processor.tick_data.index[-1].isoformat()
            },
            'H4': htf_analysis,
            'M15': mtf_analysis,
            'M5': ltf_analysis,
            'scan_time': datetime.now().isoformat()
        }

    def find_setup(self, symbol='XAUUSD'):
        """Find trading setup"""
        scan = self.scan_market(symbol)

        # Check confluence
        h4_bias = scan['H4']['bias']
        m15_bias = scan['M15']['bias']
        current_price = scan['current']['bid']

        if h4_bias == m15_bias and h4_bias != 'NEUTRAL':
            # Find nearest support/resistance
            if h4_bias == 'BULLISH':
                supports = [l for l in scan['H4']['key_levels'] if l['type'] == 'support' and l['level'] < current_price]
                if supports:
                    sl = supports[0]['level'] - 0.5
                    tp = current_price + (current_price - sl) * 3
                    return {
                        'signal': 'BUY',
                        'entry': current_price,
                        'stop_loss': round(sl, 2),
                        'take_profit': round(tp, 2),
                        'risk_reward': 3.0,
                        'reason': f"H4 and M15 aligned {h4_bias}",
                        'timestamp': datetime.now().isoformat()
                    }

        return {
            'signal': 'NO_TRADE',
            'reason': 'No confluence between timeframes',
            'h4_bias': h4_bias,
            'm15_bias': m15_bias
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

    for path in csv_paths:
        try:
            engine_instance = TradingEngine(path)
            logger.info(f"Engine loaded with data from {path}")
            break
        except:
            continue
    else:
        logger.error("Could not find tick data CSV!")

    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="ncOS Real Data API",
    version="24.0",
    lifespan=lifespan
)

class CommandRequest(BaseModel):
    command: str

@app.get("/")
def root():
    return {"status": "ncOS Real Data API", "engine": "REAL_DATA"}

@app.post("/scan")
def scan_endpoint():
    if not engine_instance:
        raise HTTPException(500, "Engine not loaded")
    return engine_instance.scan_market()

@app.post("/setup")
def setup_endpoint():
    if not engine_instance:
        raise HTTPException(500, "Engine not loaded")
    return engine_instance.find_setup()

@app.get("/data_status")
def data_status():
    if not engine_instance:
        return {"error": "No engine loaded"}

    return {
        'ticks_loaded': len(engine_instance.processor.tick_data),
        'timeframes': list(engine_instance.processor.ohlc_data.keys()),
        'last_tick_time': engine_instance.processor.tick_data.index[-1].isoformat(),
        'candles_per_timeframe': {
            tf: len(data) for tf, data in engine_instance.processor.ohlc_data.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
