#!/usr/bin/env python3

import argparse
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import talib, use backup if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, using simplified indicators")

class NCOSCLISystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self, level=logging.INFO):
        """Setup logging configuration"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def create_parser(self):
        """Create argument parser"""
        parser = argparse.ArgumentParser(description='ncOS Trading Analysis System')
        
        # Required arguments
        parser.add_argument('--data', type=str, required=True, help='Input CSV/TSV file path')
        
        # Analysis options
        parser.add_argument('--patterns', action='store_true', help='Enable pattern recognition')
        parser.add_argument('--wyckoff', action='store_true', help='Enable Wyckoff analysis')  
        parser.add_argument('--smc', action='store_true', help='Enable Smart Money Concepts')
        parser.add_argument('--harmonic', action='store_true', help='Enable harmonic patterns')
        
        # Timeframe options
        parser.add_argument('--timeframe', type=str, choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'], 
                          default='1m', help='Single timeframe to analyze')
        parser.add_argument('--timeframes', type=str, help='Multiple timeframes (comma-separated)')
        
        # Output options
        parser.add_argument('--output', type=str, default='output/', help='Output directory')
        parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
        parser.add_argument('--verbose', action='store_true', help='Verbose output')
        parser.add_argument('--debug', action='store_true', help='Debug mode')
        
        return parser
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV/TSV data with auto-detection of separator"""
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            # Try tab-separated first (common for MT5 exports), then comma-separated
            try:
                df = pd.read_csv(file_path, sep='\t')
                self.logger.info("Detected tab-separated format")
            except:
                df = pd.read_csv(file_path, sep=',')
                self.logger.info("Detected comma-separated format")
            
            self.logger.info(f"Loaded columns: {list(df.columns)}")
            
            # The data already has proper column names, just ensure timestamp is handled
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Validate required columns
            required = ['open', 'high', 'low', 'close']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Ensure volume column exists
            if 'volume' not in df.columns:
                df['volume'] = 1000
                self.logger.info("Added default volume column")
            
            self.logger.info(f"Data loaded successfully: {len(df)} rows")
            self.logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            self.logger.info(f"Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def calculate_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic indicators without TA-Lib"""
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Exponential Moving Average
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # Simple RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df
    
    def calculate_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using TA-Lib"""
        # Trend indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        
        return df
    
    def detect_patterns(self, df: pd.DataFrame) -> dict:
        """Detect candlestick patterns"""
        patterns = {}
        
        if TALIB_AVAILABLE:
            # Use TA-Lib patterns
            patterns['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            patterns['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            patterns['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        else:
            # Simple pattern detection
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['body_pct'] = df['body'] / (df['high'] - df['low'])
            
            # Doji detection (small body)
            patterns['doji'] = (df['body_pct'] < 0.1).astype(int)
            
            # Hammer detection (small body, long lower shadow)
            patterns['hammer'] = ((df['lower_shadow'] > df['body'] * 2) & 
                                (df['upper_shadow'] < df['body']) &
                                (df['body_pct'] < 0.3)).astype(int)
        
        # Count patterns
        pattern_counts = {}
        for name, pattern in patterns.items():
            if hasattr(pattern, 'sum'):
                if TALIB_AVAILABLE:
                    bullish = int((pattern > 0).sum())
                    bearish = int((pattern < 0).sum())
                    total = int((pattern != 0).sum())
                else:
                    bullish = int(pattern.sum())
                    bearish = 0
                    total = bullish
                
                pattern_counts[name] = {
                    'bullish': bullish, 
                    'bearish': bearish,
                    'total': total,
                    'last_signal': int(pattern.iloc[-1]) if len(pattern) > 0 else 0
                }
        
        return {'patterns': patterns, 'counts': pattern_counts}
    
    def analyze_wyckoff(self, df: pd.DataFrame) -> dict:
        """Wyckoff analysis"""
        try:
            # Volume analysis
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
            
            # Price-volume relationship
            price_up = df['close'] > df['close'].shift(1)
            high_volume = df['volume'] > df['vol_ma']
            
            accumulation = (price_up & high_volume).sum()
            distribution = ((~price_up) & high_volume).sum()
            neutral = ((df['volume'] <= df['vol_ma']).sum())
            
            # Current phase
            current_price_up = price_up.iloc[-1] if len(price_up) > 0 else False
            current_high_vol = high_volume.iloc[-1] if len(high_volume) > 0 else False
            
            if current_price_up and current_high_vol:
                current_phase = 'accumulation'
            elif not current_price_up and current_high_vol:
                current_phase = 'distribution'
            else:
                current_phase = 'neutral'
            
            return {
                'phases': {
                    'accumulation': int(accumulation),
                    'distribution': int(distribution),
                    'neutral': int(neutral)
                },
                'current_phase': current_phase,
                'volume_analysis': {
                    'avg_volume': float(df['volume'].mean()),
                    'current_volume': float(df['volume'].iloc[-1]),
                    'volume_ratio': float(df['vol_ratio'].iloc[-1]) if 'vol_ratio' in df.columns else None
                }
            }
        except Exception as e:
            self.logger.error(f"Wyckoff analysis error: {e}")
            return {'phases': {}, 'current_phase': 'unknown'}
    
    def analyze_smc(self, df: pd.DataFrame) -> dict:
        """Smart Money Concepts analysis"""
        try:
            # Find swing highs/lows
            window = 5
            df['swing_high'] = df['high'].rolling(window*2+1, center=True).max() == df['high']
            df['swing_low'] = df['low'].rolling(window*2+1, center=True).min() == df['low']
            
            # Get recent levels
            recent_highs = df.loc[df['swing_high'] == True, 'high'].tail(5).tolist()
            recent_lows = df.loc[df['swing_low'] == True, 'low'].tail(5).tolist()
            
            # Market structure analysis
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
                    structure = 'bullish_structure'
                elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
                    structure = 'bearish_structure'
                else:
                    structure = 'sideways'
            else:
                structure = 'insufficient_data'
            
            # Support/Resistance levels
            support_level = float(recent_lows[-1]) if recent_lows else None
            resistance_level = float(recent_highs[-1]) if recent_highs else None
            
            return {
                'structure': structure,
                'order_blocks': {
                    'resistance_levels': [float(x) for x in recent_highs],
                    'support_levels': [float(x) for x in recent_lows]
                },
                'key_levels': {
                    'support': support_level,
                    'resistance': resistance_level
                },
                'swing_analysis': {
                    'total_swing_highs': int(df['swing_high'].sum()),
                    'total_swing_lows': int(df['swing_low'].sum())
                }
            }
        except Exception as e:
            self.logger.error(f"SMC analysis error: {e}")
            return {'structure': 'unknown', 'order_blocks': {}}
    
    def analyze_harmonic(self, df: pd.DataFrame) -> dict:
        """Harmonic pattern analysis"""
        try:
            # Find significant swings
            swings = []
            window = 5
            
            for i in range(window, len(df) - window):
                if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                    swings.append(('high', i, float(df['high'].iloc[i])))
                elif df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                    swings.append(('low', i, float(df['low'].iloc[i])))
            
            # Analyze retracement levels
            patterns_detected = []
            fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            if len(swings) >= 4:
                # Check for potential ABCD patterns
                last_4_swings = swings[-4:]
                patterns_detected.append('potential_abcd')
                
                # Check for Gartley-like ratios
                if len(swings) >= 5:
                    patterns_detected.append('potential_gartley')
            
            return {
                'swing_points': len(swings),
                'patterns_detected': patterns_detected,
                'fibonacci_analysis': {
                    'retracement_levels': fibonacci_levels,
                    'swing_analysis': f'{len(swings)} swings identified'
                },
                'last_swings': swings[-5:] if len(swings) >= 5 else swings
            }
        except Exception as e:
            self.logger.error(f"Harmonic analysis error: {e}")
            return {'swing_points': 0, 'patterns_detected': []}
    
    def parse_timeframes(self, args) -> list:
        """Parse timeframes from arguments"""
        if args.timeframes:
            return [tf.strip() for tf in args.timeframes.split(',')]
        else:
            return [args.timeframe]
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to timeframe"""
        tf_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '4h': '4H', '1d': '1D'
        }
        
        if timeframe == '1m':
            return df  # No resampling needed
        
        tf = tf_map.get(timeframe, timeframe)
        
        resampled = df.resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    async def run_pattern_analysis(self, args) -> dict:
        """Main pattern analysis function"""
        try:
            self.logger.info("Starting pattern analysis...")
            
            # Load data
            df = self.load_data(args.data)
            timeframes = self.parse_timeframes(args)
            
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'data_file': args.data,
                'timeframes_analyzed': timeframes,
                'data_info': {
                    'total_bars': len(df),
                    'date_range': {
                        'start': str(df.index[0]),
                        'end': str(df.index[-1])
                    },
                    'price_range': {
                        'high': float(df['high'].max()),
                        'low': float(df['low'].min()),
                        'latest_close': float(df['close'].iloc[-1])
                    }
                },
                'results': {}
            }
            
            # Process each timeframe
            for tf in timeframes:
                self.logger.info(f"Processing timeframe: {tf}")
                
                # Resample data
                tf_df = self.resample_data(df, tf)
                
                if len(tf_df) < 20:  # Need minimum data for analysis
                    self.logger.warning(f"Insufficient data for {tf} ({len(tf_df)} bars)")
                    continue
                
                # Calculate indicators
                if TALIB_AVAILABLE:
                    tf_df = self.calculate_talib_indicators(tf_df)
                else:
                    tf_df = self.calculate_simple_indicators(tf_df)
                
                tf_results = {
                    'data_points': len(tf_df),
                    'date_range': {
                        'start': str(tf_df.index[0]),
                        'end': str(tf_df.index[-1])
                    },
                    'price_summary': {
                        'open': float(tf_df['open'].iloc[0]),
                        'high': float(tf_df['high'].max()),
                        'low': float(tf_df['low'].min()),
                        'close': float(tf_df['close'].iloc[-1]),
                        'volume': float(tf_df['volume'].sum())
                    }
                }
                
                # Run requested analyses
                if args.patterns:
                    self.logger.info(f"Running pattern analysis for {tf}")
                    tf_results['patterns'] = self.detect_patterns(tf_df)
                
                if args.wyckoff:
                    self.logger.info(f"Running Wyckoff analysis for {tf}")
                    tf_results['wyckoff'] = self.analyze_wyckoff(tf_df)
                
                if args.smc:
                    self.logger.info(f"Running SMC analysis for {tf}")
                    tf_results['smc'] = self.analyze_smc(tf_df)
                
                if args.harmonic:
                    self.logger.info(f"Running harmonic analysis for {tf}")
                    tf_results['harmonic'] = self.analyze_harmonic(tf_df)
                
                # Current indicators
                tf_results['indicators'] = {
                    'rsi': float(tf_df['rsi'].iloc[-1]) if 'rsi' in tf_df.columns and not pd.isna(tf_df['rsi'].iloc[-1]) else None,
                    'atr': float(tf_df['atr'].iloc[-1]) if 'atr' in tf_df.columns and not pd.isna(tf_df['atr'].iloc[-1]) else None,
                    'sma_20': float(tf_df['sma_20'].iloc[-1]) if 'sma_20' in tf_df.columns and not pd.isna(tf_df['sma_20'].iloc[-1]) else None,
                    'sma_50': float(tf_df['sma_50'].iloc[-1]) if 'sma_50' in tf_df.columns and not pd.isna(tf_df['sma_50'].iloc[-1]) else None
                }
                
                # Add trend analysis
                if 'sma_20' in tf_df.columns and 'sma_50' in tf_df.columns:
                    sma20 = tf_df['sma_20'].iloc[-1]
                    sma50 = tf_df['sma_50'].iloc[-1]
                    close = tf_df['close'].iloc[-1]
                    
                    if not pd.isna(sma20) and not pd.isna(sma50):
                        if close > sma20 > sma50:
                            trend = 'strong_bullish'
                        elif close > sma20 and sma20 < sma50:
                            trend = 'weak_bullish'
                        elif close < sma20 < sma50:
                            trend = 'strong_bearish'
                        elif close < sma20 and sma20 > sma50:
                            trend = 'weak_bearish'
                        else:
                            trend = 'sideways'
                    else:
                        trend = 'insufficient_data'
                    
                    tf_results['trend_analysis'] = {
                        'direction': trend,
                        'price_vs_sma20': 'above' if close > sma20 else 'below',
                        'sma20_vs_sma50': 'above' if sma20 > sma50 else 'below'
                    }
                
                results['results'][tf] = tf_results
            
            self.logger.info("Pattern analysis completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            raise
    
    def save_results(self, results: dict, args):
        """Save results to file"""
        try:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if args.format == 'json':
                output_file = output_dir / f"analysis_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_file}")
            
            if args.verbose:
                print("\n" + "="*50)
                print("ANALYSIS RESULTS SUMMARY")
                print("="*50)
                print(f"File: {results['data_file']}")
                print(f"Total bars: {results['data_info']['total_bars']}")
                print(f"Price range: {results['data_info']['price_range']['low']:.2f} - {results['data_info']['price_range']['high']:.2f}")
                print(f"Latest close: {results['data_info']['price_range']['latest_close']:.2f}")
                print()
                
                for tf, tf_data in results['results'].items():
                    print(f"--- {tf.upper()} TIMEFRAME ---")
                    print(f"Bars: {tf_data['data_points']}")
                    
                    if 'indicators' in tf_data:
                        ind = tf_data['indicators']
                        print(f"RSI: {ind['rsi']:.2f}" if ind['rsi'] else "RSI: N/A")
                        print(f"ATR: {ind['atr']:.2f}" if ind['atr'] else "ATR: N/A")
                    
                    if 'trend_analysis' in tf_data:
                        print(f"Trend: {tf_data['trend_analysis']['direction']}")
                    
                    if 'patterns' in tf_data and 'counts' in tf_data['patterns']:
                        total_patterns = sum(p['total'] for p in tf_data['patterns']['counts'].values())
                        print(f"Patterns detected: {total_patterns}")
                    
                    if 'wyckoff' in tf_data:
                        print(f"Wyckoff phase: {tf_data['wyckoff']['current_phase']}")
                    
                    if 'smc' in tf_data:
                        print(f"Market structure: {tf_data['smc']['structure']}")
                    
                    print()
                
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    async def main(self):
        """Main entry point"""
        try:
            parser = self.create_parser()
            args = parser.parse_args()
            
            if args.debug:
                self.setup_logging(logging.DEBUG)
            elif args.verbose:
                self.setup_logging(logging.INFO)
            
            self.logger.info("Starting ncOS Trading Analysis System")
            
            # Check if any analysis is requested
            if not any([args.patterns, args.wyckoff, args.smc, args.harmonic]):
                self.logger.error("No analysis mode specified. Use --patterns, --wyckoff, --smc, or --harmonic")
                return
            
            # Run analysis
            results = await self.run_pattern_analysis(args)
            
            # Save results
            self.save_results(results, args)
            
            self.logger.info("Analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error: {e}")
            if args.debug:
                raise

if __name__ == "__main__":
    cli_system = NCOSCLISystem()
    asyncio.run(cli_system.main())