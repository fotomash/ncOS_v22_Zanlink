"""
Enhanced Divergence Strategy Agent with Predictive Scoring
Integrates the NCOS Predictive Engine for quality-based trade filtering
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd

from ncOS.ncos_data_enricher import DataEnricher
from ncOS.ncos_feature_extractor import FeatureExtractor
from ncOS.ncos_predictive_engine import NCOSPredictiveEngine
from ncos_base_agent import NCOSBaseAgent
from ncos_predictive_schemas import PredictiveEngineConfig
from ncos_risk_engine import calculate_sl_and_risk

logger = logging.getLogger(__name__)


class DivergenceStrategyAgent(NCOSBaseAgent):
    """
    Executes a trading strategy based on RSI divergence with predictive quality scoring.
    Only trades high-quality setups as determined by the Predictive Engine.
    """

    def __init__(self, orchestrator, agent_id, config):
        super().__init__(orchestrator, agent_id, config)

        # Core trading state
        self.position = 'flat'
        self.active_trades = []

        # Risk management config
        self.account_balance = self.config.get('account_balance', 100000)
        self.base_conviction_score = self.config.get('risk_conviction_score', 4)
        self.risk_config = self.config.get('risk_engine_config', {})
        self.atr_config = self.config.get('atr_config', {})
        self.rr_ratio = self.config.get('risk_reward_ratio', 1.5)

        # Predictive engine configuration
        predictive_config_dict = self.config.get('predictive_engine_config', {})
        self.predictive_config = PredictiveEngineConfig(**predictive_config_dict)

        # Initialize predictive components
        self.predictive_engine = NCOSPredictiveEngine(self.predictive_config)
        self.feature_extractor = FeatureExtractor(self.predictive_config.feature_extractor.dict())
        self.data_enricher = DataEnricher(self.predictive_config.data_enricher)

        # Quality-based trade filtering
        self.min_grade_to_trade = self.config.get('min_grade_to_trade', 'B')
        self.grade_risk_multipliers = self.config.get('grade_risk_multipliers', {
            'A': 1.2,  # 20% more risk for A-grade setups
            'B': 1.0,  # Normal risk for B-grade
            'C': 0.7,  # Reduced risk for C-grade
            'D': 0.0  # No trade for D-grade
        })

        # Data buffer for calculations
        self.max_history_size = self.config.get('max_history_size', 100)
        self.historical_data = pd.DataFrame()

        # Statistics tracking
        self.trade_stats = {
            'setups_evaluated': 0,
            'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
            'trades_executed': 0,
            'trades_skipped': 0
        }

        logger.info(f"Enhanced DivergenceStrategyAgent initialized with Predictive Engine")
        logger.info(f"Minimum grade to trade: {self.min_grade_to_trade}")

    async def handle_trigger(self, trigger_name, payload, session_state):
        """Handle incoming triggers."""
        if trigger_name == 'data.bar.enriched.xauusd_4h':
            self._update_history(payload)
            await self.evaluate_strategy(payload)
        elif trigger_name == 'event.trade.closed':
            if payload.get('agent_id') == self.agent_id:
                self._handle_trade_closure(payload)

    def _update_history(self, bar_data):
        """Update historical data buffer with enriched data."""
        try:
            new_row = pd.DataFrame([bar_data])
            new_row['timestamp'] = pd.to_datetime(new_row['timestamp'])
            new_row = new_row.set_index('timestamp')

            # Enrich the historical data if enricher is enabled
            if self.data_enricher.config.enabled and len(self.historical_data) > 0:
                new_row, _ = self.data_enricher.enrich_bar_data(
                    new_row.iloc[0],
                    self.historical_data
                )
                new_row = pd.DataFrame([new_row]).set_index('timestamp')

            self.historical_data = pd.concat([self.historical_data, new_row])
            if len(self.historical_data) > self.max_history_size:
                self.historical_data = self.historical_data.iloc[-self.max_history_size:]

        except Exception as e:
            logger.error(f"Failed to update historical data: {e}")

    async def evaluate_strategy(self, bar_data):
        """Evaluate strategy with predictive quality scoring."""
        if self.position != 'flat':
            return

        bar = pd.Series(bar_data)

        # Check for basic signal presence
        signal_type = self._detect_divergence_signal(bar)
        if not signal_type:
            return

        # Increment evaluation counter
        self.trade_stats['setups_evaluated'] += 1

        # Prepare context for feature extraction
        context = self._build_evaluation_context(bar, signal_type)

        try:
            # Extract features for predictive scoring
            features = self.feature_extractor.extract_features(
                bar,
                self.historical_data,
                context
            )

            # Get predictive evaluation
            evaluation = self.predictive_engine.evaluate_setup(
                timestamp=pd.to_datetime(bar['timestamp']),
                symbol='XAUUSD',
                features=features,
                active_trades=self.active_trades,
                context=context
            )

            # Extract scoring results
            scoring = evaluation['scoring']
            grade = scoring['grade']
            maturity_score = scoring['maturity_score']
            potential_entry = scoring['potential_entry']

            # Update grade distribution
            self.trade_stats['grade_distribution'][grade] += 1

            # Log the evaluation
            logger.info(
                f"Setup Evaluated - Grade: {grade}, Score: {maturity_score:.3f}, "
                f"Direction: {signal_type}, Potential Entry: {potential_entry}"
            )

            # Check if setup meets quality threshold
            if not self._should_take_trade(grade, evaluation):
                self.trade_stats['trades_skipped'] += 1
                logger.info(f"Setup skipped - Grade {grade} below minimum {self.min_grade_to_trade}")
                return

            # Calculate risk parameters with quality adjustment
            await self._execute_quality_adjusted_trade(bar, signal_type, grade, maturity_score)

        except Exception as e:
            logger.error(f"Error during predictive evaluation: {e}", exc_info=True)

    def _detect_divergence_signal(self, bar: pd.Series) -> Optional[str]:
        """Detect basic divergence signal from bar data."""
        required_cols = ['rsi_bull_div', 'rsi_bear_div', 'close', 'sma_20', 'structure']
        if bar.isnull().any() or not all(col in bar for col in required_cols):
            return None

        # Minimum data check
        if len(self.historical_data) < self.atr_config.get('period', 14) + 2:
            return None

        # Check for divergence signals
        if bar['rsi_bull_div'] and bar['close'] < bar['sma_20'] and bar['structure'] != 'bearish':
            return 'buy'
        elif bar['rsi_bear_div'] and bar['close'] > bar['sma_20'] and bar['structure'] != 'bullish':
            return 'sell'

        return None

    def _build_evaluation_context(self, bar: pd.Series, signal_type: str) -> Dict[str, Any]:
        """Build context for feature extraction and scoring."""
        context = {
            'direction': signal_type,
            'structure': bar.get('structure', 'neutral'),
            'timestamp': bar['timestamp']
        }

        # Add HTF bias if available
        if 'structure' in bar:
            context['htf_bias'] = bar['structure']

        # Simulate pattern detection data (in real implementation, this would come from pattern detection modules)
        # For now, we'll use available data to create pseudo-context

        # Inducement data (simulated based on price action)
        if len(self.historical_data) >= 10:
            recent_highs = self.historical_data['high'].tail(10)
            recent_lows = self.historical_data['low'].tail(10)

            if signal_type == 'buy':
                # Check if recent low was swept
                min_low = recent_lows.min()
                if bar['low'] < min_low and bar['close'] > min_low:
                    context['inducement_data'] = {
                        'clear_sweep': True,
                        'touch_count': 2,
                        'volume_spike': bar.get('volume', 0) > self.historical_data['volume'].mean() * 1.5
                    }
            else:
                # Check if recent high was swept
                max_high = recent_highs.max()
                if bar['high'] > max_high and bar['close'] < max_high:
                    context['inducement_data'] = {
                        'clear_sweep': True,
                        'touch_count': 2,
                        'volume_spike': bar.get('volume', 0) > self.historical_data['volume'].mean() * 1.5
                    }

        # Add sweep data (simulated)
        context['sweep_data'] = {
            'magnitude_pips': abs(bar['high'] - bar['low']) * 10000 / 100,  # Convert to pips
            'velocity': 1.0,  # Placeholder
            'rejection_strength': 0.7 if bar.get('volume', 0) > self.historical_data['volume'].mean() else 0.3
        }

        # Add CHoCH data (simulated based on structure)
        context['choch_data'] = {
            'break_strength': 0.8 if bar.get('structure') == signal_type.replace('buy', 'bullish').replace('sell',
                                                                                                           'bearish') else 0.3,
            'volume_on_break': bar.get('volume', 0) > self.historical_data['volume'].mean() * 1.2,
            'follow_through_bars': 2  # Placeholder
        }

        # Add POI data (simulated)
        context['poi_data'] = {
            'historical_touches': 3,
            'times_respected': 2,
            'confluence_factors': 2  # SMA + Structure
        }

        # Add average volume for tick density calculation
        if 'volume' in self.historical_data.columns:
            context['avg_volume'] = self.historical_data['volume'].mean()

        return context

    def _should_take_trade(self, grade: str, evaluation: Dict[str, Any]) -> bool:
        """Determine if trade should be taken based on grade and conflicts."""
        # Check minimum grade requirement
        grade_hierarchy = ['D', 'C', 'B', 'A']
        min_grade_index = grade_hierarchy.index(self.min_grade_to_trade)
        current_grade_index = grade_hierarchy.index(grade)

        if current_grade_index < min_grade_index:
            return False

        # Check for conflicts
        conflict_analysis = evaluation.get('conflict_analysis', {})
        if conflict_analysis.get('has_conflict'):
            conflicts = conflict_analysis.get('conflicts', [])

            # If any conflict suggests not taking the trade, skip it
            for conflict in conflicts:
                if conflict.get('recommendation') == 'ignore_new_setup':
                    logger.warning(f"Conflict detected: {conflict.get('reason')}")
                    return False

        return True

    async def _execute_quality_adjusted_trade(
            self,
            bar: pd.Series,
            trade_type: str,
            grade: str,
            maturity_score: float
    ):
        """Execute trade with quality-adjusted risk parameters."""
        try:
            # Adjust conviction score based on setup grade
            grade_multiplier = self.grade_risk_multipliers.get(grade, 1.0)
            adjusted_conviction = min(5, int(self.base_conviction_score * grade_multiplier))

            # Prepare risk engine parameters
            entry_time = pd.to_datetime(bar['timestamp']).tz_localize('UTC')
            ohlc_for_engine = self.historical_data.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
            })

            engine_params = {
                'account_balance': self.account_balance,
                'conviction_score': adjusted_conviction,
                'entry_price': bar['close'],
                'entry_time': entry_time,
                'trade_type': trade_type,
                'symbol': 'XAUUSD',
                'ohlc_data': ohlc_for_engine,
                'risk_config': self.risk_config,
                'atr_config': self.atr_config
            }

            # Calculate risk parameters
            risk_result = calculate_sl_and_risk(**engine_params)

            if risk_result and risk_result.get('status') == 'success':
                self.position = 'long' if trade_type == 'buy' else 'short'

                stop_loss = risk_result['final_sl']
                lot_size = risk_result['lot_size']

                # Adjust R:R ratio based on grade (optional enhancement)
                quality_adjusted_rr = self.rr_ratio
                if grade == 'A':
                    quality_adjusted_rr *= 1.2  # Higher reward for A-grade setups

                # Calculate take profit
                sl_distance = abs(bar['close'] - stop_loss)
                tp_distance = sl_distance * quality_adjusted_rr
                take_profit = bar['close'] + tp_distance if trade_type == 'buy' else bar['close'] - tp_distance

                # Create trade payload with quality metadata
                trade_payload = {
                    'agent_id': self.agent_id,
                    'symbol': 'XAUUSD',
                    'direction': self.position,
                    'entry_price': bar['close'],
                    'stop_loss': round(stop_loss, 2),
                    'take_profit': round(take_profit, 2),
                    'lot_size': lot_size,
                    'timestamp': bar['timestamp'],
                    'metadata': {
                        'setup_grade': grade,
                        'maturity_score': round(maturity_score, 3),
                        'adjusted_conviction': adjusted_conviction,
                        'risk_percent': risk_result['risk_percent_final']
                    }
                }

                # Track active trade
                self.active_trades.append({
                    'id': f"{self.agent_id}_{bar['timestamp']}",
                    'direction': self.position,
                    'entry_time': bar['timestamp'],
                    'grade': grade
                })

                # Update statistics
                self.trade_stats['trades_executed'] += 1

                # Log and execute
                logger.warning(
                    f"🎯 GRADE {grade} {self.position.upper()} SIGNAL - Score: {maturity_score:.3f} | "
                    f"Entry: {bar['close']:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | "
                    f"Lots: {lot_size:.2f} | Risk: {risk_result['risk_percent_final']:.1f}%"
                )

                await self.orchestrator.route_trigger('action.trade.execute', trade_payload, {})

            else:
                logger.error(f"Risk Engine failed: {risk_result.get('error')}")

        except Exception as e:
            logger.error(f"Error executing quality-adjusted trade: {e}", exc_info=True)

    def _handle_trade_closure(self, payload: Dict[str, Any]):
        """Handle trade closure event."""
        trade_id = payload.get('trade_id')

        # Remove from active trades
        self.active_trades = [t for t in self.active_trades if t.get('id') != trade_id]

        # Reset position if no active trades
        if not self.active_trades:
            self.position = 'flat'

        # Log closure with grade info
        logger.info(f"Trade closed: {trade_id}. Active trades: {len(self.active_trades)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Return agent statistics including predictive engine metrics."""
        total_evaluated = self.trade_stats['setups_evaluated']

        if total_evaluated > 0:
            grade_percentages = {
                grade: (count / total_evaluated * 100)
                for grade, count in self.trade_stats['grade_distribution'].items()
            }
            execution_rate = self.trade_stats['trades_executed'] / total_evaluated * 100
        else:
            grade_percentages = {grade: 0.0 for grade in ['A', 'B', 'C', 'D']}
            execution_rate = 0.0

        return {
            'agent_id': self.agent_id,
            'position': self.position,
            'active_trades': len(self.active_trades),
            'statistics': {
                'setups_evaluated': total_evaluated,
                'trades_executed': self.trade_stats['trades_executed'],
                'trades_skipped': self.trade_stats['trades_skipped'],
                'execution_rate': f"{execution_rate:.1f}%",
                'grade_distribution': self.trade_stats['grade_distribution'],
                'grade_percentages': {k: f"{v:.1f}%" for k, v in grade_percentages.items()}
            },
            'configuration': {
                'min_grade_to_trade': self.min_grade_to_trade,
                'predictive_engine_enabled': self.predictive_config.predictive_scorer.enabled
            }
        }
