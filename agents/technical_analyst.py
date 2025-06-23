"""
TechnicalAnalyst - NCOS v21 Agent
Fixed version with proper config fields
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class TechnicalAnalystConfig:
    """Configuration for TechnicalAnalyst"""
    agent_id: str = "technical_analyst"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    indicators: Optional[List[str]] = field(default_factory=lambda: ["sma", "ema", "rsi", "macd"])
    timeframes: Optional[List[str]] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    lookback_period: Optional[int] = 100
    custom_params: Dict[str, Any] = field(default_factory=dict)


class TechnicalAnalyst:
    """
    TechnicalAnalyst - Technical analysis and indicator calculation
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = TechnicalAnalystConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "analyses_performed": 0,
            "indicators_calculated": 0
        }

        self.indicators = {}
        self.timeframes = self.config.timeframes
        self.analysis_cache = {}

        self.logger.info(f"{self.config.agent_id} initialized")

    async def initialize(self):
        """Initialize the agent"""
        try:
            self.logger.info(f"Initializing {self.config.agent_id}")
            await self._setup()
            self.status = "ready"
            self.logger.info(f"{self.config.agent_id} ready for operation")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.agent_id}: {e}")
            self.status = "error"
            raise

    async def _setup(self):
        """Agent-specific setup logic"""
        # Initialize technical indicators
        for indicator in self.config.indicators:
            self.indicators[indicator] = {
                "enabled": True,
                "last_calculated": None,
                "parameters": self._get_default_params(indicator)
            }
            self.logger.info(f"Initialized indicator: {indicator}")

    def _get_default_params(self, indicator: str) -> Dict[str, Any]:
        """Get default parameters for an indicator"""
        defaults = {
            "sma": {"period": 20},
            "ema": {"period": 20},
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9}
        }
        return defaults.get(indicator, {})

    async def _perform_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis on data"""
        symbol = data.get("symbol", "UNKNOWN")
        prices = data.get("prices", [])

        if not prices:
            return {"error": "No price data provided"}

        analysis_result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "indicators": {},
            "signals": []
        }

        # Calculate each enabled indicator
        for indicator_name, indicator_config in self.indicators.items():
            if indicator_config["enabled"]:
                try:
                    result = await self._calculate_indicator_value(indicator_name, prices,
                                                                   indicator_config["parameters"])
                    analysis_result["indicators"][indicator_name] = result
                    self.metrics["indicators_calculated"] += 1
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator_name}: {e}")
                    analysis_result["indicators"][indicator_name] = {"error": str(e)}

        # Generate trading signals
        signals = self._generate_signals(analysis_result["indicators"])
        analysis_result["signals"] = signals

        self.metrics["analyses_performed"] += 1
        self.analysis_cache[symbol] = analysis_result

        return analysis_result

    async def _calculate_indicator_value(self, indicator: str, prices: List[float], params: Dict[str, Any]) -> Dict[
        str, Any]:
        """Calculate a specific indicator value"""
        if indicator == "sma":
            return await self._calculate_sma(prices, params.get("period", 20))
        elif indicator == "ema":
            return await self._calculate_ema(prices, params.get("period", 20))
        elif indicator == "rsi":
            return await self._calculate_rsi(prices, params.get("period", 14))
        elif indicator == "macd":
            return await self._calculate_macd(prices, params.get("fast", 12), params.get("slow", 26),
                                              params.get("signal", 9))
        else:
            return {"error": f"Unknown indicator: {indicator}"}

    async def _calculate_sma(self, prices: List[float], period: int) -> Dict[str, Any]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return {"error": f"Insufficient data for SMA({period})"}

        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)

        return {
            "type": "sma",
            "period": period,
            "values": sma_values,
            "current": sma_values[-1] if sma_values else None
        }

    async def _calculate_ema(self, prices: List[float], period: int) -> Dict[str, Any]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return {"error": f"Insufficient data for EMA({period})"}

        multiplier = 2 / (period + 1)
        ema_values = [prices[0]]  # Start with first price

        for price in prices[1:]:
            ema = (price * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)

        return {
            "type": "ema",
            "period": period,
            "values": ema_values,
            "current": ema_values[-1] if ema_values else None
        }

    async def _calculate_rsi(self, prices: List[float], period: int) -> Dict[str, Any]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return {"error": f"Insufficient data for RSI({period})"}

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]

        # Calculate average gains and losses
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return {
            "type": "rsi",
            "period": period,
            "current": rsi,
            "signal": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        }

    async def _calculate_macd(self, prices: List[float], fast: int, slow: int, signal: int) -> Dict[str, Any]:
        """Calculate MACD"""
        if len(prices) < slow:
            return {"error": f"Insufficient data for MACD({fast},{slow},{signal})"}

        # Calculate EMAs
        ema_fast = await self._calculate_ema(prices, fast)
        ema_slow = await self._calculate_ema(prices, slow)

        if "error" in ema_fast or "error" in ema_slow:
            return {"error": "Error calculating MACD EMAs"}

        # Calculate MACD line
        macd_line = ema_fast["current"] - ema_slow["current"]

        return {
            "type": "macd",
            "fast": fast,
            "slow": slow,
            "signal": signal,
            "macd_line": macd_line,
            "signal_line": 0,  # Simplified
            "histogram": macd_line
        }

    def _generate_signals(self, indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals based on indicators"""
        signals = []

        # RSI signals
        if "rsi" in indicators and "current" in indicators["rsi"]:
            rsi_value = indicators["rsi"]["current"]
            if rsi_value > 70:
                signals.append({"type": "sell", "reason": "RSI overbought", "strength": "medium"})
            elif rsi_value < 30:
                signals.append({"type": "buy", "reason": "RSI oversold", "strength": "medium"})

        # MACD signals
        if "macd" in indicators and "macd_line" in indicators["macd"]:
            macd_line = indicators["macd"]["macd_line"]
            if macd_line > 0:
                signals.append({"type": "buy", "reason": "MACD bullish", "strength": "weak"})
            elif macd_line < 0:
                signals.append({"type": "sell", "reason": "MACD bearish", "strength": "weak"})

        return signals

    async def _calculate_indicator(self, indicator: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a specific indicator"""
        prices = data.get("prices", [])
        params = data.get("parameters", {})

        return await self._calculate_indicator_value(indicator, prices, params)

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message"""
        try:
            self.metrics["messages_processed"] += 1
            self.logger.debug(f"Processing message: {message.get('type', 'unknown')}")

            result = await self._handle_message(message)

            return {
                "status": "success",
                "agent_id": self.config.agent_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error processing message: {e}")
            return {
                "status": "error",
                "agent_id": self.config.agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_message(self, message: Dict[str, Any]) -> Any:
        """Agent-specific message handling"""
        # Handle technical analysis requests
        msg_type = message.get("type")
        if msg_type == "analyze":
            return await self._perform_analysis(message.get("data"))
        elif msg_type == "calculate_indicator":
            return await self._calculate_indicator(message.get("indicator"), message.get("data"))
        elif msg_type == "get_indicators":
            return {"indicators": list(self.indicators.keys())}
        elif msg_type == "get_analysis":
            symbol = message.get("symbol")
            return self.analysis_cache.get(symbol, {"error": "No analysis found"})

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "enabled_indicators": list(self.indicators.keys()),
            "cached_analyses": len(self.analysis_cache)
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")
        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> TechnicalAnalyst:
    """Factory function to create TechnicalAnalyst instance"""
    return TechnicalAnalyst(config)


# Export the agent class
__all__ = ["TechnicalAnalyst", "TechnicalAnalystConfig", "create_agent"]
