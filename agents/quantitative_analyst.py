"""
ncOS Unified v5.0 - Quantitative Analyst Agent
Industry-standard quantitative analysis with vector-native pattern recognition
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class QuantitativeAnalyst:
    """Quantitative analysis following industry standards"""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.analysis_cache = {}
        self.pattern_library = {}

    async def process(self, zbars: List, embeddings: np.ndarray) -> Dict:
        """Perform quantitative analysis"""

        # Statistical analysis
        statistical_metrics = self._statistical_analysis(zbars)

        # Pattern recognition using embeddings
        patterns = self._pattern_recognition(embeddings)

        # Regime detection
        regimes = self._regime_detection(zbars, embeddings)

        # Correlation analysis
        correlations = self._correlation_analysis(zbars)

        # Predictive modeling
        predictions = self._predictive_modeling(zbars, embeddings)

        return {
            "agent": "quantitative_analyst",
            "timestamp": datetime.now().isoformat(),
            "statistical_metrics": statistical_metrics,
            "patterns": patterns,
            "regimes": regimes,
            "correlations": correlations,
            "predictions": predictions,
            "confidence": self._calculate_confidence(statistical_metrics, patterns, regimes)
        }

    def _statistical_analysis(self, zbars: List) -> Dict:
        """Perform statistical analysis on price data"""
        if len(zbars) < 30:
            return {"status": "insufficient_data"}

        # Extract price series
        prices = np.array([getattr(zbar, 'close', 0) for zbar in zbars])
        returns = np.diff(prices) / prices[:-1]

        # Calculate statistical metrics
        metrics = {
            "mean_return": np.mean(returns),
            "volatility": np.std(returns),
            "skewness": stats.skew(returns),
            "kurtosis": stats.kurtosis(returns),
            "sharpe_ratio": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(prices),
            "var_95": np.percentile(returns, 5),
            "var_99": np.percentile(returns, 1)
        }

        # Normality test
        _, p_value = stats.jarque_bera(returns)
        metrics["normality_p_value"] = p_value
        metrics["is_normal"] = p_value > 0.05

        return metrics

    def _pattern_recognition(self, embeddings: np.ndarray) -> Dict:
        """Recognize patterns using vector embeddings"""
        if embeddings.size == 0:
            return {"patterns_found": 0}

        # Reshape embeddings if needed
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Cluster similar patterns
        n_clusters = min(5, len(embeddings))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)

            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_size = np.sum(cluster_mask)
                cluster_stats[f"cluster_{i}"] = {
                    "size": int(cluster_size),
                    "percentage": float(cluster_size / len(clusters))
                }
        else:
            cluster_stats = {"cluster_0": {"size": len(embeddings), "percentage": 1.0}}

        # Find similar patterns using cosine similarity
        if len(embeddings) > 1:
            similarity_matrix = cosine_similarity(embeddings)
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        else:
            avg_similarity = 1.0

        return {
            "patterns_found": len(cluster_stats),
            "cluster_distribution": cluster_stats,
            "average_similarity": float(avg_similarity),
            "pattern_strength": "high" if avg_similarity > 0.8 else "medium" if avg_similarity > 0.6 else "low"
        }

    def _regime_detection(self, zbars: List, embeddings: np.ndarray) -> Dict:
        """Detect market regimes"""
        if len(zbars) < 50:
            return {"regime": "unknown", "confidence": 0}

        # Extract volatility and returns
        prices = np.array([getattr(zbar, 'close', 0) for zbar in zbars])
        returns = np.diff(prices) / prices[:-1]

        # Rolling volatility
        window = 20
        rolling_vol = pd.Series(returns).rolling(window).std().values

        # Regime classification based on volatility
        recent_vol = np.mean(rolling_vol[-10:])
        historical_vol = np.mean(rolling_vol[:-10])

        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1

        if vol_ratio > 1.5:
            regime = "high_volatility"
            confidence = min((vol_ratio - 1) / 0.5, 1.0)
        elif vol_ratio < 0.7:
            regime = "low_volatility"
            confidence = min((1 - vol_ratio) / 0.3, 1.0)
        else:
            regime = "normal"
            confidence = 1 - abs(vol_ratio - 1) / 0.5

        # Trend detection
        recent_returns = returns[-20:]
        trend_strength = np.mean(recent_returns) / np.std(recent_returns) if np.std(recent_returns) > 0 else 0

        if abs(trend_strength) > 1:
            trend = "strong_trend"
        elif abs(trend_strength) > 0.5:
            trend = "weak_trend"
        else:
            trend = "sideways"

        return {
            "volatility_regime": regime,
            "trend_regime": trend,
            "volatility_ratio": float(vol_ratio),
            "trend_strength": float(trend_strength),
            "confidence": float(confidence)
        }

    def _correlation_analysis(self, zbars: List) -> Dict:
        """Analyze correlations between different metrics"""
        if len(zbars) < 30:
            return {"correlations": {}}

        # Extract multiple time series
        data = {
            "close": [getattr(zbar, 'close', 0) for zbar in zbars],
            "volume": [getattr(zbar, 'volume', 0) for zbar in zbars],
            "high": [getattr(zbar, 'high', 0) for zbar in zbars],
            "low": [getattr(zbar, 'low', 0) for zbar in zbars]
        }

        df = pd.DataFrame(data)
        correlation_matrix = df.corr()

        return {
            "price_volume_corr": float(correlation_matrix.loc["close", "volume"]),
            "high_low_corr": float(correlation_matrix.loc["high", "low"]),
            "correlation_matrix": correlation_matrix.to_dict()
        }

    def _predictive_modeling(self, zbars: List, embeddings: np.ndarray) -> Dict:
        """Simple predictive modeling"""
        if len(zbars) < 50:
            return {"prediction": "insufficient_data"}

        # Extract features
        prices = np.array([getattr(zbar, 'close', 0) for zbar in zbars])
        returns = np.diff(prices) / prices[:-1]

        # Simple momentum prediction
        recent_momentum = np.mean(returns[-5:])
        historical_momentum = np.mean(returns[:-5])

        momentum_signal = "bullish" if recent_momentum > historical_momentum else "bearish"

        # Volatility prediction
        recent_vol = np.std(returns[-20:])
        historical_vol = np.std(returns[:-20])

        vol_prediction = "increasing" if recent_vol > historical_vol else "decreasing"

        return {
            "momentum_signal": momentum_signal,
            "volatility_prediction": vol_prediction,
            "confidence": min(abs(recent_momentum - historical_momentum) * 100, 1.0)
        }

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    def _calculate_confidence(self, statistical_metrics: Dict, patterns: Dict, regimes: Dict) -> float:
        """Calculate overall confidence in analysis"""
        confidence_factors = []

        # Statistical confidence
        if statistical_metrics.get("status") != "insufficient_data":
            confidence_factors.append(0.8)

        # Pattern confidence
        pattern_strength = patterns.get("pattern_strength", "low")
        pattern_conf = {"high": 0.9, "medium": 0.7, "low": 0.5}.get(pattern_strength, 0.5)
        confidence_factors.append(pattern_conf)

        # Regime confidence
        regime_conf = regimes.get("confidence", 0.5)
        confidence_factors.append(regime_conf)

        return float(np.mean(confidence_factors))

    def get_health_status(self) -> str:
        """Return agent health status"""
        return "healthy"
