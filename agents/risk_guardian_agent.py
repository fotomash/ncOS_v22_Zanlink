#!/usr/bin/env python3
'''
Enhanced Risk Guardian Agent for ZANFLOW v18
Advanced risk management with portfolio protection
'''

from datetime import datetime
from typing import Dict

from production_logging import get_logger


class RiskGuardianAgent:
    '''Advanced risk management agent with veto power'''

    def __init__(self, config: Dict):
        self.config = config
        self.agent_id = "risk_guardian"
        self.logger = get_logger(f"agent.{self.agent_id}", agent_id=self.agent_id)
        self.active = True
        self.analysis_history = []
        self.veto_power = config.get('veto_power', True)

        # Risk parameters
        self.max_exposure = config.get('max_exposure', 0.05)  # 5% max exposure
        self.max_drawdown = config.get('max_drawdown', 0.10)  # 10% max drawdown
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% daily loss limit
        self.position_size_limit = config.get('position_size_limit', 0.02)  # 2% per trade

        # Portfolio tracking
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.open_positions = []

    def analyze(self, request: Dict) -> Dict:
        '''Analyze risk for trading decisions'''
        try:
            data = request.get('data', {})
            analysis_type = request.get('type', 'risk_assessment')

            if analysis_type == 'trade_decision':
                return self._analyze_trade_risk(data)
            elif analysis_type == 'portfolio_risk':
                return self._analyze_portfolio_risk()
            else:
                return self._default_risk_analysis(data)

        except Exception as e:
            return {
                'decision': 'block',
                'confidence': 1.0,
                'reasoning': f'Risk analysis error - blocking for safety: {str(e)}',
                'risk_assessment': {'risk_score': 100}
            }

    def _analyze_trade_risk(self, data: Dict) -> Dict:
        '''Analyze risk for specific trade'''
        symbol = data.get('symbol', 'UNKNOWN')
        position_size = data.get('position_size', self.position_size_limit)

        # Risk assessment components
        risk_assessment = {
            'position_size_risk': self._assess_position_size_risk(position_size),
            'exposure_risk': self._assess_exposure_risk(position_size),
            'correlation_risk': self._assess_correlation_risk(symbol),
            'drawdown_risk': self._assess_drawdown_risk(),
            'daily_loss_risk': self._assess_daily_loss_risk(),
            'market_risk': self._assess_market_risk(data)
        }

        # Calculate overall risk score
        risk_score = self._calculate_overall_risk_score(risk_assessment)

        # Make risk decision
        decision = self._make_risk_decision(risk_score, risk_assessment)
        confidence = 1.0  # Risk guardian always confident
        reasoning = self._generate_risk_reasoning(risk_assessment, risk_score, decision)

        result = {
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_assessment': {
                **risk_assessment,
                'risk_score': risk_score,
                'risk_level': self._get_risk_level(risk_score)
            },
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat()
        }

        self.analysis_history.append(result)
        return result

    def _assess_position_size_risk(self, position_size: float) -> Dict:
        '''Assess position size risk'''
        risk_ratio = position_size / self.position_size_limit

        return {
            'position_size': position_size,
            'limit': self.position_size_limit,
            'risk_ratio': risk_ratio,
            'risk_score': min(risk_ratio * 50, 100),  # Scale to 0-100
            'status': 'high' if risk_ratio > 1.0 else 'acceptable'
        }

    def _assess_exposure_risk(self, additional_position: float) -> Dict:
        '''Assess total exposure risk'''
        new_exposure = self.current_exposure + additional_position
        risk_ratio = new_exposure / self.max_exposure

        return {
            'current_exposure': self.current_exposure,
            'additional_position': additional_position,
            'new_exposure': new_exposure,
            'limit': self.max_exposure,
            'risk_ratio': risk_ratio,
            'risk_score': min(risk_ratio * 60, 100),
            'status': 'high' if risk_ratio > 1.0 else 'acceptable'
        }

    def _assess_correlation_risk(self, symbol: str) -> Dict:
        '''Assess correlation risk with existing positions'''
        # Simplified correlation assessment
        correlated_positions = [pos for pos in self.open_positions
                                if self._are_correlated(symbol, pos.get('symbol', ''))]

        correlation_exposure = sum(pos.get('size', 0) for pos in correlated_positions)

        return {
            'symbol': symbol,
            'correlated_positions': len(correlated_positions),
            'correlation_exposure': correlation_exposure,
            'risk_score': min(len(correlated_positions) * 20, 100),
            'status': 'high' if len(correlated_positions) > 2 else 'acceptable'
        }

    def _assess_drawdown_risk(self) -> Dict:
        '''Assess current drawdown risk'''
        # Simplified drawdown calculation
        current_drawdown = abs(min(self.daily_pnl, 0))
        risk_ratio = current_drawdown / self.max_drawdown

        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': self.max_drawdown,
            'risk_ratio': risk_ratio,
            'risk_score': min(risk_ratio * 80, 100),
            'status': 'high' if risk_ratio > 0.8 else 'acceptable'
        }

    def _assess_daily_loss_risk(self) -> Dict:
        '''Assess daily loss limit risk'''
        daily_loss = abs(min(self.daily_pnl, 0))
        risk_ratio = daily_loss / self.max_daily_loss

        return {
            'daily_pnl': self.daily_pnl,
            'daily_loss': daily_loss,
            'daily_limit': self.max_daily_loss,
            'risk_ratio': risk_ratio,
            'risk_score': min(risk_ratio * 90, 100),
            'status': 'critical' if risk_ratio > 0.9 else 'acceptable'
        }

    def _assess_market_risk(self, data: Dict) -> Dict:
        '''Assess general market risk'''
        # Simplified market risk assessment
        spread = data.get('spread', 0)
        volatility = data.get('volatility', 0.5)

        # Higher spread and volatility = higher risk
        spread_risk = min(spread * 10000, 50)  # Convert to pips and scale
        volatility_risk = min(volatility * 100, 50)

        market_risk_score = spread_risk + volatility_risk

        return {
            'spread': spread,
            'volatility': volatility,
            'spread_risk': spread_risk,
            'volatility_risk': volatility_risk,
            'risk_score': market_risk_score,
            'status': 'high' if market_risk_score > 70 else 'acceptable'
        }

    def _calculate_overall_risk_score(self, risk_assessment: Dict) -> float:
        '''Calculate weighted overall risk score'''
        weights = {
            'position_size_risk': 0.2,
            'exposure_risk': 0.25,
            'correlation_risk': 0.15,
            'drawdown_risk': 0.2,
            'daily_loss_risk': 0.15,
            'market_risk': 0.05
        }

        total_score = 0
        total_weight = 0

        for risk_type, weight in weights.items():
            if risk_type in risk_assessment:
                score = risk_assessment[risk_type].get('risk_score', 0)
                total_score += score * weight
                total_weight += weight

        risk_score = total_score / total_weight if total_weight > 0 else 0

        if risk_score > 20:
            contributions = {}
            for risk_type, weight in weights.items():
                if risk_type in risk_assessment:
                    contribution = risk_assessment[risk_type].get('risk_score', 0) * weight
                    contributions[risk_type] = contribution
                    # Record each component's contribution to the overall risk score
                    self.logger.info(
                        f"{risk_type} contribution: {contribution:.2f}"
                    )
                    self.logger.debug(
                        f"{risk_type} contribution: {contribution:.2f}",
                    )
            self.analysis_history.append({'component_contributions': contributions})

        return risk_score

    def _make_risk_decision(self, risk_score: float, risk_assessment: Dict) -> str:
        '''Make risk decision based on assessment'''
        # Critical risk factors that trigger immediate block
        daily_loss_status = risk_assessment.get('daily_loss_risk', {}).get('status')
        if daily_loss_status == 'critical':
            return 'block'

        # High risk score threshold
        if risk_score >= 80:
            return 'block'
        elif risk_score >= 60:
            return 'reduce'  # Suggest position size reduction
        elif risk_score >= 40:
            return 'caution'  # Proceed with caution
        else:
            return 'approve'

    def _get_risk_level(self, risk_score: float) -> str:
        '''Get risk level description'''
        if risk_score >= 80:
            return 'critical'
        elif risk_score >= 60:
            return 'high'
        elif risk_score >= 40:
            return 'medium'
        elif risk_score >= 20:
            return 'low'
        else:
            return 'minimal'

    def _generate_risk_reasoning(self, risk_assessment: Dict, risk_score: float, decision: str) -> str:
        '''Generate risk reasoning'''
        reasoning_parts = []

        reasoning_parts.append(f"Overall risk score: {risk_score:.1f}")

        # Highlight critical factors
        for risk_type, assessment in risk_assessment.items():
            if isinstance(assessment, dict) and assessment.get('status') in ['high', 'critical']:
                reasoning_parts.append(f"{risk_type}: {assessment['status']}")

        reasoning_parts.append(f"Risk decision: {decision}")

        return f"Risk Assessment: {'; '.join(reasoning_parts)}"

    def _are_correlated(self, symbol1: str, symbol2: str) -> bool:
        '''Check if two symbols are correlated'''
        # Simplified correlation check
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']

        # Same base or quote currency
        if len(symbol1) >= 6 and len(symbol2) >= 6:
            base1, quote1 = symbol1[:3], symbol1[3:6]
            base2, quote2 = symbol2[:3], symbol2[3:6]

            return base1 == base2 or quote1 == quote2 or base1 == quote2 or quote1 == base2

        return False

    def assess_portfolio_risk(self) -> Dict:
        '''Assess overall portfolio risk'''
        portfolio_risk = {
            'total_exposure': self.current_exposure,
            'exposure_utilization': self.current_exposure / self.max_exposure,
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
            'risk_score': self._calculate_portfolio_risk_score()
        }

        return {
            'decision': 'assess',
            'confidence': 1.0,
            'reasoning': f"Portfolio risk score: {portfolio_risk['risk_score']:.1f}",
            'risk_assessment': portfolio_risk,
            'agent_id': self.agent_id
        }

    def _calculate_portfolio_risk_score(self) -> float:
        '''Calculate overall portfolio risk score'''
        factors = []

        # Exposure factor
        exposure_ratio = self.current_exposure / self.max_exposure
        factors.append(min(exposure_ratio * 100, 100))

        # Daily loss factor
        daily_loss = abs(min(self.daily_pnl, 0))
        loss_ratio = daily_loss / self.max_daily_loss
        factors.append(min(loss_ratio * 100, 100))

        # Position count factor
        position_factor = min(len(self.open_positions) * 10, 50)
        factors.append(position_factor)

        return sum(factors) / len(factors) if factors else 0

    def _analyze_portfolio_risk(self) -> Dict:
        '''Analyze portfolio risk'''
        return self.assess_portfolio_risk()

    def _default_risk_analysis(self, data: Dict) -> Dict:
        '''Default risk analysis'''
        return {
            'decision': 'approve',
            'confidence': 0.8,
            'reasoning': 'Default risk analysis - no specific risks identified',
            'risk_assessment': {'risk_score': 20},
            'agent_id': self.agent_id
        }

    def update_position(self, symbol: str, size: float, pnl: float = 0):
        '''Update position tracking'''
        # Update exposure
        self.current_exposure += abs(size)

        # Update daily P&L
        self.daily_pnl += pnl

        # Add to open positions
        if size != 0:
            self.open_positions.append({
                'symbol': symbol,
                'size': size,
                'timestamp': datetime.now().isoformat()
            })

    def close_position(self, symbol: str, pnl: float):
        '''Close position and update tracking'''
        # Remove from open positions
        self.open_positions = [pos for pos in self.open_positions
                               if pos.get('symbol') != symbol]

        # Update daily P&L
        self.daily_pnl += pnl

    def reset_daily_tracking(self):
        '''Reset daily tracking (call at start of new day)'''
        self.daily_pnl = 0.0

    def is_active(self) -> bool:
        return self.active

    def emergency_stop(self):
        self.active = False
        print(f"🛑 {self.agent_id} emergency stopped - ALL TRADING BLOCKED")
