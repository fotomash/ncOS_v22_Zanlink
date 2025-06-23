"""
Neural Mesh - Multi-agent consensus and signal validation system
Phoenix v18 neural mesh implementation ported to v11
"""
import numpy as np
from typing import Dict, List, Any

class NeuralMesh:
    """
    Neural mesh for multi-agent consensus and signal validation.

    Features:
    - Dynamic weight adjustment based on agent performance
    - Trust score tracking
    - Consensus calculation with configurable thresholds
    """

    def __init__(self, num_agents: int = 7, weight_decay: float = 0.95):
        """
        Initialize neural mesh.

        Args:
            num_agents: Number of agents in the mesh
            weight_decay: Decay factor for weight updates (0-1)
        """
        self.num_agents = num_agents
        self.weight_decay = weight_decay

        # Initialize weight matrix (agents x agents)
        self.weights = np.eye(num_agents) * 0.5
        random_weights = np.random.random((num_agents, num_agents)) * 0.1
        self.weights += random_weights
        self.weights = self.weights / np.sum(self.weights, axis=1, keepdims=True)

        self.trust_scores = np.ones(num_agents)
        self.performance_history = []

        print(f"🔗 NeuralMesh initialized with {num_agents} agents")

    def consensus_score(self, signals: List[Dict[str, Any]]) -> float:
        """
        Calculate consensus score for a set of trading signals.

        Args:
            signals: List of trading signal dictionaries

        Returns:
            Consensus score (0-1)
        """
        if not signals:
            return 0.0

        scores = []
        for signal in signals:
            agent_id = signal.get('agent_id', 0)
            if agent_id >= self.num_agents: agent_id = 0

            base_confidence = signal.get('confidence', 0.5)
            trust_weight = self.trust_scores[agent_id]
            connectivity_weight = np.mean(self.weights[agent_id, :])

            weighted_score = base_confidence * trust_weight * (1 + connectivity_weight)
            scores.append(weighted_score)

        if scores:
            consensus = np.mean(scores)
            consensus = 1 / (1 + np.exp(-5 * (consensus - 0.5))) # Sigmoid activation
            return float(consensus)

        return 0.0

    def get_mesh_state(self) -> Dict[str, Any]:
        """Get current mesh state for monitoring."""
        return {
            'num_agents': self.num_agents,
            'weights': self.weights.tolist(),
            'trust_scores': self.trust_scores.tolist(),
            'average_connectivity': float(np.mean(self.weights)),
        }