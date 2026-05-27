from __future__ import annotations

import numpy as np

from rlevolution.config import SimulationConfig
from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.domain.fitness import FitnessCalculator


class FitnessProportionateSelector:
    def __init__(
        self,
        config: SimulationConfig,
        fitness: FitnessCalculator | None = None,
    ):
        self.config = config
        self.fitness = fitness or FitnessCalculator()

    def select(
        self,
        agents: list[AdaptiveAgent],
        rng: np.random.Generator,
    ) -> AdaptiveAgent:
        scores = np.array([self.fitness.score(agent) for agent in agents], dtype=float)
        weights = np.power(np.maximum(scores, 0.1), self.config.selection_pressure)
        probabilities = weights / weights.sum()
        return agents[int(rng.choice(len(agents), p=probabilities))]
