from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlevolution.domain.fitness import FitnessCalculator
from rlevolution.domain.snapshot import AgentSnapshot
from rlevolution.domain.traits import Traits
from rlevolution.rl.q_learning import QLearningModel


@dataclass
class AdaptiveAgent:
    agent_id: int
    position: np.ndarray
    traits: Traits
    model: QLearningModel
    generation: int
    energy: float
    age: int = 0
    alive: bool = True
    total_reward: float = 0.0
    food_eaten: int = 0
    reproduction_count: int = 0
    hazard_hits: int = 0

    def fitness_score(self, calculator: FitnessCalculator | None = None) -> float:
        return (calculator or FitnessCalculator()).score(self)

    def snapshot(self, calculator: FitnessCalculator | None = None) -> AgentSnapshot:
        return AgentSnapshot(
            agent_id=self.agent_id,
            generation=self.generation,
            x=float(self.position[0]),
            y=float(self.position[1]),
            energy=float(self.energy),
            age=self.age,
            alive=self.alive,
            fitness=self.fitness_score(calculator),
            food_eaten=self.food_eaten,
            reproduction_count=self.reproduction_count,
            hazard_hits=self.hazard_hits,
            speed=self.traits.speed,
            endurance=self.traits.endurance,
            foraging=self.traits.foraging,
            reproduction=self.traits.reproduction,
        )
