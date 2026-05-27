from __future__ import annotations

import numpy as np

from rlevolution.config import SimulationConfig
from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.domain.traits import Traits
from rlevolution.rl.q_learning import QLearningModel


class GeneticOperator:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def create_offspring(
        self,
        first: AdaptiveAgent,
        second: AdaptiveAgent,
        *,
        agent_id: int,
        generation: int,
        immediate_birth: bool,
    ) -> AdaptiveAgent:
        position = self._child_position(first, second, immediate_birth)
        start_energy = (
            self.config.start_energy * 0.62
            if immediate_birth
            else self.config.start_energy
        )

        return AdaptiveAgent(
            agent_id=agent_id,
            generation=generation,
            position=position,
            traits=self._mix_traits(first.traits, second.traits),
            model=QLearningModel.crossover(first.model, second.model, self.rng).mutated(
                self.config.mutation_rate,
                self.config.mutation_strength,
                self.rng,
            ),
            energy=start_energy,
        )

    def _child_position(
        self,
        first: AdaptiveAgent,
        second: AdaptiveAgent,
        immediate_birth: bool,
    ) -> np.ndarray:
        if immediate_birth:
            position = (first.position + second.position) / 2.0
            position = position + self.rng.normal(0.0, 2.2, size=2)
        else:
            position = np.array(
                [
                    self.rng.uniform(0, self.config.world_width),
                    self.rng.uniform(0, self.config.world_height),
                ],
                dtype=float,
            )

        return np.clip(
            position,
            [0.0, 0.0],
            [float(self.config.world_width), float(self.config.world_height)],
        )

    def _mix_traits(self, first: Traits, second: Traits) -> Traits:
        return Traits(
            speed=self._mutate_trait(
                (first.speed + second.speed) / 2.0,
                self.config.speed_min,
                self.config.speed_max,
            ),
            endurance=self._mutate_trait(
                (first.endurance + second.endurance) / 2.0,
                self.config.endurance_min,
                self.config.endurance_max,
            ),
            foraging=self._mutate_trait(
                (first.foraging + second.foraging) / 2.0,
                self.config.foraging_min,
                self.config.foraging_max,
            ),
            reproduction=self._mutate_trait(
                (first.reproduction + second.reproduction) / 2.0,
                self.config.reproduction_min,
                self.config.reproduction_max,
            ),
        )

    def _mutate_trait(self, value: float, lower: float, upper: float) -> float:
        if self.rng.random() < self.config.mutation_rate:
            value += self.rng.normal(0.0, (upper - lower) * self.config.mutation_strength)
        return float(np.clip(value, lower, upper))
