from __future__ import annotations

import numpy as np

from rlevolution.config import SimulationConfig
from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.domain.traits import Traits
from rlevolution.rl.q_learning import QLearningModel


class AgentFactory:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def create_random(self, agent_id: int, generation: int) -> AdaptiveAgent:
        return AdaptiveAgent(
            agent_id=agent_id,
            generation=generation,
            position=np.array(
                [
                    self.rng.uniform(0, self.config.world_width),
                    self.rng.uniform(0, self.config.world_height),
                ],
                dtype=float,
            ),
            traits=Traits(
                speed=float(self.rng.uniform(self.config.speed_min, self.config.speed_max)),
                endurance=float(
                    self.rng.uniform(self.config.endurance_min, self.config.endurance_max)
                ),
                foraging=float(
                    self.rng.uniform(self.config.foraging_min, self.config.foraging_max)
                ),
                reproduction=float(
                    self.rng.uniform(
                        self.config.reproduction_min,
                        self.config.reproduction_max,
                    )
                ),
            ),
            model=QLearningModel(),
            energy=self.config.start_energy,
        )
