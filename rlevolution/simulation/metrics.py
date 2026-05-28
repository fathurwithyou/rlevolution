from __future__ import annotations

import numpy as np

from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.domain.fitness import FitnessCalculator
from rlevolution.simulation.results import GenerationMetrics


class GenerationMetricsBuilder:
    def __init__(self, fitness: FitnessCalculator | None = None):
        self.fitness = fitness or FitnessCalculator()

    def build(
        self,
        *,
        generation: int,
        steps: int,
        agents: list[AdaptiveAgent],
        births: int,
    ) -> GenerationMetrics:
        total_agents = len(agents)
        alive_agents = [agent for agent in agents if agent.alive]
        fitness_values = np.array([self.fitness.score(agent) for agent in agents], dtype=float)
        energies = np.array([agent.energy for agent in alive_agents], dtype=float)
        trait_matrix = np.array(
            [
                [
                    agent.traits.speed,
                    agent.traits.endurance,
                    agent.traits.foraging,
                    agent.traits.reproduction,
                ]
                for agent in agents
            ],
            dtype=float,
        )

        return GenerationMetrics(
            generation=generation,
            steps=steps,
            population_end=len(alive_agents),
            population_total=total_agents,
            survival_rate=len(alive_agents) / max(1, total_agents),
            avg_fitness=float(fitness_values.mean()) if total_agents else 0.0,
            best_fitness=float(fitness_values.max()) if total_agents else 0.0,
            median_fitness=float(np.median(fitness_values)) if total_agents else 0.0,
            avg_energy_alive=float(energies.mean()) if len(energies) else 0.0,
            food_eaten=int(sum(agent.food_eaten for agent in agents)),
            reproductions=int(births),
            hazard_hits=int(sum(agent.hazard_hits for agent in agents)),
            avg_speed=float(trait_matrix[:, 0].mean()) if total_agents else 0.0,
            avg_endurance=float(trait_matrix[:, 1].mean()) if total_agents else 0.0,
            avg_foraging=float(trait_matrix[:, 2].mean()) if total_agents else 0.0,
            avg_reproduction=float(trait_matrix[:, 3].mean()) if total_agents else 0.0,
            avg_q_states=float(
                np.mean([agent.model.q_state_count() for agent in agents])
            )
            if total_agents
            else 0.0,
        )
