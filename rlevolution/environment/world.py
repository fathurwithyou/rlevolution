from __future__ import annotations

import math

import numpy as np

from rlevolution.config import ACTION_NAMES, SimulationConfig
from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.environment.types import WorldStepInfo, WorldStepResult
from rlevolution.rl import State


class NaturalSelectionWorld:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.food: np.ndarray = np.empty((0, 2), dtype=float)
        self.hazards: list[tuple[np.ndarray, float]] = []
        self.reset()

    def reset(self) -> None:
        self.food = self._random_positions(self.config.food_count)
        self.hazards = [
            (
                np.array(
                    [
                        self.rng.uniform(10, self.config.world_width - 10),
                        self.rng.uniform(10, self.config.world_height - 10),
                    ],
                    dtype=float,
                ),
                self.config.hazard_radius * float(self.rng.uniform(0.85, 1.25)),
            )
            for _ in range(self.config.hazard_count)
        ]

    def add_food(self, count: int = 1) -> None:
        if count <= 0:
            return
        self.food = np.vstack([self.food, self._random_positions(count)])

    def state_for(self, agent: AdaptiveAgent) -> State:
        food_index, food_distance, food_delta = self.nearest_food(agent)
        hazard_distance, hazard_delta, _ = self.nearest_hazard(agent)
        diagonal = math.hypot(self.config.world_width, self.config.world_height)

        return (
            self._bin(agent.position[0], self.config.world_width, 5),
            self._bin(agent.position[1], self.config.world_height, 5),
            self._direction_bin(food_delta[0]) if food_index is not None else 1,
            self._direction_bin(food_delta[1]) if food_index is not None else 1,
            self._distance_bin(food_distance, diagonal),
            self._direction_bin(hazard_delta[0]) if self.hazards else 1,
            self._direction_bin(hazard_delta[1]) if self.hazards else 1,
            self._distance_bin(max(0.0, hazard_distance), diagonal),
            self._bin(agent.energy, self.config.max_energy, 5),
            self._bin(agent.age, max(1, self.config.max_agent_age), 5),
        )

    def step_agent(
        self,
        agent: AdaptiveAgent,
        action_index: int,
        agents: list[AdaptiveAgent],
    ) -> WorldStepResult:
        if not agent.alive:
            return WorldStepResult(
                state=self.state_for(agent),
                reward=0.0,
                terminated=True,
                info=WorldStepInfo(),
            )

        action = ACTION_NAMES[action_index]
        reward = self.config.survival_reward
        info = WorldStepInfo()
        movement = self._movement_vector(action)

        if np.linalg.norm(movement) > 0:
            agent.position += movement * agent.traits.speed
            agent.position = np.clip(
                agent.position,
                [0.0, 0.0],
                [float(self.config.world_width), float(self.config.world_height)],
            )

        agent.age += 1
        agent.energy -= self._movement_cost(agent, movement)

        if action == "reproduce":
            agent.energy -= self.config.reproduce_search_cost
            reproduction_reward, info = self._try_reproduction(agent, agents)
            reward += reproduction_reward

        reward += self._consume_food(agent)
        reward += self._apply_hazard(agent)

        done = False
        if agent.energy <= 0 or agent.age >= self.config.max_agent_age:
            agent.alive = False
            done = True
            reward += self.config.death_reward

        agent.energy = float(np.clip(agent.energy, 0.0, self.config.max_energy))
        agent.total_reward += reward
        return WorldStepResult(
            state=self.state_for(agent),
            reward=float(reward),
            terminated=done,
            info=info,
        )

    def nearest_food(self, agent: AdaptiveAgent) -> tuple[int | None, float, np.ndarray]:
        if len(self.food) == 0:
            return None, math.inf, np.zeros(2)
        deltas = self.food - agent.position
        distances = np.linalg.norm(deltas, axis=1)
        index = int(np.argmin(distances))
        return index, float(distances[index]), deltas[index]

    def nearest_hazard(self, agent: AdaptiveAgent) -> tuple[float, np.ndarray, float]:
        if not self.hazards:
            return math.inf, np.zeros(2), 0.0

        best_distance = math.inf
        best_delta = np.zeros(2)
        best_radius = 0.0
        for center, radius in self.hazards:
            delta = center - agent.position
            edge_distance = float(np.linalg.norm(delta) - radius)
            if edge_distance < best_distance:
                best_distance = edge_distance
                best_delta = delta
                best_radius = radius
        return best_distance, best_delta, best_radius

    def _random_positions(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, 2), dtype=float)
        return np.column_stack(
            [
                self.rng.uniform(0, self.config.world_width, count),
                self.rng.uniform(0, self.config.world_height, count),
            ]
        ).astype(float)

    @staticmethod
    def _movement_vector(action: str) -> np.ndarray:
        movement = np.zeros(2, dtype=float)
        if action == "up":
            movement[1] = 1.0
        elif action == "down":
            movement[1] = -1.0
        elif action == "left":
            movement[0] = -1.0
        elif action == "right":
            movement[0] = 1.0
        return movement

    def _movement_cost(self, agent: AdaptiveAgent, movement: np.ndarray) -> float:
        return (
            self.config.base_metabolism
            * (1.0 + 0.38 * float(np.linalg.norm(movement)) * agent.traits.speed)
            / agent.traits.endurance
        )

    def _try_reproduction(
        self,
        agent: AdaptiveAgent,
        agents: list[AdaptiveAgent],
    ) -> tuple[float, WorldStepInfo]:
        if agent.energy < self.config.reproduction_energy_threshold:
            return -0.4, WorldStepInfo()

        candidates = [
            other
            for other in agents
            if other.alive
            and other.agent_id != agent.agent_id
            and other.energy >= self.config.reproduction_energy_threshold
            and np.linalg.norm(other.position - agent.position)
            <= self.config.reproduction_radius
        ]
        if not candidates:
            return -0.25, WorldStepInfo()

        partner = min(candidates, key=lambda other: np.linalg.norm(other.position - agent.position))
        probability = (agent.traits.reproduction + partner.traits.reproduction) / 2.0
        if self.rng.random() > probability:
            return -0.2, WorldStepInfo()

        agent.energy -= self.config.reproduction_energy_cost
        partner.energy -= self.config.reproduction_energy_cost * 0.65
        agent.reproduction_count += 1
        partner.reproduction_count += 1
        return self.config.reproduction_reward, WorldStepInfo(
            reproduced=True,
            partner=partner,
        )

    def _consume_food(self, agent: AdaptiveAgent) -> float:
        food_index, food_distance, _ = self.nearest_food(agent)
        if food_index is None:
            return 0.0

        effective_radius = self.config.food_radius * agent.traits.foraging
        if food_distance > effective_radius:
            return 0.0

        agent.energy += self.config.food_energy * agent.traits.foraging
        agent.food_eaten += 1
        self.food = np.delete(self.food, food_index, axis=0)
        self.add_food(1)
        return self.config.food_reward

    def _apply_hazard(self, agent: AdaptiveAgent) -> float:
        penalty = 0.0
        for center, radius in self.hazards:
            distance = float(np.linalg.norm(center - agent.position))
            if distance <= radius:
                agent.energy -= self.config.hazard_energy_penalty / agent.traits.endurance
                agent.hazard_hits += 1
                penalty += self.config.hazard_reward
        return penalty

    @staticmethod
    def _bin(value: float, maximum: float, bins: int) -> int:
        if maximum <= 0:
            return 0
        scaled = int((value / maximum) * bins)
        return max(0, min(bins - 1, scaled))

    @staticmethod
    def _distance_bin(distance: float, diagonal: float) -> int:
        if math.isinf(distance):
            return 4
        ratio = distance / max(diagonal, 1.0)
        if ratio < 0.08:
            return 0
        if ratio < 0.18:
            return 1
        if ratio < 0.34:
            return 2
        if ratio < 0.55:
            return 3
        return 4

    @staticmethod
    def _direction_bin(delta: float) -> int:
        if delta < -2.0:
            return 0
        if delta > 2.0:
            return 2
        return 1
