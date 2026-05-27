from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlevolution.config import ACTION_NAMES, SimulationConfig
from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.domain.factory import AgentFactory
from rlevolution.environment.types import (
    GymInfo,
    GymStepResult,
    RenderFrame,
    WorldStepInfo,
)
from rlevolution.environment.world import NaturalSelectionWorld
from rlevolution.rl import State


class NaturalSelectionGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.rng = rng or np.random.default_rng(config.random_seed)
        self.render_mode = render_mode
        self.world = NaturalSelectionWorld(config, self.rng)
        self.population: list[AdaptiveAgent] = []
        self.controlled_agent: AdaptiveAgent | None = None
        self.elapsed_steps = 0

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.MultiDiscrete(
            np.array([5, 5, 3, 3, 5, 3, 3, 5, 5, 5], dtype=np.int64)
        )

    @property
    def food(self) -> np.ndarray:
        return self.world.food

    @property
    def hazards(self) -> list[tuple[np.ndarray, float]]:
        return self.world.hazards

    def reset_world(self) -> None:
        self.world.reset()
        self.elapsed_steps = 0

    def set_population(self, agents: list[AdaptiveAgent]) -> None:
        self.population = agents
        self.controlled_agent = self._first_alive_agent(agents)

    def observe_agent(self, agent: AdaptiveAgent) -> np.ndarray:
        return np.array(self.world.state_for(agent), dtype=np.int64)

    def state_for(self, agent: AdaptiveAgent) -> State:
        return tuple(int(value) for value in self.observe_agent(agent))

    def step_agent(
        self,
        agent: AdaptiveAgent,
        action: int,
        population: list[AdaptiveAgent],
    ) -> GymStepResult:
        self.population = population
        self.controlled_agent = agent
        result = self.world.step_agent(
            agent,
            int(action),
            population,
        )
        truncated = agent.age >= self.config.max_agent_age
        return GymStepResult(
            observation=np.array(result.state, dtype=np.int64),
            reward=result.reward,
            terminated=result.terminated,
            truncated=truncated,
            info=result.info,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, GymInfo]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.world = NaturalSelectionWorld(self.config, self.rng)
        else:
            self.reset_world()

        options = options or {}
        population = options.get("population")
        if isinstance(population, list) and population:
            self.set_population(population)
        elif not self.population:
            self.population = [AgentFactory(self.config, self.rng).create_random(1, 0)]
            self.controlled_agent = self.population[0]
        else:
            self.controlled_agent = self._first_alive_agent(self.population)

        assert self.controlled_agent is not None
        return self.observe_agent(self.controlled_agent), self._info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, GymInfo]:
        if self.controlled_agent is None:
            observation, info = self.reset()
            return observation, 0.0, False, False, info

        result = self.step_agent(
            self.controlled_agent,
            int(action),
            self.population,
        )
        self.elapsed_steps += 1
        truncated = result.truncated
        if self.elapsed_steps >= self.config.episode_steps:
            truncated = True
        return (
            result.observation,
            result.reward,
            result.terminated,
            truncated,
            self._info(result.info),
        )

    def render(self) -> RenderFrame:
        return RenderFrame(
            food=self.food.copy(),
            hazards=[(center.copy(), radius) for center, radius in self.hazards],
            agents=[agent.snapshot() for agent in self.population],
        )

    def _info(self, world_info: WorldStepInfo | None = None) -> GymInfo:
        info = world_info or WorldStepInfo()
        alive_agents = [agent for agent in self.population if agent.alive]
        return {
            "reproduced": info.reproduced,
            "partner_id": info.partner.agent_id if info.partner is not None else None,
            "alive_agents": len(alive_agents),
            "population_size": len(self.population),
            "elapsed_steps": self.elapsed_steps,
        }

    @staticmethod
    def _first_alive_agent(agents: list[AdaptiveAgent]) -> AdaptiveAgent | None:
        return next((agent for agent in agents if agent.alive), None)
