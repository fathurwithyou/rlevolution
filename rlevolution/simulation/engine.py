from __future__ import annotations

import numpy as np

from rlevolution.config import SimulationConfig
from rlevolution.domain.agent import AdaptiveAgent
from rlevolution.domain.factory import AgentFactory
from rlevolution.domain.fitness import FitnessCalculator
from rlevolution.domain.snapshot import AgentSnapshot
from rlevolution.environment import NaturalSelectionGymEnv
from rlevolution.environment.types import WorldStepInfo
from rlevolution.evolution import FitnessProportionateSelector, GeneticOperator
from rlevolution.persistence import CsvHistoryWriter
from rlevolution.simulation.metrics import GenerationMetricsBuilder
from rlevolution.simulation.results import EngineStepResult, GenerationMetrics


class EvolutionEngine:
    def __init__(
        self,
        config: SimulationConfig,
        history_writer: CsvHistoryWriter | None = None,
    ):
        self.config = config
        self.history_writer = history_writer or CsvHistoryWriter()
        self.fitness = FitnessCalculator()
        self.metrics = GenerationMetricsBuilder(self.fitness)
        self.rng = np.random.default_rng(config.random_seed)
        self.agent_factory = AgentFactory(config, self.rng)
        self.selector = FitnessProportionateSelector(config, self.fitness)
        self.genetics = GeneticOperator(config, self.rng)
        self.environment = NaturalSelectionGymEnv(config, self.rng)
        self.agents: list[AdaptiveAgent] = []
        self.history: list[GenerationMetrics] = []
        self.current_generation = 0
        self.step_in_generation = 0
        self.next_agent_id = 1
        self.births_this_generation = 0
        self.last_summary: GenerationMetrics | None = None
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.config.random_seed)
        self.agent_factory = AgentFactory(self.config, self.rng)
        self.selector = FitnessProportionateSelector(self.config, self.fitness)
        self.genetics = GeneticOperator(self.config, self.rng)
        self.environment = NaturalSelectionGymEnv(self.config, self.rng)
        self.current_generation = 0
        self.step_in_generation = 0
        self.next_agent_id = 1
        self.births_this_generation = 0
        self.history = []
        self.last_summary = None
        self.agents = [
            self._new_random_agent(self.current_generation)
            for _ in range(self.config.initial_agents)
        ]
        self.environment.set_population(self.agents)

    def alive_agents(self) -> list[AdaptiveAgent]:
        return [agent for agent in self.agents if agent.alive]

    def exploration_rate(self) -> float:
        decayed = self.config.exploration_rate * (
            self.config.exploration_decay ** self.current_generation
        )
        return max(self.config.min_exploration_rate, decayed)

    def step(self) -> EngineStepResult:
        if self._generation_should_finish():
            self.finish_generation()
            return EngineStepResult(True, self.current_generation, self.step_in_generation)

        epsilon = self.exploration_rate()
        for agent in list(self.agents):
            if agent.alive:
                self._train_agent(agent, epsilon)

        self.step_in_generation += 1
        if self._generation_should_finish():
            self.finish_generation()
            return EngineStepResult(True, self.current_generation, self.step_in_generation)
        return EngineStepResult(False, self.current_generation, self.step_in_generation)

    def run_generation(self) -> GenerationMetrics:
        target_generation = self.current_generation
        while self.current_generation == target_generation:
            self.step()
        assert self.last_summary is not None
        return self.last_summary

    def run_experiment(self, generations: int | None = None) -> list[GenerationMetrics]:
        target = generations or self.config.max_generations
        while len(self.history) < target:
            self.run_generation()
        self.history_writer.write(self.history)
        return self.history

    def finish_generation(self) -> None:
        summary = self.metrics.build(
            generation=self.current_generation,
            steps=self.step_in_generation,
            agents=self.agents,
            births=self.births_this_generation,
        )
        self.history.append(summary)
        self.last_summary = summary
        self.history_writer.write(self.history)
        self._create_next_generation()

    def agent_snapshots(self) -> list[AgentSnapshot]:
        return [agent.snapshot(self.fitness) for agent in self.agents]

    def _train_agent(self, agent: AdaptiveAgent, epsilon: float) -> None:
        state = self.environment.state_for(agent)
        action = agent.model.choose_action(state, epsilon, self.rng)
        result = self.environment.step_agent(
            agent,
            action,
            self.agents,
        )
        next_state = tuple(int(value) for value in result.observation)
        agent.model.update(
            state,
            action,
            result.reward,
            next_state,
            result.terminated or result.truncated,
            self.config.learning_rate,
            self.config.discount_factor,
        )
        self._handle_reproduction(agent, result.info)

    def _handle_reproduction(self, agent: AdaptiveAgent, info: WorldStepInfo) -> None:
        partner = info.partner
        if (
            info.reproduced
            and isinstance(partner, AdaptiveAgent)
            and len(self.alive_agents()) < self.config.max_population
        ):
            child = self.genetics.create_offspring(
                agent,
                partner,
                agent_id=self._take_agent_id(),
                generation=self.current_generation,
                immediate_birth=True,
            )
            self.agents.append(child)
            self.births_this_generation += 1
            self.environment.set_population(self.agents)

    def _create_next_generation(self) -> None:
        parent_pool = self.agents or [self._new_random_agent(self.current_generation)]
        next_generation = self.current_generation + 1
        next_agents = [
            self._next_generation_child(parent_pool, next_generation)
            for _ in range(self.config.initial_agents)
        ]

        self.agents = next_agents
        self.environment.reset_world()
        self.environment.set_population(self.agents)
        self.current_generation = next_generation
        self.step_in_generation = 0
        self.births_this_generation = 0

    def _next_generation_child(
        self,
        parent_pool: list[AdaptiveAgent],
        next_generation: int,
    ) -> AdaptiveAgent:
        first = self.selector.select(parent_pool, self.rng)
        second = self.selector.select(parent_pool, self.rng)
        return self.genetics.create_offspring(
            first,
            second,
            agent_id=self._take_agent_id(),
            generation=next_generation,
            immediate_birth=False,
        )

    def _new_random_agent(self, generation: int) -> AdaptiveAgent:
        return self.agent_factory.create_random(self._take_agent_id(), generation)

    def _take_agent_id(self) -> int:
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id

    def _generation_should_finish(self) -> bool:
        return (
            self.step_in_generation >= self.config.episode_steps
            or not self.alive_agents()
        )
