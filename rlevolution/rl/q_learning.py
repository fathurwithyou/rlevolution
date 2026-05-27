from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rlevolution.config import ACTION_NAMES


State = tuple[int, ...]


@dataclass
class QLearningModel:
    action_count: int = len(ACTION_NAMES)
    q_table: dict[State, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.q_table = {} if self.q_table is None else {
            tuple(state): np.array(values, dtype=float).copy()
            for state, values in self.q_table.items()
        }

    def values(self, state: State) -> np.ndarray:
        assert self.q_table is not None
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count, dtype=float)
        return self.q_table[state]

    def choose_action(self, state: State, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.action_count))

        values = self.values(state)
        best_value = float(values.max())
        best_actions = np.flatnonzero(np.isclose(values, best_value))
        return int(rng.choice(best_actions))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
        learning_rate: float,
        discount_factor: float,
    ) -> None:
        current = self.values(state)
        next_best = 0.0 if done else float(self.values(next_state).max())
        target = reward + discount_factor * next_best
        current[action] = current[action] + learning_rate * (target - current[action])

    def copy(self) -> "QLearningModel":
        assert self.q_table is not None
        return QLearningModel(
            action_count=self.action_count,
            q_table={state: values.copy() for state, values in self.q_table.items()},
        )

    def mutated(
        self,
        mutation_rate: float,
        mutation_strength: float,
        rng: np.random.Generator,
    ) -> "QLearningModel":
        child = self.copy()
        assert child.q_table is not None
        for values in child.q_table.values():
            mask = rng.random(values.shape) < mutation_rate
            noise = rng.normal(0.0, mutation_strength, size=values.shape)
            values += mask * noise
        return child

    def q_state_count(self) -> int:
        return len(self.q_table or {})

    @classmethod
    def crossover(
        cls,
        first: "QLearningModel",
        second: "QLearningModel",
        rng: np.random.Generator,
    ) -> "QLearningModel":
        assert first.q_table is not None
        assert second.q_table is not None
        states: Iterable[State] = set(first.q_table.keys()) | set(second.q_table.keys())
        table: dict[State, np.ndarray] = {}

        for state in states:
            if state in first.q_table and state in second.q_table:
                values_a = first.q_table[state]
                values_b = second.q_table[state]
                mask = rng.random(first.action_count) < 0.5
                table[state] = np.where(mask, values_a, values_b).astype(float)
            elif state in first.q_table:
                table[state] = first.q_table[state].copy()
            else:
                table[state] = second.q_table[state].copy()

        return cls(action_count=first.action_count, q_table=table)
