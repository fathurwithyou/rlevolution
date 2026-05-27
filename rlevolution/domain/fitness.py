from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlevolution.domain.agent import AdaptiveAgent


class FitnessCalculator:
    def score(self, agent: "AdaptiveAgent") -> float:
        alive_bonus = 12.0 if agent.alive else 0.0
        score = (
            agent.total_reward
            + agent.age * 0.09
            + agent.food_eaten * 9.5
            + agent.reproduction_count * 22.0
            - agent.hazard_hits * 4.0
            + alive_bonus
        )
        return max(0.1, float(score))
