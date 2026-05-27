from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from rlevolution.domain.snapshot import AgentSnapshot
from rlevolution.rl import State

if TYPE_CHECKING:
    from rlevolution.domain.agent import AdaptiveAgent


@dataclass(frozen=True, slots=True)
class WorldStepInfo:
    reproduced: bool = False
    partner: AdaptiveAgent | None = None


@dataclass(frozen=True, slots=True)
class WorldStepResult:
    state: State
    reward: float
    terminated: bool
    info: WorldStepInfo


@dataclass(frozen=True, slots=True)
class GymStepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: WorldStepInfo


class GymInfo(TypedDict):
    reproduced: bool
    partner_id: int | None
    alive_agents: int
    population_size: int
    elapsed_steps: int


@dataclass(frozen=True, slots=True)
class RenderFrame:
    food: np.ndarray
    hazards: list[tuple[np.ndarray, float]]
    agents: list[AgentSnapshot]
