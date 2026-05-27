from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    agent_id: int
    generation: int
    x: float
    y: float
    energy: float
    age: int
    alive: bool
    fitness: float
    food_eaten: int
    reproduction_count: int
    hazard_hits: int
    speed: float
    endurance: float
    foraging: float
    reproduction: float
