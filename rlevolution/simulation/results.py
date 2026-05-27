from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineStepResult:
    generation_complete: bool
    generation: int
    step: int


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    generation: int
    steps: int
    population_end: int
    population_total: int
    survival_rate: float
    avg_fitness: float
    best_fitness: float
    median_fitness: float
    avg_energy_alive: float
    food_eaten: int
    reproductions: int
    hazard_hits: int
    avg_speed: float
    avg_endurance: float
    avg_foraging: float
    avg_reproduction: float
    avg_q_states: float
