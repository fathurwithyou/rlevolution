from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ACTION_NAMES = ("up", "down", "left", "right", "stay", "reproduce")
ACTION_LABELS = {
    "up": "Atas",
    "down": "Bawah",
    "left": "Kiri",
    "right": "Kanan",
    "stay": "Diam",
    "reproduce": "Reproduksi",
}

DATA_DIR = Path("data")
HISTORY_CSV = DATA_DIR / "experiment_history.csv"


@dataclass(slots=True)
class SimulationConfig:
    world_width: int = 100
    world_height: int = 72
    initial_agents: int = 36
    max_population: int = 90
    food_count: int = 85
    hazard_count: int = 5
    hazard_radius: float = 7.5
    food_radius: float = 2.4
    episode_steps: int = 420
    max_generations: int = 24

    learning_rate: float = 0.22
    discount_factor: float = 0.92
    exploration_rate: float = 0.18
    min_exploration_rate: float = 0.04
    exploration_decay: float = 0.985

    mutation_rate: float = 0.12
    mutation_strength: float = 0.08
    selection_pressure: float = 1.35

    start_energy: float = 68.0
    max_energy: float = 120.0
    base_metabolism: float = 0.32
    food_energy: float = 28.0
    hazard_energy_penalty: float = 18.0
    max_agent_age: int = 520

    survival_reward: float = 0.08
    food_reward: float = 8.0
    hazard_reward: float = -9.0
    death_reward: float = -18.0
    reproduction_reward: float = 18.0
    reproduce_search_cost: float = 0.6
    reproduction_energy_threshold: float = 58.0
    reproduction_energy_cost: float = 20.0
    reproduction_radius: float = 6.0

    speed_min: float = 0.65
    speed_max: float = 2.25
    endurance_min: float = 0.7
    endurance_max: float = 1.9
    foraging_min: float = 0.75
    foraging_max: float = 1.85
    reproduction_min: float = 0.08
    reproduction_max: float = 0.86

    random_seed: int = 3211

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)
