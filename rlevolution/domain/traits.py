from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Traits:
    speed: float
    endurance: float
    foraging: float
    reproduction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "speed": self.speed,
            "endurance": self.endurance,
            "foraging": self.foraging,
            "reproduction": self.reproduction,
        }
