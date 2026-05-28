from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from collections.abc import Sequence

from rlevolution.config import DATA_DIR, HISTORY_CSV
from rlevolution.simulation.results import GenerationMetrics


class CsvHistoryWriter:
    def __init__(self, path: Path = HISTORY_CSV):
        self.path = path

    def write(self, history: Sequence[GenerationMetrics]) -> None:
        if not history:
            return

        DATA_DIR.mkdir(exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            records = [asdict(metric) for metric in history]
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
