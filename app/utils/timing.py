"""Timing utilities for the Credit Risk Research Lab."""
import time
from pathlib import Path

from app.utils.io import save_json


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return False

    def to_dict(self) -> dict:
        return {"name": self.name, "elapsed_seconds": round(self.elapsed, 3)}


class TimingTracker:
    """Track timing for multiple operations."""

    def __init__(self):
        self.timings: list[dict] = []

    def add(self, name: str, elapsed: float):
        self.timings.append({"name": name, "elapsed_seconds": round(elapsed, 3)})

    def get_total(self) -> float:
        return sum(t["elapsed_seconds"] for t in self.timings)

    def to_dict(self) -> dict:
        return {
            "timings": self.timings,
            "total_seconds": round(self.get_total(), 3),
        }

    def save(self, path: str | Path):
        save_json(self.to_dict(), path)

    def summary(self) -> str:
        lines = ["Timing Report:"]
        for t in self.timings:
            lines.append(f"  {t['name']}: {t['elapsed_seconds']:.3f}s")
        lines.append(f"  TOTAL: {self.get_total():.3f}s")
        return "\n".join(lines)
