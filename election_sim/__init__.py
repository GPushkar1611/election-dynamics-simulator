"""Reproducible single-constituency synthetic election baseline."""

from .config import SimulationConfig
from .simulation import RunResult, simulate

__all__ = ["SimulationConfig", "RunResult", "simulate"]
