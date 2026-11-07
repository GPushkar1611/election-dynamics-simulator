"""Reproducible single-constituency synthetic election baseline."""

from .config import SimulationConfig
from .experiments import PairedEventResult, compare_event_shock
from .simulation import RunResult, simulate

__all__ = ["SimulationConfig", "RunResult", "simulate", "PairedEventResult",
           "compare_event_shock"]
