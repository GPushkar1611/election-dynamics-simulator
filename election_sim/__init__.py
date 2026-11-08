"""Reproducible single-constituency synthetic election simulator."""

from .agents import CandidateAgent, load_agents
from .config import SimulationConfig
from .experiments import PairedEventResult, compare_event_shock
from .simulation import RunResult, load_history, simulate

__all__ = ["NewsAgent", "SimulationConfig", "RunResult", "simulate", "load_history", "CandidateAgent",
           "load_agents", "PairedEventResult", "compare_event_shock"]

from .news import NewsAgent
