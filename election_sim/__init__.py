"""Reproducible single-constituency synthetic election simulator."""

from .agents import CandidateAgent, load_agents
from .config import SimulationConfig
from .experiments import (EventFactorialResult, EventSeedSweepResult,
                          PairedEventResult, SupportSensitivityResult,
                          compare_campaign, compare_event_shock, run_batch,
                          run_event_factorial_sweep, run_event_seed_sweep,
                          run_support_sensitivity_sweep, summarize_batch)
from .simulation import RunResult, load_history, simulate
from .news import NewsAgent
from .region import RegionResult, load_region, simulate_region, spillover_sweep

__all__ = ["SimulationConfig", "RunResult", "simulate", "load_history", "CandidateAgent",
           "load_agents", "PairedEventResult", "compare_event_shock",
           "EventSeedSweepResult", "run_event_seed_sweep", "EventFactorialResult",
           "run_event_factorial_sweep", "SupportSensitivityResult",
           "run_support_sensitivity_sweep", "run_batch", "compare_campaign",
           "summarize_batch", "NewsAgent", "RegionResult", "load_region",
           "simulate_region", "spillover_sweep"]
