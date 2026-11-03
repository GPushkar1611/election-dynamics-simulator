"""Baseline assumptions implementing plan sections 3.1, 3.2, 3.6 and 3.7.

C01 runs for 30 days; day 0 records the initial state. Defaults are scenario
assumptions, not fitted estimates. Candidate ideology compresses five issue
positions to their arithmetic mean. Popularity logits clamp at 1e-6 and
1-1e-6; utility weights and softmax temperature default to 1.

Social strength defaults to 0.2 and sender exponent to 1. Campaign strength
defaults to 0, so existing baseline configurations remain campaign-free.
Confidence is the
target's inclusive threshold; confidence/susceptibility multipliers clip to
their allowed ranges. All social updates read the same snapshot. The frozen
control uses social strength 0. Campaign and event effects are disabled.

Baseline turnout uses the supplied propensity without mobilization. Voting
samples final candidate support using a separate RNG. Canonical ID ordering
defines random-draw assignment and argmax tie breaking. Election ties list
all tied winners. Zero turnout has no winner and zero vote shares.

Polarization means population opinion variance. These are synthetic scenarios,
not empirical validation. Full campaign, targeting, event, experiment and UI
work remains as specified in the original implementation plan.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimulationConfig:
    constituency_id: int = 1
    days: int = 30
    lambda_social: float = 0.2
    confidence_multiplier: float = 1.0
    global_confidence: float | None = None
    susceptibility_multiplier: float = 1.0
    rho: float = 1.0
    b_pop: float = 1.0
    b_ideol: float = 1.0
    tau: float = 1.0
    vote_rule: str = "sample"
    lambda_campaign: float = 0.0

    def __post_init__(self):
        for name in ("constituency_id", "days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("lambda_social", "confidence_multiplier", "susceptibility_multiplier",
                     "rho", "b_pop", "b_ideol", "tau", "lambda_campaign"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lambda_social > 1 or self.lambda_campaign > 1 or self.tau == 0:
            raise ValueError("lambda_social and lambda_campaign must be <= 1 and tau must be > 0")
        if self.global_confidence is not None:
            if not math.isfinite(self.global_confidence) or not 0 <= self.global_confidence <= 2:
                raise ValueError("global_confidence must be in [0, 2]")
        if self.vote_rule not in {"sample", "argmax"}:
            raise ValueError("vote_rule must be sample or argmax")
