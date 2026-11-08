"""Baseline assumptions implementing plan sections 3.1, 3.2, 3.6 and 3.7.

C01 runs for 30 days; day 0 records the initial state. Defaults are scenario
assumptions, not fitted estimates. Candidate ideology compresses five issue
positions to their arithmetic mean. Popularity logits clamp at 1e-6 and
1-1e-6; utility weights and softmax temperature default to 1.

Social strength defaults to 0.2 and sender exponent to 1. Campaign strength
defaults to 0; scheduled event strength defaults to 0.2. Event memory defaults
to a 7-day half-life after its scheduled active duration; reminders default off.
Confidence is the
target's inclusive threshold; confidence/susceptibility multipliers clip to
their allowed ranges. All social updates read the same snapshot. The frozen
control disables social and event effects. Campaigns (ordinary, misleading
and smear) are optional: scripted from JSON, or chosen during the run by
candidate agents that read polls of `poll_sample_size` voters with
`poll_noise`. Smear rumours spread along social ties only when
`smear_spread_probability` > 0 (default 0: only direct recipients believe).

Turnout uses the supplied propensity by default; gamma_mob and gamma_event
optionally shift its logit by cumulative campaign/news exposure voter-days.
random_news_rate defaults to zero (no generated news). Voting
samples final candidate support using a separate RNG. Canonical ID ordering
defines random-draw assignment and argmax tie breaking. Election ties list
all tied winners. Zero turnout has no winner and zero vote shares.

Polarization means population opinion variance. These are synthetic scenarios,
not empirical validation.
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
    lambda_event: float = 0.2
    event_memory_half_life_days: float = 7.0
    event_reminder_probability: float = 0.0
    event_reminder_strength: float = 0.25
    smear_spread_probability: float = 0.0
    poll_sample_size: int = 300
    poll_noise: float = 0.02
    gamma_mob: float = 0.0
    gamma_event: float = 0.0
    random_news_rate: float = 0.0

    def __post_init__(self):
        for name in ("constituency_id", "days", "poll_sample_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("lambda_social", "confidence_multiplier", "susceptibility_multiplier",
                     "rho", "b_pop", "b_ideol", "tau", "lambda_campaign", "lambda_event",
                     "event_memory_half_life_days", "event_reminder_probability",
                     "event_reminder_strength", "smear_spread_probability", "poll_noise"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("gamma_mob", "gamma_event", "random_news_rate"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.random_news_rate < 0:
            raise ValueError("random_news_rate must be nonnegative")
        if self.lambda_social > 1 or self.lambda_campaign > 1 or self.lambda_event > 1 or self.tau == 0:
            raise ValueError("lambda_social, lambda_campaign and lambda_event must be <= 1 and tau must be > 0")
        if (self.event_reminder_probability > 1 or self.event_reminder_strength > 1
                or self.smear_spread_probability > 1):
            raise ValueError("event reminder probability/strength and smear_spread_probability must be <= 1")
        if self.global_confidence is not None:
            if not math.isfinite(self.global_confidence) or not 0 <= self.global_confidence <= 2:
                raise ValueError("global_confidence must be in [0, 2]")
        if self.vote_rule not in {"sample", "argmax"}:
            raise ValueError("vote_rule must be sample or argmax")
