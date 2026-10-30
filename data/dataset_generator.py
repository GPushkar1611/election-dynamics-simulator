#!/usr/bin/env python3
"""
Synthetic election dataset generator (Multi-Agent Systems project).

Generates a fictional state with regions, constituencies, synthetic voters,
a directed social network, candidates and political events, and writes four
CSV files:

    voters.csv, social_network.csv, candidates.csv, political_events.csv

Generation is hierarchical (state -> region -> constituency -> voter), fully
reproducible from a seed, and validated before anything is written.
All values are synthetic model inputs, not measurements of real people.

Usage:
    python dataset_generator.py
    python dataset_generator.py --voters-per-constituency 5000 --seed 123
    python dataset_generator.py --help
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION  (edit here, or override from the command line)
# =============================================================================

# State configuration
NUM_REGIONS = 3
CONSTITUENCIES_PER_REGION = 7
# Optional: total number of constituencies. If None, it is
# NUM_REGIONS * CONSTITUENCIES_PER_REGION. If set, constituencies are split as
# evenly as possible across regions (the total need not be divisible).
NUM_CONSTITUENCIES = None

# Population configuration
VOTERS_PER_CONSTITUENCY = 3000

# Candidate configuration
NUM_CANDIDATES_PER_CONSTITUENCY = 3
NUM_POLITICAL_PARTIES = 3
# Probability that a constituency has a sitting (incumbent) candidate.
INCUMBENT_PROBABILITY = 0.7

# Election configuration
ELECTION_DURATION_DAYS = 30
EVENTS_PER_CONSTITUENCY = 12      # constituency-level events, per constituency
EVENTS_PER_REGION = 6             # region-level events, per region
STATE_LEVEL_EVENTS = 8            # state-level events, in total

# Social network configuration
AVG_CONNECTIONS_PER_VOTER = 8     # average in-degree == average out-degree
AVG_COMMUNITY_SIZE = 150          # target voters per social community
CROSS_COMMUNITY_CONNECTION_PROBABILITY = 0.08
CROSS_CONSTITUENCY_CONNECTION_PROBABILITY = 0.01
# Sources are picked proportionally to influence_power ** this exponent, so
# influential voters end up with more outgoing edges.
INFLUENCE_DEGREE_EXPONENT = 1.5

# Voter behavioural feature bounds
INFLUENCE_POWER_MAX = 5.0         # influence_power lies in (0, INFLUENCE_POWER_MAX]

# Reproducibility and output
RANDOM_SEED = 42
OUTPUT_DIR = "synthetic_election_dataset"

# =============================================================================
# MODEL CONSTANTS (distribution assumptions; rarely need changing)
# =============================================================================

ISSUES = ["employment", "economy", "education", "healthcare", "agriculture"]
ISSUE_COLS = [f"issue_{i}" for i in ISSUES]
POSITION_COLS = [f"position_{i}" for i in ISSUES]

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-59", "60+"]
AGE_BASE_PROBS = np.array([0.14, 0.22, 0.22, 0.26, 0.16])
GENDERS = ["Male", "Female", "Other"]
GENDER_PROBS = np.array([0.495, 0.50, 0.005])
EDUCATION_LEVELS = ["No formal", "Primary", "Secondary", "Graduate", "Postgraduate"]
EDU_CUTPOINTS = [-1.0, -0.2, 0.7, 1.5]          # latent education score -> level
WEALTH_LEVELS = ["Low", "Middle", "High"]
WEALTH_CUTPOINTS = [-0.3, 1.0]                  # latent wealth score -> level
URBAN_RURAL = ["Urban", "Rural"]

OCCUPATIONS = [
    "Farmer", "Agricultural labourer", "Unskilled labourer", "Skilled worker",
    "Private employee", "Government employee", "Business owner", "Professional",
    "Student", "Homemaker", "Unemployed", "Retired",
]
OCC_INDEX = {name: i for i, name in enumerate(OCCUPATIONS)}
# Base occupation weights and (log-scale) effects of demographics. Missing = 0.
OCC_BASE_WEIGHT = {
    "Farmer": 0.10, "Agricultural labourer": 0.08, "Unskilled labourer": 0.12,
    "Skilled worker": 0.10, "Private employee": 0.14, "Government employee": 0.06,
    "Business owner": 0.08, "Professional": 0.05, "Student": 0.06,
    "Homemaker": 0.10, "Unemployed": 0.07, "Retired": 0.04,
}
OCC_RURAL_COEF = {
    "Farmer": 2.0, "Agricultural labourer": 2.0, "Unskilled labourer": 0.3,
    "Skilled worker": -0.2, "Private employee": -1.2, "Government employee": -0.4,
    "Business owner": -0.3, "Professional": -1.2, "Student": -0.3, "Homemaker": 0.1,
}
OCC_EDU_COEF = {
    "Farmer": -0.4, "Agricultural labourer": -0.9, "Unskilled labourer": -0.8,
    "Skilled worker": -0.1, "Private employee": 0.4, "Government employee": 0.8,
    "Business owner": 0.1, "Professional": 1.2, "Student": 0.5, "Homemaker": -0.1,
}
OCC_WEALTH_COEF = {
    "Farmer": 0.3, "Agricultural labourer": -0.8, "Unskilled labourer": -0.7,
    "Private employee": 0.2, "Government employee": 0.4, "Business owner": 0.9,
    "Professional": 0.8, "Homemaker": 0.1, "Unemployed": -0.4, "Retired": 0.2,
}
# Additive shift to log(influence_power) by occupation (social standing).
OCC_INFLUENCE_BONUS = {
    "Government employee": 0.30, "Professional": 0.35, "Business owner": 0.30,
    "Farmer": 0.10, "Student": -0.10, "Unemployed": -0.10,
}

# Constituency political archetypes and how often they occur.
ARCHETYPE_WEIGHTS = {"strong": 0.30, "moderate": 0.30, "competitive": 0.25, "polarized": 0.15}
PARTY_PREFERENCE_SHARPNESS = 3.0   # how strongly a region's lean selects the favoured party
PARTY_IDEOLOGY_RANGE = 0.7         # parties are spread over [-range, +range]
PARTY_ISSUE_SPREAD = 0.25          # party-specific deviations per issue
CANDIDATE_POSITION_NOISE = 0.12    # candidate deviation from party positions
INDEPENDENT_POSITION_SD = 0.45

PARTY_NAME_POOL = [
    "Progressive Alliance", "National Unity Party", "People's Front",
    "Democratic Reform Party", "Green Development League", "Social Justice Forum",
    "Heritage Congress", "Workers' Union Party",
]
INDEPENDENT_ID = "IND"
INDEPENDENT_NAME = "Independent"

# Social network
RELATIONSHIP_TYPES = ["Family", "Friends", "Colleagues", "Community", "Other"]
# Scope codes: 0 = same community, 1 = other community (same constituency),
# 2 = neighbouring constituency.
RELATIONSHIP_PROBS_BY_SCOPE = {
    0: [0.20, 0.30, 0.20, 0.25, 0.05],
    1: [0.05, 0.25, 0.30, 0.10, 0.30],
    2: [0.15, 0.30, 0.20, 0.05, 0.30],
}
RELATIONSHIP_WEIGHT_MEAN = {"Family": 0.75, "Friends": 0.60, "Colleagues": 0.45,
                            "Community": 0.35, "Other": 0.20}
MIN_COMMUNITY_SIZE = 3
DEGREE_DISPERSION = 3.0            # negative-binomial r (smaller = more skewed degrees)
MAX_EDGE_ROUNDS = 8                # re-draw rounds used to top up duplicate/self draws

# Political events
EVENT_SOURCES = ["Candidate", "News", "External"]
EVENT_TYPES = [
    "Manifesto announcement", "Campaign rally", "Policy announcement", "Controversy",
    "Economic development", "Employment announcement", "Agricultural issue",
    "Healthcare issue", "Education initiative", "External economic shock", "Other",
]
TARGET_GROUPS = ["General", "Rural", "Urban", "Students", "Farmers", "Unemployed",
                 "Low-income", "Middle-income", "High-income"]
EVENT_ISSUES = ISSUES + ["general"]
EVENT_LEVELS = ["constituency", "region", "state"]

EVENT_SOURCE_WEIGHTS = {
    "constituency": {"Candidate": 0.55, "News": 0.35, "External": 0.10},
    "region": {"Candidate": 0.25, "News": 0.45, "External": 0.30},
    "state": {"News": 0.60, "External": 0.40},
}
NEWS_CANDIDATE_PROB = {"constituency": 0.60, "region": 0.30, "state": 0.0}
EVENT_TYPE_WEIGHTS = {
    "Candidate": {"Manifesto announcement": 0.20, "Campaign rally": 0.35,
                  "Policy announcement": 0.20, "Employment announcement": 0.08,
                  "Education initiative": 0.07, "Controversy": 0.05, "Other": 0.05},
    "News": {"Controversy": 0.25, "Economic development": 0.15,
             "Employment announcement": 0.08, "Agricultural issue": 0.12,
             "Healthcare issue": 0.12, "Policy announcement": 0.10,
             "Education initiative": 0.05, "Other": 0.13},
    "External": {"External economic shock": 0.35, "Economic development": 0.20,
                 "Agricultural issue": 0.15, "Healthcare issue": 0.15, "Other": 0.15},
}
# Per event type: mean intensity, mean duration (days), mean and SD of the
# (signed) candidate_effect before scaling by intensity/credibility.
EVENT_PARAMS = {
    "Manifesto announcement": dict(intensity=0.55, duration=3, effect=0.25, effect_sd=0.30),
    "Campaign rally": dict(intensity=0.60, duration=1, effect=0.20, effect_sd=0.30),
    "Policy announcement": dict(intensity=0.50, duration=4, effect=0.15, effect_sd=0.35),
    "Controversy": dict(intensity=0.60, duration=5, effect=-0.40, effect_sd=0.40),
    "Economic development": dict(intensity=0.50, duration=7, effect=0.10, effect_sd=0.35),
    "Employment announcement": dict(intensity=0.50, duration=5, effect=0.20, effect_sd=0.30),
    "Agricultural issue": dict(intensity=0.50, duration=6, effect=-0.05, effect_sd=0.40),
    "Healthcare issue": dict(intensity=0.50, duration=6, effect=-0.05, effect_sd=0.40),
    "Education initiative": dict(intensity=0.45, duration=5, effect=0.15, effect_sd=0.30),
    "External economic shock": dict(intensity=0.70, duration=10, effect=-0.15, effect_sd=0.45),
    "Other": dict(intensity=0.35, duration=2, effect=0.0, effect_sd=0.25),
}
CREDIBILITY_MEAN_BY_SOURCE = {"Candidate": 0.55, "News": 0.70, "External": 0.80}
# Issue chosen for event types that do not imply one. "salience" = weighted by
# the local average issue importance of voters.
EVENT_FIXED_ISSUE = {
    "Employment announcement": "employment", "Agricultural issue": "agriculture",
    "Healthcare issue": "healthcare", "Education initiative": "education",
}
EVENT_ISSUE_CHOICES = {  # type -> {issue: weight}; weights None => salience-based
    "Economic development": {"economy": 0.5, "employment": 0.3, "agriculture": 0.2},
    "External economic shock": {"economy": 0.5, "employment": 0.25, "agriculture": 0.25},
}
EVENT_GENERAL_PROB = {"Campaign rally": 0.25, "Controversy": 0.70, "Other": 0.60}
TARGET_GROUP_WEIGHTS = {
    "employment": {"General": 0.20, "Unemployed": 0.35, "Students": 0.20,
                   "Low-income": 0.15, "Urban": 0.05, "Rural": 0.05},
    "economy": {"General": 0.30, "Middle-income": 0.20, "High-income": 0.15,
                "Low-income": 0.10, "Urban": 0.15, "Rural": 0.10},
    "education": {"General": 0.25, "Students": 0.50, "Low-income": 0.10,
                  "Rural": 0.10, "Urban": 0.05},
    "healthcare": {"General": 0.40, "Low-income": 0.25, "Rural": 0.15,
                   "Urban": 0.10, "Middle-income": 0.10},
    "agriculture": {"Farmers": 0.55, "Rural": 0.30, "General": 0.10, "Low-income": 0.05},
    "general": {"General": 0.70, "Urban": 0.07, "Rural": 0.07, "Middle-income": 0.06,
                "Low-income": 0.05, "High-income": 0.05},
}
ISSUE_PHRASE = {
    "employment": "jobs and employment", "economy": "the economy",
    "education": "education", "healthcare": "healthcare",
    "agriculture": "agriculture and farmers", "general": "general local concerns",
}
EVENT_DESCRIPTIONS = {
    "Manifesto announcement": ["Manifesto announced in {place} with a focus on {issue}.",
                               "Campaign promises released for {place}, emphasising {issue}."],
    "Campaign rally": ["Campaign rally held in {place} centred on {issue}.",
                       "Well-attended rally in {place} addressing {issue}."],
    "Policy announcement": ["Policy announcement for {place} concerning {issue}.",
                            "New policy proposal on {issue} announced in {place}."],
    "Controversy": ["Controversy reported in {place} touching on {issue}.",
                    "Public row in {place} drew media attention ({issue})."],
    "Economic development": ["Economic development news in {place} relating to {issue}.",
                             "New development project reported in {place} ({issue})."],
    "Employment announcement": ["Employment drive announced in {place}.",
                                "Job-creation announcement made in {place}."],
    "Agricultural issue": ["Agricultural issue raised in {place} affecting farmers.",
                           "Farm-sector concern reported in {place}."],
    "Healthcare issue": ["Healthcare issue reported in {place}.",
                         "Public health concern raised in {place}."],
    "Education initiative": ["Education initiative announced in {place}.",
                             "New schooling and education plan unveiled in {place}."],
    "External economic shock": ["External economic shock affecting {place} ({issue}).",
                                "Sudden economic disruption reaching {place} ({issue})."],
    "Other": ["Miscellaneous political development in {place} ({issue}).",
              "Minor political news in {place} ({issue})."],
}
SOURCE_ACTOR_VERB = {"Candidate": "Led by", "News": "Concerns", "External": "Affects"}


# =============================================================================
# Configuration object, validation of configuration, CLI
# =============================================================================

@dataclass
class Config:
    """All generator parameters. Defaults come from the constants above."""
    num_regions: int = NUM_REGIONS
    constituencies_per_region: int = CONSTITUENCIES_PER_REGION
    num_constituencies: int | None = NUM_CONSTITUENCIES
    voters_per_constituency: int = VOTERS_PER_CONSTITUENCY
    num_candidates_per_constituency: int = NUM_CANDIDATES_PER_CONSTITUENCY
    num_political_parties: int = NUM_POLITICAL_PARTIES
    incumbent_probability: float = INCUMBENT_PROBABILITY
    election_duration_days: int = ELECTION_DURATION_DAYS
    events_per_constituency: int = EVENTS_PER_CONSTITUENCY
    events_per_region: int = EVENTS_PER_REGION
    state_level_events: int = STATE_LEVEL_EVENTS
    avg_connections_per_voter: float = AVG_CONNECTIONS_PER_VOTER
    avg_community_size: int = AVG_COMMUNITY_SIZE
    cross_community_connection_probability: float = CROSS_COMMUNITY_CONNECTION_PROBABILITY
    cross_constituency_connection_probability: float = CROSS_CONSTITUENCY_CONNECTION_PROBABILITY
    influence_degree_exponent: float = INFLUENCE_DEGREE_EXPONENT
    influence_power_max: float = INFLUENCE_POWER_MAX
    seed: int = RANDOM_SEED
    output_dir: str = OUTPUT_DIR

    # ---- derived quantities -------------------------------------------------
    @property
    def total_constituencies(self) -> int:
        if self.num_constituencies is not None:
            return self.num_constituencies
        return self.num_regions * self.constituencies_per_region

    @property
    def total_voters(self) -> int:
        return self.total_constituencies * self.voters_per_constituency

    @property
    def total_candidates(self) -> int:
        return self.total_constituencies * self.num_candidates_per_constituency

    @property
    def expected_edges(self) -> int:
        return int(round(self.total_voters * self.avg_connections_per_voter))

    @property
    def total_events(self) -> int:
        return (self.total_constituencies * self.events_per_constituency
                + self.num_regions * self.events_per_region + self.state_level_events)

    def validate(self) -> None:
        """Raise ValueError for impossible configurations."""
        def need(cond, msg):
            if not cond:
                raise ValueError(f"Invalid configuration: {msg}")
        need(self.num_regions >= 1, "num_regions must be >= 1")
        need(self.total_constituencies >= self.num_regions,
             "need at least one constituency per region")
        need(self.voters_per_constituency >= 2, "voters_per_constituency must be >= 2")
        need(self.num_candidates_per_constituency >= 1, "need >= 1 candidate per constituency")
        need(self.num_political_parties >= 1, "need >= 1 political party")
        need(0 <= self.incumbent_probability <= 1, "incumbent_probability must be in [0, 1]")
        need(self.election_duration_days >= 1, "election_duration_days must be >= 1")
        need(min(self.events_per_constituency, self.events_per_region,
                 self.state_level_events) >= 0, "event counts must be >= 0")
        need(self.avg_connections_per_voter >= 0, "avg_connections_per_voter must be >= 0")
        need(self.avg_community_size >= 2, "avg_community_size must be >= 2")
        p1 = self.cross_community_connection_probability
        p2 = self.cross_constituency_connection_probability
        need(0 <= p1 <= 1 and 0 <= p2 <= 1 and p1 + p2 <= 1,
             "cross-connection probabilities must be in [0, 1] and sum to <= 1")
        need(self.influence_degree_exponent >= 0, "influence_degree_exponent must be >= 0")
        need(self.influence_power_max > 0.05, "influence_power_max must be > 0.05")


def parse_args(argv=None) -> Config:
    """Build a Config from defaults plus optional command-line overrides."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic election dataset (four CSV files).")
    flags = {
        "num_regions": ("--num-regions", int),
        "constituencies_per_region": ("--constituencies-per-region", int),
        "num_constituencies": ("--num-constituencies", int),
        "voters_per_constituency": ("--voters-per-constituency", int),
        "num_candidates_per_constituency": ("--candidates-per-constituency", int),
        "num_political_parties": ("--num-parties", int),
        "incumbent_probability": ("--incumbent-probability", float),
        "election_duration_days": ("--election-days", int),
        "events_per_constituency": ("--events-per-constituency", int),
        "events_per_region": ("--events-per-region", int),
        "state_level_events": ("--state-events", int),
        "avg_connections_per_voter": ("--avg-connections", float),
        "avg_community_size": ("--avg-community-size", int),
        "cross_community_connection_probability": ("--cross-community-prob", float),
        "cross_constituency_connection_probability": ("--cross-constituency-prob", float),
        "influence_degree_exponent": ("--influence-degree-exponent", float),
        "influence_power_max": ("--influence-power-max", float),
        "seed": ("--seed", int),
        "output_dir": ("--output-dir", str),
    }
    defaults = Config()
    for name, (flag, typ) in flags.items():
        parser.add_argument(flag, dest=name, type=typ, default=None,
                            help=f"(default: {getattr(defaults, name)})")
    ns = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(ns).items() if v is not None}
    cfg = replace(defaults, **overrides)
    # An explicit --constituencies-per-region without --num-constituencies must
    # win over a NUM_CONSTITUENCIES constant set in the file.
    if "constituencies_per_region" in overrides and "num_constituencies" not in overrides:
        cfg = replace(cfg, num_constituencies=None)
    try:
        cfg.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return cfg


# =============================================================================
# Small numeric helpers
# =============================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _beta_from_mean(rng, mean, kappa, size=None):
    """Beta draw with a given mean and concentration kappa (bigger = tighter)."""
    mean = np.clip(mean, 0.02, 0.98)
    return rng.beta(mean * kappa, (1 - mean) * kappa, size=size)


def _sample_from_probs(rng, probs):
    """Row-wise categorical sampling from an (n, k) matrix of unnormalised probs."""
    p = probs / probs.sum(axis=1, keepdims=True)
    cum = p.cumsum(axis=1)
    u = rng.random(len(p))[:, None]
    return np.minimum((cum < u).sum(axis=1), probs.shape[1] - 1)


def _sample_from_logits(rng, logits):
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return _sample_from_probs(rng, z)


def _coef_array(mapping, names):
    return np.array([mapping.get(n, 0.0) for n in names], dtype=float)


def _weighted_key(rng, mapping):
    """Pick a key from {key: weight} (insertion order => deterministic)."""
    keys = list(mapping.keys())
    w = np.array([mapping[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


# =============================================================================
# Hierarchical profiles: state -> region -> constituency
# =============================================================================

@dataclass
class StateProfile:
    urban_share: float
    edu_mean: float
    wealth_mean: float
    issue_salience: np.ndarray


@dataclass
class RegionProfile:
    region_id: int
    urban_share: float
    edu_mean: float
    wealth_mean: float
    issue_salience: np.ndarray
    political_lean: float        # tendency on the [-1, 1] opinion axis
    agri_shift: float
    turnout_shift: float
    rural_effect: float          # demographic effects on opinion (region-specific)
    wealth_effect: float
    edu_effect: float
    youth_effect: float


@dataclass
class ConstituencyProfile:
    constituency_id: int
    region_id: int
    urban_share: float
    edu_mean: float
    wealth_mean: float
    issue_salience: np.ndarray
    turnout_shift: float
    youth_shift: float
    agri_shift: float
    archetype: str
    cluster_weights: np.ndarray
    cluster_centers: np.ndarray
    cluster_stds: np.ndarray


def build_state_profile(rng) -> StateProfile:
    """State-level baseline characteristics."""
    base_salience = np.array([0.58, 0.60, 0.52, 0.55, 0.45]) + rng.normal(0, 0.03, 5)
    return StateProfile(
        urban_share=float(rng.beta(4, 6)),
        edu_mean=float(rng.normal(0, 0.15)),
        wealth_mean=float(rng.normal(0, 0.15)),
        issue_salience=np.clip(base_salience, 0.2, 0.85),
    )


def build_region_profiles(cfg: Config, state: StateProfile, rng) -> list[RegionProfile]:
    """Region-level profiles that deviate from the state baseline."""
    regions = []
    for r in range(1, cfg.num_regions + 1):
        urban = float(_sigmoid(_logit(state.urban_share) + rng.normal(0, 0.6)))
        salience = state.issue_salience + rng.normal(0, 0.06, 5)
        salience[ISSUES.index("agriculture")] += 0.2 * (1 - urban)  # rural regions care more
        regions.append(RegionProfile(
            region_id=r,
            urban_share=urban,
            edu_mean=float(state.edu_mean + rng.normal(0, 0.35)),
            wealth_mean=float(state.wealth_mean + rng.normal(0, 0.35)),
            issue_salience=np.clip(salience, 0.2, 0.85),
            political_lean=float(np.clip(rng.normal(0, 0.35), -0.7, 0.7)),
            agri_shift=float(rng.normal(0, 0.4)),
            turnout_shift=float(rng.normal(0, 0.05)),
            rural_effect=float(rng.normal(0, 0.12)),
            wealth_effect=float(rng.normal(0, 0.10)),
            edu_effect=float(rng.normal(0, 0.10)),
            youth_effect=float(rng.normal(0, 0.10)),
        ))
    return regions


def build_party_landscape(cfg: Config, rng) -> list[dict]:
    """Statewide parties with fixed ideology and issue positions."""
    p = cfg.num_political_parties
    if p == 1:
        ideologies = np.array([0.5])
    else:
        ideologies = np.linspace(-PARTY_IDEOLOGY_RANGE, PARTY_IDEOLOGY_RANGE, p)
    ideologies = np.clip(ideologies + rng.normal(0, 0.05, p), -0.9, 0.9)
    parties = []
    for i in range(p):
        name = PARTY_NAME_POOL[i] if i < len(PARTY_NAME_POOL) else f"Party {i + 1}"
        positions = np.clip(ideologies[i] + rng.normal(0, PARTY_ISSUE_SPREAD, len(ISSUES)), -1, 1)
        parties.append({"party_id": f"P{i + 1}", "party_name": name,
                        "ideology": float(ideologies[i]), "positions": positions})
    return parties


def _assign_regions(cfg: Config) -> list[int]:
    """Region id of each constituency (index 0 -> constituency 1), as even as possible."""
    total, regions = cfg.total_constituencies, cfg.num_regions
    base, extra = divmod(total, regions)
    out = []
    for r in range(regions):
        out += [r + 1] * (base + (1 if r < extra else 0))
    return out


def _build_opinion_clusters(archetype, region_lean, ideologies, rng):
    """Return (weights, centers, stds) of the opinion mixture for a constituency."""
    p = len(ideologies)
    pref = np.exp(-PARTY_PREFERENCE_SHARPNESS * np.abs(ideologies - region_lean))
    main = int(rng.choice(p, p=pref / pref.sum()))
    if archetype == "strong":
        others = [i for i in range(p) if i != main] or [main]
        other = int(rng.choice(others))
        centers = [ideologies[main] * rng.uniform(0.75, 1.0), ideologies[other] * 0.8]
        weights, stds = [0.85, 0.15], [rng.uniform(0.15, 0.25), 0.25]
    elif archetype == "moderate":
        centers = [0.55 * ideologies[main] + rng.normal(0, 0.05)]
        weights, stds = [1.0], [rng.uniform(0.25, 0.35)]
    elif archetype == "competitive":
        centers = [0.3 * region_lean + rng.normal(0, 0.1)]
        weights, stds = [1.0], [rng.uniform(0.35, 0.50)]
    else:  # polarized: two opinion clusters
        if p >= 2:
            pair = rng.choice(p, size=2, replace=False)
            centers = list(ideologies[pair] * rng.uniform(0.7, 1.0, size=2))
        else:
            centers = [0.6, -0.6]
        w = rng.uniform(0.4, 0.6)
        weights, stds = [w, 1 - w], [0.18, 0.18]
    return (np.array(weights, float), np.clip(np.array(centers, float), -1, 1),
            np.array(stds, float))


def build_constituency_profiles(cfg: Config, regions: list[RegionProfile],
                                parties: list[dict], rng) -> list[ConstituencyProfile]:
    """Constituency-level profiles: similar to their region but not identical."""
    ideologies = np.array([p["ideology"] for p in parties])
    archetypes = list(ARCHETYPE_WEIGHTS.keys())
    arch_p = np.array([ARCHETYPE_WEIGHTS[a] for a in archetypes])
    arch_p = arch_p / arch_p.sum()
    region_of = _assign_regions(cfg)
    out = []
    for c, r in enumerate(region_of, start=1):
        reg = regions[r - 1]
        archetype = archetypes[int(rng.choice(len(archetypes), p=arch_p))]
        weights, centers, stds = _build_opinion_clusters(
            archetype, reg.political_lean, ideologies, rng)
        out.append(ConstituencyProfile(
            constituency_id=c, region_id=r,
            urban_share=float(_sigmoid(_logit(reg.urban_share) + rng.normal(0, 0.5))),
            edu_mean=float(reg.edu_mean + rng.normal(0, 0.15)),
            wealth_mean=float(reg.wealth_mean + rng.normal(0, 0.15)),
            issue_salience=np.clip(reg.issue_salience + rng.normal(0, 0.04, 5), 0.15, 0.9),
            turnout_shift=float(reg.turnout_shift + rng.normal(0, 0.04)),
            youth_shift=float(rng.normal(0, 0.3)),
            agri_shift=float(reg.agri_shift + rng.normal(0, 0.25)),
            archetype=archetype, cluster_weights=weights,
            cluster_centers=centers, cluster_stds=stds,
        ))
    return out


# =============================================================================
# Voters
# =============================================================================

def _generate_constituency_voters(cfg: Config, cons: ConstituencyProfile,
                                  reg: RegionProfile, rng, cw: int, vw: int) -> pd.DataFrame:
    """Generate all voters of one constituency (vectorised)."""
    n = cfg.voters_per_constituency
    urban = (rng.random(n) < cons.urban_share).astype(float)
    rural = 1.0 - urban
    age_centered = np.arange(len(AGE_GROUPS)) - 2

    # --- demographics ---------------------------------------------------------
    youth_c = cons.youth_shift + 0.3 * urban            # urban areas skew younger
    age_logits = np.log(AGE_BASE_PROBS)[None, :] - 0.5 * youth_c[:, None] * age_centered[None, :]
    age_idx = _sample_from_logits(rng, age_logits)
    gender_idx = rng.choice(len(GENDERS), size=n, p=GENDER_PROBS)
    female = (gender_idx == 1).astype(float)

    edu_score = (cons.edu_mean + 0.6 * urban - 0.2 * (age_idx - 1)
                 - 0.15 * female * rural + rng.normal(0, 0.9, n))
    edu_idx = np.digitize(edu_score, EDU_CUTPOINTS)
    wealth_score = (cons.wealth_mean + 0.35 * (edu_idx - 2) + 0.3 * urban
                    + rng.normal(0, 0.8, n))
    wealth_idx = np.digitize(wealth_score, WEALTH_CUTPOINTS)

    # --- occupation (probabilistic, driven by urbanity, education, wealth, life stage)
    k = len(OCCUPATIONS)
    base_log = np.log(_coef_array(OCC_BASE_WEIGHT, OCCUPATIONS))
    logits = (base_log[None, :]
              + rural[:, None] * _coef_array(OCC_RURAL_COEF, OCCUPATIONS)[None, :]
              + (edu_idx - 2)[:, None] * _coef_array(OCC_EDU_COEF, OCCUPATIONS)[None, :]
              + (wealth_idx - 1)[:, None] * _coef_array(OCC_WEALTH_COEF, OCCUPATIONS)[None, :])
    for name in ("Farmer", "Agricultural labourer"):
        logits[:, OCC_INDEX[name]] += cons.agri_shift
    logits[:, OCC_INDEX["Student"]] += np.where(age_idx == 0, 2.5, np.where(age_idx == 1, 0.0, -4.0))
    logits[:, OCC_INDEX["Retired"]] += np.where(age_idx == 4, 3.5, -5.0)
    logits[:, OCC_INDEX["Unemployed"]] += np.where(age_idx <= 1, 0.6, -0.2)
    logits[:, OCC_INDEX["Homemaker"]] += np.where(female == 1, 1.5, -2.0)
    for name in ("Government employee", "Professional"):
        logits[:, OCC_INDEX[name]] += np.where(age_idx == 0, -1.0, 0.0)
    occ_idx = _sample_from_logits(rng, logits)

    def is_occ(*names):
        return np.isin(occ_idx, [OCC_INDEX[x] for x in names]).astype(float)

    # --- issue importance (Beta around demographic- and area-driven means) -----
    low_wealth = (wealth_idx == 0).astype(float)
    high_wealth = (wealth_idx == 2).astype(float)
    adj = np.zeros((n, len(ISSUES)))
    adj[:, 0] = 0.20 * is_occ("Unemployed", "Student") + 0.08 * low_wealth + 0.05 * (age_idx <= 1)
    adj[:, 1] = 0.12 * is_occ("Business owner", "Professional", "Private employee") + 0.08 * high_wealth
    adj[:, 2] = 0.20 * is_occ("Student") + 0.08 * ((age_idx == 1) | (age_idx == 2)) + 0.10 * (edu_idx >= 3)
    adj[:, 3] = 0.10 * (age_idx >= 3) + 0.06 * low_wealth + 0.05 * female
    adj[:, 4] = 0.35 * is_occ("Farmer", "Agricultural labourer") + 0.08 * rural - 0.10 * urban
    issue_mean = cons.issue_salience[None, :] + adj + rng.normal(0, 0.08, (n, len(ISSUES)))
    issues = _beta_from_mean(rng, np.clip(issue_mean, 0.05, 0.95), 8.0)

    # --- political opinion: mixture around constituency leaning + demographic shifts
    comp = rng.choice(len(cons.cluster_weights), size=n, p=cons.cluster_weights)
    shift = (reg.rural_effect * (rural - urban) + reg.wealth_effect * (wealth_idx - 1)
             + reg.edu_effect * (edu_idx - 2) / 2 + reg.youth_effect * (age_idx - 2) / 2)
    opinion = np.clip(cons.cluster_centers[comp]
                      + cons.cluster_stds[comp] * rng.normal(size=n) + shift, -1, 1)

    # --- behavioural parameters ----------------------------------------------
    susceptibility = _beta_from_mean(
        rng, 0.42 - 0.04 * (edu_idx - 2) + 0.02 * (age_idx - 2) + rng.normal(0, 0.05, n), 6.0)
    event_sensitivity = _beta_from_mean(
        rng, 0.45 + 0.06 * urban - 0.03 * (age_idx - 2) + 0.03 * (edu_idx - 2)
        + rng.normal(0, 0.05, n), 6.0)
    turnout = _beta_from_mean(
        rng, 0.62 + 0.05 * (age_idx - 2) + 0.03 * (edu_idx - 2) - 0.03 * urban
        + cons.turnout_shift + rng.normal(0, 0.04, n), 6.0)
    conf_mean = np.clip(1.0 - 0.35 * np.abs(opinion) + 0.08 * (edu_idx - 2)
                        + rng.normal(0, 0.08, n), 0.1, 1.9)
    confidence = 2.0 * _beta_from_mean(rng, conf_mean / 2.0, 5.0)
    occ_bonus = _coef_array(OCC_INFLUENCE_BONUS, OCCUPATIONS)[occ_idx]
    log_power = (np.log(0.8) + 0.10 * (edu_idx - 2) + 0.10 * (wealth_idx - 1)
                 + occ_bonus + rng.normal(0, 0.6, n))
    influence = np.clip(np.exp(log_power), 0.05, cfg.influence_power_max)

    voter_ids = [f"C{cons.constituency_id:0{cw}d}_V{i:0{vw}d}" for i in range(1, n + 1)]
    df = pd.DataFrame({
        "voter_id": voter_ids,
        "constituency_id": cons.constituency_id,
        "region_id": cons.region_id,
        "age_group": np.array(AGE_GROUPS)[age_idx],
        "gender": np.array(GENDERS)[gender_idx],
        "education_level": np.array(EDUCATION_LEVELS)[edu_idx],
        "wealth_level": np.array(WEALTH_LEVELS)[wealth_idx],
        "occupation": np.array(OCCUPATIONS)[occ_idx],
        "urban_rural": np.where(urban == 1, "Urban", "Rural"),
        "initial_opinion": np.round(opinion, 4),
        "event_sensitivity": np.round(event_sensitivity, 4),
        "turnout_propensity": np.round(turnout, 4),
        "susceptibility": np.round(susceptibility, 4),
        "influence_power": np.round(influence, 4),
        "confidence_threshold": np.round(confidence, 4),
    })
    for j, col in enumerate(ISSUE_COLS):
        df[col] = np.round(issues[:, j], 4)
    return df


def generate_voters(cfg: Config, constituencies, regions, rng) -> pd.DataFrame:
    """Generate all voters, constituency by constituency."""
    cw = max(2, len(str(cfg.total_constituencies)))
    vw = max(3, len(str(cfg.voters_per_constituency)))
    frames = [_generate_constituency_voters(cfg, c, regions[c.region_id - 1], rng, cw, vw)
              for c in constituencies]
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Social network
# =============================================================================

def _assign_communities(cfg: Config, urban: np.ndarray, rng) -> np.ndarray:
    """Split a constituency's voters into communities of unequal size."""
    n = len(urban)
    k = max(1, int(round(n / cfg.avg_community_size)))
    if k == 1:
        return np.zeros(n, dtype=int)
    size_w = rng.dirichlet(np.full(k, 4.0))
    affinity = rng.beta(2, 2, size=k)                  # some communities lean urban, some rural
    factor = np.where(urban[:, None], 0.4 + affinity[None, :], 1.4 - affinity[None, :])
    comm = _sample_from_probs(rng, size_w[None, :] * factor)
    counts = np.bincount(comm, minlength=k)
    tiny = np.flatnonzero(counts < MIN_COMMUNITY_SIZE)
    comm[np.isin(comm, tiny)] = counts.argmax()        # fold tiny communities into the largest
    _, comm = np.unique(comm, return_inverse=True)
    return comm.astype(int)


def _draw_in_degrees(cfg: Config, n: int, rng) -> np.ndarray:
    """Heterogeneous in-degrees (>= 1 when the mean is >= 1) with the configured mean."""
    mu = cfg.avg_connections_per_voter
    if mu <= 0:
        return np.zeros(n, dtype=np.int64)
    if mu < 1:
        d = rng.binomial(1, mu, n).astype(np.int64)
    else:
        d = np.ones(n, dtype=np.int64)
        extra = mu - 1.0
        if extra > 0:
            r = DEGREE_DISPERSION
            d += rng.negative_binomial(r, r / (r + extra), n)
    return np.minimum(d, n - 1)


def generate_social_network(cfg: Config, voters: pd.DataFrame, rng) -> pd.DataFrame:
    """
    Directed, community-structured network. An edge source -> target means the
    source voter influences the target voter. Each target draws its in-degree;
    each incoming edge's source is drawn from the same community (default),
    another community of the constituency, or a neighbouring constituency
    (id +/- 1), weighted by the source's influence_power.
    """
    n_total = len(voters)
    n_cons = cfg.total_constituencies
    cons_ids = voters["constituency_id"].to_numpy()
    region_ids = voters["region_id"].to_numpy()
    voter_ids = voters["voter_id"].to_numpy()
    urban_all = voters["urban_rural"].to_numpy() == "Urban"
    weights = voters["influence_power"].to_numpy() ** cfg.influence_degree_exponent
    bounds = np.searchsorted(cons_ids, np.arange(1, n_cons + 2))   # voters sorted by constituency

    src_parts, tgt_parts, scope_parts = [], [], []
    for c in range(1, n_cons + 1):
        s, e = int(bounds[c - 1]), int(bounds[c])
        n = e - s
        comm = _assign_communities(cfg, urban_all[s:e], rng)
        k = int(comm.max()) + 1
        members = [s + np.flatnonzero(comm == j) for j in range(k)]
        within_p = [weights[m] / weights[m].sum() for m in members]
        nonmembers = [s + np.flatnonzero(comm != j) for j in range(k)] if k > 1 else None
        nonmember_p = ([weights[m] / weights[m].sum() for m in nonmembers] if k > 1 else None)
        neighbours = [j for j in (c - 1, c + 1) if 1 <= j <= n_cons]
        nb_ids = [np.arange(bounds[j - 1], bounds[j]) for j in neighbours]
        nb_p = [weights[a] / weights[a].sum() for a in nb_ids]
        p_cons = cfg.cross_constituency_connection_probability if neighbours else 0.0
        p_comm = cfg.cross_community_connection_probability if k > 1 else 0.0

        deficit = _draw_in_degrees(cfg, n, rng)
        existing = np.empty(0, dtype=np.int64)
        c_src, c_tgt, c_scope = [], [], []
        for _ in range(MAX_EDGE_ROUNDS):
            tgt = np.repeat(np.arange(s, e), deficit)
            if len(tgt) == 0:
                break
            m = len(tgt)
            tgt_comm = comm[tgt - s]
            u = rng.random(m)
            scope = np.where(u < p_cons, 2, np.where(u < p_cons + p_comm, 1, 0)).astype(np.int8)
            src = np.empty(m, dtype=np.int64)
            for j in range(k):
                mask = (scope == 0) & (tgt_comm == j)
                if mask.any():
                    src[mask] = rng.choice(members[j], size=int(mask.sum()), p=within_p[j])
                if k > 1:
                    mask = (scope == 1) & (tgt_comm == j)
                    if mask.any():
                        src[mask] = rng.choice(nonmembers[j], size=int(mask.sum()), p=nonmember_p[j])
            cross = np.flatnonzero(scope == 2)
            if len(cross):
                which = rng.integers(0, len(neighbours), size=len(cross))
                for j in range(len(neighbours)):
                    sel = cross[which == j]
                    if len(sel):
                        src[sel] = rng.choice(nb_ids[j], size=len(sel), p=nb_p[j])
            keep = src != tgt                                   # no self-edges
            src, tgt, scope = src[keep], tgt[keep], scope[keep]
            key = src * n_total + tgt
            key_u, first = np.unique(key, return_index=True)    # no duplicates within round
            fresh = ~np.isin(key_u, existing)                   # ... nor across rounds
            first = first[fresh]
            c_src.append(src[first]); c_tgt.append(tgt[first]); c_scope.append(scope[first])
            existing = np.concatenate([existing, key_u[fresh]])
            deficit = deficit - np.bincount(tgt[first] - s, minlength=n)
            deficit = np.maximum(deficit, 0)
        if c_src:
            src, tgt, scope = (np.concatenate(c_src), np.concatenate(c_tgt),
                               np.concatenate(c_scope))
            order = np.lexsort((src, tgt))
            src_parts.append(src[order]); tgt_parts.append(tgt[order]); scope_parts.append(scope[order])

    if src_parts:
        src = np.concatenate(src_parts)
        tgt = np.concatenate(tgt_parts)
        scope = np.concatenate(scope_parts)
    else:
        src = tgt = np.empty(0, dtype=np.int64)
        scope = np.empty(0, dtype=np.int8)
    m = len(src)

    # Relationship type depends on scope; edge weight depends on relationship type.
    rel_idx = np.zeros(m, dtype=int)
    for sc, probs in RELATIONSHIP_PROBS_BY_SCOPE.items():
        mask = scope == sc
        if mask.any():
            rel_idx[mask] = rng.choice(len(RELATIONSHIP_TYPES), size=int(mask.sum()), p=probs)
    mean_w = np.array([RELATIONSHIP_WEIGHT_MEAN[r] for r in RELATIONSHIP_TYPES])[rel_idx]
    edge_w = np.clip(_beta_from_mean(rng, mean_w, 6.0), 0.01, 1.0) if m else np.empty(0)

    id_width = max(7, len(str(m)))
    return pd.DataFrame({
        "edge_id": [f"E{i:0{id_width}d}" for i in range(1, m + 1)],
        "constituency_id": cons_ids[tgt],            # constituency of the target voter
        "region_id": region_ids[tgt],
        "source_voter_id": voter_ids[src],
        "target_voter_id": voter_ids[tgt],
        "edge_weight": np.round(edge_w, 4),
        "relationship_type": np.array(RELATIONSHIP_TYPES)[rel_idx] if m else [],
    })


# =============================================================================
# Candidates
# =============================================================================

def _party_support(voters: pd.DataFrame, parties: list[dict], n_cons: int) -> np.ndarray:
    """Share of each constituency's voters whose opinion is closest to each party."""
    ideo = np.array([p["ideology"] for p in parties])
    nearest = np.abs(voters["initial_opinion"].to_numpy()[:, None] - ideo[None, :]).argmin(axis=1)
    cons = voters["constituency_id"].to_numpy() - 1
    counts = np.bincount(cons * len(parties) + nearest,
                         minlength=n_cons * len(parties)).reshape(n_cons, len(parties))
    return counts / counts.sum(axis=1, keepdims=True)


def generate_candidates(cfg: Config, constituencies, parties, voters, rng) -> pd.DataFrame:
    """Candidates per constituency; popularity reflects local opinion distribution."""
    support = _party_support(voters, parties, len(constituencies))
    n_par = len(parties)
    n_cand = cfg.num_candidates_per_constituency
    rows = []
    for cons in constituencies:
        sup = support[cons.constituency_id - 1]
        if n_cand <= n_par:   # a subset of the parties contests, favouring locally strong ones
            idx = rng.choice(n_par, size=n_cand, replace=False, p=(sup + 0.05) / (sup + 0.05).sum())
            chosen = sorted(int(i) for i in idx)
        else:                 # every party contests, remaining slots are independents
            chosen = list(range(n_par))
        entries = [{"party": parties[i], "support": float(sup[i])} for i in chosen]
        entries += [{"party": None, "support": 0.0} for _ in range(n_cand - len(chosen))]

        # at most one incumbent, drawn among party candidates by local strength
        incumbent = -1
        party_slots = [i for i, e in enumerate(entries) if e["party"] is not None]
        if party_slots and rng.random() < cfg.incumbent_probability:
            w = np.array([entries[i]["support"] + 0.05 for i in party_slots])
            incumbent = party_slots[int(rng.choice(len(party_slots), p=w / w.sum()))]

        max_sup = max([e["support"] for e in entries] + [1e-9])
        for i, e in enumerate(entries):
            party = e["party"]
            is_inc = i == incumbent
            if is_inc:
                experience = int(rng.choice([2, 3, 4, 5], p=[0.15, 0.30, 0.30, 0.25]))
            else:
                experience = int(rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.25, 0.20, 0.18, 0.12]))
            if party is not None:
                positions = np.clip(party["positions"]
                                    + rng.normal(0, CANDIDATE_POSITION_NOISE, len(ISSUES)), -1, 1)
                rel = e["support"] / max_sup
            else:
                positions = np.clip(rng.normal(0, INDEPENDENT_POSITION_SD, len(ISSUES)), -1, 1)
                rel = rng.uniform(0.1, 0.3)
            pop_mean = (0.15 + 0.5 * rel + 0.12 * is_inc + 0.03 * (experience - 1)
                        + rng.normal(0, 0.04))
            popularity = float(_beta_from_mean(rng, pop_mean, 12.0))
            row = {
                "constituency_id": cons.constituency_id, "region_id": cons.region_id,
                "party_id": party["party_id"] if party else INDEPENDENT_ID,
                "party_name": party["party_name"] if party else INDEPENDENT_NAME,
                "is_incumbent": bool(is_inc), "experience_level": experience,
                "local_popularity": round(popularity, 4),
            }
            for col, val in zip(POSITION_COLS, positions):
                row[col] = round(float(val), 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    width = max(4, len(str(len(df))))
    df.insert(0, "candidate_id", [f"CAND_{i:0{width}d}" for i in range(1, len(df) + 1)])
    return df


# =============================================================================
# Political events
# =============================================================================

def _tilted_target_weights(issue: str, urban_share: float) -> dict:
    w = dict(TARGET_GROUP_WEIGHTS[issue])
    if "Urban" in w:
        w["Urban"] *= 0.5 + urban_share
    if "Rural" in w:
        w["Rural"] *= 1.5 - urban_share
    return w


def _build_event(cfg: Config, rng, level: str, region_id, cons_id,
                 pool: pd.DataFrame | None, salience: np.ndarray, urban_share: float) -> dict:
    """Create one event record for the given geographic level."""
    days = cfg.election_duration_days
    source = _weighted_key(rng, EVENT_SOURCE_WEIGHTS[level])
    event_type = _weighted_key(rng, EVENT_TYPE_WEIGHTS[source])
    params = EVENT_PARAMS[event_type]

    # candidate association (optional)
    candidate_id = None
    if pool is not None and len(pool):
        associate = source == "Candidate" or (
            source == "News" and rng.random() < NEWS_CANDIDATE_PROB[level])
        if associate:
            w = pool["local_popularity"].to_numpy() + 0.1
            pick = int(rng.choice(len(pool), p=w / w.sum()))
            candidate_id = pool["candidate_id"].iloc[pick]
            party_name = pool["party_name"].iloc[pick]

    # issue
    if event_type in EVENT_FIXED_ISSUE:
        issue = EVENT_FIXED_ISSUE[event_type]
    elif event_type in EVENT_ISSUE_CHOICES:
        issue = _weighted_key(rng, EVENT_ISSUE_CHOICES[event_type])
    elif rng.random() < EVENT_GENERAL_PROB.get(event_type, 0.0):
        issue = "general"
    else:
        issue = ISSUES[int(rng.choice(len(ISSUES), p=salience / salience.sum()))]

    # timing: candidate campaigns intensify towards polling day
    if source == "Candidate":
        day = min(days, 1 + int(rng.beta(1.6, 1.2) * days))
    else:
        day = int(rng.integers(1, days + 1))
    duration = int(max(1, round(rng.lognormal(np.log(params["duration"]), 0.5))))
    duration = min(duration, days - day + 1)

    intensity = float(_beta_from_mean(rng, params["intensity"], 5.0))
    cred_mean = CREDIBILITY_MEAN_BY_SOURCE[source] - (0.10 if event_type == "Controversy" else 0.0)
    credibility = float(_beta_from_mean(rng, cred_mean, 6.0))
    target = _weighted_key(rng, _tilted_target_weights(issue, urban_share))

    effect = np.nan
    if candidate_id is not None:
        raw = rng.normal(params["effect"], params["effect_sd"])
        effect = float(np.clip(raw * (0.5 + 0.5 * intensity) * (0.5 + 0.5 * credibility), -1, 1))

    place = {"constituency": f"Constituency {cons_id}", "region": f"Region {region_id}",
             "state": "the state"}[level]
    text = EVENT_DESCRIPTIONS[event_type][int(rng.integers(len(EVENT_DESCRIPTIONS[event_type])))]
    description = text.format(place=place, issue=ISSUE_PHRASE[issue])
    if candidate_id is not None:
        actor = (f"{party_name} candidate {candidate_id}" if party_name != INDEPENDENT_NAME
                 else f"Independent candidate {candidate_id}")
        description += f" {SOURCE_ACTOR_VERB[source]} {actor}."

    return {
        "region_id": region_id, "constituency_id": cons_id, "event_day": day,
        "source_type": source, "event_type": event_type, "issue": issue,
        "candidate_id": candidate_id, "intensity": round(intensity, 4),
        "credibility": round(credibility, 4), "duration": duration,
        "target_group": target,
        "candidate_effect": round(effect, 4) if candidate_id is not None else np.nan,
        "description": description,
        "_rank": {"state": 0, "region": 1, "constituency": 2}[level],
    }


def generate_events(cfg: Config, constituencies, candidates: pd.DataFrame,
                    voters: pd.DataFrame, rng) -> pd.DataFrame:
    """Constituency-, region- and state-level events in one table."""
    cons_sal = voters.groupby("constituency_id")[ISSUE_COLS].mean()
    reg_sal = voters.groupby("region_id")[ISSUE_COLS].mean()
    is_urban = voters["urban_rural"] == "Urban"
    cons_urban = is_urban.groupby(voters["constituency_id"]).mean()
    reg_urban = is_urban.groupby(voters["region_id"]).mean()
    cand_by_cons = {c: g for c, g in candidates.groupby("constituency_id")}
    cand_by_reg = {r: g for r, g in candidates.groupby("region_id")}

    rows = []
    for cons in constituencies:
        c = cons.constituency_id
        for _ in range(cfg.events_per_constituency):
            rows.append(_build_event(cfg, rng, "constituency", cons.region_id, c,
                                     cand_by_cons[c], cons_sal.loc[c].to_numpy(),
                                     float(cons_urban.loc[c])))
    for r in range(1, cfg.num_regions + 1):
        for _ in range(cfg.events_per_region):
            rows.append(_build_event(cfg, rng, "region", r, None, cand_by_reg[r],
                                     reg_sal.loc[r].to_numpy(), float(reg_urban.loc[r])))
    for _ in range(cfg.state_level_events):
        rows.append(_build_event(cfg, rng, "state", None, None, None,
                                 voters[ISSUE_COLS].mean().to_numpy(), float(is_urban.mean())))

    columns = ["event_id", "region_id", "constituency_id", "event_day", "source_type",
               "event_type", "issue", "candidate_id", "intensity", "credibility", "duration",
               "target_group", "candidate_effect", "description"]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    df["region_id"] = df["region_id"].astype("Int64")
    df["constituency_id"] = df["constituency_id"].astype("Int64")
    df = (df.assign(_r=df["region_id"].fillna(0), _c=df["constituency_id"].fillna(0))
            .sort_values(["event_day", "_rank", "_r", "_c"], kind="stable")
            .reset_index(drop=True))
    width = max(5, len(str(len(df))))
    df.insert(0, "event_id", [f"EVT_{i:0{width}d}" for i in range(1, len(df) + 1)])
    return df[columns]


# =============================================================================
# Validation
# =============================================================================

class ValidationError(Exception):
    """Raised when the generated dataset violates a consistency rule."""


def _check(cond, message: str) -> None:
    if not bool(cond):
        raise ValidationError(message)


def _check_range(df, col, lo, hi, name, lo_open=False):
    s = df[col]
    _check(s.notna().all(), f"{name}.{col} contains missing values")
    bad = (s <= lo) if lo_open else (s < lo)
    _check(not (bad | (s > hi)).any(), f"{name}.{col} outside range "
           f"{'(' if lo_open else '['}{lo}, {hi}]")


def validate_dataset(cfg: Config, voters, network, candidates, events) -> None:
    """Raise ValidationError if any consistency rule is violated."""
    cons_to_region = dict(zip(range(1, cfg.total_constituencies + 1), _assign_regions(cfg)))

    # ---- voters -------------------------------------------------------------
    _check(not voters.isna().any().any(), "voters.csv has missing values")
    _check(voters["voter_id"].is_unique, "voter_id values are not unique")
    _check(len(voters) == cfg.total_voters, "voter count does not match configuration")
    _check(voters["constituency_id"].isin(cons_to_region.keys()).all(),
           "voters reference an invalid constituency_id")
    _check((voters["constituency_id"].map(cons_to_region) == voters["region_id"]).all(),
           "voter region_id inconsistent with its constituency")
    _check((voters.groupby("constituency_id").size() == cfg.voters_per_constituency).all(),
           "a constituency has the wrong number of voters")
    for col, allowed in [("age_group", AGE_GROUPS), ("gender", GENDERS),
                         ("education_level", EDUCATION_LEVELS), ("wealth_level", WEALTH_LEVELS),
                         ("occupation", OCCUPATIONS), ("urban_rural", URBAN_RURAL)]:
        _check(voters[col].isin(allowed).all(), f"voters.{col} has an invalid category")
    _check_range(voters, "initial_opinion", -1, 1, "voters")
    for col in ["susceptibility", "event_sensitivity", "turnout_propensity"] + ISSUE_COLS:
        _check_range(voters, col, 0, 1, "voters")
    _check_range(voters, "confidence_threshold", 0, 2, "voters")
    _check_range(voters, "influence_power", 0, cfg.influence_power_max, "voters", lo_open=True)

    # ---- social network -------------------------------------------------------
    _check(not network.isna().any().any(), "social_network.csv has missing values")
    _check(network["edge_id"].is_unique, "edge_id values are not unique")
    voter_cons = voters.set_index("voter_id")["constituency_id"]
    _check(network["source_voter_id"].isin(voter_cons.index).all(), "edge source voter does not exist")
    _check(network["target_voter_id"].isin(voter_cons.index).all(), "edge target voter does not exist")
    _check((network["source_voter_id"] != network["target_voter_id"]).all(), "self-edge found")
    _check(not network.duplicated(["source_voter_id", "target_voter_id"]).any(),
           "duplicate directed edge found")
    _check((network["edge_weight"] > 0).all() and np.isfinite(network["edge_weight"]).all(),
           "invalid edge_weight (must be positive and finite)")
    _check(network["relationship_type"].isin(RELATIONSHIP_TYPES).all(), "invalid relationship_type")
    src_c = network["source_voter_id"].map(voter_cons)
    tgt_c = network["target_voter_id"].map(voter_cons)
    _check((network["constituency_id"] == tgt_c).all(),
           "edge constituency_id must equal the target voter's constituency")
    _check((network["region_id"] == tgt_c.map(cons_to_region)).all(), "edge region_id inconsistent")
    cross = src_c != tgt_c
    _check(((src_c - tgt_c).abs()[cross] == 1).all(),
           "cross-constituency edge between non-neighbouring constituencies")

    # ---- candidates -----------------------------------------------------------
    _check(not candidates.isna().any().any(), "candidates.csv has missing values")
    _check(candidates["candidate_id"].is_unique, "candidate_id values are not unique")
    _check(candidates["constituency_id"].isin(cons_to_region.keys()).all(),
           "candidate references invalid constituency")
    _check((candidates["constituency_id"].map(cons_to_region) == candidates["region_id"]).all(),
           "candidate region_id inconsistent")
    valid_parties = {f"P{i}" for i in range(1, cfg.num_political_parties + 1)} | {INDEPENDENT_ID}
    _check(candidates["party_id"].isin(valid_parties).all(), "invalid party_id")
    _check((candidates.groupby("constituency_id").size()
            == cfg.num_candidates_per_constituency).all()
           and candidates["constituency_id"].nunique() == cfg.total_constituencies,
           "candidate count per constituency does not match configuration")
    _check(len(candidates) == cfg.total_candidates, "total candidate count mismatch")
    named = candidates[candidates["party_id"] != INDEPENDENT_ID]
    _check(not named.duplicated(["constituency_id", "party_id"]).any(),
           "a party fields two candidates in one constituency")
    _check((candidates.groupby("constituency_id")["is_incumbent"].sum() <= 1).all(),
           "more than one incumbent in a constituency")
    _check_range(candidates, "local_popularity", 0, 1, "candidates")
    for col in POSITION_COLS:
        _check_range(candidates, col, -1, 1, "candidates")
    _check(candidates["experience_level"].between(1, 5).all(), "experience_level outside 1-5")

    # ---- events ---------------------------------------------------------------
    _check(events["event_id"].is_unique, "event_id values are not unique")
    _check(len(events) == cfg.total_events, "event count does not match configuration")
    if len(events):
        _check(events["event_day"].between(1, cfg.election_duration_days).all(),
               "event_day outside the election period")
        _check(events["source_type"].isin(EVENT_SOURCES).all(), "invalid source_type")
        _check(events["event_type"].isin(EVENT_TYPES).all(), "invalid event_type")
        _check(events["issue"].isin(EVENT_ISSUES).all(), "invalid issue")
        _check(events["target_group"].isin(TARGET_GROUPS).all(), "invalid target_group")
        _check_range(events, "intensity", 0, 1, "events")
        _check_range(events, "credibility", 0, 1, "events")
        _check((events["duration"] >= 1).all()
               and ((events["event_day"] + events["duration"] - 1)
                    <= cfg.election_duration_days).all(), "invalid event duration")
        has_cand = events["candidate_id"].notna()
        _check(events.loc[has_cand, "candidate_id"].isin(candidates["candidate_id"]).all(),
               "event references unknown candidate")
        _check(events.loc[has_cand, "candidate_effect"].between(-1, 1).all()
               and events.loc[~has_cand, "candidate_effect"].isna().all(),
               "candidate_effect inconsistent with candidate_id")
        region_na, cons_na = events["region_id"].isna(), events["constituency_id"].isna()
        _check((region_na | events["region_id"].isin(range(1, cfg.num_regions + 1))).all(),
               "event references invalid region")
        _check((cons_na | events["constituency_id"].isin(cons_to_region.keys())).all(),
               "event references invalid constituency")
        _check(not (region_na & ~cons_na).any(),
               "event has a constituency_id but no region_id")
        both = ~region_na & ~cons_na
        _check((events.loc[both, "constituency_id"].map(cons_to_region)
                == events.loc[both, "region_id"]).all(), "event region/constituency mismatch")
        cand_cons = candidates.set_index("candidate_id")["constituency_id"]
        cand_reg = candidates.set_index("candidate_id")["region_id"]
        m = has_cand & both
        _check((events.loc[m, "candidate_id"].map(cand_cons) == events.loc[m, "constituency_id"]).all(),
               "constituency-level event refers to a candidate from another constituency")
        m = has_cand & ~region_na & cons_na
        _check((events.loc[m, "candidate_id"].map(cand_reg) == events.loc[m, "region_id"]).all(),
               "region-level event refers to a candidate from another region")
        _check(not (has_cand & region_na).any(), "state-level event must not name a candidate")


# =============================================================================
# Orchestration, output, summary
# =============================================================================

def generate_dataset(cfg: Config) -> dict[str, pd.DataFrame]:
    """Run the full hierarchy and return the four tables (validated)."""
    cfg.validate()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(cfg.seed).spawn(6)]
    rng_profile, rng_party, rng_voter, rng_net, rng_cand, rng_event = streams

    state = build_state_profile(rng_profile)
    regions = build_region_profiles(cfg, state, rng_profile)
    parties = build_party_landscape(cfg, rng_party)
    constituencies = build_constituency_profiles(cfg, regions, parties, rng_profile)

    voters = generate_voters(cfg, constituencies, regions, rng_voter)
    network = generate_social_network(cfg, voters, rng_net)
    candidates = generate_candidates(cfg, constituencies, parties, voters, rng_cand)
    events = generate_events(cfg, constituencies, candidates, voters, rng_event)

    validate_dataset(cfg, voters, network, candidates, events)
    return {"voters": voters, "social_network": network,
            "candidates": candidates, "political_events": events}


def write_dataset(cfg: Config, tables: dict[str, pd.DataFrame]) -> Path:
    """Write the four CSV files (4 decimal places for floats)."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
    return out


def print_summary(cfg: Config, tables: dict[str, pd.DataFrame], out: Path) -> None:
    print("Dataset generation completed.\n")
    print(f"Regions: {cfg.num_regions}")
    print(f"Constituencies: {cfg.total_constituencies}")
    print(f"Total voters: {len(tables['voters']):,}")
    print(f"Total candidates: {len(tables['candidates']):,}")
    print(f"Total social connections: {len(tables['social_network']):,} "
          f"(expected approx. {cfg.expected_edges:,})")
    print(f"Total political events: {len(tables['political_events']):,}")
    print(f"\nOutput directory: {out.resolve()}")
    print("Output files:")
    for name in ["voters", "social_network", "candidates", "political_events"]:
        print(f"- {name}.csv")


def main(argv=None) -> int:
    cfg = parse_args(argv)
    try:
        tables = generate_dataset(cfg)
    except ValidationError as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1
    out = write_dataset(cfg, tables)
    print_summary(cfg, tables, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
