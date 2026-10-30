"""Integrity checks for the synthetic election CSV files."""

from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input table cannot safely be used by the loader."""


def validate_data(voters: pd.DataFrame, social_network: pd.DataFrame,
                  candidates: pd.DataFrame, events: pd.DataFrame) -> None:
    """Check IDs, references, ranges, and geography; raise with all findings."""
    problems: list[str] = []

    required = {
        "voters": (voters, {"voter_id", "constituency_id", "region_id", "initial_opinion", "susceptibility", "influence_power", "confidence_threshold", "turnout_propensity", "event_sensitivity"}),
        "social_network": (social_network, {"edge_id", "constituency_id", "region_id", "source_voter_id", "target_voter_id", "edge_weight"}),
        "candidates": (candidates, {"candidate_id", "constituency_id", "region_id", "local_popularity"}),
        "events": (events, {"event_id", "region_id", "constituency_id", "event_day", "duration", "intensity", "credibility", "candidate_id", "candidate_effect"}),
    }
    for name, (table, columns) in required.items():
        missing = columns - set(table.columns)
        if missing:
            problems.append(f"{name} is missing columns: {', '.join(sorted(missing))}")
    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))

    for name, table, col in (("voters", voters, "voter_id"), ("social_network", social_network, "edge_id"),
                             ("candidates", candidates, "candidate_id"), ("events", events, "event_id")):
        if table[col].isna().any() or table[col].duplicated().any():
            problems.append(f"{name}.{col} must be present and unique")

    voter_ids = set(voters.voter_id)
    candidate_ids = set(candidates.candidate_id)
    constituency_regions = voters.groupby("constituency_id").region_id.nunique()
    if (constituency_regions != 1).any():
        problems.append("each constituency must belong to exactly one region in voters")
    voter_region = voters.drop_duplicates("constituency_id").set_index("constituency_id").region_id

    for col, valid in (("initial_opinion", (-1, 1)), ("susceptibility", (0, 1)),
                       ("turnout_propensity", (0, 1)), ("event_sensitivity", (0, 1)),
                       ("confidence_threshold", (0, 2))):
        values = pd.to_numeric(voters[col], errors="coerce")
        if values.isna().any() or not values.between(*valid).all():
            problems.append(f"voters.{col} must be numeric and in [{valid[0]}, {valid[1]}]")
    influence = pd.to_numeric(voters.influence_power, errors="coerce")
    if influence.isna().any() or (influence <= 0).any():
        problems.append("voters.influence_power must be numeric and positive")

    if not set(social_network.source_voter_id).issubset(voter_ids) or not set(social_network.target_voter_id).issubset(voter_ids):
        problems.append("social_network contains an endpoint that does not exist in voters")
    if (social_network.source_voter_id == social_network.target_voter_id).any():
        problems.append("social_network contains a self-edge")
    if social_network.duplicated(["source_voter_id", "target_voter_id"]).any():
        problems.append("social_network contains duplicate directed edges")
    weights = pd.to_numeric(social_network.edge_weight, errors="coerce")
    if weights.isna().any() or (weights <= 0).any():
        problems.append("social_network.edge_weight must be numeric and positive")
    voter_geo = voters.set_index("voter_id")[["constituency_id", "region_id"]]
    if not social_network.empty:
        target_geo = voter_geo.reindex(social_network.target_voter_id)
        if not (target_geo.constituency_id.to_numpy() == social_network.constituency_id.to_numpy()).all() or not (target_geo.region_id.to_numpy() == social_network.region_id.to_numpy()).all():
            problems.append("social_network geography must describe each edge's target voter")

    if not candidates.empty:
        if not candidates.constituency_id.isin(voter_geo.constituency_id.unique()).all():
            problems.append("candidates references a constituency with no voters")
        regions = candidates.constituency_id.map(voter_region)
        if regions.isna().any() or not (regions.to_numpy() == candidates.region_id.to_numpy()).all():
            problems.append("candidate region_id does not match its constituency")
        popularity = pd.to_numeric(candidates.local_popularity, errors="coerce")
        if popularity.isna().any() or not popularity.between(0, 1).all():
            problems.append("candidates.local_popularity must be in [0, 1]")

    if not events.empty:
        day = pd.to_numeric(events.event_day, errors="coerce")
        duration = pd.to_numeric(events.duration, errors="coerce")
        if day.isna().any() or (day < 1).any() or duration.isna().any() or (duration < 1).any():
            problems.append("events require event_day >= 1 and duration >= 1")
        for col in ("intensity", "credibility"):
            values = pd.to_numeric(events[col], errors="coerce")
            if values.isna().any() or not values.between(0, 1).all():
                problems.append(f"events.{col} must be in [0, 1]")
        constituency_scope = events.constituency_id.notna()
        region_scope = events.region_id.notna()
        if (constituency_scope & ~region_scope).any():
            problems.append("constituency-scoped events must also include region_id")
        scoped = events[constituency_scope]
        if not scoped.empty:
            mapped = scoped.constituency_id.map(voter_region)
            if mapped.isna().any() or not (mapped.to_numpy() == scoped.region_id.to_numpy()).all():
                problems.append("event region_id does not match constituency_id")
        named = events.candidate_id.notna() & (events.candidate_id.astype(str).str.strip() != "")
        if not set(events.loc[named, "candidate_id"]).issubset(candidate_ids):
            problems.append("events references a candidate_id that does not exist")
        effects = pd.to_numeric(events.loc[events.candidate_effect.notna(), "candidate_effect"], errors="coerce")
        if effects.isna().any() or not effects.between(-1, 1).all():
            problems.append("events.candidate_effect must be in [-1, 1] when present")

    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))
