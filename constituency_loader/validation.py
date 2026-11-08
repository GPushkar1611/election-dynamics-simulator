"""Integrity checks for the synthetic election CSV files."""

from __future__ import annotations

import pandas as pd
import numpy as np


ISSUES = ("employment", "economy", "education", "healthcare", "agriculture")


class DataValidationError(ValueError):
    """Raised when an input table cannot safely be used by the loader."""


def validate_data(voters: pd.DataFrame, social_network: pd.DataFrame,
                  candidates: pd.DataFrame, events: pd.DataFrame,
                  election_duration_days: int = 30) -> None:
    """Check IDs, references, ranges, and geography; raise with all findings."""
    problems: list[str] = []
    if isinstance(election_duration_days, bool) or not isinstance(election_duration_days, int) or election_duration_days < 1:
        raise ValueError("election_duration_days must be a positive integer")

    required = {
        "voters": (voters, {"voter_id", "constituency_id", "region_id", "initial_opinion", "susceptibility", "influence_power", "confidence_threshold", "turnout_propensity", "event_sensitivity"}),
        "social_network": (social_network, {"edge_id", "constituency_id", "region_id", "source_voter_id", "target_voter_id", "edge_weight"}),
        "candidates": (candidates, {"candidate_id", "constituency_id", "region_id", "local_popularity", *[f"position_{issue}" for issue in ISSUES]}),
        "events": (events, {"event_id", "region_id", "constituency_id", "event_day", "duration", "intensity", "credibility", "candidate_id", "candidate_effect", "source_type", "issue", "target_group"}),
    }
    for name, (table, columns) in required.items():
        missing = columns - set(table.columns)
        if missing:
            problems.append(f"{name} is missing columns: {', '.join(sorted(missing))}")
    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))

    for name, table, col in (("voters", voters, "voter_id"), ("social_network", social_network, "edge_id"),
                             ("candidates", candidates, "candidate_id"), ("events", events, "event_id")):
        if table[col].isna().any() or table[col].astype(str).str.strip().eq("").any() or table[col].duplicated().any():
            problems.append(f"{name}.{col} must be present and unique")

    for name, table in (("voters", voters), ("candidates", candidates), ("social_network", social_network), ("events", events)):
        for column in ("constituency_id", "region_id"):
            values = pd.to_numeric(table[column], errors="coerce")
            present = table[column].notna() if name == "events" else pd.Series(True, index=table.index)
            valid = (np.isfinite(values) & values.gt(0) & values.mod(1).eq(0)).fillna(False)
            if not valid.loc[present].all():
                problems.append(f"{name}.{column} must contain positive integer IDs")

    category_rules = {
        "age_group": {"18-24", "25-34", "35-44", "45-59", "60+"},
        "gender": {"Male", "Female", "Other"},
        "education_level": {"No formal", "Primary", "Secondary", "Graduate", "Postgraduate"},
        "wealth_level": {"Low", "Middle", "High"},
        "urban_rural": {"Urban", "Rural"},
        "occupation": {"Farmer", "Agricultural labourer", "Unskilled labourer", "Skilled worker",
                       "Private employee", "Government employee", "Business owner", "Professional",
                       "Student", "Homemaker", "Unemployed", "Retired"},
    }
    for column, allowed in category_rules.items():
        if column in voters and not voters[column].isin(allowed).all():
            problems.append(f"voters.{column} contains an unknown or missing category")
    for issue in ISSUES:
        column = f"issue_{issue}"
        if column in voters:
            values = pd.to_numeric(voters[column], errors="coerce")
            if values.isna().any() or not values.between(0, 1).all():
                problems.append(f"voters.{column} must be in [0, 1]")
        values = pd.to_numeric(candidates[f"position_{issue}"], errors="coerce")
        if values.isna().any() or not values.between(-1, 1).all():
            problems.append(f"candidates.position_{issue} must be in [-1, 1]")

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
    if not np.isfinite(influence).all() or (influence <= 0).any():
        problems.append("voters.influence_power must be finite and positive")

    if not set(social_network.source_voter_id).issubset(voter_ids) or not set(social_network.target_voter_id).issubset(voter_ids):
        problems.append("social_network contains an endpoint that does not exist in voters")
    if (social_network.source_voter_id == social_network.target_voter_id).any():
        problems.append("social_network contains a self-edge")
    if social_network.duplicated(["source_voter_id", "target_voter_id"]).any():
        problems.append("social_network contains duplicate directed edges")
    weights = pd.to_numeric(social_network.edge_weight, errors="coerce")
    if not np.isfinite(weights).all() or not ((weights > 0) & (weights <= 1)).all():
        problems.append("social_network.edge_weight must be finite and in (0, 1]")
    if "relationship_type" in social_network:
        if not social_network.relationship_type.isin({"Family", "Friends", "Colleagues", "Community", "Other"}).all():
            problems.append("social_network.relationship_type contains an unknown or missing category")
    voter_geo = voters.drop_duplicates("voter_id").set_index("voter_id")[["constituency_id", "region_id"]]
    if not social_network.empty:
        target_geo = voter_geo.reindex(social_network.target_voter_id)
        if not (target_geo.constituency_id.to_numpy() == social_network.constituency_id.to_numpy()).all() or not (target_geo.region_id.to_numpy() == social_network.region_id.to_numpy()).all():
            problems.append("social_network geography must describe each edge's target voter")

    if not candidates.empty:
        if not set(voters.constituency_id).issubset(set(candidates.constituency_id)):
            problems.append("each voter constituency must have at least one candidate")
        if not candidates.constituency_id.isin(voter_geo.constituency_id.unique()).all():
            problems.append("candidates references a constituency with no voters")
        regions = candidates.constituency_id.map(voter_region)
        if regions.isna().any() or not (regions.to_numpy() == candidates.region_id.to_numpy()).all():
            problems.append("candidate region_id does not match its constituency")
        popularity = pd.to_numeric(candidates.local_popularity, errors="coerce")
        if popularity.isna().any() or not popularity.between(0, 1).all():
            problems.append("candidates.local_popularity must be in [0, 1]")
        if "experience_level" in candidates:
            experience = pd.to_numeric(candidates.experience_level, errors="coerce")
            if experience.isna().any() or not experience.between(1, 5).all() or not experience.mod(1).eq(0).all():
                problems.append("candidates.experience_level must be an integer in [1, 5]")
        if "is_incumbent" in candidates:
            if not candidates.is_incumbent.isin([True, False]).all():
                problems.append("candidates.is_incumbent must be Boolean")
            elif candidates.groupby("constituency_id").is_incumbent.sum().gt(1).any():
                problems.append("at most one incumbent is allowed per constituency")
    elif not voters.empty:
        problems.append("each voter constituency must have at least one candidate")

    if not events.empty:
        day = pd.to_numeric(events.event_day, errors="coerce")
        duration = pd.to_numeric(events.duration, errors="coerce")
        if (not np.isfinite(day).all() or not day.between(1, election_duration_days).all()
                or not day.mod(1).eq(0).all()):
            problems.append(f"events.event_day must be an integer in [1, {election_duration_days}]")
        if not np.isfinite(duration).all() or (duration < 1).any() or not duration.mod(1).eq(0).all():
            problems.append("events.duration must be a positive finite integer")
        for column, allowed in (("source_type", {"Candidate", "News", "External"}),
                                ("issue", {*ISSUES, "general"}),
                                ("target_group", {"General", "Rural", "Urban", "Students", "Farmers", "Unemployed", "Low-income", "Middle-income", "High-income"})):
            if not events[column].isin(allowed).all():
                problems.append(f"events.{column} contains an unknown or missing category")
        if "event_type" in events:
            allowed_types = {"Manifesto announcement", "Campaign rally", "Policy announcement", "Controversy",
                             "Economic development", "Employment announcement", "Agricultural issue",
                             "Healthcare issue", "Education initiative", "External economic shock", "Other", "fact_check", "poll_release", "random_news"}
            if not events.event_type.isin(allowed_types).all():
                problems.append("events.event_type contains an unknown or missing category")
        for col in ("intensity", "credibility"):
            values = pd.to_numeric(events[col], errors="coerce")
            if values.isna().any() or not values.between(0, 1).all():
                problems.append(f"events.{col} must be in [0, 1]")
        constituency_scope = events.constituency_id.notna()
        region_scope = events.region_id.notna()
        if (constituency_scope & ~region_scope).any():
            problems.append("constituency-scoped events must also include region_id")
        if not events.loc[region_scope, "region_id"].isin(voters.region_id).all():
            problems.append("event region_id does not exist in voters")
        scoped = events[constituency_scope]
        if not scoped.empty:
            mapped = scoped.constituency_id.map(voter_region)
            if mapped.isna().any() or not (mapped.to_numpy() == scoped.region_id.to_numpy()).all():
                problems.append("event region_id does not match constituency_id")
        named = events.candidate_id.notna() & (events.candidate_id.astype(str).str.strip() != "")
        if not set(events.loc[named, "candidate_id"]).issubset(candidate_ids):
            problems.append("events references a candidate_id that does not exist")
        if (events.loc[~named, "candidate_effect"].notna()).any() or events.loc[named, "candidate_effect"].isna().any():
            problems.append("candidate_effect must be present exactly when a candidate is named")
        if (events.source_type.eq("Candidate") & ~named).any() or (events.source_type.eq("External") & named).any():
            problems.append("Candidate events require a candidate; External events must not name one")
        if (named & ~region_scope).any():
            problems.append("state-level events must not name a candidate")
        candidate_geography = candidates.drop_duplicates("candidate_id").set_index("candidate_id")[["constituency_id", "region_id"]]
        named_events = events.loc[named & events.candidate_id.isin(candidate_ids)]
        mapped = candidate_geography.reindex(named_events.candidate_id)
        for column in ("constituency_id", "region_id"):
            scoped_mask = named_events[column].notna().to_numpy()
            if not (mapped[column].to_numpy()[scoped_mask] == named_events[column].to_numpy()[scoped_mask]).all():
                problems.append(f"event candidate must belong to its scoped {column}")
        effects = pd.to_numeric(events.loc[events.candidate_effect.notna(), "candidate_effect"], errors="coerce")
        if effects.isna().any() or not effects.between(-1, 1).all():
            problems.append("events.candidate_effect must be in [-1, 1] when present")

    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))
