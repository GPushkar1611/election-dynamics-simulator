from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # Supports both package imports and direct use from this folder.
    from .validation import validate_data
except ImportError:  # pragma: no cover - used by the example notebook
    from validation import validate_data


def load_constituency(data_dir: str | Path, constituency_id: int = 1,
                      validate: bool = True) -> dict[str, object]:
    """Load a constituency slice from a folder containing the four CSV files.

    Returns a dictionary with ``voters``, ``candidates``, ``social_network``,
    ``events``, and ``diagnostics`` DataFrames/dictionary. Edges are directed
    (source influences target); only edges with both endpoints in this
    constituency are returned. Event scope includes constituency, its region,
    and state-wide events.
    """
    data_dir = Path(data_dir)
    voters = pd.read_csv(data_dir / "voters.csv")
    social_network = pd.read_csv(data_dir / "social_network.csv")
    candidates = pd.read_csv(data_dir / "candidates.csv")
    events = pd.read_csv(data_dir / "political_events.csv",
                         dtype={"region_id": "Int64", "constituency_id": "Int64"})

    if validate:
        validate_data(voters, social_network, candidates, events)

    local_voters = voters.loc[voters.constituency_id == constituency_id].copy()
    if local_voters.empty:
        available = sorted(voters.constituency_id.dropna().unique().tolist())
        raise ValueError(f"Constituency {constituency_id} was not found. Available IDs: {available}")
    region_id = int(local_voters.region_id.iloc[0])
    local_candidates = candidates.loc[candidates.constituency_id == constituency_id].copy()
    local_ids = set(local_voters.voter_id)

    touches_local = (social_network.source_voter_id.isin(local_ids) |
                     social_network.target_voter_id.isin(local_ids))
    local_edge_mask = (social_network.source_voter_id.isin(local_ids) &
                       social_network.target_voter_id.isin(local_ids))
    local_edges = social_network.loc[local_edge_mask].copy()
    boundary_edges = social_network.loc[touches_local & ~local_edge_mask].copy()

    event_region = events.region_id.eq(region_id)
    state_scope = events.region_id.isna() & events.constituency_id.isna()
    local_scope = events.constituency_id.eq(constituency_id)
    local_events = events.loc[local_scope | event_region | state_scope].copy()

    return {
        "voters": local_voters.reset_index(drop=True),
        "candidates": local_candidates.reset_index(drop=True),
        "social_network": local_edges.reset_index(drop=True),
        "events": local_events.reset_index(drop=True),
        "diagnostics": {
            "constituency_id": constituency_id,
            "region_id": region_id,
            "voter_count": len(local_voters),
            "candidate_count": len(local_candidates),
            "internal_edge_count": len(local_edges),
            "excluded_boundary_edge_count": len(boundary_edges),
            "event_count": len(local_events),
        },
    }
