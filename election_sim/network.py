"""Canonical incoming adjacency: every source -> target retains its direction."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IncomingGraph:
    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray


def build_incoming(voters: pd.DataFrame, edges: pd.DataFrame, rho: float) -> IncomingGraph:
    ids = pd.Index(voters.voter_id)
    if ids.has_duplicates or ids.isna().any():
        raise ValueError("voter IDs must be present and unique")
    ordered = edges.sort_values(["target_voter_id", "source_voter_id"])
    sources = ids.get_indexer(ordered.source_voter_id)
    targets = ids.get_indexer(ordered.target_voter_id)
    if (sources < 0).any() or (targets < 0).any():
        raise ValueError("all edge endpoints must belong to the selected constituency")
    if (sources == targets).any() or ordered.duplicated(["source_voter_id", "target_voter_id"]).any():
        raise ValueError("self and duplicate directed edges are not allowed")
    ties = ordered.edge_weight.to_numpy(dtype=float)
    powers = voters.influence_power.to_numpy(dtype=float)
    if not np.isfinite(ties).all() or not ((ties > 0) & (ties <= 1)).all():
        raise ValueError("edge weights must be finite and in (0, 1]")
    if not np.isfinite(powers).all() or not (powers > 0).all():
        raise ValueError("influence power must be finite and positive")
    # Stable target-index ordering makes offsets valid even for nonalphabetic input IDs.
    order = np.argsort(targets, kind="stable")
    sources, targets = sources[order], targets[order]
    weights = ties[order] * powers[sources] ** rho
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("sender weights overflowed or underflowed; reduce rho")
    offsets = np.r_[0, np.cumsum(np.bincount(targets, minlength=len(voters)))]
    return IncomingGraph(sources, targets, weights, offsets)
