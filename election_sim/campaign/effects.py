"""Campaign recipients and daily effects (plan sections 3.3 and 3.4)."""

import hashlib
import math

import numpy as np
import pandas as pd

from .models import CampaignAction


def action_rng(campaign_seed: list[int], action_id: str) -> np.random.Generator:
    """One independent stream per action, so adding or reordering actions
    never changes another action's recipients."""
    salt = int(hashlib.sha256(action_id.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([*campaign_seed, salt])


def budget(voter_count: int, reach_fraction: float) -> int:
    """Number of recipients; identical for every targeting policy."""
    return math.floor(voter_count * reach_fraction + 0.5)


GROUP_MATCHERS = {
    "General": lambda voters: np.ones(len(voters), dtype=bool),
    "Rural": lambda voters: voters.urban_rural.eq("Rural").to_numpy(),
    "Urban": lambda voters: voters.urban_rural.eq("Urban").to_numpy(),
    "Students": lambda voters: voters.occupation.eq("Student").to_numpy(),
    "Farmers": lambda voters: voters.occupation.isin(["Farmer", "Agricultural labourer"]).to_numpy(),
    "Unemployed": lambda voters: voters.occupation.eq("Unemployed").to_numpy(),
    "Low-income": lambda voters: voters.wealth_level.eq("Low").to_numpy(),
    "Middle-income": lambda voters: voters.wealth_level.eq("Middle").to_numpy(),
    "High-income": lambda voters: voters.wealth_level.eq("High").to_numpy(),
}


def outgoing_influence_scores(voters: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """Weighted out-degree times influence_power, in canonical voter order."""
    voter_index = {voter_id: index for index, voter_id in enumerate(voters.voter_id)}
    source_indices = edges.source_voter_id.map(voter_index).to_numpy(dtype=int)
    weighted_out_degree = np.bincount(
        source_indices, weights=edges.edge_weight.to_numpy(dtype=float), minlength=len(voters))
    return weighted_out_degree * voters.influence_power.to_numpy(dtype=float)


def assign_communities(voters: pd.DataFrame, edges: pd.DataFrame,
                       rng: np.random.Generator, max_iterations: int = 100) -> np.ndarray:
    """Infer weakly connected social communities using seeded label propagation.

    Direction is ignored for community membership. Labels are canonicalized by
    the lowest voter index in each community, making IDs stable for a fixed
    voter order, graph, and seed. This is a model-derived partition, not a
    supplied real-world community label.
    """
    count = len(voters)
    index = {voter_id: i for i, voter_id in enumerate(voters.voter_id)}
    neighbors = [set() for _ in range(count)]
    for source, target in zip(edges.source_voter_id, edges.target_voter_id):
        source_index, target_index = index[source], index[target]
        neighbors[source_index].add(target_index)
        neighbors[target_index].add(source_index)
    labels = np.arange(count, dtype=int)
    for _ in range(max_iterations):
        changed = 0
        for node in rng.permutation(count):
            if not neighbors[node]:
                continue
            neighbor_labels, frequencies = np.unique(
                labels[list(neighbors[node])], return_counts=True)
            leaders = neighbor_labels[frequencies == frequencies.max()]
            new_label = int(rng.choice(leaders))
            if new_label != labels[node]:
                labels[node] = new_label
                changed += 1
        if changed == 0:
            break
    roots = sorted(set(labels.tolist()), key=lambda root: int(np.flatnonzero(labels == root).min()))
    root_to_community = {root: i for i, root in enumerate(roots)}
    return np.array([root_to_community[label] for label in labels], dtype=int)


def select_recipients(action: CampaignAction, voters: pd.DataFrame, edges: pd.DataFrame,
                      rng: np.random.Generator,
                      susceptibility: np.ndarray | None = None,
                      communities: np.ndarray | None = None) -> np.ndarray:
    """Select a fixed constituency-wide exposure budget using the action policy.

    If a group is smaller than the budget, all members are selected and the
    caller can report the shortfall. Ranking policies randomize exact ties
    using the action's seeded generator.
    """
    voter_count = len(voters)
    count = budget(voter_count, action.reach_fraction)
    if action.target_policy == "uniform":
        return np.sort(rng.choice(voter_count, size=count, replace=False))
    if action.target_policy == "group":
        eligible = np.flatnonzero(GROUP_MATCHERS[action.target_group](voters))
        return np.sort(rng.choice(eligible, size=min(count, len(eligible)), replace=False))
    if action.target_policy == "community":
        if communities is None or len(communities) != voter_count:
            raise ValueError("community assignments are required for community targeting")
        labels, sizes = np.unique(communities, return_counts=True)
        if action.community_id is not None:
            if action.community_id not in labels:
                raise ValueError(f"unknown community_id {action.community_id}")
            community = action.community_id
        else:
            sufficiently_large = labels[sizes >= count]
            candidates = sufficiently_large if len(sufficiently_large) else labels[sizes == sizes.max()]
            community = int(rng.choice(candidates))
        eligible = np.flatnonzero(communities == community)
        return np.sort(rng.choice(eligible, size=min(count, len(eligible)), replace=False))
    if action.target_policy == "high_susceptibility":
        scores = (voters.susceptibility.to_numpy(dtype=float) if susceptibility is None
                  else np.asarray(susceptibility, dtype=float))
    elif action.target_policy == "high_influence":
        scores = outgoing_influence_scores(voters, edges)
    else:  # The dataclass validates policies; keep this guard for direct callers.
        raise ValueError(f"unknown target_policy {action.target_policy}")
    tie_break = rng.random(voter_count)
    return np.lexsort((tie_break, -scores))[:count].astype(int)


def message_position(action: CampaignAction, platform: float, opinions: np.ndarray,
                     recipients: np.ndarray) -> float:
    """Where a message pulls recipients: the real platform for ordinary
    messages; for misleading ones, the platform moved toward the audience's
    mean opinion by `distortion` (0 = honest, 1 = tells them what they think)."""
    if action.message_type != "misleading" or len(recipients) == 0:
        return platform
    return platform + action.distortion * (float(np.mean(opinions[recipients])) - platform)


def apply_ordinary(opinions: np.ndarray, susceptibility: np.ndarray, recipients: np.ndarray,
                   position: float, action: CampaignAction, lambda_campaign: float) -> np.ndarray:
    """x_i <- clip(x_i + a * s_i * lambda * (position - x_i)) for recipients only.

    Used for ordinary and misleading messages; see message_position."""
    updated = np.asarray(opinions, dtype=float).copy()
    movement = action.intensity * lambda_campaign * susceptibility[recipients] * (position - updated[recipients])
    updated[recipients] = np.clip(updated[recipients] + movement, -1, 1)
    return updated


# Belief in a smear claim ("candidate c did something bad"), per voter and candidate.
UNAWARE, BELIEVES, REJECTS = 0, 1, 2


def apply_smear(penalty: np.ndarray, susceptibility: np.ndarray, recipients: np.ndarray,
                target_index: int, action: CampaignAction, lambda_campaign: float) -> np.ndarray:
    """Add a * s_i * lambda to recipients' smear strength for the target candidate.

    The strength accumulates over active days. It lowers the voter's utility
    for that candidate for as long as the voter believes the claim.
    """
    updated = np.asarray(penalty, dtype=float).copy()
    updated[recipients, target_index] += action.intensity * lambda_campaign * susceptibility[recipients]
    return updated


def spread_smears(belief: np.ndarray, strength: np.ndarray, sources: np.ndarray,
                  targets: np.ndarray, edge_weights: np.ndarray, susceptibility: np.ndarray,
                  probability: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One day of smear rumours spreading along social ties.

    belief and strength are voters x candidates arrays (one smear claim per
    candidate). Every believer tells each person it influences (source ->
    target tie) with probability `probability * edge_weight`. A listener who
    had not heard the claim believes it with probability equal to their
    susceptibility, otherwise rejects it for good. A new believer takes the
    teller's strength scaled by the tie weight (the strongest teller if
    several), so rumours weaken as they travel. Everyone reacts to the
    same start-of-step snapshot. Returns updated copies.
    """
    new_belief, new_strength = belief.copy(), strength.copy()
    for candidate in range(belief.shape[1]):
        believers = belief[:, candidate] == BELIEVES
        if not believers.any():
            continue
        tells = believers[sources] & (rng.random(len(sources)) < probability * edge_weights)
        heard_strength = np.zeros(len(belief))
        np.maximum.at(heard_strength, targets[tells],
                      edge_weights[tells] * strength[sources[tells], candidate])
        heard = np.zeros(len(belief), dtype=bool)
        heard[targets[tells]] = True
        listeners = np.flatnonzero(heard & (belief[:, candidate] == UNAWARE))
        convinced = rng.random(len(listeners)) < susceptibility[listeners]
        new_belief[listeners[convinced], candidate] = BELIEVES
        new_belief[listeners[~convinced], candidate] = REJECTS
        new_strength[listeners[convinced], candidate] = heard_strength[listeners[convinced]]
    return new_belief, new_strength
