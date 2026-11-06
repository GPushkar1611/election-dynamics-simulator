"""Campaign recipients and daily effects (plan sections 3.3 and 3.4)."""

import hashlib
import math

import numpy as np

from .models import CampaignAction


def action_rng(campaign_seed: list[int], action_id: str) -> np.random.Generator:
    """One independent stream per action, so adding or reordering actions
    never changes another action's recipients."""
    salt = int(hashlib.sha256(action_id.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([*campaign_seed, salt])


def budget(voter_count: int, reach_fraction: float) -> int:
    """Number of recipients; identical for every targeting policy."""
    return math.floor(voter_count * reach_fraction + 0.5)


def select_recipients(action: CampaignAction, voter_count: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Pick an action's recipients according to its target_policy.

    Targeting policies (group, high susceptibility, high influence, ...) are
    added here. Each must return ``budget(...)`` unique, sorted voter indices.
    """
    count = budget(voter_count, action.reach_fraction)
    if action.target_policy == "uniform":
        return np.sort(rng.choice(voter_count, size=count, replace=False))
    raise ValueError(f"unknown target_policy {action.target_policy}")


def apply_ordinary(opinions: np.ndarray, susceptibility: np.ndarray, recipients: np.ndarray,
                   platform: float, action: CampaignAction, lambda_campaign: float) -> np.ndarray:
    """x_i <- clip(x_i + a * s_i * lambda * (platform - x_i)) for recipients only."""
    updated = np.asarray(opinions, dtype=float).copy()
    movement = action.intensity * lambda_campaign * susceptibility[recipients] * (platform - updated[recipients])
    updated[recipients] = np.clip(updated[recipients] + movement, -1, 1)
    return updated


def apply_smear(penalty: np.ndarray, susceptibility: np.ndarray, recipients: np.ndarray,
                target_index: int, action: CampaignAction, lambda_campaign: float) -> np.ndarray:
    """Add a * s_i * lambda to recipients' utility penalty for the target candidate.

    The penalty accumulates over active days and persists to the election.
    """
    updated = np.asarray(penalty, dtype=float).copy()
    updated[recipients, target_index] += action.intensity * lambda_campaign * susceptibility[recipients]
    return updated
