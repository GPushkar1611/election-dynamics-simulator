"""Uniform ordinary-campaign exposure and opinion movement."""

import numpy as np

from .models import CampaignAction


def select_uniform_recipients(voter_count: int, reach_fraction: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Choose a fixed number of unique voters without replacement."""
    count = int(round(voter_count * reach_fraction))
    if count == 0:
        return np.empty(0, dtype=int)
    return np.sort(rng.choice(voter_count, size=count, replace=False))


def apply_campaign_action(opinions: np.ndarray, susceptibility: np.ndarray,
                          recipients: np.ndarray, message_position: float,
                          action: CampaignAction, lambda_campaign: float) -> np.ndarray:
    """Move exposed voters toward an ordinary or declared misleading message."""
    updated = np.asarray(opinions, dtype=float).copy()
    if not np.isfinite(message_position) or not -1 <= message_position <= 1:
        raise ValueError("message position must be finite and in [-1, 1]")
    if not np.isfinite(lambda_campaign) or not 0 <= lambda_campaign <= 1:
        raise ValueError("lambda_campaign must be finite and in [0, 1]")
    if recipients.size:
        movement = (action.intensity * lambda_campaign * susceptibility[recipients]
                    * (message_position - updated[recipients]))
        updated[recipients] = np.clip(updated[recipients] + movement, -1, 1)
    return updated
