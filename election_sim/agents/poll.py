"""Opinion polls: the only view of the electorate that candidate agents get."""

import numpy as np
import pandas as pd

from ..campaign.effects import GROUP_MATCHERS

# A group is reported only if at least this many respondents belong to it.
MIN_GROUP_RESPONDENTS = 10


def run_poll(support: np.ndarray, voters: pd.DataFrame, candidate_ids: list[str],
             rng: np.random.Generator, day: int, sample_size: int = 300,
             noise: float = 0.0) -> dict:
    """Ask a random sample of voters which candidate they currently lean to.

    Each respondent names their highest-support candidate. Shares get
    Gaussian noise with standard deviation `noise` and are renormalised.
    Shares are also reported per targeting group (the GROUP_MATCHERS labels,
    e.g. "Rural", "Students") so agents can aim group campaigns.

    Returns {"day", "sample_size", "shares": {candidate: share},
             "group_shares": {group: {candidate: share}}}.
    """
    size = min(sample_size, len(voters))
    sample = np.sort(rng.choice(len(voters), size=size, replace=False))
    answers = support[sample].argmax(axis=1)

    def shares(respondents: np.ndarray) -> dict:
        counts = np.bincount(answers[respondents], minlength=len(candidate_ids))
        values = np.clip(counts / counts.sum() + rng.normal(0, noise, len(counts)), 0, None)
        if values.sum() == 0:  # extreme noise wiped everything out
            values = np.ones(len(values))
        return {candidate: float(value) for candidate, value in zip(candidate_ids, values / values.sum())}

    overall = shares(np.ones(size, dtype=bool))
    group_shares = {}
    for group, matches in GROUP_MATCHERS.items():
        if group == "General":
            continue
        try:
            members = matches(voters)[sample]
        except AttributeError:  # these voters lack the attribute this group uses
            continue
        if members.sum() >= MIN_GROUP_RESPONDENTS:
            group_shares[group] = shares(members)
    return {"day": day, "sample_size": size, "shares": overall, "group_shares": group_shares}
