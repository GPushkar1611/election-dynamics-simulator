"""Strategic voting after public polls: where epistemic logic enters the model.

Knowledge, not just opinion, drives this step:

- A candidate agent's own poll is *private*: only that agent knows the
  result, so voters cannot act on it.
- A `poll_release` news event is a *public announcement*: every voter
  receives the same poll and knows that everyone else received it too. The
  result becomes *common knowledge*.

Only with common knowledge can a supporter of a hopeless candidate expect
other supporters to give up on that candidate as well. Switching to a
front-runner is then rational, which is Duverger's law for single-winner
plurality races.

Model: the top VIABLE_COUNT candidates in the published poll are viable.
Every voter who received the announcement lowers their utility for each
non-viable candidate by `strength` (config.strategic_voting). Voters who like
a non-viable candidate much more than the alternatives still vote for it. A
later public poll replaces what voters know. Voters react to the published
numbers, noise included, not to the true state of the race.
"""

import numpy as np

# In a single-winner plurality race only the two front-runners can win.
VIABLE_COUNT = 2


def viable_candidates(shares: dict, candidate_ids: list[str]) -> np.ndarray:
    """True for the candidates a public poll shows as able to win (its top two).

    Ties at the cut-off go to the earlier candidate ID.
    """
    values = np.array([shares[candidate] for candidate in candidate_ids])
    order = np.argsort(-values, kind="stable")
    viable = np.zeros(len(candidate_ids), dtype=bool)
    viable[order[:VIABLE_COUNT]] = True
    return viable


def strategic_adjustment(viable: np.ndarray, strength: float) -> np.ndarray:
    """Utility change per candidate: 0 for viable candidates, -strength for the rest."""
    return np.where(viable, 0.0, -strength)
