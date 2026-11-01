"""Section 3.6 baseline: propensity-only turnout and a separate vote RNG."""

import numpy as np
import pandas as pd


def election_result(propensity: np.ndarray, support: np.ndarray,
                    candidate_ids: list[str], turnout_rng: np.random.Generator,
                    vote_rng: np.random.Generator, rule: str = "sample") -> tuple[pd.DataFrame, dict]:
    p = np.asarray(propensity, dtype=float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("turnout propensity must be finite and in [0, 1]")
    if support.shape != (len(p), len(candidate_ids)) or not len(candidate_ids):
        raise ValueError("support shape must match voters and candidates")
    if not np.isfinite(support).all() or (support < 0).any() or not np.allclose(support.sum(axis=1), 1):
        raise ValueError("support rows must be probability distributions")
    participating = turnout_rng.random(len(p)) < p
    # Draw once per voter even for abstainers to preserve paired random streams.
    if rule == "sample":
        draws = vote_rng.random(len(p))
        choices = (draws[:, None] >= support.cumsum(axis=1)).sum(axis=1)
        choices = np.minimum(choices, len(candidate_ids) - 1)
    elif rule == "argmax":
        choices = support.argmax(axis=1)
    else:
        raise ValueError("vote rule must be sample or argmax")
    counts = np.bincount(choices[participating], minlength=len(candidate_ids))
    turnout = int(participating.sum())
    results = pd.DataFrame({"candidate_id": candidate_ids, "votes": counts,
                            "vote_share": counts / turnout if turnout else np.zeros(len(counts))})
    maximum = int(counts.max())
    winners = [candidate_ids[i] for i in np.flatnonzero(counts == maximum)] if turnout else []
    ranked = np.sort(counts)[::-1]
    summary = {"population": len(p), "turnout": turnout, "abstentions": len(p) - turnout,
               "turnout_rate": turnout / len(p) if len(p) else 0.0,
               "winner_ids": winners, "tied": len(winners) > 1,
               "margin_votes": int(ranked[0] - ranked[1]) if len(ranked) > 1 else maximum,
               "zero_turnout": turnout == 0}
    assert int(counts.sum()) == turnout
    assert turnout + summary["abstentions"] == len(p)
    return results, summary
