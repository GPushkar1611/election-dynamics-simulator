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


def mobilized_propensity(propensity, campaign_exposure, event_exposure, gamma_mob=0., gamma_event=0.):
    """Exposure is cumulative voter-days, counting each channel once per day.

    Preserve exact baseline probabilities when coefficients are zero; for
    nonzero shifts clamp baseline endpoints to 1e-6 before taking logits.
    """
    p = np.asarray(propensity, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("propensity must be finite and in [0, 1]")
    shift = gamma_mob * np.asarray(campaign_exposure) + gamma_event * np.asarray(event_exposure)
    if shift.shape != p.shape or not np.isfinite(shift).all():
        raise ValueError("mobilization shift must be finite and match propensity")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(clipped) - np.log1p(-clipped) + shift
    adjusted = np.exp(-np.logaddexp(0., -z))
    return np.where(shift == 0, p, adjusted)
