"""Candidate utility from popularity and either scalar or issue-aware policy fit."""

import numpy as np
import pandas as pd

ISSUES = ("employment", "economy", "education", "healthcare", "agriculture")


def candidate_support(opinions: np.ndarray, candidates: pd.DataFrame,
                      b_pop: float = 1.0, b_ideol: float = 1.0,
                      tau: float = 1.0, penalty: np.ndarray | None = None,
                      utility_adjustment: np.ndarray | None = None,
                      issue_preferences: np.ndarray | None = None,
                      issue_importance: np.ndarray | None = None) -> np.ndarray:
    if not np.isfinite([b_pop, b_ideol, tau]).all() or min(b_pop, b_ideol) < 0 or tau <= 0:
        raise ValueError("support coefficients must be nonnegative and tau positive")
    x = np.asarray(opinions, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or (np.abs(x) > 1).any():
        raise ValueError("opinions must be a finite vector in [-1, 1]")
    positions = candidates[[f"position_{issue}" for issue in ISSUES]].to_numpy(dtype=float)
    popularity = candidates.local_popularity.to_numpy(dtype=float)
    if len(candidates) == 0 or not np.isfinite(positions).all() or (np.abs(positions) > 1).any():
        raise ValueError("candidates need five finite issue positions in [-1, 1]")
    if not np.isfinite(popularity).all() or not ((popularity >= 0) & (popularity <= 1)).all():
        raise ValueError("candidate popularity must be finite and in [0, 1]")
    p = np.clip(popularity, 1e-6, 1 - 1e-6)
    utility = b_pop * (np.log(p) - np.log1p(-p))[None, :]
    if issue_preferences is None and issue_importance is None:
        policy_distance = np.abs(x[:, None] - positions.mean(axis=1)[None, :])
    else:
        if issue_preferences is None or issue_importance is None:
            raise ValueError("issue_preferences and issue_importance must be supplied together")
        preferences = np.asarray(issue_preferences, dtype=float)
        importance = np.asarray(issue_importance, dtype=float)
        expected_shape = (len(x), len(ISSUES))
        if (preferences.shape != expected_shape or not np.isfinite(preferences).all()
                or (np.abs(preferences) > 1).any()):
            raise ValueError("issue_preferences must be finite voters x issues values in [-1, 1]")
        if (importance.shape != expected_shape or not np.isfinite(importance).all()
                or (importance < 0).any()):
            raise ValueError("issue_importance must be finite, nonnegative voters x issues values")
        totals = importance.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise ValueError("each voter must assign positive total importance across issues")
        weights = importance / totals
        policy_distance = (weights[:, None, :] *
                           np.abs(preferences[:, None, :] - positions[None, :, :])).sum(axis=2)
    utility = utility - b_ideol * policy_distance
    if penalty is not None:
        if penalty.shape != utility.shape or not np.isfinite(penalty).all() or (penalty < 0).any():
            raise ValueError("penalty must be a finite nonnegative voters x candidates array")
        utility = utility - penalty
    if utility_adjustment is not None:
        if (utility_adjustment.shape != utility.shape or not np.isfinite(utility_adjustment).all()):
            raise ValueError("utility_adjustment must be a finite voters x candidates array")
        utility = utility + utility_adjustment
    if not np.isfinite(utility).all():
        raise ValueError("candidate utility overflowed; reduce support coefficients")
    # Subtract before dividing to avoid positive overflow at low temperature.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((utility - utility.max(axis=1, keepdims=True)) / tau)
    return weights / weights.sum(axis=1, keepdims=True)


def synthetic_issue_offsets(voter_count: int, dispersion: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Create zero-mean per-issue deviations around a scalar opinion.

    These offsets are a transparent sensitivity-analysis assumption, not
    measured voter positions. Each row averages to zero before clipping.
    """
    if isinstance(voter_count, bool) or not isinstance(voter_count, int) or voter_count < 1:
        raise ValueError("voter_count must be a positive integer")
    if not np.isfinite(dispersion) or not 0 <= dispersion <= 1:
        raise ValueError("dispersion must be finite and in [0, 1]")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    offsets = rng.uniform(-1, 1, size=(voter_count, len(ISSUES)))
    offsets -= offsets.mean(axis=1, keepdims=True)
    max_abs = np.abs(offsets).max(axis=1, keepdims=True)
    normalized = np.divide(offsets, max_abs, out=np.zeros_like(offsets), where=max_abs > 0)
    return normalized * dispersion
