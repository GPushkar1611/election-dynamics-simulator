"""Section 3.1: popularity prior plus distance to mean candidate ideology,
minus any per-voter smear penalty (section 3.3)."""

import numpy as np
import pandas as pd

ISSUES = ("employment", "economy", "education", "healthcare", "agriculture")


def candidate_support(opinions: np.ndarray, candidates: pd.DataFrame,
                      b_pop: float = 1.0, b_ideol: float = 1.0,
                      tau: float = 1.0, penalty: np.ndarray | None = None) -> np.ndarray:
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
    utility = utility - b_ideol * np.abs(x[:, None] - positions.mean(axis=1)[None, :])
    if penalty is not None:
        if penalty.shape != utility.shape or not np.isfinite(penalty).all() or (penalty < 0).any():
            raise ValueError("penalty must be a finite nonnegative voters x candidates array")
        utility = utility - penalty
    if not np.isfinite(utility).all():
        raise ValueError("candidate utility overflowed; reduce support coefficients")
    # Subtract before dividing to avoid positive overflow at low temperature.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((utility - utility.max(axis=1, keepdims=True)) / tau)
    return weights / weights.sum(axis=1, keepdims=True)
