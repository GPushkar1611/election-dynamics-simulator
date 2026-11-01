"""Section 3.2: synchronous DeGroot update with target bounded confidence."""

import numpy as np

from .network import IncomingGraph


def social_step(opinions: np.ndarray, susceptibility: np.ndarray,
                confidence: np.ndarray, graph: IncomingGraph,
                lambda_social: float) -> tuple[np.ndarray, int]:
    # All means read this same snapshot. There is no in-place voter update.
    x = np.asarray(opinions, dtype=float)
    eligible = np.abs(x[graph.sources] - x[graph.targets]) <= confidence[graph.targets]
    sources, targets = graph.sources[eligible], graph.targets[eligible]
    weights = graph.weights[eligible]
    totals = np.bincount(targets, weights=weights, minlength=len(x))
    weighted = np.bincount(targets, weights=weights * x[sources], minlength=len(x))
    means = x.copy()
    np.divide(weighted, totals, out=means, where=totals > 0)
    updated = np.clip(x + susceptibility * lambda_social * (means - x), -1, 1)
    return updated, int(eligible.sum())
