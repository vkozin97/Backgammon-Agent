"""Utilities for match win probabilities.

`get_match_win_probs(n)` returns an `n x n` table where item `[i, j]`
represents the probability to win the match when the current player is
`i + 1` points away and the opponent is `j + 1` points away.

Values are in [0, 1].
"""

from __future__ import annotations

import numpy as np


def _woosley_like_prob(my_away: int, opp_away: int) -> float:
    """Smooth Woosley-style approximation in [0, 1]."""
    if my_away <= 0:
        return 1.0
    if opp_away <= 0:
        return 0.0
    # Symmetric logistic approximation around equal-away scores.
    # Tuned to be steeper near the end of the match.
    scale = 1.6 + 0.12 * min(my_away, opp_away)
    x = (opp_away - my_away) / scale
    return float(1.0 / (1.0 + np.exp(-x)))


def get_match_win_probs(n: int) -> np.ndarray:
    """Build an n*n Woosley match equity table in the [0, 1] range."""
    if n <= 0:
        raise ValueError("n must be > 0")
    table = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            table[i, j] = np.float32(_woosley_like_prob(i + 1, j + 1))

    # Enforce exact anti-symmetry: P(a,b) + P(b,a) = 1.
    for i in range(n):
        for j in range(i + 1, n):
            p = float(table[i, j])
            table[j, i] = np.float32(1.0 - p)
    np.fill_diagonal(table, np.float32(0.5))
    return table
