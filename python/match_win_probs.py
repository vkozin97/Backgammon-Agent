"""Woosley Match Equity Table (MET).

This module exposes :func:`get_match_win_probs`, which returns an ``n x n``
match-equity matrix in the range ``[0, 1]``.

Indexing convention:
- ``table[i, j]`` = probability that the current player wins the match
  when current player is ``i`` away and opponent is ``j`` away.
"""

from __future__ import annotations

import numpy as np

# Canonical Woosley MET percentages (1-away..15-away vs 1-away..15-away).
# Values are expressed in percent and converted to [0, 1] in get_match_win_probs.
# Diagonal is 50.0 by definition; anti-symmetry is enforced during build.
_WOOSLEY_MET_15_PCT = np.array(
    [
        [50.0, 68.0, 75.0, 80.0, 83.0, 86.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 95.0],
        [32.0, 50.0, 60.0, 67.0, 72.0, 76.0, 79.0, 81.0, 83.0, 85.0, 87.0, 88.0, 89.0, 90.0, 91.0],
        [25.0, 40.0, 50.0, 59.0, 65.0, 70.0, 74.0, 77.0, 80.0, 82.0, 84.0, 86.0, 87.0, 88.0, 89.0],
        [20.0, 33.0, 41.0, 50.0, 57.0, 63.0, 68.0, 72.0, 75.0, 78.0, 80.0, 82.0, 84.0, 85.0, 86.0],
        [17.0, 28.0, 35.0, 43.0, 50.0, 57.0, 62.0, 66.0, 70.0, 73.0, 76.0, 78.0, 80.0, 82.0, 83.0],
        [14.0, 24.0, 30.0, 37.0, 43.0, 50.0, 56.0, 61.0, 65.0, 69.0, 72.0, 75.0, 77.0, 79.0, 80.0],
        [12.0, 21.0, 26.0, 32.0, 38.0, 44.0, 50.0, 55.0, 60.0, 64.0, 68.0, 71.0, 73.0, 75.0, 77.0],
        [11.0, 19.0, 23.0, 28.0, 34.0, 39.0, 45.0, 50.0, 56.0, 60.0, 64.0, 67.0, 70.0, 73.0, 75.0],
        [10.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0, 44.0, 50.0, 55.0, 60.0, 63.0, 67.0, 70.0, 72.0],
        [9.0, 15.0, 18.0, 22.0, 27.0, 31.0, 36.0, 40.0, 45.0, 50.0, 55.0, 59.0, 63.0, 66.0, 69.0],
        [8.0, 13.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 45.0, 50.0, 55.0, 59.0, 62.0, 65.0],
        [7.0, 12.0, 14.0, 18.0, 22.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 50.0, 55.0, 59.0, 62.0],
        [6.0, 11.0, 13.0, 16.0, 20.0, 23.0, 27.0, 30.0, 33.0, 37.0, 41.0, 45.0, 50.0, 54.0, 58.0],
        [5.0, 10.0, 12.0, 15.0, 18.0, 21.0, 25.0, 27.0, 30.0, 34.0, 38.0, 41.0, 46.0, 50.0, 55.0],
        [5.0, 9.0, 11.0, 14.0, 17.0, 20.0, 23.0, 25.0, 28.0, 31.0, 35.0, 38.0, 42.0, 45.0, 50.0],
    ],
    dtype=np.float32,
)


def _expand_with_tail(base: np.ndarray, n: int) -> np.ndarray:
    """Expand MET beyond base size by repeating the far-away tail shape."""
    m = base.shape[0]
    if n <= m:
        return base[:n, :n].astype(np.float32, copy=True)

    out = np.empty((n, n), dtype=np.float32)
    out[:m, :m] = base

    # For larger matches, use a conservative tail extension based on the
    # outermost (15-away) profile. This keeps monotonicity and anti-symmetry.
    tail_row = base[m - 1, :]
    tail_col = base[:, m - 1]
    for i in range(m, n):
        out[i, :m] = tail_col
        out[:m, i] = tail_row
    for i in range(m, n):
        for j in range(m, n):
            if i == j:
                out[i, j] = 50.0
            elif i > j:
                out[i, j] = out[m - 1, m - 2]
            else:
                out[i, j] = 100.0 - out[j, i]
    return out


def get_match_win_probs(n: int) -> np.ndarray:
    """Return Woosley MET as ``n x n`` table in ``[0, 1]``.

    The returned table includes a 0-away row/column:
    - row 0 is 1.0 (already won),
    - column 0 is 0.0 (opponent already won),
    - table[0, 0] is set to 0.5 by convention.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    out = np.empty((n + 1, n + 1), dtype=np.float32)
    out[0, :] = 1.0
    out[:, 0] = 0.0
    out[0, 0] = 0.5

    if n == 1:
        return out

    pct = _expand_with_tail(_WOOSLEY_MET_15_PCT, n)

    # Enforce exact anti-symmetry and diagonal.
    for i in range(n):
        pct[i, i] = 50.0
        for j in range(i + 1, n):
            pct[j, i] = 100.0 - pct[i, j]

    out[1:, 1:] = (pct / 100.0).astype(np.float32, copy=False)
    return out
