from __future__ import annotations

import numpy as np

POINTS_DIM = 24
VECTOR_CHANNELS = 6
SCALAR_FEATURES_DIM = 14
OBSERVATION_DIM = VECTOR_CHANNELS * POINTS_DIM + SCALAR_FEATURES_DIM


def state_to_observation(raw_state: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32)
    points = raw[:24]
    opp_points = raw[24:48]
    bar, off, opp_bar, opp_off = raw[48:52]

    blots = (points == 1.0).astype(np.float32)
    opp_blots = (opp_points == 1.0).astype(np.float32)
    anchors = (points >= 2.0).astype(np.float32)
    opp_anchors = (opp_points >= 2.0).astype(np.float32)

    pip_mine = float(np.dot(points, np.arange(1, 25, dtype=np.float32)) + bar * 25.0)
    pip_opp = float(np.dot(opp_points, np.arange(24, 0, -1, dtype=np.float32)) + opp_bar * 25.0)
    blots_mine = float(np.sum(blots))
    blots_opp = float(np.sum(opp_blots))
    anchors_mine = float(np.sum(anchors))
    anchors_opp = float(np.sum(opp_anchors))
    blot_pips_mine = float(np.dot(blots, np.arange(1, 25, dtype=np.float32)))
    blot_pips_opp = float(np.dot(opp_blots, np.arange(24, 0, -1, dtype=np.float32)))
    anchor_pips_mine = float(np.dot(anchors, np.arange(1, 25, dtype=np.float32)))
    anchor_pips_opp = float(np.dot(opp_anchors, np.arange(24, 0, -1, dtype=np.float32)))

    scalars = np.array(
        [
            bar,
            off,
            opp_bar,
            opp_off,
            pip_mine,
            pip_opp,
            blots_mine,
            blots_opp,
            anchors_mine,
            anchors_opp,
            blot_pips_mine,
            blot_pips_opp,
            anchor_pips_mine,
            anchor_pips_opp,
        ],
        dtype=np.float32,
    )
    return np.concatenate([points, opp_points, blots, opp_blots, anchors, opp_anchors, scalars], dtype=np.float32)

