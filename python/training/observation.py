from __future__ import annotations

import numpy as np

from .observation_layout import (
    OBSERVATION_DIM,
    POINTS_DIM,
    RAW_BLACK_SCORE,
    RAW_CUBE_AVAILABLE_MINE,
    RAW_CUBE_AVAILABLE_OPP,
    RAW_DAVE_VALUE,
    RAW_DOUBLE_OFFERED,
    RAW_IS_CRAWFORD_GAME,
    RAW_N_GAMES,
    RAW_WHITE_SCORE,
    RAW_WHITE_TO_MOVE,
    VECTOR_CHANNELS,
)


def _legal_steps(points: np.ndarray, opp_points: np.ndarray, bar: int, die: int) -> list[tuple[int, int]]:
    if bar > 0:
        to = 24 - die
        if 0 <= to < POINTS_DIM and opp_points[to] < 2:
            return [(-1, to)]
        return []

    steps: list[tuple[int, int]] = []
    for from_idx in range(POINTS_DIM):
        if points[from_idx] <= 0:
            continue
        to = from_idx - die
        if 0 <= to < POINTS_DIM and opp_points[to] < 2:
            steps.append((from_idx, to))
    return steps


def _can_hit_with_sequence(
    points: np.ndarray,
    opp_points: np.ndarray,
    bar: int,
    opp_bar: int,
    target_idx: int,
    dice_seq: tuple[int, ...],
) -> bool:
    def dfs(p: np.ndarray, o: np.ndarray, b: int, ob: int, k: int, already_hit: bool) -> bool:
        if already_hit:
            return True
        if k >= len(dice_seq):
            return False
        steps = _legal_steps(p, o, b, int(dice_seq[k]))
        if not steps:
            return dfs(p, o, b, ob, k + 1, already_hit)
        for step in steps:
            from_idx, to_idx = step
            landed_on_target = to_idx == target_idx and o[target_idx] == 1

            if from_idx == -1:
                next_b = b - 1
            else:
                p[from_idx] -= 1
                next_b = b

            hit = o[to_idx] == 1
            if hit:
                o[to_idx] = 0
            p[to_idx] += 1

            if dfs(p, o, next_b, ob + int(hit), k + 1, already_hit or landed_on_target):
                p[to_idx] -= 1
                if hit:
                    o[to_idx] = 1
                if from_idx != -1:
                    p[from_idx] += 1
                return True

            p[to_idx] -= 1
            if hit:
                o[to_idx] = 1
            if from_idx != -1:
                p[from_idx] += 1
        return False

    return dfs(points, opp_points, bar, opp_bar, 0, False)


def _can_cover_with_sequence(points: np.ndarray, opp_points: np.ndarray, bar: int, target_idx: int, dice_seq: tuple[int, ...]) -> bool:
    def dfs(p: np.ndarray, o: np.ndarray, b: int, k: int) -> bool:
        if p[target_idx] >= 2:
            return True
        if k >= len(dice_seq):
            return False
        steps = _legal_steps(p, o, b, int(dice_seq[k]))
        if not steps:
            return dfs(p, o, b, k + 1)
        for step in steps:
            from_idx, to_idx = step
            if from_idx == -1:
                next_b = b - 1
            else:
                p[from_idx] -= 1
                next_b = b

            hit = o[to_idx] == 1
            if hit:
                o[to_idx] = 0
            p[to_idx] += 1

            if dfs(p, o, next_b, k + 1):
                p[to_idx] -= 1
                if hit:
                    o[to_idx] = 1
                if from_idx != -1:
                    p[from_idx] += 1
                return True

            p[to_idx] -= 1
            if hit:
                o[to_idx] = 1
            if from_idx != -1:
                p[from_idx] += 1
        return False

    return dfs(points, opp_points, bar, 0)


def _dice_roll_sequences() -> list[tuple[tuple[int, ...], float]]:
    seqs: list[tuple[tuple[int, ...], float]] = []
    for a in range(1, 7):
        for b in range(1, 7):
            seq = (a, a, a, a) if a == b else (a, b)
            seqs.append((seq, 1.0 / 36.0))
    return seqs


_DICE_SEQUENCES = _dice_roll_sequences()


def _prob_vectors(points: np.ndarray, opp_points: np.ndarray, bar: int, opp_bar: int) -> tuple[np.ndarray, np.ndarray]:
    threatened = np.zeros((POINTS_DIM,), dtype=np.float32)
    cover = np.zeros((POINTS_DIM,), dtype=np.float32)
    for i in range(POINTS_DIM):
        if points[i] != 1.0:
            continue
        p_hit = 0.0
        p_cover = 0.0
        for dice_seq, weight in _DICE_SEQUENCES:
            if _can_hit_with_sequence(opp_points, points, int(opp_bar), int(bar), i, dice_seq):
                p_hit += weight
            if _can_cover_with_sequence(points, opp_points, int(bar), i, dice_seq):
                p_cover += weight
        threatened[i] = np.float32(p_hit)
        cover[i] = np.float32(p_cover)
    return threatened, cover


def state_to_observation(raw_state: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_state, dtype=np.float32)
    points = raw[:24]
    opp_points = raw[24:48]
    bar, off, opp_bar, opp_off = raw[48:52]

    blots = (points == 1.0).astype(np.float32)
    opp_blots = (opp_points == 1.0).astype(np.float32)
    anchors = (points >= 2.0).astype(np.float32)
    opp_anchors = (opp_points >= 2.0).astype(np.float32)
    hit_prob_mine, cover_prob_mine = _prob_vectors(points, opp_points, int(bar), int(opp_bar))

    rev_points = opp_points[::-1].copy()
    rev_opp_points = points[::-1].copy()
    hit_prob_opp_rev, cover_prob_opp_rev = _prob_vectors(rev_points, rev_opp_points, int(opp_bar), int(bar))
    hit_prob_opp = hit_prob_opp_rev[::-1]
    cover_prob_opp = cover_prob_opp_rev[::-1]

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
    mine_all_in_home = float(bar == 0.0 and np.sum(points[6:]) == 0.0)
    opp_all_in_home = float(opp_bar == 0.0 and np.sum(opp_points[:18]) == 0.0)
    mine_last = int(np.max(np.where(points > 0.0)[0])) if np.any(points > 0.0) else -1
    opp_last = int(np.min(np.where(opp_points > 0.0)[0])) if np.any(opp_points > 0.0) else POINTS_DIM
    race_stage_no_hit = float(bar == 0.0 and opp_bar == 0.0 and mine_last < opp_last)

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
            mine_all_in_home,
            opp_all_in_home,
            race_stage_no_hit,
        ],
        dtype=np.float32,
    )
    white_score = float(raw[RAW_WHITE_SCORE]) if raw.shape[0] > RAW_WHITE_SCORE else 0.0
    black_score = float(raw[RAW_BLACK_SCORE]) if raw.shape[0] > RAW_BLACK_SCORE else 0.0
    white_to_move = bool(raw.shape[0] > RAW_WHITE_TO_MOVE and raw[RAW_WHITE_TO_MOVE] > 0.5)
    if white_to_move:
        mine_score = white_score
        opp_score = black_score
    else:
        mine_score = black_score
        opp_score = white_score
    dave_value = float(raw[RAW_DAVE_VALUE]) if raw.shape[0] > RAW_DAVE_VALUE else 1.0
    n_games = float(raw[RAW_N_GAMES]) if raw.shape[0] > RAW_N_GAMES else 11.0
    cube_available_mine = float(raw[RAW_CUBE_AVAILABLE_MINE]) if raw.shape[0] > RAW_CUBE_AVAILABLE_MINE else 0.0
    cube_available_opp = float(raw[RAW_CUBE_AVAILABLE_OPP]) if raw.shape[0] > RAW_CUBE_AVAILABLE_OPP else 0.0
    my_left = 1.0 if n_games < 0 else max(n_games - mine_score, 0.0)
    opp_left = 1.0 if n_games < 0 else max(n_games - opp_score, 0.0)
    is_crawford_game = float(raw[RAW_IS_CRAWFORD_GAME]) if raw.shape[0] > RAW_IS_CRAWFORD_GAME else 0.0
    double_offered = float(raw[RAW_DOUBLE_OFFERED]) if raw.shape[0] > RAW_DOUBLE_OFFERED else 0.0
    # Scalar order is the C++ order from bg/src/obs.cpp.
    scalars = np.concatenate(
        [
            scalars[:14],
            np.asarray(
                [
                    mine_score,
                    opp_score,
                    dave_value,
                    my_left,
                    opp_left,
                    cube_available_mine,
                    cube_available_opp,
                    is_crawford_game,
                    double_offered,
                    mine_all_in_home,
                    opp_all_in_home,
                    race_stage_no_hit,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return np.concatenate(
        [
            points,
            opp_points,
            blots,
            opp_blots,
            anchors,
            opp_anchors,
            hit_prob_mine,
            cover_prob_mine,
            hit_prob_opp,
            cover_prob_opp,
            scalars,
        ],
        dtype=np.float32,
    )
