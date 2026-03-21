import sys
import json
import sqlite3
from pathlib import Path
from typing import Optional
import numpy as np
import pygame

import bg_env  # pybind11 module
from training.agents import build_trainable_agents, get_double_hint_metrics
from training.config import ExperimentConfig
from training.league import ConservativeBaselineAgent
from training.observation import state_to_observation


# ----------------- raw decode -----------------
def decode_raw(raw: np.ndarray):
    raw = np.asarray(raw)
    mine = raw[0:24].astype(int)
    opp = raw[24:48].astype(int)
    mine_bar, mine_off, opp_bar, opp_off, ply = map(int, raw[48:53])
    white_score, black_score, dave_value, n_games, white_to_move = map(int, raw[53:58])
    return mine, opp, mine_bar, mine_off, opp_bar, opp_off, ply, white_score, black_score, dave_value, n_games, white_to_move


def transform_point_for_display(pt: int, turn_white: bool) -> int:
    if pt == 255:
        return 255
    if 1 <= pt <= 24 and not turn_white:
        return 25 - int(pt)
    return int(pt)


def move_to_str(mv8: np.ndarray, turn_white: bool = True):
    mv8 = np.asarray(mv8).astype(int)
    steps = []
    for k in range(4):
        fr = transform_point_for_display(mv8[2 * k], turn_white)
        to = transform_point_for_display(mv8[2 * k + 1], turn_white)
        if fr == 255 or to == 255:
            continue
        steps.append(f"{fr}->{to}")
    return " | ".join(steps) if steps else "(empty)"


def map_move_to_display(mv8: np.ndarray, turn_white: bool):
    mv = np.asarray(mv8).astype(np.uint8).copy()
    for k in range(4):
        fr = int(mv[2 * k])
        to = int(mv[2 * k + 1])
        if fr == 255 or to == 255:
            continue
        mv[2 * k] = np.uint8(transform_point_for_display(fr, turn_white))
        mv[2 * k + 1] = np.uint8(transform_point_for_display(to, turn_white))
    return mv


W, H = 1320, 720
FPS = 60
PANEL_W = 420
BOARD_W = W - PANEL_W
FONT_NAME = None

APP_BG = (255, 255, 255)
BOARD_BG = (242, 224, 194)
FRAME = (169, 129, 91)
TRI_A = (196, 132, 78)
TRI_B = (148, 96, 60)
WHITE = (245, 245, 245)
BLACK = (35, 35, 35)
OUTLINE = (10, 10, 10)
WHITE_OUTLINE = (156, 121, 86)
BROWN_DIE = (62, 40, 24)
TEXT = (20, 20, 20)
SUBTEXT = (60, 60, 60)
HEADER_BG = (248, 238, 220)
DIV = (128, 102, 76)
ACCENT = (214, 143, 45)
SUCCESS = (126, 191, 114)

MARGIN = 18
GAP = 0
BAR_W = 52
OFF_W = BAR_W
OFF_GAP = 0
CUBE_LANE_W = 110
HEADER_H = 90
TOP = HEADER_H + 30
BOTTOM = H - 40
MID_Y = (TOP + BOTTOM) // 2
PLAY_W = BOARD_W - OFF_W - OFF_GAP - CUBE_LANE_W
POINT_W = (PLAY_W - 2 * MARGIN - BAR_W - GAP * 2) // 12
POINT_H = (BOTTOM - TOP - GAP) // 2
CHECKER_R = min(POINT_W // 2 - 2, 18)
STACK_DY = CHECKER_R * 2 - 4
TRI_MARGIN = 10
DICE_SIZE = 42
DICE_GAP = 12
LEGAL_MOVES_UNIQUE = True
CUBE_SIZE = 52

OBS_BASE_HIT_SELF = 144
OBS_BASE_COVER_SELF = 168
OBS_BASE_HIT_OPP = 192
OBS_BASE_COVER_OPP = 216
OBS_POINTS = 24
MATCH_VECTOR_DIM = 12
REWARD_VECTOR_DIM = 6
REWARD_VALUES = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float32)

# Viewer hyperparameters
agent_mode = "hint"  # "none" | "hint" | "play" | "replay"
viewer_n_games = 100  # number of games in endless mode or points to win the match in regular mode
viewer_endless_mode = True
agent_id = "trainable_1"
agent_epoch = 332
agent_checkpoint_dir = "training_stats/checkpoints"
replay_storage_dir = "training_stats/replay"
replay_match_id = None
replay_game_number_in_match = None

RAW_STATE_FULL_DIM = 69


def _require_full_state69(state: np.ndarray) -> np.ndarray:
    a = np.asarray(state, dtype=np.int16).reshape(-1)
    if a.size != RAW_STATE_FULL_DIM:
        raise ValueError(f"Expected full state length {RAW_STATE_FULL_DIM}, got {a.size}")
    return a


def load_replay_steps(storage_dir: str, match_id: str, game_number_in_match: int) -> list[dict]:
    db_path = Path(storage_dir) / "replay.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(f"Replay DB not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT step_index, action_meta
            FROM replay
            WHERE game_id = ? AND COALESCE(game_number_in_match, 1) = ?
            ORDER BY recency_index ASC, step_index ASC
            """,
            (match_id, int(game_number_in_match)),
        ).fetchall()
    out = []
    for step_index, action_meta_json in rows:
        meta = {}
        if action_meta_json:
            try:
                meta = json.loads(action_meta_json)
            except Exception:
                meta = {}
        out.append({"step_index": int(step_index), "action_meta": meta})
    return out




def _set_ui_dice_from_values(dice: list[int]):
    d = [int(x) for x in dice]
    if len(d) != 2:
        return [], [], [], False
    if d[0] == d[1]:
        values = [d[0], d[1]]
        required = [2, 2]
    else:
        values = sorted(d, reverse=True)
        required = [1, 1]
    used = [0] * len(values)
    return values, used, required, True


def _agent_index_from_id(agent_id: str) -> int:
    if not agent_id.startswith("trainable_"):
        raise ValueError(f"Unsupported trainable agent_id={agent_id!r}. Expected trainable_N.")
    return int(agent_id.split("_", 1)[1])


def load_eval_agent(agent_id: str, agent_epoch: int, checkpoint_dir: str):
    if agent_id == "conservative_baseline":
        return ConservativeBaselineAgent()

    ckpt = Path(checkpoint_dir) / f"epoch_{agent_epoch:04d}"
    cfg_data = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig.from_dict(cfg_data)
    agents = build_trainable_agents(cfg, cfg.train.seed)
    states = json.loads((ckpt / "agents.json").read_text(encoding="utf-8"))
    for a, s in zip(agents, states):
        a.load_state_dict(s)
    idx = _agent_index_from_id(agent_id)
    if not (0 <= idx < len(agents)):
        raise ValueError(f"agent_id index out of range: {agent_id!r}")
    return agents[idx]


def evaluate_moves(env, moves: np.ndarray, agent, turn_white: bool):
    def snapshot(e):
        if hasattr(e, "get_state_full"):
            return np.asarray(e.get_state_full(), dtype=np.int16)
        return np.asarray(e.get_state_raw(), dtype=np.int16)

    def restore(e, state):
        a = np.asarray(state, dtype=np.int16).reshape(-1)
        e.set_state_full(_require_full_state69(a))

    if agent is None:
        return [
            (i, np.asarray(mv, dtype=np.uint8), None, None)
            for i, mv in enumerate(sorted_moves_for_panel(moves, turn_white=turn_white))
        ]

    if getattr(agent, "agent_id", "") == "conservative_baseline":
        if len(moves) == 0:
            return []
        state0 = snapshot(env)
        sim = bg_env.Env(0)
        scored = []
        for i, mv in enumerate(moves):
            restore(sim, state0)
            sim.step_move(np.asarray(mv, dtype=np.uint8), 0, 1)
            post = np.asarray(sim.get_state_raw(), dtype=np.int16)
            score = agent._score_move(state0, post)
            scored.append((i, np.asarray(mv, dtype=np.uint8), score))

        scored.sort(key=lambda x: (tuple(-v for v in x[2]), tuple(int(v) for v in x[1].tolist())))
        return [(i, mv, None, None) for i, mv, _ in scored]

    sim = bg_env.Env(0)
    state0 = snapshot(env)
    result = []
    for i, mv in enumerate(moves):
        restore(sim, state0)
        reward, _, _, done = sim.step_move(np.asarray(mv, dtype=np.uint8), 0, 1)
        if done:
            value = 1.0
            reward_vec_swapped = _reward_vector_for_terminal(float(reward))
        else:
            raw = np.asarray(sim.get_state_raw(), dtype=np.float32)
            obs = state_to_observation(raw).reshape(1, -1)
            pred = np.asarray(agent.predict_proba(obs), dtype=np.float32).reshape(-1)
            value = 1.0 - float(pred[0])
            reward_head = pred[MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]
            reward_vec_swapped = _swap_reward_vector_perspective(reward_head)
        result.append((i, np.asarray(mv, dtype=np.uint8), float(value), reward_vec_swapped))
    result.sort(
        key=lambda x: (
            -float(np.dot(REWARD_VALUES, np.asarray(x[3], dtype=np.float32))) if x[3] is not None else -x[2],
            tuple(int(v) for v in x[1].tolist()),
        )
    )
    return result


def _swap_reward_vector_perspective(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != REWARD_VECTOR_DIM:
        return arr
    return arr[::-1].copy()


def _reward_vector_for_terminal(reward_for_mover: float) -> np.ndarray:
    reward_for_opponent = -float(reward_for_mover)
    idx = int(np.argmin(np.abs(REWARD_VALUES - reward_for_opponent)))
    out = np.zeros((REWARD_VECTOR_DIM,), dtype=np.float32)
    out[idx] = 1.0
    return out


def _format_vec_percent(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join(f"{(float(v) * 100.0):.1f}" for v in arr.tolist()) + "]"


def _agent_hint_lines(agent, raw_state: np.ndarray, raw_after_selected_move: np.ndarray, dice_values: list[int]) -> list[str]:
    if agent is None or getattr(agent, "agent_id", "") == "conservative_baseline":
        return []

    endless = int(np.asarray(raw_state, dtype=np.int16).reshape(-1)[56]) < 0 if np.asarray(raw_state).size > 56 else False
    obs_now = state_to_observation(np.asarray(raw_state, dtype=np.float32))
    obs_after = state_to_observation(np.asarray(raw_after_selected_move, dtype=np.float32))
    m = get_double_hint_metrics(agent, obs_now, obs_after, endless=endless)
    line0 = f"Кубики: {dice_values if dice_values else '-'}"
    line1 = f"R6={_format_vec_percent(m.reward_vec)} | EV(noD)={m.exp_no_double:.3f} | EV(D)={m.exp_double:.3f} | P(acc)={(m.p_accept * 100.0):.1f}%"
    line2 = f"postR6={_format_vec_percent(m.reward_vec_after_move)} | EV(rej)={m.exp_reject:.3f} | EV(acc)={m.exp_accept:.3f}"
    line3 = f"Удв: {'Да' if m.apply_double else 'Нет'}. Прин: {'Да' if m.accept_double else 'Нет'}"
    return [line0, line1, line2, line3]


def draw_text(surf, font, text, x, y, color=TEXT):
    img = font.render(text, True, color)
    surf.blit(img, (x, y))


def point_x(idx: int) -> int:
    left_start = MARGIN
    right_start = MARGIN + 6 * POINT_W + GAP + BAR_W + GAP

    def col_center(start_x, col):
        return start_x + col * POINT_W + POINT_W // 2

    col = idx - 12 if 12 <= idx <= 23 else 11 - idx
    return col_center(left_start, col) if col <= 5 else col_center(right_start, col - 6)


def point_is_top(idx: int) -> bool:
    return 12 <= idx <= 23


def point_base_y(idx: int) -> int:
    if point_is_top(idx):
        return TOP + TRI_MARGIN + CHECKER_R
    return BOTTOM - 8 - CHECKER_R


def stack_step(count: int) -> float:
    if count <= 1:
        return 0.0
    if count <= 7:
        return float(STACK_DY)
    return float((6 * STACK_DY) / (count - 1))


def stack_y_positions(base: int, count: int, top_side: bool):
    step = stack_step(count)
    out = []
    for k in range(count):
        y = base + (k * step if top_side else -k * step)
        out.append(int(round(y)))
    return out


def draw_triangle(surface, x_center, y_top, height, upward: bool, color):
    half = POINT_W // 2 - 2
    if upward:
        pts = [(x_center - half, y_top + height), (x_center + half, y_top + height), (x_center, y_top)]
    else:
        pts = [(x_center - half, y_top), (x_center + half, y_top), (x_center, y_top + height)]
    pygame.draw.polygon(surface, color, pts)


def draw_checker(surface, x, y, is_white: bool):
    fill = WHITE if is_white else BLACK
    outline = WHITE_OUTLINE if is_white else OUTLINE
    pygame.draw.circle(surface, fill, (x, y), CHECKER_R)
    pygame.draw.circle(surface, outline, (x, y), CHECKER_R, 2)




def checker_position_for_state(state, disp_point, is_white):
    mine, opp, mine_bar, mine_off, opp_bar, opp_off, _, *_ = decode_raw(state)
    stack = mine if is_white else opp
    if disp_point == "BAR":
        count = int(mine_bar if is_white else opp_bar)
        center_gap = 10
        y0 = MID_Y - center_gap - CHECKER_R if is_white else MID_Y + center_gap + CHECKER_R
        positions = stack_y_positions(y0, count, top_side=not is_white)
        return (MARGIN + 6 * POINT_W + GAP + BAR_W // 2, positions[-1] if positions else y0)
    if disp_point == "OFF":
        off_x0 = PLAY_W + OFF_GAP
        if is_white:
            rect = pygame.Rect(off_x0, MID_Y + 3, OFF_W, BOTTOM - MID_Y - 3)
            count = int(mine_off)
            y = rect.top + 10 + max(0, count - 1) * 6
        else:
            rect = pygame.Rect(off_x0, TOP, OFF_W, MID_Y - TOP - 3)
            count = int(opp_off)
            y = rect.bottom - 10 - max(0, count - 1) * 6
        return (rect.centerx, int(y))
    idx = 24 - int(disp_point)
    x = point_x(idx)
    top = point_is_top(idx)
    count = int(stack[idx])
    ys = stack_y_positions(point_base_y(idx), count, top)
    return (x, ys[-1] if ys else point_base_y(idx))


def lerp(a, b, t):
    return a + (b - a) * t

def draw_die(surface, rect: pygame.Rect, value: int, active=False, used=False, black_turn=False):
    base_fill = BROWN_DIE if black_turn else (235, 235, 235)
    fill = (130, 130, 130) if used else base_fill
    pygame.draw.rect(surface, fill, rect, border_radius=8)
    pygame.draw.rect(surface, ACCENT if active else OUTLINE, rect, 3 if active else 2, border_radius=8)
    cx, cy = rect.center
    off = rect.width // 4
    pips = {
        1: [(0, 0)],
        2: [(-1, -1), (1, 1)],
        3: [(-1, -1), (0, 0), (1, 1)],
        4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
        6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
    }
    for ox, oy in pips.get(int(value), []):
        pip_color = WHITE if black_turn else OUTLINE
        pygame.draw.circle(surface, pip_color, (cx + ox * off, cy + oy * off), max(3, rect.width // 11))


def current_active_die_idx(used_counts, required_counts):
    for i, used in enumerate(used_counts):
        if used < required_counts[i]:
            return i
    return -1


def resolve_active_die_idx(selected_idx, used_counts, required_counts):
    if selected_idx is not None and 0 <= selected_idx < len(used_counts):
        if used_counts[selected_idx] < required_counts[selected_idx]:
            return selected_idx
    return current_active_die_idx(used_counts, required_counts)


def move_steps_from_mv(mv8, turn_white=True):
    mv8 = np.asarray(mv8).astype(int)
    out = []
    for k in range(4):
        fr = transform_point_for_display(mv8[2 * k], turn_white)
        to = transform_point_for_display(mv8[2 * k + 1], turn_white)
        if fr == 255 or to == 255:
            continue
        out.append((fr, to))
    return out


def normalize_steps(steps):
    def _norm_point(v):
        if isinstance(v, str):
            if v == "BAR":
                return 30
            if v == "OFF":
                return 25
        return int(v)

    return tuple(sorted((_norm_point(fr), _norm_point(to)) for fr, to in steps))


def matching_move_indices(moves, manual_steps, turn_white):
    target = normalize_steps(manual_steps)
    target_len = len(manual_steps)
    result = []
    for i, mv in enumerate(moves):
        mv_steps = move_steps_from_mv(mv, turn_white=turn_white)
        if len(mv_steps) != target_len:
            continue
        if normalize_steps(mv_steps) == target:
            result.append(i)
    return result


def max_micro_steps_in_moves(moves, turn_white):
    if len(moves) == 0:
        return 0
    return max(len(move_steps_from_mv(mv, turn_white=turn_white)) for mv in moves)







def first_micro_step_from_env(mv8, turn_white):
    mv = np.asarray(mv8).astype(int)
    fr = int(mv[0])
    to = int(mv[1])
    if fr == 255 or to == 255:
        return -1

    fr_display = transform_point_for_display(fr, turn_white)
    if 1 <= fr_display <= 24 and not turn_white:
        return 25 - fr_display
    return fr_display


def sorted_moves_for_panel(moves, turn_white):
    return sorted(
        [np.asarray(mv).astype(np.uint8) for mv in moves],
        key=lambda mv: (-first_micro_step_from_env(mv, turn_white), tuple(int(x) for x in mv.tolist())),
    )

def draw_undo_icon(surface, rect: pygame.Rect):
    color = (235, 235, 235)
    y = rect.centery
    x0 = rect.x + 7
    x1 = rect.right - 7
    pygame.draw.line(surface, color, (x0 + 6, y), (x1, y), 3)
    pygame.draw.polygon(surface, color, [(x0, y), (x0 + 8, y - 6), (x0 + 8, y + 6)])


def draw_check_icon(surface, rect: pygame.Rect, enabled: bool):
    color = (30, 30, 30) if enabled else (160, 160, 160)
    p1 = (rect.x + 7, rect.centery)
    p2 = (rect.x + 13, rect.bottom - 8)
    p3 = (rect.right - 7, rect.y + 8)
    pygame.draw.lines(surface, color, False, [p1, p2, p3], 4)


def draw_button(surface, font, rect: pygame.Rect, label: str, enabled: bool = True, fill=None):
    base = fill if fill is not None else (SUCCESS if enabled else (125, 125, 125))
    pygame.draw.rect(surface, base, rect, border_radius=8)
    pygame.draw.rect(surface, (40, 40, 40), rect, 2, border_radius=8)
    txt = font.render(label, True, (20, 20, 20) if enabled else (70, 70, 70))
    surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))


def draw_doubling_cube(surface, font, rect: pygame.Rect, value: int, clickable: bool):
    fill = (240, 223, 182) if clickable else (218, 204, 176)
    pygame.draw.rect(surface, fill, rect, border_radius=8)
    pygame.draw.rect(surface, FRAME, rect, 2, border_radius=8)
    txt = font.render(str(int(value)), True, TEXT)
    surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))


def draw_board(surface, font, mine, opp, mine_bar, mine_off, opp_bar, opp_off, ply, turn_white,
               dice_values, used_dice, required_dice, active_die_idx, can_submit,
               dave_value, n_games, white_score, black_score,
               cube_owner_visual, cube_clickable, cube_offer_pending,
               cube_offer_to_white, show_roll_button,
               endless_mode=False, endless_white_score=0, endless_black_score=0, endless_game_number=1,
               piece_anim=None, shake_anim=None, cube_shake_anim=None, cube_move_anim=None):
    surface.fill(APP_BG)
    pygame.draw.rect(surface, FRAME, (0, 0, BOARD_W, H))
    inner = pygame.Rect(8, HEADER_H, BOARD_W - 16, H - HEADER_H - 8)
    pygame.draw.rect(surface, BOARD_BG, inner)
    pygame.draw.rect(surface, HEADER_BG, (0, 0, BOARD_W, HEADER_H))
    pygame.draw.line(surface, DIV, (0, HEADER_H), (BOARD_W, HEADER_H), 2)

    draw_text(surface, font, f"turn={ply}", 16, 10)
    if endless_mode:
        draw_text(surface, font, f"endless game #{endless_game_number} | white: {endless_white_score}  black: {endless_black_score}", 150, 10)
    else:
        draw_text(surface, font, f"match to {n_games} | white: {white_score}  black: {black_score}", 150, 10)

    bar_x0 = MARGIN + 6 * POINT_W + GAP
    bar_rect = pygame.Rect(bar_x0, TOP, BAR_W, BOTTOM - TOP)
    bar_off_fill = (226, 200, 166)
    bar_off_border = (148, 120, 88)
    pygame.draw.rect(surface, bar_off_fill, bar_rect)
    pygame.draw.rect(surface, bar_off_border, bar_rect, 2)

    off_x0 = PLAY_W + OFF_GAP
    off_top_rect = pygame.Rect(off_x0, TOP, OFF_W, MID_Y - TOP - 3)
    off_bottom_rect = pygame.Rect(off_x0, MID_Y + 3, OFF_W, BOTTOM - MID_Y - 3)
    pygame.draw.rect(surface, bar_off_fill, off_top_rect)
    pygame.draw.rect(surface, bar_off_fill, off_bottom_rect)
    pygame.draw.rect(surface, bar_off_border, off_top_rect, 2)
    pygame.draw.rect(surface, bar_off_border, off_bottom_rect, 2)

    point_rects = {}
    for idx in range(24):
        x = point_x(idx)
        top = point_is_top(idx)
        col = idx - 12 if top else 11 - idx
        tri_color = (TRI_B if col % 2 == 0 else TRI_A) if top else (TRI_A if col % 2 == 0 else TRI_B)
        if top:
            draw_triangle(surface, x, TOP + 8, POINT_H - 16, upward=False, color=tri_color)
            img = font.render(str(24 - idx), True, SUBTEXT)
            surface.blit(img, (x - img.get_width() // 2, TOP - 28))
            rect = pygame.Rect(x - POINT_W // 2, TOP + 8, POINT_W, POINT_H - 16)
        else:
            draw_triangle(surface, x, MID_Y + 8, POINT_H - 16, upward=True, color=tri_color)
            img = font.render(str(24 - idx), True, SUBTEXT)
            surface.blit(img, (x - img.get_width() // 2, BOTTOM + 6))
            rect = pygame.Rect(x - POINT_W // 2, MID_Y + 8, POINT_W, POINT_H - 16)
        point_rects[idx] = rect

    for idx in range(24):
        x = point_x(idx)
        base = point_base_y(idx)
        top = point_is_top(idx)
        mine_count = int(mine[idx])
        opp_count = int(opp[idx])
        for y in stack_y_positions(base, mine_count, top):
            draw_checker(surface, x, y, is_white=True)
        for y in stack_y_positions(base, opp_count, top):
            draw_checker(surface, x, y, is_white=False)

    center_gap = 10
    for y in stack_y_positions(MID_Y - center_gap - CHECKER_R, int(mine_bar), top_side=False):
        draw_checker(surface, bar_rect.centerx, y, is_white=True)
    for y in stack_y_positions(MID_Y + center_gap + CHECKER_R, int(opp_bar), top_side=True):
        draw_checker(surface, bar_rect.centerx, y, is_white=False)

    def draw_off_stack(rect: pygame.Rect, count: int, is_white: bool, from_center_down: bool):
        if count <= 0:
            return
        piece_h = int(round(6 * 1.8))
        pad = 8
        anchor = rect.top + pad if from_center_down else rect.bottom - pad - piece_h
        avail = max(1, rect.height - 2 * pad - piece_h)
        step = (piece_h + 2) if count <= 1 else min(piece_h + 2, avail / (count - 1))
        for k in range(count):
            y = anchor + (k * step if from_center_down else -k * step)
            piece = pygame.Rect(rect.x + 6, int(round(y)), rect.width - 12, piece_h)
            pygame.draw.rect(surface, WHITE if is_white else BLACK, piece, border_radius=2)
            pygame.draw.rect(surface, OUTLINE, piece, 1, border_radius=2)

    draw_off_stack(off_bottom_rect, int(mine_off), is_white=True, from_center_down=True)
    draw_off_stack(off_top_rect, int(opp_off), is_white=False, from_center_down=False)

    cube_x = off_x0 + OFF_W + 18
    cube_y_center = MID_Y - CUBE_SIZE // 2
    cube_y_top = TOP + 24
    cube_y_bottom = BOTTOM - CUBE_SIZE - 24
    if cube_owner_visual is None:
        base_cube_y = cube_y_center
    else:
        base_cube_y = cube_y_bottom if cube_owner_visual else cube_y_top
    cube_rect = pygame.Rect(cube_x, base_cube_y, CUBE_SIZE, CUBE_SIZE)
    if cube_move_anim is not None:
        t = min(1.0, max(0.0, cube_move_anim["t"]))
        cy = int(round(lerp(cube_move_anim["start"][1], cube_move_anim["end"][1], t)))
        cube_rect = pygame.Rect(cube_x, cy - CUBE_SIZE // 2, CUBE_SIZE, CUBE_SIZE)
    draw_doubling_cube(surface, font, cube_rect, dave_value, cube_clickable)

    if cube_shake_anim is not None:
        t = min(1.0, max(0.0, cube_shake_anim["t"]))
        amp = 8 * (1.0 - t)
        sx = int(round(cube_rect.centerx + np.sin(t * np.pi * 4) * amp))
        sy = cube_rect.centery
        shake_rect = pygame.Rect(0, 0, cube_rect.width, cube_rect.height)
        shake_rect.center = (sx, sy)
        draw_doubling_cube(surface, font, shake_rect, dave_value, False)

    accept_rect = None
    reject_rect = None

    def dice_anchor(white_side):
        x0 = int(BOARD_W * (0.66 if white_side else 0.18))
        return x0, MID_Y - DICE_SIZE // 2

    dx, dy = dice_anchor(turn_white)

    if cube_offer_pending:
        btn_w, btn_h = 160, 46
        rx, ry = dice_anchor(cube_offer_to_white)
        accept_rect = pygame.Rect(rx, ry - (btn_h + 8), btn_w, btn_h)
        reject_rect = pygame.Rect(rx, ry + 8, btn_w, btn_h)
        draw_button(surface, font, accept_rect, "Принять", True, fill=(156, 211, 133))
        draw_button(surface, font, reject_rect, "Отклонить", True, fill=(219, 146, 125))

    dice_rects = []
    if not show_roll_button:
        for i, val in enumerate(dice_values):
            size = DICE_SIZE + (10 if i == active_die_idx else 0)
            rect = pygame.Rect(dx + i * (DICE_SIZE + DICE_GAP), dy + (DICE_SIZE - size) // 2, size, size)
            exhausted = used_dice[i] >= required_dice[i]
            draw_die(surface, rect, val, active=(i == active_die_idx), used=exhausted, black_turn=not turn_white)
            dice_rects.append(rect)

    roll_rect = None
    if show_roll_button:
        roll_rect = pygame.Rect(dx, dy - 2, 160, 46)
        draw_button(surface, font, roll_rect, "Бросить кубы", True, fill=(243, 219, 119))

    undo_rect = pygame.Rect(dx - 72, dy + 6, 30, 30)
    pygame.draw.rect(surface, (110, 110, 130), undo_rect, border_radius=6)
    draw_undo_icon(surface, undo_rect)

    ok_x = (roll_rect.right + 10) if roll_rect is not None else (dx + len(dice_values) * (DICE_SIZE + DICE_GAP) + 8)
    ok_rect = pygame.Rect(ok_x, dy + 6, 30, 30)
    pygame.draw.rect(surface, SUCCESS if can_submit else (80, 80, 80), ok_rect, border_radius=6)
    draw_check_icon(surface, ok_rect, can_submit)

    if piece_anim is not None:
        t = min(1.0, max(0.0, piece_anim["t"]))
        x = int(round(lerp(piece_anim["start"][0], piece_anim["end"][0], t)))
        y = int(round(lerp(piece_anim["start"][1], piece_anim["end"][1], t)))
        draw_checker(surface, x, y, is_white=piece_anim["is_white"])

    if shake_anim is not None:
        t = min(1.0, max(0.0, shake_anim["t"]))
        amp = 8 * (1.0 - t)
        x = int(round(shake_anim["pos"][0] + np.sin(t * np.pi * 4) * amp))
        y = int(round(shake_anim["pos"][1]))
        draw_checker(surface, x, y, is_white=shake_anim["is_white"])

    pygame.draw.line(surface, DIV, (BOARD_W, 0), (BOARD_W, H), 2)
    return point_rects, bar_rect, dice_rects, undo_rect, ok_rect, cube_rect, roll_rect, accept_rect, reject_rect


def _global_indexed_prob_row(vec: np.ndarray) -> list[int]:
    return [int(vec[24 - g] * 36.0) for g in range(24, 0, -1)]


def _format_prob_row(label: str, values: list[int]) -> str:
    return f"{label:>2} " + " ".join(f"{v:>2d}" for v in values)


def draw_panel(surface, font, small_font, tiny_font, moves, info_lines, manual_steps, turn_white, move_hints, selected_hint_idx, prob_table_lines, hint_lines, agent_thinking=False):
    x, y = BOARD_W + 12, 16
    draw_text(surface, font, "Controls:", x, y)
    y += 28
    for line in ["LMB on point - use active die", "R - roll/reset turn", "N - reset env", "ESC - quit"]:
        draw_text(surface, small_font, line, x, y, (40, 40, 40))
        y += 20

    y += 8
    draw_text(surface, font, f"Legal moves: {len(moves)}", x, y)
    y += 24
    draw_text(surface, small_font, f"Manual: {' | '.join(f'{a}->{b}' for a,b in manual_steps) or '(empty)'}", x, y, ACCENT)

    if hint_lines:
        y += 24
        draw_text(surface, small_font, "Hint:", x, y, (25, 25, 25))
        y += 20
        for line in hint_lines:
            draw_text(surface, small_font, line, x, y, (35, 35, 35))
            y += 18

    y += 28
    for line in prob_table_lines:
        draw_text(surface, tiny_font, line, x, y, (45, 45, 45))
        y += 16

    y += 10
    max_lines = (H - y - 150) // 18
    if agent_thinking:
        panel_moves = []
    elif move_hints:
        panel_moves = [mv for _, mv, _, _ in move_hints]
    else:
        panel_moves = sorted_moves_for_panel(moves, turn_white)
    for i in range(min(len(panel_moves), max_lines)):
        steps = move_steps_from_mv(panel_moves[i], turn_white=turn_white)
        line = " | ".join(f"{a}->{b}" for a, b in steps) if steps else "(empty)"
        color = (45, 45, 45)
        if i == selected_hint_idx:
            color = (220, 180, 0)
        if move_hints and i < len(move_hints):
            value_hint = move_hints[i][2]
            vec_hint = move_hints[i][3]
            if vec_hint is not None:
                ev_match = float(np.dot(REWARD_VALUES, np.asarray(vec_hint, dtype=np.float32)))
                line = f"{line}  revR6={_format_vec_percent(vec_hint)} EV={ev_match:.3f}"
            elif value_hint is not None:
                line = f"{line}  1-p={value_hint:.4f}"
        draw_text(surface, small_font, f"[{i:3d}] {line}", x, y, color)
        y += 18

    y = H - 120
    pygame.draw.line(surface, (60, 60, 60), (BOARD_W, y - 8), (W, y - 8), 1)
    for line in info_lines[-5:]:
        draw_text(surface, small_font, line, x, y, (60, 60, 80))
        y += 18


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Backgammon Viewer (manual dice UI)")
    font = pygame.font.Font(FONT_NAME, 22)
    small = pygame.font.Font(FONT_NAME, 16)
    tiny = pygame.font.Font(FONT_NAME, 13)
    clock = pygame.time.Clock()

    env = bg_env.Env(123, n_games=int(viewer_n_games), endless_mode=bool(viewer_endless_mode))
    env.reset()

    replay_steps = []
    replay_idx = 0
    replay_accept_for_next_offer = 1
    if agent_mode == "replay":
        if not replay_match_id or replay_game_number_in_match is None:
            raise ValueError("For replay mode set replay_match_id and replay_game_number_in_match")
        replay_steps = load_replay_steps(replay_storage_dir, str(replay_match_id), int(replay_game_number_in_match))
        if not replay_steps:
            raise ValueError(f"Replay steps are empty for match_id={replay_match_id}, game_number_in_match={replay_game_number_in_match}")

    agent = None
    if agent_mode in ("hint", "play"):
        agent = load_eval_agent(agent_id, agent_epoch, agent_checkpoint_dir)

    def is_white_turn_from_env() -> bool:
        return int(get_env_state()[57]) == 1

    def get_env_state() -> np.ndarray:
        if hasattr(env, "get_state_full"):
            return np.asarray(env.get_state_full(), dtype=np.int16)
        return np.asarray(env.get_state_raw(), dtype=np.int16)

    def set_env_state(state: np.ndarray):
        a = np.asarray(state, dtype=np.int16).reshape(-1)
        env.set_state_full(_require_full_state69(a))


    turn_white = is_white_turn_from_env()
    mode_label = f"endless ({int(viewer_n_games)} games)" if bool(viewer_endless_mode) else f"match to {int(viewer_n_games)}"
    if agent_mode == "replay":
        mode_label = f"replay match {replay_match_id}, game {replay_game_number_in_match}"
        info_lines = [f"Started. Mode: {mode_label}. Press SPACE to play next recorded action."]
    else:
        info_lines = [f"Started. Mode: {mode_label}. Click points (or bar) to move checkers."]
    endless_white_score = 0
    endless_black_score = 0
    endless_game_number = 1

    def on_endless_game_finished(mover_was_white: bool, scored_points: int):
        nonlocal endless_white_score, endless_black_score, endless_game_number
        if scored_points <= 0:
            return
        if mover_was_white:
            endless_white_score += int(scored_points)
            endless_black_score -= int(scored_points)
        else:
            endless_white_score -= int(scored_points)
            endless_black_score += int(scored_points)
        endless_game_number += 1

    def refresh_moves():
        double_possible, arr = env.legal_moves(LEGAL_MOVES_UNIQUE)
        mv = np.asarray(arr, dtype=np.uint8)
        if mv.ndim == 1:
            mv = mv.reshape(0, 8)
        return int(double_possible), mv

    def roll_current_dice():
        nonlocal dice_values, used_dice, required_dice, manual_steps, history, selected_die_idx, turn_start_state, moves, turn_move_hints, selected_hint_idx, macro_pending_submit, dice_rolled, cube_deactivated_for_turn, hint_pending, hint_pending_started
        d = list(map(int, env.roll_dice()))
        if d[0] == d[1]:
            dice_values = [d[0], d[1]]
            required_dice = [2, 2]
        else:
            dice_values = sorted(d, reverse=True)
            required_dice = [1, 1]
        used_dice = [0] * len(dice_values)
        manual_steps = []
        history = []
        env.set_dice(np.asarray(d, dtype=np.uint8))
        selected_die_idx = 0
        turn_start_state = get_env_state()
        _, moves = refresh_moves()
        turn_move_hints = evaluate_moves(env, moves, agent, turn_white)
        selected_hint_idx = 0
        macro_pending_submit = False
        dice_rolled = True
        cube_deactivated_for_turn = True
        if agent_mode == "play" and not turn_white:
            info_lines.append(f"Opponent dice: {dice_values}")

    def sync_cube_visual_from_env(state: Optional[np.ndarray] = None):
        nonlocal cube_owner_visual
        st = get_env_state() if state is None else np.asarray(state, dtype=np.int16)
        if st.shape[0] >= 64:
            owner = int(st[63])
            cube_owner_visual = None if owner < 0 else bool(owner == 0)

    def start_turn():
        nonlocal dice_values, used_dice, required_dice, manual_steps, history, selected_die_idx, turn_start_state, moves, turn_move_hints, selected_hint_idx, macro_pending_submit, double_possible, dice_rolled, cube_deactivated_for_turn, hint_pending, hint_pending_started
        manual_steps = []
        history = []
        selected_die_idx = 0
        turn_start_state = get_env_state()
        sync_cube_visual_from_env(turn_start_state)
        double_possible, _ = refresh_moves()
        is_opponent_turn_local = agent_mode == "play" and not turn_white
        if double_possible and not is_opponent_turn_local:
            dice_values, used_dice, required_dice = [], [], []
            moves = np.empty((0, 8), dtype=np.uint8)
            turn_move_hints = []
            selected_hint_idx = 0
            macro_pending_submit = False
            dice_rolled = False
            cube_deactivated_for_turn = False
            hint_pending = True
            hint_pending_started = False
        else:
            roll_current_dice()

    def infer_die_index_for_step(env_from: int, env_to: int, used_counts: list[int]) -> int:
        remaining = [i for i in range(len(dice_values)) if used_counts[i] < required_dice[i]]
        if not remaining:
            return -1
        if env_from == 30:
            need = int(env_to)
        elif env_to == 25:
            need = int(abs(25 - env_from))
        else:
            need = int(abs(env_to - env_from))
        for i in remaining:
            if dice_values[i] == need:
                return i
        return remaining[0]

    def cube_center_for_owner(owner_white):
        cube_x = PLAY_W + OFF_GAP + OFF_W + 18 + CUBE_SIZE // 2
        if owner_white is None:
            cube_y = MID_Y
        else:
            cube_y = BOTTOM - CUBE_SIZE // 2 - 24 if owner_white else TOP + CUBE_SIZE // 2 + 24
        return cube_x, cube_y

    def apply_committed_move(
        chosen_mv: np.ndarray,
        value_hint=None,
        use_current_state: bool = False,
        apply_double: int = 0,
        accept_double: int = 1,
    ):
        nonlocal turn_white, macro_pending_submit, dice_rolled, moves, selected_hint_idx
        mover_was_white = bool(turn_white)
        mv = np.asarray(chosen_mv, dtype=np.uint8)
        if use_current_state:
            mv = np.full((8,), 255, dtype=np.uint8)
        else:
            set_env_state(turn_start_state)
        dave_before = int(turn_start_state[55]) if turn_start_state.shape[0] > 55 else 1
        reward, dave_after, accepted, done_code = env.step_move(
            mv,
            apply_double=int(apply_double),
            accept_double=int(accept_double),
        )
        if int(n_games) <= 0 and int(done_code) in (1, 2):
            scored_points = int(round(float(reward))) * max(1, int(dave_before))
            on_endless_game_finished(mover_was_white, scored_points)
        value_suffix = f" | 1-p={value_hint:.4f}" if value_hint is not None else ""
        info_lines.append(
            f"Apply: {move_to_str(chosen_mv, turn_white=turn_white)} | r={reward} dave={dave_after} acc={int(accepted)} done={done_code}{value_suffix}"
        )
        if int(apply_double) and int(accepted) == 0 and int(done_code) in (1, 2):
            info_lines.append(f"Double offer was rejected/effective loss. Match/game finished with r={reward}, dave={dave_after}, done={done_code}.")
        if done_code == 2:
            env.reset()
        turn_white = is_white_turn_from_env()
        if agent_mode == "replay":
            manual_steps.clear()
            history.clear()
            if 0 <= replay_idx < len(replay_steps):
                next_meta = replay_steps[replay_idx].get("action_meta", {})
                dice_values[:], used_dice[:], required_dice[:], next_rolled = _set_ui_dice_from_values(list(next_meta.get("dice", [])))
                dice_rolled = bool(next_rolled)
            else:
                dice_values[:] = []
                used_dice[:] = []
                required_dice[:] = []
                dice_rolled = False
            moves = np.empty((0, 8), dtype=np.uint8)
            turn_move_hints.clear()
            selected_hint_idx = 0
        else:
            start_turn()
        macro_pending_submit = False

    def apply_replay_step():
        nonlocal replay_idx, turn_white, turn_start_state, dice_values, used_dice, required_dice, dice_rolled, cube_move_anim, cube_owner_visual, replay_accept_for_next_offer
        if replay_idx >= len(replay_steps):
            info_lines.append("Replay finished.")
            return
        if piece_anim is not None or macro_anim_steps:
            info_lines.append("Replay animation in progress. Wait until it finishes.")
            return
        meta = replay_steps[replay_idx].get("action_meta", {})
        raw_state = np.asarray(meta.get("raw_state", []), dtype=np.int16)
        if raw_state.size != RAW_STATE_FULL_DIM:
            raise ValueError(f"Replay state has invalid length {raw_state.size}; expected {RAW_STATE_FULL_DIM}")
        set_env_state(raw_state)
        turn_white = is_white_turn_from_env()
        turn_start_state = get_env_state()

        dice = list(meta.get("dice", []))
        dice_values, used_dice, required_dice, dice_rolled = _set_ui_dice_from_values(dice)
        if dice_rolled:
            env.set_dice(np.asarray(dice, dtype=np.uint8))

        move = np.asarray(meta.get("move", [255] * 8), dtype=np.uint8)
        apply_double = int(meta.get("apply_double", 0))
        accept_double_next = int(meta.get("accept_double_for_next_offer", 1))

        accept_for_current_offer = int(replay_accept_for_next_offer)
        if apply_double:
            offer_step = int(meta.get("step_index", replay_idx))
            if accept_for_current_offer:
                start_pos = cube_center_for_owner(cube_owner_visual)
                end_pos = cube_center_for_owner(not turn_white)
                cube_owner_visual = bool(not turn_white)
                cube_move_anim = {
                    "start": start_pos,
                    "end": end_pos,
                    "start_time": pygame.time.get_ticks() / 1000.0,
                    "t": 0.0,
                }
                info_lines.append(
                    f"Replay: double on step {offer_step} accepted by {'white' if (not turn_white) else 'black'} (decision from previous step)."
                )
            else:
                rej_reward = meta.get("reward", None)
                rej_done = meta.get("done_code", None)
                info_lines.append(
                    f"Replay: double on step {offer_step} rejected by {'white' if (not turn_white) else 'black'} (reward={rej_reward}, done={rej_done})."
                )

        info_lines.append(f"Replay step {replay_idx + 1}/{len(replay_steps)}: dice={dice} move={move_to_str(move, turn_white)}")
        start_macro_animation(
            move,
            None,
            auto_commit=True,
            commit_apply_double=apply_double,
            commit_accept_double=accept_double_next,
        )
        replay_idx += 1
        replay_accept_for_next_offer = accept_double_next

    dice_values, used_dice, required_dice, manual_steps, history = [], [], [], [], []
    selected_die_idx = 0
    selected_hint_idx = 0
    turn_start_state = get_env_state()
    moves = np.empty((0, 8), dtype=np.uint8)
    double_possible = 0
    dice_rolled = True
    cube_deactivated_for_turn = False
    cube_owner_visual = None
    cube_offer_pending = False
    cube_offer_from_white = None
    cube_offer_to_white = False
    cube_shake_anim = None
    cube_move_anim = None
    turn_move_hints = []
    hint_lines_cache = []
    hint_key_prev = None
    hint_pending = True
    hint_pending_started = False
    if agent_mode == "replay":
        first_meta = replay_steps[0].get("action_meta", {}) if replay_steps else {}
        first_raw = np.asarray(first_meta.get("raw_state", []), dtype=np.int16)
        if first_raw.size != RAW_STATE_FULL_DIM:
            raise ValueError(f"Replay initial state has invalid length {first_raw.size}; expected {RAW_STATE_FULL_DIM}")
        set_env_state(first_raw)
        turn_white = is_white_turn_from_env()
        turn_start_state = get_env_state()
        first_dice = list(first_meta.get("dice", []))
        dice_values, used_dice, required_dice, dice_rolled = _set_ui_dice_from_values(first_dice)
        info_lines.append(f"Loaded replay start. Steps: {len(replay_steps)}")
    else:
        start_turn()
        info_lines.append(f"First turn: {'white' if turn_white else 'black'}")

    piece_anim = None
    shake_anim = None
    macro_anim_steps = []
    macro_anim_idx = 0
    macro_anim_turn_white = True
    macro_anim_value = None
    macro_anim_move = np.full((8,), 255, dtype=np.uint8)
    macro_anim_auto_commit = True
    macro_pending_submit = False
    macro_anim_history = []
    macro_anim_manual_steps = []
    macro_anim_used_dice = []
    macro_anim_commit_apply_double = 0
    macro_anim_commit_accept_double = 1

    def start_macro_animation(
        chosen_mv: np.ndarray,
        chosen_value,
        auto_commit: bool = True,
        commit_apply_double: int = 0,
        commit_accept_double: int = 1,
    ):
        nonlocal macro_anim_steps, macro_anim_idx, macro_anim_turn_white, macro_anim_value, macro_anim_move, piece_anim, macro_anim_auto_commit, macro_pending_submit, manual_steps, history, used_dice, macro_anim_history, macro_anim_manual_steps, macro_anim_used_dice, macro_anim_commit_apply_double, macro_anim_commit_accept_double
        mv = np.asarray(chosen_mv, dtype=np.uint8)
        steps = []
        for k in range(4):
            fr = int(mv[2 * k])
            to = int(mv[2 * k + 1])
            if fr == 255 or to == 255:
                continue
            steps.append((fr, to))
        macro_anim_steps = steps
        macro_anim_idx = 0
        macro_anim_turn_white = turn_white
        macro_anim_value = chosen_value
        macro_anim_move = mv.copy()
        macro_anim_auto_commit = bool(auto_commit)
        macro_pending_submit = False
        macro_anim_history = []
        macro_anim_manual_steps = []
        macro_anim_used_dice = [0] * len(dice_values)
        macro_anim_commit_apply_double = int(commit_apply_double)
        macro_anim_commit_accept_double = int(commit_accept_double)
        if not auto_commit:
            manual_steps = []
            history = []
            used_dice = [0] * len(dice_values)
        set_env_state(turn_start_state)
        if not macro_anim_steps:
            if auto_commit:
                apply_committed_move(
                    np.asarray(macro_anim_move, dtype=np.uint8),
                    macro_anim_value,
                    use_current_state=False,
                    apply_double=macro_anim_commit_apply_double,
                    accept_double=macro_anim_commit_accept_double,
                )
            return
        prev_state = get_env_state()
        env_from, env_to = macro_anim_steps[macro_anim_idx]
        die_idx = infer_die_index_for_step(env_from, env_to, macro_anim_used_dice)
        die_value = int(dice_values[die_idx]) if die_idx >= 0 and die_idx < len(dice_values) else 0
        if die_value > 0:
            ok, _ = env.apply_micro_step(int(env_from), int(env_to), die_value)
        else:
            ok, _ = env.apply_micro_step(int(env_from), int(env_to))
        if not ok:
            set_env_state(prev_state)
            ok, _ = env.apply_micro_step(int(env_from), int(env_to))
        if not ok:
            macro_anim_steps = []
            info_lines.append("Failed to animate selected macro-step.")
            return
        post_state = get_env_state()
        from_disp = "BAR" if env_from == 30 else transform_point_for_display(env_from, macro_anim_turn_white)
        to_disp = "OFF" if env_to == 25 else transform_point_for_display(env_to, macro_anim_turn_white)
        if die_idx >= 0:
            macro_anim_used_dice[die_idx] += 1
        macro_anim_manual_steps.append((from_disp, to_disp))
        macro_anim_history.append((prev_state.copy(), from_disp, to_disp, die_idx))
        piece_anim = {
            "start": checker_position_for_state(prev_state, from_disp, macro_anim_turn_white),
            "end": checker_position_for_state(post_state, to_disp, macro_anim_turn_white),
            "is_white": macro_anim_turn_white,
            "start_time": pygame.time.get_ticks() / 1000.0,
            "t": 0.0,
        }
        macro_anim_idx += 1
    running = True
    while running:
        clock.tick(FPS)
        raw = get_env_state()
        base_mine, base_opp, mine_bar, base_mine_off, opp_bar, base_opp_off, ply, white_score, black_score, dave_value, n_games, _ = decode_raw(raw)
        endless_mode = int(n_games) < 0
        dave_value_ui = int(dave_value)

        if turn_white:
            white_base, black_base = base_mine.copy(), base_opp.copy()
            white_off, black_off = base_mine_off, base_opp_off
            white_bar, black_bar = mine_bar, opp_bar
        else:
            white_base, black_base = base_opp[::-1].copy(), base_mine[::-1].copy()
            white_off, black_off = base_opp_off, base_mine_off
            white_bar, black_bar = opp_bar, mine_bar

        view_mine, view_opp = white_base.copy(), black_base.copy()
        view_mine_off, view_opp_off = white_off, black_off

        active_idx = resolve_active_die_idx(selected_die_idx, used_dice, required_dice) if dice_rolled else -1
        selected_die_idx = active_idx
        required_steps = max_micro_steps_in_moves(moves, turn_white) if dice_rolled else 0
        is_opponent_turn = agent_mode == "play" and not turn_white
        can_submit = dice_rolled and (is_opponent_turn or len(moves) == 0 or len(manual_steps) == required_steps)
        show_roll_button = (agent_mode != "replay") and (not dice_rolled) and (not cube_offer_pending)
        cube_clickable = (not dice_rolled) and (not cube_deactivated_for_turn) and bool(double_possible) and (not cube_offer_pending) and (not is_opponent_turn)
        move_hints = turn_move_hints
        if move_hints:
            selected_hint_idx = max(0, min(selected_hint_idx, len(move_hints) - 1))
        else:
            selected_hint_idx = 0

        now = pygame.time.get_ticks() / 1000.0
        if piece_anim is not None:
            t = (now - piece_anim["start_time"]) / 0.2
            if t >= 1.0:
                piece_anim = None
            else:
                piece_anim["t"] = t
        if shake_anim is not None:
            t = (now - shake_anim["start_time"]) / 0.2
            if t >= 1.0:
                shake_anim = None
            else:
                shake_anim["t"] = t
        if cube_shake_anim is not None:
            t = (now - cube_shake_anim["start_time"]) / 0.2
            if t >= 1.0:
                cube_shake_anim = None
            else:
                cube_shake_anim["t"] = t
        if cube_move_anim is not None:
            t = (now - cube_move_anim["start_time"]) / 0.25
            if t >= 1.0:
                cube_move_anim = None
            else:
                cube_move_anim["t"] = t

        if piece_anim is None and macro_anim_steps:
            if macro_anim_idx < len(macro_anim_steps):
                prev_state = get_env_state()
                env_from, env_to = macro_anim_steps[macro_anim_idx]
                die_idx = infer_die_index_for_step(env_from, env_to, macro_anim_used_dice)
                die_value = int(dice_values[die_idx]) if die_idx >= 0 and die_idx < len(dice_values) else 0
                if die_value > 0:
                    ok, _ = env.apply_micro_step(int(env_from), int(env_to), die_value)
                else:
                    ok, _ = env.apply_micro_step(int(env_from), int(env_to))
                if not ok:
                    set_env_state(prev_state)
                    ok, _ = env.apply_micro_step(int(env_from), int(env_to))
                if not ok:
                    info_lines.append("Failed to animate selected macro-step.")
                    macro_anim_steps = []
                else:
                    post_state = get_env_state()
                    from_disp = "BAR" if env_from == 30 else transform_point_for_display(env_from, macro_anim_turn_white)
                    to_disp = "OFF" if env_to == 25 else transform_point_for_display(env_to, macro_anim_turn_white)
                    if die_idx >= 0:
                        macro_anim_used_dice[die_idx] += 1
                    macro_anim_manual_steps.append((from_disp, to_disp))
                    macro_anim_history.append((prev_state.copy(), from_disp, to_disp, die_idx))
                    piece_anim = {
                        "start": checker_position_for_state(prev_state, from_disp, macro_anim_turn_white),
                        "end": checker_position_for_state(post_state, to_disp, macro_anim_turn_white),
                        "is_white": macro_anim_turn_white,
                        "start_time": pygame.time.get_ticks() / 1000.0,
                        "t": 0.0,
                    }
                    macro_anim_idx += 1
            else:
                value_suffix = f" | 1-p={macro_anim_value:.4f}" if macro_anim_value is not None else ""
                info_lines.append(f"Macro: {move_to_str(macro_anim_move, turn_white=macro_anim_turn_white)}{value_suffix}")
                if not macro_anim_auto_commit:
                    macro_pending_submit = True
                    manual_steps = list(macro_anim_manual_steps)
                    history = list(macro_anim_history)
                    used_dice = list(macro_anim_used_dice)
                else:
                    apply_committed_move(
                        np.asarray(macro_anim_move, dtype=np.uint8),
                        macro_anim_value,
                        use_current_state=True,
                        apply_double=macro_anim_commit_apply_double,
                        accept_double=macro_anim_commit_accept_double,
                    )
                macro_anim_steps = []

        point_rects, bar_rect, dice_rects, undo_rect, ok_rect, cube_rect, roll_rect, accept_rect, reject_rect = draw_board(
            screen, font, view_mine, view_opp, white_bar, view_mine_off, black_bar, view_opp_off,
            ply, turn_white, dice_values, used_dice, required_dice, active_idx, can_submit,
            dave_value_ui, n_games, white_score, black_score,
            cube_owner_visual=cube_owner_visual, cube_clickable=cube_clickable, cube_offer_pending=cube_offer_pending,
            cube_offer_to_white=cube_offer_to_white, show_roll_button=show_roll_button,
            endless_mode=endless_mode, endless_white_score=endless_white_score,
            endless_black_score=endless_black_score, endless_game_number=endless_game_number,
            piece_anim=piece_anim, shake_anim=shake_anim, cube_shake_anim=cube_shake_anim, cube_move_anim=cube_move_anim
        )
        obs_extended = np.asarray(env.get_obs_extended(), dtype=np.float32)
        sh = _global_indexed_prob_row(obs_extended[OBS_BASE_HIT_SELF:OBS_BASE_HIT_SELF + OBS_POINTS])
        sc = _global_indexed_prob_row(obs_extended[OBS_BASE_COVER_SELF:OBS_BASE_COVER_SELF + OBS_POINTS])
        oh = _global_indexed_prob_row(obs_extended[OBS_BASE_HIT_OPP:OBS_BASE_HIT_OPP + OBS_POINTS])
        oc = _global_indexed_prob_row(obs_extended[OBS_BASE_COVER_OPP:OBS_BASE_COVER_OPP + OBS_POINTS])
        header = "   " + " ".join(f"{g:>2d}" for g in range(24, 0, -1))
        prob_table_lines = [
            header,
            _format_prob_row("SH", sh),
            _format_prob_row("SC", sc),
            _format_prob_row("OH", oh),
            _format_prob_row("OC", oc),
        ]
        agent_thinking = False
        hint_lines = []
        if agent_mode == "hint":
            key = (bytes(np.asarray(raw, dtype=np.int16).tobytes()), int(selected_hint_idx), len(history), bool(can_submit), tuple(int(x) for x in dice_values))
            if key != hint_key_prev:
                hint_key_prev = key
                hint_pending = True
                hint_pending_started = False
            if hint_pending and not hint_pending_started:
                hint_pending_started = True
                agent_thinking = True
                hint_lines = ["Агент думает..."]
            elif hint_pending:
                raw_after_selected = np.asarray(raw, dtype=np.int16)
                if can_submit and len(history) > 0:
                    raw_after_selected = np.asarray(raw, dtype=np.int16)
                elif len(move_hints) > 0:
                    sim_hint = bg_env.Env(0)
                    sim_hint.set_state_raw(np.asarray(turn_start_state, dtype=np.int16))
                    try:
                        sim_hint.step_move(np.asarray(move_hints[selected_hint_idx][1], dtype=np.uint8), 0, 1)
                    except TypeError:
                        sim_hint.step_move(np.asarray(move_hints[selected_hint_idx][1], dtype=np.uint8))
                    raw_after_selected = np.asarray(sim_hint.get_state_raw(), dtype=np.int16)
                hint_lines_cache = _agent_hint_lines(agent, raw, raw_after_selected, dice_values)
                hint_pending = False
                hint_pending_started = False
                hint_lines = hint_lines_cache
            else:
                hint_lines = hint_lines_cache
        draw_panel(
            screen,
            font,
            small,
            tiny,
            moves,
            info_lines,
            manual_steps,
            turn_white,
            ([] if agent_thinking else move_hints),
            selected_hint_idx,
            prob_table_lines,
            hint_lines,
            agent_thinking=agent_thinking,
        )
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif macro_anim_steps:
                    continue
                elif event.key == pygame.K_n and agent_mode != "replay":
                    env.reset()
                    turn_white = is_white_turn_from_env()
                    cube_owner_visual = None
                    cube_offer_pending = False
                    cube_offer_from_white = None
                    cube_offer_to_white = False
                    cube_move_anim = None
                    start_turn()
                    info_lines.append(f"Reset env. First turn: {'white' if turn_white else 'black'}")
                elif event.key == pygame.K_r and agent_mode != "replay":
                    if show_roll_button:
                        roll_current_dice()
                    else:
                        start_turn()
                    info_lines.append(f"Reroll: {dice_values}")
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    selected_hint_idx = min(event.key - pygame.K_0, max(0, len(move_hints) - 1)); hint_pending = True; hint_pending_started = False
                elif event.key == pygame.K_UP:
                    selected_hint_idx = max(0, selected_hint_idx - 1); hint_pending = True; hint_pending_started = False
                elif event.key == pygame.K_DOWN:
                    selected_hint_idx = min(max(0, len(move_hints) - 1), selected_hint_idx + 1); hint_pending = True; hint_pending_started = False
                elif event.key == pygame.K_SPACE:
                    if agent_mode == "replay":
                        apply_replay_step()
                    elif show_roll_button and agent_mode != "replay":
                        roll_current_dice()
                        info_lines.append(f"Roll: {dice_values}")
                    elif is_opponent_turn and can_submit:
                        if len(move_hints) > 0:
                            _, chosen_mv, chosen_v, _ = move_hints[0]
                            value_suffix = f" | v={chosen_v:.4f}" if chosen_v is not None else ""
                            info_lines.append(f"Agent({agent_id}) black: {move_to_str(chosen_mv, turn_white=False)}{value_suffix}")
                            start_macro_animation(chosen_mv, chosen_v)
                        else:
                            info_lines.append(f"Agent({agent_id}) black: (pass)")
                            env.commit_turn()
                            turn_white = not turn_white
                            start_turn()
                    elif agent_mode in ("none", "hint"):
                        if can_submit:
                            if macro_pending_submit:
                                chosen_mv = np.asarray(macro_anim_move, dtype=np.uint8)
                                chosen_v = macro_anim_value
                            elif len(move_hints) > 0:
                                _, chosen_mv, chosen_v, _ = move_hints[selected_hint_idx]
                            else:
                                chosen_mv, chosen_v = np.full((8,), 255, dtype=np.uint8), None
                            apply_committed_move(np.asarray(chosen_mv, dtype=np.uint8), chosen_v, use_current_state=(len(history) > 0))
                        elif len(history) == 0 and len(move_hints) > 0:
                            _, chosen_mv, chosen_v, _ = move_hints[selected_hint_idx]
                            start_macro_animation(chosen_mv, chosen_v, auto_commit=False)
            if macro_anim_steps:
                continue
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                if cube_offer_pending and accept_rect is not None and accept_rect.collidepoint(mx, my):
                    dave_after, accepted, done_code = env.resolve_pending_double(1)
                    if not accepted:
                        info_lines.append("No pending double to accept.")
                        cube_offer_pending = False
                        continue
                    cube_offer_pending = False
                    cube_owner_visual = cube_offer_to_white
                    cube_offer_from_white = None
                    cube_offer_to_white = False
                    cube_deactivated_for_turn = True
                    sync_cube_visual_from_env()
                    cube_move_anim = None
                    info_lines.append("Double accepted.")
                    if done_code == 0:
                        roll_current_dice()
                    continue

                if cube_offer_pending and reject_rect is not None and reject_rect.collidepoint(mx, my):
                    offerer_is_white = bool(cube_offer_from_white)
                    dave_before_reject = int(dave_value_ui)
                    _dave_after, _accepted, done_code = env.resolve_pending_double(0)
                    if endless_mode and agent_mode == "none" and int(done_code) in (1, 2):
                        on_endless_game_finished(offerer_is_white, dave_before_reject)
                    turn_white = is_white_turn_from_env()
                    cube_owner_visual = None
                    cube_offer_pending = False
                    cube_offer_from_white = None
                    cube_offer_to_white = False
                    cube_move_anim = None
                    info_lines.append("Double declined.")
                    if done_code == 2:
                        info_lines.append("Match finished.")
                    start_turn()
                    continue

                if roll_rect is not None and roll_rect.collidepoint(mx, my):
                    roll_current_dice()
                    info_lines.append(f"Roll: {dice_values}")
                    continue

                if cube_rect.collidepoint(mx, my):
                    if cube_clickable:
                        if not env.request_double():
                            cube_shake_anim = {"start_time": pygame.time.get_ticks() / 1000.0, "t": 0.0}
                            continue
                        cube_offer_pending = True
                        cube_offer_from_white = turn_white
                        cube_offer_to_white = not turn_white
                        start_pos = cube_center_for_owner(cube_owner_visual)
                        end_pos = cube_center_for_owner(cube_offer_to_white)
                        cube_owner_visual = cube_offer_to_white
                        cube_move_anim = {
                            "start": start_pos,
                            "end": end_pos,
                            "start_time": pygame.time.get_ticks() / 1000.0,
                            "t": 0.0,
                        }
                        dice_rolled = False
                        dice_values, used_dice, required_dice = [], [], []
                        moves = np.empty((0, 8), dtype=np.uint8)
                        info_lines.append("Double offered. Opponent must accept or reject.")
                    else:
                        cube_shake_anim = {"start_time": pygame.time.get_ticks() / 1000.0, "t": 0.0}
                    continue

                if not dice_rolled:
                    continue

                if undo_rect.collidepoint(mx, my) and history:
                    prev_state, fr, to, die_idx = history.pop()
                    set_env_state(prev_state)
                    manual_steps.pop()
                    if 0 <= die_idx < len(used_dice):
                        used_dice[die_idx] = max(0, used_dice[die_idx] - 1)
                        selected_die_idx = die_idx
                    info_lines.append(f"Undo {fr}->{to}")
                    macro_pending_submit = False
                    continue

                if ok_rect.collidepoint(mx, my) and can_submit:
                    if is_opponent_turn:
                        if len(move_hints) > 0:
                            _, chosen_mv, chosen_v, _ = move_hints[0]
                            value_suffix = f" | v={chosen_v:.4f}" if chosen_v is not None else ""
                            info_lines.append(f"Agent({agent_id}) black: {move_to_str(chosen_mv, turn_white=False)}{value_suffix}")
                            start_macro_animation(chosen_mv, chosen_v)
                        else:
                            info_lines.append(f"Agent({agent_id}) black: (pass)")
                            env.commit_turn()
                            turn_white = not turn_white
                            start_turn()
                        continue

                    if macro_pending_submit:
                        apply_committed_move(np.asarray(macro_anim_move, dtype=np.uint8), macro_anim_value, use_current_state=True)
                        continue

                    chosen_value = None
                    chosen_mv = np.full((8,), 255, dtype=np.uint8)
                    matched = matching_move_indices(moves, manual_steps, turn_white)
                    if matched:
                        idx = matched[0]
                        chosen_mv = np.asarray(moves[idx], dtype=np.uint8)
                        if move_hints and move_hints[0][2] is not None:
                            matched_set = set(matched)
                            for move_i, _, value, _ in move_hints:
                                if move_i in matched_set:
                                    chosen_value = value
                                    break
                    apply_committed_move(chosen_mv, chosen_value, use_current_state=True)
                    continue

                clicked_die = next((i for i, rect in enumerate(dice_rects) if rect.collidepoint(mx, my)), None)
                if clicked_die is not None:
                    if used_dice[clicked_die] < required_dice[clicked_die]:
                        selected_die_idx = clicked_die
                    continue

                if is_opponent_turn:
                    continue

                has_bar_checker = mine_bar > 0
                clicked_bar = bar_rect.collidepoint(mx, my)
                clicked_idx = next((i for i, rect in point_rects.items() if rect.collidepoint(mx, my)), None)
                if clicked_idx is None and not (has_bar_checker and clicked_bar):
                    continue

                if active_idx < 0:
                    continue
                clicked_idx = next((i for i, rect in point_rects.items() if rect.collidepoint(mx, my)), None)
                if clicked_idx is None and not (has_bar_checker and clicked_bar):
                    continue

                if has_bar_checker:
                    if clicked_bar:
                        die_candidates = [active_idx] + [
                            i for i in range(len(dice_values))
                            if i != active_idx and used_dice[i] < required_dice[i]
                        ]
                    else:
                        wanted_die = (24 - clicked_idx) if turn_white else (clicked_idx + 1)
                        die_candidates = [
                            i for i, val in enumerate(dice_values)
                            if val == wanted_die and used_dice[i] < required_dice[i]
                        ]
                        if not die_candidates:
                            info_lines.append(f"No available die for bar entry to point {wanted_die}.")
                            shake_anim = {"pos": checker_position_for_state(raw, "BAR", turn_white), "is_white": turn_white, "start_time": pygame.time.get_ticks() / 1000.0, "t": 0.0}
                            continue

                    prev_state = get_env_state()
                    used_idx = None
                    env_from, env_to = 30, -1
                    for die_idx in die_candidates:
                        attempt_to = int(dice_values[die_idx])
                        ok, _ = env.apply_micro_step(int(env_from), attempt_to)
                        if ok:
                            used_idx = die_idx
                            env_to = attempt_to
                            break
                        set_env_state(prev_state)

                    if used_idx is None:
                        info_lines.append("Invalid bar entry.")
                        shake_anim = {"pos": checker_position_for_state(prev_state, "BAR", turn_white), "is_white": turn_white, "start_time": pygame.time.get_ticks() / 1000.0, "t": 0.0}
                        continue

                    from_disp = "BAR"
                    to_disp = transform_point_for_display(env_to, turn_white)
                    post_state = get_env_state()
                    piece_anim = {"start": checker_position_for_state(prev_state, from_disp, turn_white), "end": checker_position_for_state(post_state, to_disp, turn_white), "is_white": turn_white, "start_time": pygame.time.get_ticks() / 1000.0, "t": 0.0}
                    manual_steps.append((from_disp, to_disp))
                    used_dice[used_idx] += 1
                    selected_die_idx = active_idx
                    history.append((prev_state, from_disp, to_disp, used_idx))
                    macro_pending_submit = False
                    continue

                own = view_mine if turn_white else view_opp
                if own[clicked_idx] <= 0:
                    info_lines.append("No checker of current color on point.")
                    continue

                env_from = (24 - clicked_idx) if turn_white else (clicked_idx + 1)
                active_die = dice_values[active_idx]
                projected_to = env_from + active_die
                off_attempt = projected_to >= 25 or projected_to <= 0
                env_to = 25 if off_attempt else projected_to

                candidate_die_indices = [active_idx]
                if off_attempt:
                    available = [i for i in range(len(dice_values)) if used_dice[i] < required_dice[i]]
                    if available:
                        smallest_idx = min(available, key=lambda i: dice_values[i])
                        smallest_die = dice_values[smallest_idx]
                        off_distance = abs(25 - env_from)
                        if smallest_die < active_die and off_distance <= smallest_die:
                            candidate_die_indices = [smallest_idx, active_idx]
                candidate_die_indices.extend(
                    i for i in range(len(dice_values))
                    if i not in candidate_die_indices and used_dice[i] < required_dice[i]
                )

                prev_state = get_env_state()
                used_idx = None
                for die_idx in candidate_die_indices:
                    attempt_to = 25 if off_attempt else (env_from + dice_values[die_idx])
                    ok, _ = env.apply_micro_step(int(env_from), int(attempt_to), int(dice_values[die_idx]))
                    if ok:
                        used_idx = die_idx
                        env_to = attempt_to
                        break
                    set_env_state(prev_state)

                if used_idx is None:
                    info_lines.append("Invalid micro-step.")
                    shake_anim = {"pos": checker_position_for_state(prev_state, transform_point_for_display(env_from, turn_white), turn_white), "is_white": turn_white, "start_time": pygame.time.get_ticks() / 1000.0, "t": 0.0}
                    continue

                from_disp = transform_point_for_display(env_from, turn_white)
                to_disp = "OFF" if env_to == 25 else transform_point_for_display(env_to, turn_white)
                post_state = get_env_state()
                piece_anim = {"start": checker_position_for_state(prev_state, from_disp, turn_white), "end": checker_position_for_state(post_state, to_disp, turn_white), "is_white": turn_white, "start_time": pygame.time.get_ticks() / 1000.0, "t": 0.0}
                manual_steps.append((from_disp, to_disp))
                used_dice[used_idx] += 1
                selected_die_idx = active_idx
                history.append((prev_state, from_disp, to_disp, used_idx))
                macro_pending_submit = False


    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
