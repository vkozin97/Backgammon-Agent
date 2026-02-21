import sys
import numpy as np
import pygame

import bg_env  # pybind11 module


# ----------------- raw decode -----------------
def decode_raw(raw: np.ndarray):
    raw = np.asarray(raw)
    mine = raw[0:24].astype(int)
    opp = raw[24:48].astype(int)
    mine_bar, mine_off, opp_bar, opp_off, ply = map(int, raw[48:53])
    return mine, opp, mine_bar, mine_off, opp_bar, opp_off, ply


def transform_point_for_display(pt: int, turn_white: bool) -> int:
    if pt == 255:
        return 255
    if 0 <= pt <= 23 and not turn_white:
        return 23 - int(pt)
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


W, H = 1100, 720
FPS = 60
PANEL_W = 420
BOARD_W = W - PANEL_W
FONT_NAME = None

BOARD_BG = (40, 30, 20)
FRAME = (90, 70, 50)
TRI_A = (140, 85, 45)
TRI_B = (80, 45, 25)
WHITE = (245, 245, 245)
BLACK = (25, 25, 25)
OUTLINE = (10, 10, 10)
TEXT = (235, 235, 235)
SUBTEXT = (190, 190, 190)
HEADER_BG = (18, 18, 18)
DIV = (60, 60, 60)
ACCENT = (255, 214, 102)
SUCCESS = (90, 200, 120)

MARGIN = 18
GAP = 20
BAR_W = 52
HEADER_H = 90
TOP = HEADER_H + 30
BOTTOM = H - 40
MID_Y = (TOP + BOTTOM) // 2
POINT_W = (BOARD_W - 2 * MARGIN - BAR_W - GAP * 2) // 12
POINT_H = (BOTTOM - TOP - GAP) // 2
CHECKER_R = min(POINT_W // 2 - 2, 18)
STACK_DY = CHECKER_R * 2 - 4
DICE_SIZE = 42
DICE_GAP = 12


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
    return TOP + CHECKER_R + 6 if point_is_top(idx) else BOTTOM - CHECKER_R - 6


def draw_triangle(surface, x_center, y_top, height, upward: bool, color):
    half = POINT_W // 2 - 2
    if upward:
        pts = [(x_center - half, y_top + height), (x_center + half, y_top + height), (x_center, y_top)]
    else:
        pts = [(x_center - half, y_top), (x_center + half, y_top), (x_center, y_top + height)]
    pygame.draw.polygon(surface, color, pts)


def draw_checker(surface, x, y, is_white: bool):
    fill = WHITE if is_white else BLACK
    pygame.draw.circle(surface, fill, (x, y), CHECKER_R)
    pygame.draw.circle(surface, OUTLINE, (x, y), CHECKER_R, 2)


def draw_die(surface, rect: pygame.Rect, value: int, active=False, used=False):
    fill = (130, 130, 130) if used else (235, 235, 235)
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
        pygame.draw.circle(surface, OUTLINE, (cx + ox * off, cy + oy * off), max(3, rect.width // 11))


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
    return tuple(sorted((int(fr), int(to)) for fr, to in steps))


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


def draw_board(surface, font, mine, opp, mine_bar, mine_off, opp_bar, opp_off, ply, turn_white,
               dice_values, used_dice, required_dice, active_die_idx, can_submit):
    surface.fill((25, 25, 25))
    pygame.draw.rect(surface, FRAME, (0, 0, BOARD_W, H))
    inner = pygame.Rect(8, HEADER_H, BOARD_W - 16, H - HEADER_H - 8)
    pygame.draw.rect(surface, BOARD_BG, inner)
    pygame.draw.rect(surface, HEADER_BG, (0, 0, BOARD_W, HEADER_H))
    pygame.draw.line(surface, DIV, (0, HEADER_H), (BOARD_W, HEADER_H), 2)

    turn_label = "WHITE" if turn_white else "BLACK"
    draw_text(surface, font, f"Turn: {turn_label}    ply={ply}", 16, 10)
    draw_text(surface, font, f"White: bar={mine_bar} off={mine_off}", 16, 36)
    draw_text(surface, font, f"Black: bar={opp_bar} off={opp_off}", 16, 62)

    bar_x0 = MARGIN + 6 * POINT_W + GAP
    bar_rect = pygame.Rect(bar_x0, TOP, BAR_W, BOTTOM - TOP)
    pygame.draw.rect(surface, (55, 40, 28), bar_rect)
    pygame.draw.rect(surface, (25, 18, 12), bar_rect, 2)
    pygame.draw.line(surface, (70, 55, 40), (MARGIN, MID_Y), (BOARD_W - MARGIN, MID_Y), 2)

    point_rects = {}
    for idx in range(24):
        x = point_x(idx)
        top = point_is_top(idx)
        col = idx - 12 if top else 11 - idx
        tri_color = TRI_A if col % 2 == 0 else TRI_B
        if top:
            draw_triangle(surface, x, TOP + 8, POINT_H - 16, upward=False, color=tri_color)
            img = font.render(str(idx), True, SUBTEXT)
            surface.blit(img, (x - img.get_width() // 2, TOP - 28))
            rect = pygame.Rect(x - POINT_W // 2, TOP + 8, POINT_W, POINT_H - 16)
        else:
            draw_triangle(surface, x, MID_Y + 8, POINT_H - 16, upward=True, color=tri_color)
            img = font.render(str(idx), True, SUBTEXT)
            surface.blit(img, (x - img.get_width() // 2, BOTTOM + 6))
            rect = pygame.Rect(x - POINT_W // 2, MID_Y + 8, POINT_W, POINT_H - 16)
        point_rects[idx] = rect

    for idx in range(24):
        x = point_x(idx)
        base = point_base_y(idx)
        top = point_is_top(idx)
        for k in range(min(int(mine[idx]), 6)):
            y = base + (k * STACK_DY if top else -k * STACK_DY)
            draw_checker(surface, x, y, is_white=True)
        for k in range(min(int(opp[idx]), 6)):
            y = base + (k * STACK_DY if top else -k * STACK_DY)
            draw_checker(surface, x + CHECKER_R + 2, y, is_white=False)

    def dice_anchor(white_side):
        x0 = int(BOARD_W * (0.66 if white_side else 0.18))
        return x0, MID_Y - DICE_SIZE // 2

    dx, dy = dice_anchor(turn_white)
    dice_rects = []
    for i, val in enumerate(dice_values):
        size = DICE_SIZE + (10 if i == active_die_idx else 0)
        rect = pygame.Rect(dx + i * (DICE_SIZE + DICE_GAP), dy + (DICE_SIZE - size) // 2, size, size)
        exhausted = used_dice[i] >= required_dice[i]
        draw_die(surface, rect, val, active=(i == active_die_idx), used=exhausted)
        dice_rects.append(rect)

    undo_rect = pygame.Rect(dx - 44, dy + 6, 30, 30)
    pygame.draw.rect(surface, (110, 110, 130), undo_rect, border_radius=6)
    draw_undo_icon(surface, undo_rect)

    ok_rect = pygame.Rect(dx + len(dice_values) * (DICE_SIZE + DICE_GAP) + 8, dy + 6, 30, 30)
    pygame.draw.rect(surface, SUCCESS if can_submit else (80, 80, 80), ok_rect, border_radius=6)
    draw_check_icon(surface, ok_rect, can_submit)

    pygame.draw.line(surface, DIV, (BOARD_W, 0), (BOARD_W, H), 2)
    return point_rects, dice_rects, undo_rect, ok_rect


def draw_panel(surface, font, small_font, moves, info_lines, manual_steps, turn_white):
    x, y = BOARD_W + 12, 16
    draw_text(surface, font, "Controls:", x, y)
    y += 28
    for line in ["LMB on point - use active die", "R - roll/reset turn", "N - reset env", "ESC - quit"]:
        draw_text(surface, small_font, line, x, y, (200, 200, 200))
        y += 20

    y += 8
    draw_text(surface, font, f"Legal moves: {len(moves)}", x, y)
    y += 24
    draw_text(surface, small_font, f"Manual: {' | '.join(f'{a}->{b}' for a,b in manual_steps) or '(empty)'}", x, y, ACCENT)

    y += 30
    max_lines = (H - y - 150) // 18
    for i in range(min(len(moves), max_lines)):
        draw_text(surface, small_font, f"[{i:3d}] {move_to_str(moves[i], turn_white=turn_white)}", x, y, (210, 210, 210))
        y += 18

    y = H - 120
    pygame.draw.line(surface, (60, 60, 60), (BOARD_W, y - 8), (W, y - 8), 1)
    for line in info_lines[-5:]:
        draw_text(surface, small_font, line, x, y, (180, 180, 220))
        y += 18


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Backgammon Viewer (manual dice UI)")
    font = pygame.font.Font(FONT_NAME, 22)
    small = pygame.font.Font(FONT_NAME, 16)
    clock = pygame.time.Clock()

    env = bg_env.Env(123)
    env.reset()

    turn_white = True
    info_lines = ["Started. Click point columns to move by active die."]

    def refresh_moves():
        mv = np.asarray(env.legal_moves(), dtype=np.uint8)
        if mv.ndim == 1:
            mv = mv.reshape(0, 8)
        return mv

    def start_turn():
        nonlocal dice_values, used_dice, required_dice, manual_steps, history, selected_die_idx
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

    dice_values, used_dice, required_dice, manual_steps, history = [], [], [], [], []
    selected_die_idx = 0
    start_turn()
    moves = refresh_moves()

    running = True
    while running:
        clock.tick(FPS)
        raw = np.asarray(env.get_state_raw(), dtype=np.int16)
        base_mine, base_opp, mine_bar, base_mine_off, opp_bar, base_opp_off, ply = decode_raw(raw)

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

        active_idx = resolve_active_die_idx(selected_die_idx, used_dice, required_dice)
        selected_die_idx = active_idx
        required_steps = max_micro_steps_in_moves(moves, turn_white)
        can_submit = len(moves) == 0 or len(manual_steps) == required_steps

        point_rects, dice_rects, undo_rect, ok_rect = draw_board(
            screen, font, view_mine, view_opp, white_bar, view_mine_off, black_bar, view_opp_off,
            ply, turn_white, dice_values, used_dice, required_dice, active_idx, can_submit
        )
        draw_panel(screen, font, small, moves, info_lines, manual_steps, turn_white)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    env.reset()
                    turn_white = True
                    start_turn()
                    moves = refresh_moves()
                    info_lines.append("Reset env.")
                elif event.key == pygame.K_r:
                    start_turn()
                    moves = refresh_moves()
                    info_lines.append(f"Reroll: {dice_values}")
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if undo_rect.collidepoint(mx, my) and history:
                    prev_state, fr, to, die_idx = history.pop()
                    env.set_state_raw(prev_state)
                    manual_steps.pop()
                    used_dice[die_idx] = max(0, used_dice[die_idx] - 1)
                    selected_die_idx = die_idx
                    moves = refresh_moves()
                    info_lines.append(f"Undo {fr}->{to}")
                    continue

                if ok_rect.collidepoint(mx, my) and can_submit:
                    done = int(np.asarray(env.get_state_raw())[49]) >= 15
                    reward = 1.0 if done else 0.0
                    if not done:
                        env.commit_turn()
                    info_lines.append(f"Apply: {' | '.join(f'{a}->{b}' for a,b in manual_steps)} | r={reward} done={done}")
                    if done:
                        env.reset()
                        turn_white = True
                    else:
                        turn_white = not turn_white
                    start_turn()
                    moves = refresh_moves()
                    continue

                clicked_die = next((i for i, rect in enumerate(dice_rects) if rect.collidepoint(mx, my)), None)
                if clicked_die is not None:
                    if used_dice[clicked_die] < required_dice[clicked_die]:
                        selected_die_idx = clicked_die
                    continue

                if active_idx < 0:
                    continue
                die = dice_values[active_idx]
                clicked_idx = next((i for i, rect in point_rects.items() if rect.collidepoint(mx, my)), None)
                if clicked_idx is None:
                    continue

                own = view_mine if turn_white else view_opp
                for fr, to in manual_steps:
                    if turn_white and fr == clicked_idx and own[fr] <= 0:
                        break
                if own[clicked_idx] <= 0:
                    info_lines.append("No checker of current color on point.")
                    continue

                env_from = clicked_idx if turn_white else 23 - clicked_idx
                env_to = env_from - die
                env_to = env_to if env_to >= 0 else 25

                prev_state = np.asarray(env.get_state_raw(), dtype=np.int16)
                ok, _ = env.apply_micro_step(int(env_from), int(env_to))
                if not ok:
                    info_lines.append("Invalid micro-step.")
                    continue
                to = env_to if turn_white else (23 - env_to if env_to <= 23 else env_to)
                manual_steps.append((clicked_idx, to))
                used_dice[active_idx] += 1
                selected_die_idx = active_idx
                history.append((prev_state, clicked_idx, to, active_idx))
                moves = refresh_moves()


    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
