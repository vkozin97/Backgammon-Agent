import sys
import numpy as np
import pygame

import bg_env  # pybind11 module


# ----------------- raw decode -----------------
# raw layout (len=53):
# [0..23] mine points
# [24..47] opp points
# [48] mine_bar, [49] mine_off, [50] opp_bar, [51] opp_off, [52] ply
def decode_raw(raw: np.ndarray):
    raw = np.asarray(raw)
    mine = raw[0:24].astype(int)
    opp = raw[24:48].astype(int)
    mine_bar, mine_off, opp_bar, opp_off, ply = map(int, raw[48:53])
    return mine, opp, mine_bar, mine_off, opp_bar, opp_off, ply


def move_to_str(mv8: np.ndarray):
    mv8 = np.asarray(mv8).astype(int)
    steps = []
    for k in range(4):
        fr = mv8[2 * k]
        to = mv8[2 * k + 1]
        if fr == 255 or to == 255:
            continue
        steps.append(f"{fr}->{to}")
    return " | ".join(steps) if steps else "(empty)"


# ----------------- pygame config -----------------
W, H = 1100, 720
FPS = 60

PANEL_W = 420
BOARD_W = W - PANEL_W

FONT_NAME = None  # default system font


# ----- board look -----
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

# geometry
MARGIN = 18
GAP = 20
BAR_W = 52

HEADER_H = 90  # <-- fixes overlap: header is separate area

TOP = HEADER_H + 30
BOTTOM = H - 40
MID_Y = (TOP + BOTTOM) // 2

# point sizes
POINT_W = (BOARD_W - 2 * MARGIN - BAR_W - GAP * 2) // 12
POINT_H = (BOTTOM - TOP - GAP) // 2

CHECKER_R = min(POINT_W // 2 - 2, 18)
STACK_DY = CHECKER_R * 2 - 4


def draw_text(surf, font, text, x, y, color=TEXT):
    img = font.render(text, True, color)
    surf.blit(img, (x, y))


# ----------------- coordinate helpers -----------------
def point_x(idx: int) -> int:
    """
    X-center of point idx (0..23) in a realistic layout:
      Top row shows 12..23 left->right (23 on the right)
      Bottom row shows 11..0  left->right (0 on the right)
      With a bar in the middle.
    """
    left_start = MARGIN
    right_start = MARGIN + 6 * POINT_W + GAP + BAR_W + GAP

    def col_center(start_x, col):
        return start_x + col * POINT_W + POINT_W // 2

    if 12 <= idx <= 23:
        col = idx - 12  # 0..11
    else:
        col = 11 - idx  # idx=11->0 ... idx=0->11

    if col <= 5:
        return col_center(left_start, col)
    else:
        return col_center(right_start, col - 6)


def point_is_top(idx: int) -> bool:
    return 12 <= idx <= 23


def point_base_y(idx: int) -> int:
    """Base Y for stacking checkers on that point."""
    if point_is_top(idx):
        return TOP + CHECKER_R + 6
    else:
        return BOTTOM - CHECKER_R - 6


# ----------------- drawing primitives -----------------
def draw_triangle(surface, x_center, y_top, height, upward: bool, color):
    half = POINT_W // 2 - 2
    if upward:
        pts = [
            (x_center - half, y_top + height),
            (x_center + half, y_top + height),
            (x_center, y_top),
        ]
    else:
        pts = [
            (x_center - half, y_top),
            (x_center + half, y_top),
            (x_center, y_top + height),
        ]
    pygame.draw.polygon(surface, color, pts)


def draw_checker(surface, x, y, is_white: bool):
    fill = WHITE if is_white else BLACK
    pygame.draw.circle(surface, fill, (x, y), CHECKER_R)
    pygame.draw.circle(surface, OUTLINE, (x, y), CHECKER_R, 2)


# ----------------- main board draw -----------------
def draw_board(surface, font, mine, opp, mine_bar, mine_off, opp_bar, opp_off, dice, ply):
    # outer background
    surface.fill((25, 25, 25))

    # board frame and inner
    pygame.draw.rect(surface, FRAME, (0, 0, BOARD_W, H))
    inner = pygame.Rect(8, HEADER_H, BOARD_W - 16, H - HEADER_H - 8)
    pygame.draw.rect(surface, BOARD_BG, inner)

    # header bar (separate area, no overlap)
    pygame.draw.rect(surface, HEADER_BG, (0, 0, BOARD_W, HEADER_H))
    pygame.draw.line(surface, DIV, (0, HEADER_H), (BOARD_W, HEADER_H), 2)

    draw_text(surface, font, f"Dice: {dice[0]}, {dice[1]}   ply={ply}", 16, 10, TEXT)
    draw_text(surface, font, f"White (mine): bar={mine_bar} off={mine_off}", 16, 36, TEXT)
    draw_text(surface, font, f"Black (opp):  bar={opp_bar} off={opp_off}", 16, 62, TEXT)

    # bar area
    bar_x0 = MARGIN + 6 * POINT_W + GAP
    bar_rect = pygame.Rect(bar_x0, TOP, BAR_W, BOTTOM - TOP)
    pygame.draw.rect(surface, (55, 40, 28), bar_rect)
    pygame.draw.rect(surface, (25, 18, 12), bar_rect, 2)

    # middle separator line
    pygame.draw.line(surface, (70, 55, 40), (MARGIN, MID_Y), (BOARD_W - MARGIN, MID_Y), 2)

    # triangles + indices
    for idx in range(24):
        x = point_x(idx)
        top = point_is_top(idx)

        # visual column in its row (0..11 from left->right)
        if top:
            col = idx - 12
        else:
            col = 11 - idx
        tri_color = TRI_A if (col % 2 == 0) else TRI_B

        if top:
            draw_triangle(surface, x, TOP + 8, POINT_H - 16, upward=False, color=tri_color)
            img = font.render(str(idx), True, SUBTEXT)
            surface.blit(img, (x - img.get_width() // 2, TOP - 28))
        else:
            draw_triangle(surface, x, MID_Y + 8, POINT_H - 16, upward=True, color=tri_color)
            img = font.render(str(idx), True, SUBTEXT)
            surface.blit(img, (x - img.get_width() // 2, BOTTOM + 6))

    # checkers stacks
    for idx in range(24):
        x = point_x(idx)
        base = point_base_y(idx)
        top = point_is_top(idx)

        # mine = white
        nW = int(mine[idx])
        for k in range(min(nW, 6)):
            y = base + (k * STACK_DY if top else -k * STACK_DY)
            draw_checker(surface, x, y, is_white=True)
        if nW > 6:
            img = font.render(f"+{nW-6}", True, TEXT)
            y = base + (6 * STACK_DY if top else -6 * STACK_DY)
            surface.blit(img, (x - img.get_width() // 2, y - img.get_height() // 2))

        # opp = black (shifted a bit, only for debug if both appear on same point)
        nB = int(opp[idx])
        x2 = x + CHECKER_R + 2
        for k in range(min(nB, 6)):
            y = base + (k * STACK_DY if top else -k * STACK_DY)
            draw_checker(surface, x2, y, is_white=False)
        if nB > 6:
            img = font.render(f"+{nB-6}", True, TEXT)
            y = base + (6 * STACK_DY if top else -6 * STACK_DY)
            surface.blit(img, (x2 - img.get_width() // 2, y - img.get_height() // 2))

    # bar stacks
    def draw_bar_stack(count, is_white, y_start, direction):
        for k in range(min(count, 6)):
            y = y_start + direction * k * STACK_DY
            draw_checker(surface, bar_rect.centerx, y, is_white=is_white)
        if count > 6:
            img = font.render(f"+{count-6}", True, TEXT)
            surface.blit(
                img,
                (bar_rect.centerx - img.get_width() // 2,
                 y_start + direction * 6 * STACK_DY - img.get_height() // 2)
            )

    draw_bar_stack(mine_bar, True, TOP + CHECKER_R + 10, +1)
    draw_bar_stack(opp_bar, False, BOTTOM - CHECKER_R - 10, -1)

    # divider to panel
    pygame.draw.line(surface, DIV, (BOARD_W, 0), (BOARD_W, H), 2)


# ----------------- right panel -----------------
def draw_panel(surface, font, small_font, moves, selected_idx, info_lines):
    x = BOARD_W + 12
    y = 16

    draw_text(surface, font, "Controls:", x, y)
    y += 28
    for line in [
        "R  - roll dice",
        "N  - reset",
        "UP/DOWN - select move",
        "ENTER - apply selected move",
        "ESC - quit",
    ]:
        draw_text(surface, small_font, line, x, y, (200, 200, 200))
        y += 20

    y += 10
    draw_text(surface, font, f"Legal moves: {len(moves)}", x, y)
    y += 28

    list_y0 = y
    max_lines = (H - list_y0 - 150) // 18
    start = 0
    if selected_idx >= max_lines:
        start = selected_idx - max_lines + 1

    for i in range(start, min(len(moves), start + max_lines)):
        line = f"[{i:3d}] {move_to_str(moves[i])}"
        color = (255, 230, 120) if i == selected_idx else (210, 210, 210)
        draw_text(surface, small_font, line, x, y, color)
        y += 18

    # info bottom
    y = H - 120
    pygame.draw.line(surface, (60, 60, 60), (BOARD_W, y - 8), (W, y - 8), 1)
    for line in info_lines[-5:]:
        draw_text(surface, small_font, line, x, y, (180, 180, 220))
        y += 18


# ----------------- main loop -----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Backgammon Viewer (pybind11 + pygame)")

    font = pygame.font.Font(FONT_NAME, 22)
    small = pygame.font.Font(FONT_NAME, 16)
    clock = pygame.time.Clock()

    env = bg_env.Env(123)
    env.reset()
    dice = env.roll_dice()

    selected = 0
    info_lines = ["Viewer started. Press R to roll dice, ENTER to apply move."]

    def refresh_moves():
        nonlocal selected
        mv = env.legal_moves()
        mv = np.asarray(mv, dtype=np.uint8)
        if mv.ndim == 1:
            mv = mv.reshape(0, 8)
        if len(mv) == 0:
            selected = 0
        else:
            selected = max(0, min(selected, len(mv) - 1))
        return mv

    moves = refresh_moves()

    running = True
    while running:
        clock.tick(FPS)

        raw = env.get_state_raw()
        mine, opp, mine_bar, mine_off, opp_bar, opp_off, ply = decode_raw(raw)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_n:
                    env.reset()
                    dice = env.roll_dice()
                    moves = refresh_moves()
                    info_lines.append("Reset.")

                elif event.key == pygame.K_r:
                    dice = env.roll_dice()
                    moves = refresh_moves()
                    info_lines.append(f"Rolled dice: {dice[0]},{dice[1]} (moves={len(moves)})")

                elif event.key == pygame.K_UP:
                    if len(moves) > 0:
                        selected = max(0, selected - 1)

                elif event.key == pygame.K_DOWN:
                    if len(moves) > 0:
                        selected = min(len(moves) - 1, selected + 1)

                elif event.key == pygame.K_RETURN:
                    if len(moves) > 0:
                        mv = moves[selected]
                        reward, done = env.step_move(mv)
                        info_lines.append(f"Applied #{selected}: {move_to_str(mv)} | r={reward} done={done}")
                        if done:
                            info_lines.append("Terminal. Auto-reset.")
                            env.reset()
                            dice = env.roll_dice()
                        moves = refresh_moves()

        # draw
        draw_board(screen, font, mine, opp, mine_bar, mine_off, opp_bar, opp_off, dice, ply)
        draw_panel(screen, font, small, moves, selected, info_lines)
        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())