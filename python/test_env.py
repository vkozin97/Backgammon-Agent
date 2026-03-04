import bg_env
import numpy as np

env = bg_env.Env(123, n_games=5)
env.reset()
env.roll_dice()

double_possible, moves = env.legal_moves()
moves = np.asarray(moves, dtype=np.uint8)
if len(moves):
    r, dave_after, accepted, done = env.step_move(moves[0], apply_double=0, accept_double=1)
    print("step_move:", r, dave_after, accepted, done, "double_possible=", double_possible)
else:
    print("no legal moves")
