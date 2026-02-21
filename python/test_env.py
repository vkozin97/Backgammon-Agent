import bg_env

env = bg_env.Env(123)
env.reset()
env.roll_dice()

moves = env.legal_moves()
r, done = env.step_move(moves[0])
print("step_move:", r, done)
