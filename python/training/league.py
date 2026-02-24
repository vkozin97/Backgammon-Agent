from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import torch

from .agents import ValueAgent

try:
    import bg_env
except Exception:  # pragma: no cover
    bg_env = None


class _FallbackEnv:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.turn = 0
        self.state = np.zeros(53, dtype=np.int16)

    def roll_dice(self):
        self.dice = (int(self.rng.integers(1,7)), int(self.rng.integers(1,7)))
        return self.dice

    def legal_moves(self):
        return np.array([[0,1,0,0,0,0,0,0],[1,2,0,0,0,0,0,0]], dtype=np.uint8)

    def step_move(self, mv):
        self.turn += 1
        self.state[52] = self.turn
        done = self.turn >= 8
        reward = 1.0 if done and (int(mv[1]) % 2 == 0) else ( -1.0 if done else 0.0)
        return reward, done

    def get_state_raw(self):
        st = self.state.copy()
        st[:24] = self.turn
        st[24:48] = 8 - self.turn
        return st

    def set_state_raw(self, st):
        self.state = np.array(st, dtype=np.int16)
        self.turn = int(self.state[52])


@dataclass
class GameResult:
    game_id: str
    steps: list[dict]
    winner: str
    turns: int
    player_1_id: str
    player_2_id: str


class RandomAgent:
    agent_id = "random"

    def select(self, env) -> np.ndarray:
        moves = env.legal_moves()
        if len(moves) == 0:
            return pass_move()
        return moves[np.random.randint(len(moves))]


class BaselineAgent:
    agent_id = "baseline"

    def select(self, env) -> np.ndarray:
        moves = env.legal_moves()
        if len(moves) == 0:
            return pass_move()
        scores = moves[:, 1::2].sum(axis=1)
        return moves[int(np.argmax(scores))]


def pass_move() -> np.ndarray:
    # All 255 means "no micro-steps"; env will commit turn on step_move.
    return np.full((8,), 255, dtype=np.uint8)


class LeagueController:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.random = RandomAgent()
        self.baseline = BaselineAgent()

    @staticmethod
    def state_vector(env) -> np.ndarray:
        raw = np.asarray(env.get_state_raw(), dtype=np.float32)
        return raw[:52]

    def _select_move_value(self, env, agent: ValueAgent) -> np.ndarray:
        moves = env.legal_moves()
        if len(moves) == 0:
            return pass_move()
        if len(moves) == 1:
            return moves[0]
        state = np.asarray(env.get_state_raw(), dtype=np.int16)
        obs_batch = []
        done_mask = []
        for mv in moves:
            env.set_state_raw(state)
            _, done = env.step_move(mv)
            obs_batch.append(self.state_vector(env))
            done_mask.append(done)
        env.set_state_raw(state)

        obs_t = torch.as_tensor(np.stack(obs_batch).astype(np.float32), dtype=torch.float32, device=agent.device)
        probs = agent.predict_proba_tensor(obs_t).reshape(-1).detach().cpu().numpy()
        vals = np.where(np.asarray(done_mask, dtype=bool), 1.0, probs)
        return moves[int(np.argmax(vals))]

    def play_game(self, p1, p2, game_id: str, epoch: int):
        env = (bg_env.Env(self.seed + hash(game_id) % 100000) if bg_env is not None else _FallbackEnv(self.seed + hash(game_id) % 100000))
        env.reset()
        history = []
        players = [p1, p2]
        turn = 0
        done = False
        winner = players[0].agent_id
        while not done and turn < self.cfg.max_turns_per_game:
            env.roll_dice()
            actor = players[turn % 2]
            opp = players[(turn + 1) % 2]
            state = self.state_vector(env)
            move = actor.select(env) if hasattr(actor, "select") else self._select_move_value(env, actor)
            reward, done = env.step_move(move)
            history.append({
                "state_vector": state,
                "agent_id": actor.agent_id,
                "opponent_id": opp.agent_id,
                "game_id": game_id,
                "step_index": turn,
                "epoch": epoch,
            })
            if done:
                winner = actor.agent_id if reward > 0 else opp.agent_id
            turn += 1
        return GameResult(
            game_id=game_id,
            steps=history,
            winner=winner,
            turns=turn,
            player_1_id=p1.agent_id,
            player_2_id=p2.agent_id,
        )

    def run_epoch(self, trainable_agents: list[ValueAgent], epoch: int):
        t0 = time.time()
        results = []
        opponents = [self.random, self.baseline]
        for i, a in enumerate(trainable_agents):
            for b in trainable_agents[i + 1:]:
                for g in range(self.cfg.games_per_pair):
                    results.append(self.play_game(a, b, f"e{epoch}_t{i}_{b.agent_id}_{g}", epoch))
            for opp in opponents:
                for g in range(self.cfg.games_per_pair):
                    results.append(self.play_game(a, opp, f"e{epoch}_{a.agent_id}_{opp.agent_id}_{g}", epoch))
        dt = max(time.time() - t0, 1e-6)
        return results, len(results) / dt
