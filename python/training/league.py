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

try:
    import batched_bg_env
except Exception:  # pragma: no cover
    batched_bg_env = None


class _FallbackEnv:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.turn = 0
        self.state = np.zeros(53, dtype=np.int16)

    def roll_dice(self):
        self.dice = (int(self.rng.integers(1, 7)), int(self.rng.integers(1, 7)))
        return self.dice

    def legal_moves(self):
        return np.array([[0, 1, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    def step_move(self, mv):
        self.turn += 1
        self.state[52] = self.turn
        done = self.turn >= 8
        reward = 1.0 if done and (int(mv[1]) % 2 == 0) else (-1.0 if done else 0.0)
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


@dataclass
class _GameSpec:
    game_id: str
    p1: object
    p2: object


class RandomAgent:
    agent_id = "random"


class BaselineAgent:
    agent_id = "baseline"


def pass_move() -> np.ndarray:
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

    def _score_baseline(self, moves: np.ndarray) -> np.ndarray:
        if len(moves) == 0:
            return pass_move()
        scores = moves[:, 1::2].sum(axis=1)
        return moves[int(np.argmax(scores))]

    def _score_random(self, moves: np.ndarray) -> np.ndarray:
        if len(moves) == 0:
            return pass_move()
        return moves[np.random.randint(len(moves))]

    def _select_moves_value_batched(self, states: np.ndarray, legal_moves_list: list[np.ndarray], agent: ValueAgent) -> np.ndarray:
        actions = np.full((len(legal_moves_list), 8), 255, dtype=np.uint8)
        candidate_obs: list[np.ndarray] = []
        candidate_done: list[bool] = []
        candidate_owner: list[int] = []

        sim = bg_env.Env(int(self.seed)) if bg_env is not None else _FallbackEnv(int(self.seed))
        for i, moves in enumerate(legal_moves_list):
            if len(moves) == 0:
                continue
            if len(moves) == 1:
                actions[i] = moves[0]
                continue
            state_i = states[i]
            for mv in moves:
                sim.set_state_raw(state_i)
                _, done = sim.step_move(mv)
                candidate_obs.append(self.state_vector(sim))
                candidate_done.append(done)
                candidate_owner.append(i)

        if candidate_obs:
            obs_t = torch.as_tensor(np.stack(candidate_obs).astype(np.float32), dtype=torch.float32, device=agent.device)
            probs = agent.predict_proba_tensor(obs_t).reshape(-1).detach().cpu().numpy()
            vals = np.where(np.asarray(candidate_done, dtype=bool), 1.0, probs)

            grouped_vals: dict[int, list[float]] = {}
            for owner, val in zip(candidate_owner, vals):
                grouped_vals.setdefault(owner, []).append(float(val))
            for i, moves in enumerate(legal_moves_list):
                if len(moves) <= 1:
                    continue
                actions[i] = moves[int(np.argmax(grouped_vals[i]))]

        return actions

    def _play_all_games_batched(self, game_specs: list[_GameSpec], epoch: int) -> list[GameResult]:
        if batched_bg_env is None:
            return [self.play_game(spec.p1, spec.p2, spec.game_id, epoch) for spec in game_specs]

        n_games = len(game_specs)
        env = batched_bg_env.Env(n_games, self.seed + epoch * 100_000)
        env.reset()

        histories = [[] for _ in range(n_games)]
        winners = [spec.p1.agent_id for spec in game_specs]
        turns = [0 for _ in range(n_games)]
        done = np.zeros((n_games,), dtype=bool)

        for turn in range(self.cfg.max_turns_per_game):
            active_idx = np.flatnonzero(~done)
            if len(active_idx) == 0:
                break

            env.roll_dice()
            states = np.asarray(env.get_states_raw(), dtype=np.int16)
            legal_moves = list(env.legal_moves())
            actions = np.full((n_games, 8), 255, dtype=np.uint8)

            by_group: dict[str, list[int]] = {"A": [], "B": [], "C": []}
            for i in active_idx:
                spec = game_specs[int(i)]
                actor = spec.p1 if turn % 2 == 0 else spec.p2
                if isinstance(actor, ValueAgent):
                    by_group[actor.group].append(int(i))
                elif actor.agent_id == "baseline":
                    actions[i] = self._score_baseline(legal_moves[i])
                else:
                    actions[i] = self._score_random(legal_moves[i])

            for group_name in ("A", "B", "C"):
                idxs = by_group[group_name]
                if not idxs:
                    continue

                # Формируем максимально крупный батч по группе архитектуры.
                by_agent: dict[str, list[int]] = {}
                agents: dict[str, ValueAgent] = {}
                for i in idxs:
                    actor = game_specs[i].p1 if turn % 2 == 0 else game_specs[i].p2
                    by_agent.setdefault(actor.agent_id, []).append(i)
                    agents[actor.agent_id] = actor

                for agent_id, game_ids in by_agent.items():
                    agent = agents[agent_id]
                    local_states = states[game_ids]
                    local_moves = [legal_moves[i] for i in game_ids]
                    local_actions = self._select_moves_value_batched(local_states, local_moves, agent)
                    actions[np.asarray(game_ids, dtype=np.int64)] = local_actions

            rewards, done_step = env.step_apply(actions)
            rewards = np.asarray(rewards, dtype=np.float32)
            done_step = np.asarray(done_step, dtype=np.uint8).astype(bool)

            for i in active_idx:
                spec = game_specs[int(i)]
                actor = spec.p1 if turn % 2 == 0 else spec.p2
                opp = spec.p2 if turn % 2 == 0 else spec.p1
                histories[i].append({
                    "state_vector": states[i][:52].astype(np.float32),
                    "agent_id": actor.agent_id,
                    "opponent_id": opp.agent_id,
                    "game_id": spec.game_id,
                    "step_index": turns[i],
                    "epoch": epoch,
                })
                turns[i] += 1
                if done_step[i]:
                    winners[i] = actor.agent_id if rewards[i] > 0 else opp.agent_id
                    done[i] = True

        return [
            GameResult(
                game_id=game_specs[i].game_id,
                steps=histories[i],
                winner=winners[i],
                turns=turns[i],
                player_1_id=game_specs[i].p1.agent_id,
                player_2_id=game_specs[i].p2.agent_id,
            )
            for i in range(n_games)
        ]

    def play_game(self, p1, p2, game_id: str, epoch: int):
        env = bg_env.Env(self.seed + hash(game_id) % 100000) if bg_env is not None else _FallbackEnv(self.seed + hash(game_id) % 100000)
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
            if isinstance(actor, ValueAgent):
                move = self._select_moves_value_batched(np.asarray([env.get_state_raw()], dtype=np.int16), [env.legal_moves()], actor)[0]
            elif actor.agent_id == "baseline":
                move = self._score_baseline(env.legal_moves())
            else:
                move = self._score_random(env.legal_moves())
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
        opponents = [self.random, self.baseline]
        specs: list[_GameSpec] = []

        for i, a in enumerate(trainable_agents):
            for j in range(i, len(trainable_agents)):
                b = trainable_agents[j]
                for g in range(self.cfg.games_per_pair):
                    specs.append(_GameSpec(game_id=f"e{epoch}_t{i}_{b.agent_id}_{g}", p1=a, p2=b))
            for opp in opponents:
                for g in range(self.cfg.games_per_pair):
                    specs.append(_GameSpec(game_id=f"e{epoch}_{a.agent_id}_{opp.agent_id}_{g}", p1=a, p2=opp))

        results = self._play_all_games_batched(specs, epoch)
        dt = max(time.time() - t0, 1e-6)
        return results, len(results) / dt
