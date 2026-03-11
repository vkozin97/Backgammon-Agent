from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np
import torch

from match_win_probs import get_match_win_probs
from .agents import ValueAgent
from .observation import state_to_observation

try:
    from torch.func import functional_call, stack_module_state, vmap
except Exception:  # pragma: no cover
    functional_call = None
    stack_module_state = None
    vmap = None

try:
    import bg_env
except Exception:  # pragma: no cover
    bg_env = None

try:
    import batched_bg_env
except Exception:  # pragma: no cover
    batched_bg_env = None


MATCH_VECTOR_DIM = 12
MODEL_OUTPUT_DIM = 31

REWARD_VECTOR_DIM = 6
REWARD_VALUES = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float32)

def _reward_expectation(probs_row: np.ndarray) -> float:
    p = np.asarray(probs_row, dtype=np.float32)
    reward_head = p[MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]
    if reward_head.size != REWARD_VECTOR_DIM:
        return 0.0
    return float(np.dot(REWARD_VALUES, reward_head))


def _mask_and_normalize_head(head_probs: np.ndarray, left_to_win: int) -> np.ndarray:
    h = np.asarray(head_probs, dtype=np.float32).copy()
    left = int(np.clip(left_to_win, 0, MATCH_VECTOR_DIM - 1))
    for i in range(MATCH_VECTOR_DIM):
        if i > left:
            h[i] = 0.0
    total = float(np.sum(h))
    if total <= 0.0:
        h.fill(0.0)
        h[left] = 1.0
        return h
    return h / total


def _mask_eval_outputs(probs_row: np.ndarray, obs: np.ndarray) -> np.ndarray:
    out = np.asarray(probs_row, dtype=np.float32).copy()
    my_left = int(np.clip(np.round(float(obs[-6])) if obs.size >= 6 else MATCH_VECTOR_DIM - 1, 0, MATCH_VECTOR_DIM - 1))
    opp_left = int(np.clip(np.round(float(obs[-5])) if obs.size >= 5 else MATCH_VECTOR_DIM - 1, 0, MATCH_VECTOR_DIM - 1))
    out[:MATCH_VECTOR_DIM] = _mask_and_normalize_head(out[:MATCH_VECTOR_DIM], my_left)
    out[MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2] = _mask_and_normalize_head(out[MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2], opp_left)
    out[MATCH_VECTOR_DIM * 2] = float(np.clip(out[MATCH_VECTOR_DIM * 2], 0.0, 1.0))
    return out


def _match_win_probability(masked_probs_row: np.ndarray) -> float:
    return float(np.asarray(masked_probs_row, dtype=np.float32)[0])

# MET is immutable and shared across all self-play calls.
# Keep one float32 table in memory and reuse it in all evaluations.
MET_TABLE = np.asarray(get_match_win_probs(MATCH_VECTOR_DIM - 1), dtype=np.float32)


def _redistribute_forbidden_stay_mass(my_probs: np.ndarray, opp_probs: np.ndarray, my_left: int, opp_left: int) -> tuple[np.ndarray, np.ndarray]:
    my = np.asarray(my_probs, dtype=np.float32).copy()
    opp = np.asarray(opp_probs, dtype=np.float32).copy()
    my_left = int(np.clip(my_left, 0, MATCH_VECTOR_DIM - 1))
    opp_left = int(np.clip(opp_left, 0, MATCH_VECTOR_DIM - 1))

    stay_mass = float(my[my_left] + opp[opp_left])
    my[my_left] = 0.0
    opp[opp_left] = 0.0

    my_rest = float(np.sum(my))
    opp_rest = float(np.sum(opp))
    rest_total = my_rest + opp_rest
    if stay_mass <= 0.0 or rest_total <= 1e-8:
        return my, opp

    my_add = stay_mass * (my_rest / rest_total)
    opp_add = stay_mass * (opp_rest / rest_total)

    if my_rest > 1e-8:
        my *= (my_rest + my_add) / my_rest
    if opp_rest > 1e-8:
        opp *= (opp_rest + opp_add) / opp_rest
    return my, opp


def _expected_match_win_prob(my_probs: np.ndarray, opp_probs: np.ndarray) -> float:
    my = np.asarray(my_probs, dtype=np.float32).reshape(-1)
    opp = np.asarray(opp_probs, dtype=np.float32).reshape(-1)
    met_view = MET_TABLE[: my.shape[0], : opp.shape[0]]
    # Avoid temporary outer-product allocations in hot self-play path.
    return float(my @ met_view @ opp)


def _extract_obs_controls(obs: np.ndarray) -> tuple[int, int, int, int, int]:
    v = np.asarray(obs, dtype=np.float32).reshape(-1)
    my_left = int(np.clip(np.round(float(v[-6])) if v.size >= 6 else MATCH_VECTOR_DIM - 1, 0, MATCH_VECTOR_DIM - 1))
    opp_left = int(np.clip(np.round(float(v[-5])) if v.size >= 5 else MATCH_VECTOR_DIM - 1, 0, MATCH_VECTOR_DIM - 1))
    dave_val = int(max(1, round(float(v[-7])))) if v.size >= 7 else 1
    my_double_avail = int(round(float(v[-4]))) if v.size >= 4 else 0
    opp_double_avail = int(round(float(v[-3]))) if v.size >= 3 else 0
    return my_left, opp_left, dave_val, my_double_avail, opp_double_avail


def _set_obs_double_state(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32).copy()
    if x.size >= 7:
        x[-7] = x[-7] * 2.0
    if x.size >= 4:
        x[-4] = 0.0
    if x.size >= 3:
        x[-3] = 1.0
    if x.size >= 1:
        x[-1] = 1.0
    return x


def _head_win_eval(probs_row: np.ndarray, obs_row: np.ndarray) -> float:
    masked = _mask_eval_outputs(probs_row, obs_row)
    my_left, opp_left, _, _, _ = _extract_obs_controls(obs_row)
    my = masked[:MATCH_VECTOR_DIM]
    opp = masked[MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2]
    my_r, opp_r = _redistribute_forbidden_stay_mass(my, opp, my_left, opp_left)
    return _expected_match_win_prob(my_r, opp_r)


def _decide_apply_double(probs_now: np.ndarray, probs_after_double: np.ndarray, obs_now: np.ndarray) -> int:
    my_left, opp_left, dave_val, my_double_avail, _ = _extract_obs_controls(obs_now)
    if my_double_avail <= 0:
        return 0

    p_win = _head_win_eval(probs_now, obs_now)

    my_after_reject = max(my_left - int(max(dave_val, 1)), 0)
    p_win_rejected = float(MET_TABLE[my_after_reject, opp_left])

    obs_double = _set_obs_double_state(obs_now)
    p_win_accepted = _head_win_eval(probs_after_double, obs_double)

    p_accept = float(np.clip(np.asarray(probs_now, dtype=np.float32)[MATCH_VECTOR_DIM * 2], 0.0, 1.0))
    p_win_double = p_accept * p_win_accepted + (1.0 - p_accept) * p_win_rejected
    return int(p_win_double > p_win)


def _set_obs_opponent_double_offer(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32).copy()
    if x.size >= 7:
        x[-7] = x[-7] * 2.0
    if x.size >= 4:
        x[-4] = 1.0
    if x.size >= 3:
        x[-3] = 0.0
    if x.size >= 1:
        x[-1] = 1.0
    return x


def _reject_double_match_win_prob(obs_now: np.ndarray) -> float:
    my_left, opp_left, dave_val, _, _ = _extract_obs_controls(obs_now)
    opp_after = max(opp_left - int(max(dave_val, 1)), 0)
    return float(MET_TABLE[my_left, opp_after])


def _is_endless_state(raw_state: np.ndarray) -> bool:
    rs = np.asarray(raw_state, dtype=np.int16).reshape(-1)
    return rs.size > 56 and int(rs[56]) < 0

def _decide_accept_double_from_probs(probs_if_opp_doubles: np.ndarray, obs_now: np.ndarray) -> int:
    my_left, opp_left, _, _, opp_double_avail = _extract_obs_controls(obs_now)
    if opp_double_avail <= 0:
        return 0
    p_accept = _head_win_eval(probs_if_opp_doubles, _set_obs_opponent_double_offer(obs_now))
    p_reject = _reject_double_match_win_prob(obs_now)
    return int(p_accept >= p_reject)

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
    winner_player_index: int = 0
    points_won: int = 1
    reward_value: int = 1


@dataclass
class _GameSpec:
    game_id: str
    p1: object
    p2: object


class RandomAgent:
    agent_id = "random"

    def select(self, env) -> np.ndarray:
        moves = env.legal_moves()
        if len(moves) == 0:
            return pass_move()
        return moves[np.random.randint(len(moves))]


class ConservativeBaselineAgent:
    agent_id = "conservative_baseline"

    @staticmethod
    def _all_in_home(points: np.ndarray) -> bool:
        return bool(np.sum(points[6:]) == 0)

    @staticmethod
    def _to_mover_perspective(before_state: np.ndarray, after_state: np.ndarray) -> np.ndarray:
        if int(after_state[52]) == int(before_state[52]) + 1:
            converted = np.asarray(after_state, dtype=np.int16).copy()
            converted[:24] = after_state[24:48][::-1]
            converted[24:48] = after_state[:24][::-1]
            converted[48] = after_state[50]
            converted[49] = after_state[51]
            converted[50] = after_state[48]
            converted[51] = after_state[49]
            converted[52] = before_state[52]
            return converted
        return after_state

    @staticmethod
    def _dangerous_home_blots(raw_state: np.ndarray) -> int:
        points = raw_state[:24]
        opp_points = raw_state[24:48]
        opp_bar = int(raw_state[50])
        home_blots = np.where(points[:6] == 1)[0]
        if home_blots.size == 0:
            return 0
        if opp_bar > 0:
            return int(home_blots.size)
        dangerous = 0
        for idx in home_blots:
            if np.any(opp_points[:idx] > 0):
                dangerous += 1
        return dangerous

    def _score_move(self, before_state: np.ndarray, after_state: np.ndarray) -> tuple[float, ...]:
        before_state = np.asarray(before_state, dtype=np.int16)
        after_state = self._to_mover_perspective(before_state, np.asarray(after_state, dtype=np.int16))

        before_points = before_state[:24]
        points = after_state[:24]
        opp_bar_before = int(before_state[50])
        opp_bar_after = int(after_state[50])

        home_anchors = int(np.sum(points[:6] >= 2))
        total_anchors = int(np.sum(points >= 2))

        blot_idxs = np.where(points == 1)[0]
        blots = int(blot_idxs.size)
        blot_distance_sum = int(np.sum(blot_idxs))
        home_blots = int(np.sum(blot_idxs < 6))
        opp_home_blots = int(np.sum(blot_idxs >= 18))
        hits = int(max(0, opp_bar_after - opp_bar_before))
        off = int(after_state[49])

        moved_from = np.where(before_points > points)[0]
        moved_from_outside_home = int(np.sum(moved_from >= 6))
        moved_from_outside_home_pips = int(np.sum(moved_from[moved_from >= 6]))
        moved_from_home = int(np.sum(moved_from < 6))

        in_bearoff = self._all_in_home(before_points)
        if in_bearoff:
            dangerous_home_blots = self._dangerous_home_blots(after_state)
            return (
                float(-dangerous_home_blots),
                float(off),
                float(hits),
                float(-home_blots),
                float(opp_home_blots),
                float(-blots),
                float(moved_from_outside_home),
                float(moved_from_outside_home_pips),
                float(-moved_from_home),
                float(blot_distance_sum),
                float(home_anchors),
                float(total_anchors),
            )

        return (
            float(home_anchors),
            float(hits),
            float(-home_blots),
            float(opp_home_blots),
            float(-blots),
            float(moved_from_outside_home),
            float(moved_from_outside_home_pips),
            float(-moved_from_home),
            float(blot_distance_sum),
            float(total_anchors),
            float(off),
        )

    def select(self, env) -> np.ndarray:
        moves = env.legal_moves()
        if len(moves) == 0:
            return pass_move()
        if len(moves) == 1:
            return moves[0]

        before_state = np.asarray(env.get_state_raw(), dtype=np.int16)
        sim = bg_env.Env(0) if bg_env is not None else _FallbackEnv(0)
        best_idx = 0
        best_score: tuple[float, ...] | None = None
        for idx, mv in enumerate(moves):
            sim.set_state_raw(before_state)
            sim.step_move(mv)
            after_state = np.asarray(sim.get_state_raw(), dtype=np.int16)
            score = self._score_move(before_state, after_state)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        return moves[best_idx]


def pass_move() -> np.ndarray:
    return np.full((8,), 255, dtype=np.uint8)


def _unwrap_legal_moves_entry(entry) -> np.ndarray:
    """Normalize legal_moves entry from env/batched_env to ndarray[(N,8), uint8]."""
    moves = entry[1] if isinstance(entry, tuple) else entry
    arr = np.asarray(moves, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise RuntimeError(f"Unexpected legal_moves format: shape={arr.shape}, type={type(entry)}")
    return arr

class LeagueController:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.random = RandomAgent()
        self.conservative_baseline = ConservativeBaselineAgent()
        self.decision_temperature = float(getattr(cfg, "selfplay_temperature", 0.0))
        self.choose_best_probability = float(np.clip(getattr(cfg, "choose_best_probability", 0.5), 0.0, 1.0))
        self._decision_topk_hits = np.zeros((10,), dtype=np.float64)
        self._decision_count = 0
        self._baseline_eval_agent: ValueAgent | None = None
        self._ensemble_base_model_cache: dict[tuple, torch.nn.Module] = {}
        self._obs_probe_env = None
        self._move_eval_env = bg_env.Env(int(seed) + 11) if bg_env is not None else _FallbackEnv(int(seed) + 11)
        self._accept_eval_env = bg_env.Env(int(seed) + 17) if bg_env is not None else _FallbackEnv(int(seed) + 17)
        if bg_env is not None:
            try:
                self._obs_probe_env = bg_env.Env(int(seed), n_games=int(getattr(self.cfg, "games_in_match", 11)))
            except TypeError:
                try:
                    self._obs_probe_env = bg_env.Env(int(seed))
                except Exception:
                    self._obs_probe_env = None

    def set_decision_temperature(self, temperature: float) -> None:
        self.decision_temperature = float(temperature)

    def set_choose_best_probability(self, choose_best_probability: float) -> None:
        self.choose_best_probability = float(np.clip(choose_best_probability, 0.0, 1.0))

    def reset_decision_stats(self) -> None:
        self._decision_topk_hits.fill(0.0)
        self._decision_count = 0

    def get_decision_stats(self) -> dict:
        if self._decision_count <= 0:
            freqs = np.zeros((10,), dtype=np.float64)
        else:
            freqs = self._decision_topk_hits / float(self._decision_count)
        return {
            "decision_count": int(self._decision_count),
            "topk_freq": freqs.astype(np.float32).tolist(),
        }

    def _sample_action_index(self, values: np.ndarray) -> int:
        vals = np.asarray(values, dtype=np.float64)
        if vals.size == 0:
            return 0

        if self.rng.random() < float(np.clip(self.choose_best_probability, 0.0, 1.0)):
            return int(np.argmax(vals))

        temp = float(self.decision_temperature)
        if temp <= 0.0 or not np.all(np.isfinite(vals)):
            return int(np.argmax(vals))

        inv_temp = 1.0 / max(temp, 1e-8)
        logits = (vals - float(np.max(vals))) * inv_temp
        exp_logits = np.exp(logits)
        probs_sum = float(np.sum(exp_logits))
        if not np.isfinite(probs_sum) or probs_sum <= 0.0:
            return int(np.argmax(vals))

        probs = exp_logits / probs_sum
        idx = int(self.rng.choice(len(vals), p=probs))
        return idx

    def _record_topk_hit(self, values: np.ndarray, selected_idx: int) -> None:
        sorted_idx = np.argsort(-values, kind="mergesort")
        rank = int(np.flatnonzero(sorted_idx == selected_idx)[0]) + 1
        for k in range(1, 11):
            if rank <= k:
                self._decision_topk_hits[k - 1] += 1.0
        self._decision_count += 1

    @staticmethod
    def state_vector(env) -> np.ndarray:
        if bg_env is not None and hasattr(env, "get_obs_extended"):
            return np.asarray(env.get_obs_extended(), dtype=np.float32)
        raw = np.asarray(env.get_state_raw(), dtype=np.float32)
        return state_to_observation(raw)

    def state_vector_from_raw(self, raw: np.ndarray) -> np.ndarray:
        raw_np = np.asarray(raw, dtype=np.int16)
        if self._obs_probe_env is not None and hasattr(self._obs_probe_env, "set_state_raw") and hasattr(self._obs_probe_env, "get_obs_extended"):
            try:
                self._obs_probe_env.set_state_raw(raw_np)
                return np.asarray(self._obs_probe_env.get_obs_extended(), dtype=np.float32)
            except Exception:
                pass
        return state_to_observation(np.asarray(raw_np, dtype=np.float32))

    def _score_random(self, moves: np.ndarray) -> np.ndarray:
        if len(moves) == 0:
            return pass_move()
        return moves[np.random.randint(len(moves))]

    def _score_conservative_baseline(self, state: np.ndarray, moves: np.ndarray) -> np.ndarray:
        if len(moves) == 0:
            return pass_move()
        if len(moves) == 1:
            return moves[0]

        before_state = np.asarray(state, dtype=np.int16)
        sim = bg_env.Env(int(self.seed)) if bg_env is not None else _FallbackEnv(int(self.seed))

        best_idx = 0
        best_score: tuple[float, ...] | None = None
        for idx, mv in enumerate(moves):
            sim.set_state_raw(before_state)
            sim.step_move(mv)
            after_state = np.asarray(sim.get_state_raw(), dtype=np.int16)
            score = self.conservative_baseline._score_move(before_state, after_state)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        return moves[best_idx]

    def _decide_doubles_for_fixed_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        evaluator: ValueAgent,
        obs_batch: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(len(actions))
        apply_doubles = np.zeros((n,), dtype=np.uint8)
        accept_doubles = np.zeros((n,), dtype=np.uint8)
        if n == 0:
            return apply_doubles, accept_doubles

        base_obs: list[np.ndarray] = []
        for i in range(n):
            if obs_batch is not None:
                base_obs.append(np.asarray(obs_batch[i], dtype=np.float32))
            else:
                base_obs.append(self.state_vector_from_raw(states[i]))

        eval_agents = [evaluator for _ in range(n)]
        probs_now = self._predict_probs_single_cuda_call(eval_agents, np.stack(base_obs).astype(np.float32))
        obs_double_batch = np.stack([_set_obs_double_state(x) for x in base_obs]).astype(np.float32)
        probs_double = self._predict_probs_single_cuda_call(eval_agents, obs_double_batch)
        for i in range(n):
            if _is_endless_state(states[i]):
                p_accept = float(np.clip(probs_now[i][MATCH_VECTOR_DIM * 2], 0.0, 1.0))
                exp_no_double = _reward_expectation(probs_now[i])
                exp_double = p_accept * 2.0 * _reward_expectation(probs_double[i]) + (1.0 - p_accept) * 1.0
                apply_doubles[i] = int(exp_double > exp_no_double)
            else:
                apply_doubles[i] = _decide_apply_double(probs_now[i], probs_double[i], base_obs[i])

        accept_eval_obs: list[np.ndarray] = []
        accept_eval_owner: list[int] = []
        sim2 = bg_env.Env(int(self.seed) + 19) if bg_env is not None else _FallbackEnv(int(self.seed) + 19)
        for i in range(n):
            mv = np.asarray(actions[i], dtype=np.uint8)
            if int(mv[0]) == 255:
                continue
            sim2.set_state_raw(states[i])
            try:
                sim2.step_move(mv, apply_double=int(apply_doubles[i]), accept_double=1)
            except TypeError:
                sim2.step_move(mv)
            post_obs = self.state_vector(sim2)
            _, _, _, _, opp_double_avail = _extract_obs_controls(post_obs)
            if opp_double_avail <= 0:
                continue
            accept_eval_obs.append(_set_obs_opponent_double_offer(post_obs))
            accept_eval_owner.append(i)

        if accept_eval_obs:
            probs_accept = self._predict_probs_single_cuda_call([evaluator for _ in range(len(accept_eval_obs))], np.stack(accept_eval_obs).astype(np.float32))
            for owner, p_row, obs_h in zip(accept_eval_owner, probs_accept, accept_eval_obs):
                obs_pre = np.asarray(obs_h, dtype=np.float32).copy()
                if obs_pre.size >= 7:
                    obs_pre[-7] = obs_pre[-7] / 2.0
                if obs_pre.size >= 4:
                    obs_pre[-4] = 0.0
                if obs_pre.size >= 3:
                    obs_pre[-3] = 1.0
                if _is_endless_state(states[owner]):
                    exp_keep = -_reward_expectation(p_row)
                    exp_double = -2.0 * _reward_expectation(p_row)
                    accept_doubles[owner] = int(exp_double >= exp_keep)
                else:
                    accept_doubles[owner] = _decide_accept_double_from_probs(p_row, obs_pre)

        return apply_doubles, accept_doubles

    def _predict_probs_single_cuda_call(self, agents_for_samples: list[ValueAgent], obs_np: np.ndarray) -> np.ndarray:
        """Predict probabilities for mixed-agent samples with one CUDA call per architecture group."""
        if len(obs_np) == 0:
            return np.zeros((0, MODEL_OUTPUT_DIM), dtype=np.float32)

        first_agent = agents_for_samples[0]
        device = first_agent.device
        for ag in agents_for_samples:
            if ag.device != device:
                raise RuntimeError("Mixed devices in one architecture group are not supported")

        unique_agents: list[ValueAgent] = []
        agent_to_idx: dict[str, int] = {}
        model_idx = np.empty((len(agents_for_samples),), dtype=np.int64)
        for i, ag in enumerate(agents_for_samples):
            if ag.agent_id not in agent_to_idx:
                agent_to_idx[ag.agent_id] = len(unique_agents)
                unique_agents.append(ag)
            model_idx[i] = agent_to_idx[ag.agent_id]

        expected_dims = sorted({int(getattr(ag.model.cfg, "input_dim", -1)) for ag in agents_for_samples})
        model_in_dim = int(getattr(first_agent.model.cfg, "input_dim", obs_np.shape[1]))
        obs_arr = np.asarray(obs_np, dtype=np.float32)
        if obs_arr.ndim != 2 or int(obs_arr.shape[1]) != model_in_dim:
            sample_agents = [ag.agent_id for ag in agents_for_samples[:5]]
            raise RuntimeError(
                "Observation dim mismatch in league inference: "
                f"got batch shape={obs_arr.shape}, expected input_dim={model_in_dim}, "
                f"dims_in_group={expected_dims}, sample_agents={sample_agents}. "
                "Root cause is usually config/input mismatch (e.g. model_group_*.input_dim vs env.get_obs_extended dim). "
                "Verify checkpoint config and current environment build use same observation layout."
            )
        x_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)

        # Fast path: one model in group -> ordinary single forward.
        if len(unique_agents) == 1:
            probs = unique_agents[0].predict_proba_tensor(x_t)
            return probs.detach().cpu().numpy()

        # Vectorized ensemble forward: one CUDA call for all models in group.
        if functional_call is not None and stack_module_state is not None and vmap is not None:
            models = [ag.model for ag in unique_agents]

            # stack_module_state требует одинаковый train/eval режим у всех моделей.
            prev_modes = [m.training for m in models]
            try:
                for m in models:
                    m.eval()

                m0 = models[0]
                cfg0 = getattr(m0, "cfg", None)
                cache_key = (
                    type(m0),
                    str(device),
                    int(getattr(cfg0, "input_dim", -1)),
                    int(getattr(cfg0, "output_dim", -1)),
                )
                base_model = self._ensemble_base_model_cache.get(cache_key)
                if base_model is None:
                    base_model = copy.deepcopy(m0).to(device)
                    base_model.eval()
                    self._ensemble_base_model_cache[cache_key] = base_model
                params, buffers = stack_module_state(models)

                def _fmodel(p, b, x):
                    return functional_call(base_model, (p, b), (x,))

                logits_all = vmap(_fmodel, in_dims=(0, 0, None))(params, buffers, x_t)
                my = torch.softmax(logits_all[:, :, :MATCH_VECTOR_DIM], dim=-1)
                opp = torch.softmax(logits_all[:, :, MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2], dim=-1)
                acc = torch.sigmoid(logits_all[:, :, MATCH_VECTOR_DIM * 2: MATCH_VECTOR_DIM * 2 + 1])
                rew = torch.softmax(logits_all[:, :, MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM], dim=-1)
                probs_all = torch.cat([my, opp, acc, rew], dim=-1)
                model_idx_t = torch.as_tensor(model_idx, dtype=torch.long, device=device)
                sample_idx_t = torch.arange(x_t.shape[0], dtype=torch.long, device=device)
                probs = probs_all[model_idx_t, sample_idx_t, :]
                return probs.detach().cpu().numpy()
            finally:
                for m, was_training in zip(models, prev_modes):
                    m.train(was_training)

        # Conservative fallback when torch.func is unavailable.
        out = np.empty((len(agents_for_samples), MODEL_OUTPUT_DIM), dtype=np.float32)
        for ag_id, m_idx in agent_to_idx.items():
            sel = np.where(model_idx == m_idx)[0]
            p = unique_agents[m_idx].predict_proba_tensor(x_t[torch.as_tensor(sel, dtype=torch.long, device=device)])
            out[sel] = p.detach().cpu().numpy()
        return out

    def _select_group_actions_single_call(self, states: np.ndarray, legal_moves_list: list[np.ndarray], actors: list[ValueAgent], obs_batch: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.full((len(legal_moves_list), 8), 255, dtype=np.uint8)
        apply_doubles = np.zeros((len(legal_moves_list),), dtype=np.uint8)
        accept_doubles = np.zeros((len(legal_moves_list),), dtype=np.uint8)

        if obs_batch is not None:
            base_obs = np.asarray(obs_batch, dtype=np.float32)
        else:
            base_obs = np.stack([self.state_vector_from_raw(states[i]) for i in range(len(legal_moves_list))]).astype(np.float32)

        if actors:
            probs_now = self._predict_probs_single_cuda_call(actors, base_obs)
            obs_double_batch = np.stack([_set_obs_double_state(x) for x in base_obs]).astype(np.float32)
            probs_double = self._predict_probs_single_cuda_call(actors, obs_double_batch)
            for i in range(len(actors)):
                if _is_endless_state(states[i]):
                    p_accept = float(np.clip(probs_now[i][MATCH_VECTOR_DIM * 2], 0.0, 1.0))
                    exp_no_double = _reward_expectation(probs_now[i])
                    exp_double = p_accept * 2.0 * _reward_expectation(probs_double[i]) + (1.0 - p_accept) * 1.0
                    apply_doubles[i] = int(exp_double > exp_no_double)
                else:
                    apply_doubles[i] = _decide_apply_double(probs_now[i], probs_double[i], base_obs[i])

        candidate_obs: list[np.ndarray] = []
        candidate_done: list[bool] = []
        candidate_owner: list[int] = []
        candidate_actor: list[ValueAgent] = []

        sim = self._move_eval_env
        for i, moves in enumerate(legal_moves_list):
            if len(moves) == 0:
                continue
            if len(moves) == 1:
                actions[i] = moves[0]
                continue
            state_i = states[i]
            for mv in moves:
                sim.set_state_raw(state_i)
                try:
                    _, _, _, done = sim.step_move(np.asarray(mv, dtype=np.uint8), apply_double=int(apply_doubles[i]), accept_double=1)
                except TypeError:
                    _, done = sim.step_move(np.asarray(mv, dtype=np.uint8))
                candidate_obs.append(self.state_vector(sim))
                candidate_done.append(done)
                candidate_owner.append(i)
                candidate_actor.append(actors[i])

        probs = None
        if candidate_obs:
            candidate_obs_np = np.stack(candidate_obs).astype(np.float32)
            probs = self._predict_probs_single_cuda_call(candidate_actor, candidate_obs_np)
            vals = []
            done_np = np.asarray(candidate_done, dtype=bool)
            for is_done, p_row, obs_row in zip(done_np, probs, candidate_obs_np):
                if is_done:
                    vals.append(0.0)
                else:
                    vals.append(_head_win_eval(p_row, obs_row))

            grouped_vals: dict[int, list[float]] = {}
            for owner, val in zip(candidate_owner, vals):
                grouped_vals.setdefault(owner, []).append(float(val))
            for i, moves in enumerate(legal_moves_list):
                if len(moves) <= 1:
                    continue
                values_i = np.asarray(grouped_vals[i], dtype=np.float32)
                if _is_endless_state(states[i]):
                    values_i = -np.asarray([_reward_expectation(probs[j]) for j,o in enumerate(candidate_owner) if o==i], dtype=np.float32)
                    selected_idx = self._sample_action_index(values_i)
                    self._record_topk_hit(values_i, selected_idx)
                else:
                    selected_idx = self._sample_action_index(-values_i)
                    self._record_topk_hit(-values_i, selected_idx)
                actions[i] = moves[selected_idx]

        # Decide accept_double for opponent's potential next-turn double using post-move state
        accept_eval_obs: list[np.ndarray] = []
        accept_eval_actor: list[ValueAgent] = []
        accept_eval_owner: list[int] = []
        sim2 = self._accept_eval_env
        for i, mv in enumerate(actions):
            if int(mv[0]) == 255:
                accept_doubles[i] = 0
                continue
            sim2.set_state_raw(states[i])
            try:
                sim2.step_move(np.asarray(mv, dtype=np.uint8), apply_double=int(apply_doubles[i]), accept_double=1)
            except TypeError:
                sim2.step_move(np.asarray(mv, dtype=np.uint8))
            post_obs = self.state_vector(sim2)
            _, _, _, _, opp_double_avail = _extract_obs_controls(post_obs)
            if opp_double_avail <= 0:
                accept_doubles[i] = 0
                continue
            accept_eval_obs.append(_set_obs_opponent_double_offer(post_obs))
            accept_eval_actor.append(actors[i])
            accept_eval_owner.append(i)

        if accept_eval_obs:
            accept_eval_obs_np = np.stack(accept_eval_obs).astype(np.float32)
            probs_accept = self._predict_probs_single_cuda_call(accept_eval_actor, accept_eval_obs_np)
            for owner, p_row, obs_h in zip(accept_eval_owner, probs_accept, accept_eval_obs_np):
                # reject threshold uses pre-offer post-move state (halve cube back)
                obs_pre = np.asarray(obs_h, dtype=np.float32).copy()
                if obs_pre.size >= 7:
                    obs_pre[-7] = obs_pre[-7] / 2.0
                if obs_pre.size >= 4:
                    obs_pre[-4] = 0.0
                if obs_pre.size >= 3:
                    obs_pre[-3] = 1.0
                if _is_endless_state(states[owner]):
                    exp_keep = -_reward_expectation(p_row)
                    exp_double = -2.0 * _reward_expectation(p_row)
                    accept_doubles[owner] = int(exp_double >= exp_keep)
                else:
                    accept_doubles[owner] = _decide_accept_double_from_probs(p_row, obs_pre)

        return actions, apply_doubles, accept_doubles

    def _play_all_games_batched(self, game_specs: list[_GameSpec], epoch: int) -> list[GameResult]:
        if batched_bg_env is None:
            return [self.play_game(spec.p1, spec.p2, spec.game_id, epoch) for spec in game_specs]

        n_games = len(game_specs)
        env = batched_bg_env.Env(
            n_matches=n_games,
            n_games=int(getattr(self.cfg, "games_in_match", 11)),
            seed=self.seed + epoch * 100_000,
        )
        env.reset()

        histories = [[] for _ in range(n_games)]
        turns = [0 for _ in range(n_games)]
        finished_games = [0 for _ in range(n_games)]
        game_results: list[GameResult] = []
        done = np.zeros((n_games,), dtype=bool)

        cfg_games_in_match = int(getattr(self.cfg, "games_in_match", 11))
        target_games_in_match = int(self.cfg.matches_per_pair) if cfg_games_in_match < 0 else max(1, cfg_games_in_match)

        turn = 0
        while True:
            active_idx = np.flatnonzero(~done)
            if len(active_idx) == 0:
                break

            t0= time.time()
            env.roll_dice()
            states = np.asarray(env.get_states_raw(), dtype=np.int16)
            obs_extended_batch = None
            if hasattr(env, "get_obs_extended"):
                obs_extended_batch = np.asarray(env.get_obs_extended(getattr(self.cfg, "batched_obs_threads", 0)), dtype=np.float32)
            legal_moves_raw = list(env.legal_moves())
            legal_moves = [_unwrap_legal_moves_entry(x) for x in legal_moves_raw]
            white_to_move = (states[:, 57] > 0) if states.shape[1] > 57 else ((turn % 2) == 0) * np.ones((n_games,), dtype=bool)
            actions = np.full((n_games, 8), 255, dtype=np.uint8)
            apply_doubles = np.zeros((n_games,), dtype=np.uint8)
            accept_doubles = np.zeros((n_games,), dtype=np.uint8)

            by_group: dict[str, list[int]] = {"A": [], "C": [], "D": []}
            baseline_idxs: list[int] = []
            dave_before = np.maximum(states[:, 55].astype(np.int32), 1) if states.shape[1] > 55 else np.ones((n_games,), dtype=np.int32)

            for i in active_idx:
                spec = game_specs[int(i)]
                actor = spec.p1 if bool(white_to_move[i]) else spec.p2
                if isinstance(actor, ValueAgent):
                    by_group[actor.group].append(int(i))
                elif actor.agent_id == self.conservative_baseline.agent_id:
                    actions[i] = self._score_conservative_baseline(states[i], legal_moves[i])
                    baseline_idxs.append(int(i))
                else:
                    actions[i] = self._score_random(legal_moves[i])

            for group_name in ("A", "C", "D"):
                idxs = by_group[group_name]
                if not idxs:
                    continue

                idxs_np = np.asarray(idxs, dtype=np.int64)
                local_states = states[idxs_np]
                local_moves = [legal_moves[i] for i in idxs]
                local_actors = [(game_specs[i].p1 if bool(white_to_move[i]) else game_specs[i].p2) for i in idxs]
                local_obs = obs_extended_batch[idxs_np] if obs_extended_batch is not None else None
                local_actions, local_apply, local_accept = self._select_group_actions_single_call(local_states, local_moves, local_actors, local_obs)
                actions[idxs_np] = local_actions
                apply_doubles[idxs_np] = local_apply
                accept_doubles[idxs_np] = local_accept

            if baseline_idxs and self._baseline_eval_agent is not None:
                b_idx = np.asarray(baseline_idxs, dtype=np.int64)
                b_states = states[b_idx]
                b_actions = actions[b_idx]
                b_obs = obs_extended_batch[b_idx] if obs_extended_batch is not None else None
                b_apply, b_accept = self._decide_doubles_for_fixed_actions(b_states, b_actions, self._baseline_eval_agent, b_obs)
                apply_doubles[b_idx] = b_apply
                accept_doubles[b_idx] = b_accept

            try:
                step_ret = env.step_apply(actions, apply_doubles, accept_doubles)
            except TypeError:
                step_ret = env.step_apply(actions)
            if isinstance(step_ret, tuple) and len(step_ret) >= 2:
                rewards, done_step = step_ret[0], step_ret[-1]
                accepted_step = step_ret[2] if len(step_ret) >= 3 else np.zeros((n_games,), dtype=np.uint8)
            else:
                rewards, done_step = step_ret
                accepted_step = np.zeros((n_games,), dtype=np.uint8)
            rewards = np.asarray(rewards, dtype=np.float32)
            done_code = np.asarray(done_step, dtype=np.uint8)
            accepted_step = np.asarray(accepted_step, dtype=np.uint8)
            states_after = None
            if np.any(done_code[active_idx] > 0):
                states_after = np.asarray(env.get_states_raw(), dtype=np.int16)

            for i in active_idx:
                spec = game_specs[int(i)]
                actor = spec.p1 if bool(white_to_move[i]) else spec.p2
                opp = spec.p2 if bool(white_to_move[i]) else spec.p1
                actor_player_index = 0 if bool(white_to_move[i]) else 1
                state_vector = (
                    obs_extended_batch[i].copy()
                    if obs_extended_batch is not None
                    else self.state_vector_from_raw(states[i])
                )
                histories[i].append({
                    "state_vector": state_vector,
                    "agent_id": actor.agent_id,
                    "opponent_id": opp.agent_id,
                    "game_id": spec.game_id,
                    "step_index": turns[i],
                    "player_index": actor_player_index,
                    "epoch": epoch,
                    "double_offered_by_agent": bool(apply_doubles[i]),
                    "double_was_accepted": bool(accepted_step[i]) if bool(apply_doubles[i]) else False,
                    "accept_double_opponent": bool(accept_doubles[i]),
                })
                turns[i] += 1
                if int(done_code[i]) in (1, 2):
                    winner = actor.agent_id if rewards[i] > 0 else opp.agent_id
                    winner_player_index = actor_player_index if rewards[i] > 0 else (1 - actor_player_index)
                    reward_value = max(1, int(round(abs(float(rewards[i])))))
                    points_won = max(1, int(round(abs(float(rewards[i])) * int(dave_before[i]))))
                    finished_games[i] += 1

                    game_results.append(
                        GameResult(
                            game_id=f"{spec.game_id}_g{finished_games[i]}",
                            steps=histories[i],
                            winner=winner,
                            turns=turns[i],
                            player_1_id=spec.p1.agent_id,
                            player_2_id=spec.p2.agent_id,
                            winner_player_index=winner_player_index,
                            points_won=points_won,
                            reward_value=reward_value,
                        )
                    )

                    white_score = int(states_after[i][53]) if states_after is not None and states_after.shape[1] > 53 else -1
                    black_score = int(states_after[i][54]) if states_after is not None and states_after.shape[1] > 54 else -1
                    dave_after = int(states_after[i][55]) if states_after is not None and states_after.shape[1] > 55 else int(dave_before[i])
                    # print(
                    #     "[self-play] game finished "
                    #     f"pair={spec.p1.agent_id} vs {spec.p2.agent_id}, "
                    #     f"match={spec.game_id}, game_in_match={finished_games[i]}, "
                    #     f"winner={winner}, reward={float(rewards[i]):.3f}, dave={dave_after}, "
                    #     f"match_score={white_score}:{black_score}"
                    # )

                    histories[i] = []
                    turns[i] = 0
                    if int(done_code[i]) == 2 or finished_games[i] >= target_games_in_match:
                        done[i] = True
                    
            dt = time.time() - t0
            # print(f"Ran step {turn} via {dt} sec")
            turn += 1

        return game_results

    def play_game(self, p1, p2, game_id: str, epoch: int):
        env = (
            bg_env.Env(self.seed + hash(game_id) % 100000, n_games=int(getattr(self.cfg, "games_in_match", 11)))
            if bg_env is not None
            else _FallbackEnv(self.seed + hash(game_id) % 100000)
        )
        env.reset()
        history = []
        players = [p1, p2]
        turn = 0
        done = False
        winner = players[0].agent_id
        winner_player_index = 0
        points_won = 1
        reward_value = 1
        cfg_games_in_match = int(getattr(self.cfg, "games_in_match", 11))
        target_games_in_match = int(self.cfg.matches_per_pair) if cfg_games_in_match < 0 else max(1, cfg_games_in_match)
        finished_games = 0
        while not done:
            env.roll_dice()
            raw_turn = np.asarray(env.get_state_raw(), dtype=np.int16)
            white_to_move = bool(raw_turn[57]) if raw_turn.shape[0] > 57 else ((turn % 2) == 0)
            actor_player_index = 0 if white_to_move else 1
            actor = players[actor_player_index]
            opp = players[1 - actor_player_index]
            state = self.state_vector(env)
            if isinstance(actor, ValueAgent):
                lm = env.legal_moves()
                local_moves = lm[1] if isinstance(lm, tuple) else lm
                move, apply_double, accept_double = self._select_group_actions_single_call(
                    np.asarray([env.get_state_raw()], dtype=np.int16),
                    [local_moves],
                    [actor],
                    np.asarray([state], dtype=np.float32),
                )
                move = move[0]
                apply_double = int(apply_double[0])
                accept_double = int(accept_double[0])
            elif actor.agent_id == self.conservative_baseline.agent_id:
                lm = env.legal_moves()
                local_moves = lm[1] if isinstance(lm, tuple) else lm
                move = self._score_conservative_baseline(np.asarray(env.get_state_raw(), dtype=np.int16), local_moves)
                if self._baseline_eval_agent is not None:
                    b_apply, b_accept = self._decide_doubles_for_fixed_actions(
                        np.asarray([env.get_state_raw()], dtype=np.int16),
                        np.asarray([move], dtype=np.uint8),
                        self._baseline_eval_agent,
                        np.asarray([state], dtype=np.float32),
                    )
                    apply_double = int(b_apply[0])
                    accept_double = int(b_accept[0])
                else:
                    apply_double = 0
                    accept_double = 0
            else:
                lm = env.legal_moves()
                local_moves = lm[1] if isinstance(lm, tuple) else lm
                move = self._score_random(local_moves)
                apply_double = 0
                accept_double = 0
            raw_before = np.asarray(env.get_state_raw(), dtype=np.int16)
            dave_value = int(raw_before[55]) if raw_before.shape[0] > 55 else 1
            try:
                reward, _dave_after, accepted, done = env.step_move(move, apply_double=apply_double, accept_double=accept_double)
            except TypeError:
                reward, done = env.step_move(move)
                accepted = 0
            history.append({
                "state_vector": state,
                "agent_id": actor.agent_id,
                "opponent_id": opp.agent_id,
                "game_id": game_id,
                "step_index": turn,
                "player_index": actor_player_index,
                "epoch": epoch,
                "double_offered_by_agent": bool(apply_double),
                "double_was_accepted": bool(accepted) if bool(apply_double) else False,
                "accept_double_opponent": bool(accept_double),
            })
            if int(done) in (1, 2):
                winner = actor.agent_id if reward > 0 else opp.agent_id
                winner_player_index = actor_player_index if reward > 0 else (1 - actor_player_index)
                reward_value = max(1, int(round(abs(float(reward)))))
                points_won = max(1, int(round(abs(float(reward)) * max(dave_value, 1))))
                finished_games += 1
                done = bool(int(done) == 2 or finished_games >= target_games_in_match)
            else:
                done = False
            turn += 1
        return GameResult(
            game_id=game_id,
            steps=history,
            winner=winner,
            turns=turn,
            player_1_id=p1.agent_id,
            player_2_id=p2.agent_id,
            winner_player_index=winner_player_index,
            points_won=points_won,
            reward_value=reward_value,
        )

    def run_epoch(self, trainable_agents: list[ValueAgent], epoch: int):
        t0 = time.time()
        self.reset_decision_stats()
        if trainable_agents:
            self._baseline_eval_agent = trainable_agents[int(self.rng.integers(0, len(trainable_agents)))]
        else:
            self._baseline_eval_agent = None
        opponents = [self.conservative_baseline]
        specs: list[_GameSpec] = []

        endless_mode = int(getattr(self.cfg, "games_in_match", 11)) < 0
        matches_per_pair = 1 if endless_mode else int(self.cfg.matches_per_pair)

        for i, a in enumerate(trainable_agents):
            for j in range(i, len(trainable_agents)):
                b = trainable_agents[j]
                for g in range(matches_per_pair):
                    specs.append(_GameSpec(game_id=f"e{epoch}_t{i}_{b.agent_id}_{g}", p1=a, p2=b))
            for opp in opponents:
                for g in range(matches_per_pair):
                    specs.append(_GameSpec(game_id=f"e{epoch}_{a.agent_id}_{opp.agent_id}_{g}", p1=a, p2=opp))

        results = self._play_all_games_batched(specs, epoch)
        dt = max(time.time() - t0, 1e-6)
        
        return results, len(results) / dt
