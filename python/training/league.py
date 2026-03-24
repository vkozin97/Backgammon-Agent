from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np
import torch

from .agents import (
    ValueAgent,
    REWARD_VECTOR_DIM,
    decide_apply_double_from_probs,
    decide_accept_double_from_probs,
    extract_obs_controls,
    flip_observation_perspective,
    head_win_eval,
    reward_expectation,
    set_obs_double_state,
    set_obs_opponent_double_offer,
)
from .observation import OBSERVATION_DIM, state_to_observation

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


def _is_endless_state(raw_state: np.ndarray) -> bool:
    rs = np.asarray(raw_state, dtype=np.int16).reshape(-1)
    return rs.size > 56 and int(rs[56]) < 0


def _compact_agent_token(agent_id: str) -> str:
    if agent_id.startswith("trainable_"):
        return f"t{agent_id.split('_', 1)[1]}"
    if agent_id == "conservative_baseline":
        return "b"
    return agent_id

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

    def current_dice(self):
        return getattr(self, "dice", (1, 1))

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
    match_id: str
    match_number: int
    game_number_in_match: int
    steps: list[dict]
    winner: str
    turns: int
    player_1_id: str
    player_2_id: str
    winner_player_index: int = 0
    points_won: int = 1
    reward_value: int = 1
    final_dave_value: int = 1
    ended_by_double_reject: bool = False
    max_steps_reached: bool = False


@dataclass
class _GameSpec:
    match_id: str
    match_number: int
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
        self._agents_double_decision_enabled_by_agent: dict[str, bool] = {}
        self._ensemble_base_model_cache: dict[tuple, torch.nn.Module] = {}
        self._obs_probe_env = None
        self._move_eval_env = bg_env.Env(int(seed) + 11) if bg_env is not None else _FallbackEnv(int(seed) + 11)
        self._post_action_opponent_double_eval_env = (
            bg_env.Env(int(seed) + 17) if bg_env is not None else _FallbackEnv(int(seed) + 17)
        )
        if bg_env is not None:
            try:
                self._obs_probe_env = bg_env.Env(int(seed), n_games=int(getattr(self.cfg, "n_games_per_match", 11)), endless_mode=bool(getattr(self.cfg, "endless_mode", False)))
            except TypeError:
                try:
                    self._obs_probe_env = bg_env.Env(int(seed))
                except Exception:
                    self._obs_probe_env = None

    def set_decision_temperature(self, temperature: float) -> None:
        self.decision_temperature = float(temperature)

    def set_choose_best_probability(self, choose_best_probability: float) -> None:
        self.choose_best_probability = float(np.clip(choose_best_probability, 0.0, 1.0))

    def set_agents_double_decision_enabled(self, enabled: bool) -> None:
        self._agents_double_decision_enabled_by_agent = {k: bool(enabled) for k in self._agents_double_decision_enabled_by_agent}

    def _is_agent_double_decision_enabled(self, agent_id: str) -> bool:
        return bool(self._agents_double_decision_enabled_by_agent.get(agent_id, True))

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
        if n == 0:
            return np.zeros((n,), dtype=np.uint8), np.zeros((n,), dtype=np.uint8)

        base_obs = self._base_observations_for_states(states, obs_batch)
        eval_agents = [evaluator] * n
        apply_doubles = self._decide_apply_doubles(states, base_obs, eval_agents)
        accept_doubles = self._decide_accept_doubles_for_actions(states, actions, apply_doubles, eval_agents)
        return apply_doubles, accept_doubles

    def _base_observations_for_states(self, states: np.ndarray, obs_batch: np.ndarray | None = None) -> np.ndarray:
        if obs_batch is not None:
            return np.asarray(obs_batch, dtype=np.float32)
        return np.stack([self.state_vector_from_raw(states[i]) for i in range(len(states))]).astype(np.float32)

    def _decide_apply_doubles(
        self,
        states: np.ndarray,
        base_obs: np.ndarray,
        evaluators: list[ValueAgent],
        enabled_mask: list[bool] | np.ndarray | None = None,
    ) -> np.ndarray:
        n = int(len(evaluators))
        apply_doubles = np.zeros((n,), dtype=np.uint8)
        if n == 0:
            return apply_doubles

        if enabled_mask is None:
            enabled_idx = np.arange(n, dtype=np.int64)
        else:
            enabled_idx = np.flatnonzero(np.asarray(enabled_mask, dtype=bool))
        if enabled_idx.size == 0:
            return apply_doubles

        enabled_obs = np.asarray(base_obs[enabled_idx], dtype=np.float32)
        enabled_evaluators = [evaluators[int(i)] for i in enabled_idx]
        probs_now = self._predict_probs_single_cuda_call(enabled_evaluators, enabled_obs)
        probs_double = self._predict_probs_single_cuda_call(
            enabled_evaluators,
            np.stack([set_obs_double_state(x) for x in enabled_obs]).astype(np.float32),
        )
        for local_i, i in enumerate(enabled_idx):
            owner = int(i)
            apply_doubles[owner] = decide_apply_double_from_probs(
                probs_now[local_i],
                probs_double[local_i],
                enabled_obs[local_i],
                endless=_is_endless_state(states[owner]),
            )
        return apply_doubles

    def _decide_accept_doubles_for_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        apply_doubles: np.ndarray,
        evaluators: list[ValueAgent],
        enabled_mask: list[bool] | np.ndarray | None = None,
        accept_if_disabled: int = 1,
    ) -> np.ndarray:
        n = int(len(actions))
        accept_doubles = np.zeros((n,), dtype=np.uint8)
        if n == 0:
            return accept_doubles

        enabled_arr = None if enabled_mask is None else np.asarray(enabled_mask, dtype=bool)
        accept_eval_obs: list[np.ndarray] = []
        accept_eval_owner: list[int] = []
        accept_eval_evaluators: list[ValueAgent] = []
        sim = self._post_action_opponent_double_eval_env
        for i, mv in enumerate(actions):
            mv_arr = np.asarray(mv, dtype=np.uint8)
            if int(mv_arr[0]) == 255:
                continue
            sim.set_state_raw(states[i])
            try:
                sim.step_move(mv_arr, apply_double=int(apply_doubles[i]), accept_double=1)
            except TypeError:
                sim.step_move(mv_arr)
            post_obs = self.state_vector(sim)
            current_player_post_obs = flip_observation_perspective(post_obs)
            _, _, _, _, opp_double_avail = extract_obs_controls(current_player_post_obs)
            if opp_double_avail <= 0:
                continue
            if enabled_arr is not None and not bool(enabled_arr[i]):
                accept_doubles[i] = int(accept_if_disabled)
                continue
            accept_eval_obs.append(set_obs_opponent_double_offer(current_player_post_obs))
            accept_eval_owner.append(i)
            accept_eval_evaluators.append(evaluators[i])

        if accept_eval_obs:
            probs_accept = self._predict_probs_single_cuda_call(
                accept_eval_evaluators,
                np.stack(accept_eval_obs).astype(np.float32),
            )
            for owner, p_row, obs_h in zip(accept_eval_owner, probs_accept, accept_eval_obs):
                obs_pre = np.asarray(obs_h, dtype=np.float32).copy()
                if obs_pre.size >= 7:
                    obs_pre[-7] = obs_pre[-7] / 2.0
                if obs_pre.size >= 4:
                    obs_pre[-4] = 1.0
                if obs_pre.size >= 3:
                    obs_pre[-3] = 1.0
                if obs_pre.size >= 1:
                    obs_pre[-1] = 0.0
                accept_doubles[owner] = decide_accept_double_from_probs(
                    p_row,
                    obs_pre,
                    endless=_is_endless_state(states[owner]),
                )

        return accept_doubles

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
                    tuple(int(x) for x in getattr(cfg0, "conv_channels", [])),
                    tuple(int(x) for x in getattr(cfg0, "conv_kernel_sizes", [])),
                    int(getattr(cfg0, "conv_pool_every", 0)),
                    tuple(int(x) for x in getattr(cfg0, "hidden_dims", [])),
                    tuple(int(x) for x in getattr(cfg0, "head_hidden_dims", [])),
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
        base_obs = self._base_observations_for_states(states, obs_batch)

        enabled_for_double = [self._is_agent_double_decision_enabled(a.agent_id) for a in actors]
        apply_doubles = self._decide_apply_doubles(states, base_obs, actors, enabled_mask=enabled_for_double)

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
                    vals.append(head_win_eval(p_row, obs_row))

            grouped_vals: dict[int, list[float]] = {}
            for owner, val in zip(candidate_owner, vals):
                grouped_vals.setdefault(owner, []).append(float(val))
            for i, moves in enumerate(legal_moves_list):
                if len(moves) <= 1:
                    continue
                values_i = np.asarray(grouped_vals[i], dtype=np.float32)
                if _is_endless_state(states[i]):
                    values_i = -np.asarray([reward_expectation(probs[j]) for j,o in enumerate(candidate_owner) if o==i], dtype=np.float32)
                    selected_idx = self._sample_action_index(values_i)
                    self._record_topk_hit(values_i, selected_idx)
                else:
                    selected_idx = self._sample_action_index(-values_i)
                    self._record_topk_hit(-values_i, selected_idx)
                actions[i] = moves[selected_idx]

        accept_doubles = self._decide_accept_doubles_for_actions(
            states,
            actions,
            apply_doubles,
            actors,
            enabled_mask=enabled_for_double,
            accept_if_disabled=1,
        )
        return actions, apply_doubles, accept_doubles

    def _play_all_games_batched(self, game_specs: list[_GameSpec], epoch: int) -> list[GameResult]:
        if batched_bg_env is None:
            return [result for spec in game_specs for result in self.play_game(spec.p1, spec.p2, spec.match_id, epoch, spec.match_number)]

        n_games = len(game_specs)
        env = batched_bg_env.Env(
            n_matches=n_games,
            n_games=int(getattr(self.cfg, "n_games_per_match", 11)),
            endless_mode=bool(getattr(self.cfg, "endless_mode", False)),
            seed=self.seed + epoch * 100_000,
        )
        env.reset()

        histories = [[] for _ in range(n_games)]
        turns = [0 for _ in range(n_games)]
        finished_games = [0 for _ in range(n_games)]
        game_results: list[GameResult] = []
        done = np.zeros((n_games,), dtype=bool)

        target_games_in_match = max(1, int(getattr(self.cfg, "n_games_per_match", 1)))
        max_steps_per_game = max(1, int(getattr(self.cfg, "max_steps_per_game", 200)))

        turn = 0
        while True:
            active_idx = np.flatnonzero(~done)
            if len(active_idx) == 0:
                break

            t0= time.time()
            rolled_dice = np.asarray(env.roll_dice(), dtype=np.uint8)
            states = np.asarray(env.get_states_raw(), dtype=np.int16)
            obs_extended_batch = None
            if hasattr(env, "get_obs_extended"):
                obs_extended_batch = np.asarray(env.get_obs_extended(getattr(self.cfg, "batched_obs_threads", 0)), dtype=np.float32)
                
            # avg_pips_mine = round(obs_extended_batch[active_idx, 244].mean(), 1)
            # avg_pips_opp = round(obs_extended_batch[active_idx, 245].mean(), 1)
            # avg_bar_mine = round(obs_extended_batch[active_idx, 240].mean(), 1)
            # avg_bar_opp = round(obs_extended_batch[active_idx, 242].mean(), 1)
            # avg_off_mine = round(obs_extended_batch[active_idx, 241].mean(), 1)
            # avg_off_opp = round(obs_extended_batch[active_idx, 243].mean(), 1)
            # avg_blots_mine = round(obs_extended_batch[active_idx, 246].mean(), 1)
            # avg_blots_opp = round(obs_extended_batch[active_idx, 247].mean(), 1)
            
            # print(f"Turn {turn}, active players: {len(active_idx)}\n"
            #       f"pips:  {avg_pips_mine:.1f} \t {avg_pips_opp:.1f}\n"
            #       f"bar:   {avg_bar_mine:.1f} \t {avg_bar_opp:.1f}\n"
            #       f"off:   {avg_off_mine:.1f} \t {avg_off_opp:.1f}\n"
            #       f"blots: {avg_blots_mine:.1f} \t {avg_blots_opp:.1f}\n"
            #       )
            
            legal_moves_raw = list(env.legal_moves())
            legal_moves = [_unwrap_legal_moves_entry(x) for x in legal_moves_raw]
            white_to_move = (states[:, 57] > 0) if states.shape[1] > 57 else ((turn % 2) == 0) * np.ones((n_games,), dtype=bool)
            actions = np.full((n_games, 8), 255, dtype=np.uint8)
            apply_doubles = np.zeros((n_games,), dtype=np.uint8)
            accept_doubles = np.ones((n_games,), dtype=np.uint8)

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

            if baseline_idxs:
                b_idx = np.asarray(baseline_idxs, dtype=np.int64)
                if self._baseline_eval_agent is not None:
                    b_states = states[b_idx]
                    b_actions = actions[b_idx]
                    b_obs = obs_extended_batch[b_idx] if obs_extended_batch is not None else None
                    b_apply, b_accept = self._decide_doubles_for_fixed_actions(b_states, b_actions, self._baseline_eval_agent, b_obs)
                    apply_doubles[b_idx] = b_apply
                    accept_doubles[b_idx] = b_accept
                else:
                    apply_doubles[b_idx] = 0
                    accept_doubles[b_idx] = 1

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
                    "game_id": spec.match_id,
                    "match_id": spec.match_id,
                    "match_number": int(spec.match_number),
                    "game_number_in_match": int(finished_games[i] + 1),
                    "step_index": turns[i],
                    "player_index": actor_player_index,
                    "epoch": epoch,
                    "double_offered_by_agent": bool(apply_doubles[i]),
                    "double_was_accepted": bool(accepted_step[i]) if bool(apply_doubles[i]) else False,
                    "accept_double_opponent": bool(accept_doubles[i]),
                    "accept_double_opportunity": bool(states[i][65] >= 0) if states.shape[1] > 65 else False,
                    "action_meta": {
                        "dice": [int(rolled_dice[i][0]), int(rolled_dice[i][1])],
                        "move": [int(x) for x in actions[i].tolist()],
                        "apply_double": int(apply_doubles[i]),
                        "accept_double_for_next_offer": int(accept_doubles[i]),
                        "raw_state": [int(x) for x in states[i].tolist()],
                    },
                })
                turns[i] += 1
                game_finished = int(done_code[i]) in (1, 2)
                max_steps_reached = turns[i] >= max_steps_per_game
                if game_finished or max_steps_reached:
                    histories[i][-1]["action_meta"]["reward"] = float(rewards[i])
                    histories[i][-1]["action_meta"]["done_code"] = int(done_code[i])
                    histories[i][-1]["action_meta"]["double_was_accepted"] = int(accepted_step[i])
                    dave_after = int(states_after[i][55]) if states_after is not None and states_after.shape[1] > 55 else int(dave_before[i])
                    winner = actor.agent_id if rewards[i] > 0 else opp.agent_id
                    winner_player_index = actor_player_index if rewards[i] > 0 else (1 - actor_player_index)
                    reward_value = max(1, int(round(abs(float(rewards[i])))))
                    points_won = max(1, int(round(abs(float(rewards[i])) * int(dave_before[i]))))
                    finished_games[i] += 1

                    ended_by_double_reject = bool(states[i][65] >= 0 and int(accept_doubles[i]) == 0 and game_finished) if states.shape[1] > 65 else False
                    game_results.append(
                        GameResult(
                            game_id=f"{spec.match_id}_g{finished_games[i]}",
                            match_id=spec.match_id,
                            match_number=int(spec.match_number),
                            game_number_in_match=int(finished_games[i]),
                            steps=histories[i],
                            winner=winner,
                            turns=turns[i],
                            player_1_id=spec.p1.agent_id,
                            player_2_id=spec.p2.agent_id,
                            winner_player_index=winner_player_index,
                            points_won=points_won,
                            reward_value=reward_value,
                            final_dave_value=dave_after,
                            ended_by_double_reject=ended_by_double_reject,
                            max_steps_reached=max_steps_reached and not game_finished,
                        )
                    )

                    white_score = int(states_after[i][53]) if states_after is not None and states_after.shape[1] > 53 else -1
                    black_score = int(states_after[i][54]) if states_after is not None and states_after.shape[1] > 54 else -1
                    # print(
                    #     "[self-play] game finished "
                    #     f"pair={spec.p1.agent_id} vs {spec.p2.agent_id}, "
                    #     f"match={spec.game_id}, game_in_match={finished_games[i]}, "
                    #     f"winner={winner}, reward={float(rewards[i]):.3f}, dave={dave_after}, "
                    #     f"match_score={white_score}:{black_score}"
                    # )

                    histories[i] = []
                    turns[i] = 0
                    if max_steps_reached:
                        done[i] = True
                    elif int(done_code[i]) == 2 or finished_games[i] >= target_games_in_match:
                        done[i] = True
                    
            dt = time.time() - t0
            # print(f"Ran step {turn} via {dt} sec")
            turn += 1

        return game_results

    def play_game(self, p1, p2, game_id: str, epoch: int, match_number: int = 0) -> list[GameResult]:
        env = (
            bg_env.Env(
                self.seed + hash(game_id) % 100000,
                n_games=int(getattr(self.cfg, "n_games_per_match", 11)),
                endless_mode=bool(getattr(self.cfg, "endless_mode", False)),
            )
            if bg_env is not None
            else _FallbackEnv(self.seed + hash(game_id) % 100000)
        )
        env.reset()
        players = [p1, p2]
        turn = 0
        done = False
        target_games_in_match = max(1, int(getattr(self.cfg, "n_games_per_match", 1)))
        max_steps_per_game = max(1, int(getattr(self.cfg, "max_steps_per_game", 200)))
        finished_games = 0
        current_game_turns = 0
        history: list[dict] = []
        results: list[GameResult] = []

        while not done:
            env.roll_dice()
            rolled_dice = np.asarray(env.current_dice(), dtype=np.uint8)
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
                    accept_double = 1
            else:
                lm = env.legal_moves()
                local_moves = lm[1] if isinstance(lm, tuple) else lm
                move = self._score_random(local_moves)
                apply_double = 0
                accept_double = 0

            raw_before = np.asarray(env.get_state_raw(), dtype=np.int16)
            dave_value = int(raw_before[55]) if raw_before.shape[0] > 55 else 1
            try:
                reward, _dave_after, accepted, done_code = env.step_move(move, apply_double=apply_double, accept_double=accept_double)
            except TypeError:
                reward, done_code = env.step_move(move)
                accepted = 0

            current_game_number = finished_games + 1
            history.append({
                "state_vector": state,
                "agent_id": actor.agent_id,
                "opponent_id": opp.agent_id,
                "game_id": game_id,
                "match_id": game_id,
                "match_number": int(match_number),
                "game_number_in_match": int(current_game_number),
                "step_index": current_game_turns,
                "player_index": actor_player_index,
                "epoch": epoch,
                "double_offered_by_agent": bool(apply_double),
                "double_was_accepted": bool(accepted) if bool(apply_double) else False,
                "accept_double_opponent": bool(accept_double),
                "accept_double_opportunity": bool(raw_before[65] >= 0) if raw_before.shape[0] > 65 else False,
                "action_meta": {
                    "dice": [int(rolled_dice[0]), int(rolled_dice[1])],
                    "move": [int(x) for x in np.asarray(move, dtype=np.uint8).tolist()],
                    "apply_double": int(apply_double),
                    "accept_double_for_next_offer": int(accept_double),
                    "raw_state": [int(x) for x in raw_before.tolist()],
                },
            })
            history[-1]["action_meta"]["reward"] = float(reward)
            history[-1]["action_meta"]["done_code"] = int(done_code)
            history[-1]["action_meta"]["double_was_accepted"] = int(accepted)

            turn += 1
            current_game_turns += 1
            game_finished = int(done_code) in (1, 2)
            if game_finished:
                winner = actor.agent_id if reward > 0 else opp.agent_id
                winner_player_index = actor_player_index if reward > 0 else (1 - actor_player_index)
                reward_value = max(1, int(round(abs(float(reward)))))
                points_won = max(1, int(round(abs(float(reward)) * max(dave_value, 1))))
                finished_games += 1
                raw_after = np.asarray(env.get_state_raw(), dtype=np.int16)
                ended_by_double_reject = bool(raw_before[65] >= 0 and int(accept_double) == 0 and int(done_code) in (1, 2)) if raw_before.shape[0] > 65 else False
                results.append(
                    GameResult(
                        game_id=f"{game_id}_g{finished_games}",
                        match_id=game_id,
                        match_number=int(match_number),
                        game_number_in_match=int(finished_games),
                        steps=history,
                        winner=winner,
                        turns=current_game_turns,
                        player_1_id=p1.agent_id,
                        player_2_id=p2.agent_id,
                        winner_player_index=winner_player_index,
                        points_won=points_won,
                        reward_value=reward_value,
                        final_dave_value=int(raw_after[55]) if raw_after.shape[0] > 55 else 1,
                        ended_by_double_reject=ended_by_double_reject,
                        max_steps_reached=False,
                    )
                )
                history = []
                current_game_turns = 0
                done = bool(int(done_code) == 2 or finished_games >= target_games_in_match)
            elif current_game_turns >= max_steps_per_game:
                results.append(
                    GameResult(
                        game_id=f"{game_id}_g{current_game_number}",
                        match_id=game_id,
                        match_number=int(match_number),
                        game_number_in_match=int(current_game_number),
                        steps=history,
                        winner=actor.agent_id,
                        turns=current_game_turns,
                        player_1_id=p1.agent_id,
                        player_2_id=p2.agent_id,
                        winner_player_index=actor_player_index,
                        points_won=1,
                        reward_value=1,
                        final_dave_value=int(raw_before[55]) if raw_before.shape[0] > 55 else 1,
                        ended_by_double_reject=False,
                        max_steps_reached=True,
                    )
                )
                done = True
            else:
                done = False

        return results

    def _configure_doubling_mode(self, trainable_agents: list[ValueAgent], enable_agent_decisions: bool) -> None:
        baseline_copy_prob = float(np.clip(getattr(self.cfg, "conservative_baseline_double_copy_prob", 0.0), 0.0, 1.0))
        if trainable_agents and self.rng.random() < baseline_copy_prob:
            self._baseline_eval_agent = trainable_agents[int(self.rng.integers(0, len(trainable_agents)))]
        else:
            self._baseline_eval_agent = None

        if not enable_agent_decisions:
            self._agents_double_decision_enabled_by_agent = {a.agent_id: False for a in trainable_agents}
            return

        agents_double_decision_prob = float(np.clip(getattr(self.cfg, "agents_double_decision_prob", 0.0), 0.0, 1.0))
        self._agents_double_decision_enabled_by_agent = {
            a.agent_id: bool(self.rng.random() < agents_double_decision_prob)
            for a in trainable_agents
        }

    def run_calibration_epoch(self, trainable_agents: list[ValueAgent], epoch: int):
        t0 = time.time()
        self.reset_decision_stats()
        self._configure_doubling_mode(trainable_agents, enable_agent_decisions=True)

        all_agents: list[object] = list(trainable_agents) + [self.conservative_baseline]
        specs: list[_GameSpec] = []
        matches_per_pair = max(1, int(getattr(self.cfg, "calibrate_matches_per_pair", 1)))
        match_counter = 0
        for i, a in enumerate(all_agents):
            for j in range(i, len(all_agents)):
                b = all_agents[j]
                pair_token = f"{_compact_agent_token(a.agent_id)}_{_compact_agent_token(b.agent_id)}"
                for match_number in range(matches_per_pair):
                    specs.append(
                        _GameSpec(
                            match_id=f"e{epoch}_cal_{pair_token}_m{match_number}",
                            match_number=match_counter,
                            p1=a,
                            p2=b,
                        )
                    )
                    match_counter += 1

        results = self._play_all_games_batched(specs, epoch)
        dt = max(time.time() - t0, 1e-6)
        return results, len(results) / dt

    def run_training_epoch(
        self,
        trainable_agents: list[ValueAgent],
        epoch: int,
        decayed_winrates: np.ndarray,
        all_agent_ids: list[str],
    ):
        t0 = time.time()
        self.reset_decision_stats()
        self._configure_doubling_mode(trainable_agents, enable_agent_decisions=True)

        all_agents: list[object] = list(trainable_agents) + [self.conservative_baseline]
        n_agents = len(all_agents)
        matches_per_agent = max(1, int(getattr(self.cfg, "matches_per_agent", 1)))
        matches_left = np.full((n_agents,), matches_per_agent, dtype=np.int64)
        sigma = max(float(getattr(self.cfg, "matchmaking_sigma", 0.2)), 1e-6)

        pairs: list[tuple[int, int]] = []
        dist_probs: list[float] = []
        norm_const = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        for i in range(n_agents):
            for j in range(i, n_agents):
                if all_agents[i].agent_id == self.conservative_baseline.agent_id and all_agents[j].agent_id == self.conservative_baseline.agent_id:
                    continue
                pairs.append((i, j))
                diff = float(decayed_winrates[i] - decayed_winrates[j])
                dist_probs.append(float(norm_const * np.exp(-0.5 * (diff / sigma) ** 2)))
        dist_probs_arr = np.asarray(dist_probs, dtype=np.float64)

        specs: list[_GameSpec] = []
        pair_match_numbers: dict[tuple[int, int], int] = {}
        match_counts = {aid: {opp: 0 for opp in all_agent_ids} for aid in all_agent_ids}

        while int(np.sum(matches_left)) > 0:
            matches_left_probs = np.asarray(
                [max(0, min(int(matches_left[i]), int(matches_left[j]))) for i, j in pairs],
                dtype=np.float64,
            )
            sampling_probs = matches_left_probs * dist_probs_arr
            total = float(np.sum(sampling_probs))
            if total <= 0.0:
                break
            sampling_probs = sampling_probs / total
            chosen_idx = int(self.rng.choice(len(pairs), p=sampling_probs))
            i, j = pairs[chosen_idx]
            a = all_agents[i]
            b = all_agents[j]

            pair_key = (i, j)
            match_number = pair_match_numbers.get(pair_key, 0)
            pair_match_numbers[pair_key] = match_number + 1
            pair_token = f"{_compact_agent_token(a.agent_id)}_{_compact_agent_token(b.agent_id)}"
            specs.append(
                _GameSpec(
                    match_id=f"e{epoch}_train_{pair_token}_m{match_number}",
                    match_number=match_number,
                    p1=a,
                    p2=b,
                )
            )

            matches_left[i] = max(int(matches_left[i]) - 1, 0)
            if j != i:
                matches_left[j] = max(int(matches_left[j]) - 1, 0)

            aid = all_agent_ids[i]
            bid = all_agent_ids[j]
            match_counts[aid][bid] += 1
            if i != j:
                match_counts[bid][aid] += 1

        results = self._play_all_games_batched(specs, epoch)
        dt = max(time.time() - t0, 1e-6)
        return results, len(results) / dt, match_counts

    # Backward-compatible API used in tests and legacy callers.
    def run_epoch(self, trainable_agents: list[ValueAgent], epoch: int):
        return self.run_calibration_epoch(trainable_agents=trainable_agents, epoch=epoch)
