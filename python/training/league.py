from __future__ import annotations

from dataclasses import dataclass
import copy
import time

import numpy as np
import torch

from .agents import ValueAgent

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
    return np.full((8,), 255, dtype=np.uint8)


class LeagueController:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.random = RandomAgent()
        self.baseline = BaselineAgent()
        self.decision_temperature = float(getattr(cfg, "selfplay_temperature", 0.0))
        self._decision_topk_hits = np.zeros((10,), dtype=np.float64)
        self._decision_count = 0

    def set_decision_temperature(self, temperature: float) -> None:
        self.decision_temperature = float(temperature)

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
        temp = float(self.decision_temperature)
        if temp <= 0.0:
            return int(np.argmax(values))

        centered = values - float(np.max(values))
        logits = centered / temp
        exp_logits = np.exp(logits)
        probs = exp_logits / float(np.sum(exp_logits))
        idx = int(self.rng.choice(len(values), p=probs))
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

    def _predict_probs_single_cuda_call(self, agents_for_samples: list[ValueAgent], obs_np: np.ndarray) -> np.ndarray:
        """Predict probabilities for mixed-agent samples with one CUDA call per architecture group."""
        if len(obs_np) == 0:
            return np.zeros((0,), dtype=np.float32)

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

        x_t = torch.as_tensor(obs_np.astype(np.float32), dtype=torch.float32, device=device)

        # Fast path: one model in group -> ordinary single forward.
        if len(unique_agents) == 1:
            probs = unique_agents[0].predict_proba_tensor(x_t).reshape(-1)
            return probs.detach().cpu().numpy()

        # Vectorized ensemble forward: one CUDA call for all models in group.
        if functional_call is not None and stack_module_state is not None and vmap is not None:
            models = [ag.model for ag in unique_agents]

            # stack_module_state требует одинаковый train/eval режим у всех моделей.
            prev_modes = [m.training for m in models]
            try:
                for m in models:
                    m.eval()

                base_model = copy.deepcopy(models[0]).to(device)
                base_model.eval()
                params, buffers = stack_module_state(models)

                def _fmodel(p, b, x):
                    return functional_call(base_model, (p, b), (x,))

                logits_all = vmap(_fmodel, in_dims=(0, 0, None))(params, buffers, x_t).squeeze(-1)
                probs_all = torch.sigmoid(logits_all)
                model_idx_t = torch.as_tensor(model_idx, dtype=torch.long, device=device)
                sample_idx_t = torch.arange(x_t.shape[0], dtype=torch.long, device=device)
                probs = probs_all[model_idx_t, sample_idx_t]
                return probs.detach().cpu().numpy()
            finally:
                for m, was_training in zip(models, prev_modes):
                    m.train(was_training)

        # Conservative fallback when torch.func is unavailable.
        out = np.empty((len(agents_for_samples),), dtype=np.float32)
        for ag_id, m_idx in agent_to_idx.items():
            sel = np.where(model_idx == m_idx)[0]
            p = unique_agents[m_idx].predict_proba_tensor(x_t[torch.as_tensor(sel, dtype=torch.long, device=device)]).reshape(-1)
            out[sel] = p.detach().cpu().numpy()
        return out

    def _select_group_actions_single_call(self, states: np.ndarray, legal_moves_list: list[np.ndarray], actors: list[ValueAgent]) -> np.ndarray:
        actions = np.full((len(legal_moves_list), 8), 255, dtype=np.uint8)
        candidate_obs: list[np.ndarray] = []
        candidate_done: list[bool] = []
        candidate_owner: list[int] = []
        candidate_actor: list[ValueAgent] = []

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
                candidate_actor.append(actors[i])

        if candidate_obs:
            probs = self._predict_probs_single_cuda_call(candidate_actor, np.stack(candidate_obs).astype(np.float32))
            vals = np.where(np.asarray(candidate_done, dtype=bool), 1.0, probs)

            grouped_vals: dict[int, list[float]] = {}
            for owner, val in zip(candidate_owner, vals):
                grouped_vals.setdefault(owner, []).append(float(val))
            for i, moves in enumerate(legal_moves_list):
                if len(moves) <= 1:
                    continue
                values_i = np.asarray(grouped_vals[i], dtype=np.float32)
                selected_idx = self._sample_action_index(values_i)
                self._record_topk_hit(values_i, selected_idx)
                actions[i] = moves[selected_idx]

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

                local_states = states[idxs]
                local_moves = [legal_moves[i] for i in idxs]
                local_actors = [(game_specs[i].p1 if turn % 2 == 0 else game_specs[i].p2) for i in idxs]
                local_actions = self._select_group_actions_single_call(local_states, local_moves, local_actors)
                actions[np.asarray(idxs, dtype=np.int64)] = local_actions

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
                move = self._select_group_actions_single_call(
                    np.asarray([env.get_state_raw()], dtype=np.int16),
                    [env.legal_moves()],
                    [actor],
                )[0]
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
        self.reset_decision_stats()
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
