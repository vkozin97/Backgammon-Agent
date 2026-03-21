from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ModelConfig, TrainConfig
from .observation import POINTS_DIM, VECTOR_CHANNELS
from match_win_probs import get_match_win_probs


MATCH_VECTOR_DIM = 12
REWARD_VECTOR_DIM = 6
ACCEPT_HEAD_DIM = 1
TOTAL_OUTPUT_DIM = MATCH_VECTOR_DIM * 2 + ACCEPT_HEAD_DIM + REWARD_VECTOR_DIM
REWARD_VALUES = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=np.float32)
MET_TABLE = np.asarray(get_match_win_probs(MATCH_VECTOR_DIM - 1), dtype=np.float32)


@dataclass
class DoubleHintMetrics:
    reward_vec: np.ndarray
    exp_no_double: float
    exp_double: float
    p_accept: float
    reward_vec_after_move: np.ndarray
    exp_reject: float
    exp_accept: float
    apply_double: int
    accept_double: int


def reward_expectation(probs_row: np.ndarray) -> float:
    p = np.asarray(probs_row, dtype=np.float32).reshape(-1)
    reward_head = p[MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]
    if reward_head.size != REWARD_VECTOR_DIM:
        return 0.0
    return float(np.dot(REWARD_VALUES, reward_head))


def extract_obs_controls(obs: np.ndarray) -> tuple[int, int, int, int, int]:
    v = np.asarray(obs, dtype=np.float32).reshape(-1)
    my_left = int(np.clip(np.round(float(v[-6])) if v.size >= 6 else MATCH_VECTOR_DIM - 1, 0, MATCH_VECTOR_DIM - 1))
    opp_left = int(np.clip(np.round(float(v[-5])) if v.size >= 5 else MATCH_VECTOR_DIM - 1, 0, MATCH_VECTOR_DIM - 1))
    dave_val = int(max(1, round(float(v[-7])))) if v.size >= 7 else 1
    my_double_avail = int(round(float(v[-4]))) if v.size >= 4 else 0
    opp_double_avail = int(round(float(v[-3]))) if v.size >= 3 else 0
    return my_left, opp_left, dave_val, my_double_avail, opp_double_avail


def set_obs_double_state(obs: np.ndarray) -> np.ndarray:
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


def set_obs_opponent_double_offer(obs: np.ndarray) -> np.ndarray:
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


def set_obs_double_offer_for_receiver(obs: np.ndarray) -> np.ndarray:
    return set_obs_opponent_double_offer(flip_observation_perspective(set_obs_double_state(obs)))


def flip_observation_perspective(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x.copy()

    out = x.copy()

    def _flip_pair(start_a: int, start_b: int, width: int = POINTS_DIM) -> None:
        out[start_a:start_a + width] = x[start_b:start_b + width][::-1]
        out[start_b:start_b + width] = x[start_a:start_a + width][::-1]

    _flip_pair(0, POINTS_DIM)  # points / opp_points
    _flip_pair(POINTS_DIM * 2, POINTS_DIM * 3)  # blots / opp_blots
    _flip_pair(POINTS_DIM * 4, POINTS_DIM * 5)  # anchors / opp_anchors
    _flip_pair(POINTS_DIM * 6, POINTS_DIM * 8)  # hit_prob_mine / hit_prob_opp
    _flip_pair(POINTS_DIM * 7, POINTS_DIM * 9)  # cover_prob_mine / cover_prob_opp

    scalar_base = VECTOR_CHANNELS * POINTS_DIM
    if out.size >= scalar_base + 14:
        out[scalar_base + 0] = x[scalar_base + 2]  # bar
        out[scalar_base + 1] = x[scalar_base + 3]  # off
        out[scalar_base + 2] = x[scalar_base + 0]  # opp_bar
        out[scalar_base + 3] = x[scalar_base + 1]  # opp_off
        out[scalar_base + 4] = x[scalar_base + 5]  # pip_mine
        out[scalar_base + 5] = x[scalar_base + 4]  # pip_opp
        out[scalar_base + 6] = x[scalar_base + 7]  # blots_mine
        out[scalar_base + 7] = x[scalar_base + 6]  # blots_opp
        out[scalar_base + 8] = x[scalar_base + 9]  # anchors_mine
        out[scalar_base + 9] = x[scalar_base + 8]  # anchors_opp
        out[scalar_base + 10] = x[scalar_base + 11]  # blot_pips_mine
        out[scalar_base + 11] = x[scalar_base + 10]  # blot_pips_opp
        out[scalar_base + 12] = x[scalar_base + 13]  # anchor_pips_mine
        out[scalar_base + 13] = x[scalar_base + 12]  # anchor_pips_opp

    match_base = scalar_base + 14
    if out.size >= match_base + 9:
        out[match_base + 0] = x[match_base + 1]  # mine_score
        out[match_base + 1] = x[match_base + 0]  # opp_score
        out[match_base + 2] = x[match_base + 2]  # dave_value
        out[match_base + 3] = x[match_base + 4]  # my_left
        out[match_base + 4] = x[match_base + 3]  # opp_left
        out[match_base + 5] = x[match_base + 6]  # cube_available_mine
        out[match_base + 6] = x[match_base + 5]  # cube_available_opp
        out[match_base + 7] = x[match_base + 7]  # is_crawford_game
        out[match_base + 8] = x[match_base + 8]  # double_offered

    return out


def reject_double_equity(obs_now: np.ndarray, endless: bool = False) -> float:
    if endless:
        return -1.0
    my_left, opp_left, dave_val, _, opp_double_avail = extract_obs_controls(obs_now)
    if opp_double_avail <= 0:
        return 0.0
    opp_after = max(opp_left - int(max(dave_val, 1)), 0)
    return float(MET_TABLE[my_left, opp_after])


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
    my_left, opp_left, _, _, _ = extract_obs_controls(obs)
    out[:MATCH_VECTOR_DIM] = _mask_and_normalize_head(out[:MATCH_VECTOR_DIM], my_left)
    out[MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2] = _mask_and_normalize_head(out[MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2], opp_left)
    out[MATCH_VECTOR_DIM * 2] = float(np.clip(out[MATCH_VECTOR_DIM * 2], 0.0, 1.0))
    return out


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
    return float(my @ met_view @ opp)


def head_win_eval(probs_row: np.ndarray, obs_row: np.ndarray) -> float:
    masked = _mask_eval_outputs(probs_row, obs_row)
    my_left, opp_left, _, _, _ = extract_obs_controls(obs_row)
    my = masked[:MATCH_VECTOR_DIM]
    opp = masked[MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2]
    my_r, opp_r = _redistribute_forbidden_stay_mass(my, opp, my_left, opp_left)
    return _expected_match_win_prob(my_r, opp_r)


def decide_apply_double_from_probs(
    probs_now: np.ndarray,
    probs_after_double: np.ndarray,
    obs_now: np.ndarray,
    endless: bool = False,
    probs_offer_receiver: np.ndarray | None = None,
) -> int:
    p_accept_source = probs_after_double if probs_offer_receiver is None else probs_offer_receiver
    p_accept = float(np.clip(np.asarray(p_accept_source, dtype=np.float32)[MATCH_VECTOR_DIM * 2], 0.0, 1.0))
    if endless:
        _, _, _, my_double_avail, _ = extract_obs_controls(obs_now)
        if my_double_avail <= 0:
            return 0
        exp_no_double = reward_expectation(probs_now)
        exp_double = p_accept * 2.0 * reward_expectation(probs_after_double) + (1.0 - p_accept) * 1.0
        return int(exp_double > exp_no_double)

    my_left, opp_left, dave_val, my_double_avail, _ = extract_obs_controls(obs_now)
    if my_double_avail <= 0:
        return 0
    p_win = head_win_eval(probs_now, obs_now)
    my_after_reject = max(my_left - int(max(dave_val, 1)), 0)
    p_win_rejected = float(MET_TABLE[my_after_reject, opp_left])
    p_win_accepted = head_win_eval(probs_after_double, set_obs_double_state(obs_now))
    p_win_double = p_accept * p_win_accepted + (1.0 - p_accept) * p_win_rejected
    return int(p_win_double > p_win)


def decide_accept_double_from_probs(probs_if_opp_doubles: np.ndarray, obs_now: np.ndarray, endless: bool = False) -> int:
    _, _, _, _, opp_double_avail = extract_obs_controls(obs_now)
    if opp_double_avail <= 0:
        return 0
    if endless:
        exp_reject = reject_double_equity(obs_now, endless=True)
        exp_accept = 2.0 * reward_expectation(probs_if_opp_doubles)
        return int(exp_accept >= exp_reject)

    p_accept = head_win_eval(probs_if_opp_doubles, set_obs_opponent_double_offer(obs_now))
    p_reject = reject_double_equity(obs_now, endless=False)
    return int(p_accept >= p_reject)


def get_double_hint_metrics(
    agent: "ValueAgent",
    obs_now_current: np.ndarray,
    obs_post_turn_swapped: np.ndarray,
    endless: bool = False,
    canonical_post_reward_vec: Optional[np.ndarray] = None,
) -> DoubleHintMetrics:
    obs_now_2d = np.asarray(obs_now_current, dtype=np.float32).reshape(1, -1)
    probs_now = np.asarray(agent.predict_proba(obs_now_2d), dtype=np.float32).reshape(-1)
    probs_double_now = np.asarray(agent.predict_proba(set_obs_double_state(obs_now_current).reshape(1, -1)), dtype=np.float32).reshape(-1)
    probs_double_offer_receiver = np.asarray(
        agent.predict_proba(set_obs_double_offer_for_receiver(obs_now_current).reshape(1, -1)),
        dtype=np.float32,
    ).reshape(-1)

    reward_vec = probs_now[MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]
    p_accept = float(np.clip(probs_double_offer_receiver[MATCH_VECTOR_DIM * 2], 0.0, 1.0))
    exp_no_double = reward_expectation(probs_now)
    exp_double = p_accept * (2.0 * reward_expectation(probs_double_now)) + (1.0 - p_accept) * 1.0
    apply_double = decide_apply_double_from_probs(
        probs_now,
        probs_double_now,
        obs_now_current,
        endless=endless,
        probs_offer_receiver=probs_double_offer_receiver,
    )

    obs_post_turn = np.asarray(obs_post_turn_swapped, dtype=np.float32).reshape(-1)
    probs_keep = np.asarray(agent.predict_proba(obs_post_turn.reshape(1, -1)), dtype=np.float32).reshape(-1)
    reward_vec_after_move = probs_keep[MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM].copy()[::-1]
    obs_post_current = flip_observation_perspective(obs_post_turn)
    probs_offer = np.asarray(
        agent.predict_proba(set_obs_opponent_double_offer(obs_post_current).reshape(1, -1)),
        dtype=np.float32,
    ).reshape(-1)
    exp_reject = reject_double_equity(obs_post_current, endless=endless)
    canonical_post_reward = (
        np.asarray(canonical_post_reward_vec, dtype=np.float32).reshape(-1)
        if canonical_post_reward_vec is not None else
        reward_vec_after_move
    )
    exp_accept = (
        2.0 * float(np.dot(REWARD_VALUES, canonical_post_reward))
        if endless else
        head_win_eval(probs_offer, set_obs_opponent_double_offer(obs_post_current))
    )
    accept_double = (
        int(extract_obs_controls(obs_post_current)[4] > 0 and exp_accept >= exp_reject)
        if endless else
        decide_accept_double_from_probs(probs_offer, obs_post_current, endless=endless)
    )

    return DoubleHintMetrics(
        reward_vec=reward_vec,
        exp_no_double=float(exp_no_double),
        exp_double=float(exp_double),
        p_accept=float(p_accept),
        reward_vec_after_move=reward_vec_after_move,
        exp_reject=float(exp_reject),
        exp_accept=float(exp_accept),
        apply_double=int(apply_double),
        accept_double=int(accept_double),
    )


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    activation_name: str,
    dropout_enabled: bool,
    dropout_layout: list[int],
    p_dropout: float,
) -> nn.Sequential:
    dims = [input_dim] + hidden_dims
    blocks: list[nn.Module] = []
    dropout_layers = set(dropout_layout) if dropout_enabled else set()
    for i in range(len(dims) - 1):
        blocks.append(nn.Linear(dims[i], dims[i + 1]))
        blocks.append(_activation(activation_name))
        if (i + 1) in dropout_layers:
            blocks.append(nn.Dropout(p_dropout))
    return nn.Sequential(*blocks)


@dataclass
class Param:
    w: np.ndarray
    b: np.ndarray


class TorchMLPValueModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _build_mlp(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            activation_name=cfg.activation_fn,
            dropout_enabled=cfg.dropout_enabled,
            dropout_layout=cfg.dropout_layout,
            p_dropout=cfg.p_dropout,
        )
        self.out = nn.Linear(cfg.hidden_dims[-1], cfg.output_dim)
        with torch.no_grad():
            self.out.bias.fill_(cfg.final_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(x))


class TorchConvHeadValueModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(
            in_channels=VECTOR_CHANNELS,
            out_channels=cfg.conv_out_channels,
            kernel_size=6,
        )
        self.conv_activation = _activation(cfg.conv_activation)
        self.scalar_dim = max(int(cfg.input_dim) - VECTOR_CHANNELS * POINTS_DIM, 0)
        conv_len = POINTS_DIM - 6 + 1
        mlp_in = cfg.conv_out_channels * conv_len + self.scalar_dim
        self.head = _build_mlp(
            input_dim=mlp_in,
            hidden_dims=cfg.head_hidden_dims,
            activation_name=cfg.activation_fn,
            dropout_enabled=cfg.dropout_enabled,
            dropout_layout=cfg.dropout_layout,
            p_dropout=cfg.p_dropout,
        )
        self.out = nn.Linear(cfg.head_hidden_dims[-1], cfg.output_dim)
        with torch.no_grad():
            self.out.bias.fill_(cfg.final_bias_init)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = x[:, : VECTOR_CHANNELS * POINTS_DIM].reshape(-1, VECTOR_CHANNELS, POINTS_DIM)
        scalars = x[:, VECTOR_CHANNELS * POINTS_DIM : VECTOR_CHANNELS * POINTS_DIM + self.scalar_dim]
        return vectors, scalars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vectors, scalars = self._split(x)
        c = self.conv_activation(self.conv(vectors)).flatten(1)
        feat = torch.cat([c, scalars], dim=1)
        return self.out(self.head(feat))


class TorchDeepConvValueModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        channels = [VECTOR_CHANNELS] + list(cfg.conv_channels)
        kernels = list(cfg.conv_kernel_sizes)
        conv_blocks: list[nn.Module] = []
        cur_len = POINTS_DIM
        pool_every = max(int(getattr(cfg, "conv_pool_every", 0)), 0)
        for i, k in enumerate(kernels):
            conv_blocks.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=k, padding="same"))
            conv_blocks.append(_activation(cfg.conv_activation))
            if pool_every > 0 and (i + 1) % pool_every == 0:
                conv_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
                cur_len = max(cur_len // 2, 1)
        self.conv_stack = nn.Sequential(*conv_blocks)

        self.scalar_dim = max(int(cfg.input_dim) - VECTOR_CHANNELS * POINTS_DIM, 0)
        head_in = channels[-1] * cur_len + self.scalar_dim
        self.head = _build_mlp(
            input_dim=head_in,
            hidden_dims=cfg.head_hidden_dims,
            activation_name=cfg.activation_fn,
            dropout_enabled=cfg.dropout_enabled,
            dropout_layout=cfg.dropout_layout,
            p_dropout=cfg.p_dropout,
        )
        self.out = nn.Linear(cfg.head_hidden_dims[-1], cfg.output_dim)
        with torch.no_grad():
            self.out.bias.fill_(cfg.final_bias_init)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = x[:, : VECTOR_CHANNELS * POINTS_DIM].reshape(-1, VECTOR_CHANNELS, POINTS_DIM)
        scalars = x[:, VECTOR_CHANNELS * POINTS_DIM : VECTOR_CHANNELS * POINTS_DIM + self.scalar_dim]
        return vectors, scalars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vectors, scalars = self._split(x)
        c = self.conv_stack(vectors).flatten(1)
        feat = torch.cat([scalars, c], dim=1)
        return self.out(self.head(feat))


class ValueAgent:
    def __init__(self, agent_id: str, group: str, model_cfg: ModelConfig, train_cfg: TrainConfig, seed: int):
        self.agent_id = agent_id
        self.group = group
        torch.manual_seed(seed)
        requested = train_cfg.train_device
        if requested.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(requested)
        else:
            self.device = torch.device("cpu")

        if group in {"C", "D"}:
            self.model = TorchDeepConvValueModel(model_cfg)
        else:
            self.model = TorchMLPValueModel(model_cfg)
        self.model.to(self.device)

        if train_cfg.optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay, betas=train_cfg.betas
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_cfg.learning_rate,
                momentum=train_cfg.momentum,
                weight_decay=train_cfg.weight_decay,
            )
        self.loss_type = train_cfg.loss_type
        self.loss_weights = np.asarray(train_cfg.loss_weights, dtype=np.float32)
        self.target_expansion = train_cfg.target_expansion
        self.grad_clip_norm = train_cfg.grad_clip_norm
        self.lr_decay_factor = train_cfg.lr_decay_factor
        self.lr_decay_every_steps = train_cfg.lr_decay_every_steps
        self.min_learning_rate = float(train_cfg.min_learning_rate)
        self.lr_schedule_base = float(train_cfg.learning_rate)
        self.lr_schedule_step_offset = 0
        self.output_layer_only_training = False
        self.train_step = 0

    def predict_logits(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self.model.train(training)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(xt)
        return logits.detach().cpu().numpy()

    def _probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        my_logits = logits[:, :MATCH_VECTOR_DIM]
        opp_logits = logits[:, MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2]
        accept_logits = logits[:, MATCH_VECTOR_DIM * 2: MATCH_VECTOR_DIM * 2 + 1]
        reward_logits = logits[:, MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]
        my_probs = torch.softmax(my_logits, dim=1)
        opp_probs = torch.softmax(opp_logits, dim=1)
        accept_prob = torch.sigmoid(accept_logits)
        reward_probs = torch.softmax(reward_logits, dim=1)
        return torch.cat([my_probs, opp_probs, accept_prob, reward_probs], dim=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x, training=False)
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        probs = self._probs_from_logits(logits_t)
        return probs.detach().cpu().numpy()

    def predict_proba_tensor(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self._probs_from_logits(self.model(x.to(self.device)))

    def _output_parameter_ids(self) -> set[int]:
        return {id(param) for param in self.model.out.parameters()}

    def set_output_layer_only_training(self, enabled: bool) -> None:
        enabled = bool(enabled)
        output_param_ids = self._output_parameter_ids()
        for param in self.model.parameters():
            param.requires_grad = (id(param) in output_param_ids) if enabled else True
        self.output_layer_only_training = enabled

    def _scheduled_learning_rate(self) -> float:
        if int(self.lr_decay_every_steps) <= 0:
            return float(max(self.lr_schedule_base, self.min_learning_rate))
        phase_steps = max(int(self.train_step) - int(self.lr_schedule_step_offset), 0)
        decay_events = phase_steps // int(self.lr_decay_every_steps)
        lr = float(self.lr_schedule_base) * (float(self.lr_decay_factor) ** int(decay_events))
        return float(max(lr, self.min_learning_rate))

    def _apply_current_learning_rate(self) -> float:
        lr = self._scheduled_learning_rate()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def configure_training_phase(
        self,
        learning_rate: float,
        lr_decay_factor: float,
        schedule_step_offset: int,
        freeze_to_output_layer: bool,
    ) -> float:
        self.lr_schedule_base = float(learning_rate)
        self.lr_decay_factor = float(lr_decay_factor)
        self.lr_schedule_step_offset = int(schedule_step_offset)
        self.set_output_layer_only_training(freeze_to_output_layer)
        return self._apply_current_learning_rate()

    def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return self.train_batch_tensor(x_t, y_t)

    def train_batch_tensor(self, x_t: torch.Tensor, y_t: torch.Tensor) -> float:
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x_t)
        y_expanded = self._expand_targets(y_t, logits.shape[1])

        my_logits = logits[:, :MATCH_VECTOR_DIM]
        opp_logits = logits[:, MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2]
        accept_logits = logits[:, MATCH_VECTOR_DIM * 2: MATCH_VECTOR_DIM * 2 + 1]

        my_target = y_expanded[:, :MATCH_VECTOR_DIM]
        opp_target = y_expanded[:, MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2]
        accept_target = y_expanded[:, MATCH_VECTOR_DIM * 2: MATCH_VECTOR_DIM * 2 + 1]
        reward_target = y_expanded[:, MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]

        if self.loss_type == "mse":
            probs = self._probs_from_logits(logits)
            loss_main = torch.mean((probs[:, : MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM] - y_expanded[:, : MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]) ** 2)
        elif self.loss_type == "smooth_l1":
            probs = self._probs_from_logits(logits)
            loss_main = torch.nn.functional.smooth_l1_loss(probs[:, : MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM], y_expanded[:, : MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM])
        else:
            loss_my = -torch.mean(torch.sum(my_target * torch.log_softmax(my_logits, dim=1), dim=1))
            loss_opp = -torch.mean(torch.sum(opp_target * torch.log_softmax(opp_logits, dim=1), dim=1))
            loss_reward = -torch.mean(torch.sum(reward_target * torch.log_softmax(logits[:, MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM], dim=1), dim=1))
            loss_main = (loss_my + loss_opp + loss_reward) / 3.0

        loss_accept = torch.nn.functional.binary_cross_entropy_with_logits(accept_logits, accept_target)

        loss = loss_main + loss_accept
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.train_step += 1
        self._apply_current_learning_rate()
        return float(loss.detach().cpu().item())

    def state_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "group": self.group,
            "model": {k: v.detach().cpu().numpy().tolist() for k, v in self.model.state_dict().items()},
        }

    def load_state_dict(self, state: dict) -> None:
        model_state = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in state["model"].items()}
        current_state = self.model.state_dict()
        adapted_state: dict[str, torch.Tensor] = {}
        for key, cur_tensor in current_state.items():
            old_tensor = model_state.get(key)
            if old_tensor is None:
                adapted_state[key] = cur_tensor
                continue
            if old_tensor.shape == cur_tensor.shape:
                adapted_state[key] = old_tensor
                continue
            if key in {"out.weight", "out.bias"} and old_tensor.ndim == cur_tensor.ndim:
                merged = cur_tensor.clone()
                copy_n = min(old_tensor.shape[0], cur_tensor.shape[0])
                merged[:copy_n] = old_tensor[:copy_n]
                adapted_state[key] = merged
                continue
            adapted_state[key] = cur_tensor
        self.model.load_state_dict(adapted_state, strict=False)

    def _expand_targets(self, y_t: torch.Tensor, output_dim: int) -> torch.Tensor:
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)

        if y_t.shape[1] == output_dim + 1:
            y_main = y_t[:, :output_dim]
        elif y_t.shape[1] == output_dim:
            y_main = y_t
        else:
            raise ValueError(f"Target dim {y_t.shape[1]} does not match model output dim {output_dim} (or {output_dim + 1} legacy format)")

        my = y_main[:, :MATCH_VECTOR_DIM]
        opp = y_main[:, MATCH_VECTOR_DIM: MATCH_VECTOR_DIM * 2]
        acc = y_main[:, MATCH_VECTOR_DIM * 2: MATCH_VECTOR_DIM * 2 + 1]
        reward = y_main[:, MATCH_VECTOR_DIM * 2 + 1: MATCH_VECTOR_DIM * 2 + 1 + REWARD_VECTOR_DIM]

        my = my / torch.clamp(torch.sum(my, dim=1, keepdim=True), min=1e-8)
        opp = opp / torch.clamp(torch.sum(opp, dim=1, keepdim=True), min=1e-8)
        reward = reward / torch.clamp(torch.sum(reward, dim=1, keepdim=True), min=1e-8)
        acc = torch.clamp(acc, 0.0, 1.0)

        return torch.cat([my, opp, acc, reward], dim=1)

    def _loss_weights_tensor(self, output_dim: int, device: torch.device) -> torch.Tensor:
        if self.loss_weights.size == output_dim:
            weights = self.loss_weights
        elif self.loss_weights.size == 1:
            weights = np.full((output_dim,), float(self.loss_weights[0]), dtype=np.float32)
        else:
            weights = np.ones((output_dim,), dtype=np.float32)
        return torch.as_tensor(weights.reshape(1, output_dim), dtype=torch.float32, device=device)


def build_trainable_agents(cfg, seed: int = 0) -> list[ValueAgent]:
    agents: list[ValueAgent] = []
    groups = ["A"] * 3 + ["C"] * 3 + ["D"] * 3
    for i, g in enumerate(groups):
        mcfg = (
            cfg.model_group_a
            if g == "A"
            else cfg.model_group_c
            if g == "C"
            else cfg.model_group_d
        )
        agents.append(ValueAgent(f"trainable_{i}", g, mcfg, cfg.train, seed + i))
    return agents
