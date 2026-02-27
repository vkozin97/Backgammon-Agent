from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ModelConfig, TrainConfig
from .observation import POINTS_DIM, SCALAR_FEATURES_DIM, VECTOR_CHANNELS


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
        conv_len = POINTS_DIM - 6 + 1
        mlp_in = cfg.conv_out_channels * conv_len + SCALAR_FEATURES_DIM
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
        scalars = x[:, VECTOR_CHANNELS * POINTS_DIM : VECTOR_CHANNELS * POINTS_DIM + SCALAR_FEATURES_DIM]
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
        for i, k in enumerate(kernels):
            conv_blocks.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=k))
            conv_blocks.append(_activation(cfg.conv_activation))
            cur_len = cur_len - k + 1
            if i < len(kernels) - 1:
                conv_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
                cur_len = max(cur_len // 2, 1)
        self.conv_stack = nn.Sequential(*conv_blocks)
        self.proj = nn.Linear(channels[-1] * cur_len, cfg.conv_output_dim)

        head_in = cfg.conv_output_dim + SCALAR_FEATURES_DIM
        self.head = _build_mlp(
            input_dim=head_in,
            hidden_dims=cfg.hidden_dims,
            activation_name=cfg.activation_fn,
            dropout_enabled=cfg.dropout_enabled,
            dropout_layout=cfg.dropout_layout,
            p_dropout=cfg.p_dropout,
        )
        self.out = nn.Linear(cfg.hidden_dims[-1], cfg.output_dim)
        with torch.no_grad():
            self.out.bias.fill_(cfg.final_bias_init)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = x[:, : VECTOR_CHANNELS * POINTS_DIM].reshape(-1, VECTOR_CHANNELS, POINTS_DIM)
        scalars = x[:, VECTOR_CHANNELS * POINTS_DIM : VECTOR_CHANNELS * POINTS_DIM + SCALAR_FEATURES_DIM]
        return vectors, scalars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vectors, scalars = self._split(x)
        c = self.conv_stack(vectors).flatten(1)
        c = self.proj(c)
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

        if group == "C":
            self.model: nn.Module = TorchConvHeadValueModel(model_cfg)
        elif group == "D":
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
        self.train_step = 0

    def predict_logits(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self.model.train(training)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(xt)
        return logits.detach().cpu().numpy()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x, training=False)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba_tensor(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(x.to(self.device)))

    def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return self.train_batch_tensor(x_t, y_t)

    def train_batch_tensor(self, x_t: torch.Tensor, y_t: torch.Tensor) -> float:
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x_t)
        y_expanded = self._expand_targets(y_t, logits.shape[1])
        if self.loss_type == "mse":
            probs = torch.sigmoid(logits)
            loss = torch.mean((probs - y_expanded) ** 2)
        elif self.loss_type == "smooth_l1":
            probs = torch.sigmoid(logits)
            loss = torch.nn.functional.smooth_l1_loss(probs, y_expanded)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_expanded, reduction="none")
            if logits.shape[1] > 1:
                loss = loss * self._loss_weights_tensor(logits.shape[1], logits.device)
            loss = torch.mean(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.train_step += 1
        if self.lr_decay_every_steps > 0 and self.train_step % self.lr_decay_every_steps == 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= self.lr_decay_factor
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
        if y_t.shape[1] == output_dim:
            return y_t
        if output_dim == 1:
            return y_t[:, :1]
        if self.target_expansion == "first_head_only":
            expanded = torch.zeros((y_t.shape[0], output_dim), dtype=y_t.dtype, device=y_t.device)
            expanded[:, :1] = y_t[:, :1]
            return expanded
        return y_t[:, :1].repeat(1, output_dim)

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
    groups = ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3
    for i, g in enumerate(groups):
        mcfg = (
            cfg.model_group_a
            if g == "A"
            else cfg.model_group_b
            if g == "B"
            else cfg.model_group_c
            if g == "C"
            else cfg.model_group_d
        )
        agents.append(ValueAgent(f"trainable_{i}", g, mcfg, cfg.train, seed + i))
    return agents
