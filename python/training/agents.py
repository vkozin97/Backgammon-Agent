from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ModelConfig, TrainConfig


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


@dataclass
class Param:
    w: np.ndarray
    b: np.ndarray


class TorchMLPValueModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dims = [cfg.input_dim] + cfg.hidden_dims + [1]
        blocks: list[nn.Module] = []
        self.dropout_layers = set(cfg.dropout_layout) if cfg.dropout_enabled else set()
        for i in range(len(dims) - 2):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            blocks.append(_activation(cfg.activation_fn))
            if (i + 1) in self.dropout_layers:
                blocks.append(nn.Dropout(cfg.p_dropout))
        self.backbone = nn.Sequential(*blocks)
        self.out = nn.Linear(dims[-2], 1)
        with torch.no_grad():
            self.out.bias.fill_(cfg.final_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(x))


class TorchConvHeadValueModel(TorchMLPValueModel):
    def __init__(self, cfg: ModelConfig) -> None:
        conv_cfg = ModelConfig(**{**cfg.__dict__})
        conv_cfg.input_dim = sum(cfg.conv_channels) + 4
        conv_cfg.hidden_dims = cfg.head_hidden_dims
        conv_cfg.num_layers = 1 + len(cfg.head_hidden_dims)
        super().__init__(conv_cfg)
        self.cfg_orig = cfg
        self.conv_1 = nn.Linear(24, cfg.conv_channels[0])
        self.conv_2 = nn.Linear(24, cfg.conv_channels[1])
        self.conv_activation = _activation(cfg.conv_activation)

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        mine = x[:, :24]
        opp = x[:, 24:48]
        extra = x[:, 48:52]
        f1 = self.conv_activation(self.conv_1(mine))
        f2 = self.conv_activation(self.conv_2(opp))
        return torch.cat([f1, f2, extra], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self._conv_features(x))


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
        if self.loss_type == "mse":
            probs = torch.sigmoid(logits)
            loss = torch.mean((probs - y_t) ** 2)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
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
        self.model.load_state_dict(model_state)


def build_trainable_agents(cfg, seed: int = 0) -> list[ValueAgent]:
    agents: list[ValueAgent] = []
    groups = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    for i, g in enumerate(groups):
        mcfg = cfg.model_group_a if g == "A" else cfg.model_group_b if g == "B" else cfg.model_group_c
        agents.append(ValueAgent(f"trainable_{i}", g, mcfg, cfg.train, seed + i))
    return agents
