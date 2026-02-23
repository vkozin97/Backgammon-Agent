from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ModelConfig, TrainConfig


def _act(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    return np.maximum(x, 0.0)


def _act_grad(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        t = np.tanh(x)
        return 1.0 - t * t
    return (x > 0).astype(np.float32)


@dataclass
class Param:
    w: np.ndarray
    b: np.ndarray


class MLPValueModel:
    def __init__(self, cfg: ModelConfig, seed: int) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        dims = [cfg.input_dim] + cfg.hidden_dims + [1]
        self.layers: list[Param] = []
        for i in range(len(dims) - 1):
            w = rng.normal(0, np.sqrt(2.0 / max(dims[i], 1)), size=(dims[i], dims[i + 1])).astype(np.float32)
            b = np.zeros((dims[i + 1],), dtype=np.float32)
            if i == len(dims) - 2:
                b[:] = cfg.final_bias_init
            self.layers.append(Param(w=w, b=b))

    def forward(self, x: np.ndarray, training: bool) -> tuple[np.ndarray, dict]:
        cache: dict = {"a0": x}
        a = x
        for i, layer in enumerate(self.layers[:-1], start=1):
            z = a @ layer.w + layer.b
            a = _act(z, self.cfg.activation_fn)
            if self.cfg.dropout_enabled and training and i in set(self.cfg.dropout_layout):
                keep = 1.0 - self.cfg.p_dropout
                m = (np.random.rand(*a.shape) < keep).astype(np.float32) / keep
                a = a * m
                cache[f"drop{i}"] = m
            cache[f"z{i}"] = z
            cache[f"a{i}"] = a
        out = a @ self.layers[-1].w + self.layers[-1].b
        cache["out_in"] = a
        return out, cache

    def backward(self, cache: dict, dloss_dout: np.ndarray) -> list[Param]:
        grads = [Param(np.zeros_like(l.w), np.zeros_like(l.b)) for l in self.layers]
        grads[-1].w[:] = cache["out_in"].T @ dloss_dout / dloss_dout.shape[0]
        grads[-1].b[:] = dloss_dout.mean(axis=0)
        da = dloss_dout @ self.layers[-1].w.T
        for idx in reversed(range(1, len(self.layers))):
            if f"drop{idx}" in cache:
                da = da * cache[f"drop{idx}"]
            dz = da * _act_grad(cache[f"z{idx}"], self.cfg.activation_fn)
            a_prev = cache["a0"] if idx == 1 else cache[f"a{idx - 1}"]
            grads[idx - 1].w[:] = a_prev.T @ dz / dz.shape[0]
            grads[idx - 1].b[:] = dz.mean(axis=0)
            da = dz @ self.layers[idx - 1].w.T
        return grads


class ConvHeadValueModel(MLPValueModel):
    def __init__(self, cfg: ModelConfig, seed: int) -> None:
        conv_cfg = ModelConfig(**{**cfg.__dict__})
        conv_cfg.input_dim = sum(cfg.conv_channels) + 4
        conv_cfg.hidden_dims = cfg.head_hidden_dims
        conv_cfg.num_layers = 1 + len(cfg.head_hidden_dims)
        super().__init__(conv_cfg, seed)
        rng = np.random.default_rng(seed + 10)
        self.conv_w1 = rng.normal(0, 0.1, size=(24, cfg.conv_channels[0])).astype(np.float32)
        self.conv_w2 = rng.normal(0, 0.1, size=(24, cfg.conv_channels[1])).astype(np.float32)
        self.cfg_orig = cfg

    def _conv_features(self, x: np.ndarray) -> np.ndarray:
        mine = x[:, :24]
        opp = x[:, 24:48]
        extra = x[:, 48:52]
        f1 = _act(mine @ self.conv_w1, self.cfg_orig.conv_activation)
        f2 = _act(opp @ self.conv_w2, self.cfg_orig.conv_activation)
        return np.concatenate([f1, f2, extra], axis=1)

    def forward(self, x: np.ndarray, training: bool) -> tuple[np.ndarray, dict]:
        fx = self._conv_features(x)
        return super().forward(fx, training)


class SGD:
    def __init__(self, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.vw: list[np.ndarray] = []
        self.vb: list[np.ndarray] = []

    def step(self, params: list[Param], grads: list[Param]) -> None:
        if not self.vw:
            self.vw = [np.zeros_like(p.w) for p in params]
            self.vb = [np.zeros_like(p.b) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            gw = g.w + self.weight_decay * p.w
            gb = g.b
            self.vw[i] = self.momentum * self.vw[i] + (1 - self.momentum) * gw
            self.vb[i] = self.momentum * self.vb[i] + (1 - self.momentum) * gb
            p.w -= self.lr * self.vw[i]
            p.b -= self.lr * self.vb[i]


class ValueAgent:
    def __init__(self, agent_id: str, group: str, model_cfg: ModelConfig, train_cfg: TrainConfig, seed: int):
        self.agent_id = agent_id
        self.group = group
        if group == "C":
            self.model = ConvHeadValueModel(model_cfg, seed)
        else:
            self.model = MLPValueModel(model_cfg, seed)
        self.optimizer = SGD(train_cfg.learning_rate, momentum=train_cfg.momentum, weight_decay=train_cfg.weight_decay)
        self.loss_type = train_cfg.loss_type

    def predict_logits(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        logits, _ = self.model.forward(x.astype(np.float32), training)
        return logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x, training=False)
        return 1.0 / (1.0 + np.exp(-logits))

    def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, cache = self.model.forward(x.astype(np.float32), training=True)
        if self.loss_type == "mse":
            probs = 1.0 / (1.0 + np.exp(-logits))
            loss = np.mean((probs - y) ** 2)
            d = 2.0 * (probs - y) * probs * (1 - probs) / y.shape[0]
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))
            eps = 1e-8
            loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
            d = (probs - y)
        grads = self.model.backward(cache, d)
        self.optimizer.step(self.model.layers, grads)
        return float(loss)

    def state_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "group": self.group,
            "layers": [{"w": p.w.tolist(), "b": p.b.tolist()} for p in self.model.layers],
        }

    def load_state_dict(self, state: dict) -> None:
        for p, ps in zip(self.model.layers, state["layers"]):
            p.w[:] = np.asarray(ps["w"], dtype=np.float32)
            p.b[:] = np.asarray(ps["b"], dtype=np.float32)


def build_trainable_agents(cfg, seed: int = 0) -> list[ValueAgent]:
    agents: list[ValueAgent] = []
    groups = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    for i, g in enumerate(groups):
        mcfg = cfg.model_group_a if g == "A" else cfg.model_group_b if g == "B" else cfg.model_group_c
        agents.append(ValueAgent(f"trainable_{i}", g, mcfg, cfg.train, seed + i))
    return agents
