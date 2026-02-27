from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class ModelConfig:
    input_dim: int = 158
    output_mode: str = "logit"
    activation_fn: str = "relu"
    weight_init: str = "xavier"
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    dropout_enabled: bool = True
    p_dropout: float = 0.1
    dropout_layout: list[int] = field(default_factory=lambda: [1, 2])
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    num_layers: int = 3
    residual_connections: bool = False
    final_bias_init: float = 0.0
    conv_channels: list[int] = field(default_factory=lambda: [8, 16])
    conv_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    conv_strides: list[int] = field(default_factory=lambda: [1, 1])
    conv_paddings: list[int] = field(default_factory=lambda: [1, 1])
    conv_activation: str = "relu"
    conv_pooling_type: str = "avg"
    conv_pooling_params: dict[str, Any] = field(default_factory=dict)
    fusion_mode: str = "concat"
    head_hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    conv_out_channels: int = 64
    conv_output_dim: int = 512


@dataclass
class TrainConfig:
    num_epochs: int = 400
    updates_per_epoch_per_agent: int = 40
    batch_size: int = 10_000
    optimizer_type: str = "adam"
    learning_rate: float = 1e-3
    lr_decay_factor: float = 0.96
    lr_decay_every_steps: int = 40
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    grad_clip_norm: float = 1.0
    lr_scheduler_type: str = "none"
    lr_scheduler_params: dict[str, Any] = field(default_factory=dict)
    mixed_precision_enabled: bool = False
    grad_accum_steps: int = 1
    train_device: str = "cuda"
    eval_device: str = "cuda"
    seed: int = 42
    loss_type: str = "bce_with_logits"
    plot_every_k_epochs: int = 20
    winrate_window_size: int = 10


@dataclass
class LeagueConfig:
    games_per_pair: int = 5
    replay_storage_dir: str = "training_stats/replay"
    min_replay_size_to_train: int = 100
    alpha_recency: float = 1.0
    alpha_uniform: float = 0.0
    recency_window: int = 2
    recency_decay: float = 0.98
    recency_center_mass_ratio: float = 0.9
    sampling_mode: str = "window"
    parallel_env_workers: int = 1
    selfplay_temperature: float = 1.0
    temperature_decay: float = 0.9
    evaluation_games_per_pair: int = 1
    checkpoint_frequency_epochs: int = 1
    metrics_flush_frequency: int = 1


@dataclass
class ExperimentConfig:
    model_group_a: ModelConfig = field(default_factory=lambda: ModelConfig(hidden_dims=[128, 64], num_layers=3, p_dropout=0.10, dropout_layout=[1, 2]))
    model_group_b: ModelConfig = field(default_factory=lambda: ModelConfig(hidden_dims=[256, 256, 128, 128, 64], num_layers=6, p_dropout=0.15, dropout_layout=[1, 2, 3, 4, 5]))
    model_group_c: ModelConfig = field(default_factory=lambda: ModelConfig(p_dropout=0.10, dropout_layout=[0, 2], hidden_dims=[128, 64], conv_out_channels=64, conv_kernel_sizes=[6]))
    model_group_d: ModelConfig = field(default_factory=lambda: ModelConfig(
        p_dropout=0.10,
        dropout_layout=[1, 2],
        hidden_dims=[128, 64],
        conv_channels=[64, 64, 64],
        conv_kernel_sizes=[3, 3, 2],
        conv_pooling_type="max",
        conv_output_dim=256,
    ))
    train: TrainConfig = field(default_factory=TrainConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    checkpoint_dir: str = "training_stats/checkpoints"
    plots_dir: str = "training_stats/plots"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        league_data = dict(data.get("league", {}))
        # Backward compatibility with old checkpoints/configs.
        league_data.pop("max_turns_per_game", None)
        return cls(
            model_group_a=ModelConfig(**data.get("model_group_a", {})),
            model_group_b=ModelConfig(**data.get("model_group_b", {})),
            model_group_c=ModelConfig(**data.get("model_group_c", {})),
            model_group_d=ModelConfig(**data.get("model_group_d", {})),
            train=TrainConfig(**data.get("train", {})),
            league=LeagueConfig(**league_data),
            checkpoint_dir=data.get("checkpoint_dir", "python/training_stats"),
        )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def load_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(data)
