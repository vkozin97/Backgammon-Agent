from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any

from .observation_layout import OBSERVATION_DIM


@dataclass
class ModelConfig:
    input_dim: int = OBSERVATION_DIM
    output_dim: int = 31
    activation_fn: str = "relu"
    dropout_enabled: bool = False
    p_dropout: float = 0.1
    dropout_layout: list[int] = field(default_factory=list)
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    final_bias_init: float = 0.0
    conv_channels: list[int] = field(default_factory=lambda: [8, 16])
    conv_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    conv_pool_every: int = 0
    conv_activation: str = "relu"
    head_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    conv_out_channels: int = 64
    conv_output_dim: int = 0


@dataclass
class TrainConfig:
    num_epochs: int = 700
    updates_per_epoch_per_agent: int = 100
    batch_size: int = 10_000
    optimizer_type: str = "adam"
    learning_rate: float = 1e-4
    min_learning_rate: float = 5e-7
    lr_decay_factor: float = 0.997
    lr_decay_every_steps: int = 50
    freeze_weights_from_epoch: int = 300
    freeze_weights_till_epoch: int = 400
    lr_during_freeze: float = 1e-5
    lr_decay_during_freeze: float = 0.98
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    grad_clip_norm: float = 1.0
    train_device: str = "cuda"
    seed: int = 42
    loss_type: str = "bce_with_logits"
    loss_weights: list[float] = field(default_factory=lambda: [1.0])
    target_expansion: str = "repeat"
    plot_every_k_epochs: int = 20
    winrate_window_size: int = 20
    value_window_size: int = 20
    matchmaking_window_size: int = 20


@dataclass
class LeagueConfig:
    matches_per_agent: int = 12
    n_games_per_match: int = 4
    endless_mode: bool = True
    replay_storage_dir: str = "training_stats/replay"
    min_replay_size_to_train: int = 100
    alpha_recency: float = 0.8
    alpha_uniform: float = 0.2
    recency_decay: float = 0.95
    replay_window_epochs: int = 40
    sigmoid_parameter: float = 6.74755607143124
    batched_obs_threads: int = 8
    selfplay_temperature: float = 0.1
    temperature_decay: float = 0.98
    choose_best_probability: float = 0.3
    choose_best_decay: float = 0.9
    conservative_baseline_double_copy_prob: float = 0.0
    baseline_conservative_double_copy_start_epoch: int = 300
    baseline_conservative_double_copy_end_epoch: int = 350
    agents_double_decision_prob: float = 0.0
    agents_double_decision_start_epoch: int = 300
    agents_double_decision_end_epoch: int = 350
    checkpoint_frequency_epochs: int = 1
    max_steps_per_game: int = 200
    calibrate_every_k_epochs: int = 1
    calibrate_matches_per_pair: int = 1
    calibrate_winrates_decay: float = 0.9
    matchmaking_sigma: float = 0.2


@dataclass
class ExperimentConfig:
    model_group_a: ModelConfig = field(default_factory=lambda: ModelConfig(
        hidden_dims=[128, 64],
        dropout_enabled=False,
        p_dropout=0.0,
        dropout_layout=[],
    ))
    model_group_c: ModelConfig = field(default_factory=lambda: ModelConfig(
        dropout_enabled=False,
        p_dropout=0.0,
        dropout_layout=[],
        hidden_dims=[64, 32],
        conv_channels=[32, 16, 32, 16, 32, 16],
        conv_kernel_sizes=[3, 3, 3, 3, 3, 3],
        conv_pool_every=2,
        head_hidden_dims=[64, 32],
        conv_output_dim=0,
    ))
    model_group_d: ModelConfig = field(default_factory=lambda: ModelConfig(
        dropout_enabled=False,
        p_dropout=0.0,
        dropout_layout=[],
        hidden_dims=[64, 32],
        conv_channels=[32, 16, 16, 8],
        conv_kernel_sizes=[6, 3, 6, 3],
        conv_pool_every=2,
        head_hidden_dims=[64, 32],
        conv_output_dim=0,
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
        if "games_in_match" in league_data:
            legacy_games_in_match = int(league_data.pop("games_in_match"))
            if "endless_mode" not in league_data:
                league_data["endless_mode"] = legacy_games_in_match < 0
            if "n_games_per_match" not in league_data:
                if legacy_games_in_match < 0:
                    legacy_matches = max(1, int(league_data.get("matches_per_pair", 1)))
                    league_data["n_games_per_match"] = legacy_matches
                    if "matches_per_pair" in league_data:
                        league_data["matches_per_pair"] = 1
                else:
                    league_data["n_games_per_match"] = max(1, legacy_games_in_match)
        if "games_per_pair" in league_data and "matches_per_pair" not in league_data:
            league_data["matches_per_pair"] = league_data.pop("games_per_pair")
        if "pages_per_pair" in league_data and "matches_per_pair" not in league_data:
            league_data["matches_per_pair"] = league_data.pop("pages_per_pair")
        if "matches_per_pair" in league_data and "matches_per_agent" not in league_data:
            legacy_matches_per_pair = int(league_data.pop("matches_per_pair"))
            legacy_games_per_epoch = max(legacy_matches_per_pair, 1)
            league_data["matches_per_agent"] = legacy_games_per_epoch
        if "baseline_conservative_double_copy_prob" in league_data and "conservative_baseline_double_copy_prob" not in league_data:
            league_data["conservative_baseline_double_copy_prob"] = league_data.pop("baseline_conservative_double_copy_prob")
        league_data.pop("recency_window", None)
        league_data.pop("recency_center_mass_ratio", None)

        # Migrate deprecated decay-based schedules to epoch-window schedules.
        if "conservative_baseline_double_copy_decay" in league_data:
            league_data.pop("conservative_baseline_double_copy_decay", None)
        if "agents_double_decision_decay" in league_data:
            league_data.pop("agents_double_decision_decay", None)
        model_a = dict(data.get("model_group_a", {}))
        model_c = dict(data.get("model_group_c", {}))
        model_d = dict(data.get("model_group_d", {}))
        train_data = dict(data.get("train", {}))

        removed_model_keys = {
            "output_mode", "weight_init", "use_layer_norm", "use_batch_norm",
            "num_layers", "residual_connections", "conv_strides", "conv_paddings",
            "conv_pooling_type", "conv_pooling_params", "fusion_mode",
        }
        removed_train_keys = {"lr_scheduler_type", "lr_scheduler_params", "mixed_precision_enabled", "grad_accum_steps", "eval_device"}
        removed_league_keys = {"sampling_mode", "parallel_env_workers", "evaluation_games_per_pair", "metrics_flush_frequency"}

        for k in removed_model_keys:
            model_a.pop(k, None)
            model_c.pop(k, None)
            model_d.pop(k, None)
        for k in removed_train_keys:
            train_data.pop(k, None)
        for k in removed_league_keys:
            league_data.pop(k, None)
        return cls(
            model_group_a=ModelConfig(**model_a),
            model_group_c=ModelConfig(**model_c),
            model_group_d=ModelConfig(**model_d),
            train=TrainConfig(**train_data),
            league=LeagueConfig(**league_data),
            checkpoint_dir=data.get("checkpoint_dir", "python/training_stats"),
        )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def load_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(data)
