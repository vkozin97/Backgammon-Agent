from pathlib import Path
import importlib.util

from training.config import ExperimentConfig, save_config, load_config
from training.pipeline import run_training, load_checkpoint, _games_for_pair
from training.agents import build_trainable_agents
from training.league import BaselineAgent, RandomAgent, pass_move, GameResult


class _NoMoveEnv:
    def legal_moves(self):
        import numpy as np

        return np.empty((0, 8), dtype=np.uint8)


def test_config_roundtrip(tmp_path: Path):
    cfg = ExperimentConfig()
    p = tmp_path / "cfg.json"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.train.num_epochs == cfg.train.num_epochs
    assert loaded.train.winrate_window_size == cfg.train.winrate_window_size
    assert loaded.checkpoint_dir == cfg.checkpoint_dir


def test_one_training_epoch_and_checkpoint(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 1
    cfg.train.updates_per_epoch_per_agent = 1
    cfg.train.batch_size = 8
    cfg.league.games_per_pair = 1
    cfg.league.min_replay_size_to_train = 1
    cfg.league.max_turns_per_game = 10
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)
    assert len(metrics) == 1
    assert "decision_temperature" in metrics[0]
    assert "decision_topk_freq" in metrics[0]
    assert len(metrics[0]["decision_topk_freq"]) == 10
    ck = Path(cfg.checkpoint_dir) / "epoch_0000" / "agents.json"
    assert ck.exists()
    winrates_dir = Path(cfg.plots_dir) / "winrates"
    loss_dir = Path(cfg.plots_dir) / "loss"
    lr_dir = Path(cfg.plots_dir) / "lr"
    winrates_windowed_dir = Path(cfg.plots_dir) / "winrates_windowed"
    decision_dir = Path(cfg.plots_dir) / "decision_temperature"
    if importlib.util.find_spec("matplotlib") is not None:
        assert winrates_dir.exists()
        assert loss_dir.exists()
        trainable_agents = build_trainable_agents(cfg, cfg.train.seed)
        total_agents = len(trainable_agents) + 2
        opponents_per_agent = total_agents - 1
        assert len(list(winrates_dir.glob("*.png"))) == total_agents * opponents_per_agent
        assert len(list(winrates_windowed_dir.glob("*.png"))) == total_agents * opponents_per_agent
        assert len(list(loss_dir.glob("*.png"))) == len(trainable_agents)
        assert len(list(lr_dir.glob("*.png"))) == len(trainable_agents)
        assert (decision_dir / "decision_temperature.png").exists()
        assert len(list(decision_dir.glob("selected_action_top_*.png"))) == 10

    agents = build_trainable_agents(cfg, cfg.train.seed)
    load_checkpoint(cfg, agents, 0)
    assert agents[0].agent_id == "trainable_0"


def test_fixed_agents_support_empty_legal_moves():
    env = _NoMoveEnv()
    expected = pass_move()
    assert (RandomAgent().select(env) == expected).all()
    assert (BaselineAgent().select(env) == expected).all()


def test_pair_matching_uses_participants_not_game_id():
    games = [
        GameResult("opaque_1", [], "trainable_0", 7, "trainable_0", "trainable_1"),
        GameResult("opaque_2", [], "trainable_1", 7, "trainable_1", "trainable_2"),
        GameResult("opaque_3", [], "random", 7, "trainable_0", "random"),
    ]
    pair = _games_for_pair(games, "trainable_0", "trainable_1")
    assert len(pair) == 1
    assert pair[0].game_id == "opaque_1"


def test_temperature_decay_progression(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 2
    cfg.train.updates_per_epoch_per_agent = 0
    cfg.league.games_per_pair = 1
    cfg.league.max_turns_per_game = 2
    cfg.league.selfplay_temperature = 1.0
    cfg.league.temperature_decay = 0.9
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)
    assert len(metrics) == 2
    assert metrics[0]["decision_temperature"] == 1.0
    assert abs(metrics[1]["decision_temperature"] - 0.9) < 1e-9
