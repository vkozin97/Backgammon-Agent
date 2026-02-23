from pathlib import Path

from training.config import ExperimentConfig, save_config, load_config
from training.pipeline import run_training, load_checkpoint
from training.agents import build_trainable_agents
from training.league import BaselineAgent, RandomAgent, pass_move


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


def test_one_training_epoch_and_checkpoint(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 1
    cfg.train.updates_per_epoch_per_agent = 1
    cfg.train.batch_size = 8
    cfg.league.games_per_pair = 1
    cfg.league.min_replay_size_to_train = 1
    cfg.league.max_turns_per_game = 10
    cfg.checkpoint_dir = str(tmp_path / "ckpt")

    metrics = run_training(cfg)
    assert len(metrics) == 1
    ck = Path(cfg.checkpoint_dir) / "epoch_0000" / "agents.json"
    assert ck.exists()

    agents = build_trainable_agents(cfg, cfg.train.seed)
    load_checkpoint(cfg, agents, 0)
    assert agents[0].agent_id == "trainable_0"


def test_fixed_agents_support_empty_legal_moves():
    env = _NoMoveEnv()
    expected = pass_move()
    assert (RandomAgent().select(env) == expected).all()
    assert (BaselineAgent().select(env) == expected).all()
