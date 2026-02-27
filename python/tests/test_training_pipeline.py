from pathlib import Path
import importlib.util
import sqlite3
import numpy as np

from training.config import ExperimentConfig, save_config, load_config
from training.pipeline import run_training, load_checkpoint, _games_for_pair
from training.agents import build_trainable_agents
from training.league import RandomAgent, pass_move, GameResult


class _NoMoveEnv:
    def legal_moves(self):
        return np.empty((0, 8), dtype=np.uint8)


def test_replay_sample_with_agent_ids_returns_agent_labels(tmp_path: Path):
    from training.replay import ReplayBuffer

    replay = ReplayBuffer(storage_dir=str(tmp_path / "replay"))
    for i in range(40):
        aid = "trainable_0" if i % 2 == 0 else "trainable_1"
        replay.add(
            state_vector=[float(i), 0.0],
            agent_id=aid,
            opponent_id="opp",
            game_id=f"g_{i // 2}",
            step_index=i,
            epoch=0,
            terminal_outcome=1.0 if aid == "trainable_0" else 0.0,
        )

    x, y, a = replay.sample_with_agent_ids(64, alpha_recency=0.8, alpha_uniform=0.2, recency_window=2)
    replay.close()

    assert x.shape[0] == 64
    assert y.shape == (64, 1)
    assert a.shape == (64,)
    assert set(a.tolist()).issubset({"trainable_0", "trainable_1"})


def test_replay_sample_stratified_with_agent_ids_returns_requested_counts(tmp_path: Path):
    from training.replay import ReplayBuffer

    replay = ReplayBuffer(storage_dir=str(tmp_path / "replay"))
    for i in range(60):
        aid = "trainable_0" if i % 3 != 0 else "trainable_1"
        replay.add(
            state_vector=[float(i), 1.0],
            agent_id=aid,
            opponent_id="opp",
            game_id=f"g_{i // 2}",
            step_index=i,
            epoch=0,
            terminal_outcome=1.0 if aid == "trainable_0" else 0.0,
        )

    sampled = replay.sample_stratified_with_agent_ids(
        {"trainable_0": 32, "trainable_1": 24, "unknown": 16},
        alpha_recency=0.8,
        alpha_uniform=0.2,
        recency_window=2,
    )
    replay.close()

    x0, y0 = sampled["trainable_0"]
    x1, y1 = sampled["trainable_1"]
    xu, yu = sampled["unknown"]

    assert x0.shape[0] == 32
    assert y0.shape == (32, 1)
    assert x1.shape[0] == 24
    assert y1.shape == (24, 1)
    assert xu.shape[0] == 0
    assert yu.shape == (0, 1)

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
    replay_dir = Path(cfg.plots_dir) / "replay"
    winrates_windowed_dir = Path(cfg.plots_dir) / "winrates_windowed"
    decision_dir = Path(cfg.plots_dir) / "decision_temperature"
    if importlib.util.find_spec("matplotlib") is not None:
        assert winrates_dir.exists()
        assert loss_dir.exists()
        trainable_agents = build_trainable_agents(cfg, cfg.train.seed)
        total_agents = len(trainable_agents) + 1
        opponents_per_agent = total_agents - 1
        assert len(list(winrates_dir.glob("*.png"))) == total_agents * opponents_per_agent
        assert len(list(winrates_windowed_dir.glob("*.png"))) == total_agents * opponents_per_agent
        assert len(list(loss_dir.glob("*.png"))) == len(trainable_agents)
        assert len(list(lr_dir.glob("*.png"))) == len(trainable_agents) * 2
        assert (replay_dir / "replay_size.png").exists()
        assert (decision_dir / "decision_temperature.png").exists()
        assert len(list(decision_dir.glob("selected_action_top_*.png"))) == 10

    agents = build_trainable_agents(cfg, cfg.train.seed)
    load_checkpoint(cfg, agents, 0)
    assert agents[0].agent_id == "trainable_0"


def test_fixed_agents_support_empty_legal_moves():
    env = _NoMoveEnv()
    expected = pass_move()
    assert (RandomAgent().select(env) == expected).all()


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


def test_run_training_clears_replay_db_on_fresh_start(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    db_path = replay_dir / "replay.sqlite3"

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE stale (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO stale (id) VALUES (1)")
    conn.commit()
    conn.close()

    cfg = ExperimentConfig()
    cfg.train.num_epochs = 0
    cfg.league.replay_storage_dir = str(replay_dir)
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)
    assert metrics == []

    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "replay" in tables
    assert "stale" not in tables
