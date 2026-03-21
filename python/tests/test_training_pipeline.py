from pathlib import Path
import importlib.util
import sqlite3
import numpy as np

from training.config import ExperimentConfig, save_config, load_config
from training.pipeline import (
    run_training,
    load_checkpoint,
    _games_for_pair,
    _terminal_outcome_for_step,
    _bootstrap_outcomes_for_unfinished_game,
    _sigmoid_growth_probability,
    _learning_rate_for_epoch,
    _epoch_uses_output_freeze,
)
from training.agents import (
    MATCH_VECTOR_DIM,
    ValueAgent,
    build_trainable_agents,
    decide_apply_double_from_probs,
    decide_accept_double_from_probs,
    flip_observation_perspective,
    reject_double_equity,
)
from training.league import ConservativeBaselineAgent, RandomAgent, pass_move, GameResult


class _NoMoveEnv:
    def legal_moves(self):
        return np.empty((0, 8), dtype=np.uint8)


def _outcome_vec(v: float) -> np.ndarray:
    out = np.zeros((26,), dtype=np.float32)
    out[0 if v > 0 else 1] = 1.0
    out[12 + (0 if v > 0 else 1)] = 1.0
    return out


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
            terminal_outcome=_outcome_vec(1.0 if aid == "trainable_0" else 0.0),
        )

    x, y, a = replay.sample_with_agent_ids(64, alpha_recency=0.8, alpha_uniform=0.2)
    replay.close()

    assert x.shape[0] == 64
    assert y.shape == (64, 26)
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
            terminal_outcome=_outcome_vec(1.0 if aid == "trainable_0" else 0.0),
        )

    sampled = replay.sample_stratified_with_agent_ids(
        {"trainable_0": 32, "trainable_1": 24, "unknown": 16},
        alpha_recency=0.8,
        alpha_uniform=0.2,
    )
    replay.close()

    x0, y0 = sampled["trainable_0"]
    x1, y1 = sampled["trainable_1"]
    xu, yu = sampled["unknown"]

    assert x0.shape[0] == 32
    assert y0.shape == (32, 26)
    assert x1.shape[0] == 24
    assert y1.shape == (24, 26)
    assert xu.shape[0] == 0
    assert yu.shape == (0, 0)



def test_replay_delete_from_epoch_keeps_only_older_rows(tmp_path: Path):
    from training.replay import ReplayBuffer

    replay = ReplayBuffer(storage_dir=str(tmp_path / "replay"))
    for i in range(6):
        replay.add(
            state_vector=[float(i), 0.0],
            agent_id="trainable_0",
            opponent_id="opp",
            game_id=f"g_{i}",
            step_index=i,
            epoch=i // 2,
            terminal_outcome=_outcome_vec(1.0),
        )

    replay.delete_from_epoch(2)
    meta = replay.get_meta()
    replay.close()

    assert int(meta["size"]) == 4


def test_config_roundtrip(tmp_path: Path):
    cfg = ExperimentConfig()
    p = tmp_path / "cfg.json"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.train.num_epochs == cfg.train.num_epochs
    assert loaded.train.winrate_window_size == cfg.train.winrate_window_size
    assert loaded.train.value_window_size == cfg.train.value_window_size
    assert loaded.checkpoint_dir == cfg.checkpoint_dir


def test_one_training_epoch_and_checkpoint(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 1
    cfg.train.updates_per_epoch_per_agent = 1
    cfg.train.batch_size = 8
    cfg.league.matches_per_pair = 1
    cfg.league.min_replay_size_to_train = 1
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)
    assert len(metrics) == 1
    assert "decision_temperature" in metrics[0]
    assert "decision_topk_freq" in metrics[0]
    assert len(metrics[0]["decision_topk_freq"]) == 10
    ck = Path(cfg.checkpoint_dir) / "epoch_0000" / "agents.json"
    assert ck.exists()
    loss_dir = Path(cfg.plots_dir) / "loss"
    lr_dir = Path(cfg.plots_dir) / "lr"
    replay_dir = Path(cfg.plots_dir) / "replay"
    winrates_windowed_dir = Path(cfg.plots_dir) / "winrates_windowed"
    value_windowed_dir = Path(cfg.plots_dir) / "value_windowed"
    decision_dir = Path(cfg.plots_dir) / "decision_temperature"
    if importlib.util.find_spec("matplotlib") is not None:
        assert loss_dir.exists()
        trainable_agents = build_trainable_agents(cfg, cfg.train.seed)
        total_agents = len(trainable_agents) + 1
        opponents_per_agent = total_agents - 1
        assert len(list(winrates_windowed_dir.glob("*.png"))) == total_agents * opponents_per_agent + 1
        assert len(list(value_windowed_dir.glob("*.png"))) == total_agents * opponents_per_agent + 1
        assert len(list(loss_dir.glob("*.png"))) == len(trainable_agents)
        assert len(list(lr_dir.glob("*.png"))) == len(trainable_agents) * 2
        assert not (replay_dir / "replay_size.png").exists()
        assert (decision_dir / "decision_temperature.png").exists()
        assert len(list(decision_dir.glob("selected_action_top_*.png"))) == 10

    agents = build_trainable_agents(cfg, cfg.train.seed)
    load_checkpoint(cfg, agents, 0)
    assert agents[0].agent_id == "trainable_0"


def test_fixed_agents_support_empty_legal_moves():
    env = _NoMoveEnv()
    expected = pass_move()
    assert (RandomAgent().select(env) == expected).all()
    assert (ConservativeBaselineAgent().select(env) == expected).all()


def test_pair_matching_uses_participants_not_game_id():
    games = [
        GameResult("opaque_1", "opaque_1", 0, 1, [], "trainable_0", 7, "trainable_0", "trainable_1"),
        GameResult("opaque_2", "opaque_2", 0, 1, [], "trainable_1", 7, "trainable_1", "trainable_2"),
        GameResult("opaque_3", "opaque_3", 0, 1, [], "random", 7, "trainable_0", "random"),
    ]
    pair = _games_for_pair(games, "trainable_0", "trainable_1")
    assert len(pair) == 1
    assert pair[0].game_id == "opaque_1"


def test_temperature_decay_progression(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 2
    cfg.train.updates_per_epoch_per_agent = 0
    cfg.league.matches_per_pair = 1
    cfg.league.selfplay_temperature = 1.0
    cfg.league.temperature_decay = 0.9
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")

    metrics = run_training(cfg)
    assert len(metrics) == 2
    assert metrics[0]["decision_temperature"] == 1.0
    assert abs(metrics[1]["decision_temperature"] - 0.9) < 1e-9




def test_sampling_concentrates_on_best_value_when_temperature_goes_to_zero():
    from training.league import LeagueController

    cfg = ExperimentConfig().league
    cfg.selfplay_temperature = 1.0
    league = LeagueController(cfg, seed=7)

    values = np.array([1.0, 0.8, 0.4, -0.2], dtype=np.float32)

    league.set_decision_temperature(1.0)
    top1_warm = 0
    n = 4000
    for _ in range(n):
        idx = league._sample_action_index(values)
        if idx == 0:
            top1_warm += 1

    league.set_decision_temperature(1e-4)
    top1_cool = 0
    for _ in range(n):
        idx = league._sample_action_index(values)
        if idx == 0:
            top1_cool += 1

    assert top1_cool / n > top1_warm / n
    assert top1_cool / n > 0.99


def test_ensemble_cache_key_separates_conv_architectures():
    import training.league as league_module
    from training.league import LeagueController

    if league_module.functional_call is None or league_module.stack_module_state is None or league_module.vmap is None:
        return

    cfg = ExperimentConfig()
    agents = build_trainable_agents(cfg, cfg.train.seed)
    league = LeagueController(cfg.league, seed=11)

    x = np.zeros((2, cfg.model_group_c.input_dim), dtype=np.float32)
    league._predict_probs_single_cuda_call([agents[3], agents[4]], x)
    league._predict_probs_single_cuda_call([agents[5], agents[6]], x)

    assert len(league._ensemble_base_model_cache) == 2
    cached_channels = {tuple(getattr(model.cfg, "conv_channels", [])) for model in league._ensemble_base_model_cache.values()}
    assert tuple(cfg.model_group_c.conv_channels) in cached_channels
    assert tuple(cfg.model_group_d.conv_channels) in cached_channels

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


def test_league_run_epoch_includes_self_mirror_games():
    from training.league import LeagueController, _GameSpec

    cfg = ExperimentConfig().league
    cfg.matches_per_pair = 1
    league = LeagueController(cfg, seed=123)

    class DummyAgent:
        def __init__(self, agent_id: str):
            self.agent_id = agent_id

    agents = [DummyAgent("trainable_0"), DummyAgent("trainable_1"), DummyAgent("trainable_2")]

    captured_specs: list[_GameSpec] = []

    def fake_play(specs, epoch):
        captured_specs.extend(specs)
        return []

    league._play_all_games_batched = fake_play  # type: ignore[method-assign]
    league.run_epoch(agents, epoch=0)

    assert captured_specs
    assert any(spec.p1.agent_id == spec.p2.agent_id for spec in captured_specs)
    assert all("trainable" not in spec.match_id for spec in captured_specs)
    assert all("conservative_baseline" not in spec.match_id for spec in captured_specs)
    assert any("_b_" in spec.match_id for spec in captured_specs)


def test_terminal_outcome_uses_player_index_for_same_agent_ids():
    game = GameResult(
        game_id="g_same",
        match_id="g_same",
        match_number=0,
        game_number_in_match=1,
        steps=[],
        winner="trainable_0",
        turns=2,
        player_1_id="trainable_0",
        player_2_id="trainable_0",
        winner_player_index=1,
    )
    step_p1 = {"agent_id": "trainable_0", "player_index": 0}
    step_p2 = {"agent_id": "trainable_0", "player_index": 1}

    out_p1 = _terminal_outcome_for_step(game, {**step_p1, "state_vector": np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)})
    out_p2 = _terminal_outcome_for_step(game, {**step_p2, "state_vector": np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)})
    assert out_p1.shape == (31,)
    assert out_p2.shape == (31,)
    assert np.isclose(np.sum(out_p1[:12]), 1.0)
    assert np.isclose(np.sum(out_p1[12:24]), 1.0)
    assert np.isclose(np.sum(out_p2[:12]), 1.0)
    assert np.isclose(np.sum(out_p2[12:24]), 1.0)
    assert int(np.argmax(out_p1[:12])) != int(np.argmax(out_p2[:12]))




def test_run_training_resume_drops_replay_rows_from_start_epoch(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    from training.replay import ReplayBuffer

    replay = ReplayBuffer(storage_dir=str(replay_dir))
    for i in range(6):
        replay.add(
            state_vector=[float(i), 0.0],
            agent_id="trainable_0",
            opponent_id="opp",
            game_id=f"g_{i}",
            step_index=i,
            epoch=i,
            terminal_outcome=_outcome_vec(1.0),
        )
    replay.close()

    cfg = ExperimentConfig()
    cfg.train.num_epochs = 3
    cfg.league.replay_storage_dir = str(replay_dir)
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")

    agents = build_trainable_agents(cfg, cfg.train.seed)
    ckpt = Path(cfg.checkpoint_dir) / "epoch_0002"
    ckpt.mkdir(parents=True, exist_ok=True)
    ckpt_agents = __import__("json").dumps([a.state_dict() for a in agents])
    (ckpt / "agents.json").write_text(ckpt_agents, encoding="utf-8")

    run_training(cfg, start_epoch=3)

    conn = sqlite3.connect(replay_dir / "replay.sqlite3")
    remaining = conn.execute("SELECT COUNT(*) FROM replay WHERE epoch >= 3").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM replay").fetchone()[0]
    conn.close()

    assert remaining == 0
    assert total == 3


def test_run_training_deletes_stale_checkpoints_from_start_epoch(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 2
    cfg.train.updates_per_epoch_per_agent = 0
    cfg.league.matches_per_pair = 1
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")
    cfg.league.replay_storage_dir = str(tmp_path / "replay")

    agents = build_trainable_agents(cfg, cfg.train.seed)
    ckpt_root = Path(cfg.checkpoint_dir)
    for epoch in range(4):
        ckpt = ckpt_root / f"epoch_{epoch:04d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        ckpt_agents = __import__("json").dumps([a.state_dict() for a in agents])
        (ckpt / "agents.json").write_text(ckpt_agents, encoding="utf-8")
        (ckpt / "metrics.json").write_text(__import__("json").dumps({"epoch": epoch}), encoding="utf-8")

    run_training(cfg, start_epoch=2)

    assert (ckpt_root / "epoch_0000").exists()
    assert (ckpt_root / "epoch_0001").exists()
    assert not (ckpt_root / "epoch_0003").exists()


def test_run_training_uses_config_params_when_calculation_disabled(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 3
    cfg.train.updates_per_epoch_per_agent = 1
    cfg.train.batch_size = 4
    cfg.train.learning_rate = 1e-3
    cfg.train.lr_decay_factor = 0.5
    cfg.train.lr_decay_every_steps = 1000
    cfg.league.matches_per_pair = 1
    cfg.league.min_replay_size_to_train = 1
    cfg.league.selfplay_temperature = 0.7
    cfg.league.temperature_decay = 0.1
    cfg.league.choose_best_probability = 0.2
    cfg.league.choose_best_decay = 0.3
    cfg.league.conservative_baseline_double_copy_prob = 0.15
    cfg.league.agents_double_decision_prob = 0.35
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")
    cfg.league.replay_storage_dir = str(tmp_path / "replay")

    run_training(cfg, start_epoch=0)
    metrics = run_training(cfg, start_epoch=2, calculate_learning_params=False)

    assert len(metrics) == 3
    resumed = metrics[-1]
    assert resumed["epoch"] == 2
    assert np.isclose(resumed["decision_temperature"], cfg.league.selfplay_temperature)
    assert np.isclose(resumed["choose_best_probability"], cfg.league.choose_best_probability)
    assert np.isclose(resumed["conservative_baseline_double_copy_prob"], cfg.league.conservative_baseline_double_copy_prob)
    assert np.isclose(resumed["agents_double_decision_prob"], cfg.league.agents_double_decision_prob)
    for aid, stats in resumed["agents"].items():
        assert stats["learning_rate_steps_epoch"]
        assert np.isclose(stats["learning_rate_steps_epoch"][0], cfg.train.learning_rate)


def test_run_training_can_resume_from_epoch(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 1
    cfg.train.updates_per_epoch_per_agent = 0
    cfg.league.matches_per_pair = 1
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")
    cfg.league.replay_storage_dir = str(tmp_path / "replay")

    run_training(cfg)

    cfg_resume = ExperimentConfig()
    cfg_resume.train.num_epochs = 2
    cfg_resume.train.updates_per_epoch_per_agent = 0
    cfg_resume.league.matches_per_pair = 1
    cfg_resume.checkpoint_dir = cfg.checkpoint_dir
    cfg_resume.plots_dir = cfg.plots_dir
    cfg_resume.league.replay_storage_dir = cfg.league.replay_storage_dir

    metrics = run_training(cfg_resume, start_epoch=1)
    assert len(metrics) == 2
    assert metrics[0]["epoch"] == 0
    assert metrics[1]["epoch"] == 1
    assert (Path(cfg.checkpoint_dir) / "epoch_0001" / "agents.json").exists()


def test_load_checkpoint_keeps_old_head_and_initializes_new_heads(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.train.num_epochs = 1
    cfg.train.updates_per_epoch_per_agent = 0
    cfg.league.matches_per_pair = 1
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")

    run_training(cfg)

    import json

    states = json.loads((Path(cfg.checkpoint_dir) / "epoch_0000" / "agents.json").read_text(encoding="utf-8"))
    old_w = np.asarray(states[0]["model"]["out.weight"], dtype=np.float32)
    old_b = np.asarray(states[0]["model"]["out.bias"], dtype=np.float32)

    cfg_new = ExperimentConfig()
    cfg_new.model_group_a.output_dim = 3
    cfg_new.model_group_c.output_dim = 3
    cfg_new.model_group_d.output_dim = 3
    cfg_new.checkpoint_dir = cfg.checkpoint_dir

    agents = build_trainable_agents(cfg_new, cfg_new.train.seed)
    load_checkpoint(cfg_new, agents, 0)

    new_w = agents[0].model.state_dict()["out.weight"].detach().cpu().numpy()
    new_b = agents[0].model.state_dict()["out.bias"].detach().cpu().numpy()

    assert new_w.shape[0] == 3
    assert np.allclose(new_w[0], old_w[0])
    assert np.allclose(new_b[0], old_b[0])


def test_bootstrap_outcomes_for_unfinished_game_uses_last_step_per_player():
    class DummyAgent:
        def __init__(self, value: float):
            self.value = value

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            out = np.zeros((x.shape[0], 31), dtype=np.float32)
            out[:, 0] = self.value
            return out

    game = GameResult(
        game_id="g_truncated",
        match_id="g_truncated",
        match_number=0,
        game_number_in_match=1,
        steps=[
            {"player_index": 0, "step_index": 0, "agent_id": "a0", "state_vector": np.array([1.0, 0.0], dtype=np.float32)},
            {"player_index": 1, "step_index": 1, "agent_id": "a1", "state_vector": np.array([2.0, 0.0], dtype=np.float32)},
            {"player_index": 0, "step_index": 2, "agent_id": "a0", "state_vector": np.array([3.0, 0.0], dtype=np.float32)},
        ],
        winner="a0",
        turns=3,
        player_1_id="a0",
        player_2_id="a1",
        max_steps_reached=True,
    )

    targets = _bootstrap_outcomes_for_unfinished_game(game, {"a0": DummyAgent(0.25), "a1": DummyAgent(0.75)})
    assert set(targets.keys()) == {0, 1}
    assert targets[0].shape == (31,)
    assert targets[1].shape == (31,)
    assert np.isclose(targets[0][0], 0.25)
    assert np.isclose(targets[1][0], 0.75)


def test_play_game_respects_max_steps_per_game():
    from training.league import LeagueController

    cfg = ExperimentConfig().league
    cfg.max_steps_per_game = 2
    league = LeagueController(cfg, seed=1)

    result = league.play_game(RandomAgent(), RandomAgent(), game_id="g_cap", epoch=0)
    assert len(result) == 1
    assert result[0].max_steps_reached is True
    assert result[0].turns == 2


def test_terminal_outcome_accept_target_uses_pending_accept():
    step = {
        "agent_id": "a",
        "player_index": 0,
        "state_vector": np.zeros((263,), dtype=np.float32),
        "accept_double_opponent": False,
        "action_meta": {"raw_state": [0] * 65 + [1]},
    }
    game = GameResult(
        game_id="g",
        match_id="g",
        match_number=0,
        game_number_in_match=1,
        steps=[step],
        winner="a",
        turns=1,
        player_1_id="a",
        player_2_id="b",
        winner_player_index=0,
        points_won=1,
        reward_value=1,
    )
    out = _terminal_outcome_for_step(game, step)
    assert out[24] == 1.0



from training.observation import state_to_observation


def test_state_to_observation_uses_cube_scalars_from_state_raw():
    raw = np.zeros((69,), dtype=np.float32)
    raw[53] = 2
    raw[54] = 1
    raw[55] = 4
    raw[56] = 7
    raw[59] = 1
    raw[66] = 1
    raw[67] = 0
    raw[68] = 1
    obs = state_to_observation(raw)
    # match scalars are the last 9 values; cube flags are at positions 5,6,7,8 in that block
    ms = obs[-9:]
    assert ms[5] == 1.0
    assert ms[6] == 0.0
    assert ms[7] == 1.0
    assert ms[8] == 1.0


def test_sigmoid_growth_probability_schedule():
    base = 0.2
    sigmoid_parameter = 6.74755607143124
    assert np.isclose(_sigmoid_growth_probability(base, epoch=0, start_epoch=5, end_epoch=10, sigmoid_parameter=sigmoid_parameter), base)
    at_start = _sigmoid_growth_probability(base, epoch=5, start_epoch=5, end_epoch=10, sigmoid_parameter=sigmoid_parameter)
    quarter_rise = _sigmoid_growth_probability(base, epoch=6.75, start_epoch=5, end_epoch=10, sigmoid_parameter=sigmoid_parameter)
    mid = _sigmoid_growth_probability(base, epoch=7.5, start_epoch=5, end_epoch=10, sigmoid_parameter=sigmoid_parameter)
    at_end = _sigmoid_growth_probability(base, epoch=10, start_epoch=5, end_epoch=10, sigmoid_parameter=sigmoid_parameter)
    expected_quarter_rise = (base + 0.001) + 0.25 * (0.999 - (base + 0.001))
    assert np.isclose(at_start, base + 0.001)
    assert np.isclose(quarter_rise, expected_quarter_rise, atol=1e-6)
    assert at_start < quarter_rise < mid < at_end
    assert np.isclose(at_end, 0.999)
    assert np.isclose(_sigmoid_growth_probability(base, epoch=50, start_epoch=5, end_epoch=10, sigmoid_parameter=sigmoid_parameter), 1.0)


def test_learning_rate_for_epoch_accounts_for_updates_and_decay_steps():
    lr = _learning_rate_for_epoch(
        base_learning_rate=1e-3,
        min_learning_rate=1e-7,
        lr_decay_factor=0.5,
        lr_decay_every_steps=50,
        updates_per_epoch_per_agent=100,
        epoch=3,
    )
    assert np.isclose(lr, 1e-3 * (0.5 ** 6))


def test_decide_accept_double_from_probs_endless_sign():
    probs = np.zeros((31,), dtype=np.float32)
    # reward head indices 25..30 for [-3,-2,-1,+1,+2,+3]
    probs[MATCH_VECTOR_DIM * 2 + 1 + 5] = 1.0  # certain +3 for chooser
    obs = np.zeros((263,), dtype=np.float32)
    obs[-3] = 1.0
    assert decide_accept_double_from_probs(probs, obs, endless=True) == 1


def test_decide_apply_double_from_probs_endless_requires_cube_availability():
    probs_now = np.zeros((31,), dtype=np.float32)
    probs_after_double = np.zeros((31,), dtype=np.float32)
    probs_now[MATCH_VECTOR_DIM * 2] = 0.9
    probs_now[MATCH_VECTOR_DIM * 2 + 1 + 2] = 1.0  # certain -1
    probs_after_double[MATCH_VECTOR_DIM * 2 + 1 + 4] = 1.0  # certain +2

    obs = np.zeros((263,), dtype=np.float32)
    obs[-4] = 0.0
    assert decide_apply_double_from_probs(probs_now, probs_after_double, obs, endless=True) == 0

    obs[-4] = 1.0
    assert decide_apply_double_from_probs(probs_now, probs_after_double, obs, endless=True) == 1


def test_flip_observation_perspective_swaps_sides_and_controls():
    obs = np.zeros((263,), dtype=np.float32)
    obs[0] = 1.0
    obs[24 + 5] = 2.0
    obs[240:254] = np.asarray([1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float32)
    obs[254:263] = np.asarray([7, 8, 2, 4, 5, 1, 0, 1, 0], dtype=np.float32)

    flipped = flip_observation_perspective(obs)

    assert flipped[23 - (24 + 5 - 24)] == 2.0
    assert flipped[24 + 23] == 1.0
    assert np.allclose(flipped[240:244], np.asarray([3, 4, 1, 2], dtype=np.float32))
    assert np.allclose(flipped[254:261], np.asarray([8, 7, 2, 5, 4, 0, 1], dtype=np.float32))
    assert flipped[261] == 1.0
    assert flipped[262] == 0.0


def test_reject_double_equity_endless_is_immediate_loss():
    obs = np.zeros((263,), dtype=np.float32)
    obs[-3] = 1.0
    assert reject_double_equity(obs, endless=True) == -1.0


def test_learning_rate_for_epoch_can_restart_decay_from_freeze_start_epoch():
    lr = _learning_rate_for_epoch(
        base_learning_rate=1e-5,
        min_learning_rate=1e-7,
        lr_decay_factor=0.98,
        lr_decay_every_steps=50,
        updates_per_epoch_per_agent=100,
        epoch=251,
        start_epoch=250,
    )
    assert np.isclose(lr, 1e-5 * (0.98 ** 2))


def test_configure_training_phase_does_not_double_apply_prior_epoch_decay():
    cfg = ExperimentConfig()
    cfg.train.learning_rate = 1e-3
    cfg.train.lr_decay_factor = 0.5
    cfg.train.lr_decay_every_steps = 50
    agent = ValueAgent("trainable_0", "A", cfg.model_group_a, cfg.train, seed=0)

    current_epoch = 1
    completed_updates = current_epoch * cfg.train.updates_per_epoch_per_agent
    current_lr = _learning_rate_for_epoch(
        base_learning_rate=cfg.train.learning_rate,
        min_learning_rate=cfg.train.min_learning_rate,
        lr_decay_factor=cfg.train.lr_decay_factor,
        lr_decay_every_steps=cfg.train.lr_decay_every_steps,
        updates_per_epoch_per_agent=cfg.train.updates_per_epoch_per_agent,
        epoch=current_epoch,
    )

    agent.train_step = completed_updates
    configured_lr = agent.configure_training_phase(
        learning_rate=current_lr,
        lr_decay_factor=cfg.train.lr_decay_factor,
        schedule_step_offset=completed_updates,
        freeze_to_output_layer=False,
    )

    assert np.isclose(current_lr, 1e-3 * (0.5 ** 2))
    assert np.isclose(configured_lr, current_lr)

    agent.train_step = completed_updates + cfg.train.lr_decay_every_steps
    assert np.isclose(agent._apply_current_learning_rate(), 1e-3 * (0.5 ** 3))


def test_epoch_uses_output_freeze_is_inclusive():
    assert not _epoch_uses_output_freeze(epoch=249, freeze_from_epoch=250, freeze_till_epoch=400)
    assert _epoch_uses_output_freeze(epoch=250, freeze_from_epoch=250, freeze_till_epoch=400)
    assert _epoch_uses_output_freeze(epoch=400, freeze_from_epoch=250, freeze_till_epoch=400)
    assert not _epoch_uses_output_freeze(epoch=401, freeze_from_epoch=250, freeze_till_epoch=400)


def test_value_agent_can_freeze_all_but_output_layer():
    cfg = ExperimentConfig()
    agent = ValueAgent("trainable_0", "A", cfg.model_group_a, cfg.train, seed=0)

    agent.set_output_layer_only_training(True)

    frozen = [name for name, param in agent.model.named_parameters() if not name.startswith("out.") and not param.requires_grad]
    trainable = [name for name, param in agent.model.named_parameters() if param.requires_grad]
    assert frozen
    assert trainable
    assert all(name.startswith("out.") for name in trainable)

    agent.set_output_layer_only_training(False)
    assert all(param.requires_grad for _, param in agent.model.named_parameters())


def test_run_training_switches_lr_policy_during_freeze_and_restores_regular_schedule(tmp_path: Path, monkeypatch):
    import training.pipeline as pipeline_mod

    class FakeAgent:
        def __init__(self, agent_id: str):
            self.agent_id = agent_id
            self.optimizer = type("Opt", (), {"param_groups": [{"lr": 0.0}]})()
            self.train_step = 0
            self.phase_history: list[tuple[float, float, int, bool]] = []

        def configure_training_phase(self, learning_rate: float, lr_decay_factor: float, schedule_step_offset: int, freeze_to_output_layer: bool) -> float:
            self.phase_history.append((learning_rate, lr_decay_factor, schedule_step_offset, freeze_to_output_layer))
            self.optimizer.param_groups[0]["lr"] = float(learning_rate)
            return float(learning_rate)

        def state_dict(self) -> dict:
            return {"agent_id": self.agent_id, "group": "A", "model": {}}

    class FakeLeague:
        def __init__(self, *_args, **_kwargs):
            self._decision_stats = {"decision_count": 0, "topk_freq": [0.0] * 10}

        def set_decision_temperature(self, _value: float) -> None:
            return None

        def set_choose_best_probability(self, _value: float) -> None:
            return None

        def run_epoch(self, _agents, _epoch: int):
            return [], 0.0

        def get_decision_stats(self) -> dict:
            return self._decision_stats

    class FakeReplay:
        def __init__(self, *_args, **_kwargs):
            self.size = 0

        def __len__(self) -> int:
            return self.size

        def add_many(self, _records) -> None:
            return None

        def get_meta(self) -> dict:
            return {"size": self.size}

    fake_agents = [FakeAgent("trainable_0"), FakeAgent("trainable_1")]
    monkeypatch.setattr(pipeline_mod, "build_trainable_agents", lambda cfg, seed: fake_agents)
    monkeypatch.setattr(pipeline_mod, "LeagueController", FakeLeague)
    monkeypatch.setattr(pipeline_mod, "ReplayBuffer", FakeReplay)
    monkeypatch.setattr(pipeline_mod, "load_metrics_history_from_checkpoints", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline_mod, "plot_metrics_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_mod, "save_checkpoint", lambda *_args, **_kwargs: None)

    cfg = ExperimentConfig()
    cfg.train.num_epochs = 3
    cfg.train.updates_per_epoch_per_agent = 1
    cfg.train.learning_rate = 1e-3
    cfg.train.lr_decay_factor = 0.5
    cfg.train.lr_decay_every_steps = 1
    cfg.train.freeze_weights_from_epoch = 1
    cfg.train.freeze_weights_till_epoch = 1
    cfg.train.lr_during_freeze = 5e-4
    cfg.train.lr_decay_during_freeze = 0.1
    cfg.league.min_replay_size_to_train = 10**9
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.plots_dir = str(tmp_path / "plots")
    cfg.league.replay_storage_dir = str(tmp_path / "replay")

    metrics = run_training(cfg)

    assert len(metrics) == 3
    first_agent_id = "trainable_0"
    assert np.isclose(metrics[0]["agents"][first_agent_id]["learning_rate"], 1e-3)
    assert not metrics[0]["freeze_output_layer_only"]
    assert np.isclose(metrics[1]["agents"][first_agent_id]["learning_rate"], 5e-4)
    assert metrics[1]["freeze_output_layer_only"]
    assert np.isclose(metrics[2]["agents"][first_agent_id]["learning_rate"], 1e-3 * (0.5 ** 2))
    assert not metrics[2]["freeze_output_layer_only"]
