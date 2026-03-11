from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from .agents import build_trainable_agents
from .config import ExperimentConfig, save_config
from .league import LeagueController
from .replay import ReplayBuffer
from .plotting import load_metrics_history_from_checkpoints, plot_metrics_history


MATCH_VECTOR_DIM = 12
REWARD_VECTOR_DIM = 6
MODEL_OUTPUT_DIM = 31


def _left_to_win_from_state_vector(state_vector: np.ndarray) -> tuple[int, int]:
    vec = np.asarray(state_vector, dtype=np.float32).reshape(-1)
    if vec.size < 6:
        return MATCH_VECTOR_DIM - 1, MATCH_VECTOR_DIM - 1
    my_left = int(np.clip(np.round(float(vec[-6])), 0, MATCH_VECTOR_DIM - 1))
    opp_left = int(np.clip(np.round(float(vec[-5])), 0, MATCH_VECTOR_DIM - 1))
    return my_left, opp_left

def _games_for_pair(game_results: list, agent_id: str, opponent_id: str) -> list:
    pair = []
    for g in game_results:
        participants = {g.player_1_id, g.player_2_id}
        if {agent_id, opponent_id} == participants:
            pair.append(g)
    return pair


def _terminal_outcome_for_step(game, step: dict) -> np.ndarray:
    player_index = step.get("player_index")
    if player_index is not None:
        won = int(player_index) == int(getattr(game, "winner_player_index", 0))
    else:
        won = step["agent_id"] == game.winner

    my_left, opp_left = _left_to_win_from_state_vector(step.get("state_vector", np.empty((0,), dtype=np.float32)))
    points_won = int(max(getattr(game, "points_won", 1), 1))

    my_next = max(my_left - points_won, 0) if won else my_left
    opp_next = opp_left if won else max(opp_left - points_won, 0)

    my_vec = np.zeros((MATCH_VECTOR_DIM,), dtype=np.float32)
    opp_vec = np.zeros((MATCH_VECTOR_DIM,), dtype=np.float32)
    my_vec[my_next] = 1.0
    opp_vec[opp_next] = 1.0

    accept_target = np.array([1.0 if bool(step.get("accept_double_opponent", False)) else 0.0], dtype=np.float32)

    reward_value = int(np.clip(getattr(game, "reward_value", 1), 1, 3))
    signed_reward = reward_value if won else -reward_value
    reward_vec = np.zeros((REWARD_VECTOR_DIM,), dtype=np.float32)
    reward_vec[signed_reward + 3 - (1 if signed_reward > 0 else 0)] = 1.0

    out = np.concatenate([my_vec, opp_vec, accept_target, reward_vec], dtype=np.float32)
    return out


def save_checkpoint(cfg: ExperimentConfig, agents, replay: ReplayBuffer, epoch: int, metrics: dict) -> None:
    d = Path(cfg.checkpoint_dir) / f"epoch_{epoch:04d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "agents.json").write_text(json.dumps([a.state_dict() for a in agents]), encoding="utf-8")
    (d / "replay_meta.json").write_text(json.dumps(replay.get_meta()), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_config(cfg, d / "config.json")


def load_checkpoint(cfg: ExperimentConfig, agents, epoch: int) -> None:
    d = Path(cfg.checkpoint_dir) / f"epoch_{epoch:04d}"
    states = json.loads((d / "agents.json").read_text(encoding="utf-8"))
    for a, s in zip(agents, states):
        a.load_state_dict(s)


def _clear_replay_storage(storage_dir: str) -> None:
    replay_dir = Path(storage_dir)
    for p in replay_dir.glob("replay.sqlite3*"):
        if not p.is_file():
            continue
        try:
            p.unlink()
        except PermissionError:
            # On Windows a live SQLite handle can keep the file locked.
            # Fallback cleanup is performed by ReplayBuffer(clear_existing=True).
            continue


def run_training(cfg: ExperimentConfig, start_epoch: int = 0) -> list[dict]:
    np.random.seed(cfg.train.seed)
    agents = build_trainable_agents(cfg, cfg.train.seed)
    league = LeagueController(cfg.league, seed=cfg.train.seed)
    fresh_start = start_epoch <= 0
    if fresh_start:
        _clear_replay_storage(cfg.league.replay_storage_dir)
    replay = ReplayBuffer(
        cfg.league.replay_storage_dir,
        recency_decay=cfg.league.recency_decay,
        recency_center_mass_ratio=cfg.league.recency_center_mass_ratio,
        clear_existing=fresh_start,
    )

    if start_epoch > 0:
        load_checkpoint(cfg, agents, start_epoch - 1)
    metrics_history = load_metrics_history_from_checkpoints(Path(cfg.checkpoint_dir), start_epoch)

    all_agent_ids = [x.agent_id for x in agents] + ["conservative_baseline"]

    current_temperature = float(cfg.league.selfplay_temperature) * (float(cfg.league.temperature_decay) ** max(start_epoch, 0))
    current_choose_best_probability = float(cfg.league.choose_best_probability)

    for epoch in range(start_epoch, cfg.train.num_epochs):
        epoch_t0 = time.time()
        print(f"Epoch {epoch}\n")
        print("[1/6] Self-play started")
        play_t0 = time.time()
        league.set_decision_temperature(current_temperature)
        league.set_choose_best_probability(current_choose_best_probability)
        game_results, games_sec = league.run_epoch(agents, epoch)
        decision_stats = league.get_decision_stats()
        play_dt = max(time.time() - play_t0, 1e-6)
        print(f"[1/6] Self-play took {play_dt:.2f} seconds")

        replay_add_t0 = time.time()
        for game in game_results:
            records = []
            for st in game.steps:
                outcome = _terminal_outcome_for_step(game, st)
                records.append({**st, "terminal_outcome": outcome})
            replay.add_many(records)
        replay_add_dt = max(time.time() - replay_add_t0, 1e-6)
        print(f"[2/6] Replay append took {replay_add_dt:.2f} seconds")

        winrates_t0 = time.time()
        winrates_vs_baseline = []
        for agent in agents:
            pair_baseline = _games_for_pair(game_results, agent.agent_id, "conservative_baseline")
            wr_baseline = sum(1 for g in pair_baseline if g.winner == agent.agent_id) / len(pair_baseline) if pair_baseline else 0.0
            winrates_vs_baseline.append(round(wr_baseline * 100.0, 2))
        winrates_dt = max(time.time() - winrates_t0, 1e-6)
        print(f"[3/6] Winrate aggregation took {winrates_dt:.2f} seconds")

        print(f"Winrates vs conservative baseline: {winrates_vs_baseline}\n")
        print("[4/6] Training started")

        train_losses: dict[str, list[float]] = {a.agent_id: [] for a in agents}
        train_lrs_steps: dict[str, list[float]] = {a.agent_id: [] for a in agents}
        t0 = time.time()
        replay_sample_time_total = 0.0
        replay_sample_calls = 0
        if len(replay) >= cfg.league.min_replay_size_to_train:
            batch_by_agent = {agent.agent_id: cfg.train.batch_size for agent in agents}

            for _ in range(cfg.train.updates_per_epoch_per_agent):
                sample_t0 = time.time()
                stratified_batches = replay.sample_stratified_with_agent_ids(
                    batch_by_agent,
                    cfg.league.alpha_recency,
                    cfg.league.alpha_uniform,
                    cfg.league.recency_window,
                )
                replay_sample_time_total += time.time() - sample_t0
                replay_sample_calls += 1

                for agent in agents:
                    x_np, y_np = stratified_batches.get(
                        agent.agent_id,
                        (np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)),
                    )
                    if x_np.shape[0] == 0:
                        continue

                    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=agent.device)
                    y_t = torch.as_tensor(y_np, dtype=torch.float32, device=agent.device)
                    train_losses[agent.agent_id].append(agent.train_batch_tensor(x_t, y_t))
                    train_lrs_steps[agent.agent_id].append(float(agent.optimizer.param_groups[0]["lr"]))
        train_dt = max(time.time() - t0, 1e-6)
        steps_per_sec = (cfg.train.batch_size * cfg.train.updates_per_epoch_per_agent * len(agents)) / train_dt

        per_agent = {aid: {
            "train_loss_epoch": None,
            "train_loss_steps_epoch": [],
            "learning_rate": None,
            "learning_rate_steps_epoch": [],
            "learning_steps_epoch": 0,
            "winrate_vs_random": 0.0,
            "winrate_vs_baseline": 0.0,
            "aggregate_winrate_vs_trainable": 0.0,
            "avg_game_length": float(np.mean([g.turns for g in game_results]) if game_results else 0.0),
            "winrate_vs_opponents": {},
        } for aid in all_agent_ids}

        for a in agents:
            per_agent[a.agent_id]["train_loss_epoch"] = float(np.mean(train_losses[a.agent_id]) if train_losses[a.agent_id] else 0.0)
            per_agent[a.agent_id]["train_loss_steps_epoch"] = train_losses[a.agent_id]
            per_agent[a.agent_id]["learning_rate"] = float(a.optimizer.param_groups[0]["lr"])
            per_agent[a.agent_id]["learning_rate_steps_epoch"] = train_lrs_steps[a.agent_id]
            per_agent[a.agent_id]["learning_steps_epoch"] = len(train_lrs_steps[a.agent_id])

        for aid in all_agent_ids:
            for opp in all_agent_ids:
                if opp == aid:
                    continue
                pair = _games_for_pair(game_results, aid, opp)
                if not pair:
                    wr = 0.0
                else:
                    wr = sum(1 for g in pair if g.winner == aid) / len(pair)
                per_agent[aid]["winrate_vs_opponents"][opp] = wr

            trainable_opponents = [x.agent_id for x in agents if x.agent_id != aid]
            if trainable_opponents:
                per_agent[aid]["aggregate_winrate_vs_trainable"] = float(np.mean([per_agent[aid]["winrate_vs_opponents"].get(t, 0.0) for t in trainable_opponents]))

            per_agent[aid]["winrate_vs_random"] = 0.0
            per_agent[aid]["winrate_vs_baseline"] = per_agent[aid]["winrate_vs_opponents"].get("conservative_baseline", 0.0)

        avg_sample_ms = (replay_sample_time_total / replay_sample_calls * 1000.0) if replay_sample_calls else 0.0
        pure_train_dt = max(train_dt - replay_sample_time_total, 0.0)
        print(f"[4/6] Training (model update only) took {pure_train_dt:.2f} seconds")
        print(f"[5/6] Replay sampling took {replay_sample_time_total:.2f} seconds (avg={avg_sample_ms:.2f}ms, calls={replay_sample_calls})")
        print(f"Losses: {[round(float(np.mean(train_losses[a.agent_id]) if train_losses[a.agent_id] else 0.0), 6) for a in agents]}\n")

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            torch.cuda.reset_peak_memory_stats()

        metrics = {
            "epoch": epoch,
            "epoch_total_sec": max(time.time() - epoch_t0, 1e-6),
            "games_sec": games_sec,
            "steps_sec": steps_per_sec,
            "gpu_mem_mb": gpu_mem,
            "replay_sampling_total_sec": replay_sample_time_total,
            "replay_sampling_avg_ms": avg_sample_ms,
            "replay_size": int(len(replay)),
            "timings": {
                "selfplay_sec": play_dt,
                "replay_append_sec": replay_add_dt,
                "winrate_aggregation_sec": winrates_dt,
                "training_total_sec": train_dt,
                "training_model_only_sec": pure_train_dt,
                "replay_sampling_sec": replay_sample_time_total,
            },
            "decision_temperature": float(current_temperature),
            "choose_best_probability": float(current_choose_best_probability),
            "decision_count": int(decision_stats["decision_count"]),
            "decision_topk_freq": decision_stats["topk_freq"],
            "agents": per_agent,
        }
        metrics_history.append(metrics)

        epoch_total_dt = float(metrics["epoch_total_sec"])
        print(f"[6/6] Epoch total took {epoch_total_dt:.2f} seconds\n")

        should_plot = (epoch + 1) % max(cfg.train.plot_every_k_epochs, 1) == 0 or epoch == cfg.train.num_epochs - 1
        if should_plot:
            plot_metrics_history(
                metrics_history,
                Path(cfg.plots_dir),
                cfg.train.winrate_window_size,
                cfg.league.alpha_recency,
                cfg.league.alpha_uniform,
                cfg.league.recency_center_mass_ratio,
            )

        if epoch % cfg.league.checkpoint_frequency_epochs == 0:
            save_checkpoint(cfg, agents, replay, epoch, metrics)

        current_temperature *= float(cfg.league.temperature_decay)
        current_choose_best_probability = 1.0 - (1.0 - current_choose_best_probability) * float(cfg.league.choose_best_decay)

    return metrics_history
