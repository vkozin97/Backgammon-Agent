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


def _games_for_pair(game_results: list, agent_id: str, opponent_id: str) -> list:
    pair = []
    for g in game_results:
        participants = {g.player_1_id, g.player_2_id}
        if {agent_id, opponent_id} == participants:
            pair.append(g)
    return pair


def _terminal_outcome_for_step(game, step: dict) -> float:
    player_index = step.get("player_index")
    if player_index is not None:
        return 1.0 if int(player_index) == int(getattr(game, "winner_player_index", 0)) else 0.0
    return 1.0 if step["agent_id"] == game.winner else 0.0


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


def _exp_windowed(values: list[float], window_size: int) -> list[float]:
    if not values:
        return []
    w = max(int(window_size), 1)
    out: list[float] = []
    for i in range(len(values)):
        left = max(0, i - w + 1)
        chunk = values[left:i + 1]
        # Newer points get larger weights: exp(0), exp(-1), exp(-2), ...
        age = np.arange(len(chunk) - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-age)
        out.append(float(np.dot(np.asarray(chunk, dtype=np.float64), weights) / np.sum(weights)))
    return out


def _uniform_windowed(values: list[float], window_size: int) -> list[float]:
    if not values:
        return []
    w = max(int(window_size), 1)
    out: list[float] = []
    for i in range(len(values)):
        left = max(0, i - w + 1)
        chunk = values[left:i + 1]
        out.append(float(np.mean(np.asarray(chunk, dtype=np.float64))))
    return out



def _print_plot_timing(stage: str, started_at: float, total_started_at: float) -> None:
    stage_dt = max(time.perf_counter() - started_at, 0.0)
    total_dt = max(time.perf_counter() - total_started_at, 0.0)
    print(f"[plot] {stage} took {stage_dt:.2f}s (total={total_dt:.2f}s)")


def _load_metrics_history_from_checkpoints(checkpoint_dir: Path, end_epoch_exclusive: int) -> list[dict]:
    if end_epoch_exclusive <= 0:
        return []
    loaded: list[dict] = []
    for epoch in range(end_epoch_exclusive):
        metrics_path = checkpoint_dir / f"epoch_{epoch:04d}" / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(metrics, dict):
            loaded.append(metrics)
    return loaded

def _plot(
    metrics_history: list[dict],
    out_dir: Path,
    winrate_window_size: int,
    alpha_recency: float,
    alpha_uniform: float,
    recency_center_mass_ratio: float,
) -> None:
    plot_total_t0 = time.perf_counter()
    print(f"[plot] Start plotting for {len(metrics_history)} epochs")
    step_t0 = time.perf_counter()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib is unavailable, skip plotting")
        return
    _print_plot_timing("matplotlib import", step_t0, plot_total_t0)
    winrates_dir = out_dir / "winrates"
    winrates_windowed_dir = out_dir / "winrates_windowed"
    loss_dir = out_dir / "loss"
    lr_dir = out_dir / "lr"
    decision_dir = out_dir / "decision_temperature"
    step_t0 = time.perf_counter()
    winrates_dir.mkdir(parents=True, exist_ok=True)
    winrates_windowed_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    _print_plot_timing("prepare output directories", step_t0, plot_total_t0)

    agents = sorted(metrics_history[-1]["agents"].keys())


    xs = [m["epoch"] for m in metrics_history]

    step_t0 = time.perf_counter()
    temps = [float(m.get("decision_temperature", np.nan)) for m in metrics_history]
    if any(np.isfinite(v) for v in temps):
        plt.figure(figsize=(18, 9))
        plt.plot(xs, temps)
        plt.title("Decision temperature")
        plt.xlabel("epoch")
        plt.ylabel("temperature")
        plt.tight_layout()
        plt.savefig(decision_dir / "decision_temperature.png")
        plt.close()

    for k in range(1, 11):
        topk_series = []
        for m in metrics_history:
            vals = m.get("decision_topk_freq", [])
            topk_series.append(float(vals[k - 1]) * 100.0 if len(vals) >= k else np.nan)
        if any(np.isfinite(v) for v in topk_series):
            plt.figure(figsize=(18, 9))
            plt.plot(xs, topk_series)
            plt.title(f"Selected action in top-{k} values")
            plt.xlabel("epoch")
            plt.ylabel("frequency (%)")
            plt.ylim(0.0, 100.0)
            plt.tight_layout()
            plt.savefig(decision_dir / f"selected_action_top_{k}.png")
            plt.close()
    _print_plot_timing("decision-temperature and top-k plots", step_t0, plot_total_t0)

    step_t0 = time.perf_counter()
    fixed_opponents = {"conservative_baseline"}
    for aid in agents:
        agent_t0 = time.perf_counter()
        latest_opponents = set(metrics_history[-1]["agents"][aid].get("winrate_vs_opponents", {}).keys())
        opponents_for_agent = sorted((latest_opponents | fixed_opponents) - {aid})
        xs = [m["epoch"] for m in metrics_history]

        # Winrate graphs: each figure contains all opponent lines,
        # with one opponent highlighted in the title/line style.
        series_by_opp = {
            opp: [m["agents"][aid]["winrate_vs_opponents"].get(opp, 0.0) * 100.0 for m in metrics_history]
            for opp in opponents_for_agent
        }
        windowed_series_by_opp = {
            opp: _uniform_windowed(series_by_opp[opp], winrate_window_size)
            for opp in opponents_for_agent
        }

        for focus_opp in opponents_for_agent:
            plt.figure(figsize=(18, 9))
            for opp in opponents_for_agent:
                lw = 2.4 if opp == focus_opp else 1.0
                alpha = 1.0 if opp == focus_opp else 0.4
                plt.plot(xs, series_by_opp[opp], label=opp, linewidth=lw, alpha=alpha)
            plt.title(f"{aid} winrates (focus: {focus_opp})")
            plt.xlabel("epoch")
            plt.ylabel("winrate (%)")
            plt.ylim(0.0, 100.0)
            plt.legend(fontsize=6, ncol=2)
            plt.tight_layout()
            plt.savefig(winrates_dir / f"{aid}_winrates_focus_{focus_opp}.png")
            plt.close()

            plt.figure(figsize=(18, 9))
            for opp in opponents_for_agent:
                lw = 2.4 if opp == focus_opp else 1.0
                alpha = 1.0 if opp == focus_opp else 0.4
                plt.plot(xs, windowed_series_by_opp[opp], label=opp, linewidth=lw, alpha=alpha)
            plt.title(f"{aid} winrates windowed (focus: {focus_opp}, w={max(int(winrate_window_size), 1)})")
            plt.xlabel("epoch")
            plt.ylabel("windowed winrate (%)")
            plt.ylim(0.0, 100.0)
            plt.legend(fontsize=6, ncol=2)
            plt.tight_layout()
            plt.savefig(winrates_windowed_dir / f"{aid}_winrates_windowed_focus_{focus_opp}.png")
            plt.close()

        _print_plot_timing(f"{aid}: winrate plots", agent_t0, plot_total_t0)
        agent_t0 = time.perf_counter()

        loss_steps: list[float] = []
        loss_epoch_end_steps: list[int] = []
        loss_cursor = 0
        for m in metrics_history:
            epoch_loss_steps = m["agents"][aid].get("train_loss_steps_epoch", [])
            if epoch_loss_steps:
                sanitized_epoch_loss = [float(v) for v in epoch_loss_steps]
                loss_steps.extend(sanitized_epoch_loss)
            loss_cursor += len(epoch_loss_steps)
            loss_epoch_end_steps.append(loss_cursor)

        if loss_steps:
            loss_xs = list(range(1, len(loss_steps) + 1))
            loss_epoch_boundaries = sorted({x for x in loss_epoch_end_steps[:-1] if 0 < x < len(loss_steps)})
            plt.figure(figsize=(18, 9))
            plt.plot(loss_xs, loss_steps, linewidth=1.2)
            for boundary in loss_epoch_boundaries:
                plt.axvline(x=boundary + 0.5, linestyle=":", color="gray", linewidth=0.8, alpha=0.8)
            plt.title(f"{aid} train loss")
            plt.xlabel("learning step")
            plt.ylabel("loss")
            plt.tight_layout()
            plt.savefig(loss_dir / f"{aid}_loss.png")
            plt.close()
        _print_plot_timing(f"{aid}: loss plot", agent_t0, plot_total_t0)
        agent_t0 = time.perf_counter()

        lr_steps: list[float] = []
        epoch_end_steps: list[int] = []
        learning_steps_per_epoch: list[int] = []
        cursor = 0
        for m in metrics_history:
            epoch_lr_steps = m["agents"][aid].get("learning_rate_steps_epoch", [])
            if epoch_lr_steps:
                sanitized_epoch_lr = [float(v) for v in epoch_lr_steps]
                lr_steps.extend(sanitized_epoch_lr)
                learning_steps_per_epoch.append(len(sanitized_epoch_lr))
            cursor += len(epoch_lr_steps)
            epoch_end_steps.append(cursor)

        if lr_steps:
            lr_xs = list(range(1, len(lr_steps) + 1))
            epoch_boundaries = sorted({x for x in epoch_end_steps[:-1] if 0 < x < len(lr_steps)})

            plt.figure(figsize=(18, 9))
            plt.plot(lr_xs, lr_steps)
            for boundary in epoch_boundaries:
                plt.axvline(x=boundary + 0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.8)
            plt.title(f"{aid} learning rate")
            plt.xlabel("learning step")
            plt.ylabel("lr")
            plt.tight_layout()
            plt.savefig(lr_dir / f"{aid}_lr.png")
            plt.close()

            base_steps = learning_steps_per_epoch[0] if learning_steps_per_epoch else len(lr_steps)
            lr_window = max(int(round(base_steps * 0.25)), 1)
            lr_steps_windowed = _exp_windowed(lr_steps, lr_window)
            plt.figure(figsize=(18, 9))
            plt.plot(lr_xs, lr_steps_windowed)
            for boundary in epoch_boundaries:
                plt.axvline(x=boundary + 0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.8)
            plt.title(f"{aid} learning rate windowed (w={lr_window})")
            plt.xlabel("learning step")
            plt.ylabel("windowed lr")
            plt.tight_layout()
            plt.savefig(lr_dir / f"{aid}_lr_windowed.png")
            plt.close()
        _print_plot_timing(f"{aid}: lr plots", agent_t0, plot_total_t0)

    _print_plot_timing("all per-agent plot groups", step_t0, plot_total_t0)

    step_t0 = time.perf_counter()
    overall_series: dict[str, list[float]] = {}
    for aid in agents:
        vals = []
        for m in metrics_history:
            opp_vals = list(m["agents"][aid].get("winrate_vs_opponents", {}).values())
            vals.append(float(np.mean(opp_vals) * 100.0) if opp_vals else 0.0)
        overall_series[aid] = vals

    if overall_series:
        plt.figure(figsize=(18, 9))
        for aid in agents:
            plt.plot(xs, overall_series[aid], label=aid, linewidth=1.4)
        plt.title("Agents overall average winrate vs all opponents")
        plt.xlabel("epoch")
        plt.ylabel("winrate (%)")
        plt.ylim(0.0, 100.0)
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(winrates_dir / "overall_avg_winrates.png")
        plt.close()

        plt.figure(figsize=(18, 9))
        for aid in agents:
            plt.plot(xs, _uniform_windowed(overall_series[aid], winrate_window_size), label=aid, linewidth=1.4)
        plt.title(f"Agents overall average winrate windowed (w={max(int(winrate_window_size), 1)})")
        plt.xlabel("epoch")
        plt.ylabel("windowed winrate (%)")
        plt.ylim(0.0, 100.0)
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(winrates_windowed_dir / "overall_avg_winrates_windowed.png")
        plt.close()

    _print_plot_timing("overall average winrate plots", step_t0, plot_total_t0)
    _print_plot_timing("full plotting stage", plot_total_t0, plot_total_t0)


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
    metrics_history = _load_metrics_history_from_checkpoints(Path(cfg.checkpoint_dir), start_epoch)

    all_agent_ids = [x.agent_id for x in agents] + ["conservative_baseline"]

    current_temperature = float(cfg.league.selfplay_temperature) * (float(cfg.league.temperature_decay) ** max(start_epoch, 0))

    for epoch in range(start_epoch, cfg.train.num_epochs):
        epoch_t0 = time.time()
        print(f"Epoch {epoch}\n")
        print("[1/6] Self-play started")
        play_t0 = time.time()
        league.set_decision_temperature(current_temperature)
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
                        (np.empty((0, 0), dtype=np.float32), np.empty((0, 1), dtype=np.float32)),
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
            "decision_count": int(decision_stats["decision_count"]),
            "decision_topk_freq": decision_stats["topk_freq"],
            "agents": per_agent,
        }
        metrics_history.append(metrics)

        epoch_total_dt = float(metrics["epoch_total_sec"])
        print(f"[6/6] Epoch total took {epoch_total_dt:.2f} seconds\n")

        should_plot = (epoch + 1) % max(cfg.train.plot_every_k_epochs, 1) == 0 or epoch == cfg.train.num_epochs - 1
        if should_plot:
            _plot(
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

    return metrics_history
