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


def _plot(metrics_history: list[dict], out_dir: Path, winrate_window_size: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    winrates_dir = out_dir / "winrates"
    winrates_windowed_dir = out_dir / "winrates_windowed"
    loss_dir = out_dir / "loss"
    lr_dir = out_dir / "lr"
    winrates_dir.mkdir(parents=True, exist_ok=True)
    winrates_windowed_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    agents = sorted(metrics_history[-1]["agents"].keys())

    for aid in agents:
        opponents_for_agent = sorted(metrics_history[-1]["agents"][aid]["winrate_vs_opponents"].keys())
        xs = [m["epoch"] for m in metrics_history]

        # Winrate graphs: each figure contains all opponent lines,
        # with one opponent highlighted in the title/line style.
        series_by_opp = {
            opp: [m["agents"][aid]["winrate_vs_opponents"].get(opp, 0.0) * 100.0 for m in metrics_history]
            for opp in opponents_for_agent
        }
        windowed_series_by_opp = {
            opp: _exp_windowed(series_by_opp[opp], winrate_window_size)
            for opp in opponents_for_agent
        }

        for focus_opp in opponents_for_agent:
            plt.figure(figsize=(6, 3))
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

            plt.figure(figsize=(6, 3))
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

        losses = [m["agents"][aid].get("train_loss_epoch") for m in metrics_history]
        if any(v is not None for v in losses):
            sanitized_loss = [float(v) if v is not None else np.nan for v in losses]
            plt.figure(figsize=(6, 3))
            plt.plot(xs, sanitized_loss)
            plt.title(f"{aid} train loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.tight_layout()
            plt.savefig(loss_dir / f"{aid}_loss.png")
            plt.close()

        lrs = [m["agents"][aid].get("learning_rate") for m in metrics_history]
        if any(v is not None for v in lrs):
            sanitized_lr = [float(v) if v is not None else np.nan for v in lrs]
            plt.figure(figsize=(6, 3))
            plt.plot(xs, sanitized_lr)
            plt.title(f"{aid} learning rate")
            plt.xlabel("epoch")
            plt.ylabel("lr")
            plt.tight_layout()
            plt.savefig(lr_dir / f"{aid}_lr.png")
            plt.close()


def run_training(cfg: ExperimentConfig) -> list[dict]:
    np.random.seed(cfg.train.seed)
    agents = build_trainable_agents(cfg, cfg.train.seed)
    league = LeagueController(cfg.league, seed=cfg.train.seed)
    replay = ReplayBuffer(cfg.league.replay_storage_dir, recency_decay=cfg.league.recency_decay)
    metrics_history: list[dict] = []

    all_agent_ids = [x.agent_id for x in agents] + ["random", "baseline"]

    for epoch in range(cfg.train.num_epochs):
        print(f"Epoch {epoch}\n")
        print("Playing started")
        play_t0 = time.time()
        game_results, games_sec = league.run_epoch(agents, epoch)
        play_dt = max(time.time() - play_t0, 1e-6)
        print(f"Playing took {play_dt:.2f} seconds")
        for game in game_results:
            for st in game.steps:
                outcome = 1.0 if st["agent_id"] == game.winner else 0.0
                replay.add(**st, terminal_outcome=outcome)

        winrates_vs_baseline = []
        winrates_vs_random = []
        for agent in agents:
            pair_baseline = _games_for_pair(game_results, agent.agent_id, "baseline")
            pair_random = _games_for_pair(game_results, agent.agent_id, "random")
            wr_baseline = sum(1 for g in pair_baseline if g.winner == agent.agent_id) / len(pair_baseline) if pair_baseline else 0.0
            wr_random = sum(1 for g in pair_random if g.winner == agent.agent_id) / len(pair_random) if pair_random else 0.0
            winrates_vs_baseline.append(round(wr_baseline * 100.0, 2))
            winrates_vs_random.append(round(wr_random * 100.0, 2))

        print(f"Winrates vs baseline: {winrates_vs_baseline}")
        print(f"Winrates vs random: {winrates_vs_random}\n")
        print("Training started")

        train_losses: dict[str, list[float]] = {a.agent_id: [] for a in agents}
        t0 = time.time()
        replay_sample_time_total = 0.0
        replay_sample_calls = 0
        if len(replay) >= cfg.league.min_replay_size_to_train:
            agents_by_group: dict[str, list] = {}
            for agent in agents:
                agents_by_group.setdefault(agent.group, []).append(agent)

            for group_agents in agents_by_group.values():
                for _ in range(cfg.train.updates_per_epoch_per_agent):
                    # One replay sample and one host->device transfer per architecture group.
                    sample_t0 = time.time()
                    x_np, y_np = replay.sample(
                        cfg.train.batch_size,
                        cfg.league.alpha_recency,
                        cfg.league.alpha_uniform,
                        cfg.league.recency_window,
                    )
                    replay_sample_time_total += time.time() - sample_t0
                    replay_sample_calls += 1

                    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=group_agents[0].device)
                    y_t = torch.as_tensor(y_np, dtype=torch.float32, device=group_agents[0].device)

                    for agent in group_agents:
                        train_losses[agent.agent_id].append(agent.train_batch_tensor(x_t, y_t))
        train_dt = max(time.time() - t0, 1e-6)
        steps_per_sec = (cfg.train.batch_size * cfg.train.updates_per_epoch_per_agent * len(agents)) / train_dt

        per_agent = {aid: {
            "train_loss_epoch": None,
            "learning_rate": None,
            "winrate_vs_random": 0.0,
            "winrate_vs_baseline": 0.0,
            "aggregate_winrate_vs_trainable": 0.0,
            "avg_game_length": float(np.mean([g.turns for g in game_results]) if game_results else 0.0),
            "winrate_vs_opponents": {},
        } for aid in all_agent_ids}

        for a in agents:
            per_agent[a.agent_id]["train_loss_epoch"] = float(np.mean(train_losses[a.agent_id]) if train_losses[a.agent_id] else 0.0)
            per_agent[a.agent_id]["learning_rate"] = float(a.optimizer.param_groups[0]["lr"])

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

            per_agent[aid]["winrate_vs_random"] = per_agent[aid]["winrate_vs_opponents"].get("random", 0.0)
            per_agent[aid]["winrate_vs_baseline"] = per_agent[aid]["winrate_vs_opponents"].get("baseline", 0.0)

        avg_sample_ms = (replay_sample_time_total / replay_sample_calls * 1000.0) if replay_sample_calls else 0.0
        print(f"Training took {train_dt:.2f} seconds")
        print(f"Replay sampling time: total={replay_sample_time_total:.2f}s avg={avg_sample_ms:.2f}ms calls={replay_sample_calls}")
        print(f"Losses: {[round(float(np.mean(train_losses[a.agent_id]) if train_losses[a.agent_id] else 0.0), 6) for a in agents]}\n")

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            torch.cuda.reset_peak_memory_stats()

        metrics = {
            "epoch": epoch,
            "games_sec": games_sec,
            "steps_sec": steps_per_sec,
            "gpu_mem_mb": gpu_mem,
            "replay_sampling_total_sec": replay_sample_time_total,
            "replay_sampling_avg_ms": avg_sample_ms,
            "agents": per_agent,
        }
        metrics_history.append(metrics)

        should_plot = (epoch + 1) % max(cfg.train.plot_every_k_epochs, 1) == 0 or epoch == cfg.train.num_epochs - 1
        if should_plot:
            _plot(metrics_history, Path(cfg.plots_dir), cfg.train.winrate_window_size)

        if epoch % cfg.league.checkpoint_frequency_epochs == 0:
            save_checkpoint(cfg, agents, replay, epoch, metrics)

    return metrics_history
