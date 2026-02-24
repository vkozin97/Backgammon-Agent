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
    (d / "replay_meta.json").write_text(json.dumps({"size": len(replay), "counter": replay._counter}), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_config(cfg, d / "config.json")


def load_checkpoint(cfg: ExperimentConfig, agents, epoch: int) -> None:
    d = Path(cfg.checkpoint_dir) / f"epoch_{epoch:04d}"
    states = json.loads((d / "agents.json").read_text(encoding="utf-8"))
    for a, s in zip(agents, states):
        a.load_state_dict(s)


def _plot(metrics_history: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    winrates_dir = out_dir / "winrates"
    loss_dir = out_dir / "loss"
    winrates_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    agents = sorted(metrics_history[-1]["agents"].keys())
    all_opponents = sorted(metrics_history[-1]["agents"][agents[0]]["winrate_vs_opponents"].keys())

    for aid in agents:
        xs = [m["epoch"] for m in metrics_history]

        # 1..13) Winrate graphs: each figure contains all 13 opponent lines,
        # with one opponent highlighted in the title/line style.
        for focus_opp in all_opponents:
            plt.figure(figsize=(6, 3))
            for opp in all_opponents:
                ys = [m["agents"][aid]["winrate_vs_opponents"].get(opp, 0.0) for m in metrics_history]
                lw = 2.4 if opp == focus_opp else 1.0
                alpha = 1.0 if opp == focus_opp else 0.4
                plt.plot(xs, ys, label=opp, linewidth=lw, alpha=alpha)
            plt.title(f"{aid} winrates (focus: {focus_opp})")
            plt.xlabel("epoch")
            plt.ylabel("winrate")
            plt.legend(fontsize=6, ncol=2)
            plt.tight_layout()
            plt.savefig(winrates_dir / f"{aid}_winrates_focus_{focus_opp}.png")
            plt.close()

        # 14) Loss graph.
        loss = [m["agents"][aid]["train_loss_epoch"] for m in metrics_history]
        plt.figure(figsize=(6, 3))
        plt.plot(xs, loss)
        plt.title(f"{aid} train loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(loss_dir / f"{aid}_loss.png")
        plt.close()


def run_training(cfg: ExperimentConfig) -> list[dict]:
    np.random.seed(cfg.train.seed)
    agents = build_trainable_agents(cfg, cfg.train.seed)
    league = LeagueController(cfg.league, seed=cfg.train.seed)
    replay = ReplayBuffer(cfg.league.replay_capacity)
    metrics_history: list[dict] = []

    for epoch in range(cfg.train.num_epochs):
        game_results, games_sec = league.run_epoch(agents, epoch)
        for game in game_results:
            for st in game.steps:
                outcome = 1.0 if st["agent_id"] == game.winner else 0.0
                replay.add(**st, terminal_outcome=outcome)

        train_losses: dict[str, list[float]] = {a.agent_id: [] for a in agents}
        t0 = time.time()
        if len(replay) >= cfg.league.min_replay_size_to_train:
            agents_by_group: dict[str, list] = {}
            for agent in agents:
                agents_by_group.setdefault(agent.group, []).append(agent)

            for group_agents in agents_by_group.values():
                for _ in range(cfg.train.updates_per_epoch_per_agent):
                    # One replay sample and one host->device transfer per architecture group.
                    batch = replay.sample(
                        cfg.train.batch_size,
                        cfg.league.alpha_recency,
                        cfg.league.alpha_uniform,
                        cfg.league.recency_window,
                        cfg.league.max_samples_per_game_in_batch,
                    )
                    x_np = np.stack([b.state_vector for b in batch]).astype(np.float32)
                    y_np = np.array([b.terminal_outcome for b in batch], dtype=np.float32).reshape(-1, 1)

                    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=group_agents[0].device)
                    y_t = torch.as_tensor(y_np, dtype=torch.float32, device=group_agents[0].device)

                    for agent in group_agents:
                        train_losses[agent.agent_id].append(agent.train_batch_tensor(x_t, y_t))
        train_dt = max(time.time() - t0, 1e-6)
        steps_per_sec = (cfg.train.batch_size * cfg.train.updates_per_epoch_per_agent * len(agents)) / train_dt

        per_agent = {a.agent_id: {
            "train_loss_epoch": float(np.mean(train_losses[a.agent_id]) if train_losses[a.agent_id] else 0.0),
            "winrate_vs_random": 0.0,
            "winrate_vs_baseline": 0.0,
            "aggregate_winrate_vs_trainable": 0.0,
            "avg_game_length": float(np.mean([g.turns for g in game_results]) if game_results else 0.0),
            "winrate_vs_opponents": {},
        } for a in agents}

        opponents = [x.agent_id for x in agents] + ["random", "baseline"]
        for a in agents:
            for opp in opponents:
                if opp == a.agent_id:
                    continue
                pair = _games_for_pair(game_results, a.agent_id, opp)
                if not pair:
                    wr = 0.0
                else:
                    wr = sum(1 for g in pair if g.winner == a.agent_id) / len(pair)
                per_agent[a.agent_id]["winrate_vs_opponents"][opp] = wr
            tr_ids = [x.agent_id for x in agents if x.agent_id != a.agent_id]
            per_agent[a.agent_id]["aggregate_winrate_vs_trainable"] = float(np.mean([per_agent[a.agent_id]["winrate_vs_opponents"].get(t, 0.0) for t in tr_ids]))
            per_agent[a.agent_id]["winrate_vs_random"] = per_agent[a.agent_id]["winrate_vs_opponents"].get("random", 0.0)
            per_agent[a.agent_id]["winrate_vs_baseline"] = per_agent[a.agent_id]["winrate_vs_opponents"].get("baseline", 0.0)

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            torch.cuda.reset_peak_memory_stats()

        metrics = {
            "epoch": epoch,
            "games_sec": games_sec,
            "steps_sec": steps_per_sec,
            "gpu_mem_mb": gpu_mem,
            "agents": per_agent,
        }
        metrics_history.append(metrics)

        if epoch % cfg.league.checkpoint_frequency_epochs == 0:
            save_checkpoint(cfg, agents, replay, epoch, metrics)

    _plot(metrics_history, Path(cfg.plots_dir))
    return metrics_history
