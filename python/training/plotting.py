from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

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


def _sampling_probability_series(
    xs: list[int],
    replay_sizes: list[int],
    alpha_recency: float,
    alpha_uniform: float,
    recency_decay: float,
    replay_window_epochs: int,
) -> list[float]:
    if not xs:
        return []

    additions: list[int] = []
    prev_size = 0
    for size in replay_sizes:
        cur = max(int(size), 0)
        additions.append(max(cur - prev_size, 0))
        prev_size = cur

    window = min(max(int(replay_window_epochs), 1), len(xs))
    start = len(xs) - window
    total_states = max(int(sum(additions[start:])), 1)

    tail_weights = []
    for i in range(window):
        ratio_from_last = (float(window - 1 - i) / float(max(window, 1)))
        exponent = ratio_from_last * float(replay_window_epochs)
        weight = float(alpha_uniform) + float(alpha_recency) * (float(recency_decay) ** exponent)
        tail_weights.append(max(weight, 0.0))
    weight_sum = float(sum(tail_weights))
    if weight_sum <= 0.0:
        tail_probs = [1.0 / float(total_states)] * window
    else:
        tail_probs = [w / weight_sum for w in tail_weights]

    probs = [float("nan")] * len(xs)
    for i in range(window):
        probs[start + i] = float(tail_probs[i])
    return probs


def _plot_sampling_probability_overlay(plt_module, xs: list[int], sampling_probs: list[float]) -> None:
    if not xs or not sampling_probs:
        return
    ax = plt_module.gca()
    ax_prob = ax.twinx()
    ax_prob.plot(
        xs,
        sampling_probs,
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.7,
        label="_nolegend_",
    )
    ax_prob.set_ylabel("sample probability")
    ax_prob.set_ylim(bottom=0.0)
    plt_module.sca(ax)



def _apply_percent_y_grid(ax) -> None:
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0.0, 101.0, 10.0))
    ax.set_yticks(np.arange(0.0, 101.0, 5.0), minor=True)
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.7, alpha=0.35)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.25)



def _apply_loss_y_grid(ax) -> None:
    ax.set_ylim(0.0, 4.0)
    ax.set_yticks(np.arange(0.0, 4.01, 0.4))
    ax.set_yticks(np.arange(0.0, 4.01, 0.2), minor=True)
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.7, alpha=0.35)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.25)


def _rolling_percentiles(values: list[float], window_size: int, percentiles: tuple[float, ...]) -> dict[float, list[float]]:
    if not values:
        return {p: [] for p in percentiles}
    w = max(int(window_size), 1)
    arr = np.asarray(values, dtype=np.float64)
    out: dict[float, list[float]] = {p: [] for p in percentiles}
    for i in range(len(arr)):
        left = max(0, i - w + 1)
        chunk = arr[left:i + 1]
        for p in percentiles:
            out[p].append(float(np.percentile(chunk, p)))
    return out


def _loss_plot_name_from_metric_key(metric_key: str) -> str:
    if metric_key == "train_loss_steps_epoch":
        return "total"
    name = metric_key
    if name.endswith("_steps_epoch"):
        name = name[: -len("_steps_epoch")]
    if name.endswith("_epoch"):
        name = name[: -len("_epoch")]
    if name.startswith("train_"):
        name = name[len("train_"):]
    return name



def _format_sci_short(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent):+d}"



def _annotate_last_value(ax, xs: list[int], ys: list[float]) -> None:
    if not xs or not ys:
        return
    x_last = xs[-1]
    y_last = ys[-1]
    if not np.isfinite(y_last):
        return
    ax.annotate(
        _format_sci_short(float(y_last)),
        xy=(x_last, y_last),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
    )



def _print_plot_timing(stage: str, started_at: float, total_started_at: float) -> None:
    stage_dt = max(time.perf_counter() - started_at, 0.0)
    total_dt = max(time.perf_counter() - total_started_at, 0.0)
    print(f"[plot] {stage} took {stage_dt:.2f}s (total={total_dt:.2f}s)")


def load_metrics_history_from_checkpoints(checkpoint_dir: Path, end_epoch_exclusive: int) -> list[dict]:
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

def plot_metrics_history(
    metrics_history: list[dict],
    out_dir: Path,
    winrate_window_size: int,
    value_window_size: int,
    matchmaking_window_size: int,
    alpha_recency: float,
    alpha_uniform: float,
    recency_decay: float,
    replay_window_epochs: int,
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
    winrates_windowed_dir = out_dir / "winrates_windowed"
    value_windowed_dir = out_dir / "value_windowed"
    loss_dir = out_dir / "loss"
    loss_total_dir = loss_dir / "total"
    lr_dir = out_dir / "lr"
    decision_dir = out_dir / "decision_temperature"
    replay_dir = out_dir / "replay"
    games_stats_dir = out_dir / "games_stats"
    matchmaking_dir = out_dir / "matchmaking"
    step_t0 = time.perf_counter()
    winrates_windowed_dir.mkdir(parents=True, exist_ok=True)
    value_windowed_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)
    loss_total_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    games_stats_dir.mkdir(parents=True, exist_ok=True)
    matchmaking_dir.mkdir(parents=True, exist_ok=True)
    _print_plot_timing("prepare output directories", step_t0, plot_total_t0)

    agents = sorted(metrics_history[-1]["agents"].keys())


    xs = [m["epoch"] for m in metrics_history]
    replay_sizes = [int(m.get("replay_size", 0)) for m in metrics_history]
    sampling_probs = _sampling_probability_series(
        xs,
        replay_sizes,
        alpha_recency,
        alpha_uniform,
        recency_decay,
        replay_window_epochs,
    )

    step_t0 = time.perf_counter()
    if replay_sizes:
        plt.figure(figsize=(18, 9))
        plt.plot(xs, replay_sizes)
        plt.title("Replay size")
        plt.xlabel("epoch")
        plt.ylabel("samples")
        plt.tight_layout()
        plt.savefig(replay_dir / "replay_size.png")
        plt.close()

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

    choose_best_probs = [float(m.get("choose_best_probability", np.nan)) for m in metrics_history]
    if any(np.isfinite(v) for v in choose_best_probs):
        plt.figure(figsize=(18, 9))
        plt.plot(xs, choose_best_probs)
        plt.title("Choose-best probability")
        plt.xlabel("epoch")
        plt.ylabel("probability")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(decision_dir / "choose_best_probability.png")
        plt.close()

    baseline_copy_probs = [float(m.get("conservative_baseline_double_copy_prob", np.nan)) for m in metrics_history]
    agents_double_probs = [float(m.get("agents_double_decision_prob", np.nan)) for m in metrics_history]
    if any(np.isfinite(v) for v in baseline_copy_probs) or any(np.isfinite(v) for v in agents_double_probs):
        plt.figure(figsize=(18, 9))
        if any(np.isfinite(v) for v in baseline_copy_probs):
            plt.plot(xs, baseline_copy_probs, label="conservative_baseline_double_copy_prob")
        if any(np.isfinite(v) for v in agents_double_probs):
            plt.plot(xs, agents_double_probs, label="agents_double_decision_prob")
        plt.title("Double decision probabilities")
        plt.xlabel("epoch")
        plt.ylabel("probability")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(decision_dir / "double_decision_probabilities.png")
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
    reward_labels = ["1", "2", "3"]
    reward_colors = ["#66a61e", "#1b9e77", "#1f78b4"]
    signed_reward_probs_series = [np.asarray(m.get("games_stats", {}).get("signed_reward_probs", [np.nan] * 6), dtype=np.float64) for m in metrics_history]
    if signed_reward_probs_series:
        arr6 = np.vstack(signed_reward_probs_series)
        if np.any(np.isfinite(arr6)):
            arr = np.column_stack([arr6[:, 2] + arr6[:, 3], arr6[:, 1] + arr6[:, 4], arr6[:, 0] + arr6[:, 5]])
            plt.figure(figsize=(18, 9))
            plt.stackplot(xs, [arr[:, i] * 100.0 for i in range(3)], labels=reward_labels, colors=reward_colors, alpha=0.9)
            plt.title("Reward distribution")
            plt.xlabel("epoch")
            plt.ylabel("probability (%)")
            ax = plt.gca()
            _apply_percent_y_grid(ax)
            last = arr[-1]
            if np.all(np.isfinite(last)):
                cumulative = np.cumsum(last * 100.0)
                lower = np.concatenate(([0.0], cumulative[:-1]))
                mids = (lower + cumulative) / 2.0
                for lbl, val, ymid in zip(reward_labels, last, mids):
                    if val > 0.0:
                        ax.annotate(f"{lbl}: {val * 100.0:.1f}%", (xs[-1], ymid), textcoords="offset points", xytext=(0, 0), ha="center", va="center", fontsize=9, color="white")
            plt.legend(loc="upper right", ncol=3, fontsize=8)
            plt.tight_layout()
            plt.savefig(games_stats_dir / "reward_distribution.png")
            plt.close()

    ended_natural = [float(m.get("games_stats", {}).get("ended_natural_freq", np.nan)) for m in metrics_history]
    if any(np.isfinite(v) for v in ended_natural):
        plt.figure(figsize=(18, 9))
        plt.plot(xs, ended_natural)
        plt.title("Frequency of naturally finished games")
        plt.xlabel("epoch")
        plt.ylabel("frequency")
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(games_stats_dir / "natural_finish_frequency.png")
        plt.close()

    avg_steps = [float(m.get("games_stats", {}).get("avg_steps_per_game", np.nan)) for m in metrics_history]
    min_steps = [float(m.get("games_stats", {}).get("min_steps_per_game", np.nan)) for m in metrics_history]
    max_steps = [float(m.get("games_stats", {}).get("max_steps_per_game", np.nan)) for m in metrics_history]
    if any(np.isfinite(v) for v in avg_steps):
        plt.figure(figsize=(18, 9))
        avg_arr = np.asarray(avg_steps, dtype=np.float64)
        min_arr = np.asarray(min_steps, dtype=np.float64)
        max_arr = np.asarray(max_steps, dtype=np.float64)
        lower = np.where(np.isfinite(min_arr), np.maximum(avg_arr - min_arr, 0.0), np.nan)
        upper = np.where(np.isfinite(max_arr), np.maximum(max_arr - avg_arr, 0.0), np.nan)

        plt.errorbar(xs, avg_steps, yerr=[lower, upper], fmt="-o", capsize=4, linewidth=1.5, markersize=4, label="avg [min..max]")

        for x, avg_v, min_v, max_v in zip(xs, avg_arr, min_arr, max_arr):
            if not np.isfinite(avg_v):
                continue
            plt.annotate(f"{avg_v:.1f}", (x, avg_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
            if np.isfinite(min_v):
                plt.annotate(f"min {min_v:.0f}", (x, min_v), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6, color="#666666")
            if np.isfinite(max_v):
                plt.annotate(f"max {max_v:.0f}", (x, max_v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color="#666666")

        plt.title("Average steps per game (with min/max whiskers)")
        plt.xlabel("epoch")
        plt.ylabel("steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(games_stats_dir / "avg_steps_per_game.png")
        plt.close()

    agent_ids = sorted(metrics_history[-1].get("games_stats", {}).get("offers_per_game_by_agent", {}).keys())
    if agent_ids:
        plt.figure(figsize=(18, 9))
        for aid in agent_ids:
            series = [float(m.get("games_stats", {}).get("offers_per_game_by_agent", {}).get(aid, np.nan)) for m in metrics_history]
            plt.plot(xs, series, alpha=0.55, linewidth=1.2, label=aid)
        mean_series = [float(m.get("games_stats", {}).get("offers_per_game_mean", np.nan)) for m in metrics_history]
        plt.plot(xs, mean_series, color="black", linewidth=2.8, label="mean")
        plt.title("Double offers per game")
        plt.xlabel("epoch")
        plt.ylabel("offers / game")
        plt.legend(fontsize=6, ncol=3)
        plt.tight_layout()
        plt.savefig(games_stats_dir / "double_offers_per_game_by_agent.png")
        plt.close()

        plt.figure(figsize=(18, 9))
        for aid in agent_ids:
            series = [float(m.get("games_stats", {}).get("accept_prob_by_agent", {}).get(aid, np.nan)) for m in metrics_history]
            plt.plot(xs, series, alpha=0.55, linewidth=1.2, label=aid)
        mean_series = [float(m.get("games_stats", {}).get("accept_prob_mean", np.nan)) for m in metrics_history]
        plt.plot(xs, mean_series, color="black", linewidth=2.8, label="mean")
        plt.title("Double acceptance probability")
        plt.xlabel("epoch")
        plt.ylabel("probability")
        plt.ylim(0.0, 1.0)
        plt.legend(fontsize=6, ncol=3)
        plt.tight_layout()
        plt.savefig(games_stats_dir / "double_accept_prob_by_agent.png")
        plt.close()
    _print_plot_timing("games-stats plots", step_t0, plot_total_t0)

    step_t0 = time.perf_counter()
    fixed_opponents = {"conservative_baseline"}
    for aid in agents:
        agent_t0 = time.perf_counter()
        latest_opponents = set(metrics_history[-1]["agents"][aid].get("winrate_vs_opponents", {}).keys())
        opponents_for_agent = sorted((latest_opponents | fixed_opponents) - {aid})
        xs = [m["epoch"] for m in metrics_history]

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
                plt.plot(xs, windowed_series_by_opp[opp], label=opp, linewidth=lw, alpha=alpha)
            plt.title(f"{aid} winrates windowed (focus: {focus_opp}, w={max(int(winrate_window_size), 1)})")
            plt.xlabel("epoch")
            plt.ylabel("windowed winrate (%)")
            ax = plt.gca()
            _apply_percent_y_grid(ax)
            _plot_sampling_probability_overlay(plt, xs, sampling_probs)
            plt.legend(fontsize=6, ncol=2)
            plt.tight_layout()
            plt.savefig(winrates_windowed_dir / f"{aid}_winrates_windowed_focus_{focus_opp}.png")
            plt.close()

        _print_plot_timing(f"{aid}: winrate plots", agent_t0, plot_total_t0)
        agent_t0 = time.perf_counter()

        value_series_by_opp = {
            opp: [float(m["agents"][aid].get("average_value_vs_opponents", {}).get(opp, 0.0)) for m in metrics_history]
            for opp in opponents_for_agent
        }
        windowed_value_series_by_opp = {
            opp: _uniform_windowed(value_series_by_opp[opp], value_window_size)
            for opp in opponents_for_agent
        }

        for focus_opp in opponents_for_agent:
            plt.figure(figsize=(18, 9))
            for opp in opponents_for_agent:
                lw = 2.4 if opp == focus_opp else 1.0
                alpha = 1.0 if opp == focus_opp else 0.4
                plt.plot(xs, windowed_value_series_by_opp[opp], label=opp, linewidth=lw, alpha=alpha)
            plt.title(f"{aid} average value windowed (focus: {focus_opp}, w={max(int(value_window_size), 1)})")
            plt.xlabel("epoch")
            plt.ylabel("windowed avg value / game")
            plt.axhline(y=0.0, linestyle="--", color="gray", linewidth=0.9, alpha=0.8)
            _plot_sampling_probability_overlay(plt, xs, sampling_probs)
            plt.legend(fontsize=6, ncol=2)
            plt.tight_layout()
            plt.savefig(value_windowed_dir / f"{aid}_value_windowed_focus_{focus_opp}.png")
            plt.close()

        _print_plot_timing(f"{aid}: value plots", agent_t0, plot_total_t0)
        agent_t0 = time.perf_counter()

        latest_matchmaking_opponents = set(metrics_history[-1]["agents"][aid].get("matches_vs_opponents", {}).keys())
        matchmaking_opponents = sorted(latest_matchmaking_opponents | {aid, "conservative_baseline"})
        matchmaking_series_by_opp = {
            opp: [float(m["agents"][aid].get("matches_vs_opponents", {}).get(opp, 0.0)) for m in metrics_history]
            for opp in matchmaking_opponents
        }
        windowed_matchmaking_series = {
            opp: _uniform_windowed(matchmaking_series_by_opp[opp], matchmaking_window_size)
            for opp in matchmaking_opponents
        }

        for focus_opp in matchmaking_opponents:
            plt.figure(figsize=(18, 9))
            for opp in matchmaking_opponents:
                lw = 2.4 if opp == focus_opp else 1.0
                alpha = 1.0 if opp == focus_opp else 0.4
                plt.plot(xs, windowed_matchmaking_series[opp], label=opp, linewidth=lw, alpha=alpha)
            plt.title(f"{aid} matchmaking windowed (focus: {focus_opp}, w={max(int(matchmaking_window_size), 1)})")
            plt.xlabel("epoch")
            plt.ylabel("windowed matches per epoch")
            _plot_sampling_probability_overlay(plt, xs, sampling_probs)
            plt.legend(fontsize=6, ncol=2)
            plt.tight_layout()
            plt.savefig(matchmaking_dir / f"{aid}_matchmaking_windowed_focus_{focus_opp}.png")
            plt.close()

        _print_plot_timing(f"{aid}: matchmaking plots", agent_t0, plot_total_t0)
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
        loss_step_metric_keys = sorted({
            k
            for m in metrics_history
            for k, v in m["agents"][aid].items()
            if ("loss" in k.lower()) and k.endswith("_steps_epoch") and isinstance(v, list)
        })
        if "train_loss_steps_epoch" in loss_step_metric_keys:
            loss_step_metric_keys = ["train_loss_steps_epoch"] + [k for k in loss_step_metric_keys if k != "train_loss_steps_epoch"]

        for loss_metric_key in loss_step_metric_keys:
            loss_steps: list[float] = []
            loss_epoch_end_steps: list[int] = []
            loss_cursor = 0
            for m in metrics_history:
                epoch_loss_steps = m["agents"][aid].get(loss_metric_key, [])
                if epoch_loss_steps:
                    sanitized_epoch_loss = [float(v) for v in epoch_loss_steps]
                    loss_steps.extend(sanitized_epoch_loss)
                loss_cursor += len(epoch_loss_steps)
                loss_epoch_end_steps.append(loss_cursor)

            if not loss_steps:
                continue

            loss_name = _loss_plot_name_from_metric_key(loss_metric_key)
            loss_subdir = loss_dir / loss_name
            loss_subdir.mkdir(parents=True, exist_ok=True)

            loss_xs = list(range(1, len(loss_steps) + 1))
            loss_epoch_boundaries = sorted({x for x in loss_epoch_end_steps[:-1] if 0 < x < len(loss_steps)})
            window_size = max(int(round(np.mean(np.diff([0] + loss_epoch_end_steps)))) if len(loss_epoch_end_steps) > 1 else len(loss_steps), 1)
            percs = _rolling_percentiles(loss_steps, window_size=window_size, percentiles=(50.0, 90.0, 99.0))

            plt.figure(figsize=(18, 9))
            plt.plot(loss_xs, loss_steps, linewidth=1.0, alpha=0.35, label="raw")
            plt.plot(loss_xs, percs[50.0], linewidth=1.8, label=f"p50 (rolling, w={window_size})")
            plt.plot(loss_xs, percs[90.0], linewidth=1.5, label=f"p90 (rolling, w={window_size})")
            plt.plot(loss_xs, percs[99.0], linewidth=1.2, label=f"p99 (rolling, w={window_size})")
            for boundary in loss_epoch_boundaries:
                plt.axvline(x=boundary + 0.5, linestyle=":", color="gray", linewidth=0.8, alpha=0.8)
            plt.title(f"{aid} train {loss_name} loss")
            plt.xlabel("learning step")
            plt.ylabel("loss")
            _apply_loss_y_grid(plt.gca())
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(loss_subdir / f"{aid}_{loss_name}_loss.png")
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
            _annotate_last_value(plt.gca(), lr_xs, lr_steps)
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
            _annotate_last_value(plt.gca(), lr_xs, lr_steps_windowed)
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
            plt.plot(xs, _uniform_windowed(overall_series[aid], winrate_window_size), label=aid, linewidth=1.4)
        plt.title(f"Agents overall average winrate windowed (w={max(int(winrate_window_size), 1)})")
        plt.xlabel("epoch")
        plt.ylabel("windowed winrate (%)")
        ax = plt.gca()
        _apply_percent_y_grid(ax)
        _plot_sampling_probability_overlay(plt, xs, sampling_probs)
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(winrates_windowed_dir / "overall_avg_winrates_windowed.png")
        plt.close()

    overall_value_series: dict[str, list[float]] = {}
    for aid in agents:
        vals = []
        for m in metrics_history:
            opp_vals = list(m["agents"][aid].get("average_value_vs_opponents", {}).values())
            vals.append(float(np.mean(opp_vals)) if opp_vals else 0.0)
        overall_value_series[aid] = vals

    if overall_value_series:
        plt.figure(figsize=(18, 9))
        for aid in agents:
            plt.plot(xs, _uniform_windowed(overall_value_series[aid], value_window_size), label=aid, linewidth=1.4)
        plt.title(f"Agents overall average value windowed (w={max(int(value_window_size), 1)})")
        plt.xlabel("epoch")
        plt.ylabel("windowed avg value / game")
        plt.axhline(y=0.0, linestyle="--", color="gray", linewidth=0.9, alpha=0.8)
        _plot_sampling_probability_overlay(plt, xs, sampling_probs)
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(value_windowed_dir / "overall_avg_value_windowed.png")
        plt.close()

    _print_plot_timing("overall average winrate/value plots", step_t0, plot_total_t0)

    step_t0 = time.perf_counter()
    total_loss_by_agent: dict[str, list[float]] = {}
    for aid in agents:
        vals = [float(m["agents"][aid].get("train_loss_epoch", np.nan)) for m in metrics_history]
        total_loss_by_agent[aid] = vals

    if total_loss_by_agent:
        total_arr = np.asarray([total_loss_by_agent[aid] for aid in agents], dtype=np.float64)
        mean_total = np.nanmean(total_arr, axis=0)
        p50_total = np.nanpercentile(total_arr, 50, axis=0)
        p90_total = np.nanpercentile(total_arr, 90, axis=0)
        p99_total = np.nanpercentile(total_arr, 99, axis=0)

        plt.figure(figsize=(18, 9))
        for aid in agents:
            plt.plot(xs, total_loss_by_agent[aid], alpha=0.18, linewidth=0.9, label="_nolegend_")
        plt.plot(xs, mean_total, linewidth=2.2, color="black", label="mean")
        plt.plot(xs, p50_total, linewidth=1.8, label="p50 across agents")
        plt.plot(xs, p90_total, linewidth=1.5, label="p90 across agents")
        plt.plot(xs, p99_total, linewidth=1.2, label="p99 across agents")
        plt.title("Overall total train loss (epoch)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        _apply_loss_y_grid(plt.gca())
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(loss_total_dir / "overall_total_loss_epoch.png")
        plt.close()
    _print_plot_timing("overall total-loss plots", step_t0, plot_total_t0)
    _print_plot_timing("full plotting stage", plot_total_t0, plot_total_t0)
