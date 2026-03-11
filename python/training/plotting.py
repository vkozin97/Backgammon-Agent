from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .replay import build_recency_weights


SAMPLING_PROBABILITY_TICKS = 1000


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
    recency_center_mass_ratio: float,
    ticks: int = SAMPLING_PROBABILITY_TICKS,
) -> list[float]:
    if not xs:
        return []

    point_count = len(xs)
    if point_count == 1:
        normalized_positions = [0.0]
    else:
        normalized_positions = [i / float(point_count - 1) for i in range(point_count)]

    probs: list[float] = []
    for norm_pos, replay_size in zip(normalized_positions, replay_sizes):
        full_size = max(int(replay_size), 1)
        size = min(full_size, max(int(ticks), 1))
        uniform_prob = 1.0 / float(size)

        recency_weights = build_recency_weights(size, recency_center_mass_ratio)
        recency_sum = float(np.sum(recency_weights))
        if recency_sum <= 0.0:
            recency_prob = uniform_prob
        else:
            recency_index = int(round(norm_pos * float(size - 1)))
            recency_index = min(max(recency_index, 0), size - 1)
            recency_prob = float(recency_weights[recency_index] / recency_sum)

        probs.append(float(alpha_uniform * uniform_prob + alpha_recency * recency_prob))
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
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_yticks(np.arange(0.0, 1.01, 0.05), minor=True)
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.7, alpha=0.35)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.25)



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
    replay_sizes = [int(m.get("replay_size", 0)) for m in metrics_history]
    sampling_probs = _sampling_probability_series(
        xs,
        replay_sizes,
        alpha_recency,
        alpha_uniform,
        recency_center_mass_ratio,
    )

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
            ax = plt.gca()
            _apply_percent_y_grid(ax)
            _plot_sampling_probability_overlay(plt, xs, sampling_probs)
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
            ax = plt.gca()
            _apply_percent_y_grid(ax)
            _plot_sampling_probability_overlay(plt, xs, sampling_probs)
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
            _apply_loss_y_grid(plt.gca())
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
            plt.plot(xs, overall_series[aid], label=aid, linewidth=1.4)
        plt.title("Agents overall average winrate vs all opponents")
        plt.xlabel("epoch")
        plt.ylabel("winrate (%)")
        ax = plt.gca()
        _apply_percent_y_grid(ax)
        _plot_sampling_probability_overlay(plt, xs, sampling_probs)
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
        ax = plt.gca()
        _apply_percent_y_grid(ax)
        _plot_sampling_probability_overlay(plt, xs, sampling_probs)
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(winrates_windowed_dir / "overall_avg_winrates_windowed.png")
        plt.close()

    _print_plot_timing("overall average winrate plots", step_t0, plot_total_t0)
    _print_plot_timing("full plotting stage", plot_total_t0, plot_total_t0)


