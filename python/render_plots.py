from __future__ import annotations

import argparse
from pathlib import Path

from training.config import ExperimentConfig
from training.plotting import load_metrics_history_from_checkpoints, plot_metrics_history


def main() -> int:
    parser = argparse.ArgumentParser(description="Render training plots from saved checkpoints.")
    parser.add_argument("--experiment-dir", default="training_stats", help="Experiment root containing checkpoints/ and plots/")
    parser.add_argument("--end-epoch-exclusive", type=int, default=None, help="Optional end epoch (exclusive). Defaults to latest+1")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    cfg = ExperimentConfig()
    cfg.checkpoint_dir = str(experiment_dir / "checkpoints")
    cfg.plots_dir = str(experiment_dir / "plots")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    epoch_dirs = sorted(
        int(p.name.split("_")[1])
        for p in checkpoint_dir.glob("epoch_*")
        if p.is_dir() and p.name.split("_")[-1].isdigit()
    )
    if not epoch_dirs:
        raise SystemExit(f"No checkpoints found in {checkpoint_dir}")

    end_epoch_exclusive = args.end_epoch_exclusive
    if end_epoch_exclusive is None:
        end_epoch_exclusive = max(epoch_dirs) + 1

    metrics_history = load_metrics_history_from_checkpoints(checkpoint_dir, int(end_epoch_exclusive))
    if not metrics_history:
        raise SystemExit("No metrics were loaded from checkpoints.")

    plot_metrics_history(
        metrics_history,
        Path(cfg.plots_dir),
        cfg.train.winrate_window_size,
        cfg.league.alpha_recency,
        cfg.league.alpha_uniform,
        cfg.league.recency_decay,
        cfg.league.replay_window_epochs,
    )
    print(f"Rendered plots for {len(metrics_history)} epochs into {cfg.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
