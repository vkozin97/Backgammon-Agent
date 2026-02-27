from pathlib import Path

from training.config import ExperimentConfig
from training.pipeline import run_training


if __name__ == "__main__":
    experiment_dir = Path("training_stats/default_experiment")
    epoch: int | None = None

    cfg = ExperimentConfig()
    cfg.checkpoint_dir = str(experiment_dir / "checkpoints")
    cfg.plots_dir = str(experiment_dir / "plots")
    cfg.league.replay_storage_dir = str(experiment_dir / "replay")

    start_epoch = 0 if epoch in (None, 0) else int(epoch)
    run_training(cfg, start_epoch=start_epoch)
    print("training finished")
