from pathlib import Path

from training.config import ExperimentConfig
from training.pipeline import run_training


if __name__ == "__main__":
    experiment_dir = "training_stats"
    experiment_dir_path = Path(experiment_dir)
    epoch = 307
    calculate_learning_params = True

    cfg = ExperimentConfig()
    cfg.checkpoint_dir = str(experiment_dir_path / "checkpoints")
    cfg.plots_dir = str(experiment_dir_path / "plots")
    cfg.league.replay_storage_dir = str(experiment_dir_path / "replay")

    start_epoch = 0 if epoch in (None, 0) else int(epoch)
    run_training(cfg, start_epoch=start_epoch, calculate_learning_params=calculate_learning_params)
    print("training finished")
