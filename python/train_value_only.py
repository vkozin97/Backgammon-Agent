from training.config import ExperimentConfig
from training.pipeline import run_training


if __name__ == "__main__":
    cfg = ExperimentConfig()
    run_training(cfg)
    print("training finished")
