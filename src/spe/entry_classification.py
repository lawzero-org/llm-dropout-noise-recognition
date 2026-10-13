"""Entry point for zero-shot classification (paper Sec. 5)."""

import hydra
import omegaconf

from spe import experiment_setup
from spe.experiments import classification as exp_classification


@hydra.main(version_base=None, config_path="conf", config_name="classification")
def main(cfg: omegaconf.DictConfig) -> dict:  # type: ignore[type-arg]
    """Run the zero-shot (dropout vs noise) classification experiment."""
    model, tokenizer = experiment_setup.setup_experiment(cfg)
    metrics, results = exp_classification.run_classification(cfg, model, tokenizer)
    experiment_setup.teardown_experiment(metrics, results, mode="multiclass", cfg=cfg)
    return metrics


if __name__ == "__main__":
    main()
