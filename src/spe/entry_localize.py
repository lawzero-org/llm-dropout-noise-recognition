"""Entry point for localization (paper Sec. 4)."""

import hydra
import omegaconf

from spe import experiment_setup
from spe.experiments import localization as exp_localization


@hydra.main(version_base=None, config_path="conf", config_name="localization")
def main(cfg: omegaconf.DictConfig) -> dict:  # type: ignore[type-arg]
    """Run the localization (N way forced choice) experiment."""
    model, tokenizer = experiment_setup.setup_experiment(cfg)
    metrics, results = exp_localization.run_localization(cfg, model, tokenizer)
    experiment_setup.teardown_experiment(metrics, results, mode="localization", cfg=cfg)
    return metrics


if __name__ == "__main__":
    main()
