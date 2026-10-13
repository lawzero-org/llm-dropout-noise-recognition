"""Entry point for few-shot classification (paper Sec. 6)."""

import hydra
import omegaconf

from spe import experiment_setup
from spe.experiments import icl as exp_icl


@hydra.main(version_base=None, config_path="conf", config_name="icl")
def main(cfg: omegaconf.DictConfig) -> dict:  # type: ignore[type-arg]
    """Run the ICL teaching experiment."""
    model, tokenizer = experiment_setup.setup_experiment(cfg)
    metrics, results = exp_icl.run(cfg, model, tokenizer)
    experiment_setup.teardown_experiment(metrics, results, mode="icl_teaching", cfg=cfg)
    return metrics


if __name__ == "__main__":
    main()
