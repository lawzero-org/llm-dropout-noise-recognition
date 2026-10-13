"""Shared experiment setup and teardown used by all entry points."""

import hashlib
import json
import random
import typing

import numpy as np
import omegaconf
import torch
import transformers

from spe import model_utils, wandb_utils


def resolve_seed(cfg: omegaconf.DictConfig) -> int:
    """Return ``cfg.experiment.seed``; when it is ``"auto"``, derive it from the config.

    The auto seed is a 32 bit sha256 of the resolved config without the run
    bookkeeping (``wandb``, ``output``, ``model.device_map``) and the seed itself,
    so every distinct experiment (model, prompts, perturbation, strength, ...) gets
    its own reproducible seed, and the same experiment always gets the same one.
    """
    seed = cfg.experiment.seed
    if seed != "auto":
        return int(seed)
    payload = typing.cast(dict, omegaconf.OmegaConf.to_container(cfg, resolve=True))
    del payload["wandb"], payload["output"]
    del payload["experiment"]["seed"], payload["model"]["device_map"]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return int(digest[:8], 16)


def seed_everything(cfg: omegaconf.DictConfig) -> None:
    """Seed Python / NumPy / Torch global RNGs from ``cfg.experiment.seed``.

    Resolves an ``"auto"`` seed first and writes the integer back into the config
    so it is logged to W&B and reused by the runners. Kept separate from model
    loading so the sweep agent can reseed before every trial while loading the
    model only once.
    """
    seed = resolve_seed(cfg)
    cfg.experiment.seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def setup_experiment(
    cfg: omegaconf.DictConfig,
) -> tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    """Print config, set seeds, init wandb and load model.

    Args:
        cfg: Resolved Hydra config.

    Returns:
        Tuple of (model, tokenizer).
    """
    seed_everything(cfg)  # before the print so the resolved seed is visible in the log

    print(omegaconf.OmegaConf.to_yaml(cfg))

    wandb_utils.init_run(cfg)

    model_utils.print_hardware_info()

    model, tokenizer = model_utils.load_model_and_tokenizer(
        model_name=cfg.model.name,
        dtype=cfg.model.dtype,
        device_map=cfg.model.device_map,
        revision=cfg.model.get("revision"),
        auto_class=cfg.model.get("auto_class", "causal_lm"),
    )
    info = model_utils.runtime_info(cfg.model.name, cfg.model.get("revision"))
    print(f"Runtime: {info}")
    wandb_utils.log_runtime_info(info)

    return model, tokenizer


def teardown_experiment(
    metrics: dict[str, typing.Any],
    results: list[dict[str, typing.Any]],
    mode: str,
    cfg: omegaconf.DictConfig,
) -> None:
    """Log results to wandb and finish the run.

    Args:
        metrics: Metrics dict returned by the evaluation module.
        results: List of result dicts (one for each sample).
        mode: Experiment mode string for wandb grouping.
        cfg: Resolved Hydra config.
    """
    wandb_utils.log_results(metrics, results, mode=mode, cfg=cfg)
    wandb_utils.finish_run()

    print(f"\nExperiment complete. Mode: {mode}")
