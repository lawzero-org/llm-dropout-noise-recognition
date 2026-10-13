"""Exercise result logging with real plot files and no W&B network calls."""

import random
from pathlib import Path

import pytest
import wandb

from spe import wandb_utils
from spe.experiments import classification, icl, localization

from .test_classification_e2e import _cfg as classification_cfg
from .test_icl_e2e import _cfg as icl_cfg
from .test_localization_e2e import _cfg as localization_cfg


@pytest.mark.parametrize(
    ("mode", "num_classes"),
    [
        ("localization", 2),
        ("multiclass", 2),
        ("icl_teaching", 2),
    ],
)
def test_plots_and_full_results_are_logged(
    mode,
    num_classes,
    mock_generate,
    tmp_output_dir,
    tokenizer,
    fake_model,
    sentences_file,
    monkeypatch,
):
    random.seed(42)
    if mode == "localization":
        cfg = localization_cfg(sentences_file, num_samples=6)
        run = localization.run_localization
    elif mode == "multiclass":
        cfg = classification_cfg(num_samples=6, sentences_file=sentences_file)
        run = classification.run_classification
    else:
        cfg = icl_cfg(num_samples=6, sentences_file=sentences_file)
        run = icl.run
    cfg.model.name = "test-model"

    metrics, results = run(cfg, fake_model, tokenizer)
    # The temporary working directory cannot supply stale plots from another run.
    missing = [path for path in metrics["_plot_files"] if not Path(path).is_file()]
    assert not missing, f"Declared plots were not generated: {missing}"

    logged = []
    monkeypatch.setattr(wandb_utils, "_enabled", True)
    monkeypatch.setattr(wandb, "summary", {})
    monkeypatch.setattr(wandb, "log", logged.append)
    # Keep wandb.Image real: it must successfully read every declared image.
    wandb_utils.log_results(metrics, results, mode, cfg)
    uploads = {key: value for payload in logged for key, value in payload.items()}
    assert all(Path(path).stem in uploads for path in metrics["_plot_files"])
    assert len(uploads["results_table_full"].data) == len(results)
