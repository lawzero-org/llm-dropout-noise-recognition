"""In-process wandb sweep agent: load the model once, drain the sweep queue.

Standard wandb sweeps reload the model every trial because the CLI ``wandb agent``
spawns a fresh subprocess per run (via the sweep's ``command:``). For fixed-model
sweeps of big models that is infeasible.

This is the **function-based** agent: ``wandb.agent(sweep_id, function=run_one)``
calls ``run_one`` repeatedly in the *same* process, so a model loaded on the first
trial persists and is reused for every trial pulled from the sweep queue. Multiple
agents (one per GPU/node) can attach to the same ``sweep_id`` for dynamic,
resumable, data-parallel execution — each loads the model once for the trials it
pulls.

Create the sweep with the wandb CLI, then launch this agent against its id:

    wandb sweep sweeps/<model>/<key>.yaml          # prints the SWEEP_ID
    spe-sweep --sweep-id <entity/project/SWEEP_ID> \
        --config-name localization

Because the function agent ignores the sweep YAML's ``command:``, every override
the run needs (including fixed dims like ``model`` / ``experiment.sentences_file``)
must be a sweep ``parameter`` (fixed ones as a single ``value:``). Each trial's
``wandb.config`` is translated straight into Hydra overrides.

Scope: fixed model per agent. If a trial changes the model, the agent raises rather
than silently reloading — launch one agent per model for model-varying sweeps.
"""

import argparse
import os
import tempfile
from pathlib import Path

import hydra
import wandb

from spe import experiment_setup, model_utils, wandb_utils
from spe.experiments import classification as exp_classification
from spe.experiments import icl as exp_icl
from spe.experiments import localization as exp_localization

# experiment name -> (per-trial run function, wandb log mode). All run functions
# share the (cfg, model, tokenizer) -> (metrics, results) signature.
EXPERIMENTS = {
    "localization": (exp_localization.run_localization, "localization"),
    "multiclass": (exp_classification.run_classification, "multiclass"),
    "icl": (exp_icl.run, "icl_teaching"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="Sweep id from `wandb sweep`, ideally 'entity/project/id'.",
    )
    parser.add_argument(
        "--config-name",
        default="localization",
        help="Hydra config to compose each trial from.",
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENTS),
        default="localization",
        help="Which experiment to run per trial (selects run fn + log mode).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Max trials this agent runs (default: drain the sweep).",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Hydra overrides applied to every trial (after the sweep params).",
    )
    args = parser.parse_args()
    run_fn, log_mode = EXPERIMENTS[args.experiment]

    # Each agent runs in its own working dir so the fixed-name plot files (and the
    # local wandb/ dir) don't collide when agents share a node — or share the NFS
    # repo cwd across nodes. We symlink the repo's top-level entries into that dir
    # so relative *reads* (src/conf, data/) still resolve, while new *writes*
    # (plot PNGs) land in the isolated dir. Trials within one agent are sequential.
    repo_root = Path.cwd().resolve()
    workdir = Path(tempfile.mkdtemp(prefix="spe_agent_"))
    # Symlink only *directories* (src, data, ...) so relative reads resolve. We
    # must NOT symlink top-level files: a stale plot PNG left in the repo root
    # would then be a symlink, and savefig would write *through* it back to the
    # shared file — reintroducing the cross-agent collision. Plot files are
    # top-level, so leaving files unlinked makes each write land fresh in workdir.
    for entry in repo_root.iterdir():
        if entry.is_dir():
            (workdir / entry.name).symlink_to(entry)
    os.chdir(workdir)

    # (model, tokenizer) loaded on the first trial, reused after. List so the
    # nested run_one can mutate it without `global`.
    loaded: list = [None, None]
    model_key: list = [None]
    runtime: list = [None]

    def run_one() -> None:
        # The function agent does not pre-init the run; we must call wandb.init(),
        # which picks up this trial's params from the sweep into wandb.config.
        # console="off": do not stream the ~1000 printed per-sample rows per run to W&B; with hundreds of
        # concurrent agents that traffic hit the per-project filestream limit (HTTP 429) in the ICL main arm.
        with wandb.init(settings=wandb.Settings(console="off")):
            # `hydra_overrides` is an escape hatch for non-grid sweeps: one string of
            # space-separated Hydra overrides per trial (exact config lists). NB: the
            # key must not start with "_" -- wandb.config.items() hides such keys.
            overrides = []
            for k, v in wandb.config.items():
                overrides += str(v).split() if k == "hydra_overrides" else [f"{k}={v}"]
            overrides += list(args.extra)
            cfg = hydra.compose(config_name=args.config_name, overrides=overrides)

            experiment_setup.seed_everything(cfg)

            key = (cfg.model.name, cfg.model.get("revision"))
            if loaded[0] is None:
                loaded[0], loaded[1] = model_utils.load_model_and_tokenizer(
                    model_name=cfg.model.name,
                    dtype=cfg.model.dtype,
                    device_map=cfg.model.device_map,
                    revision=cfg.model.get("revision"),
                    auto_class=cfg.model.get("auto_class", "causal_lm"),
                )
                model_key[0] = key
                runtime[0] = model_utils.runtime_info(
                    cfg.model.name, cfg.model.get("revision")
                )
            elif key != model_key[0]:
                raise ValueError(
                    f"fixed-model agent: model changed {model_key[0]} -> {key}. "
                    "Launch one agent per model for model-varying sweeps."
                )

            # Enrich the run with the resolved nested config + tags, then log.
            wandb_utils.annotate_current_run(cfg)
            wandb_utils.log_runtime_info(runtime[0])
            metrics, results = run_fn(cfg, loaded[0], loaded[1])
            wandb_utils.log_results(metrics, results, mode=log_mode, cfg=cfg)
        # `with wandb.init()` finishes the run on exit.

    # Initialize Hydra once (package-based, so it survives the chdir above);
    # run_one composes inside this context per trial.
    with hydra.initialize_config_module(config_module="spe.conf", version_base=None):
        wandb.agent(args.sweep_id, function=run_one, count=args.count)


if __name__ == "__main__":
    main()
