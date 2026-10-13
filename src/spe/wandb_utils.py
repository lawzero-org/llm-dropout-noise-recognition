"""Weights & Biases integration for experiment tracking."""

import math
import typing

import omegaconf
import torch
import wandb

_enabled: bool = False


def _table(columns: typing.Sequence[str]) -> wandb.Table:
    """wandb>=0.30 types ``columns`` as ``list[str | int]``; lists are invariant."""
    keys: list[str | int] = list(columns)
    return wandb.Table(columns=keys)


def _multiclass_table_columns(results: list[dict]) -> list[str]:
    """Build multiclass table columns dynamically from result keys."""
    if not results:
        return []
    return [k for k in results[0] if not k.startswith("v_")]


def _localization_table_columns(results: list[dict]) -> list[str]:
    """Build localization table columns dynamically from result keys.

    Detects labels from ``prob_X`` keys where X is a single uppercase
    letter, so the table adapts to any number of sentences (2, 3, 5, ...).

    Args:
        results: List of result dicts (needs at least one row).

    Returns:
        Ordered list of column names for the clean W&B table.
    """
    base = [
        "sample_id",
        "ground_truth",
        "predicted_aggregate",
        "predicted_primary",
        "same_sentence",
        "target_perturbation",
    ]
    labels = sorted(
        k.split("_")[1]
        for k in results[0]
        if k.startswith("prob_")
        and len(k.split("_")) == 2
        and k.split("_")[1].isupper()
    )
    for lbl in labels:
        base.append(f"prob_{lbl}")
    for lbl in labels:
        base.append(f"logsumexp_{lbl}")
    for lbl in labels:
        base.append(f"primary_prob_{lbl}")
    for lbl in labels:
        base.append(f"primary_logit_{lbl}")
    for lbl in labels:
        base.append(f"primary_log_prob_{lbl}")
    for lbl in labels:
        base.append(f"aggregate_log_prob_{lbl}")
    for lbl in labels:
        col = f"sentence_log_prob_{lbl.lower()}"
        if col in results[0]:
            base.append(col)
    for lbl in labels:
        col = f"n_perturbed_tokens_{lbl.lower()}"
        if col in results[0]:
            base.append(col)
    if "sentence_n_tokens" in results[0]:
        base.append("sentence_n_tokens")
    base.extend(
        [
            "primary_log_prob_correct",
            "aggregate_log_prob_correct",
            "logit_diff",
            "logsumexp_diff",
            "logit_diff_correct_vs_incorrect",
            "logsumexp_diff_correct_vs_incorrect",
        ]
    )
    # Log probs by perturbation type of the labelled sentence (perturbed / "nothing")
    ptype_suffixes = {"dropout", "noise", "nothing"}
    for prefix in ("primary_log_prob", "aggregate_log_prob"):
        for suffix in sorted(ptype_suffixes):
            col = f"{prefix}_{suffix}"
            if col in results[0]:
                base.append(col)
    return base


def _icl_teaching_table_columns(results: list[dict]) -> list[str]:
    """Build ICL teaching table columns dynamically from result keys.

    Detects letter labels from ``sum_prob_X`` keys where X is a single
    uppercase letter.

    Args:
        results: List of result dicts (needs at least one row).

    Returns:
        Ordered list of column names for the clean W&B table.
    """
    base = [
        "sample_id",
        "test_perturbation",
        "is_baseline",
        "option_order",
        "num_teaching_examples",
        "teaching_schedule",
        "argmax_token",
        "argmax_prob",
    ]
    letters = sorted(
        k.split("_")[-1]
        for k in results[0]
        if k.startswith("sum_prob_")
        and len(k.split("_")[-1]) == 1
        and k.split("_")[-1].isupper()
    )
    for letter in letters:
        base.extend(
            [
                f"sum_prob_{letter}",
                f"logsumexp_{letter}",
                f"primary_prob_{letter}",
                f"primary_logit_{letter}",
            ]
        )
    # Detect class level probability columns (e.g. sum_prob_dropout, primary_prob_noise).
    # These are lowercase multi-character suffixes, unlike letter columns (uppercase single char).
    class_prob_keys = sorted(
        k
        for k in results[0]
        if (k.startswith("sum_prob_") or k.startswith("primary_prob_"))
        and k.split("_")[-1].islower()
        and len(k.split("_")[-1]) > 1
    )
    base.extend(class_prob_keys)
    # Detect class level logit diff columns from result keys
    logit_diff_cols = sorted(
        k
        for k in results[0]
        if k.startswith("logit_diff_") or k.startswith("logsumexp_diff_")
    )
    base.extend(logit_diff_cols)
    # Sentence log prob columns only appear on stochastic rows (not
    # baselines), so scan all rows to discover them.
    log_prob_keys: set[str] = set()
    for row in results:
        for k in row:
            if k.startswith("sentence_log_prob_"):
                log_prob_keys.add(k)
    base.extend(sorted(log_prob_keys))

    base.extend(
        [
            "predicted_label_aggregate",
            "predicted_label_primary",
            "predicted_class_aggregate",
            "predicted_class_primary",
            "ground_truth_letter",
        ]
    )
    return base


def _perturbation_type_tags(
    cfg: omegaconf.DictConfig,
    mode: str,
) -> list[str]:
    """Derive perturbation type tags from the config."""
    types: set[str] = set()
    pcfg = cfg.perturbation
    if mode == "localization":
        types.add(pcfg.target_perturbation.upper())
        if pcfg.get("num_bg_dropout", 0) > 0:
            types.add("DROPOUT")
        if pcfg.get("num_bg_noise", 0) > 0:
            types.add("NOISE")
        if pcfg.get("num_bg_nothing", 0) > 0:
            types.add("NOTHING")
    elif mode == "icl_teaching":
        class_names = list(cfg.prompts.turns.class_names)
        types.update(c.upper() for c in class_names)
    elif mode == "multiclass":
        types.update(["DROPOUT", "NOISE"])
    return sorted(t.lower() for t in types)


def _run_metadata(cfg: omegaconf.DictConfig) -> tuple[str, list[str], dict]:
    """Compute (name, tags, flat_config) for a run from the Hydra config."""
    flat_config = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    # Build a short model name from the full HuggingFace model path.
    # e.g. "Qwen/Qwen3-32B" -> "qwen3_32b"
    model_short = cfg.model.name.split("/")[-1].lower().replace("-", "_")

    mode = cfg.perturbation.mode
    tags = [model_short] + _perturbation_type_tags(cfg, mode)

    default_name = mode
    if mode == "multiclass":
        n_classes = len(cfg.prompts.turns.class_names)
        default_name = f"{mode}_{n_classes}way"

    flat_config["model_name"] = model_short  # type: ignore[index]
    flat_config["prompt_turns"] = cfg.prompts.turns.name  # type: ignore[index]
    flat_config["system_prompt"] = cfg.prompts.system.name  # type: ignore[index]

    name = cfg.wandb.get("name") or default_name
    return name, tags, flat_config  # type: ignore[return-value]


def init_run(cfg: omegaconf.DictConfig) -> None:
    """Initialize a W&B run and log the full Hydra config.

    Args:
        cfg: Resolved Hydra config.
    """
    global _enabled
    _enabled = cfg.wandb.enabled
    if not _enabled:
        return

    name, tags, flat_config = _run_metadata(cfg)
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=name,
        tags=tags,
        config=flat_config,  # type: ignore[arg-type]
    )


def log_runtime_info(info: dict[str, typing.Any]) -> None:
    """Attach ``model_utils.runtime_info`` to the active run's config under ``runtime``."""
    if not _enabled:
        return
    wandb.run.config.update({"runtime": info}, allow_val_change=True)  # type: ignore[union-attr]


def annotate_current_run(cfg: omegaconf.DictConfig) -> None:
    """Enrich the currently active W&B run with name/tags/nested config.

    Used by the sweep agent driver: it calls ``wandb.init()`` (which pulls the
    trial's sweep params into ``wandb.config``), then this adds the resolved
    nested Hydra config, model tags, and run name onto that same run — keeping
    the config layout identical to the per-process path. Must be called while a
    run is active.
    """
    global _enabled
    _enabled = True
    name, tags, flat_config = _run_metadata(cfg)
    run = wandb.run
    run.name = name  # type: ignore[union-attr]
    run.tags = tuple(run.tags or ()) + tuple(tags)  # type: ignore[union-attr]
    run.config.update(flat_config, allow_val_change=True)  # type: ignore[union-attr]


def log_results(
    metrics: dict[str, typing.Any],
    results: list[dict[str, typing.Any]],
    mode: str,
    cfg: omegaconf.DictConfig,
) -> None:
    """Log metrics, plots, and raw results to W&B.

    Args:
        metrics: Metrics dict returned by the evaluation module.
        results: List of result dicts (one for each sample).
        mode: Experiment mode ("multiclass" (zero-shot), "localization", or "icl_teaching").
        cfg: Resolved Hydra config (used for sweep summary table).
    """
    if not _enabled:
        return
    if torch.cuda.is_available():
        metrics["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # Log scalar metrics to wandb.summary using grouped keys so the
    # W&B dashboard organizes them into collapsible sections.
    # Non-finite floats (inf, NaN) and None values are skipped entirely so that
    # runs missing a metric don't appear as "null" lines in parallel
    # coordinates plots.
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            wandb.summary[key] = value
        elif isinstance(value, str):
            wandb.summary[key] = value

    # Log only the plot files generated by this experiment.
    plot_files = metrics.get("_plot_files", [])
    for path in plot_files:
        name = path.rsplit(".", 1)[0]
        wandb.log({name: wandb.Image(path)})

    # Log raw results as W&B Tables.
    if results:
        # Collect columns from ALL rows so stochastic-only keys
        # (e.g. sentence_log_prob_*) are not dropped when
        # results[0] is a baseline row that lacks them.
        seen: dict[str, None] = {}
        for row in results:
            for k in row:
                if k not in seen:
                    seen[k] = None
        all_columns = list(seen)

        # Full table with every column (data recovery if terminal is lost).
        full_table = _table(all_columns)
        for row in results:
            full_table.add_data(*[row.get(c, None) for c in all_columns])
        wandb.log({"results_table_full": full_table})

        # Clean summary table for modes with known column sets.
        if mode == "localization":
            loc_columns = _localization_table_columns(results)
            clean_table = _table(loc_columns)
            for row in results:
                clean_table.add_data(*[row.get(c, None) for c in loc_columns])
            wandb.log({"results_table": clean_table})
        elif mode == "multiclass":
            mc_columns = _multiclass_table_columns(results)
            clean_table = _table(mc_columns)
            for row in results:
                clean_table.add_data(*[row.get(c, None) for c in mc_columns])
            wandb.log({"results_table": clean_table})
        elif mode == "icl_teaching":
            icl_columns = _icl_teaching_table_columns(results)
            clean_table = _table(icl_columns)
            for row in results:
                clean_table.add_data(*[row.get(c, None) for c in icl_columns])
            wandb.log({"results_table": clean_table})
        else:
            wandb.log({"results_table": full_table})

    if mode == "multiclass":
        _log_multiclass_sweep_summary_table(metrics, cfg)
    elif mode == "icl_teaching":
        _log_icl_teaching_sweep_summary_table(metrics, cfg)


def _log_multiclass_sweep_summary_table(
    metrics: dict[str, typing.Any],
    cfg: omegaconf.DictConfig,
) -> None:
    """Log a single row W&B Table with config params and multiclass metrics.

    Columns are built dynamically so that only groups with actual
    samples appear.  Absent groups produce no columns rather than
    misleading zero filled columns.

    Args:
        metrics: Metrics dict from multiclass evaluation.
        cfg: Resolved Hydra config.
    """
    class_names = list(cfg.prompts.turns.class_names)

    # Discover which perturbation groups had samples.
    active_groups = [
        cls for cls in class_names if metrics.get(f"num_{cls.lower()}", 0) > 0
    ]

    # Accumulate columns and values in lockstep.
    columns: list[str] = []
    values: list[typing.Any] = []

    def _add(col: str, val: typing.Any) -> None:
        columns.append(col)
        values.append(val)

    def _add_if_present(col: str) -> None:
        if col in metrics:
            columns.append(col)
            values.append(metrics[col])

    # --- Config (always present) ---
    display_names = cfg.aliases.display_names
    aliases_label = (
        "none" if display_names is None else "/".join(display_names.values())
    )
    system_content = cfg.prompts.system.content
    system_label = system_content[:30] if system_content else "(empty)"

    _add("model", cfg.model.name.split("/")[-1])
    _add("dropout_rate", cfg.perturbation.dropout_rate)
    _add("noise_std", cfg.perturbation.noise_std)
    _add("last_layer", cfg.perturbation.last_layer)
    _add("hook_target", cfg.perturbation.hook_target)
    _add("system_prompt", system_label)
    _add("prompt_turns", cfg.prompts.turns.name)
    _add("aliases", aliases_label)

    # --- Accuracy and classification metrics (always present) ---
    for key in (
        "accuracy_aggregate",
        "accuracy_se_aggregate",
        "accuracy_primary",
        "accuracy_se_primary",
        "macro_precision_class_constrained_aggregate",
        "macro_recall_class_constrained_aggregate",
        "macro_f1_class_constrained_aggregate",
        "macro_precision_class_constrained_primary",
        "macro_recall_class_constrained_primary",
        "macro_f1_class_constrained_primary",
        "roc_auc_constrained_aggregate_macro",
        "roc_auc_constrained_primary_macro",
    ):
        _add(key, metrics.get(key, 0))

    # --- Sample counts ---
    _add("num_baselines", metrics.get("num_baselines", 0))
    for group in active_groups:
        _add(f"num_{group.lower()}", metrics.get(f"num_{group.lower()}", 0))

    # --- Per active group: aggregate prob, logit diff, log prob per class, and log prob correct ---
    for group in active_groups:
        gl = group.lower()
        for cls in class_names:
            cl = cls.lower()
            _add_if_present(f"{gl}_mean_aggregate_prob_{cl}")
            _add_if_present(f"{gl}_mean_logit_diff_{cl}")
            _add_if_present(f"{gl}_mean_primary_log_prob_{cl}")
            _add_if_present(f"{gl}_se_primary_log_prob_{cl}")
            _add_if_present(f"{gl}_mean_aggregate_log_prob_{cl}")
            _add_if_present(f"{gl}_se_aggregate_log_prob_{cl}")
        _add_if_present(f"{gl}_mean_primary_log_prob_correct")
        _add_if_present(f"{gl}_se_primary_log_prob_correct")
        _add_if_present(f"{gl}_mean_aggregate_log_prob_correct")
        _add_if_present(f"{gl}_se_aggregate_log_prob_correct")

    # --- Overall pairwise logit diffs ---
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            x, y = class_names[i].lower(), class_names[j].lower()
            _add_if_present(f"mean_logit_diff_{x}_vs_{y}")

    # --- Per group pairwise logit diffs ---
    for group in active_groups:
        gl = group.lower()
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                x, y = class_names[i].lower(), class_names[j].lower()
                _add_if_present(f"{gl}_mean_logit_diff_{x}_vs_{y}")

    summary_table = _table(columns)
    summary_table.add_data(*values)
    wandb.log({"sweep_summary": summary_table})


def _log_icl_teaching_sweep_summary_table(
    metrics: dict[str, typing.Any],
    cfg: omegaconf.DictConfig,
) -> None:
    """Log a single row W&B Table with config params and ICL teaching metrics.

    Args:
        metrics: Metrics dict from ICL teaching evaluation.
        cfg: Resolved Hydra config.
    """
    class_names = list(cfg.prompts.turns.class_names)

    # For each test condition X, add mean_aggregate_prob and mean_logit_diff for each target class Y.
    per_class_columns: list[str] = []
    per_class_keys: list[str] = []
    for x_cls in class_names:
        x = x_cls.lower()
        for y_cls in class_names:
            y = y_cls.lower()
            sp_key = f"{x}_mean_aggregate_prob_{y}"
            ld_key = f"{x}_mean_logit_diff_{y}"
            per_class_columns.extend([sp_key, ld_key])
            per_class_keys.extend([sp_key, ld_key])

    summary_table = _table(
        [
            "model",
            "dropout_rate",
            "noise_std",
            "last_layer",
            "hook_target",
            "num_pairs",
            "classes",
            "aliases",
            "swap_labels",
            "accuracy_aggregate",
            "accuracy_se_aggregate",
            "accuracy_primary",
            "accuracy_se_primary",
            "macro_precision_class_constrained_aggregate",
            "macro_recall_class_constrained_aggregate",
            "macro_f1_class_constrained_aggregate",
            "macro_precision_class_constrained_primary",
            "macro_recall_class_constrained_primary",
            "macro_f1_class_constrained_primary",
            "roc_auc_constrained_aggregate",
            "roc_auc_constrained_primary",
        ]
        + per_class_columns
    )

    per_class_values = [metrics.get(k, 0) for k in per_class_keys]

    display_names = cfg.aliases.display_names
    aliases_label = (
        "none" if display_names is None else "/".join(display_names.values())
    )

    summary_table.add_data(
        cfg.model.name.split("/")[-1],
        cfg.perturbation.dropout_rate,
        cfg.perturbation.noise_std,
        cfg.perturbation.last_layer,
        cfg.perturbation.hook_target,
        cfg.prompts.turns.num_pairs,
        "_vs_".join(class_names),
        aliases_label,
        cfg.prompts.turns.swap_labels,
        metrics.get("accuracy_aggregate", 0),
        metrics.get("accuracy_se_aggregate", 0),
        metrics.get("accuracy_primary", 0),
        metrics.get("accuracy_se_primary", 0),
        metrics.get("macro_precision_class_constrained_aggregate", 0),
        metrics.get("macro_recall_class_constrained_aggregate", 0),
        metrics.get("macro_f1_class_constrained_aggregate", 0),
        metrics.get("macro_precision_class_constrained_primary", 0),
        metrics.get("macro_recall_class_constrained_primary", 0),
        metrics.get("macro_f1_class_constrained_primary", 0),
        metrics.get("roc_auc_constrained_aggregate", 0),
        metrics.get("roc_auc_constrained_primary", 0),
        *per_class_values,
    )
    wandb.log({"sweep_summary": summary_table})


def finish_run() -> None:
    """Close the W&B run."""
    if not _enabled:
        return
    wandb.finish()
