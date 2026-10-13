"""Zero-shot classification metrics and plots."""

import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve


def _compute_group_stats(
    rows: list[dict[str, typing.Any]],
    class_names: list[str],
    metric_prefix: str,
) -> dict[str, float]:
    """Compute mean, std, and SEM of normalized quantities for each class.

    Only probabilities (sumP and primary P) and logit diffs are safe to
    average across runs. Raw logits and logsumexp values include the
    normalization constant, which varies across runs, so averaging them
    is meaningless.

    Args:
        rows: Result rows to aggregate.
        class_names: Class names (``["DROPOUT", "NOISE"]``).
        metric_prefix: Prefix for the returned metric keys
            (e.g. ``"dropout"`` produces ``"dropout_mean_aggregate_prob_noise"``).

    Returns:
        Dict of mean, std, and sem for sum_prob, primary_prob, logit_diff,
        and logsumexp_diff for each class.
    """
    stats: dict[str, float] = {}
    n = len(rows)
    sqrt_n = math.sqrt(n) if n > 0 else 1.0

    for cls in class_names:
        cls_lower = cls.lower()

        sp_values = [r[f"sum_prob_{cls_lower}"] for r in rows]
        pp_values = [r[f"primary_prob_{cls_lower}"] for r in rows]

        sp_std = float(np.std(sp_values)) if sp_values else 0.0
        pp_std = float(np.std(pp_values)) if pp_values else 0.0

        stats[f"{metric_prefix}_mean_aggregate_prob_{cls_lower}"] = (
            float(np.mean(sp_values)) if sp_values else 0.0
        )
        stats[f"{metric_prefix}_std_aggregate_prob_{cls_lower}"] = sp_std
        stats[f"{metric_prefix}_se_aggregate_prob_{cls_lower}"] = (
            sp_std / sqrt_n if sp_values else 0.0
        )
        stats[f"{metric_prefix}_mean_primary_prob_{cls_lower}"] = (
            float(np.mean(pp_values)) if pp_values else 0.0
        )
        stats[f"{metric_prefix}_std_primary_prob_{cls_lower}"] = pp_std
        stats[f"{metric_prefix}_se_primary_prob_{cls_lower}"] = (
            pp_std / sqrt_n if pp_values else 0.0
        )

        # Log probability metrics (safe to average: normalization constant cancels)
        plp_values = [
            r[f"primary_log_prob_{cls_lower}"]
            for r in rows
            if f"primary_log_prob_{cls_lower}" in r
        ]
        alp_values = [
            r[f"aggregate_log_prob_{cls_lower}"]
            for r in rows
            if f"aggregate_log_prob_{cls_lower}" in r
        ]

        if plp_values:
            plp_std = float(np.std(plp_values))
            stats[f"{metric_prefix}_mean_primary_log_prob_{cls_lower}"] = float(
                np.mean(plp_values)
            )
            stats[f"{metric_prefix}_std_primary_log_prob_{cls_lower}"] = plp_std
            stats[f"{metric_prefix}_se_primary_log_prob_{cls_lower}"] = (
                plp_std / math.sqrt(len(plp_values))
            )

        if alp_values:
            alp_std = float(np.std(alp_values))
            stats[f"{metric_prefix}_mean_aggregate_log_prob_{cls_lower}"] = float(
                np.mean(alp_values)
            )
            stats[f"{metric_prefix}_std_aggregate_log_prob_{cls_lower}"] = alp_std
            stats[f"{metric_prefix}_se_aggregate_log_prob_{cls_lower}"] = (
                alp_std / math.sqrt(len(alp_values))
            )

        # Logit diff metrics (safe to average: normalization constant cancels)
        ld_key = f"logit_diff_{cls_lower}"
        lsd_key = f"logsumexp_diff_{cls_lower}"
        ld_values = [r[ld_key] for r in rows if ld_key in r]
        lsd_values = [r[lsd_key] for r in rows if lsd_key in r]

        if ld_values:
            ld_std = float(np.std(ld_values))
            stats[f"{metric_prefix}_mean_logit_diff_{cls_lower}"] = float(
                np.mean(ld_values)
            )
            stats[f"{metric_prefix}_std_logit_diff_{cls_lower}"] = ld_std
            stats[f"{metric_prefix}_se_logit_diff_{cls_lower}"] = ld_std / math.sqrt(
                len(ld_values)
            )
        else:
            stats[f"{metric_prefix}_mean_logit_diff_{cls_lower}"] = float("nan")
            stats[f"{metric_prefix}_std_logit_diff_{cls_lower}"] = float("nan")
            stats[f"{metric_prefix}_se_logit_diff_{cls_lower}"] = float("nan")

        if lsd_values:
            lsd_std = float(np.std(lsd_values))
            stats[f"{metric_prefix}_mean_aggregate_logit_diff_{cls_lower}"] = float(
                np.mean(lsd_values)
            )
            stats[f"{metric_prefix}_std_aggregate_logit_diff_{cls_lower}"] = lsd_std
            stats[f"{metric_prefix}_se_aggregate_logit_diff_{cls_lower}"] = (
                lsd_std / math.sqrt(len(lsd_values))
            )
        else:
            stats[f"{metric_prefix}_mean_aggregate_logit_diff_{cls_lower}"] = float(
                "nan"
            )
            stats[f"{metric_prefix}_std_aggregate_logit_diff_{cls_lower}"] = float(
                "nan"
            )
            stats[f"{metric_prefix}_se_aggregate_logit_diff_{cls_lower}"] = float("nan")

    return stats


def _compute_cm_metrics(
    y_true: list[str],
    y_pred: list[str],
    label_list: list[str],
    exclude_from_macro: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, typing.Any]], float, float, float]:
    """Compute confusion matrix and derived metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        label_list: Ordered label set.
        exclude_from_macro: Labels to exclude from macro averages (e.g. a class with no samples).

    Returns:
        Tuple of (confusion_matrix, per_class_metrics, macro_precision,
        macro_recall, macro_f1).
    """
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    per_class: dict[str, dict[str, typing.Any]] = {}

    for i, cls in enumerate(label_list):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(label_list)) if j != i)
        fn = sum(cm[i][j] for j in range(len(label_list)) if j != i)
        support = int(sum(cm[i]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cls] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": support,
        }

    excluded = set(exclude_from_macro or [])
    macro_labels = [c for c in label_list if c not in excluded]
    macro_precision = float(np.mean([per_class[c]["precision"] for c in macro_labels]))
    macro_recall = float(np.mean([per_class[c]["recall"] for c in macro_labels]))
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in macro_labels]))

    return cm, per_class, macro_precision, macro_recall, macro_f1


def _print_cm(
    cm: np.ndarray,
    label_list: list[str],
    per_class: dict[str, dict[str, typing.Any]],
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    title: str,
    macro_note: str = "",
    show_metrics: bool = True,
) -> None:
    """Print a confusion matrix and optionally its derived metrics.

    Args:
        cm: Confusion matrix array.
        label_list: Ordered label set.
        per_class: Dict of per class metrics.
        macro_precision: Macro averaged precision.
        macro_recall: Macro averaged recall.
        macro_f1: Macro averaged F1.
        title: Section title.
        macro_note: Optional note appended to the macro average line.
        show_metrics: If False, print only the confusion matrix grid
            and skip the precision/recall/F1 table. Metrics are still
            computed and logged to W&B regardless.
    """
    print(f"\n  {title}:")
    col_width = 12
    margin = 8
    lbl_width = max(len(c) for c in label_list) + 1
    grid_width = col_width * len(label_list)
    print(f"  {'':>{margin + lbl_width}}{'Predictions':^{grid_width}}")
    print(f"  {'':>{margin + lbl_width}}", end="")
    for cls in label_list:
        print(f"{cls:>{col_width}}", end="")
    print()
    mid = len(label_list) // 2
    for i, cls in enumerate(label_list):
        tag = "Targets" if i == mid else ""
        print(f"  {tag:>{margin}}{cls:>{lbl_width}}", end="")
        for j in range(len(label_list)):
            print(f"{cm[i][j]:>{col_width}}", end="")
        print()

    if show_metrics:
        print(
            f"\n  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}"
        )
        print(f"  {'-' * 60}")
        for cls in label_list:
            m = per_class[cls]
            print(
                f"  {cls:<15} {m['precision']:>11.2%} {m['recall']:>11.2%} "
                f"{m['f1']:>11.2%} {m['support']:>9}"
            )
        note = f"  {macro_note}" if macro_note else ""
        print(
            f"\n  {'Macro avg':<15} {macro_precision:>11.2%} {macro_recall:>11.2%} "
            f"{macro_f1:>11.2%}{note}"
        )


def _print_group_stats(
    rows: list[dict[str, typing.Any]],
    stats: dict[str, float],
    class_names: list[str],
    prefix: str,
    title: str,
) -> None:
    """Print probability and logit-diff stats for a group of rows."""
    print(f"\n--- {title} ({len(rows)} rows) ---")
    if not rows:
        return
    for cls in class_names:
        cl = cls.lower()
        pp_mean = stats[f"{prefix}_mean_primary_prob_{cl}"]
        pp_std = stats[f"{prefix}_std_primary_prob_{cl}"]
        sp_mean = stats[f"{prefix}_mean_aggregate_prob_{cl}"]
        sp_std = stats[f"{prefix}_std_aggregate_prob_{cl}"]
        print(
            f"  {cls:<10}  P(primary): {pp_mean:.6f} +/- {pp_std:.6f}    "
            f"sumP({cls}): {sp_mean:.6f} +/- {sp_std:.6f}"
        )
    for cls in class_names:
        cl = cls.lower()
        ld_mean = stats[f"{prefix}_mean_logit_diff_{cl}"]
        ld_std = stats[f"{prefix}_std_logit_diff_{cl}"]
        lsd_mean = stats[f"{prefix}_mean_aggregate_logit_diff_{cl}"]
        lsd_std = stats[f"{prefix}_std_aggregate_logit_diff_{cl}"]
        if not math.isnan(ld_mean):
            print(
                f"  logit_diff({cls}): {ld_mean:+.4f} +/- {ld_std:.4f}    "
                f"logsumexp_diff({cls}): {lsd_mean:+.4f} +/- {lsd_std:.4f}"
            )


def compute_and_plot(
    results: list[dict[str, typing.Any]],
    class_names: list[str],
    labels: list[str],
    dpi: int = 300,
    fmt: str = "png",
) -> dict[str, typing.Any]:
    """Compute zero-shot classification metrics, print a summary, and save plots.

    A paper run applies a single perturbation (DROPOUT only or NOISE only), so
    the group of the other class is empty: its sample count is 0 and its
    recall is not defined (it is excluded from the macro averages).

    Args:
        results: List of result dicts from ``run_classification()``.
        class_names: Ordered class names (e.g. ``["DROPOUT", "NOISE"]``).
        labels: Ordered position labels (e.g. ``["A", "B"]``).
        dpi: Resolution for saved plots.
        fmt: Image format (e.g. ``"png"``).

    Returns:
        Dict of computed metrics.
    """
    # The runner logs no unperturbed baseline rows; num_baselines stays in the
    # metrics (always 0) because the notebooks check total_rows - num_baselines.
    baselines = [r for r in results if r.get("is_baseline", False)]
    stochastic_rows = [r for r in results if not r.get("is_baseline", False)]

    # Classes that have samples (a single one in the paper's runs)
    stochastic_classes = sorted(
        {r["perturbation"] for r in stochastic_rows},
        key=lambda c: class_names.index(c),
    )
    perturbation_classes = list(class_names)

    group_rows = {
        c: [r for r in stochastic_rows if r["perturbation"] == c]
        for c in perturbation_classes
    }

    print("\n" + "=" * 70)
    print("SUMMARY - MULTICLASS CLASSIFICATION")
    print("=" * 70)
    counts_desc = " + ".join(
        f"{len(group_rows[c])} {c.lower()}" for c in perturbation_classes
    )
    print(f"Total rows: {len(results)} ({counts_desc})")

    # --- Per perturbation group stats ---
    all_group_stats: dict[str, dict[str, float]] = {}
    for cls in perturbation_classes:
        prefix = cls.lower()
        stats = _compute_group_stats(group_rows[cls], class_names, prefix)
        all_group_stats[prefix] = stats
        _print_group_stats(group_rows[cls], stats, class_names, prefix, cls)

    # --- Accuracy ---
    y_true_class = [r["perturbation"] for r in stochastic_rows]

    y_pred_class_agg = [r["predicted_class_aggregate"] for r in stochastic_rows]
    y_pred_class_pri = [r["predicted_class_primary"] for r in stochastic_rows]
    correct_agg = sum(
        1
        for gt, pred in zip(y_true_class, y_pred_class_agg, strict=False)
        if gt == pred
    )
    correct_pri = sum(
        1
        for gt, pred in zip(y_true_class, y_pred_class_pri, strict=False)
        if gt == pred
    )
    n_stochastic = len(stochastic_rows)
    accuracy_constrained_aggregate = (
        correct_agg / n_stochastic if n_stochastic > 0 else 0.0
    )
    accuracy_constrained_primary = (
        correct_pri / n_stochastic if n_stochastic > 0 else 0.0
    )

    accuracy_se_constrained_aggregate = (
        math.sqrt(
            accuracy_constrained_aggregate
            * (1 - accuracy_constrained_aggregate)
            / n_stochastic
        )
        if n_stochastic > 0
        else float("nan")
    )
    accuracy_se_constrained_primary = (
        math.sqrt(
            accuracy_constrained_primary
            * (1 - accuracy_constrained_primary)
            / n_stochastic
        )
        if n_stochastic > 0
        else float("nan")
    )

    # Argmax accuracy: prediction based on the full-vocab argmax token
    y_pred_class_argmax = [
        r.get("predicted_class_argmax", "OTHER") for r in stochastic_rows
    ]
    correct_argmax = sum(
        1
        for gt, pred in zip(y_true_class, y_pred_class_argmax, strict=False)
        if gt == pred
    )
    accuracy_aggregate = correct_argmax / n_stochastic if n_stochastic > 0 else 0.0
    accuracy_se_aggregate = (
        math.sqrt(accuracy_aggregate * (1 - accuracy_aggregate) / n_stochastic)
        if n_stochastic > 0
        else float("nan")
    )
    argmax_is_label_count = sum(
        1
        for r in stochastic_rows
        if r.get("predicted_class_argmax", "OTHER") != "OTHER"
    )
    argmax_is_label_rate = (
        argmax_is_label_count / n_stochastic if n_stochastic > 0 else 0.0
    )
    # Strict argmax accuracy: the full-vocab argmax must be exactly the primary token " L"
    y_pred_class_argmax_primary = [
        r["predicted_class_argmax_primary"] for r in stochastic_rows
    ]
    correct_argmax_primary = sum(
        1
        for gt, pred in zip(y_true_class, y_pred_class_argmax_primary, strict=True)
        if gt == pred
    )
    accuracy_primary = correct_argmax_primary / n_stochastic
    argmax_is_primary_label_rate = (
        sum(1 for pred in y_pred_class_argmax_primary if pred != "OTHER") / n_stochastic
    )
    accuracy_se_primary = math.sqrt(
        accuracy_primary * (1 - accuracy_primary) / n_stochastic
    )

    print(f"\n--- Accuracy (stochastic rows, n={n_stochastic}) ---")
    print(
        f"  Constrained aggregate: {correct_agg}/{n_stochastic} = {accuracy_constrained_aggregate:.2%} "
        f"(SE: {accuracy_se_constrained_aggregate:.2%})"
    )
    print(
        f"  Constrained primary:   {correct_pri}/{n_stochastic} = {accuracy_constrained_primary:.2%} "
        f"(SE: {accuracy_se_constrained_primary:.2%})"
    )
    print(
        f"  Aggregate (argmax in variants): {correct_argmax}/{n_stochastic} = {accuracy_aggregate:.2%} "
        f"(SE: {accuracy_se_aggregate:.2%})"
    )
    print(
        f"  Argmax is label: {argmax_is_label_count}/{n_stochastic} = {argmax_is_label_rate:.2%}"
    )
    print(
        f"  Primary (argmax == ' L'): {correct_argmax_primary}/{n_stochastic} = "
        f"{accuracy_primary:.2%} (SE: {accuracy_se_primary:.2%})"
    )

    # Ground truth distribution
    print("\n  Ground truth distribution:")
    for cls in class_names:
        count = sum(1 for gt in y_true_class if gt == cls)
        print(f"    {cls}: {count}")

    # --- Entropy ---
    entropy_vals = [r["entropy"] for r in stochastic_rows if "entropy" in r]
    entropy_mean: float | None = None
    entropy_std: float | None = None
    entropy_se: float | None = None
    if entropy_vals:
        entropy_mean = float(np.mean(entropy_vals))
        entropy_std = float(np.std(entropy_vals))
        entropy_se = entropy_std / math.sqrt(len(entropy_vals))
        print("\n--- Entropy ---")
        print(
            f"  Mean: {entropy_mean:.4f} +/- {entropy_std:.4f} (SEM: {entropy_se:.4f})"
        )

    # --- Class Confusion Matrices ---
    print("\n--- Class Confusion Matrices ---")

    # Exclude classes with zero true samples from macro averages.
    # In single-perturbation runs (e.g. only DROPOUT), recall/precision
    # for the absent class is not scientifically useful.
    absent_classes = [c for c in class_names if c not in stochastic_classes]
    exclude_macro = absent_classes if absent_classes else None
    macro_classes = [c for c in class_names if c not in (exclude_macro or [])]
    macro_note = f"(over {' and '.join(macro_classes)} only)" if exclude_macro else ""

    cm_class_agg, pc_class_agg, mp_ca, mr_ca, mf1_ca = _compute_cm_metrics(
        y_true_class,
        y_pred_class_agg,
        class_names,
        exclude_from_macro=exclude_macro,
    )
    _print_cm(
        cm_class_agg,
        class_names,
        pc_class_agg,
        mp_ca,
        mr_ca,
        mf1_ca,
        "Class confusion matrix (aggregate)",
        macro_note,
    )

    cm_class_pri, pc_class_pri, mp_cp, mr_cp, mf1_cp = _compute_cm_metrics(
        y_true_class,
        y_pred_class_pri,
        class_names,
        exclude_from_macro=exclude_macro,
    )
    _print_cm(
        cm_class_pri,
        class_names,
        pc_class_pri,
        mp_cp,
        mr_cp,
        mf1_cp,
        "Class confusion matrix (primary)",
        macro_note,
    )

    # --- Label Confusion Matrices ---
    print("\n--- Label Confusion Matrices ---")

    y_true_label = [r["ground_truth_letter"] for r in stochastic_rows]
    y_pred_label_agg = [r["predicted_label_aggregate"] for r in stochastic_rows]
    cm_label_agg, pc_label_agg, mp_la, mr_la, mf1_la = _compute_cm_metrics(
        y_true_label,
        y_pred_label_agg,
        labels,
    )
    _print_cm(
        cm_label_agg,
        labels,
        pc_label_agg,
        mp_la,
        mr_la,
        mf1_la,
        "Label confusion matrix (aggregate)",
        show_metrics=False,
    )

    y_pred_label_pri = [r["predicted_label_primary"] for r in stochastic_rows]
    cm_label_pri, pc_label_pri, mp_lp, mr_lp, mf1_lp = _compute_cm_metrics(
        y_true_label,
        y_pred_label_pri,
        labels,
    )
    _print_cm(
        cm_label_pri,
        labels,
        pc_label_pri,
        mp_lp,
        mr_lp,
        mf1_lp,
        "Label confusion matrix (primary)",
        show_metrics=False,
    )

    # --- Position Bias ---
    print("\n--- Position Bias ---")
    position_bias_metrics: dict[str, float] = {}
    for letter in labels:
        agg_vals = [r[f"sum_prob_{letter}"] for r in stochastic_rows]
        pri_vals = [r[f"primary_prob_{letter}"] for r in stochastic_rows]
        mean_agg = float(np.mean(agg_vals))
        mean_pri = float(np.mean(pri_vals))
        print(
            f"  Mean sumP({letter}): {mean_agg:.4f}    "
            f'Mean P(" {letter}"): {mean_pri:.4f}'
        )
        position_bias_metrics[f"position_bias_aggregate_prob_{letter}"] = mean_agg
        position_bias_metrics[f"position_bias_primary_prob_{letter}"] = mean_pri

    # --- ROC AUC ---
    roc_classes = perturbation_classes
    print(f"\n--- ROC AUC ({' and '.join(roc_classes)}) ---")

    roc_metrics_agg = _compute_ovr_roc(
        stochastic_rows, roc_classes, "sum_prob", "constrained_aggregate", dpi, fmt
    )
    roc_metrics_pri = _compute_ovr_roc(
        stochastic_rows, roc_classes, "primary_prob", "constrained_primary", dpi, fmt
    )

    # --- Plots (no terminal output) ---
    _plot_cm(
        cm_class_agg,
        class_names,
        "Class Aggregate",
        "confusion_matrix_class_constrained_aggregate",
        dpi,
        fmt,
    )
    _plot_cm(
        cm_class_pri,
        class_names,
        "Class Primary",
        "confusion_matrix_class_constrained_primary",
        dpi,
        fmt,
    )
    _plot_cm(
        cm_label_agg,
        labels,
        "Label Aggregate",
        "confusion_matrix_label_constrained_aggregate",
        dpi,
        fmt,
    )
    _plot_cm(
        cm_label_pri,
        labels,
        "Label Primary",
        "confusion_matrix_label_constrained_primary",
        dpi,
        fmt,
    )

    _fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    if len(class_names) == 1:
        axes = [axes]
    for idx, cls in enumerate(class_names):
        key = f"sum_prob_{cls.lower()}"
        for pc in perturbation_classes:
            color = {"DROPOUT": "red", "NOISE": "green"}.get(pc, "blue")
            vals = [r[key] for r in group_rows[pc]]
            axes[idx].hist(
                vals, bins=15, alpha=0.6, label=pc, color=color, edgecolor="black"
            )
        axes[idx].set_xlabel(f"sumP({cls})")
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(f"Distribution of sumP({cls})")
        axes[idx].legend()
        axes[idx].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"class_probability_distributions.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()

    if torch.cuda.is_available():
        print(f"\nFinal GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    print("=" * 70)

    # --- Collect plot files ---
    plot_files = [
        f"confusion_matrix_class_constrained_aggregate.{fmt}",
        f"confusion_matrix_class_constrained_primary.{fmt}",
        f"confusion_matrix_label_constrained_aggregate.{fmt}",
        f"confusion_matrix_label_constrained_primary.{fmt}",
        f"class_probability_distributions.{fmt}",
        f"multiclass_roc_curves_constrained_aggregate.{fmt}",
        f"multiclass_roc_curves_constrained_primary.{fmt}",
    ]

    # --- Build metrics dict ---
    metrics: dict[str, typing.Any] = {
        "_plot_files": plot_files,
        "total_rows": len(results),
        "num_baselines": len(baselines),
        "metrics_version": 2,
        "accuracy_constrained_aggregate": accuracy_constrained_aggregate,
        "accuracy_constrained_primary": accuracy_constrained_primary,
        "accuracy_aggregate": accuracy_aggregate,
        "accuracy_primary": accuracy_primary,
        "accuracy_se_constrained_aggregate": accuracy_se_constrained_aggregate,
        "accuracy_se_constrained_primary": accuracy_se_constrained_primary,
        "accuracy_se_aggregate": accuracy_se_aggregate,
        "accuracy_se_primary": accuracy_se_primary,
        "argmax_is_label_count": argmax_is_label_count,
        "argmax_is_label_rate": argmax_is_label_rate,
        "argmax_is_primary_label_rate": argmax_is_primary_label_rate,
        "entropy_mean": entropy_mean if entropy_vals else None,
        "entropy_std": entropy_std if entropy_vals else None,
        "entropy_se": entropy_se if entropy_vals else None,
        "macro_precision_class_constrained_aggregate": mp_ca,
        "macro_recall_class_constrained_aggregate": mr_ca,
        "macro_f1_class_constrained_aggregate": mf1_ca,
        "macro_precision_class_constrained_primary": mp_cp,
        "macro_recall_class_constrained_primary": mr_cp,
        "macro_f1_class_constrained_primary": mf1_cp,
        "macro_precision_label_constrained_aggregate": mp_la,
        "macro_recall_label_constrained_aggregate": mr_la,
        "macro_f1_label_constrained_aggregate": mf1_la,
        "macro_precision_label_constrained_primary": mp_lp,
        "macro_recall_label_constrained_primary": mr_lp,
        "macro_f1_label_constrained_primary": mf1_lp,
    }

    # Per-group sample counts, per-class recall, and predicted counts
    for cls in perturbation_classes:
        cl = cls.lower()
        metrics[f"num_{cl}"] = len(group_rows[cls])
        metrics[f"recall_constrained_aggregate_{cl}"] = pc_class_agg.get(cls, {}).get(
            "recall"
        )
        metrics[f"recall_constrained_primary_{cl}"] = pc_class_pri.get(cls, {}).get(
            "recall"
        )
        metrics[f"predicted_count_constrained_aggregate_{cl}"] = sum(
            1 for r in stochastic_rows if r["predicted_class_aggregate"] == cls
        )
        metrics[f"predicted_count_constrained_primary_{cl}"] = sum(
            1 for r in stochastic_rows if r["predicted_class_primary"] == cls
        )

    metrics.update(roc_metrics_agg)
    metrics.update(roc_metrics_pri)
    for stats in all_group_stats.values():
        metrics.update(stats)
    metrics.update(position_bias_metrics)

    # Per-group log probability of the correct answer
    for group_name in perturbation_classes:
        gn = group_name.lower()
        for lp_key in ("primary_log_prob_correct", "aggregate_log_prob_correct"):
            lp_values = [r[lp_key] for r in group_rows[group_name] if lp_key in r]
            if lp_values:
                lp_std = float(np.std(lp_values))
                lp_n = len(lp_values)
                metrics[f"{gn}_mean_{lp_key}"] = float(np.mean(lp_values))
                metrics[f"{gn}_std_{lp_key}"] = lp_std
                metrics[f"{gn}_se_{lp_key}"] = lp_std / math.sqrt(lp_n)

    # Overall group stats across all samples (no prefix)
    overall_stats = _compute_group_stats(stochastic_rows, class_names, "overall")
    metrics.update({k.removeprefix("overall_"): v for k, v in overall_stats.items()})

    # Pairwise logit diff: overall and per-group (both primary and aggregate)
    pairwise_pairs = [
        (class_names[i].lower(), class_names[j].lower())
        for i in range(len(class_names))
        for j in range(i + 1, len(class_names))
    ]
    for x, y in pairwise_pairs:
        for prefix in ("logit_diff", "aggregate_logit_diff"):
            key = f"{prefix}_{x}_vs_{y}"
            values = [r[key] for r in stochastic_rows if key in r]
            if values:
                std = float(np.std(values))
                metrics[f"mean_{key}"] = float(np.mean(values))
                metrics[f"std_{key}"] = std
                metrics[f"se_{key}"] = std / math.sqrt(len(values))

    for group_name in perturbation_classes:
        gn = group_name.lower()
        for x, y in pairwise_pairs:
            for prefix in ("logit_diff", "aggregate_logit_diff"):
                key = f"{prefix}_{x}_vs_{y}"
                values = [r[key] for r in group_rows[group_name] if key in r]
                if values:
                    std = float(np.std(values))
                    metrics[f"{gn}_mean_{key}"] = float(np.mean(values))
                    metrics[f"{gn}_std_{key}"] = std
                    metrics[f"{gn}_se_{key}"] = std / math.sqrt(len(values))

    return metrics


def _compute_ovr_roc(
    stochastic_rows: list[dict[str, typing.Any]],
    roc_classes: list[str],
    prob_prefix: str,
    version: str,
    dpi: int,
    fmt: str,
) -> dict[str, typing.Any]:
    """Compute one vs rest ROC for given classes and plot curves.

    Args:
        stochastic_rows: Stochastic result rows.
        roc_classes: Classes to compute ROC for.
        prob_prefix: Column prefix (``"sum_prob"`` or ``"primary_prob"``).
        version: ``"aggregate"`` or ``"primary"``.
        dpi: Plot resolution.
        fmt: Image format.

    Returns:
        Dict with ``roc_auc_{version}_{cls}`` for each class and ``roc_auc_{version}_macro``.
    """
    roc_out: dict[str, typing.Any] = {}
    colors = {"DROPOUT": "red", "NOISE": "green"}
    plt.figure(figsize=(10, 8))

    auc_values = []
    for cls in roc_classes:
        y_true_binary = [1 if r["perturbation"] == cls else 0 for r in stochastic_rows]
        y_scores = [r[f"{prob_prefix}_{cls.lower()}"] for r in stochastic_rows]
        n_positive = sum(y_true_binary)
        n_negative = len(y_true_binary) - n_positive

        if n_positive > 0 and n_negative > 0:
            fpr, tpr, _thresholds = roc_curve(y_true_binary, y_scores)
            roc_auc_val = float(auc(fpr, tpr))
            auc_values.append(roc_auc_val)
            roc_out[f"roc_auc_{version}_{cls.lower()}"] = roc_auc_val

            plt.plot(
                fpr,
                tpr,
                color=colors.get(cls, "blue"),
                lw=2,
                label=f"{cls} (AUC = {roc_auc_val:.4f}, n={n_positive})",
            )
            print(
                f"    {cls:<15} AUC ({version}) = {roc_auc_val:.4f} (support = {n_positive})"
            )
        else:
            roc_out[f"roc_auc_{version}_{cls.lower()}"] = None
            print(
                f"    {cls:<15} Cannot compute ROC ({version}) "
                f"(positives={n_positive}, negatives={n_negative})"
            )

    macro_auc = float(np.mean(auc_values)) if auc_values else None
    roc_out[f"roc_auc_{version}_macro"] = macro_auc

    plt.plot(
        [0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random classifier"
    )
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.05))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"One vs Rest ROC Curves ({version})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    filename = f"multiclass_roc_curves_{version}.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()

    return roc_out


def _plot_cm(
    cm: np.ndarray,
    label_list: list[str],
    title: str,
    filename_base: str,
    dpi: int,
    fmt: str,
) -> None:
    """Save a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_list,
        yticklabels=label_list,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.title(f"Confusion Matrix: {title}")
    filename = f"{filename_base}.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
