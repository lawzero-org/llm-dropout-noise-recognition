"""Few-shot (ICL) classification metrics and plots."""

import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


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
        exclude_from_macro: Labels to exclude from macro averages.

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


def _plot_cm(
    cm: np.ndarray,
    label_list: list[str],
    title: str,
    filename_base: str,
    dpi: int,
    fmt: str,
) -> None:
    """Save a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
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


def compute_and_plot(
    results: list[dict[str, typing.Any]],
    class_names: list[str],
    dpi: int = 300,
    fmt: str = "png",
) -> dict[str, typing.Any]:
    """Compute ICL teaching metrics, print a summary, and save plots.

    Computes four confusion matrices:
    1. Class aggregate (by sumP)
    2. Class primary (by primary logit)
    3. Label aggregate (A/B by sumP)
    4. Label primary (A/B by primary logit)

    Args:
        results: List of result dicts from ``run()``.
        class_names: The two class names (``["DROPOUT", "NOISE"]``).
        dpi: Resolution for saved plots.
        fmt: Image format (e.g. ``"png"``).

    Returns:
        Dict of computed metrics.
    """
    baselines = [r for r in results if r["is_baseline"]]
    stochastic_rows = [r for r in results if not r["is_baseline"]]
    num_classes = len(class_names)
    letters = [chr(65 + i) for i in range(num_classes)]

    # Group stochastic rows by class
    rows_by_class: dict[str, list[dict]] = {}
    for cls in class_names:
        rows_by_class[cls] = [
            r for r in stochastic_rows if r["test_perturbation"] == cls
        ]

    print("\n" + "=" * 70)
    print("SUMMARY - ICL TEACHING CLASSIFICATION")
    print("=" * 70)
    print(f"Classes: {' vs '.join(class_names)}")
    counts_str = " + ".join(f"{len(rows_by_class[c])} {c}" for c in class_names)
    print(f"Total rows: {len(results)} ({len(baselines)} baseline + {counts_str})")

    # --- Baseline ---
    if baselines:
        print(f"\n--- Baseline ({len(baselines)} row(s), no perturbation) ---")
        for bl_idx, bl in enumerate(baselines):
            print(f"  Baseline {bl_idx}:")
            print(f"    Argmax token:       {bl['argmax_token']}")
            print(f"    Argmax logit:       {bl['argmax_logit']:+.4f}")
            print(f"    Argmax prob:        {bl['argmax_prob']:.6f}")
            for cls in class_names:
                print(
                    f"    P(primary of {cls}):  {bl[f'primary_prob_{cls.lower()}']:.6f}    "
                    f"sumP({cls}):  {bl[f'sum_prob_{cls.lower()}']:.6f}"
                )
            print(f"    Predicted class:    {bl['predicted_class_aggregate']}")

        if len(baselines) > 1:
            print(f"\n  Baseline aggregate ({len(baselines)} baselines):")
            for cls in class_names:
                pp_vals = [bl[f"primary_prob_{cls.lower()}"] for bl in baselines]
                sp_vals = [bl[f"sum_prob_{cls.lower()}"] for bl in baselines]
                print(
                    f"    P(primary of {cls}):  {np.mean(pp_vals):.6f} ± {np.std(pp_vals):.6f}    "
                    f"sumP({cls}):  {np.mean(sp_vals):.6f} ± {np.std(sp_vals):.6f}"
                )

    # --- Perturbed group summaries ---
    group_stats_by_class: dict[str, dict[str, float]] = {}
    for cls in class_names:
        rows = rows_by_class[cls]
        if not rows:
            print(f"\n--- {cls} (0 rows) ---")
            continue
        print(f"\n--- {cls} ({len(rows)} rows) ---")
        cls_prefix = cls.lower()
        cls_n = len(rows)
        cls_sqrt_n = math.sqrt(cls_n)
        cls_stats: dict[str, float] = {}
        for other_cls in class_names:
            other_lower = other_cls.lower()
            pp_vals = [r[f"primary_prob_{other_lower}"] for r in rows]
            sp_vals = [r[f"sum_prob_{other_lower}"] for r in rows]
            pp_std = float(np.std(pp_vals))
            sp_std = float(np.std(sp_vals))
            cls_stats[f"{cls_prefix}_mean_aggregate_prob_{other_lower}"] = float(
                np.mean(sp_vals)
            )
            cls_stats[f"{cls_prefix}_std_aggregate_prob_{other_lower}"] = sp_std
            cls_stats[f"{cls_prefix}_se_aggregate_prob_{other_lower}"] = (
                sp_std / cls_sqrt_n
            )
            cls_stats[f"{cls_prefix}_mean_primary_prob_{other_lower}"] = float(
                np.mean(pp_vals)
            )
            cls_stats[f"{cls_prefix}_std_primary_prob_{other_lower}"] = pp_std
            cls_stats[f"{cls_prefix}_se_primary_prob_{other_lower}"] = (
                pp_std / cls_sqrt_n
            )
            print(
                f"  P(primary of {other_cls}): {np.mean(pp_vals):.6f} ± {pp_std:.6f}    "
                f"sumP({other_cls}): {np.mean(sp_vals):.6f} ± {sp_std:.6f}"
            )
        for other_cls in class_names:
            other_lower = other_cls.lower()
            ld_key = f"logit_diff_{other_lower}"
            lsd_key = f"logsumexp_diff_{other_lower}"
            ld_vals = [r[ld_key] for r in rows if ld_key in r]
            lsd_vals = [r[lsd_key] for r in rows if lsd_key in r]
            if ld_vals:
                ld_mean = float(np.mean(ld_vals))
                ld_std = float(np.std(ld_vals))
                lsd_mean = float(np.mean(lsd_vals))
                lsd_std = float(np.std(lsd_vals))
                cls_stats[f"{cls_prefix}_mean_logit_diff_{other_lower}"] = ld_mean
                cls_stats[f"{cls_prefix}_std_logit_diff_{other_lower}"] = ld_std
                cls_stats[f"{cls_prefix}_se_logit_diff_{other_lower}"] = (
                    ld_std / math.sqrt(len(ld_vals))
                )
                cls_stats[f"{cls_prefix}_mean_aggregate_logit_diff_{other_lower}"] = (
                    lsd_mean
                )
                cls_stats[f"{cls_prefix}_std_aggregate_logit_diff_{other_lower}"] = (
                    lsd_std
                )
                cls_stats[f"{cls_prefix}_se_aggregate_logit_diff_{other_lower}"] = (
                    lsd_std / math.sqrt(len(lsd_vals))
                )
                print(
                    f"  logit_diff({other_cls}): {ld_mean:+.4f} ± {ld_std:.4f}    "
                    f"logsumexp_diff({other_cls}): {lsd_mean:+.4f} ± {lsd_std:.4f}"
                )
        group_stats_by_class[cls] = cls_stats

    # --- Classification on stochastic rows ---
    if not stochastic_rows:
        print("\nNo stochastic rows to evaluate.")
        return {"total_rows": len(results), "num_baselines": len(baselines)}

    y_true_class = [r["test_perturbation"] for r in stochastic_rows]
    labels_ordered = list(class_names)

    # --- Accuracy ---
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
    accuracy_constrained_aggregate = correct_agg / n_stochastic
    accuracy_constrained_primary = correct_pri / n_stochastic
    chance = 1.0 / num_classes

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
    # Primary accuracy: the full-vocab argmax must be exactly the primary token " L"
    y_pred_class_argmax_primary = [
        r["predicted_class_argmax_primary"] for r in stochastic_rows
    ]
    correct_primary = sum(
        1
        for gt, pred in zip(y_true_class, y_pred_class_argmax_primary, strict=True)
        if gt == pred
    )
    accuracy_primary = correct_primary / n_stochastic
    argmax_is_primary_label_rate = (
        sum(1 for pred in y_pred_class_argmax_primary if pred != "OTHER") / n_stochastic
    )
    accuracy_se_primary = math.sqrt(
        accuracy_primary * (1 - accuracy_primary) / n_stochastic
    )

    print(f"\n--- Accuracy (stochastic rows, n={n_stochastic}) ---")
    print(
        f"  Constrained aggregate: {correct_agg}/{n_stochastic} = {accuracy_constrained_aggregate:.2%} "
        f"(SE: {accuracy_se_constrained_aggregate:.2%}, chance = {chance:.0%})"
    )
    print(
        f"  Constrained primary:   {correct_pri}/{n_stochastic} = {accuracy_constrained_primary:.2%} "
        f"(SE: {accuracy_se_constrained_primary:.2%}, chance = {chance:.0%})"
    )
    print(
        f"  Aggregate (argmax in variants): {correct_argmax}/{n_stochastic} = {accuracy_aggregate:.2%} "
        f"(SE: {accuracy_se_aggregate:.2%}, chance = {chance:.0%})"
    )
    print(
        f"  Argmax is label: {argmax_is_label_count}/{n_stochastic} = {argmax_is_label_rate:.2%}"
    )
    print(
        f"  Primary (argmax == ' L'): {correct_primary}/{n_stochastic} = "
        f"{accuracy_primary:.2%} (SE: {accuracy_se_primary:.2%}, chance = {chance:.0%})"
    )

    # Ground truth distribution
    print("\n  Ground truth distribution:")
    for cls in labels_ordered:
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

    # --- Confusion Matrices ---
    print("\n--- Class Confusion Matrices ---")

    # 1. Class aggregate
    cm_class_agg, pc_class_agg, mp_ca, mr_ca, mf1_ca = _compute_cm_metrics(
        y_true_class,
        y_pred_class_agg,
        labels_ordered,
    )
    _print_cm(
        cm_class_agg,
        labels_ordered,
        pc_class_agg,
        mp_ca,
        mr_ca,
        mf1_ca,
        "Class confusion matrix (aggregate)",
    )

    # 2. Class primary
    cm_class_pri, pc_class_pri, mp_cp, mr_cp, mf1_cp = _compute_cm_metrics(
        y_true_class,
        y_pred_class_pri,
        labels_ordered,
    )
    _print_cm(
        cm_class_pri,
        labels_ordered,
        pc_class_pri,
        mp_cp,
        mr_cp,
        mf1_cp,
        "Class confusion matrix (primary)",
    )

    print("\n--- Label Confusion Matrices ---")

    # 3. Label aggregate
    y_true_label = [r["ground_truth_letter"] for r in stochastic_rows]
    y_pred_label_agg = [r["predicted_label_aggregate"] for r in stochastic_rows]
    cm_label_agg, pc_label_agg, mp_la, mr_la, mf1_la = _compute_cm_metrics(
        y_true_label,
        y_pred_label_agg,
        letters,
    )
    _print_cm(
        cm_label_agg,
        letters,
        pc_label_agg,
        mp_la,
        mr_la,
        mf1_la,
        "Label confusion matrix (aggregate)",
        show_metrics=False,
    )

    # 4. Label primary
    y_pred_label_pri = [r["predicted_label_primary"] for r in stochastic_rows]
    cm_label_pri, pc_label_pri, mp_lp, mr_lp, mf1_lp = _compute_cm_metrics(
        y_true_label,
        y_pred_label_pri,
        letters,
    )
    _print_cm(
        cm_label_pri,
        letters,
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
    for letter in letters:
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
    print("\n--- ROC AUC ---")
    roc_auc_constrained_aggregate = _compute_and_plot_roc(
        stochastic_rows,
        class_names,
        "sum_prob",
        "constrained_aggregate",
        dpi,
        fmt,
    )
    roc_auc_constrained_primary = _compute_and_plot_roc(
        stochastic_rows,
        class_names,
        "primary_prob",
        "constrained_primary",
        dpi,
        fmt,
    )

    # --- Plots (no terminal output) ---
    _plot_cm(
        cm_class_agg,
        labels_ordered,
        "ICL Class Aggregate",
        "icl_cm_class_constrained_aggregate",
        dpi,
        fmt,
    )
    _plot_cm(
        cm_class_pri,
        labels_ordered,
        "ICL Class Primary",
        "icl_cm_class_constrained_primary",
        dpi,
        fmt,
    )
    _plot_cm(
        cm_label_agg,
        letters,
        "ICL Label Aggregate",
        "icl_cm_label_constrained_aggregate",
        dpi,
        fmt,
    )
    _plot_cm(
        cm_label_pri,
        letters,
        "ICL Label Primary",
        "icl_cm_label_constrained_primary",
        dpi,
        fmt,
    )

    _fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]

    plot_colors = {"DROPOUT": "red", "NOISE": "green"}

    for idx, cls in enumerate(labels_ordered):
        key = f"sum_prob_{cls.lower()}"

        baseline_vals = [r[key] for r in baselines] if baselines else []
        if baseline_vals:
            axes[idx].hist(
                baseline_vals,
                bins=15,
                alpha=0.6,
                label="Baseline",
                color="lightblue",
                edgecolor="black",
            )

        for other_cls in class_names:
            vals = [r[key] for r in rows_by_class[other_cls]]
            color = plot_colors.get(other_cls, "blue")
            axes[idx].hist(
                vals,
                bins=15,
                alpha=0.6,
                label=other_cls,
                color=color,
                edgecolor="black",
            )

        axes[idx].set_xlabel(f"sumP({cls})")
        axes[idx].set_ylabel("Count")
        axes[idx].set_title(f"Distribution of sumP({cls})")
        axes[idx].legend()
        axes[idx].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"icl_probability_distributions.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()

    print("=" * 70)

    # --- Collect plot files ---
    plot_files = [
        f"icl_cm_class_constrained_aggregate.{fmt}",
        f"icl_cm_class_constrained_primary.{fmt}",
        f"icl_cm_label_constrained_aggregate.{fmt}",
        f"icl_cm_label_constrained_primary.{fmt}",
        f"icl_probability_distributions.{fmt}",
    ]
    if roc_auc_constrained_aggregate is not None:
        plot_files.append(f"icl_roc_curve_constrained_aggregate.{fmt}")
    if roc_auc_constrained_primary is not None:
        plot_files.append(f"icl_roc_curve_constrained_primary.{fmt}")

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
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "entropy_se": entropy_se,
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
        "roc_auc_constrained_aggregate": roc_auc_constrained_aggregate,
        "roc_auc_constrained_primary": roc_auc_constrained_primary,
    }

    for cls in class_names:
        cls_lower = cls.lower()
        metrics[f"num_{cls_lower}"] = len(rows_by_class[cls])

    # Baseline metrics (aggregate across all baselines)
    if baselines:
        for cls in class_names:
            cls_lower = cls.lower()
            vals = [bl[f"sum_prob_{cls_lower}"] for bl in baselines]
            metrics[f"baseline_mean_aggregate_prob_{cls_lower}"] = float(np.mean(vals))
            metrics[f"baseline_std_aggregate_prob_{cls_lower}"] = float(np.std(vals))
        metrics["baseline_argmax_token"] = baselines[0]["argmax_token"]
        metrics["baseline_predicted_class"] = baselines[0]["predicted_class_aggregate"]

    # Group stats for each class under each test condition
    for label_cls in class_names:
        if label_cls in group_stats_by_class:
            metrics.update(group_stats_by_class[label_cls])

    metrics.update(position_bias_metrics)

    return metrics


def _compute_and_plot_roc(
    stochastic_rows: list[dict[str, typing.Any]],
    class_names: list[str],
    prob_prefix: str,
    version: str,
    dpi: int,
    fmt: str,
) -> float | None:
    """Compute the ROC AUC and plot the curve for a given probability version.

    Args:
        stochastic_rows: Stochastic result rows.
        class_names: Class names.
        prob_prefix: Column prefix (``"sum_prob"`` or ``"primary_prob"``).
        version: ``"aggregate"`` or ``"primary"``.
        dpi: Plot resolution.
        fmt: Image format.

    Returns:
        AUC value (first class as positive), or None if a class is missing.
    """
    roc_auc: float | None = None

    cls_a_lower = class_names[0].lower()
    y_true_binary = [
        1 if r["test_perturbation"] == class_names[0] else 0 for r in stochastic_rows
    ]
    y_scores = [r[f"{prob_prefix}_{cls_a_lower}"] for r in stochastic_rows]
    n_positive = sum(y_true_binary)
    n_negative = len(y_true_binary) - n_positive

    if n_positive > 0 and n_negative > 0:
        fpr, tpr, _thresholds = roc_curve(y_true_binary, y_scores)
        roc_auc = float(auc(fpr, tpr))
        print(
            f"\n  ROC AUC ({version}, {class_names[0]} vs {class_names[1]}): {roc_auc:.4f}"
        )

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr,
            tpr,
            color="blue",
            lw=2,
            label=f"{class_names[0]} vs {class_names[1]} (AUC = {roc_auc:.4f})",
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color="gray",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ICL Teaching ROC ({version}): {class_names[0]} vs {class_names[1]}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f"icl_roc_curve_{version}.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        print(
            f"\n  Cannot compute ROC ({version}) (positives={n_positive}, negatives={n_negative})"
        )

    return roc_auc
