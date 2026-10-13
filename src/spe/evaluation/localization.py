"""Localization metrics and plots."""

import math
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


def compute_and_plot(
    results: list[dict[str, typing.Any]],
    labels: list[str] | None = None,
    dpi: int = 300,
    fmt: str = "png",
) -> dict[str, typing.Any]:
    """Compute localization metrics, print a summary, and save plots.

    Computes and prints two versions of each classification metric:
    - **Aggregate**: predicted label is the one with the highest ``sumP``
      (from ``predicted_aggregate``).
    - **Primary**: predicted label is the one whose primary token has the
      highest logit (from ``predicted_primary``).

    Args:
        results: List of result dicts, each with keys ``ground_truth``,
            ``predicted_aggregate``, ``predicted_primary``,
            ``prob_<L>``, ``logsumexp_<L>``, ``primary_prob_<L>``,
            ``primary_logit_<L>`` for each label L.
        labels: The two sentence labels (["A", "B"]). Auto detected from
            results if None.
        dpi: Resolution for saved plots.
        fmt: Image format (e.g. ``"png"``).

    Returns:
        Dict of computed metrics.
    """
    if labels is None:
        labels = sorted({r["ground_truth"] for r in results})

    n = len(results)
    y_true = [r["ground_truth"] for r in results]
    y_pred_aggregate = [r["predicted_aggregate"] for r in results]
    y_pred_primary = [r["predicted_primary"] for r in results]
    if len(labels) != 2:
        raise ValueError(f"localization has two sentences, got labels {labels}")

    label_str = "/".join(labels)
    print("\n" + "=" * 70)
    print(f"SUMMARY - LOCALIZATION ({label_str})")
    print("=" * 70)
    print(f"Total samples: {n}")

    # --- Distributions ---
    print("\n--- Distributions ---")
    print("\n  Ground Truth:")
    for cls in labels:
        count = sum(1 for v in y_true if v == cls)
        print(f"    {cls}: {count}")
    for y_pred, version in [
        (y_pred_aggregate, "constrained_aggregate"),
        (y_pred_primary, "constrained_primary"),
    ]:
        print(f"\n  Predictions ({version}):")
        for cls in labels:
            count = sum(1 for v in y_pred if v == cls)
            print(f"    {cls}: {count}")

    # --- Accuracy ---
    correct_agg = sum(
        1 for gt, pred in zip(y_true, y_pred_aggregate, strict=False) if gt == pred
    )
    correct_pri = sum(
        1 for gt, pred in zip(y_true, y_pred_primary, strict=False) if gt == pred
    )
    accuracy_constrained_aggregate = correct_agg / n if n > 0 else 0.0
    accuracy_constrained_primary = correct_pri / n if n > 0 else 0.0
    accuracy_se_constrained_aggregate = (
        math.sqrt(
            accuracy_constrained_aggregate * (1 - accuracy_constrained_aggregate) / n
        )
        if n > 0
        else float("nan")
    )
    accuracy_se_constrained_primary = (
        math.sqrt(accuracy_constrained_primary * (1 - accuracy_constrained_primary) / n)
        if n > 0
        else float("nan")
    )
    # Argmax accuracy: prediction based on the full-vocab argmax token
    y_pred_argmax = [r.get("predicted_argmax", "OTHER") for r in results]
    correct_argmax = sum(
        1 for gt, pred in zip(y_true, y_pred_argmax, strict=False) if gt == pred
    )
    accuracy_aggregate = correct_argmax / n if n > 0 else 0.0
    accuracy_se_aggregate = (
        math.sqrt(accuracy_aggregate * (1 - accuracy_aggregate) / n)
        if n > 0
        else float("nan")
    )
    argmax_is_label_count = sum(1 for pred in y_pred_argmax if pred != "OTHER")
    argmax_is_label_rate = argmax_is_label_count / n if n > 0 else 0.0
    # Strict argmax accuracy: the full-vocab argmax must be exactly the primary token " L"
    y_pred_argmax_primary = [r["predicted_argmax_primary"] for r in results]
    correct_argmax_primary = sum(
        1 for gt, pred in zip(y_true, y_pred_argmax_primary, strict=True) if gt == pred
    )
    accuracy_primary = correct_argmax_primary / n
    accuracy_se_primary = math.sqrt(accuracy_primary * (1 - accuracy_primary) / n)
    argmax_is_primary_label_rate = (
        sum(1 for pred in y_pred_argmax_primary if pred != "OTHER") / n
    )

    print("\n--- Accuracy ---")
    print(
        f"  Constrained aggregate: {correct_agg}/{n} = {accuracy_constrained_aggregate:.2%} "
        f"(SE: {accuracy_se_constrained_aggregate:.2%})"
    )
    print(
        f"  Constrained primary:   {correct_pri}/{n} = {accuracy_constrained_primary:.2%} "
        f"(SE: {accuracy_se_constrained_primary:.2%})"
    )
    print(
        f"  Aggregate (argmax in variants): {correct_argmax}/{n} = {accuracy_aggregate:.2%} "
        f"(SE: {accuracy_se_aggregate:.2%})"
    )
    print(
        f"  Argmax is label: {argmax_is_label_count}/{n} = {argmax_is_label_rate:.2%}"
    )
    print(
        f"  Primary (argmax == ' L'): {correct_argmax_primary}/{n} = {accuracy_primary:.2%} "
        f"(SE: {accuracy_se_primary:.2%})"
    )

    # --- Content accuracy (control prompts with sentence_groups) ---
    content_results = [r for r in results if "content_ground_truth" in r]
    content_accuracy_constrained_aggregate: float | None = None
    content_accuracy_constrained_primary: float | None = None
    content_accuracy_aggregate: float | None = None
    content_accuracy_primary: float | None = None
    content_accuracy_se_constrained_aggregate: float | None = None
    content_accuracy_se_constrained_primary: float | None = None
    content_accuracy_se_aggregate: float | None = None
    content_accuracy_se_primary: float | None = None
    if content_results:
        n_content = len(content_results)
        content_correct_agg = sum(
            1
            for r in content_results
            if r["content_ground_truth"] == r["predicted_aggregate"]
        )
        content_correct_pri = sum(
            1
            for r in content_results
            if r["content_ground_truth"] == r["predicted_primary"]
        )
        # Full-vocab argmax: predicted_argmax is "OTHER" when the top-1
        # token is not a label token, which counts as incorrect here.
        content_correct_argmax = sum(
            1
            for r in content_results
            if r["content_ground_truth"] == r["predicted_argmax"]
        )
        content_correct_argmax_primary = sum(
            1
            for r in content_results
            if r["content_ground_truth"] == r["predicted_argmax_primary"]
        )
        content_accuracy_constrained_aggregate = content_correct_agg / n_content
        content_accuracy_constrained_primary = content_correct_pri / n_content
        content_accuracy_aggregate = content_correct_argmax / n_content
        content_accuracy_primary = content_correct_argmax_primary / n_content
        content_accuracy_se_constrained_aggregate = math.sqrt(
            content_accuracy_constrained_aggregate
            * (1 - content_accuracy_constrained_aggregate)
            / n_content
        )
        content_accuracy_se_constrained_primary = math.sqrt(
            content_accuracy_constrained_primary
            * (1 - content_accuracy_constrained_primary)
            / n_content
        )
        content_accuracy_se_aggregate = math.sqrt(
            content_accuracy_aggregate * (1 - content_accuracy_aggregate) / n_content
        )
        content_accuracy_se_primary = math.sqrt(
            content_accuracy_primary * (1 - content_accuracy_primary) / n_content
        )
        print("\n--- Content Accuracy (control question) ---")
        print(
            f"  Constrained aggregate: {content_correct_agg}/{n_content} = "
            f"{content_accuracy_constrained_aggregate:.2%} (SE: {content_accuracy_se_constrained_aggregate:.2%})"
        )
        print(
            f"  Constrained primary:   {content_correct_pri}/{n_content} = "
            f"{content_accuracy_constrained_primary:.2%} (SE: {content_accuracy_se_constrained_primary:.2%})"
        )
        print(
            f"  Aggregate (argmax in variants): {content_correct_argmax}/{n_content} = "
            f"{content_accuracy_aggregate:.2%} (SE: {content_accuracy_se_aggregate:.2%})"
        )
        print(
            f"  Primary (argmax == ' L'): {content_correct_argmax_primary}/{n_content} = "
            f"{content_accuracy_primary:.2%} "
            f"(SE: {content_accuracy_se_primary:.2%})"
        )

    # --- Entropy ---
    entropy_vals = [r["entropy"] for r in results if "entropy" in r]
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

    # --- Logit diffs (config target position vs alternatives) ---
    ld_vals = [r["logit_diff"] for r in results]
    lsd_vals = [r["logsumexp_diff"] for r in results]
    ld_mean = float(np.mean(ld_vals))
    ld_std = float(np.std(ld_vals))
    ld_se = ld_std / math.sqrt(n) if n > 0 else float("nan")
    lsd_mean = float(np.mean(lsd_vals))
    lsd_std = float(np.std(lsd_vals))
    lsd_se = lsd_std / math.sqrt(n) if n > 0 else float("nan")
    print("\n--- Logit Diffs (config target position vs alternatives) ---")
    print(f"  logit_diff:     {ld_mean:+.4f} +/- {ld_std:.4f} (SEM: {ld_se:.4f})")
    print(f"  logsumexp_diff: {lsd_mean:+.4f} +/- {lsd_std:.4f} (SEM: {lsd_se:.4f})")

    # --- Logit diffs (asked about position vs alternatives) ---
    ld_corr_vals = [r["logit_diff_correct_vs_incorrect"] for r in results]
    lsd_corr_vals = [r["logsumexp_diff_correct_vs_incorrect"] for r in results]
    ld_corr_mean = float(np.mean(ld_corr_vals))
    ld_corr_std = float(np.std(ld_corr_vals))
    ld_corr_se = ld_corr_std / math.sqrt(n) if n > 0 else float("nan")
    lsd_corr_mean = float(np.mean(lsd_corr_vals))
    lsd_corr_std = float(np.std(lsd_corr_vals))
    lsd_corr_se = lsd_corr_std / math.sqrt(n) if n > 0 else float("nan")
    # --- Log-probabilities of correct answer ---
    plp_vals = [r["primary_log_prob_correct"] for r in results]
    alp_vals = [r["aggregate_log_prob_correct"] for r in results]
    plp_mean = float(np.mean(plp_vals))
    plp_std = float(np.std(plp_vals))
    plp_se = plp_std / math.sqrt(n) if n > 0 else float("nan")
    alp_mean = float(np.mean(alp_vals))
    alp_std = float(np.std(alp_vals))
    alp_se = alp_std / math.sqrt(n) if n > 0 else float("nan")
    # --- log(1 - P(correct)) ---
    p1mp_vals = [r["primary_log_one_minus_p_correct"] for r in results]
    a1mp_vals = [r["aggregate_log_one_minus_p_correct"] for r in results]
    p1mp_mean = float(np.mean(p1mp_vals))
    p1mp_std = float(np.std(p1mp_vals))
    p1mp_se = p1mp_std / math.sqrt(n) if n > 0 else float("nan")
    a1mp_mean = float(np.mean(a1mp_vals))
    a1mp_std = float(np.std(a1mp_vals))
    a1mp_se = a1mp_std / math.sqrt(n) if n > 0 else float("nan")

    print("\n--- Log-Probability of Correct Answer ---")
    print(
        f"  primary_log_prob_correct:   {plp_mean:+.4f} +/- {plp_std:.4f} (SEM: {plp_se:.4f})"
    )
    print(
        f"  aggregate_log_prob_correct: {alp_mean:+.4f} +/- {alp_std:.4f} (SEM: {alp_se:.4f})"
    )
    print(
        f"  primary_log(1-P(correct)):  {p1mp_mean:+.4f} +/- {p1mp_std:.4f} (SEM: {p1mp_se:.4f})"
    )
    print(
        f"  aggregate_log(1-P(correct)):{a1mp_mean:+.4f} +/- {a1mp_std:.4f} (SEM: {a1mp_se:.4f})"
    )

    print("\n--- Logit Diffs (asked about position vs alternatives) ---")
    print(
        f"  logit_diff_correct_vs_incorrect:     {ld_corr_mean:+.4f} "
        f"+/- {ld_corr_std:.4f} (SEM: {ld_corr_se:.4f})"
    )
    print(
        f"  logsumexp_diff_correct_vs_incorrect: {lsd_corr_mean:+.4f} "
        f"+/- {lsd_corr_std:.4f} (SEM: {lsd_corr_se:.4f})"
    )

    # --- Perturbed vs Unperturbed Joint Log-Prob (standard 2-sentence shape) ---
    plower_vals = [
        r["perturbed_lp_lower"] for r in results if "perturbed_lp_lower" in r
    ]
    delta_vals = [
        r["perturbed_unperturbed_logprob_delta"]
        for r in results
        if "perturbed_unperturbed_logprob_delta" in r
    ]
    perturbed_lp_lower_rate: float | None = None
    perturbed_lp_lower_se: float | None = None
    perturbed_unperturbed_logprob_delta_mean: float | None = None
    perturbed_unperturbed_logprob_delta_std: float | None = None
    perturbed_unperturbed_logprob_delta_se: float | None = None
    if plower_vals:
        n_lp = len(plower_vals)
        perturbed_lp_lower_rate = float(np.mean(plower_vals))
        perturbed_lp_lower_se = math.sqrt(
            perturbed_lp_lower_rate * (1 - perturbed_lp_lower_rate) / n_lp
        )
        perturbed_unperturbed_logprob_delta_mean = float(np.mean(delta_vals))
        perturbed_unperturbed_logprob_delta_std = float(np.std(delta_vals))
        perturbed_unperturbed_logprob_delta_se = (
            perturbed_unperturbed_logprob_delta_std / math.sqrt(n_lp)
        )
        print("\n--- Perturbed vs Unperturbed Joint Log-Prob ---")
        print(
            f"  perturbed_lp_lower_rate: {perturbed_lp_lower_rate:.4f} "
            f"(SE: {perturbed_lp_lower_se:.4f}, n={n_lp})"
        )
        print(
            f"  unperturbed - perturbed: "
            f"{perturbed_unperturbed_logprob_delta_mean:+.4f} "
            f"+/- {perturbed_unperturbed_logprob_delta_std:.4f} "
            f"(SEM: {perturbed_unperturbed_logprob_delta_se:.4f})"
        )

    # --- Confusion matrices (position level) ---
    print("\n--- Confusion Matrices ---")
    _print_confusion_matrix(y_true, y_pred_aggregate, labels, "constrained_aggregate")
    metrics_agg = _compute_classification_metrics(
        y_true,
        y_pred_aggregate,
        labels,
        "constrained_aggregate",
    )
    _print_confusion_matrix(y_true, y_pred_primary, labels, "constrained_primary")
    metrics_pri = _compute_classification_metrics(
        y_true,
        y_pred_primary,
        labels,
        "constrained_primary",
    )

    # --- Position bias ---
    print("\n--- Position Bias ---")
    position_bias_values: dict[str, float] = {}
    for lbl in labels:
        agg_vals = [r[f"prob_{lbl}"] for r in results]
        pri_vals = [r[f"primary_prob_{lbl}"] for r in results]
        mean_agg = float(np.mean(agg_vals))
        mean_pri = float(np.mean(pri_vals))
        position_bias_values[lbl] = mean_agg
        print(f'  Mean sumP({lbl}): {mean_agg:.4f}    Mean P(" {lbl}"): {mean_pri:.4f}')
    position_bias = position_bias_values[labels[0]]

    # --- ROC AUC ---
    print("\n--- ROC AUC ---")
    roc_auc_constrained_aggregate = _compute_roc(
        results,
        labels,
        "prob",
        dpi,
        fmt,
        "constrained_aggregate",
    )
    roc_auc_constrained_primary = _compute_roc(
        results,
        labels,
        "primary_prob",
        dpi,
        fmt,
        "constrained_primary",
    )

    # --- Plots (no terminal output) ---
    _plot_confusion_matrix(
        y_true, y_pred_aggregate, labels, dpi, fmt, "constrained_aggregate"
    )
    _plot_confusion_matrix(
        y_true, y_pred_primary, labels, dpi, fmt, "constrained_primary"
    )

    _plot_logit_distribution(results, labels, dpi, fmt)

    print("=" * 70)

    # --- Collect plot files ---
    plot_files = [
        f"localization_confusion_matrix_constrained_aggregate.{fmt}",
        f"localization_confusion_matrix_constrained_primary.{fmt}",
    ]
    # _plot_roc skips saving when AUC is None (single class).
    if roc_auc_constrained_aggregate is not None:
        plot_files.append(f"localization_roc_constrained_aggregate.{fmt}")
    if roc_auc_constrained_primary is not None:
        plot_files.append(f"localization_roc_constrained_primary.{fmt}")
    plot_files.append(f"localization_logit_distribution.{fmt}")

    # Build return dict (always includes both aggregate and primary)
    result_metrics: dict[str, typing.Any] = {
        "_plot_files": plot_files,
        "total_samples": n,
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
        "content_accuracy_constrained_aggregate": content_accuracy_constrained_aggregate,
        "content_accuracy_constrained_primary": content_accuracy_constrained_primary,
        "content_accuracy_aggregate": content_accuracy_aggregate,
        "content_accuracy_primary": content_accuracy_primary,
        "content_accuracy_se_constrained_aggregate": content_accuracy_se_constrained_aggregate,
        "content_accuracy_se_constrained_primary": content_accuracy_se_constrained_primary,
        "content_accuracy_se_aggregate": content_accuracy_se_aggregate,
        "content_accuracy_se_primary": content_accuracy_se_primary,
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "entropy_se": entropy_se,
        "position_bias": position_bias,
        "roc_auc_constrained_aggregate": roc_auc_constrained_aggregate,
        "roc_auc_constrained_primary": roc_auc_constrained_primary,
        "logit_diff_mean": ld_mean,
        "logit_diff_std": ld_std,
        "logit_diff_se": ld_se,
        "aggregate_logit_diff_mean": lsd_mean,
        "aggregate_logit_diff_std": lsd_std,
        "aggregate_logit_diff_se": lsd_se,
        "logit_diff_correct_vs_incorrect_mean": ld_corr_mean,
        "logit_diff_correct_vs_incorrect_std": ld_corr_std,
        "logit_diff_correct_vs_incorrect_se": ld_corr_se,
        "logsumexp_diff_correct_vs_incorrect_mean": lsd_corr_mean,
        "logsumexp_diff_correct_vs_incorrect_std": lsd_corr_std,
        "logsumexp_diff_correct_vs_incorrect_se": lsd_corr_se,
        "primary_log_prob_correct_mean": plp_mean,
        "primary_log_prob_correct_std": plp_std,
        "primary_log_prob_correct_se": plp_se,
        "aggregate_log_prob_correct_mean": alp_mean,
        "aggregate_log_prob_correct_std": alp_std,
        "aggregate_log_prob_correct_se": alp_se,
        "primary_log_one_minus_p_correct_mean": p1mp_mean,
        "primary_log_one_minus_p_correct_std": p1mp_std,
        "primary_log_one_minus_p_correct_se": p1mp_se,
        "aggregate_log_one_minus_p_correct_mean": a1mp_mean,
        "aggregate_log_one_minus_p_correct_std": a1mp_std,
        "aggregate_log_one_minus_p_correct_se": a1mp_se,
        "perturbed_lp_lower_rate": perturbed_lp_lower_rate,
        "perturbed_lp_lower_se": perturbed_lp_lower_se,
        "perturbed_unperturbed_logprob_delta_mean": (
            perturbed_unperturbed_logprob_delta_mean
        ),
        "perturbed_unperturbed_logprob_delta_std": (
            perturbed_unperturbed_logprob_delta_std
        ),
        "perturbed_unperturbed_logprob_delta_se": (
            perturbed_unperturbed_logprob_delta_se
        ),
    }
    result_metrics.update(metrics_agg)
    result_metrics.update(metrics_pri)

    return result_metrics


# --- Metric helpers -----------------------------------------------


def _compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    version: str,
    verbose: bool = True,
) -> dict[str, typing.Any]:
    """Compute precision, recall, F1 for a given prediction version.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Ordered label set.
        version: ``"aggregate"`` or ``"primary"``, used as key suffix.
        verbose: Whether to print the results to the terminal.

    Returns:
        Dict of ``macro_precision_{version}``, ``macro_recall_{version}``,
        ``macro_f1_{version}``.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0.0,  # type: ignore[arg-type]
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0.0,  # type: ignore[arg-type]
    )

    if verbose:
        label_width = max(8, max(len(l) for l in labels) + 2)
        print(
            f"\n  {'Label':<{label_width}} {'Precision':>10} {'Recall':>10} "
            f"{'F1':>10} {'Support':>8}"
        )
        print(f"  {'-' * (label_width + 42)}")
        for i, cls in enumerate(labels):
            print(
                f"  {cls:<{label_width}} {precision[i]:>10.2%} {recall[i]:>10.2%} "  # type: ignore[index]
                f"{f1[i]:>10.2%} {support[i]:>8}"  # type: ignore[index]
            )
        print(
            f"  {'Macro':<{label_width}} {macro_p:>10.2%} {macro_r:>10.2%} "
            f"{macro_f1:>10.2%}"
        )

    return {
        f"macro_precision_{version}": float(macro_p),
        f"macro_recall_{version}": float(macro_r),
        f"macro_f1_{version}": float(macro_f1),
    }


def _print_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    version: str,
) -> None:
    """Print confusion matrix grid to the terminal.

    Rows represent ground truth, columns represent predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Ordered label set.
        version: ``"aggregate"`` or ``"primary"``, used in the title.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    col_width = max(6, max(len(l) for l in labels) + 1)
    margin = 8  # width for "True" row label area
    row_label_width = max(len(l) for l in labels) + 1

    grid_width = col_width * len(labels)
    print(f"\n  Confusion Matrix ({version}):")
    print(f"  {'':>{margin + row_label_width}}{'Predictions':^{grid_width}}")
    print(
        f"  {'':>{margin + row_label_width}}"
        + "".join(f"{l:>{col_width}}" for l in labels)
    )
    print(f"  {'':>{margin + row_label_width}}" + "-" * grid_width)
    mid = len(labels) // 2
    for i, row_label in enumerate(labels):
        tag = "Targets" if i == mid else ""
        row_vals = "".join(f"{cm[i][j]:>{col_width}d}" for j in range(len(labels)))
        print(f"  {tag:>{margin}}{row_label:>{row_label_width}}{row_vals}")


def _compute_roc(
    results: list[dict[str, typing.Any]],
    labels: list[str],
    prob_prefix: str,
    dpi: int,
    fmt: str,
    version: str,
) -> float | None:
    """Compute ROC AUC (first label as positive), plot the curve, and return the AUC.

    Args:
        results: Result rows.
        labels: The two labels.
        prob_prefix: Column prefix (``"prob"`` for aggregate,
            ``"primary_prob"`` for primary).
        dpi: Plot resolution.
        fmt: Image format.
        version: ``"aggregate"`` or ``"primary"``.

    Returns:
        AUC value, or None if not computable.
    """
    y_true = [r["ground_truth"] for r in results]
    y_binary = [1 if gt == labels[0] else 0 for gt in y_true]
    scores = [r[f"{prob_prefix}_{labels[0]}"] for r in results]
    roc_auc_value = _compute_roc_auc(y_binary, scores)

    if roc_auc_value is not None:
        print(f"  ROC AUC ({version}): {roc_auc_value:.4f}")
    else:
        print(f"  ROC AUC ({version}): N/A (single class)")

    _plot_roc(y_binary, scores, roc_auc_value, dpi, fmt, version)

    return roc_auc_value


def _compute_roc_auc(
    y_binary: list[int],
    scores: list[float],
) -> float | None:
    """ROC AUC, or None if only one class is present."""
    if len(set(y_binary)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_binary, scores)
    return float(auc(fpr, tpr))


# --- Plot helpers --------------------------------------------------


def _plot_roc(
    y_binary: list[int],
    scores: list[float],
    roc_auc_value: float | None,
    dpi: int,
    fmt: str,
    version: str,
) -> None:
    """Save the ROC curve plot."""
    if roc_auc_value is None:
        return
    fpr, tpr, _ = roc_curve(y_binary, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc_value:.4f})")
    plt.plot(
        [0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random classifier"
    )
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.05))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: Localization ({version})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    filename = f"localization_roc_{version}.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    dpi: int,
    fmt: str,
    version: str,
) -> str:
    """Save confusion matrix heatmap.

    Returns:
        The filename of the saved plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    label_str = "/".join(labels)
    plt.title(f"Confusion Matrix: Localization ({label_str}, {version})")
    filename = f"localization_confusion_matrix_{version}.{fmt}"
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return filename


def _plot_logit_distribution(
    results: list[dict[str, typing.Any]],
    labels: list[str],
    dpi: int,
    fmt: str,
) -> None:
    """Save histogram of logsumexp(A) - logsumexp(B) split by ground truth (binary only)."""
    l0, l1 = labels[0], labels[1]
    diff_0 = [
        r[f"logsumexp_{l0}"] - r[f"logsumexp_{l1}"]
        for r in results
        if r["ground_truth"] == l0
    ]
    diff_1 = [
        r[f"logsumexp_{l0}"] - r[f"logsumexp_{l1}"]
        for r in results
        if r["ground_truth"] == l1
    ]

    plt.figure(figsize=(8, 5))
    plt.hist(
        diff_0,
        bins=20,
        alpha=0.6,
        label=f"GT: {l0} (target on {l0})",
        color="blue",
        edgecolor="black",
    )
    plt.hist(
        diff_1,
        bins=20,
        alpha=0.6,
        label=f"GT: {l1} (target on {l1})",
        color="red",
        edgecolor="black",
    )
    plt.xlabel(f"logsumexp({l0}) - logsumexp({l1})")
    plt.ylabel("Count")
    plt.title("Distribution of logsumexp Difference by Ground Truth")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.savefig(f"localization_logit_distribution.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
