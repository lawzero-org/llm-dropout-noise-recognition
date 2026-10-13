"""Zero-shot classification runner (paper Sec. 5): which perturbation was applied to one sentence."""

import functools
import itertools
import random
import typing

import numpy as np
import omegaconf
import transformers

from spe import core, generation, hooks
from spe.evaluation import classification as eval_classification
from spe.prompt_utils import (
    build_messages,
    build_messages_from_turns,
    compute_substring_token_range,
    print_token_map,
    validate_token_range,
)

# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------


def _build_classification_row(
    sample_id: int,
    perturbation: str,
    is_baseline: bool,
    permutation: tuple[str, ...],
    vdata: dict[str, typing.Any],
    generated_token_id: int,
    class_names: list[str],
    variant_token_ids: dict[str, int] | None = None,
) -> dict[str, typing.Any]:
    """Build a single result row for the multiclass experiment.

    Maps letter-level metrics (A/B) to class-level metrics
    (DROPOUT/NOISE) using the permutation.
    """
    class_to_letter = {permutation[i]: chr(65 + i) for i in range(len(permutation))}
    letter_to_class = {chr(65 + i): permutation[i] for i in range(len(permutation))}
    letters = [chr(65 + i) for i in range(len(permutation))]

    row: dict[str, typing.Any] = {
        "sample_id": sample_id,
        "perturbation": perturbation,
        "is_baseline": is_baseline,
        "option_order": "/".join(permutation),
        "generated_token_id": generated_token_id,
        "argmax_token": vdata["argmax_token"],
        "argmax_logit": vdata["argmax_logit"],
        "argmax_prob": vdata["argmax_prob"],
        "argmax_token_id": vdata["argmax_token_id"],
        "entropy": vdata["entropy"],
    }

    # Letter level metrics (position based)
    for letter in letters:
        row[f"sum_prob_{letter}"] = vdata[f"sum_prob_{letter}"]
        row[f"logsumexp_{letter}"] = vdata[f"logsumexp_{letter}"]
        row[f"primary_prob_{letter}"] = vdata[f"primary_prob_{letter}"]
        row[f"primary_logit_{letter}"] = vdata[f"primary_logit_{letter}"]
        row[f"primary_log_prob_{letter}"] = vdata[f"primary_log_prob_{letter}"]
        row[f"aggregate_log_prob_{letter}"] = vdata[f"aggregate_log_prob_{letter}"]

    # Class level metrics (semantic, mapped through the permutation)
    for cls in class_names:
        letter = class_to_letter[cls]
        row[f"sum_prob_{cls.lower()}"] = vdata[f"sum_prob_{letter}"]
        row[f"logsumexp_{cls.lower()}"] = vdata[f"logsumexp_{letter}"]
        row[f"primary_prob_{cls.lower()}"] = vdata[f"primary_prob_{letter}"]
        row[f"primary_logit_{cls.lower()}"] = vdata[f"primary_logit_{letter}"]
        row[f"primary_log_prob_{cls.lower()}"] = vdata[f"primary_log_prob_{letter}"]
        row[f"aggregate_log_prob_{cls.lower()}"] = vdata[f"aggregate_log_prob_{letter}"]

    # Logit diff: for each class X, logit(X) - logaddexp(logit(Y), logit(Z), ...)
    for cls in class_names:
        cls_lower = cls.lower()
        others_primary = [
            row[f"primary_logit_{c.lower()}"] for c in class_names if c != cls
        ]
        others_logsumexp = [
            row[f"logsumexp_{c.lower()}"] for c in class_names if c != cls
        ]
        row[f"logit_diff_{cls_lower}"] = row[
            f"primary_logit_{cls_lower}"
        ] - functools.reduce(np.logaddexp, others_primary)
        row[f"logsumexp_diff_{cls_lower}"] = row[
            f"logsumexp_{cls_lower}"
        ] - functools.reduce(np.logaddexp, others_logsumexp)

    # Pairwise logit diff: X_vs_Y = logit(X) - logit(Y)
    for i, cls_x in enumerate(class_names):
        for cls_y in class_names[i + 1 :]:
            x, y = cls_x.lower(), cls_y.lower()
            row[f"logit_diff_{x}_vs_{y}"] = (
                row[f"primary_logit_{x}"] - row[f"primary_logit_{y}"]
            )
            row[f"aggregate_logit_diff_{x}_vs_{y}"] = (
                row[f"logsumexp_{x}"] - row[f"logsumexp_{y}"]
            )

    # Predicted label (letter) and class for both aggregate and primary
    predicted_label_aggregate = max(letters, key=lambda l: row[f"sum_prob_{l}"])
    predicted_label_primary = max(letters, key=lambda l: row[f"primary_logit_{l}"])
    predicted_class_aggregate = letter_to_class[predicted_label_aggregate]
    predicted_class_primary = letter_to_class[predicted_label_primary]

    row["predicted_label_aggregate"] = predicted_label_aggregate
    row["predicted_label_primary"] = predicted_label_primary
    row["predicted_class_aggregate"] = predicted_class_aggregate
    row["predicted_class_primary"] = predicted_class_primary

    # Argmax prediction: map full-vocab argmax to a label/class (or OTHER)
    if variant_token_ids is not None:
        argmax_label = generation.argmax_to_label(
            vdata["argmax_token_id"],
            variant_token_ids,
            letters,
        )
        row["predicted_label_argmax"] = argmax_label
        row["predicted_class_argmax"] = letter_to_class.get(argmax_label, "OTHER")
        # Strict variant: the argmax must be exactly the primary token " L"
        argmax_label_primary = next(
            (
                l
                for l in letters
                if vdata["argmax_token_id"] == variant_token_ids[f" {l}"]
            ),
            "OTHER",
        )
        row["predicted_label_argmax_primary"] = argmax_label_primary
        row["predicted_class_argmax_primary"] = letter_to_class.get(
            argmax_label_primary, "OTHER"
        )
    else:
        row["predicted_label_argmax"] = "OTHER"
        row["predicted_class_argmax"] = "OTHER"
        row["predicted_label_argmax_primary"] = "OTHER"
        row["predicted_class_argmax_primary"] = "OTHER"

    # Ground truth letter: the letter that corresponds to the ground truth class
    row["ground_truth_letter"] = class_to_letter.get(perturbation, "")

    # Log probability of the correct answer
    ground_truth_cls = perturbation.lower()
    plp_key = f"primary_log_prob_{ground_truth_cls}"
    alp_key = f"aggregate_log_prob_{ground_truth_cls}"
    if plp_key in row:
        row["primary_log_prob_correct"] = row[plp_key]
    if alp_key in row:
        row["aggregate_log_prob_correct"] = row[alp_key]

    # Per variant raw values
    for key, value in vdata.items():
        if key.startswith("v_prob_") or key.startswith("v_logit_"):
            row[key] = value

    return row


def run_classification(
    cfg: omegaconf.DictConfig,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
) -> tuple[dict, list]:
    """Execute the zero-shot classification loop (paper Sec. 5).

    Each of the ``num_samples`` passes perturbs one sentence sampled from
    ``experiment.sentences_file`` and asks which perturbation was applied.
    Samples are split evenly across ``perturbation.active_perturbations``
    (default: every class; the paper runs one class at a time, e.g.
    ``["DROPOUT"]``). The option order is shuffled for each sample.

    Args:
        cfg: Resolved Hydra config.
        model: Loaded language model.
        tokenizer: Matching tokenizer.

    Returns:
        Tuple of (metrics dict, results list).
    """
    class_names = list(cfg.prompts.turns.class_names)
    labels = list(cfg.prompts.turns.labels)
    if sorted(class_names) != ["DROPOUT", "NOISE"] or len(labels) != 2:
        raise ValueError(
            "Zero-shot classification asks DROPOUT vs NOISE with two answer letters, got "
            f"class_names={class_names}, labels={labels}."
        )
    if cfg.perturbation.get("dropout_style", "post_sublayer") != "post_sublayer":
        raise ValueError(
            "Zero-shot classification supports only perturbation.dropout_style=post_sublayer "
            "(AIAYN-style dropout is implemented for localization only)."
        )

    # Resolve variant token IDs for all letter variants
    all_variants = []
    for letter in labels:
        all_variants += generation.letter_variants(letter)
    variant_token_ids = generation.resolve_token_ids_safe(tokenizer, all_variants)
    if not variant_token_ids:
        raise ValueError(
            "No letter variant resolved to a single token. Cannot run classification."
        )

    print(f"\nResolved variant token IDs: {variant_token_ids}")

    # Resolve and validate layer range once (fails fast on bad config).
    resolved_first_layer, resolved_last_layer = hooks.resolve_layer_range(
        model,
        cfg.perturbation.first_layer,
        cfg.perturbation.last_layer,
    )

    # Build display name mapping from aliases config (same pattern as ICL)
    use_aliases = cfg.aliases.display_names is not None
    if use_aliases:
        display_map: dict[str, str] = dict(cfg.aliases.display_names)
        alias_desc_map: dict[str, str] = {}
        if cfg.aliases.get("descriptions") is not None:
            alias_desc_map = dict(cfg.aliases.descriptions)
    else:
        display_map = {c: c for c in class_names}
        alias_desc_map = {}

    # Description pool: one description per class sampled for each prompt (canonical labels only)
    description_pool: dict[str, list[str]] | None = None
    if cfg.prompts.turns.get("description_pool") is not None:
        description_pool = {
            k: list(v) for k, v in cfg.prompts.turns.description_pool.items()
        }
        print(f"Description pool loaded: {list(description_pool.keys())}")

    # Prompt pool: load variant configs, validate schema.
    prompt_pool: list[dict[str, typing.Any]] | None = None
    if cfg.prompts.turns.get("prompt_pool") is not None:
        from spe.prompt_utils import load_prompt_pool

        prompt_pool = load_prompt_pool(
            list(cfg.prompts.turns.prompt_pool),
            class_names=class_names,
            labels=labels,
        )

    all_permutations = list(itertools.permutations(class_names))

    sentences_file = cfg.experiment.get("sentences_file", None)
    if sentences_file is None:
        raise ValueError(
            "Set experiment.sentences_file: the target sentences are sampled from it."
        )
    with open(sentences_file) as f:
        sentences_pool = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(sentences_pool)} sentences from {sentences_file}")

    num_samples = cfg.experiment.num_samples

    active_raw = cfg.perturbation.get("active_perturbations", None)
    active_classes = list(class_names) if active_raw is None else list(active_raw)
    for a in active_classes:
        if a not in class_names:
            raise ValueError(
                f"active_perturbations contains '{a}' which is not in "
                f"class_names {list(class_names)}."
            )

    samples_per_class = num_samples // len(active_classes)
    # Redistribute remainder to last class
    sample_counts: dict[str, int] = {c: samples_per_class for c in active_classes}
    sample_counts[active_classes[-1]] += num_samples - samples_per_class * len(
        active_classes
    )

    # --- Results table header ---
    print_agg = cfg.output.print_aggregates
    # Build class column headers with full names
    if print_agg:
        mc_cls_headers = [f"sumP({c.lower()})" for c in class_names]
    else:
        mc_cls_headers = [f"P({c.lower()})" for c in class_names]
    mc_cls_width = max(max(len(h) for h in mc_cls_headers), 9)
    mc_cls_cols = "  ".join(f"{h:>{mc_cls_width}}" for h in mc_cls_headers)
    table_width = (
        59 + 31 + 2 + mc_cls_width * len(class_names) + 2 * (len(class_names) - 1)
    )
    # Build letter column headers dynamically
    if print_agg:
        ltr_headers = [f"sumP({l})" for l in labels]
    else:
        ltr_headers = [f'P(" {l}")' for l in labels]
    ltr_cols = "  ".join(f"{h:>9}" for h in ltr_headers)

    counts_desc = "  ".join(f"{sample_counts[c]} {c.lower()}" for c in active_classes)

    print(f"\n{'=' * table_width}")
    print(
        f"CLASSIFICATION  |  {counts_desc} = {num_samples} rows  |  "
        f"dropout_rate={cfg.perturbation.dropout_rate}  |  noise_std={cfg.perturbation.noise_std}"
    )
    print(
        f"Variant IDs: {variant_token_ids}  |  Layer range: {resolved_first_layer}..{resolved_last_layer}"
    )
    print(f"{'=' * table_width}")

    def _print_table_header() -> None:
        print(
            f"{'#':>4}  {'Type':<10}  {'Order':<25}  {'Argmax':<12}  "
            f"{ltr_cols}  {mc_cls_cols}"
        )

    _print_table_header()
    print(f"{'-' * table_width}")

    results: list[dict[str, typing.Any]] = []
    num_printed_prompts = 0
    max_printed_prompts = min(3, num_samples)

    def _make_format_kwargs(
        perm: tuple[str, ...],
        sampled_descriptions: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build format kwargs for prompt templates.

        Aliased labels take the alias descriptions (none for the paper's aliases); canonical labels take a
        description sampled from the description pool, or ``core.describe_option`` without a pool.
        """
        kwargs: dict[str, str] = {}
        for i, cls in enumerate(perm):
            label = display_map[cls]
            if use_aliases:
                desc = alias_desc_map.get(cls, "")
            elif sampled_descriptions and cls in sampled_descriptions:
                desc = sampled_descriptions[cls]
            else:
                desc = core.describe_option(cls)
            if desc:
                kwargs[f"option_{chr(97 + i)}"] = f"{label} {desc}"
            else:
                kwargs[f"option_{chr(97 + i)}"] = label
        # Add type_a/type_b for prompts that reference perturbation names in their intro
        for j, pcls in enumerate(class_names):
            kwargs[f"type_{chr(97 + j)}"] = display_map[pcls]
        return kwargs

    def _print_row(
        idx: int, label: str, perm: tuple[str, ...], vdata: dict, row: dict
    ) -> None:
        order_str = "/".join(perm)
        prob_prefix = "sum_prob" if print_agg else "primary_prob"
        ltr_vals = "  ".join(f"{vdata[f'{prob_prefix}_{l}']:>9.6f}" for l in labels)
        cls_vals = "  ".join(
            f"{row[f'{prob_prefix}_{c.lower()}']:>{mc_cls_width}.6f}"
            for c in class_names
        )
        print(
            f"{idx:>4}  {label:<10}  {order_str:<25}  {vdata['argmax_token']:<12}  "
            f"{ltr_vals}  {cls_vals}"
        )

    def _print_prompt_and_token_maps(
        messages: list, first_token: int, last_token: int
    ) -> None:
        print(f"\n{'=' * 60}")
        print("PROMPT (option order varies across samples)")
        print(f"{'=' * 60}")
        for msg in messages:
            print(f"[{msg['role']}]: {msg['content']}")
        print(f"{'=' * 60}")
        for ptype in class_names:
            print_token_map(
                tokenizer,
                messages,
                first_token=first_token,
                last_token=last_token,
                label=ptype,
                enable_thinking=cfg.model.thinking,
            )
        # Re-print table header after prompt block so rows align with headers
        print(f"\n{'=' * table_width}")
        _print_table_header()
        print(f"{'-' * table_width}")

    # Build flat assignment list: [CLASS_A]*n_a + [CLASS_B]*n_b + ..., shuffled
    perturbation_assignments = []
    for cls in active_classes:
        perturbation_assignments += [cls] * sample_counts[cls]
    random.shuffle(perturbation_assignments)

    system_content = cfg.prompts.system.content

    for i in range(num_samples):
        perturbation_choice = perturbation_assignments[i]

        # Random permutation for option order
        perm = random.choice(all_permutations)

        # Sample descriptions (only in canonical mode with description_pool)
        sampled_descriptions: dict[str, str] | None = None
        if description_pool is not None:
            sampled_descriptions = {
                cls: random.choice(description_pool[cls])
                for cls in class_names
                if cls in description_pool
            }

        format_kwargs = _make_format_kwargs(perm, sampled_descriptions)
        sentence = random.choice(sentences_pool)
        format_kwargs["sentence"] = sentence

        # Sample prompt variant (or use default turns)
        selected_variant: dict[str, typing.Any] | None = None
        if prompt_pool is not None:
            selected_variant = random.choice(prompt_pool)
            messages = build_messages_from_turns(
                system_content,
                selected_variant["turns"],
                format_kwargs=format_kwargs,
            )
        else:
            messages = build_messages(cfg, format_kwargs=format_kwargs)

        first_token, last_token = compute_substring_token_range(
            tokenizer,
            messages,
            sentence,
            enable_thinking=cfg.model.thinking,
        )
        validate_token_range(first_token, last_token)

        if num_printed_prompts < max_printed_prompts:
            _print_prompt_and_token_maps(messages, first_token, last_token)
            num_printed_prompts += 1

        if perturbation_choice == "DROPOUT":
            hook_fn = hooks.create_dropout_hook(
                cfg.perturbation.dropout_rate,
                first_token=first_token,
                last_token=last_token,
            )
        else:
            hook_fn = hooks.create_noise_hook(
                cfg.perturbation.noise_std,
                first_token=first_token,
                last_token=last_token,
            )
        _response, logits, generated_token_id = core.run_single_sample(
            model,
            tokenizer,
            messages,
            hook_fn=hook_fn,
            hook_target=cfg.perturbation.hook_target,
            first_layer=cfg.perturbation.first_layer,
            last_layer=cfg.perturbation.last_layer,
            enable_thinking=cfg.model.thinking,
        )

        vdata = generation.extract_n_letter_variant_data(
            logits,
            variant_token_ids,
            tokenizer,
            labels,
        )
        row = _build_classification_row(
            sample_id=i,
            perturbation=perturbation_choice,
            is_baseline=False,
            permutation=perm,
            vdata=vdata,
            generated_token_id=generated_token_id,
            class_names=class_names,
            variant_token_ids=variant_token_ids,
        )
        row["sentence"] = sentence
        if selected_variant is not None:
            row["prompt_variant"] = selected_variant["name"]
        if sampled_descriptions is not None:
            for cls, desc in sampled_descriptions.items():
                row[f"description_{cls.lower()}"] = desc
        elif use_aliases and alias_desc_map:
            for cls, desc in alias_desc_map.items():
                row[f"description_{cls.lower()}"] = desc
        results.append(row)
        _print_row(i, perturbation_choice, perm, vdata, row)

    print(f"{'=' * table_width}")

    # --- Variant summary ---
    group_rows = {
        c: [r for r in results if r["perturbation"] == c] for c in active_classes
    }
    group_headers = "  ".join(f"{f'{c} P':>24}" for c in active_classes)
    summary_width = 12 + 24 * len(active_classes)

    print(f"\n{'=' * summary_width}")
    print(f"VARIANT SUMMARY (letter level, all {'/'.join(labels)} variants)")
    print(f"{'=' * summary_width}")
    print(f"{'Variant':<10}  {group_headers}")
    print(f"{'-' * summary_width}")

    for variant in variant_token_ids:
        parts = f"{variant:<10}  "
        for c in active_classes:
            g_probs = [r.get(f"v_prob_{variant}", 0.0) for r in group_rows[c]]
            parts += f"{np.mean(g_probs):>10.6f} ± {np.std(g_probs):<10.6f}  "
        print(parts.rstrip())

    print(f"{'=' * summary_width}")

    metrics = eval_classification.compute_and_plot(
        results,
        class_names=class_names,
        labels=labels,
        dpi=cfg.output.dpi,
        fmt=cfg.output.format,
    )
    return metrics, results
