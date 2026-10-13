"""Localization experiment runner (paper Sec. 4 and its appendix).

``run_localization``: two sentences, one of them perturbed with the target
perturbation (dropout or noise); the model is asked which one.
"""

import functools
import random
import string
import typing

import numpy as np
import omegaconf
import torch
import transformers

from spe import aiayn_hooks, generation, hooks, sentence_utils
from spe.evaluation import localization as loc_eval
from spe.prompt_utils import (
    build_messages,
    build_messages_from_turns,
    compute_substring_token_range,
    print_multi_range_token_map,
    validate_token_range,
)

# ---------------------------------------------------------------------------
# Localization: which of two sentences was perturbed
# ---------------------------------------------------------------------------


def _validate_localization_config(cfg: omegaconf.DictConfig) -> None:
    """Two sentences: one with the target perturbation, the other unperturbed.

    Raises:
        ValueError: If the config asks for another shape.
    """
    p = cfg.perturbation
    shape = (
        p.num_sentences,
        p.get("num_bg_dropout", 0),
        p.get("num_bg_noise", 0),
        p.get("num_bg_nothing", 0),
    )
    if shape != (2, 0, 0, 1):
        raise ValueError(
            "Localization uses two sentences, one perturbed and one unperturbed: set num_sentences=2, "
            f"num_bg_nothing=1, num_bg_dropout=num_bg_noise=0 (got {shape})."
        )
    if p.target_perturbation not in ("DROPOUT", "NOISE"):
        raise ValueError(
            f"target_perturbation must be DROPOUT or NOISE, got {p.target_perturbation!r}"
        )


def run_localization(
    cfg: omegaconf.DictConfig,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
) -> tuple[dict, list]:
    """Execute the localization experiment loop.

    The target perturbation is applied to one of the two sentences, chosen
    at random for each sample; the other sentence is unperturbed.

    Args:
        cfg: Resolved Hydra config.
        model: Loaded language model.
        tokenizer: Matching tokenizer.

    Returns:
        Tuple of (metrics dict, results list).
    """
    _validate_localization_config(cfg)

    dropout_style = cfg.perturbation.get("dropout_style", "post_sublayer")

    # Prompt pool: load variant configs if present.
    prompt_pool: list[dict[str, typing.Any]] | None = None
    if cfg.prompts.turns.get("prompt_pool") is not None:
        from spe.prompt_utils import load_prompt_pool

        prompt_pool = load_prompt_pool(
            list(cfg.prompts.turns.prompt_pool),
            extra_fields=["content_correct_group"],
        )

    sentence_n_tokens = cfg.experiment.get("sentence_n_tokens", None)
    sentences_file = cfg.experiment.get("sentences_file", None)
    hardcoded_sentences: list[str] | None = None

    # Grouped sentences: a list of lists. One sentence is sampled from
    # each group, guaranteeing one per category (e.g. one animal, one
    # city). The number of groups must equal num_sentences.
    sentence_groups: list[list[str]] | None = None
    if "sentence_groups" in cfg.prompts.turns:
        sentence_groups = [list(g) for g in cfg.prompts.turns.sentence_groups]

    # Priority: sentences_file > sentence_groups (control prompts) > sentence_n_tokens (generator)
    if sentences_file is not None:
        with open(sentences_file) as f:
            hardcoded_sentences = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(hardcoded_sentences)} sentences from {sentences_file}")
    if (
        hardcoded_sentences is None
        and sentence_groups is None
        and sentence_n_tokens is None
    ):
        raise ValueError(
            "Set experiment.sentences_file or experiment.sentence_n_tokens, "
            "or use a control prompt with sentence_groups"
        )
    same_sentence = cfg.perturbation.get("same_sentence", False)
    num_samples = cfg.experiment.num_samples
    num_sentences = cfg.perturbation.get("num_sentences", 2)
    sentence_rng = random.Random(cfg.experiment.seed)
    labels = list(string.ascii_uppercase[:num_sentences])

    if sentence_groups is not None and len(sentence_groups) != num_sentences:
        raise ValueError(
            f"sentence_groups has {len(sentence_groups)} groups but "
            f"num_sentences={num_sentences}. They must match."
        )

    target_perturbation = cfg.perturbation.target_perturbation

    # Resolve variant token IDs for each label
    variant_ids = {}
    for lbl in labels:
        resolved = generation.resolve_token_ids_safe(
            tokenizer,
            generation.letter_variants(lbl),
        )
        variant_ids[lbl] = list(resolved.values())

    resolved_first_layer, resolved_last_layer = hooks.resolve_layer_range(
        model,
        cfg.perturbation.first_layer,
        cfg.perturbation.last_layer,
    )

    # Generic label for the prompt ("a perturbation"), so that it never names the perturbation
    prompt_label = cfg.perturbation.get("prompt_label", None)

    # --- Header ---
    print_agg = cfg.output.print_aggregates
    _print_loc_header(
        num_samples,
        num_sentences,
        same_sentence,
        target_perturbation,
        cfg.perturbation.dropout_rate,
        cfg.perturbation.get("noise_std", 0.01),
        labels,
        variant_ids,
        resolved_first_layer,
        resolved_last_layer,
    )
    _print_loc_table_header(labels, print_aggregates=print_agg)

    results: list[dict] = []
    num_printed_prompts = 0
    max_printed_prompts = min(3, num_samples)
    aiayn_positions: dict[str, bool] | None = None
    if dropout_style == "aiayn":
        aiayn_positions = aiayn_hooks.resolve_positions(cfg)

    for i in range(num_samples):
        # group_order[j] = which group index ended up at position j
        # (e.g. group_order=[1,0] means group 1 is at position A,
        # group 0 at position B). None when sentence_groups is unused.
        group_order: list[int] | None = None
        if sentence_groups is not None:
            group_order = list(range(len(sentence_groups)))
            sampled = [random.choice(g) for g in sentence_groups]
            combined = list(zip(group_order, sampled, strict=False))
            random.shuffle(combined)
            group_order, sampled = [list(x) for x in zip(*combined, strict=False)]
        elif hardcoded_sentences is not None:
            sampled = sentence_utils.sample_or_generate_sentences(
                num_sentences,
                same_sentence,
                sentences_pool=hardcoded_sentences,
            )
        else:
            sampled = sentence_utils.sample_or_generate_sentences(
                num_sentences,
                same_sentence,
                sentence_n_tokens=sentence_n_tokens,
                tokenizer=tokenizer,
                rng=sentence_rng,
            )
        target_position = random.choice(labels)
        position_to_type = {
            lbl: target_perturbation if lbl == target_position else "NOTHING"
            for lbl in labels
        }
        sample_target = target_perturbation
        sample_ground_truth = target_position

        format_kwargs = _build_format_kwargs(
            labels,
            sampled,
            num_sentences,
            target_perturbation_name=prompt_label or sample_target,
        )

        # Sample prompt variant (or use default turns)
        selected_variant: dict[str, typing.Any] | None = None
        if prompt_pool is not None:
            selected_variant = random.choice(prompt_pool)
            system_content = cfg.prompts.system.content
            messages = build_messages_from_turns(
                system_content,
                selected_variant["turns"],
                format_kwargs=format_kwargs,
            )
        else:
            messages = build_messages(cfg, format_kwargs=format_kwargs)

        # Compute token ranges for ALL sentences (with anchors for disambiguation)
        token_ranges: dict[str, tuple[int, int]] = {}
        for lbl_idx, lbl in enumerate(labels):
            sentence = sampled[lbl_idx]
            anchor = _build_anchor(lbl, same_sentence)
            first_tok, last_tok = compute_substring_token_range(
                tokenizer,
                messages,
                sentence,
                anchor=anchor,
                enable_thinking=cfg.model.thinking,
            )
            validate_token_range(first_tok, last_tok)
            token_ranges[lbl] = (first_tok, last_tok)

        # Build perturbation entries for all sentences
        perturbation_entries: list[dict[str, typing.Any]] = []
        for lbl in labels:
            first_tok, last_tok = token_ranges[lbl]
            perturbation_entries.append(
                {
                    "type": position_to_type[lbl],
                    "first_token": first_tok,
                    "last_token": last_tok,
                    "dropout_rate": cfg.perturbation.dropout_rate,
                    "noise_std": cfg.perturbation.get("noise_std", 0.01),
                    "label": f"sentence_{lbl.lower()}",
                }
            )

        if num_printed_prompts < max_printed_prompts:
            _print_first_sample(messages)
            print_multi_range_token_map(
                tokenizer,
                messages,
                perturbation_entries,
                enable_thinking=cfg.model.thinking,
            )
            table_width = 120 if print_agg else 80
            print(f"\n{'=' * table_width}")
            _print_loc_table_header(labels, print_aggregates=print_agg)
            num_printed_prompts += 1

        # Create hook(s) and run
        if dropout_style == "aiayn":
            _response, logits, _token_id = aiayn_hooks.run_single_sample_multi(
                model,
                tokenizer,
                messages,
                perturbation_entries=perturbation_entries,
                hook_target=cfg.perturbation.hook_target,
                first_layer=cfg.perturbation.first_layer,
                last_layer=cfg.perturbation.last_layer,
                aiayn_positions=aiayn_positions,
                enable_thinking=cfg.model.thinking,
            )
            all_logits = None
        else:
            hook_fn = hooks.create_multi_perturbation_hook(perturbation_entries)
            handles = hooks.register_hooks(
                model,
                hook_fn,
                target=cfg.perturbation.hook_target,
                first_layer=cfg.perturbation.first_layer,
                last_layer=cfg.perturbation.last_layer,
            )
            try:
                _response, logits, _token_id, all_logits = (
                    generation.generate_single_token(
                        model,
                        tokenizer,
                        messages,
                        enable_thinking=cfg.model.thinking,
                        return_all_logits=True,
                    )
                )
            finally:
                hooks.remove_hooks(handles)

        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)
        log_norm = torch.logsumexp(logits, dim=-1).item()

        # Compute sentence log probs from the full logits
        sentence_log_probs: dict[str, float] = {}
        if all_logits is not None:
            is_prefill = messages[-1]["role"] == "assistant"
            text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=not is_prefill,
                continue_final_message=is_prefill,
                enable_thinking=cfg.model.thinking,
            )
            input_ids = tokenizer(text, return_tensors="pt").input_ids[0]  # type: ignore[call-overload]
            for lbl in labels:
                first_tok, last_tok = token_ranges[lbl]
                sentence_log_probs[lbl] = generation.compute_token_range_log_prob(
                    all_logits,
                    input_ids,
                    first_tok,
                    last_tok,
                )
            del all_logits

        # Compute aggregated prob and logsumexp for each label
        label_probs = {}
        label_logsumexp = {}
        for lbl in labels:
            ids = variant_ids[lbl]
            label_probs[lbl] = sum(probs[tid].item() for tid in ids)
            label_logsumexp[lbl] = torch.logsumexp(logits[ids], dim=0).item()

        # Compute primary token prob and logit for each label. The primary
        # variant " L" must be a single token, otherwise the metric is meaningless.
        label_primary_ids: dict[str, int] = {}
        label_primary_probs: dict[str, float] = {}
        label_primary_logits: dict[str, float] = {}
        for lbl in labels:
            primary_str = f" {lbl}"
            tid = generation.resolve_token_ids_safe(tokenizer, [primary_str])[
                primary_str
            ]
            label_primary_ids[lbl] = tid
            label_primary_probs[lbl] = probs[tid].item()
            label_primary_logits[lbl] = logits[tid].item()

        predicted_aggregate = max(labels, key=lambda l: label_probs[l])
        predicted_primary = max(labels, key=lambda l: label_primary_logits[l])

        top_token_id = int(logits.argmax().item())
        top_token = repr(tokenizer.decode([top_token_id]))  # type: ignore[attr-defined]
        top_logit = logits[top_token_id].item()
        top_prob = probs[top_token_id].item()

        # Argmax prediction: map full-vocab argmax to a label (or OTHER).
        # ``predicted_argmax`` accepts any casing/space variant of the letter;
        # ``predicted_argmax_primary`` only the primary variant " L".
        predicted_argmax = "OTHER"
        for lbl in labels:
            if top_token_id in variant_ids[lbl]:
                predicted_argmax = lbl
                break
        predicted_argmax_primary = next(
            (lbl for lbl in labels if top_token_id == label_primary_ids[lbl]), "OTHER"
        )

        entropy = generation.entropy_from_logits(logits)

        row = {
            "sample_id": i,
            "ground_truth": sample_ground_truth,
            "predicted": predicted_aggregate,
            "predicted_aggregate": predicted_aggregate,
            "predicted_primary": predicted_primary,
            "predicted_argmax": predicted_argmax,
            "predicted_argmax_primary": predicted_argmax_primary,
            "argmax_token": top_token,
            "argmax_token_id": top_token_id,
            "argmax_logit": top_logit,
            "argmax_prob": top_prob,
            "entropy": entropy,
            "same_sentence": same_sentence,
            "target_perturbation": sample_target,
            "first_token": token_ranges[sample_ground_truth][0],
            "last_token": token_ranges[sample_ground_truth][1],
            "sentence_n_tokens": sentence_n_tokens,
        }
        if selected_variant is not None:
            row["prompt_variant"] = selected_variant["name"]
        if group_order is not None:
            for lbl_idx, lbl in enumerate(labels):
                row[f"sentence_group_{lbl.lower()}"] = group_order[lbl_idx]
            # Content ground truth: which label has the group the
            # question asked about (e.g. "animals" = group 0).
            ccg = (
                selected_variant.get("content_correct_group")
                if selected_variant
                else None
            )
            if ccg is not None:
                row["content_ground_truth"] = labels[group_order.index(ccg)]
        for lbl in labels:
            row[f"prob_{lbl}"] = label_probs[lbl]
            row[f"logsumexp_{lbl}"] = label_logsumexp[lbl]
            row[f"primary_prob_{lbl}"] = label_primary_probs[lbl]
            row[f"primary_logit_{lbl}"] = label_primary_logits[lbl]
            row[f"primary_log_prob_{lbl}"] = label_primary_logits[lbl] - log_norm
            row[f"aggregate_log_prob_{lbl}"] = label_logsumexp[lbl] - log_norm
            row[f"sentence_{lbl.lower()}"] = sampled[labels.index(lbl)]
            row[f"n_perturbed_tokens_{lbl.lower()}"] = (
                token_ranges[lbl][1] - token_ranges[lbl][0] + 1
            )
            if lbl in sentence_log_probs:
                row[f"sentence_log_prob_{lbl.lower()}"] = sentence_log_probs[lbl]
            row[f"perturbation_{lbl.lower()}"] = position_to_type[lbl].lower()

        # Per-perturbation-type log probabilities
        for lbl in labels:
            ptype = position_to_type[lbl].lower()
            row[f"primary_log_prob_{ptype}"] = label_primary_logits[lbl] - log_norm
            row[f"aggregate_log_prob_{ptype}"] = label_logsumexp[lbl] - log_norm

        # Logit diff: config target position vs logaddexp of all alternatives
        others_primary = [
            label_primary_logits[lbl] for lbl in labels if lbl != target_position
        ]
        others_lse = [label_logsumexp[lbl] for lbl in labels if lbl != target_position]
        row["logit_diff"] = label_primary_logits[target_position] - functools.reduce(
            np.logaddexp, others_primary
        )
        row["logsumexp_diff"] = label_logsumexp[target_position] - functools.reduce(
            np.logaddexp, others_lse
        )

        # Logit diff: asked-about (perturbed) position vs the alternatives; equals logit_diff / logsumexp_diff.
        others_primary_corr = [
            label_primary_logits[lbl] for lbl in labels if lbl != sample_ground_truth
        ]
        others_lse_corr = [
            label_logsumexp[lbl] for lbl in labels if lbl != sample_ground_truth
        ]
        row["logit_diff_correct_vs_incorrect"] = label_primary_logits[
            sample_ground_truth
        ] - functools.reduce(np.logaddexp, others_primary_corr)
        row["logsumexp_diff_correct_vs_incorrect"] = label_logsumexp[
            sample_ground_truth
        ] - functools.reduce(np.logaddexp, others_lse_corr)

        # Log-probability of the correct answer: log P(correct)
        row["primary_log_prob_correct"] = (
            label_primary_logits[sample_ground_truth] - log_norm
        )
        row["aggregate_log_prob_correct"] = (
            label_logsumexp[sample_ground_truth] - log_norm
        )

        # log(1 - P(correct)): how much probability mass is NOT on the correct answer
        p_correct = label_primary_probs[sample_ground_truth]
        row["primary_log_one_minus_p_correct"] = np.log1p(-p_correct)
        p_correct_agg = label_probs[sample_ground_truth]
        row["aggregate_log_one_minus_p_correct"] = np.log1p(-p_correct_agg)

        # Joint log-prob of the perturbed sentence vs. the unperturbed (NOTHING)
        # sentence, both read off the same forward pass.
        if len(sentence_log_probs) == len(labels):
            perturbed_pos = next(l for l in labels if position_to_type[l] != "NOTHING")
            unperturbed_pos = next(
                l for l in labels if position_to_type[l] == "NOTHING"
            )
            perturbed_lp = sentence_log_probs[perturbed_pos]
            unperturbed_lp = sentence_log_probs[unperturbed_pos]
            row["perturbed_lp_lower"] = int(perturbed_lp < unperturbed_lp)
            row["perturbed_unperturbed_logprob_delta"] = unperturbed_lp - perturbed_lp

        results.append(row)

        displayed_pred = predicted_aggregate if print_agg else predicted_primary
        _print_loc_table_row(
            i,
            sample_ground_truth,
            displayed_pred,
            labels,
            label_probs,
            label_logsumexp,
            top_token,
            top_logit,
            top_prob,
            print_aggregates=print_agg,
            label_primary_probs=label_primary_probs,
            sample_target=sample_target,
            position_to_type=position_to_type,
        )

    table_width = 120 if print_agg else 80
    print(f"{'=' * table_width}")

    metrics = loc_eval.compute_and_plot(
        results,
        labels=labels,
        dpi=cfg.output.dpi,
        fmt=cfg.output.format,
    )
    metrics["target_perturbation"] = target_perturbation
    return metrics, results


_PERTURBATION_ABBREV = {"DROPOUT": "DR", "NOISE": "NS", "NOTHING": "NT"}


# --- Localization helpers ---


def _build_anchor(
    label: str,
    same_sentence: bool,
) -> str | None:
    """Return an anchor prefix for same sentence disambiguation."""
    if not same_sentence:
        return None
    return f"Sentence {label}: "


def _build_format_kwargs(
    labels: list[str],
    sampled: list[str],
    num_sentences: int,
    target_perturbation_name: str = "DROPOUT",
) -> dict[str, str]:
    """Build the format_kwargs dict for prompt template substitution."""
    parts = [f"Sentence {lbl}: {s}" for lbl, s in zip(labels, sampled, strict=False)]
    sentences_block = "\n\n".join(parts)
    opts = [f'"the answer is: {lbl}"' for lbl in labels]
    answer_instruction = "Answer only " + ", ".join(opts[:-1]) + f", or {opts[-1]}."

    return {
        "sentences_block": sentences_block,
        "answer_instruction": answer_instruction,
        "num_sentences": str(num_sentences),
        "target_perturbation_name": target_perturbation_name,
    }


def _print_loc_header(
    num_samples: int,
    num_sentences: int,
    same_sentence: bool,
    target_perturbation: str,
    dropout_rate: float,
    noise_std: float,
    labels: list[str],
    variant_ids: dict[str, list[int]],
    first_layer: int,
    last_layer: int,
) -> None:
    print(f"\n{'=' * 120}")
    print(
        f"LOCALIZATION  |  {num_samples} samples  |  "
        f"{num_sentences} sentences ({'/'.join(labels)})  |  "
        f"same_sentence={same_sentence}  |  "
        f"target={target_perturbation}  |  "
        f"dropout_rate={dropout_rate}  |  noise_std={noise_std}"
    )
    variant_summary = "  |  ".join(
        f"{lbl} variants: {len(variant_ids[lbl])} IDs" for lbl in labels
    )
    print(f"{variant_summary}  |  Layer range: {first_layer}..{last_layer}")
    print(f"{'=' * 120}")


def _print_loc_table_header(labels: list[str], print_aggregates: bool = True) -> None:
    type_cols = "  ".join(f"{l:>2}" for l in labels)
    if print_aggregates:
        prob_cols = "  ".join(f"{'sumP(' + l + ')':>10}" for l in labels)
        lse_cols = "  ".join(f"{'logsumexp(' + l + ')':>14}" for l in labels)
        print(
            f"{'#':>4}  {'Ask':>3}  {type_cols}  {'GT':>3}  {'Pred':>4}  "
            f"{prob_cols}  {lse_cols}  "
            f"{'Argmax':>8}  {'Argmax logit':>12}  {'Argmax prob':>11}"
        )
    else:
        q = '"'
        prob_cols = "  ".join(
            f"{'P(' + q + ' ' + l + ' ' + q + ')':>10}" for l in labels
        )
        print(
            f"{'#':>4}  {'Ask':>3}  {type_cols}  {'GT':>3}  {'Pred':>4}  "
            f"{prob_cols}  "
            f"{'Argmax':>8}  {'Argmax logit':>12}  {'Argmax prob':>11}"
        )
    table_width = 120 if print_aggregates else 80
    print(f"{'-' * table_width}")


def _print_loc_table_row(
    i: int,
    gt: str,
    predicted: str,
    labels: list[str],
    label_probs: dict[str, float],
    label_logits: dict[str, float],
    top_token: str,
    top_logit: float,
    top_prob: float,
    print_aggregates: bool = True,
    label_primary_probs: dict[str, float] | None = None,
    sample_target: str = "",
    position_to_type: dict[str, str] | None = None,
) -> None:
    ask = _PERTURBATION_ABBREV.get(sample_target, sample_target[:2])
    type_cols = (
        "  ".join(
            f"{_PERTURBATION_ABBREV.get(position_to_type[l], '??'):>2}" for l in labels
        )
        if position_to_type
        else "  ".join(f"{'??':>2}" for _ in labels)
    )
    if print_aggregates:
        prob_cols = "  ".join(f"{label_probs[l]:>10.2e}" for l in labels)
        logit_cols = "  ".join(f"{label_logits[l]:>14.2f}" for l in labels)
        print(
            f"{i + 1:>4}  {ask:>3}  {type_cols}  {gt:>3}  {predicted:>4}  "
            f"{prob_cols}  {logit_cols}  "
            f"{top_token:>8}  {top_logit:>12.2f}  {top_prob:>11.2e}"
        )
    else:
        probs = label_primary_probs if label_primary_probs is not None else label_probs
        prob_cols = "  ".join(f"{probs[l]:>10.2e}" for l in labels)
        print(
            f"{i + 1:>4}  {ask:>3}  {type_cols}  {gt:>3}  {predicted:>4}  "
            f"{prob_cols}  "
            f"{top_token:>8}  {top_logit:>12.2f}  {top_prob:>11.2e}"
        )


def _print_first_sample(messages: list[dict[str, str]]) -> None:
    """Print prompt for the first sample."""
    print(f"\n{'=' * 60}")
    print("PROMPT (sample 1, sentences vary across samples)")
    print(f"{'=' * 60}")
    for msg in messages:
        print(f"[{msg['role']}]: {msg['content']}")
    print(f"{'=' * 60}")
