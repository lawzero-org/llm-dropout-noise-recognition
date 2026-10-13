"""ICL teaching experiment runner (paper Sec. 6).

Teaches the model what each perturbation type "feels like" through
labeled in context examples (correct or swapped labels), then tests
whether the model can classify the perturbation of a new sentence.
"""

import functools
import itertools
import random
import typing

import numpy as np
import omegaconf
import transformers

from spe import core, generation, hooks, sentence_utils
from spe.evaluation import icl as eval_icl
from spe.prompt_utils import (
    compute_ordered_substring_token_ranges,
    print_multi_range_token_map,
)

# Fields every ICL prompt variant must define (see conf/prompts/turns/icl/main_zs.yaml).
TEMPLATE_FIELDS = (
    "intro_user",
    "intro_assistant",
    "sentence_user",
    "label_assistant",
    "test_user",
    "test_assistant",
    "question_user",
    "answer_prefill",
)


def build_pair_schedule(
    class_names: list[str],
    num_pairs: int,
) -> list[str]:
    """Build a teaching schedule with randomized pairs.

    Each pair contains one entry for each class. The order
    within each pair is shuffled randomly.

    Args:
        class_names: List of class names
            (e.g. ``["DROPOUT", "NOISE"]``).
        num_pairs: Number of teaching pairs.

    Returns:
        Flat list of class names, one for each teaching
        example. Length = ``num_pairs * len(class_names)``.
    """
    schedule: list[str] = []
    for _ in range(num_pairs):
        pair = list(class_names)
        random.shuffle(pair)
        schedule.extend(pair)
    return schedule


def _expand_pair_sentences(
    unique_sentences: list[str],
    num_classes: int,
) -> list[str]:
    """Expand pair-level sentences to per-slot sentences.

    Given ``num_pairs + 1`` unique sentences (one per pair + test),
    repeat each pair sentence ``num_classes`` times so the result
    has ``num_pairs * num_classes + 1`` entries aligned with the
    teaching schedule (which groups by pair).
    """
    expanded = []
    for s in unique_sentences[:-1]:
        expanded.extend([s] * num_classes)
    expanded.append(unique_sentences[-1])
    return expanded


def build_icl_messages(
    cfg: omegaconf.DictConfig,
    schedule: list[str],
    sentences: list[str],
    option_order: tuple[str, ...],
    template_overrides: dict[str, str] | None = None,
    sampled_descriptions: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build the full ICL teaching message list from templates.

    Constructs the conversation:
    - System prompt
    - Intro user/assistant turns
    - Teaching pairs (sentence user + label assistant) for each schedule entry
    - Test sentence user + test assistant
    - Question user
    - Answer prefill assistant

    Args:
        cfg: Resolved Hydra config.
        schedule: Teaching schedule (list of class names for each example).
        sentences: Sentences to use. First ``len(schedule)`` are for teaching,
            the next one is for the test.
        option_order: Tuple of class names in option order (e.g. ``("DROPOUT", "NOISE")``).
        template_overrides: Optional dict of template field overrides from a
            prompt pool variant. Keys are field names like ``"intro_user"``,
            ``"sentence_user"``, etc. When provided, these replace the
            corresponding fields from ``cfg.prompts.turns``.
        sampled_descriptions: Optional per-class description strings sampled
            from the description pool. When provided and aliases are not
            active, these replace the default ``core.describe_option()``
            descriptions in the question options.

    Returns:
        List of message dicts ready for ``tokenizer.apply_chat_template``.
    """
    turns_cfg = cfg.prompts.turns
    class_names = list(turns_cfg.class_names)

    def _t(field: str) -> str:
        """Resolve a template field: override if provided, else from config."""
        if template_overrides and field in template_overrides:
            return template_overrides[field]
        return getattr(turns_cfg, field)

    # Build display name mapping from aliases config
    use_aliases = cfg.aliases.display_names is not None
    if use_aliases:
        display_map: dict[str, str] = dict(cfg.aliases.display_names)
    else:
        display_map = {c: c for c in class_names}

    # Determine labels for each teaching example
    teaching_labels = list(schedule)  # default: label matches perturbation type

    if turns_cfg.swap_labels:
        # the label names the other perturbation than the one applied
        label_map = {class_names[0]: class_names[1], class_names[1]: class_names[0]}
        teaching_labels = [label_map[s] for s in schedule]

    messages = [{"role": "system", "content": cfg.prompts.system.content}]

    # Intro turns: build kwargs dynamically for however many classes
    intro_kwargs = {
        f"type_{chr(97 + i)}": display_map[cls] for i, cls in enumerate(class_names)
    }
    # Optional k-conditional fragments (templates mirroring the zero-shot prompt, e.g. icl/main_zs):
    # ``{examples_intro}`` announces the examples and ``{question_scope}`` points the question at
    # the last sentence; both render empty at k=0 so that the k=0 prompt equals the zero-shot one.
    n_teaching = len(schedule)
    intro_kwargs["examples_intro"] = (
        ""
        if n_teaching == 0
        else " " + turns_cfg.get("examples_intro", "").format(n=n_teaching)
    )
    intro_user = _t("intro_user").format(**intro_kwargs)
    messages.append({"role": "user", "content": intro_user})
    messages.append({"role": "assistant", "content": _t("intro_assistant")})

    # Teaching example pairs
    for i, label in enumerate(teaching_labels):
        sentence = sentences[i]
        sentence_content = _t("sentence_user").format(sentence=sentence)
        messages.append({"role": "user", "content": sentence_content})

        label_content = _t("label_assistant").format(label=display_map[label])
        messages.append({"role": "assistant", "content": label_content})

    # Test sentence
    test_sentence = sentences[len(schedule)]
    test_content = _t("test_user").format(sentence=test_sentence)
    messages.append({"role": "user", "content": test_content})
    messages.append({"role": "assistant", "content": _t("test_assistant")})

    # Question with option order: build kwargs dynamically
    question_kwargs = {}
    for i, cls in enumerate(option_order):
        letter = chr(97 + i)  # a, b, c, ...
        if use_aliases:
            question_kwargs[f"option_{letter}"] = display_map[cls]
        else:
            desc = (
                sampled_descriptions.get(cls, core.describe_option(cls))
                if sampled_descriptions
                else core.describe_option(cls)
            )
            question_kwargs[f"option_{letter}"] = f"{cls} {desc}"
    question_kwargs["question_scope"] = (
        "" if n_teaching == 0 else turns_cfg.get("question_scope", "")
    )
    question_content = _t("question_user").format(**question_kwargs)
    messages.append({"role": "user", "content": question_content})

    # Answer prefill
    messages.append({"role": "assistant", "content": _t("answer_prefill")})

    return messages


def _build_icl_row(
    sample_id: int,
    test_perturbation: str,
    is_baseline: bool,
    option_order: tuple[str, ...],
    num_teaching_examples: int,
    teaching_schedule_str: str,
    vdata: dict[str, typing.Any],
    generated_token_id: int,
    class_names: list[str],
    sentence_log_probs: dict[str, float] | None = None,
    variant_token_ids: dict[str, int] | None = None,
) -> dict[str, typing.Any]:
    """Build a single result row for the ICL teaching experiment.

    Maps letter metrics to class level metrics using the option order.
    """
    letters = [chr(65 + i) for i in range(len(option_order))]
    class_to_letter = {option_order[i]: letters[i] for i in range(len(option_order))}
    letter_to_class = {letters[i]: option_order[i] for i in range(len(option_order))}

    row: dict[str, typing.Any] = {
        "sample_id": sample_id,
        "test_perturbation": test_perturbation,
        "is_baseline": is_baseline,
        "option_order": "/".join(option_order),
        "num_teaching_examples": num_teaching_examples,
        "teaching_schedule": teaching_schedule_str,
        "generated_token_id": generated_token_id,
        "argmax_token": vdata["argmax_token"],
        "argmax_logit": vdata["argmax_logit"],
        "argmax_prob": vdata["argmax_prob"],
        "argmax_token_id": vdata["argmax_token_id"],
        "entropy": vdata["entropy"],
    }

    # Letter level metrics
    for letter in letters:
        row[f"sum_prob_{letter}"] = vdata[f"sum_prob_{letter}"]
        row[f"logsumexp_{letter}"] = vdata[f"logsumexp_{letter}"]
        row[f"primary_prob_{letter}"] = vdata[f"primary_prob_{letter}"]
        row[f"primary_logit_{letter}"] = vdata[f"primary_logit_{letter}"]

    # Class level metrics (mapped through option order)
    for cls in class_names:
        letter = class_to_letter[cls]
        row[f"sum_prob_{cls.lower()}"] = vdata[f"sum_prob_{letter}"]
        row[f"logsumexp_{cls.lower()}"] = vdata[f"logsumexp_{letter}"]
        row[f"primary_prob_{cls.lower()}"] = vdata[f"primary_prob_{letter}"]
        row[f"primary_logit_{cls.lower()}"] = vdata[f"primary_logit_{letter}"]

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

    # Short alias of predicted_class_aggregate (console summary)
    row["predicted_class"] = predicted_class_aggregate

    # Ground truth letter: the letter that corresponds to the ground truth class
    row["ground_truth_letter"] = class_to_letter.get(test_perturbation, "")

    # Per variant raw values
    for key, value in vdata.items():
        if key.startswith("v_prob_") or key.startswith("v_logit_"):
            row[key] = value

    # Sentence log probabilities
    if sentence_log_probs is not None:
        for key, value in sentence_log_probs.items():
            row[key] = value

    return row


def run(
    cfg: omegaconf.DictConfig,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
) -> tuple[dict, list]:
    """Execute the ICL teaching experiment loop.

    - 1 baseline pass (no hooks, fixed option order)
    - ``num_samples`` stochastic passes, split evenly between the two
      perturbation types, with shuffled option orders

    All perturbations (teaching examples + test sentence) fire in a single
    forward pass via a compound multi perturbation hook.

    Args:
        cfg: Resolved Hydra config.
        model: Loaded language model.
        tokenizer: Matching tokenizer.

    Returns:
        Tuple of (metrics dict, results list).
    """
    random.seed(cfg.experiment.seed)

    turns_cfg = cfg.prompts.turns
    class_names = list(turns_cfg.class_names)
    num_classes = len(class_names)

    if num_classes != 2:
        raise ValueError(
            f"ICL teaching requires 2 class names, got {num_classes}: {class_names}."
        )

    # same_sentence=true already uses one sentence everywhere,
    # so same_pair_sentence has no effect in that case.
    same_pair_sentence = turns_cfg.get("same_pair_sentence", False)
    if turns_cfg.same_sentence and same_pair_sentence:
        same_pair_sentence = False

    if cfg.perturbation.get("dropout_style", "post_sublayer") != "post_sublayer":
        raise ValueError(
            "ICL supports only perturbation.dropout_style=post_sublayer "
            "(AIAYN-style dropout is implemented for localization only)."
        )

    # --- Teaching schedule ---
    num_pairs = turns_cfg.num_pairs
    num_teaching = num_pairs * num_classes
    if same_pair_sentence:
        needed = num_pairs + 1  # one unique sentence per pair + 1 test
    else:
        needed = num_teaching + 1  # one sentence per teaching slot + 1 test

    print("\nICL Teaching setup:")
    print(f"  Classes: {' vs '.join(class_names)}")
    print(
        f"  Pair based schedule: {num_pairs} pairs x {num_classes} "
        f"classes = {num_teaching} examples (randomized for each sample)"
    )
    print(
        f"  Sentence mode: same_sentence={turns_cfg.same_sentence}, same_pair_sentence={same_pair_sentence}"
    )
    print(f"  Unique sentences needed per sample: {needed}")
    print(f"  swap_labels={turns_cfg.swap_labels}")

    # --- Prompt pool ---
    prompt_pool: list[dict[str, typing.Any]] | None = None
    if turns_cfg.get("prompt_pool") is not None:
        from spe.prompt_utils import load_prompt_pool

        labels = list(turns_cfg.labels)
        prompt_pool = load_prompt_pool(
            list(turns_cfg.prompt_pool),
            class_names=class_names,
            labels=labels,
            required_template_fields=TEMPLATE_FIELDS,
        )

    # --- Description pool ---
    description_pool: dict[str, list[str]] | None = None
    if turns_cfg.get("description_pool") is not None:
        description_pool = {k: list(v) for k, v in turns_cfg.description_pool.items()}
        print(f"  Description pool loaded: {list(description_pool.keys())}")

    # --- Sentence sourcing ---
    sentences_file = cfg.experiment.get("sentences_file", None)
    sentence_n_tokens = cfg.experiment.get("sentence_n_tokens", None)
    sentence_rng: random.Random = random.Random(cfg.experiment.seed)

    if sentences_file is not None:
        with open(sentences_file) as f:
            sentences_pool: list[str] | None = [
                line.strip() for line in f if line.strip()
            ]
        print(f"  Loaded {len(sentences_pool)} sentences from {sentences_file}")
        if not turns_cfg.same_sentence and len(sentences_pool) < needed:
            raise ValueError(
                f"Need at least {needed} unique sentences "
                f"({needed - 1} teaching + 1 test), "
                f"but {sentences_file} has {len(sentences_pool)}"
            )
    elif sentence_n_tokens is not None:
        sentences_pool = None
        print(f"  Sentence generation: {sentence_n_tokens} tokens each")
    else:
        raise ValueError(
            "Set experiment.sentences_file or experiment.sentence_n_tokens "
            "to provide sentences for ICL teaching"
        )

    # --- Resolve variant token IDs ---
    all_variants = []
    letters = [chr(65 + i) for i in range(num_classes)]
    for letter in letters:
        all_variants.extend(generation.letter_variants(letter))
    variant_token_ids = generation.resolve_token_ids_safe(tokenizer, all_variants)
    if not variant_token_ids:
        raise ValueError(
            "No letter variant resolved to a single token. Cannot run ICL teaching mode."
        )

    print(f"\nResolved variant token IDs: {variant_token_ids}")

    # --- Resolve and validate layer range ---
    resolved_first_layer, resolved_last_layer = hooks.resolve_layer_range(
        model,
        cfg.perturbation.first_layer,
        cfg.perturbation.last_layer,
    )

    # --- Baselines and stochastic split ---
    all_permutations = list(itertools.permutations(class_names))
    num_baselines = 1

    num_samples = cfg.experiment.num_samples
    samples_per_class = [num_samples // num_classes] * num_classes
    for i in range(num_samples % num_classes):
        samples_per_class[i] += 1
    total_rows = num_baselines + num_samples

    # Build class assignment list for stochastic phase
    stochastic_types: list[str] = []
    for cls_idx, cls in enumerate(class_names):
        stochastic_types.extend([cls] * samples_per_class[cls_idx])

    class_counts_str = " + ".join(
        f"{samples_per_class[i]} {class_names[i]}" for i in range(num_classes)
    )
    print(
        f"\nExperiment plan: {num_baselines} baseline(s) + {class_counts_str} = {total_rows} rows"
    )
    print(f"Layer range: {resolved_first_layer}..{resolved_last_layer}")

    # --- Results table header ---
    print_agg = cfg.output.print_aggregates
    if print_agg:
        letter_cols = "  ".join(f"{'sumP(' + l + ')':>9}" for l in letters)
        cls_headers = [f"sumP({c.lower()})" for c in class_names]
    else:
        q = '"'
        letter_cols = "  ".join(f"{'P(' + q + ' ' + l + q + ')':>9}" for l in letters)
        cls_headers = [f"P({c.lower()})" for c in class_names]
    cls_width = max(max(len(h) for h in cls_headers), 9)
    cls_cols = "  ".join(f"{h:>{cls_width}}" for h in cls_headers)
    ltr_cols_w = 9 * num_classes + 2 * (num_classes - 1)
    cls_cols_w = cls_width * num_classes + 2 * (num_classes - 1)
    table_width_icl = 59 + ltr_cols_w + 2 + cls_cols_w + 2 + 10
    print(f"\n{'=' * table_width_icl}")
    print(
        f"ICL TEACHING  |  {num_baselines} baselines + {num_samples} stochastic = {total_rows} rows  |  "
        f"dropout_rate={cfg.perturbation.dropout_rate}  |  noise_std={cfg.perturbation.noise_std}"
    )
    print(f"{'=' * table_width_icl}")
    print(
        f"{'#':>4}  {'TestType':<10}  {'Order':<25}  {'Argmax':<12}  "
        f"{letter_cols}  {cls_cols}  {'Pred':<10}"
    )
    print(f"{'-' * table_width_icl}")

    results: list[dict[str, typing.Any]] = []
    row_idx = 0
    num_printed_prompts = 0
    max_printed_prompts = min(3, total_rows)

    # ---- Phase 1: baseline (no hooks) ----
    baseline_orders = [(class_names[0], class_names[1])]

    # Deterministic schedule for baselines (no shuffling)
    baseline_schedule = list(class_names) * num_pairs
    baseline_schedule_str = ",".join(baseline_schedule)

    for baseline_order in baseline_orders:
        baseline_sentences = sentence_utils.sample_or_generate_sentences(
            needed,
            turns_cfg.same_sentence,
            sentences_pool=sentences_pool,
            sentence_n_tokens=sentence_n_tokens,
            tokenizer=tokenizer,
            rng=sentence_rng if sentences_pool is None else None,
        )
        if same_pair_sentence:
            baseline_sentences = _expand_pair_sentences(baseline_sentences, num_classes)
        # Sample prompt variant for baselines too
        baseline_variant: dict[str, typing.Any] | None = None
        if prompt_pool is not None:
            baseline_variant = random.choice(prompt_pool)
        messages = build_icl_messages(
            cfg,
            baseline_schedule,
            baseline_sentences,
            option_order=baseline_order,
            template_overrides=baseline_variant["templates"]
            if baseline_variant
            else None,
        )

        if num_printed_prompts < max_printed_prompts:
            print(f"\n{'=' * 60}")
            print(f"PROMPT (row {row_idx})")
            print(f"{'=' * 60}")
            for msg in messages:
                print(f"[{msg['role']}]: {msg['content']}")
            print(f"{'=' * 60}")
            # Re-print table header after prompt block so rows align with headers
            print(f"\n{'=' * table_width_icl}")
            print(
                f"{'#':>4}  {'TestType':<10}  {'Order':<25}  {'Argmax':<12}  "
                f"{letter_cols}  {cls_cols}  {'Pred':<10}"
            )
            print(f"{'-' * table_width_icl}")
            num_printed_prompts += 1

        _response, logits, generated_token_id = core.run_single_sample(
            model,
            tokenizer,
            messages,
            hook_fn=None,
            hook_target=cfg.perturbation.hook_target,
            first_layer=cfg.perturbation.first_layer,
            last_layer=cfg.perturbation.last_layer,
            enable_thinking=cfg.model.thinking,
        )

        vdata = generation.extract_n_letter_variant_data(
            logits, variant_token_ids, tokenizer, letters
        )
        row = _build_icl_row(
            sample_id=row_idx,
            test_perturbation="NOTHING",
            is_baseline=True,
            option_order=baseline_order,
            num_teaching_examples=num_teaching,
            teaching_schedule_str=baseline_schedule_str,
            vdata=vdata,
            generated_token_id=generated_token_id,
            class_names=class_names,
            variant_token_ids=variant_token_ids,
        )
        if baseline_variant is not None:
            row["prompt_variant"] = baseline_variant["name"]
        results.append(row)

        order_str = "/".join(baseline_order)
        prob_key = "sum_prob" if print_agg else "primary_prob"
        letter_vals = "  ".join(f"{vdata[f'{prob_key}_{l}']:>9.6f}" for l in letters)
        cls_vals = "  ".join(
            f"{row[f'{prob_key}_{c.lower()}']:>{cls_width}.6f}" for c in class_names
        )
        print(
            f"{row_idx:>4}  {'BASELINE':<10}  {order_str:<25}  {vdata['argmax_token']:<12}  "
            f"{letter_vals}  {cls_vals}  {row['predicted_class']:<10}"
        )
        row_idx += 1

    # ---- Phase 2: stochastic passes ----
    for i in range(num_samples):
        test_perturbation = stochastic_types[i]

        # Shuffle option order randomly
        option_order = tuple(random.choice(all_permutations))

        schedule = build_pair_schedule(class_names, num_pairs)
        teaching_schedule_str = ",".join(schedule)

        # Sample fresh sentences for each stochastic row
        sentences = sentence_utils.sample_or_generate_sentences(
            needed,
            turns_cfg.same_sentence,
            sentences_pool=sentences_pool,
            sentence_n_tokens=sentence_n_tokens,
            tokenizer=tokenizer,
            rng=sentence_rng if sentences_pool is None else None,
        )
        if same_pair_sentence:
            sentences = _expand_pair_sentences(sentences, num_classes)

        # Sample descriptions from pool (if configured)
        sampled_descriptions: dict[str, str] | None = None
        if description_pool is not None:
            sampled_descriptions = {
                cls: random.choice(description_pool[cls])
                for cls in class_names
                if cls in description_pool
            }

        # Sample prompt variant (or use default templates)
        selected_variant: dict[str, typing.Any] | None = None
        if prompt_pool is not None:
            selected_variant = random.choice(prompt_pool)
        messages = build_icl_messages(
            cfg,
            schedule,
            sentences,
            option_order=option_order,
            template_overrides=selected_variant["templates"]
            if selected_variant
            else None,
            sampled_descriptions=sampled_descriptions,
        )

        # Compute token ranges for each sentence substring (teaching + test).
        # All sentences (teaching + test) are located in one pass to handle
        # duplicate sentences across turns correctly.
        all_sentences = sentences[:num_teaching] + [sentences[len(schedule)]]
        token_ranges = compute_ordered_substring_token_ranges(
            tokenizer,
            messages,
            all_sentences,
            enable_thinking=cfg.model.thinking,
        )

        perturbation_entries: list[dict[str, typing.Any]] = []
        for teach_idx in range(num_teaching):
            first_token, last_token = token_ranges[teach_idx]
            perturbation_entries.append(
                {
                    "type": schedule[teach_idx],
                    "first_token": first_token,
                    "last_token": last_token,
                    "dropout_rate": cfg.perturbation.dropout_rate,
                    "noise_std": cfg.perturbation.noise_std,
                    "label": f"teach{teach_idx}",
                }
            )

        # Test sentence
        test_first, test_last = token_ranges[num_teaching]

        perturbation_entries.append(
            {
                "type": test_perturbation,
                "first_token": test_first,
                "last_token": test_last,
                "dropout_rate": cfg.perturbation.dropout_rate,
                "noise_std": cfg.perturbation.noise_std,
                "label": "test",
            }
        )

        if num_printed_prompts < max_printed_prompts:
            print_multi_range_token_map(
                tokenizer,
                messages,
                perturbation_entries,
                enable_thinking=cfg.model.thinking,
            )
            num_printed_prompts += 1

        # Create hook(s) and run forward pass
        hook_fn = hooks.create_multi_perturbation_hook(perturbation_entries)
        handles = hooks.register_hooks(
            model,
            hook_fn,
            target=cfg.perturbation.hook_target,
            first_layer=cfg.perturbation.first_layer,
            last_layer=cfg.perturbation.last_layer,
        )
        try:
            _response, logits, generated_token_id, all_logits = (
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

        # Compute sentence log probs from the full logits
        sent_log_probs: dict[str, float] = {}
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
            for entry in perturbation_entries:
                sent_log_probs[f"sentence_log_prob_{entry['label']}"] = (
                    generation.compute_token_range_log_prob(
                        all_logits,
                        input_ids,
                        entry["first_token"],
                        entry["last_token"],
                    )
                )
            del all_logits

        vdata = generation.extract_n_letter_variant_data(
            logits, variant_token_ids, tokenizer, letters
        )
        row = _build_icl_row(
            sample_id=row_idx,
            test_perturbation=test_perturbation,
            is_baseline=False,
            option_order=option_order,
            num_teaching_examples=num_teaching,
            teaching_schedule_str=teaching_schedule_str,
            vdata=vdata,
            generated_token_id=generated_token_id,
            class_names=class_names,
            sentence_log_probs=sent_log_probs if sent_log_probs else None,
            variant_token_ids=variant_token_ids,
        )
        if selected_variant is not None:
            row["prompt_variant"] = selected_variant["name"]
        results.append(row)

        order_str = "/".join(option_order)
        prob_key = "sum_prob" if print_agg else "primary_prob"
        letter_vals = "  ".join(f"{vdata[f'{prob_key}_{l}']:>9.6f}" for l in letters)
        cls_vals = "  ".join(
            f"{row[f'{prob_key}_{c.lower()}']:>{cls_width}.6f}" for c in class_names
        )
        print(
            f"{row_idx:>4}  {test_perturbation:<10}  {order_str:<25}  {vdata['argmax_token']:<12}  "
            f"{letter_vals}  {cls_vals}  {row['predicted_class']:<10}"
        )
        row_idx += 1

    print(f"{'=' * table_width_icl}")

    metrics = eval_icl.compute_and_plot(
        results,
        class_names=class_names,
        dpi=cfg.output.dpi,
        fmt=cfg.output.format,
    )
    return metrics, results
