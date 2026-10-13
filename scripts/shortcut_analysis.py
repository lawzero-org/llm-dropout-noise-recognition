"""Log-probability analysis of the localization task (paper appendix).

For each sample, do THREE forward passes on the same prompt:
  1. clean: no perturbation anywhere
  2. pertA: perturbation applied only to sentence A's tokens
  3. pertB: perturbation applied only to sentence B's tokens

For each pass, read out the sentence joint log probability of A and B.
That gives six atomic per sample quantities (3 runs x 2 sentences):

  lp_clean_A, lp_clean_B,
  lp_pertA_A, lp_pertA_B,
  lp_pertB_A, lp_pertB_B

These are logged as a single wandb.Table (one row per sample,
seven columns including the sample index). All downstream
analyses (effect deltas, histograms, scatter plots, ratios)
are computed locally from the downloaded table. We do not log
aggregate statistics or histograms.

Note: lp_pertB_A should equal lp_clean_A modulo floating point
noise, because A precedes B and causal attention prevents B's
perturbation from reaching A's positions. Both columns are kept
so the invariant can be checked.

The paper's 32 runs are listed by ``scripts/logprob_runs.py``:

    uv run python scripts/shortcut_analysis.py $(uv run python scripts/logprob_runs.py | sed -n 1p)
    sbatch --array=0-31 scripts/command_array.sh      # all of them on a Slurm cluster

Reuses model loading, hook machinery, sentence sampling and prompt
construction from src/spe.
"""

from __future__ import annotations

import random
import typing

import hydra
import omegaconf
import wandb

from spe import (
    experiment_setup,
    generation,
    hooks,
    sentence_utils,
    wandb_utils,
)
from spe.experiments import localization as exp_loc
from spe.prompt_utils import (
    build_messages,
    build_messages_from_turns,
    compute_substring_token_range,
    load_prompt_pool,
    print_multi_range_token_map,
    validate_token_range,
)


def _make_entries(
    labels: list[str],
    token_ranges: dict[str, tuple[int, int]],
    target_position: str | None,
    cfg: omegaconf.DictConfig,
) -> list[dict[str, typing.Any]]:
    """Build perturbation entries.

    ``target_position=None`` produces an all NOTHING entry list (clean run).
    Otherwise the named position gets ``cfg.perturbation.target_perturbation``
    and every other position is NOTHING.
    """
    entries: list[dict[str, typing.Any]] = []
    target_perturbation = cfg.perturbation.target_perturbation
    for lbl in labels:
        first_tok, last_tok = token_ranges[lbl]
        ptype = target_perturbation if lbl == target_position else "NOTHING"
        entries.append(
            {
                "type": ptype,
                "first_token": first_tok,
                "last_token": last_tok,
                "dropout_rate": cfg.perturbation.dropout_rate,
                "noise_std": cfg.perturbation.get("noise_std", 0.01),
                "label": f"sentence_{lbl.lower()}",
            }
        )
    return entries


def _forward_lp(
    model: typing.Any,
    tokenizer: typing.Any,
    messages: list[dict[str, str]],
    entries: list[dict[str, typing.Any]],
    cfg: omegaconf.DictConfig,
    token_ranges: dict[str, tuple[int, int]],
    labels: list[str],
) -> dict[str, float]:
    """One forward pass with the given perturbation entries.

    Returns ``{label: sentence_joint_log_prob}``.
    """
    hook_fn = hooks.create_multi_perturbation_hook(entries)
    handles = hooks.register_hooks(
        model,
        hook_fn,
        target=cfg.perturbation.hook_target,
        first_layer=cfg.perturbation.first_layer,
        last_layer=cfg.perturbation.last_layer,
    )
    try:
        _r, _l, _t, all_logits = generation.generate_single_token(
            model,
            tokenizer,
            messages,
            enable_thinking=cfg.model.thinking,
            return_all_logits=True,
        )
    finally:
        hooks.remove_hooks(handles)

    is_prefill = messages[-1]["role"] == "assistant"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not is_prefill,
        continue_final_message=is_prefill,
        enable_thinking=cfg.model.thinking,
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    out: dict[str, float] = {}
    for lbl in labels:
        first_tok, last_tok = token_ranges[lbl]
        out[lbl] = generation.compute_token_range_log_prob(
            all_logits, input_ids, first_tok, last_tok
        )
    return out


@hydra.main(
    version_base=None,
    config_path="../src/spe/conf",
    config_name="localization",
)
def main(cfg: omegaconf.DictConfig) -> dict:  # type: ignore[type-arg]
    """Run the shortcut analysis: 3 forward passes per sample."""
    model, tokenizer = experiment_setup.setup_experiment(cfg)

    labels: list[str] = ["A", "B"]
    num_sentences = 2
    num_samples = cfg.experiment.num_samples
    same_sentence = cfg.perturbation.same_sentence
    sentence_n_tokens = cfg.experiment.get("sentence_n_tokens", None)
    sentences_file = cfg.experiment.get("sentences_file", None)
    target_perturbation_name = (
        cfg.perturbation.get("prompt_label", None)
        or cfg.perturbation.target_perturbation
    )

    hardcoded: list[str] | None = None
    if sentences_file is not None:
        abs_path = hydra.utils.to_absolute_path(sentences_file)
        with open(abs_path) as f:
            hardcoded = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(hardcoded)} sentences from {abs_path}")

    sentence_rng = random.Random(cfg.experiment.get("seed", 42))

    # Same prompt pool as the localization runs (one variant sampled for each sample).
    prompt_pool = (
        load_prompt_pool(list(cfg.prompts.turns.prompt_pool))
        if cfg.prompts.turns.get("prompt_pool") is not None
        else None
    )

    table = wandb.Table(
        columns=[
            "sample_idx",
            "lp_clean_A",
            "lp_clean_B",
            "lp_pertA_A",
            "lp_pertA_B",
            "lp_pertB_A",
            "lp_pertB_B",
            "prompt_variant",
        ]
    )

    for i in range(num_samples):
        if hardcoded is not None:
            sampled = sentence_utils.sample_or_generate_sentences(
                num_sentences,
                same_sentence,
                sentences_pool=hardcoded,
            )
        else:
            sampled = sentence_utils.sample_or_generate_sentences(
                num_sentences,
                same_sentence,
                sentence_n_tokens=sentence_n_tokens,
                tokenizer=tokenizer,
                rng=sentence_rng,
            )

        format_kwargs = exp_loc._build_format_kwargs(
            labels,
            sampled,
            num_sentences,
            target_perturbation_name=target_perturbation_name,
        )
        variant = random.choice(prompt_pool) if prompt_pool is not None else None
        if variant is not None:
            messages = build_messages_from_turns(
                cfg.prompts.system.content,
                variant["turns"],
                format_kwargs=format_kwargs,
            )
        else:
            messages = build_messages(cfg, format_kwargs=format_kwargs)

        token_ranges: dict[str, tuple[int, int]] = {}
        for lbl_idx, lbl in enumerate(labels):
            anchor = exp_loc._build_anchor(lbl, same_sentence)
            first_tok, last_tok = compute_substring_token_range(
                tokenizer,
                messages,
                sampled[lbl_idx],
                anchor=anchor,
                enable_thinking=cfg.model.thinking,
            )
            validate_token_range(first_tok, last_tok)
            token_ranges[lbl] = (first_tok, last_tok)

        lp_clean = _forward_lp(
            model,
            tokenizer,
            messages,
            _make_entries(labels, token_ranges, target_position=None, cfg=cfg),
            cfg,
            token_ranges,
            labels,
        )
        lp_pertA = _forward_lp(
            model,
            tokenizer,
            messages,
            _make_entries(labels, token_ranges, target_position="A", cfg=cfg),
            cfg,
            token_ranges,
            labels,
        )
        lp_pertB = _forward_lp(
            model,
            tokenizer,
            messages,
            _make_entries(labels, token_ranges, target_position="B", cfg=cfg),
            cfg,
            token_ranges,
            labels,
        )

        table.add_data(
            i,
            lp_clean["A"],
            lp_clean["B"],
            lp_pertA["A"],
            lp_pertA["B"],
            lp_pertB["A"],
            lp_pertB["B"],
            variant["name"] if variant is not None else "",
        )
        if i < 2:
            exp_loc._print_first_sample(messages)
            print_multi_range_token_map(
                tokenizer,
                messages,
                _make_entries(labels, token_ranges, target_position="B", cfg=cfg),
                enable_thinking=cfg.model.thinking,
            )
            print(f"  lp clean {lp_clean} | pertA {lp_pertA} | pertB {lp_pertB}")

        if (i + 1) % 50 == 0:
            print(f"  [shortcut-analysis] {i + 1}/{num_samples} done")

    if cfg.wandb.enabled:
        wandb.log({"per_sample_logprobs": table})
    else:
        print("wandb.enabled=false: the per-sample table is not logged.")

    wandb_utils.finish_run()
    print("\nShortcut analysis complete.")
    return {}


if __name__ == "__main__":
    main()
