"""E2E tests for ``spe-icl`` (in-context learning) entrypoint.

Tests the full pipeline: config -> teaching schedule -> ICL message
construction -> multi-perturbation hook creation -> (mocked) generation ->
metric extraction -> result building, without loading a real model.
"""

from collections import Counter

import pytest
from omegaconf import OmegaConf

from spe.experiments.icl import build_icl_messages, run

# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _cfg(
    num_samples: int = 4,
    num_pairs: int = 2,
    same_sentence: bool = True,
    swap_labels: bool = False,
    sentences_file: str | None = None,
    seed: int = 42,
):
    """Build a minimal resolved ICL teaching config."""
    return OmegaConf.create(
        {
            "prompts": {
                "system": {"content": "You are helpful."},
                "turns": {
                    "name": "test",
                    "labels": ["A", "B"],
                    "class_names": ["DROPOUT", "NOISE"],
                    "prompt_pool": None,
                    "description_pool": None,
                    "num_pairs": num_pairs,
                    "teaching_schedule": None,
                    "same_sentence": same_sentence,
                    "same_pair_sentence": False,
                    "swap_labels": swap_labels,
                    # Conversation templates
                    "intro_user": "I will apply perturbations. "
                    "Types: {type_a} and {type_b}.",
                    "intro_assistant": "I understand.",
                    "sentence_user": 'Pay attention to: "{sentence}"',
                    "label_assistant": "That was {label}.",
                    "test_user": 'Test: "{sentence}"',
                    "test_assistant": "Introspecting.",
                    "question_user": "Which perturbation? A) {option_a} B) {option_b}",
                    "answer_prefill": "the answer is:",
                },
            },
            "model": {"thinking": False},
            "perturbation": {
                "mode": "icl_teaching",
                "dropout_rate": 0.3,
                "dropout_style": "post_sublayer",
                "noise_std": 0.03,
                "hook_target": "both",
                "first_layer": 0,
                "last_layer": -1,
            },
            "aliases": {"display_names": None, "descriptions": None},
            "output": {"print_aggregates": True, "dpi": 72, "format": "png"},
            "experiment": {
                "seed": seed,
                "num_samples": num_samples,
                "sentences_file": sentences_file,
                "sentence_n_tokens": None,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_runs(icl_mocks, tokenizer, fake_model, sentences_file):
    """Smoke test: full ICL pipeline completes and returns the right
    number of result rows (1 baseline + num_samples for 2-class)."""
    cfg = _cfg(num_samples=4, sentences_file=sentences_file)
    metrics, results = run(cfg, fake_model, tokenizer)

    assert isinstance(metrics, dict)
    assert isinstance(results, list)
    # 2-class mode: 1 baseline + 4 stochastic = 5 rows
    assert len(results) == 5


def test_result_row_structure(icl_mocks, tokenizer, fake_model, sentences_file):
    """Each result row contains all expected ICL-specific keys."""
    cfg = _cfg(num_samples=2, sentences_file=sentences_file)
    _, results = run(cfg, fake_model, tokenizer)

    # Pick a stochastic row (skip baseline)
    row = next(r for r in results if not r["is_baseline"])

    # Bookkeeping
    for key in (
        "sample_id",
        "test_perturbation",
        "is_baseline",
        "option_order",
        "num_teaching_examples",
        "teaching_schedule",
        "generated_token_id",
        "argmax_token",
        "argmax_logit",
        "argmax_prob",
        "entropy",
    ):
        assert key in row, f"Missing bookkeeping key: {key}"

    # Letter-level metrics
    for letter in ("A", "B"):
        for prefix in ("sum_prob", "logsumexp", "primary_prob", "primary_logit"):
            assert f"{prefix}_{letter}" in row, f"Missing {prefix}_{letter}"

    # Class-level metrics
    for cls in ("dropout", "noise"):
        for prefix in ("sum_prob", "logsumexp", "primary_prob", "primary_logit"):
            assert f"{prefix}_{cls}" in row, f"Missing {prefix}_{cls}"

    # Logit diffs
    assert "logit_diff_dropout" in row
    assert "logit_diff_noise" in row

    # Predictions
    assert "predicted_class" in row
    assert "predicted_class_aggregate" in row
    assert "predicted_class_primary" in row
    assert "predicted_class_argmax" in row
    assert "ground_truth_letter" in row


def test_baseline_present(icl_mocks, tokenizer, fake_model, sentences_file):
    """2-class mode produces exactly 1 baseline row with
    ``test_perturbation='NOTHING'``."""
    cfg = _cfg(num_samples=2, sentences_file=sentences_file)
    _, results = run(cfg, fake_model, tokenizer)

    baselines = [r for r in results if r["is_baseline"]]
    assert len(baselines) == 1
    assert baselines[0]["test_perturbation"] == "NOTHING"


def test_balanced_test_perturbation(
    icl_mocks,
    tokenizer,
    fake_model,
    sentences_file,
):
    """Stochastic samples are split evenly across perturbation types."""
    cfg = _cfg(num_samples=6, sentences_file=sentences_file)
    _, results = run(cfg, fake_model, tokenizer)

    stochastic = [r for r in results if not r["is_baseline"]]
    counts = Counter(r["test_perturbation"] for r in stochastic)
    assert counts["DROPOUT"] == 3
    assert counts["NOISE"] == 3


def test_teaching_schedule_shape(
    icl_mocks,
    tokenizer,
    fake_model,
    sentences_file,
):
    """Teaching schedule has ``num_pairs * num_classes`` entries,
    with each class represented in every pair."""
    cfg = _cfg(num_samples=4, num_pairs=3, sentences_file=sentences_file)
    _, results = run(cfg, fake_model, tokenizer)

    stochastic = [r for r in results if not r["is_baseline"]]
    for row in stochastic:
        schedule = row["teaching_schedule"].split(",")
        # 3 pairs x 2 classes = 6 entries
        assert len(schedule) == 6
        # Each entry is a valid class name
        assert all(s in ("DROPOUT", "NOISE") for s in schedule)


def test_swap_labels_control(
    icl_mocks,
    tokenizer,
    fake_model,
    sentences_file,
):
    """``swap_labels=True`` runs to completion (labels mismatch
    perturbation types)."""
    cfg = _cfg(
        num_samples=2,
        swap_labels=True,
        sentences_file=sentences_file,
    )
    _, results = run(cfg, fake_model, tokenizer)
    assert len(results) > 0


def test_swap_labels_names_the_other_perturbation():
    """With ``swap_labels=True`` each teaching sentence is labeled with the
    perturbation that was *not* applied to it."""
    schedule = ["DROPOUT", "NOISE", "NOISE", "DROPOUT"]
    sentences = ["s1", "s1", "s2", "s2", "test"]
    labels = {}
    for swap in (False, True):
        cfg = _cfg(num_pairs=2, swap_labels=swap)
        msgs = build_icl_messages(
            cfg, schedule, sentences, option_order=("DROPOUT", "NOISE")
        )
        labels[swap] = [
            m["content"] for m in msgs if m["content"].startswith("That was")
        ]
    assert labels[False] == [f"That was {c}." for c in schedule]
    assert labels[True] == [
        f"That was {'NOISE' if c == 'DROPOUT' else 'DROPOUT'}." for c in schedule
    ]


def test_three_classes_raise(tokenizer, fake_model, sentences_file):
    """Only the two-class task of the paper is supported."""
    cfg = _cfg(sentences_file=sentences_file)
    cfg.prompts.turns.class_names = ["DROPOUT", "NOISE", "NOTHING"]
    with pytest.raises(ValueError, match="2 class names"):
        run(cfg, fake_model, tokenizer)
