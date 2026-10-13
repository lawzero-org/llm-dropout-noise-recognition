"""E2E tests for ``spe-classify`` (binary classification) entrypoint.

Tests the full pipeline: config -> prompt construction -> hook creation ->
(mocked) generation -> metric extraction -> result building, without
loading a real model.
"""

from collections import Counter

from omegaconf import OmegaConf

from spe.experiments.classification import run_classification

# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _cfg(
    sentences_file: str,
    num_samples: int = 4,
    active_perturbations: list[str] | None = None,
    seed: int = 42,
):
    """Build a minimal resolved classification config."""
    user_turn = (
        "Process this sentence: {sentence}\n\n"
        "Which perturbation was applied? "
        "A) {option_a} B) {option_b}"
    )

    return OmegaConf.create(
        {
            "prompts": {
                "system": {"content": ""},
                "turns": {
                    "name": "test",
                    "labels": ["A", "B"],
                    "class_names": ["DROPOUT", "NOISE"],
                    "prompt_pool": None,
                    "description_pool": None,
                    "turns": [
                        {"role": "user", "content": user_turn},
                        {"role": "assistant", "content": "the answer is:"},
                    ],
                },
            },
            "model": {
                "name": "test",
                "dtype": "bfloat16",
                "device_map": "cpu",
                "thinking": False,
            },
            "perturbation": {
                "mode": "multiclass",
                "dropout_rate": 0.1,
                "dropout_style": "post_sublayer",
                "noise_std": 0.01,
                "hook_target": "both",
                "first_layer": 0,
                "last_layer": -1,
                "first_token": 0,
                "last_token": -1,
                "perturb_from_turn": None,
                "perturb_to_turn": None,
                "active_perturbations": active_perturbations,
            },
            "aliases": {"display_names": None, "descriptions": None},
            "output": {"print_aggregates": True, "dpi": 72, "format": "png"},
            "experiment": {
                "seed": seed,
                "num_samples": num_samples,
                "sentences_file": sentences_file,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_runs(classification_mocks, tokenizer, fake_model, sentences_file):
    """Smoke test: full classification pipeline completes and returns the
    right number of result rows."""
    cfg = _cfg(sentences_file, num_samples=4)
    metrics, results = run_classification(cfg, fake_model, tokenizer)

    assert isinstance(metrics, dict)
    assert isinstance(results, list)
    # No NOTHING class -> no baselines; results == num_samples.
    assert len(results) == 4


def test_result_row_structure(
    classification_mocks, tokenizer, fake_model, sentences_file
):
    """Each result row contains all letter-level, class-level, and
    derived metric keys."""
    cfg = _cfg(sentences_file, num_samples=2)
    _, results = run_classification(cfg, fake_model, tokenizer)
    row = results[0]

    # Bookkeeping
    for key in (
        "sample_id",
        "perturbation",
        "is_baseline",
        "option_order",
        "generated_token_id",
        "argmax_token",
        "argmax_logit",
        "argmax_prob",
        "entropy",
    ):
        assert key in row, f"Missing bookkeeping key: {key}"

    # Letter-level metrics (A, B)
    for letter in ("A", "B"):
        for prefix in (
            "sum_prob",
            "logsumexp",
            "primary_prob",
            "primary_logit",
            "primary_log_prob",
            "aggregate_log_prob",
        ):
            assert f"{prefix}_{letter}" in row, f"Missing {prefix}_{letter}"

    # Class-level metrics (dropout, noise)
    for cls in ("dropout", "noise"):
        for prefix in (
            "sum_prob",
            "logsumexp",
            "primary_prob",
            "primary_logit",
            "primary_log_prob",
            "aggregate_log_prob",
        ):
            assert f"{prefix}_{cls}" in row, f"Missing {prefix}_{cls}"

    # Logit diffs
    assert "logit_diff_dropout" in row
    assert "logit_diff_noise" in row
    assert "logit_diff_dropout_vs_noise" in row

    # Predictions
    assert "predicted_class_aggregate" in row
    assert "predicted_class_primary" in row
    assert "predicted_class_argmax" in row
    assert "ground_truth_letter" in row

    # Log prob of correct answer
    assert "primary_log_prob_correct" in row
    assert "aggregate_log_prob_correct" in row


def test_balanced_perturbation_assignment(
    classification_mocks,
    tokenizer,
    fake_model,
    sentences_file,
):
    """Samples are split evenly across DROPOUT and NOISE."""
    cfg = _cfg(sentences_file, num_samples=6)
    _, results = run_classification(cfg, fake_model, tokenizer)

    counts = Counter(r["perturbation"] for r in results)
    assert counts["DROPOUT"] == 3
    assert counts["NOISE"] == 3


def test_option_order_varies(
    classification_mocks, tokenizer, fake_model, sentences_file
):
    """With enough samples, more than one option order is used."""
    cfg = _cfg(sentences_file, num_samples=20)
    _, results = run_classification(cfg, fake_model, tokenizer)

    orders = {r["option_order"] for r in results}
    assert len(orders) > 1, "Expected multiple option orderings"


def test_active_perturbations_restriction(
    classification_mocks,
    tokenizer,
    fake_model,
    sentences_file,
):
    """``active_perturbations`` restricts which classes are sampled."""
    cfg = _cfg(sentences_file, num_samples=4, active_perturbations=["DROPOUT"])
    _, results = run_classification(cfg, fake_model, tokenizer)

    assert all(r["perturbation"] == "DROPOUT" for r in results)


def test_with_sentences(
    classification_mocks,
    tokenizer,
    fake_model,
    sentences_file,
):
    """Sentence-based mode records the sentence and computes substring
    token ranges."""
    cfg = _cfg(sentences_file, num_samples=4)
    _, results = run_classification(cfg, fake_model, tokenizer)

    assert len(results) == 4
    for r in results:
        assert "sentence" in r
        assert isinstance(r["sentence"], str)
        assert len(r["sentence"]) > 0
