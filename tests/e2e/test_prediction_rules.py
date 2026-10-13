"""The four prediction rules disagree exactly where they should.

Fake logits: the full-vocabulary argmax is the bare lowercase token ``a`` (a variant
of A but not the primary token ``" A"``), and ``" A"`` beats ``" B"``. So, for a
sample whose ground truth is A: aggregate (argmax in variants) and both constrained
rules are correct, the exact-token primary rule is not.
"""

from unittest.mock import patch

import torch

from spe.experiments.classification import _build_classification_row
from spe.experiments.localization import run_localization
from spe.generation import extract_n_letter_variant_data, resolve_token_ids_safe

from .test_localization_e2e import _cfg


def _logits(tok) -> torch.Tensor:
    logits = torch.full((tok.vocab_size,), -10.0)
    ids = resolve_token_ids_safe(tok, ["a", " A", " B"])
    logits[ids["a"]] = 5.0
    logits[ids[" A"]] = 3.0
    logits[ids[" B"]] = 1.0
    return logits


def test_localization_rules(
    tokenizer, fake_model, tmp_output_dir, sentences_file
) -> None:
    def fake(model, tok, messages, enable_thinking=False, return_all_logits=False):
        logits = _logits(tok)
        return "a", logits, int(logits.argmax()), None

    with patch("spe.generation.generate_single_token", side_effect=fake):
        metrics, results = run_localization(
            _cfg(sentences_file, num_samples=8), fake_model, tokenizer
        )

    for r in results:
        assert r["predicted_argmax"] == "A"
        assert r["predicted_argmax_primary"] == "OTHER"
        assert r["predicted_primary"] == "A"
        assert r["predicted_aggregate"] == "A"
    frac_a = sum(r["ground_truth"] == "A" for r in results) / len(results)
    assert metrics["metrics_version"] == 2
    assert metrics["accuracy_aggregate"] == frac_a
    assert metrics["accuracy_constrained_primary"] == frac_a
    assert metrics["accuracy_constrained_aggregate"] == frac_a
    assert metrics["accuracy_primary"] == 0.0
    assert metrics["argmax_is_label_rate"] == 1.0
    assert metrics["argmax_is_primary_label_rate"] == 0.0


def test_classification_row_rules(tokenizer) -> None:
    letters = ["A", "B"]
    variants = [
        v
        for letter in letters
        for v in (letter, f" {letter}", letter.lower(), f" {letter.lower()}")
    ]
    variant_ids = resolve_token_ids_safe(tokenizer, variants)
    vdata = extract_n_letter_variant_data(
        _logits(tokenizer), variant_ids, tokenizer, letters
    )
    row = _build_classification_row(
        0,
        "DROPOUT",
        False,
        ("DROPOUT", "NOISE"),
        vdata,
        vdata["argmax_token_id"],
        ["DROPOUT", "NOISE"],
        variant_ids,
    )
    assert row["predicted_class_argmax"] == "DROPOUT"
    assert row["predicted_class_argmax_primary"] == "OTHER"
    assert row["predicted_class_primary"] == "DROPOUT"
    assert row["predicted_class_aggregate"] == "DROPOUT"


def test_constrained_rules_disagree(tokenizer) -> None:
    """ " A" beats " B" on the primary logit, but the B variants together carry more mass."""
    letters = ["A", "B"]
    variants = [
        v
        for letter in letters
        for v in (letter, f" {letter}", letter.lower(), f" {letter.lower()}")
    ]
    variant_ids = resolve_token_ids_safe(tokenizer, variants)
    logits = torch.full((tokenizer.vocab_size,), -10.0)
    logits[variant_ids[" A"]] = 3.0
    for v in ("B", " B", "b", " b"):
        logits[variant_ids[v]] = 2.9
    vdata = extract_n_letter_variant_data(logits, variant_ids, tokenizer, letters)
    row = _build_classification_row(
        0,
        "DROPOUT",
        False,
        ("DROPOUT", "NOISE"),
        vdata,
        vdata["argmax_token_id"],
        ["DROPOUT", "NOISE"],
        variant_ids,
    )
    assert row["predicted_class_primary"] == "DROPOUT"
    assert row["predicted_class_aggregate"] == "NOISE"
    assert row["predicted_class_argmax"] == "DROPOUT"
    assert row["predicted_class_argmax_primary"] == "DROPOUT"
