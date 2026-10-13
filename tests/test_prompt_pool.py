"""Prompt variants load from any working directory (the pool path is resolved from the package)."""

import pytest

from spe.experiments.icl import TEMPLATE_FIELDS
from spe.prompt_utils import load_prompt_pool


@pytest.fixture(autouse=True)
def _elsewhere(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def test_turn_based_variants() -> None:
    pool = load_prompt_pool(
        ["localization/main_variants/v00", "classification/main_variants/v19"]
    )
    assert len(pool) == 2 and all(e["turns"] for e in pool)


def test_template_variants() -> None:
    pool = load_prompt_pool(
        ["icl/main_zs_variants/v00"],
        class_names=["DROPOUT", "NOISE"],
        labels=["A", "B"],
        required_template_fields=TEMPLATE_FIELDS,
    )
    assert set(pool[0]["templates"]) == set(TEMPLATE_FIELDS)
