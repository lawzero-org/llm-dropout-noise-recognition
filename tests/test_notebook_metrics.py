"""Test notebook computations without running their W&B downloads or widgets."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def notebook_functions(path, **namespace):
    """Execute the notebook's actual function definitions, without its side effects."""
    namespace = {"pd": pd, "np": np, **namespace}
    notebook = json.loads((NOTEBOOKS / path).read_text())
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            definitions.extend(n for n in tree.body if isinstance(n, ast.FunctionDef))
    exec(  # noqa: S102 -- Test the checked-in notebook definitions without network calls.
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def icl_helpers():
    return notebook_functions(
        "paper/2_few_shot.ipynb",
        pd=pd,
        np=np,
        COL_SWAP="swap",
        COL_DROPOUT="p",
        COL_NOISE="sigma",
        LABEL_RATE_COL={
            "accuracy_primary": "argmax_is_primary_label_rate",
            "accuracy_aggregate": "argmax_is_label_rate",
            "accuracy_constrained_primary": None,
        },
    )


@pytest.mark.parametrize(
    ("column", "rate_column", "expected"),
    [
        ("accuracy_primary", "argmax_is_primary_label_rate", 20),
        ("accuracy_aggregate", "argmax_is_label_rate", 40),
        ("accuracy_constrained_primary", None, 80),
    ],
)
def test_single_and_combined_heatmaps_agree(icl_helpers, column, rate_column, expected):
    row = {
        "p": 0.1,
        "sigma": 0.1,
        column: 0.2,
        "argmax_is_primary_label_rate": 0.4,
        "argmax_is_label_rate": 0.6,
    }
    df = pd.DataFrame([{**row, "swap": False}, {**row, "swap": True}])
    original = df.copy(deep=True)
    main, flipped, diff = icl_helpers["make_heatmap_data"](df, column, 100, True)
    single = icl_helpers["make_single_heatmap"](df, column, 100, True, True)
    assert main.iloc[0, 0] == pytest.approx(20)
    assert flipped.iloc[0, 0] == pytest.approx(expected)
    assert diff.iloc[0, 0] == pytest.approx(20 - expected)
    pd.testing.assert_frame_equal(single, flipped)
    pd.testing.assert_frame_equal(df, original)
    if rate_column:
        df[column] = df[rate_column] = 0.0  # all predictions are OTHER for this rule
        assert (
            icl_helpers["make_single_heatmap"](df, column, 100, True, True).iloc[0, 0]
            == 0
        )


@pytest.mark.parametrize(
    ("column", "scale", "flip", "expected"),
    [("logit_diff_correct", 1, True, -0.2), ("accuracy_se", 100, False, 20)],
)
def test_heatmap_logit_diff_and_uncertainty(icl_helpers, column, scale, flip, expected):
    df = pd.DataFrame([{"swap": True, "p": 0.1, "sigma": 0.1, column: 0.2}])
    assert (
        icl_helpers["make_single_heatmap"](df, column, scale, flip, True).iloc[0, 0]
        == expected
    )


def test_common_metric_includes_legacy_controls_and_missing_rows(icl_helpers):
    select = icl_helpers["select_accuracy_metric"]
    modern = pd.DataFrame(
        {
            "accuracy_primary": [0.2],
            "argmax_is_primary_label_rate": [0.4],
            "accuracy_constrained_primary": [0.8],
        }
    )
    legacy = modern[["accuracy_constrained_primary"]].copy()
    assert select(modern, modern) == "accuracy_primary"
    assert select(modern, legacy) == "accuracy_constrained_primary"
    mixed = pd.concat([modern, legacy], ignore_index=True)
    assert select(mixed, modern) == "accuracy_constrained_primary"
    with pytest.raises(ValueError, match="No accuracy rule"):
        select(modern, pd.DataFrame({"accuracy_primary": [float("nan")]}))
