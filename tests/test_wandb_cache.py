"""Cached sweeps load without importing W&B."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"

# Run in a fresh interpreter: other tests import wandb, which would hide an eager import.
SCRIPT = """
import sys, pathlib
sys.path.insert(0, {notebooks!r})
import wandb_cache
wandb_cache.CACHE_DIR = pathlib.Path({cache!r})
df = wandb_cache.load_sweep("abc12345", "some-project", require_version=2)
assert len(df) == 2, df
assert "wandb" not in sys.modules, "wandb imported although the sweep is cached"
print("ok")
"""


def test_cached_sweep_loads_without_wandb(tmp_path) -> None:
    pd.DataFrame({"accuracy_primary": [0.5, 0.9], "metrics_version": [2, 2]}).to_pickle(
        tmp_path / "abc12345.pkl"
    )
    out = subprocess.run(  # noqa: S603 - fixed argv: this interpreter and a literal script
        [
            sys.executable,
            "-c",
            SCRIPT.format(notebooks=str(NOTEBOOKS), cache=str(tmp_path)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
