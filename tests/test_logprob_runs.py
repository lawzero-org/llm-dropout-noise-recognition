"""The log-prob runs: 8 models x {dropout, noise} x {low, high}, from the Table 1 ranges."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import logprob_runs  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_32_distinct_runs() -> None:
    runs = logprob_runs.overrides()
    assert len(runs) == 32
    assert len({" ".join(r) for r in runs}) == 32
    assert {r[0].removeprefix("model=") for r in runs} == set(logprob_runs.MODELS)


def test_low_is_range_start_and_high_is_midpoint() -> None:
    runs = {
        (r[0], r[1], r[-1]): float(r[2].split("=")[1]) for r in logprob_runs.overrides()
    }
    # Qwen3-32B, Table 1: p in (0.18, 0.54), sigma in (0.10, 0.22)
    m = "model=qwen3_32b"
    d = "perturbation.target_perturbation=DROPOUT"
    n = "perturbation.target_perturbation=NOISE"
    assert runs[(m, d, "experiment.stage=logprob_low")] == 0.18
    assert runs[(m, d, "experiment.stage=logprob_high")] == 0.36
    assert runs[(m, n, "experiment.stage=logprob_low")] == 0.1
    assert runs[(m, n, "experiment.stage=logprob_high")] == 0.16
