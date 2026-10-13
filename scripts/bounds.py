"""Perturbation bounds of Table 1 (paper Sec. 4) from the localization and control sweeps.

    uv run python scripts/bounds.py              # every model of the manifest
    uv run python scripts/bounds.py qwen3_32b    # one model, with the curves
    uv run python scripts/bounds.py olmo3_32b --exact-token   # OLMo with the exact-token accuracy (Sec. 4 footnote)

Rule (the one Table 1 uses):
- p_min / sigma_min: the strength just before the first two consecutive strengths whose detection accuracy is
  above 50% + 2 SE (a single excursion out of the chance band is ignored);
- p_max / sigma_max: the strength just before the first strength whose mean content accuracy over the five
  control topic pairs is below 95%.
The metric is the exact-token accuracy; OLMo answers "neither" at low strengths, so for it both curves use the
constrained accuracy (logit of " A" vs " B"). Zoom sweeps listed as ``loc_<pert>_<suffix>`` / ``ctrl_<pert>_<suffix>``
in the manifest are merged with the base sweep (strengths present in both are averaged).
"""

import argparse
import math
import pathlib
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
from wandb_cache import load_sweep  # noqa: E402

MANIFEST = pathlib.Path(__file__).resolve().parents[1] / "notebooks/paper/sweeps.yaml"
STRENGTH = {"dropout": "perturbation.dropout_rate", "noise": "perturbation.noise_std"}
CONSTRAINED = {
    "olmo3_32b",
    "olmo3_32b_instruct_base",
    "olmo3_32b_instruct_sft",
    "olmo3_32b_instruct_dpo",
}
N_PER_RUN = 1000
Z = 2.0
THRESHOLD = 0.95


def load_family(sweeps: dict, prefix: str, project: str) -> pd.DataFrame:
    """Base sweep ``prefix`` plus every zoom sweep ``prefix_<suffix>`` of the same model."""
    ids = [
        sid
        for key, sid in sweeps.items()
        if key == prefix or key.startswith(prefix + "_")
    ]
    return pd.concat(
        [load_sweep(sid, project, require_version=2) for sid in ids], ignore_index=True
    )


def curve(df: pd.DataFrame, x: str, metric: str) -> pd.Series:
    """Mean metric per strength (control sweeps have one run per topic pair)."""
    return df.groupby(x)[metric].mean().sort_index()


def lower_bound(acc: pd.Series) -> float | None:
    """Strength before the first two consecutive points above the chance band (None if never)."""
    up = (acc > 0.5 + Z * math.sqrt(0.25 / N_PER_RUN)).tolist()
    idx = list(acc.index)
    for i in range(len(up) - 1):
        if up[i] and up[i + 1]:
            return idx[i - 1] if i > 0 else None
    return None


def upper_bound(content: pd.Series) -> float | None:
    """Strength before the first mean content accuracy below the threshold (None if never)."""
    idx = list(content.index)
    for i, v in enumerate(content.tolist()):
        if v < THRESHOLD:
            return idx[i - 1] if i > 0 else None
    return None


def bounds(
    model: str, manifest: dict, verbose: bool = False, exact_token: bool = False
) -> dict[str, tuple[float | None, float | None]]:
    constrained = model in CONSTRAINED and not exact_token
    metric = "accuracy_constrained_primary" if constrained else "accuracy_primary"
    out = {}
    for pert, x in STRENGTH.items():
        acc = curve(
            load_family(manifest[model], f"loc_{pert}", manifest["project"]), x, metric
        )
        content = curve(
            load_family(manifest[model], f"ctrl_{pert}", manifest["project"]),
            x,
            f"content_{metric}",
        )
        out[pert] = (lower_bound(acc), upper_bound(content))
        if verbose:
            print(f"\n=== {model} / {pert} ({metric})")
            table = pd.DataFrame({"detection": acc, "control_content": content})
            print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "models", nargs="*", help="manifest model keys (default: every model)"
    )
    ap.add_argument(
        "--exact-token", action="store_true", help="exact-token accuracy for OLMo too"
    )
    args = ap.parse_args()
    manifest = yaml.safe_load(MANIFEST.read_text())
    models = args.models or [
        m for m, v in manifest.items() if isinstance(v, dict) and "loc_dropout" in v
    ]
    rows = []
    for m in models:
        b = bounds(m, manifest, verbose=bool(args.models), exact_token=args.exact_token)
        rows.append(
            {
                "model": m,
                "p_min": b["dropout"][0],
                "p_max": b["dropout"][1],
                "sigma_min": b["noise"][0],
                "sigma_max": b["noise"][1],
            }
        )
    print("\n" + pd.DataFrame(rows).set_index("model").to_string())


if __name__ == "__main__":
    main()
