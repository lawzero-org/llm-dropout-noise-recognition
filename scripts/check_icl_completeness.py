"""Completeness of an ICL sweep arm: which grid points have a run with `accuracy_primary`, and refills for the gaps.

    uv run python scripts/check_icl_completeness.py --arm iclzs_main            # report only
    uv run python scripts/check_icl_completeness.py --arm iclzs_ctrl --write    # alias controls; write refill YAMLs

`--arm` is a manifest key of notebooks/paper/sweeps.yaml (iclzs_main, iclzs_swap, iclzs_ctrl).
The grid is every list-valued parameter of sweeps/<model>/<arm>.yaml (e.g. num_pairs x aliases x swap_labels x p x
sigma); the runs of the arm's sweep plus every `<arm>_refill*` sweep are merged. Run states on W&B are unreliable
under rate limits and a run can finish with an empty summary, so a point counts only if some run at that point has
`accuracy_primary` in its summary (client-side check; the server-side `summary_metrics` filter is not reliable).
Refill sweeps list the missing points exactly, one `hydra_overrides` string per point: same resolved config -> same
hashed seed as the original would have had. After `wandb sweep`, add the id to the manifest as `<arm>_refill` (then
`<arm>_refill2`, ...).
"""

import argparse
import itertools
import os
import pathlib

import pandas as pd
import wandb
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
# W&B project; the entity is WANDB_ENTITY or your default entity.
PROJECT = os.environ.get("WANDB_PROJECT", "llm-dropout-noise")
MODELS = [
    "llama3_8b",
    "qwen3_14b",
    "mistral_small_24b",
    "olmo3_32b",
    "qwen3_32b",
    "seed_oss_36b",
    "gemma4_31b",
    "qwen3_8_27b",
]


def cfg_get(c: dict, dotted: str):
    """Flat "a.b.c" key if present (grid sweeps), else walk the nested config (hydra_overrides sweeps)."""
    if dotted in c:
        return c[dotted]
    cur = c
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def norm(value):
    """Comparable form of a grid value: bools stay bools, numbers become floats, the rest strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(
        value, dict
    ):  # a resolved config group (e.g. aliases) has no name in the nested config
        return None
    return str(value)


def override(key: str, value) -> str:
    return f"{key}={str(value).lower() if isinstance(value, bool) else value}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", default="iclzs_main", help="manifest key, e.g. iclzs_main, iclzs_swap"
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="write sweeps/<model>/<arm>_refill.yaml for the gaps",
    )
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    arm = args.arm
    api = wandb.Api(timeout=120)
    manifest = yaml.safe_load((REPO / "notebooks/paper/sweeps.yaml").read_text())
    total_missing = 0
    for m in args.models:
        sid = manifest[m][arm]
        params = yaml.safe_load((REPO / f"sweeps/{m}/{arm}.yaml").read_text())[
            "parameters"
        ]
        dims = {
            k: [norm(v) for v in spec["values"]]
            for k, spec in params.items()
            if "values" in spec
        }
        fixed = {k: spec for k, spec in params.items() if "value" in spec}
        int_dims = {  # norm() turns 3 into 3.0; an override "num_pairs=3.0" makes Hydra pass a float (list * float fails)
            k
            for k, spec in params.items()
            if "values" in spec
            and all(
                isinstance(v, int) and not isinstance(v, bool) for v in spec["values"]
            )
        }
        grid = set(itertools.product(*dims.values()))
        sweep_ids = [sid] + [
            v for k, v in manifest[m].items() if k.startswith(f"{arm}_refill")
        ]
        rows = []
        for r in (
            run
            for s in sweep_ids
            for run in api.runs(PROJECT, filters={"sweep": s}, per_page=1000)
        ):
            c, summary = r.config, r.summary._json_dict
            point = []
            for k in dims:
                v = cfg_get(c, k)
                if (
                    v is None or norm(v) is None
                ):  # hydra_overrides refill: read the group name from the override string
                    tokens = dict(
                        t.split("=", 1)
                        for t in str(c.get("hydra_overrides", "")).split()
                        if "=" in t
                    )
                    v = tokens.get(k)
                    if v is not None and k in dims and isinstance(dims[k][0], bool):
                        v = v.lower() == "true"
                    elif v is not None and isinstance(dims[k][0], float):
                        v = float(v)
                point.append(norm(v))
            rows.append(
                {
                    "id": r.id,
                    "state": r.state,
                    "point": tuple(point),
                    "acc": "accuracy_primary" in summary,
                }
            )
        df = pd.DataFrame(rows)
        good = df[df.acc]
        covered = set(good.point)
        missing = sorted(grid - covered, key=str)
        dup = good.groupby("point").size()
        states = df.state.value_counts().to_dict()
        print(
            f"{m:18s} {'+'.join(sweep_ids)} runs={len(df)} states={states} with_acc={len(good)} "
            f"points={len(covered)}/{len(grid)} missing={len(missing)} duplicates={int((dup > 1).sum())}"
        )
        if missing:
            seen = (
                df[~df.acc]
                .groupby("point")
                .state.agg(lambda x: ",".join(sorted(set(x))))
            )
            for pt in missing:
                print(
                    "   missing",
                    dict(zip(dims, pt, strict=True)),
                    "| states seen:",
                    seen.get(pt, "never dispatched"),
                )
        total_missing += len(missing)
        if args.write and missing:
            overrides = [
                " ".join(
                    override(k, int(v) if k in int_dims else v)
                    for k, v in zip(dims, pt, strict=True)
                )
                for pt in missing
            ]
            out = {
                "project": PROJECT,
                "program": "spe-icl",
                "method": "grid",
                "name": f"{m}-{arm.replace('_', '-')}-refill",
                "metric": {"name": "accuracy_primary", "goal": "maximize"},
                "parameters": {**fixed, "hydra_overrides": {"values": overrides}},
            }
            path = REPO / f"sweeps/{m}/{arm}_refill.yaml"
            if path.exists():
                path = REPO / f"sweeps/{m}/{arm}_refill2.yaml"
            header = (
                f"# Refill of the grid points of {arm} {sid} that have no run with accuracy_primary;\n"
                "# one hydra_overrides string per missing point, same resolved config -> same hashed seed.\n"
            )
            path.write_text(header + yaml.safe_dump(out, sort_keys=False, width=250))
            print(f"  wrote {path.relative_to(REPO)} ({len(overrides)} points)")
    print(f"TOTAL missing points: {total_missing}")


if __name__ == "__main__":
    main()
