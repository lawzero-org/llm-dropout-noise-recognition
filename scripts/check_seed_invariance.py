"""Hashed seed of every grid point of the paper's sweeps, to check that a code change leaves them unchanged.

    uv run python scripts/check_seed_invariance.py --write seeds.json   # record
    uv run python scripts/check_seed_invariance.py --check seeds.json   # compare after a change

The seed is a hash of the resolved config (``experiment_setup.resolve_seed``), so an unchanged seed means that
re-running the sweep reproduces the W&B runs exactly. Sweeps are the manifest entries in
``notebooks/paper/sweeps.yaml``; the sweep ``<group>: {<key>: <id>}`` is ``sweeps/<group>/<key>.yaml``;
the 32 log-prob runs are the ones of ``scripts/logprob_runs.py``.
"""

import argparse
import itertools
import json
import multiprocessing
import pathlib
import sys

import hydra
import yaml

from spe import experiment_setup

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import logprob_runs  # noqa: E402

CONFIG = {
    "spe-localize": "localization",
    "spe-classify": "classification",
    "spe-icl": "icl",
}


def sweep_files() -> dict[str, pathlib.Path]:
    manifest = yaml.safe_load((REPO / "notebooks/paper/sweeps.yaml").read_text())
    return {
        sid: REPO / "sweeps" / group / f"{key}.yaml"
        for group, sweeps in manifest.items()
        if isinstance(sweeps, dict)
        for key, sid in sweeps.items()
    }


def _as_sent(v: object) -> object:
    """The value as the W&B agent passes it: whole-number floats arrive as ints (0.0 -> 0)."""
    return int(v) if isinstance(v, float) and v.is_integer() else v


def points(f: pathlib.Path) -> tuple[str, list[list[str]]]:
    doc = yaml.safe_load(f.read_text())
    fixed, dims = [], {}
    for k, v in doc["parameters"].items():
        if "values" in v:
            dims[k] = v["values"]
        else:
            fixed.append((k, v["value"]))
    grid = []
    for combo in itertools.product(*dims.values()):
        ov = []
        for k, v in fixed + list(zip(dims, combo, strict=True)):
            ov += str(v).split() if k == "hydra_overrides" else [f"{k}={_as_sent(v)}"]
        grid.append(ov)
    return CONFIG[doc["program"]], grid


def _init() -> None:
    hydra.initialize_config_dir(
        version_base=None, config_dir=str(REPO / "src/spe/conf")
    )


def _seed(job: tuple[str, str, list[str]]) -> tuple[str, int]:
    key, name, ov = job
    return key, experiment_setup.resolve_seed(
        hydra.compose(config_name=name, overrides=ov)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write")
    g.add_argument("--check")
    args = ap.parse_args()
    jobs = []
    for sid, f in sweep_files().items():
        name, grid = points(f)
        jobs += [(f"{sid}/{i}", name, ov) for i, ov in enumerate(grid)]
    for i, ov in enumerate(logprob_runs.overrides()):
        jobs.append((f"logprob/{i}", "localization", ov))
    with multiprocessing.Pool(initializer=_init) as pool:
        seeds = dict(pool.map(_seed, jobs, chunksize=64))
    if args.write:
        pathlib.Path(args.write).write_text(json.dumps(seeds, indent=0, sort_keys=True))
        print(f"{len(seeds)} seeds written to {args.write}")
        return
    ref = json.loads(pathlib.Path(args.check).read_text())
    changed = [k for k in ref if seeds.get(k) != ref[k]]
    missing = [k for k in ref if k not in seeds]
    print(
        f"{len(ref)} reference seeds: {len(changed)} changed, {len(missing)} missing, {len(seeds) - len(ref) + len(missing)} new"
    )
    if changed or missing:
        raise SystemExit(f"first differences: {(changed or missing)[:5]}")


if __name__ == "__main__":
    main()
