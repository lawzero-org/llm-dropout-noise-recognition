"""Shared wandb sweep data loading with local caching.

A sweep is read from ``data/cache/<sweep_id>.pkl`` when that file exists (no W&B account or network needed),
otherwise it is downloaded from the W&B project (entity: ``WANDB_ENTITY`` or your default entity) and cached.
"""

import pathlib

import pandas as pd
from tqdm.auto import tqdm

CACHE_DIR = pathlib.Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_KEY = "accuracy_primary"


def _flat_items(prefix: str, d: dict):
    """Yield ("a.b.c", value) for every nested key of ``d`` below the top level."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if prefix:
            yield key, v
        if isinstance(v, dict):
            yield from _flat_items(key, v)


_api = None


def _get_api(sweep_id: str):
    """W&B API client, imported lazily so that cached sweeps load without W&B."""
    global _api
    if _api is None:
        import wandb

        try:
            _api = wandb.Api()
        except Exception as e:
            raise RuntimeError(
                f"sweep {sweep_id} is not in {CACHE_DIR} and W&B is not available ({e}). "
                "Either put the cached pickle there, or log in to W&B (set WANDB_ENTITY if needed) "
                "and list your own sweep ids in notebooks/paper/sweeps.yaml."
            ) from e
    return _api


def load_sweep(
    sweep_id: str,
    project: str,
    required_key: str = REQUIRED_KEY,
    require_version: int | None = None,
    min_rows: int | None = None,
) -> pd.DataFrame:
    """Download all runs from a sweep, caching to a local pickle file once the sweep and all its runs have finished.

    Args:
        sweep_id: The wandb sweep identifier.
        project: The wandb project name (e.g. "llm-dropout-noise").
        required_key: Only include runs whose summary contains this key.
        require_version: If set, raise unless every run logged this ``metrics_version``
            (the paper notebooks use 2, the metrics version of every paper run).
        min_rows: If set, a sweep with fewer finished runs (e.g. one still running) is returned but not cached,
            and a cached copy with fewer rows is downloaded again.

    Returns:
        DataFrame with one row per run (config + summary merged).
    """
    df = _load_sweep(sweep_id, project, required_key, min_rows)
    if require_version is not None:
        versions = (
            df["metrics_version"]
            if "metrics_version" in df.columns
            else pd.Series(dtype=float)
        )
        if len(versions) != len(df) or (versions != require_version).any():
            raise ValueError(
                f"sweep {sweep_id}: not all runs have metrics_version={require_version}"
            )
    return df


def _load_sweep(
    sweep_id: str, project: str, required_key: str, min_rows: int | None = None
) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{sweep_id}.pkl"
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if min_rows is None or len(cached) >= min_rows:
            print(f"Loading {sweep_id} from cache: {cache_path}")
            return cached

    api = _get_api(sweep_id)
    print(f"Downloading {sweep_id} from wandb...")
    runs_iter = api.runs(project, filters={"sweep": sweep_id}, per_page=1000)

    rows = []
    skipped = 0
    active = 0
    for run in tqdm(runs_iter, desc="Fetching runs"):
        active += run.state in ("running", "pending")
        summary = run.summary._json_dict
        if required_key not in summary:
            skipped += 1
            continue
        row = {**run.config, **summary}
        # Sweeps that pass params as one override string (spe-sweep `hydra_overrides`)
        # have no flat "section.key" entries; derive them from the nested config at
        # every depth (e.g. prompts.turns.num_pairs).
        for key, value in _flat_items("", run.config):
            row.setdefault(key, value)
        # A config-group override in the string (e.g. ``aliases=masking_jitter``) leaves the flat
        # key holding the group's resolved body (a dict); keep the group name instead.
        for token in str(row.get("hydra_overrides", "")).split():
            key, _, value = token.partition("=")
            if isinstance(row.get(key), dict):
                row[key] = value
        row["_run_state"] = run.state
        rows.append(row)
    if not rows and min_rows is not None:
        print(f"  {sweep_id}: 0 of {min_rows} runs finished; not caching")
        return pd.DataFrame()
    if not rows:  # never cache an empty sweep (e.g. one still running): the pickle would shadow later runs
        raise ValueError(
            f"sweep {sweep_id}: no run has '{required_key}' ({skipped} runs seen); not caching"
        )
    df = pd.DataFrame(rows)
    if min_rows is not None and len(rows) < min_rows:
        print(f"  {sweep_id}: {len(rows)} of {min_rows} runs finished; not caching")
        return df
    # A grid sweep is FINISHED once every point is dispatched, so its runs must have stopped too.
    state = api.sweep(f"{project}/{sweep_id}").state
    if state != "FINISHED" or active:
        print(f"  {sweep_id}: sweep {state}, {active} runs still active; not caching")
        return df
    df.to_pickle(cache_path)
    print(
        f"  Total runs: {len(rows) + skipped}, with '{required_key}': {len(rows)}, missing: {skipped}"
    )
    return df


def load_sweep_tables(
    sweep_id: str,
    project: str,
    table_key: str = "results_table_full",
    required_key: str = REQUIRED_KEY,
) -> pd.DataFrame:
    """Download per-sample results tables for all runs in a sweep.

    Each run's wandb Table is fetched and concatenated into a single DataFrame.
    Run-level config columns are merged so each row has both per-sample data
    and the run's config (model, perturbation params, etc.).

    Args:
        sweep_id: The wandb sweep identifier.
        project: The wandb project name.
        table_key: The wandb log key for the table (default "results_table_full").
        required_key: Only include runs whose summary contains this key.

    Returns:
        DataFrame with one row per sample across all runs.
    """
    cache_path = CACHE_DIR / f"{sweep_id}_tables.pkl"
    if cache_path.exists():
        print(f"Loading {sweep_id} tables from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    api = _get_api(sweep_id)
    print(f"Downloading {sweep_id} per-sample tables from wandb...")
    all_runs = list(api.runs(project, filters={"sweep": sweep_id}, per_page=1000))

    all_rows = []
    skipped = 0
    for run in tqdm(all_runs, desc="Fetching tables"):
        summary = run.summary._json_dict
        if required_key not in summary:
            skipped += 1
            continue

        # Extract run-level config for merging
        run_config = {
            "_run_id": run.id,
            "_run_name": run.name,
        }
        for k, v in run.config.items():
            if not isinstance(v, dict):
                run_config[k] = v

        # Fetch the table artifact
        table_ref = summary.get(table_key)
        if table_ref is None:
            skipped += 1
            continue

        try:
            # Find the logged artifact that contains the table
            artifact = None
            for art in run.logged_artifacts():
                if table_key in [f.name.split(".table.json")[0] for f in art.files()]:
                    artifact = art
                    break
            if artifact is None:
                skipped += 1
                continue
            table_file = artifact.get(table_key)
            table_df = pd.DataFrame(table_file.data, columns=table_file.columns)
            for k, v in run_config.items():
                table_df[k] = v
            all_rows.append(table_df)
        except Exception as e:
            print(f"  Warning: failed to fetch table for run {run.id}: {e}")
            skipped += 1
            continue

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df.to_pickle(cache_path)
    print(
        f"  Total runs: {len(all_runs)}, fetched tables: {len(all_rows)}, skipped: {skipped}"
    )
    print(f"  Total samples: {len(df)}")
    return df


def clear_cache(sweep_id: str) -> None:
    """Delete cached files for a sweep, forcing re-download."""
    for name in [f"{sweep_id}.pkl", f"{sweep_id}_tables.pkl"]:
        cache_path = CACHE_DIR / name
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache: {cache_path}")
