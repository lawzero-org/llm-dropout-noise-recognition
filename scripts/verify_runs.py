"""Recompute the headline metrics of finished W&B runs from their per-sample tables.

For each run: check metrics_version, integer seed, runtime info; rebuild the four prediction
rules from the logged per-row scores; recompute accuracies (+SE), AUROC and logit-diff mean;
compare with the summary. Exact match for counts, 1e-6 for floats.

    uv run python scripts/verify_runs.py RUN_ID [RUN_ID ...]
    uv run python scripts/verify_runs.py --since 2027-01-01T12:00  # all runs created after
"""

import argparse
import json
import math
import os
import re
import tempfile
import typing

import numpy as np
import transformers
import wandb
from sklearn.metrics import roc_auc_score

from spe.generation import letter_variants, resolve_token_ids_safe

# W&B project; the entity is WANDB_ENTITY or your default entity.
PROJECT = os.environ.get("WANDB_PROJECT", "llm-dropout-noise")
TOL = 1e-6
_TOKENIZERS: dict[tuple, typing.Any] = {}


def model_key(cfg: dict) -> tuple[str, str | None]:
    """(HF name, revision). Sweep-agent runs store the Hydra model *config name* under ``model``."""
    m = cfg["model"]
    if isinstance(m, dict):
        return m["name"], m.get("revision")
    import yaml  # noqa: PLC0415

    model_cfg = yaml.safe_load(open(f"src/spe/conf/model/{m}.yaml"))
    return model_cfg["name"], model_cfg.get("revision")


def tokenizer_for(cfg: dict):
    key = model_key(cfg)
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = transformers.AutoTokenizer.from_pretrained(
            key[0], revision=key[1]
        )
    return _TOKENIZERS[key]


def argmax_labels(tok, letters: list[str], token_id: int) -> tuple[str, str]:
    """(variant label, exact-token label) of a raw argmax token id, or OTHER."""
    variant = next(
        (
            L
            for L in letters
            if token_id in resolve_token_ids_safe(tok, letter_variants(L)).values()
        ),
        "OTHER",
    )
    exact = next(
        (
            L
            for L in letters
            if resolve_token_ids_safe(tok, [f" {L}"])[f" {L}"] == token_id
        ),
        "OTHER",
    )
    return variant, exact


def table(run) -> list[dict]:
    ref = run.summary._json_dict["results_table_full"]["path"]
    with tempfile.TemporaryDirectory() as d:
        t = json.load(open(run.file(ref).download(root=d, replace=True).name))
    return [dict(zip(t["columns"], row, strict=True)) for row in t["data"]]


def acc_se(correct: list[bool]) -> tuple[float, float]:
    n = len(correct)
    p = sum(correct) / n
    return p, math.sqrt(p * (1 - p) / n)


def check(name: str, got, want, exact: bool, report: list[str]) -> None:
    ok = got == want if exact else abs(float(got) - float(want)) <= TOL
    report.append(
        f"  {'ok  ' if ok else 'FAIL'} {name}: summary={want} recomputed={got}"
    )


def verify_localization(rows: list[dict], s: dict, report: list[str], tok) -> None:
    labels = sorted(
        k[len("prob_") :] for k in rows[0] if k.startswith("prob_") and len(k) == 6
    )
    gt = [r["ground_truth"] for r in rows]
    for (
        r
    ) in rows:  # rebuild both argmax rules from the raw token id through the tokenizer
        variant, exact = argmax_labels(tok, labels, r["argmax_token_id"])
        if (variant, exact) != (r["predicted_argmax"], r["predicted_argmax_primary"]):
            raise ValueError(
                f"argmax label mismatch in sample {r['sample_id']}: {(variant, exact)}"
            )
    report.append("  ok   argmax labels rebuilt from raw token ids (all rows)")
    # rebuild the four rules from per-row scores / raw argmax token id
    pri = [max(labels, key=lambda l: r[f"primary_logit_{l}"]) for r in rows]
    agg = [max(labels, key=lambda l: r[f"prob_{l}"]) for r in rows]
    for r, p, a in zip(rows, pri, agg, strict=True):
        if r["predicted_primary"] != p or r["predicted_aggregate"] != a:
            raise ValueError(
                f"constrained prediction mismatch in sample {r['sample_id']}"
            )
    argmax_lbl = [r["predicted_argmax"] for r in rows]
    argmax_pri = [r["predicted_argmax_primary"] for r in rows]
    for r in rows:  # exact-token rule implies the variant rule
        if r["predicted_argmax_primary"] not in ("OTHER", r["predicted_argmax"]):
            raise ValueError(
                f"exact-token outside variant label in sample {r['sample_id']}"
            )
    for key, pred in [
        ("accuracy_constrained_primary", pri),
        ("accuracy_constrained_aggregate", agg),
        ("accuracy_aggregate", argmax_lbl),
        ("accuracy_primary", argmax_pri),
    ]:
        p, se = acc_se([g == q for g, q in zip(gt, pred, strict=True)])
        check(key, p, s[key], False, report)
        check(
            key.replace("accuracy", "accuracy_se"),
            se,
            s[key.replace("accuracy", "accuracy_se")],
            False,
            report,
        )
    check(
        "argmax_is_label_rate",
        sum(q != "OTHER" for q in argmax_lbl) / len(rows),
        s["argmax_is_label_rate"],
        False,
        report,
    )
    check(
        "argmax_is_primary_label_rate",
        sum(q != "OTHER" for q in argmax_pri) / len(rows),
        s["argmax_is_primary_label_rate"],
        False,
        report,
    )
    if "content_ground_truth" in rows[0]:
        cgt = [r["content_ground_truth"] for r in rows]
        for key, pred in [
            ("content_accuracy_constrained_primary", pri),
            ("content_accuracy_aggregate", argmax_lbl),
            ("content_accuracy_primary", argmax_pri),
        ]:
            p, _ = acc_se([g == q for g, q in zip(cgt, pred, strict=True)])
            check(key, p, s[key], False, report)
    if len(labels) == 2:
        y = [1 if g == labels[0] else 0 for g in gt]
        check(
            "roc_auc_constrained_primary",
            roc_auc_score(y, [r[f"primary_prob_{labels[0]}"] for r in rows]),
            s["roc_auc_constrained_primary"],
            False,
            report,
        )
        check(
            "roc_auc_constrained_aggregate",
            roc_auc_score(y, [r[f"prob_{labels[0]}"] for r in rows]),
            s["roc_auc_constrained_aggregate"],
            False,
            report,
        )
    check(
        "logit_diff_mean",
        float(np.mean([r["logit_diff"] for r in rows])),
        s["logit_diff_mean"],
        False,
        report,
    )


def verify_classification(rows: list[dict], s: dict, report: list[str], tok) -> None:
    rows = [r for r in rows if not r["is_baseline"]]
    gt_key = (
        "perturbation" if "perturbation" in rows[0] else "test_perturbation"
    )  # ICL rows
    gt = [r[gt_key] for r in rows]
    for r in rows:  # rebuild both argmax rules from the raw token id + option order
        classes = r["option_order"].split("/")
        letters = [chr(65 + i) for i in range(len(classes))]
        variant, exact = argmax_labels(tok, letters, r["argmax_token_id"])
        to_cls = dict(zip(letters, classes, strict=True))
        want = (to_cls.get(variant, "OTHER"), to_cls.get(exact, "OTHER"))
        if want != (r["predicted_class_argmax"], r["predicted_class_argmax_primary"]):
            raise ValueError(
                f"argmax class mismatch in sample {r['sample_id']}: {want}"
            )
    report.append("  ok   argmax classes rebuilt from raw token ids (all rows)")
    # one-vs-rest ROC AUC per class on the constrained scores, plus the macro average
    for version, col in [
        ("constrained_primary", "primary_prob"),
        ("constrained_aggregate", "sum_prob"),
    ]:
        aucs = []
        for cls in sorted(set(gt)):
            key = f"roc_auc_{version}_{cls.lower()}"
            if key not in s:
                continue
            auc = roc_auc_score(
                [1 if g == cls else 0 for g in gt],
                [r[f"{col}_{cls.lower()}"] for r in rows],
            )
            aucs.append(auc)
            check(key, auc, s[key], False, report)
        if f"roc_auc_{version}_macro" in s:
            check(
                f"roc_auc_{version}_macro",
                float(np.mean(aucs)),
                s[f"roc_auc_{version}_macro"],
                False,
                report,
            )
    # <group>_mean_logit_diff_<class> = mean of logit_diff_<class> over rows whose ground truth is <group>
    for key in s:
        m = re.match(r"^([a-z]+)_mean_logit_diff_([a-z]+)$", key)
        if m and any(r[gt_key].lower() == m[1] for r in rows):
            vals = [r[f"logit_diff_{m[2]}"] for r in rows if r[gt_key].lower() == m[1]]
            check(key, float(np.mean(vals)), s[key], False, report)
    for key, col in [
        ("accuracy_constrained_primary", "predicted_class_primary"),
        ("accuracy_constrained_aggregate", "predicted_class_aggregate"),
        ("accuracy_aggregate", "predicted_class_argmax"),
        ("accuracy_primary", "predicted_class_argmax_primary"),
    ]:
        p, se = acc_se([g == r[col] for g, r in zip(gt, rows, strict=True)])
        check(key, p, s[key], False, report)
        check(
            key.replace("accuracy", "accuracy_se"),
            se,
            s[key.replace("accuracy", "accuracy_se")],
            False,
            report,
        )
    check(
        "argmax_is_label_rate",
        sum(r["predicted_class_argmax"] != "OTHER" for r in rows) / len(rows),
        s["argmax_is_label_rate"],
        False,
        report,
    )
    check(
        "argmax_is_primary_label_rate",
        sum(r["predicted_class_argmax_primary"] != "OTHER" for r in rows) / len(rows),
        s["argmax_is_primary_label_rate"],
        False,
        report,
    )
    # swapped-label view used by the few-shot notebook: label predictions that were wrong
    wrong_label = sum(
        r["predicted_class_argmax"] not in ("OTHER", g)
        for g, r in zip(gt, rows, strict=True)
    ) / len(rows)
    report.append(
        f"  info swapped-view aggregate (label-rate - acc) = {s['argmax_is_label_rate'] - s['accuracy_aggregate']:.6f}, from table = {wrong_label:.6f}"
    )


COMPARE_KEYS = ("ground_truth", "prompt_variant", "argmax_token_id")


def compare(api, a_id: str, b_id: str) -> None:
    """Sample-by-sample comparison of two runs: inputs first, then model outputs."""
    a, b = (table(api.run(f"{PROJECT}/{rid}")) for rid in (a_id, b_id))
    if len(a) != len(b):
        raise ValueError(f"row counts differ: {len(a)} vs {len(b)}")
    input_keys = [
        k
        for k in a[0]
        if k.startswith("sentence_")
        or k in ("ground_truth", "prompt_variant", "option_order")
    ]
    inputs_same = all(
        all(x[k] == y[k] for k in input_keys) for x, y in zip(a, b, strict=True)
    )
    # v1 tables have no raw argmax token id; fall back to the constrained prediction
    out_key = (
        "argmax_token_id"
        if "argmax_token_id" in a[0] and "argmax_token_id" in b[0]
        else "predicted_primary"
    )
    flips = [
        x["sample_id"] for x, y in zip(a, b, strict=True) if x[out_key] != y[out_key]
    ]
    logit_keys = [k for k in a[0] if k.startswith("primary_logit_")]
    max_logit_delta = max(
        abs(x[k] - y[k]) for x, y in zip(a, b, strict=True) for k in logit_keys
    )
    print(f"== compare {a_id} vs {b_id}: n={len(a)}")
    print(
        f"  {'ok  ' if inputs_same else 'FAIL'} identical sampled inputs/assignments ({input_keys})"
    )
    print(
        f"  {'ok  ' if not flips else 'DIFF'} argmax flips: {len(flips)} {flips[:10]}"
    )
    print(f"  info max |primary logit delta| = {max_logit_delta:.6f}")
    for sid in flips[:10]:
        x, y = a[sid], b[sid]
        margins = [(k, round(x[k], 4), round(y[k], 4)) for k in logit_keys]
        print(f"    sample {sid}: {x[out_key]} -> {y[out_key]}  logits {margins}")


def notebook_functions(path: str, **namespace):
    """Exec only the function definitions of a notebook (same trick as tests/test_notebook_metrics.py)."""
    import ast  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    namespace = {"pd": pd, "np": np, **namespace}
    nb = json.load(open(path))
    defs = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            defs.extend(
                n
                for n in ast.parse("".join(cell["source"])).body
                if isinstance(n, ast.FunctionDef)
            )
    exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), namespace)  # noqa: S102
    return namespace


def swap_pair(api, plain_id: str, swapped_id: str) -> None:
    """Few-shot notebook views on two real ICL runs (swap_labels false / true, same strengths)."""
    import pandas as pd  # noqa: PLC0415

    runs = [api.run(f"{PROJECT}/{rid}") for rid in (plain_id, swapped_id)]
    rows = []
    for run in runs:
        c, s = run.config, run.summary._json_dict
        rows.append(
            {
                "swap": bool(c["prompts"]["turns"]["swap_labels"]),
                "p": c["perturbation"]["dropout_rate"],
                "sigma": c["perturbation"]["noise_std"],
                "accuracy_aggregate": s["accuracy_aggregate"],
                "argmax_is_label_rate": s["argmax_is_label_rate"],
                "accuracy_primary": s["accuracy_primary"],
                "argmax_is_primary_label_rate": s["argmax_is_primary_label_rate"],
                "accuracy_constrained_primary": s["accuracy_constrained_primary"],
            }
        )
    df = pd.DataFrame(rows)
    if (
        df["swap"].tolist() != [False, True]
        or len(set(zip(df["p"], df["sigma"], strict=True))) != 1
    ):
        raise ValueError(
            f"need one plain and one swapped run at the same strengths, got {df[['swap', 'p', 'sigma']].to_dict()}"
        )
    nbf = notebook_functions(
        "notebooks/paper/2_few_shot.ipynb",
        COL_SWAP="swap",
        COL_DROPOUT="p",
        COL_NOISE="sigma",
        LABEL_RATE_COL={
            "accuracy_primary": "argmax_is_primary_label_rate",
            "accuracy_aggregate": "argmax_is_label_rate",
            "accuracy_constrained_primary": None,
        },
    )
    swapped_rows = [r for r in table(runs[1]) if not r["is_baseline"]]
    print(
        f"== swap-pair {plain_id} (swap=False) vs {swapped_id} (swap=True), n={len(swapped_rows)}"
    )
    for col, rate_col, pred_col in [
        ("accuracy_aggregate", "argmax_is_label_rate", "predicted_class_argmax"),
        (
            "accuracy_primary",
            "argmax_is_primary_label_rate",
            "predicted_class_argmax_primary",
        ),
        ("accuracy_constrained_primary", None, "predicted_class_primary"),
    ]:
        _, flipped, _ = nbf["make_heatmap_data"](df, col, 100, True)
        single = nbf["make_single_heatmap"](df, col, 100, True, True)
        expect = (df.loc[1, rate_col] if rate_col else 1.0) - df.loc[1, col]
        from_table = sum(
            r[pred_col] not in ("OTHER", r["test_perturbation"]) for r in swapped_rows
        ) / len(swapped_rows)
        ok = (
            abs(flipped.iloc[0, 0] - expect * 100) <= 1e-9
            and abs(single.iloc[0, 0] - flipped.iloc[0, 0]) <= 1e-9
            and abs(from_table - expect) <= 1e-9
        )
        print(
            f"  {'ok  ' if ok else 'FAIL'} {col}: combined={flipped.iloc[0, 0]:.4f} single={single.iloc[0, 0]:.4f} label-rate-minus-acc={expect * 100:.4f} from-table={from_table * 100:.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*")
    ap.add_argument("--since", help="ISO time; verify all runs created after it")
    ap.add_argument(
        "--compare", nargs=2, metavar="RUN_ID", help="compare two runs sample by sample"
    )
    ap.add_argument(
        "--swap-pair", nargs=2, metavar="RUN_ID", help="plain and swapped-label ICL run"
    )
    args = ap.parse_args()
    api = wandb.Api()
    if args.compare:
        compare(api, *args.compare)
        return
    if args.swap_pair:
        swap_pair(api, *args.swap_pair)
        return
    runs = [api.run(f"{PROJECT}/{rid}") for rid in args.run_ids]
    if args.since:
        runs += list(
            api.runs(
                PROJECT,
                filters={"created_at": {"$gt": args.since}},
                order="+created_at",
            )
        )
    for run in runs:
        s = run.summary._json_dict
        cfg = run.config
        report = [f"== {run.id} {run.name} state={run.state} mode={cfg.get('mode')}"]
        check("metrics_version", s.get("metrics_version"), 2, True, report)
        seed = cfg.get("experiment", {}).get("seed", cfg.get("experiment.seed"))
        report.append(
            f"  {'ok  ' if isinstance(seed, int) else 'FAIL'} seed is int: {seed}"
        )
        report.append(f"  info runtime: {cfg.get('runtime')}")
        rows = table(run)
        report.append(
            f"  info rows={len(rows)} plots={[k for k in s if k.startswith(('localization_', 'confusion', 'icl_', 'multiclass_', 'class_prob'))]}"
        )
        tok = tokenizer_for(cfg)
        if "predicted_argmax_primary" in rows[0]:
            verify_localization(rows, s, report, tok)
        elif "predicted_class_argmax_primary" in rows[0]:
            verify_classification(rows, s, report, tok)
        if "peak_gpu_memory_gb" in s:
            report.append(f"  info peak GPU memory: {s['peak_gpu_memory_gb']:.1f} GB")
        print("\n".join(report))
        print(
            f"  => {'PASS' if not any('FAIL' in line for line in report) else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
