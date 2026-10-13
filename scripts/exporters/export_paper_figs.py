"""Export the paper's data figures (everything except the OLMo and log-prob figures) as pgfplots .tex files.

    uv run python scripts/exporters/export_paper_figs.py                               # write ./plots/...
    uv run python scripts/exporters/export_paper_figs.py --paper <paper repo> --check   # diff against the paper

The data come from the local W&B caches of the manifest sweeps (``notebooks/wandb_cache.py``). The curves and
heatmaps are computed and formatted by the paper notebooks' own cells (``notebooks/paper/*.ipynb``, run headless);
the few layouts that no notebook cell emits live in ``scripts/exporters/templates/``. A figure file that already
exists in the paper keeps its caption, label and float placement, so edits made in the paper survive a re-export.
``--check`` compares with the paper's files, ignoring comment lines. The OLMo figures come from
``export_olmo_tex.py``, the log-prob figures from ``export_shortcut_tex.py``.
"""

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).resolve().parent / "templates"
MODELS = [
    "gemma4_31b",
    "llama3_8b",
    "mistral_small_24b",
    "olmo3_32b",
    "qwen3_14b",
    "qwen3_32b",
    "qwen3_8_27b",
    "seed_oss_36b",
]
# answers "neither": its control curves use the " A" vs " B" accuracy (Table 1)
CONSTRAINED = {"olmo3_32b"}
LAYER_THIRD = {0: "first", 1: "second", 2: "third"}
THIRD_COLOR = {
    "first": "blue!60!black",
    "second": "green!55!black",
    "third": "red!55!black",
}


# --------------------------------------------------------------------------------------------------------------------
# notebooks and output
# --------------------------------------------------------------------------------------------------------------------


def notebook(name: str, cells: list[int]) -> dict:
    """Run code cells of ``notebooks/paper/<name>`` headless (widgets and prints silenced); return the namespace."""
    os.chdir(REPO / "notebooks/paper")
    sys.path.insert(0, str(REPO / "notebooks"))
    ns: dict = {"__name__": "__nb__"}
    for i in cells:
        run_cell(name, i, ns)
    return ns


def run_cell(name: str, i: int, ns: dict) -> None:
    """Execute code cell ``i`` of a paper notebook in ``ns``, output silenced."""
    nb = json.loads((REPO / "notebooks/paper" / name).read_text())
    src = "".join(nb["cells"][i]["source"])
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, f"{name}[{i}]", "exec"), ns)  # noqa: S102 -- the notebooks' own code


def _body(text: str | None) -> list[str]:
    return [line for line in (text or "").splitlines() if not line.startswith("%")]


def resized(tex: str, old: str, new: str) -> str:
    """Change one size of a main-text figure (the notebooks draw the appendix at their default sizes)."""
    if tex.count(old) < 1:
        raise ValueError(f"size {old!r} not found")
    return tex.replace(old, new)


def two_row_legend(tex: str) -> str:
    """Break the model legend of Figs. 2 and 3 after its fourth entry (4 models above, 4 below)."""
    lines = tex.split("\n")
    entries = [
        i
        for i, line in enumerate(lines)
        if line.startswith("\\tikz\\draw[") and "\\model{" in line
    ]
    if len(entries) != 8 or not lines[entries[3]].endswith("\\hspace{1em}"):
        raise ValueError("expected a legend of 8 models")
    lines[entries[3]] = lines[entries[3]][: -len("\\hspace{1em}")] + "\\\\"
    return "\n".join(lines)


def capture(fn: Callable[[], object]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


_CAPTION = re.compile(r"\\caption\{.*?\}\n\\label\{[^}]*\}", re.S)
_PLACEMENT = re.compile(r"\\begin\{figure\}\[[^\]]*\]")


class Output:
    """Collects the generated files; writes them into the paper, or compares them with it."""

    def __init__(self, paper: Path, check: bool) -> None:
        self.paper, self.check, self.diffs, self.n = paper, check, [], 0

    def __call__(self, rel: str, text: str) -> None:
        text = text.rstrip("\n") + "\n"
        path = self.paper / rel
        old = path.read_text() if path.exists() else None
        for pattern in (_CAPTION, _PLACEMENT):
            kept = pattern.findall(old or "")
            parts = pattern.split(text)
            if kept and len(kept) == len(parts) - 1:
                text = "".join(p + k for p, k in zip(parts, [*kept, ""], strict=True))
        self.n += 1
        if _body(old) == _body(text):
            return
        self.diffs.append(rel)
        if not self.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


def fill(template: str, panels: list[list[str]], **fields: str) -> str:
    """Fill a layout of ``templates/``: one list of plot lines for every %%PLOTS%% line, plus the @@NAME@@ fields."""
    out, it = [], iter(panels)
    for line in (TEMPLATES / template).read_text().splitlines():
        if line == "%%PLOTS%%":
            out.extend(next(it))
        elif not line.startswith("%%"):
            out.append(line)
    text = "\n".join(out) + "\n"
    for k, v in fields.items():
        text = text.replace(f"@@{k}@@", v)
    if next(it, None) is not None or "@@" in text:
        raise ValueError(f"{template}: panels or fields do not match the layout")
    return text


def band_and_line(x, y, e, color: str, main: bool, fmt: str = ".4f") -> list[str]:
    """The notebooks' curve style: a shaded +-SE band and the line (main aliases solid with dots, controls dashed)."""
    up = " ".join(f"({a:.1f},{b + c:{fmt}})" for a, b, c in zip(x, y, e, strict=True))
    lo = " ".join(
        f"({a:.1f},{b - c:{fmt}})"
        for a, b, c in reversed(list(zip(x, y, e, strict=True)))
    )
    style = (
        "mark=*, mark size=1, line width=1.5pt" if main else "dashed, line width=0.8pt"
    )
    return [
        f"\\addplot[{color}, fill={color}, fill opacity={'0.3' if main else '0.15'}, draw=none, forget plot] coordinates {{{up} {lo}}} --cycle;",
        f"\\addplot[{color}, {style}, forget plot] coordinates {{"
        + " ".join(f"({a:.1f},{b:{fmt}})" for a, b in zip(x, y, strict=True))
        + "};",
    ]


# --------------------------------------------------------------------------------------------------------------------
# localization (Sec. 4 and appendix)
# --------------------------------------------------------------------------------------------------------------------


def localization(out: Output) -> None:
    import pandas as pd

    ns = notebook("0_localization.ipynb", [1, 2, 3, 4])
    cc, n, load = ns["compute_curves"], ns["N_PER_RUN"], ns["_load_key"]
    cd, cn, cs, cp = (
        ns["COL_DROPOUT"],
        ns["COL_NOISE"],
        ns["COL_SENTENCES"],
        ns["COL_PROMPT"],
    )
    fig = ns["latex_figure"]
    ml = {m: ns["MODEL_MACROS_LATEX"][m][0] for m in MODELS}

    def curves(df, x, metric, models, group=cs, constrained_content=False):
        c = cc(
            df,
            x,
            metric,
            group,
            "all",
            n,
            [m for m in models if not (constrained_content and m in CONSTRAINED)],
        )
        if constrained_content:
            c.update(
                cc(
                    df,
                    x,
                    metric.replace("_primary", "_constrained_primary"),
                    group,
                    "all",
                    n,
                    [m for m in models if m in CONSTRAINED],
                )
            )
        return c

    def controls(models):
        d = curves(ns["df_ctrl_d"], cd, "content_accuracy_primary", models, cp, True)
        s = ns["clip_curves"](
            curves(ns["df_ctrl_n"], cn, "content_accuracy_primary", models, cp, True),
            None,
            0.5,
        )
        return d, s

    main_d = curves(
        ns["df_main_d"], cd, "accuracy_primary", MODELS
    )  # Fig. 2: every model
    main_n = curves(ns["df_main_n"], cn, "accuracy_primary", MODELS)
    cap_main = ns["PAPER_CAPTIONS"]["Main"][0]
    order = [
        "llama3_8b",
        "qwen3_32b",
        "gemma4_31b",
        "mistral_small_24b",
        "olmo3_32b",
        "qwen3_14b",
        "qwen3_8_27b",
        "seed_oss_36b",
    ]  # qwen32 early: drawn under the others
    out(
        "plots/00_loc/00_loc_main_15_tok.tex",
        two_row_legend(
            resized(
                fig(
                    main_d,
                    main_n,
                    "accuracy_primary",
                    "SE",
                    order,
                    cap_main,
                    "loc-main-15tok",
                    ymin=0,
                    ymax=100,
                ),
                "    height=5cm,",
                "    height=4.2cm,",
            )
        ),
    )
    d, s = controls(order)  # Fig. 3: every model
    out(
        "plots/00_loc/01_loc_controls.tex",
        two_row_legend(
            resized(
                fig(
                    d,
                    s,
                    "content_accuracy_primary",
                    "SE",
                    order,
                    "",
                    "loc-control-pooled",
                    ref_lines=[95],
                    ymin=50,
                    ymax=100,
                ),
                "    height=5cm,",
                "    height=4.2cm,",
            )
        ),
    )
    met = "accuracy_constrained_primary"
    od = cc(ns["df_main_d"], cd, met, cs, "all", n, ["olmo3_32b"])
    on = cc(ns["df_main_n"], cn, met, cs, "all", n, ["olmo3_32b"])
    tex = fig(
        od, on, met, "SE", ["olmo3_32b"], "", "olmo_primary_accuracy", ymin=0, ymax=100
    )
    out(
        "plots/11_appendix/localization/04_primary_accuracy_olmo32.tex",
        tex.replace("\\begin{figure}[ht]", "\\begin{figure}[!h]", 1),
    )

    # prompt ablations and AIAYN dropout (same figure format, the 8 models)
    for key, rel, label in (
        (
            "noprime",
            "30_loc_prompt_ablation_no_announcement",
            "loc-prompt-ablation-no-announcement",
        ),
        (
            "bare",
            "32_loc_prompt_ablation_sentences_and_question",
            "loc-prompt-ablation-sentences-and-question",
        ),
    ):
        pd_ = curves(load(f"{key}_dropout"), cd, "accuracy_primary", MODELS)
        pn_ = curves(load(f"{key}_noise"), cn, "accuracy_primary", MODELS)
        out(
            f"plots/11_appendix/localization/{rel}.tex",
            fig(
                pd_, pn_, "accuracy_primary", "SE", MODELS, "", label, ymin=0, ymax=100
            ),
        )
    std = curves(load("loc_dropout"), cd, "accuracy_primary", MODELS)
    aiayn = curves(load("aiayn_dropout"), cd, "accuracy_primary", MODELS)
    tex = fig(
        std, aiayn, "accuracy_primary", "SE", MODELS, "", "loc-aiayn", ymin=0, ymax=100
    )
    tex = tex.replace(ns["NOISE_XLABEL"]["latex"], "Dropout rate $p$ (AIAYN)").replace(
        ns["DROPOUT_XLABEL"]["latex"], "Dropout rate $p$ (output of the sublayers)", 1
    )
    out("plots/11_appendix/localization/31_loc_aiayn.tex", tex)

    # sentence length: the notebook's token-count cell on the 15-token runs + the other lengths
    tok_d = pd.concat([ns["df_main_d"], load("toklen_dropout")], ignore_index=True)
    tok_n = pd.concat([ns["df_main_n"], load("toklen_noise")], ignore_index=True)
    ns["df_main_d"], ns["df_main_n"] = tok_d, tok_n
    ns["sent_map"] = {ns["_stem"](sf): sf for sf in sorted(tok_d[cs].unique())}
    run_cell("0_localization.ipynb", 10, ns)
    for i, m in enumerate(MODELS):
        d = ns["_compute_tok_curves"](tok_d, cd, "accuracy_primary", m)
        s = ns["_compute_tok_curves"](tok_n, cn, "accuracy_primary", m)
        tex = ns["_latex_tok_figure"](
            d, s, "accuracy_primary", "SE", m, ymin=0, ymax=100
        )
        tex = _CAPTION.sub(
            lambda _, m=m: (
                f"\\caption{{Localization: {ml[m]}.}}\n\\label{{fig:loc_tok_{m}}}"
            ),
            tex,
            count=1,
        )
        key = ml[m][len("\\model{") : -1]
        out(f"plots/11_appendix/localization/2{i}_token_count_{key}.tex", tex)

    # layer thirds: 4 x 4 groupplot, each panel = the three thirds + all layers over the Table 1 range
    lines = [
        r"\centering",
        "\\tikz\\draw[gray, dashed, line width=1pt] plot coordinates {(0,0) (0.4,0)}; {\\small all layers}\\hspace{1em}"
        + "\\hspace{1em}".join(
            f"\\tikz\\draw[{THIRD_COLOR[t]}, thick, mark=*, mark size=1.5] plot coordinates {{(0,0) (0.4,0)}}; {{\\small {t} third}}"
            for t in ("first", "second", "third")
        )
        + r"\\[4pt]",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"    group style={group size=4 by 4, horizontal sep=0.9cm, vertical sep=1.3cm},",
        r"    width=0.29\linewidth, height=3.2cm, ymin=0, ymax=100, grid=major,",
        r"    tick label style={font=\tiny}, title style={font=\scriptsize, yshift=-0.8ex},",
        r"    scaled x ticks=false, xticklabel style={/pgf/number format/fixed},",
        r"]",
    ]
    for m in MODELS:
        for pert, x in (("dropout", cd), ("noise", cn)):
            lay = load(f"layers_{pert}")
            lay = lay[lay[ns["COL_MODEL"]] == m]
            first_layer = lay["perturbation.first_layer"].astype(int)
            lo, hi = lay[x].min(), lay[x].max()
            base = load(f"loc_{pert}")
            base = base[
                (base[ns["COL_MODEL"]] == m)
                & (base[x] >= lo - 1e-9)
                & (base[x] <= hi + 1e-9)
            ]
            lines.append(
                f"\\nextgroupplot[title={{{ml[m]} $\\cdot$ {pert.capitalize()}}}, xmin={lo:g}, xmax={hi:g}, xtick={{{lo:g},{hi:g}}}, xticklabels={{{{{lo:g}}},{{{hi:g}}}}}]"
            )
            g = base.groupby(x)["accuracy_primary"].mean() * 100
            lines.append(
                "\\addplot[gray, dashed, line width=1pt, forget plot] coordinates {"
                + " ".join(f"({i:g},{v:.2f})" for i, v in g.items())
                + "};"
            )
            for j, f0 in enumerate(sorted(first_layer.unique())):
                t = (
                    lay[first_layer == f0]
                    .groupby(x)["accuracy_primary"]
                    .agg(["mean", "count"])
                    .sort_index()
                )
                p = t["mean"]
                se = np.sqrt(p * (1 - p) / (t["count"] * n)) * 100
                y = p * 100
                c = THIRD_COLOR[LAYER_THIRD[j]]
                up = " ".join(
                    f"({i:g},{v + e:.2f})"
                    for i, v, e in zip(y.index, y, se, strict=True)
                )
                dn = " ".join(
                    f"({i:g},{v - e:.2f})"
                    for i, v, e in reversed(list(zip(y.index, y, se, strict=True)))
                )
                lines.append(
                    f"\\addplot[{c}, fill={c}, fill opacity=0.3, draw=none, forget plot] coordinates {{{up} {dn}}} --cycle;"
                )
                lines.append(
                    f"\\addplot[{c}, mark=*, mark size=0.8, line width=0.8pt, forget plot] coordinates {{"
                    + " ".join(f"({i:g},{v:.2f})" for i, v in y.items())
                    + "};"
                )
            lines.append(
                f"\\addplot[gray, dashed, line width=0.4pt, forget plot] coordinates {{({lo:g},50) ({hi:g},50)}};"
            )
    lines += [r"\end{groupplot}", r"\end{tikzpicture}"]
    out("plots/11_appendix/layer_ablation/layer_ablation.tex", "\n".join(lines))


# --------------------------------------------------------------------------------------------------------------------
# zero-shot classification (Sec. 6 and appendix)
# --------------------------------------------------------------------------------------------------------------------


def zero_shot(out: Output) -> None:
    ns = notebook("1_zero_shot.ipynb", [1, 2, 3, 4, 7])
    cc, n, ml = ns["compute_curves"], ns["N_STOCHASTIC"], ns["MODEL_LATEX"]
    main = ["dropout_noise", "masking_jitter"]
    color = {
        "dropout_noise": "red!70!black",
        "masking_jitter": "red!40",
        "foo_bar": "brown!70!black",
        "rotation_permutation": "blue!60!black",
        "vanilla_chocolate": "green!70!black",
        "x_y": "cyan!40!blue!70",
    }

    def panel(model, df, x, metric, aliases, chance=True, palette=None):
        c = cc(df, x, metric, aliases, n, [model])
        lines = [
            l
            for a in aliases
            if a in c and model in c[a]
            for l in band_and_line(*c[a][model][:3], (palette or color)[a], a in main)
        ]
        return lines + (
            [
                r"\addplot[gray, dashed, line width=0.5pt, forget plot] coordinates {(0,50) (100,50)};"
            ]
            if chance
            else []
        )

    d, s, cd, cn = ns["df_all_d"], ns["df_all_n"], ns["COL_DROPOUT"], ns["COL_NOISE"]
    ctrl = ["foo_bar", "rotation_permutation", "vanilla_chocolate", "x_y"]
    curves = ns["combined_curves"]("dropout_noise", "accuracy_primary", MODELS)
    bars = ns["combined_bars"](curves, "SE")
    bars_mj = ns["combined_bars"](
        ns["combined_curves"]("masking_jitter", "accuracy_primary", MODELS), "SE"
    )
    green = {"dropout_noise": "pastelgreendark", "masking_jitter": "pastelgreen"}
    qwens = {"qwen3_32b", "qwen3_8_27b"}

    def bar_plot(rows, fill_color, keep, shift):
        body = " ".join(
            f"({ml[m]},{v:.2f}) +- (0,{e:.2f})" for m, v, e in rows if keep(m)
        )
        return f"\\addplot[{fill_color}, fill={fill_color}, fill opacity=0.8, bar shift={shift}, error bars/.cd, y dir=both, y explicit, error bar style={{black, line width=0.6pt}}] coordinates {{{body}}};"

    out(
        "plots/01_zero_shot/01_zero_shot_average_qwen32_gemma31.tex",
        fill(
            "zero_shot_main.tex",
            [
                [
                    bar_plot(bars, "red!70!black", lambda m: m not in qwens, "-2.5pt"),
                    bar_plot(bars, "pastelgreendark", lambda m: m in qwens, "-2.5pt"),
                    bar_plot(bars_mj, "red!40", lambda m: m not in qwens, "2.5pt"),
                    bar_plot(bars_mj, "pastelgreen", lambda m: m in qwens, "2.5pt"),
                ],
                panel("qwen3_32b", d, cd, "accuracy_primary", main, palette=green),
                panel("qwen3_32b", s, cn, "accuracy_primary", main, palette=green),
                panel("gemma4_31b", d, cd, "accuracy_primary", main),
                panel("gemma4_31b", s, cn, "accuracy_primary", main),
            ],
            MODELS=",".join(ml[m] for m, _, _ in bars),
        ),
    )
    out(  # the same four controls in the left wrap figure of Sec. 6.2, below the prompt
        "plots/01_zero_shot/02_zero_shot_controls_qwen32_wrap.tex",
        fill(
            "zero_shot_controls_wrap.tex",
            [
                panel("qwen3_32b", d, cd, "accuracy_primary", ctrl),
                panel("qwen3_32b", s, cn, "accuracy_primary", ctrl),
            ],
        ),
    )
    out(
        "plots/11_appendix/zero_shot/00_zero_shot_controls_qwen32.tex",
        fill(
            "zero_shot_controls_qwen32.tex",
            [
                panel("qwen3_32b", d, cd, "accuracy_primary", ctrl),
                panel("qwen3_32b", s, cn, "accuracy_primary", ctrl),
            ],
        ),
    )

    four = ["dropout_noise", "masking_jitter", "rotation_permutation", "x_y"]
    ent = [cc(df, x, "entropy_mean", four, n, MODELS) for df, x in ((d, cd), (s, cn))]
    top = (
        math.ceil(
            max(
                float(np.nanmax(v[1]))
                for c in ent
                for per_model in c.values()
                for v in per_model.values()
            )
            * 2
        )
        / 2
    )
    ticks = ",".join(f"{v:g}" for v in np.round(np.linspace(0, top, 6), 2))
    pairs = [
        ("gemma4_31b", "llama3_8b"),
        ("mistral_small_24b", "olmo3_32b"),
        ("qwen3_14b", "qwen3_8_27b"),
        ("qwen3_32b", "seed_oss_36b"),
    ]
    for i, (a, b) in enumerate(pairs):
        ka, kb = ml[a][len("\\model{") : -1], ml[b][len("\\model{") : -1]
        for kind, metric, ylabel, yrange, bands, rel, cap, label in (
            (
                "acc",
                "accuracy_primary",
                "Accuracy (\\%)",
                "ymin=0.0, ymax=100.0,\n    ytick={0,20,40,60,80,100},",
                True,
                f"accuracy/1{i}_zero_shot_{ka}_{kb}",
                f"Zero-shot classification accuracy of {ml[a]} (left pair) and {ml[b]} (right pair).",
                f"fig:zero_shot_{ka}_{kb}",
            ),
            (
                "ent",
                "entropy_mean",
                "Entropy (nats)",
                f"ymin=0.0, ymax={top},\n    ytick={{{ticks}}},",
                False,
                f"entropy/1{i}_entropy_{ka}_{kb}",
                f"Zero-shot: entropy of the token distribution for {ml[a]} (left pair) and {ml[b]} (right pair).",
                f"fig:zero_shot_entropy_{ka}_{kb}",
            ),
        ):
            panels = []
            for m in (a, b):
                for df, x in ((d, cd), (s, cn)):
                    c = cc(df, x, metric, four, n, [m])
                    lines = []
                    for al in four:
                        if al in c and m in c[al]:
                            bl = band_and_line(*c[al][m][:3], color[al], al in main)
                            lines += bl if bands else bl[1:]
                    panels.append(
                        lines
                        + (
                            [
                                r"\addplot[gray, dashed, line width=0.5pt, forget plot] coordinates {(0,50) (100,50)};"
                            ]
                            if kind == "acc"
                            else []
                        )
                    )
            out(
                f"plots/11_appendix/zero_shot/{rel}.tex",
                fill(
                    "zero_shot_pair.tex",
                    panels,
                    MODEL_A=ml[a],
                    MODEL_B=ml[b],
                    YLABEL=ylabel,
                    YRANGE=yrange,
                    CAPTION=f"\\caption{{\\small {cap}}}\n\\label{{{label}}}",
                ),
            )


# --------------------------------------------------------------------------------------------------------------------
# few-shot classification (Sec. 7 and appendix)
# --------------------------------------------------------------------------------------------------------------------


def few_shot(out: Output) -> None:
    ns = notebook("2_few_shot.ipynb", [1, 2, 3, 4, 5, 9, 11, 13, 14])
    ml = ns["MODEL_MACROS_LATEX"]
    key = {m: ml[m][len("\\model{") : -1] for m in MODELS}
    z = ns["Z_METRICS"][f"{ns['COL_ACC']} (%)"]
    gen, vlim, font = (
        ns["generate_single_heatmap_latex"],
        ns["FIG6_VLIM"],
        ns["FIG6_CELL_FONT"],
    )
    dm, dca, colm, colk, cola = (
        ns["df_main"],
        ns["df_ctrl_all"],
        ns["COL_MODEL"],
        ns["COL_NUM_PAIRS"],
        ns["COL_ALIASES"],
    )

    def sel(df, model, k, alias=None):
        s = df[(df[colm] == model) & (df[colk] == k)]
        return s if alias is None else s[s[cola] == alias]

    def heat(model, k, swap):
        return ns["make_single_heatmap"](
            sel(dm, model, k), z["col"], z["scale"], z["flip_swap"], swap
        )

    def delta(model, alias):
        return ns["make_heatmap_data"](
            sel(dca, model, 1, alias), z["col"], z["scale"], z["flip_swap"]
        )[2]

    def grid_panel(h, cmap, vmin, vmax, title, pos):
        row, col = divmod(pos, 2)
        return "\\centering\n" + gen(
            h,
            cmap,
            vmin,
            vmax,
            title=title,
            show_xlabel=(row == 1),
            show_ylabel=(col == 0),
            show_colorbar=(col == 1),
            rotate_xticks=False,
            title_yshift="-1.0ex",
            cell_font=font,
        )

    # main text: Fig. 6 (heatmap + learning curves) and the qwen32 delta panels (appendix)
    out(
        "plots/02_few_shots/00_heatmap_qwen32.tex",
        "\\centering\n"
        + resized(
            gen(
                heat("qwen3_32b", 1, False),
                "pastel_rg",
                0,
                100,
                title=ml["qwen3_32b"],
                rotate_xticks=False,
            ),
            "0.82\\linewidth,",
            "0.72\\linewidth,",
        ),
    )
    ns["ld_w_swap"].value = "Both"
    text = capture(lambda: setattr(ns["ld_w_swap"], "value", "swap=False only"))
    fig = text[text.rindex("\\begin{figure}") : text.rindex("\\end{figure}")]
    body = fig.split("\\centering\n", 1)[1]
    plots = [line for line in body.splitlines() if line.startswith("\\addplot")]
    out("plots/02_few_shots/00_examples_all.tex", fill("examples_all.tex", [plots]))
    titles, files = ns["FIG6_TITLES"], ns["FIG6_FILES"]
    for pos, alias in enumerate(ns["FIG6_ALIASES"]):
        tex = grid_panel(
            delta("qwen3_32b", alias), "pastel_po", -vlim, vlim, titles[alias], pos
        )
        out(
            f"plots/02_few_shots/{files[alias]}.tex",
            f"% ===== plots/02_few_shots/{files[alias]}.tex  (qwen3_32b, k=1) =====\n"
            + tex,
        )

    def pair_panel(tex, title, left):
        """One heatmap of the 2 x 2 grid in the main text: a fixed axis size, and the y label (left column)
        or the colorbar (right column); the x label comes from grid_panel (bottom row)."""
        old = "    width=0.82\\linewidth,\n    height=0.82\\linewidth,\n"
        if old not in tex or tex.count("    title={") != 1:
            raise ValueError("delta panel: axis options changed in the notebook")
        tex = tex.replace(
            old, "    scale only axis,\n    width=3.3cm,\n    height=3.3cm,\n"
        )
        if left:
            tex = tex.replace(
                "    ylabel={\\small Noise SD $\\sigma$},\n",
                "    ylabel={\\small Noise SD $\\sigma$},\n    ylabel near ticks,\n    ylabel shift={-3pt},\n",
            )
        i = tex.index("    title={")
        return tex[:i] + f"    title={{{title}}},\n" + tex[tex.index("\n", i) + 1 :]

    for pos, (alias, title) in enumerate(
        (
            ("dropout_noise", "Dropout / Noise"),
            ("masking_jitter", "Masking / Jitter"),
            ("x_y", "X / Y"),
            ("rotation_permutation", "Rotation / Permutation"),
        )
    ):
        panel = grid_panel(
            delta("qwen3_32b", alias), "pastel_po", -vlim, vlim, titles[alias], pos
        )
        out(
            f"plots/02_few_shots/04_delta_qwen32_{alias}.tex",
            pair_panel(
                panel,
                title,
                pos % 2 == 0,
            ),
        )

    # correct vs swapped labels for every model, two rows of four
    tf, tt = ns["ldt_false"], ns["ldt_true"]
    vals = [v for t in (tf, tt) for v in t[t.model.isin(MODELS)]["acc_pct"]]
    ymin, ymax = 5 * int(min(vals) // 5), 5 * int(-(-max(vals) // 5))
    colors = ns["MODEL_COLORS_LATEX"]

    def curve(t, m, c, op, style):
        t = t[t["model"] == m].sort_values("num_pairs")
        pts = list(
            zip(t["num_pairs"], t["acc_pct"], t["acc_pooled_se_pct"], strict=True)
        )
        up = " ".join(f"({int(x)},{y + e:.2f})" for x, y, e in pts)
        lo = " ".join(f"({int(x)},{y - e:.2f})" for x, y, e in reversed(pts))
        return [
            f"\\addplot[{c}, fill={c}, fill opacity={op}, draw=none, forget plot] coordinates {{{up} {lo}}} --cycle;",
            f"\\addplot[{c}, {style}, forget plot] coordinates {{"
            + " ".join(f"({int(x)},{y:.2f})" for x, y, _ in pts)
            + "};",
        ]

    # every model in one row, shaped like the right panel of the few-shot figure; eight fit in the text width only
    # without stray spaces and with the boxes trimmed to the axes
    for pos, m in enumerate(MODELS):
        k = key[m]
        lines = [
            r"\tikzset{external/export next=false}%",
            r"\begin{tikzpicture}[trim axis right]"
            if pos == 0
            else r"\begin{tikzpicture}[trim axis left, trim axis right]",
            r"\begin{axis}[",
            r"    scale only axis,",
            r"    width=1.75cm,",
            r"    height=2.65cm,",
            r"    grid=major,",
            f"    title={{{ml[m]}}},",
            r"    title style={font=\scriptsize, yshift=-1.2ex},",
            r"    ylabel={\scriptsize Accuracy (\%)}, ylabel style={yshift=-0.8ex},"
            if pos == 0
            else r"    ylabel={}, yticklabels={},",
            f"    xtick={{0,1,3,5,7,9}}, xmin=0, xmax=9, ymin={ymin}, ymax={ymax},",
            r"    tick label style={font=\tiny},",
            r"]",
        ]
        lines += curve(tf, m, colors[m], "0.3", "mark=*, mark size=1, line width=1.2pt")
        lines += curve(
            tt, m, colors[m], "0.15", "dashed, mark=o, mark size=1, line width=0.9pt"
        )
        lines += [
            r"\addplot[gray, dashed, line width=0.5pt, forget plot] coordinates {(0,50) (9,50)};",
            r"\end{axis}",
            r"\end{tikzpicture}%",
        ]
        out(f"plots/02_few_shots/03_examples_flipped_{k}.tex", "\n".join(lines))

    # appendix: qwen32 with swapped labels (heatmap + curves), then every model
    out(
        "plots/02_few_shots/01_heatmap_qwen32_flipped.tex",
        "\\centering\n"
        + gen(
            heat("qwen3_32b", 1, True),
            "pastel_rg",
            0,
            100,
            title=ml["qwen3_32b"],
            rotate_xticks=False,
        ),
    )
    out(
        "plots/02_few_shots/01_examples_qwen32.tex",
        fill(
            "examples_qwen32_flipped.tex",
            [
                curve(
                    tf,
                    "qwen3_32b",
                    "blue!70!black",
                    "0.3",
                    "mark=o, mark size=1.5, line width=1.5pt",
                )
                + curve(
                    tt,
                    "qwen3_32b",
                    "blue!70!black",
                    "0.15",
                    "dashed, mark=o, mark size=1, line width=1pt",
                )
                + [
                    r"\addplot[gray, dashed, line width=0.5pt, forget plot] coordinates {(0,50) (9,50)};"
                ]
            ],
        ),
    )
    for g in range(2):
        for pos, m in enumerate(MODELS[4 * g : 4 * g + 4]):
            i = MODELS.index(m)
            out(
                f"plots/11_appendix/few_shot/accuracy/2{i}_heatmap_{key[m]}_k1.tex",
                grid_panel(heat(m, 1, False), "pastel_rg", 0, 100, ml[m], pos),
            )
            out(
                f"plots/11_appendix/few_shot/accuracy/3{i}_heatmap_{key[m]}_k9.tex",
                grid_panel(heat(m, 9, False), "pastel_rg", 0, 100, ml[m], pos),
            )
            out(
                f"plots/11_appendix/few_shot/accuracy/4{i}_heatmap_{key[m]}_k1_flipped.tex",
                grid_panel(heat(m, 1, True), "pastel_rg", 0, 100, ml[m], pos),
            )
            out(
                f"plots/11_appendix/few_shot/delta/1{i}_delta_{key[m]}.tex",
                grid_panel(
                    delta(m, "dropout_noise"), "pastel_po", -vlim, vlim, ml[m], pos
                ),
            )
    for m in MODELS:
        if m != "qwen3_32b":
            for pos, alias in enumerate(ns["FIG6_ALIASES"]):
                out(
                    f"plots/11_appendix/few_shot/delta/alias/{key[m]}_{alias}.tex",
                    grid_panel(
                        delta(m, alias), "pastel_po", -vlim, vlim, titles[alias], pos
                    ),
                )
    text = capture(lambda: setattr(ns["ld_w_swap"], "value", "Both"))
    fig = text[text.rindex("\\begin{figure}") : text.rindex("\\end{figure}")]
    body = fig.split("\\centering\n", 1)[1]
    out(
        "plots/11_appendix/few_shot/accuracy/50_examples_all_flipped.tex",
        "\\centering\n" + body[: body.index("\\caption{")].rstrip(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--paper",
        type=Path,
        default=Path("."),
        help="root under which plots/ is written (default: the current directory); "
        "with --check, the root of the paper's LaTeX sources",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare with the paper's files instead of writing",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        choices=("localization", "zero_shot", "few_shot"),
        default=None,
    )
    args = ap.parse_args()
    out = Output(args.paper.resolve(), args.check)
    for name, fn in (
        ("localization", localization),
        ("zero_shot", zero_shot),
        ("few_shot", few_shot),
    ):
        if args.only is None or name in args.only:
            with contextlib.redirect_stdout(io.StringIO()):
                fn(out)
    if args.check:
        print(
            f"{out.n} figure files, {len(out.diffs)} differ from the paper"
            + "".join(f"\n  {d}" for d in out.diffs)
        )
        raise SystemExit(1 if out.diffs else 0)
    print(
        f"{out.n} figure files, {len(out.diffs)} written into {out.paper}"
        + "".join(f"\n  {d}" for d in out.diffs)
    )


if __name__ == "__main__":
    main()
