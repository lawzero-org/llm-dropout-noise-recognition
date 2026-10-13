"""The 32 runs of the log-probability analysis (paper appendix, localization second controls).

8 models x {dropout, noise} x {low, high} strength, each run by ``scripts/shortcut_analysis.py``:
low = p_min / sigma_min (bin 0 of the model's Table 1 range), high = the midpoint of the range (bin 5).
The 11 strengths of each range are the ones of the model's zero-shot sweep, ``sweeps/<model>/zs_<pert>.yaml``.

    uv run python scripts/logprob_runs.py          # one line of Hydra overrides per run
    uv run python scripts/shortcut_analysis.py $(uv run python scripts/logprob_runs.py | sed -n 1p)
"""

import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
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
STRENGTH = {"dropout": "perturbation.dropout_rate", "noise": "perturbation.noise_std"}
BINS = {"low": 0, "high": 5}


def overrides() -> list[list[str]]:
    """Hydra overrides of every run, in a fixed order (model, perturbation, low / high)."""
    runs = []
    for model in MODELS:
        for pert, key in STRENGTH.items():
            sweep = yaml.safe_load(
                (REPO / f"sweeps/{model}/zs_{pert}.yaml").read_text()
            )
            strengths = sweep["parameters"][key]["values"]
            for regime, b in BINS.items():
                runs.append(
                    [
                        f"model={model}",
                        f"perturbation.target_perturbation={pert.upper()}",
                        f"{key}={strengths[b]}",
                        "experiment.sentences_file=data/sentences/15tok.txt",
                        "experiment.num_samples=1000",
                        f"experiment.stage=logprob_{regime}",
                    ]
                )
    return runs


if __name__ == "__main__":
    for run in overrides():
        print(" ".join(run))
