"""Download the OLMo-3 32B pretraining checkpoints used by the pretraining-steps figure into HF_HOME.

    HF_HOME=<cache> uv run python scripts/download_olmo_checkpoints.py

Revisions: the ten `stage1-step*` tags of the sweeps in `sweeps/olmo_steps/`, ~61 GB each.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import snapshot_download

MODEL = "allenai/Olmo-3-1125-32B"
STEPS = [0, 73000, 145000, 219000, 292000, 364000, 437000, 510000, 583000, 656000]

for step in STEPS:
    rev = f"stage1-step{step}"
    print(f"downloading {MODEL}@{rev} ...", flush=True)
    path = snapshot_download(
        MODEL,
        revision=rev,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
    )
    print(f"done {rev}: {path}", flush=True)
print("ALL DONE", flush=True)
