"""Download the paper's models at the revisions pinned in ``src/spe/conf/model/*.yaml``.

    HF_HOME=<cache> uv run python scripts/download_models.py            # every model config
    HF_HOME=<cache> uv run python scripts/download_models.py qwen3_32b  # selected configs

The OLMo pretraining checkpoints of the appendix (revisions ``stage1-step<N>``) are downloaded by
``scripts/download_olmo_checkpoints.py``.
"""

import os
import pathlib
import sys

import yaml

os.environ["HF_HUB_OFFLINE"] = "0"

from huggingface_hub import snapshot_download  # noqa: E402

CONF = pathlib.Path(__file__).resolve().parents[1] / "src/spe/conf/model"


def main() -> None:
    names = sys.argv[1:] or sorted(p.stem for p in CONF.glob("*.yaml"))
    for name in names:
        cfg = yaml.safe_load((CONF / f"{name}.yaml").read_text())
        print(f"Downloading {cfg['name']} @ {cfg.get('revision')} ({name})...")
        snapshot_download(cfg["name"], revision=cfg.get("revision"))
        print(f"Done: {name}\n")


if __name__ == "__main__":
    main()
