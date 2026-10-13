"""runtime_info must not fail a run outside a git checkout or without the model in the HF cache."""

import subprocess

import pytest

from spe import model_utils


@pytest.fixture
def no_hf_cache(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("not in the local HF cache")

    monkeypatch.setattr(model_utils.huggingface_hub, "snapshot_download", missing)


def test_outside_git_checkout(monkeypatch, tmp_path, no_hf_cache) -> None:
    # GIT_DIR pointing to a missing directory makes every git command fail
    # with "not a git repository", as in an unpacked archive of the repo.
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "missing"))
    info = model_utils.runtime_info("some/model", revision=None)
    assert info["code_commit"] is None
    assert info["code_dirty"] is None
    assert info["hf_commit"] is None
    assert info["torch"] and info["transformers"]


def test_git_not_installed(monkeypatch, no_hf_cache) -> None:
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    info = model_utils.runtime_info("some/model", revision=None)
    assert info["code_commit"] is None and info["code_dirty"] is None


def test_inside_git_checkout(no_hf_cache) -> None:
    try:
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True)  # noqa: S607
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not running from a git checkout")
    info = model_utils.runtime_info("some/model", revision=None)
    assert isinstance(info["code_commit"], str) and len(info["code_commit"]) == 40
    assert isinstance(info["code_dirty"], bool)
