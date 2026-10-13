"""Shared fixtures for end-to-end experiment tests.

Provides a real tokenizer (lightweight, no GPU required) and a fake model
with the minimum transformer layer structure needed by the hook system.
All fixtures that mock model inference live here so test files only need
to request the composite fixtures (``classification_mocks``, etc.).
"""

from unittest.mock import patch

import matplotlib
import pytest
import torch
from transformers import AutoTokenizer

# Use non-interactive backend so plots render without a display.
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
NUM_FAKE_LAYERS = 4
HIDDEN_DIM = 64


# ---------------------------------------------------------------------------
# Fake model (satisfies hook registration and layer resolution)
# ---------------------------------------------------------------------------


class _FakeSubmodule(torch.nn.Module):
    """Minimal ``nn.Module`` that supports ``register_forward_hook``."""


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeSubmodule()
        self.mlp = _FakeSubmodule()


class FakeModel(torch.nn.Module):
    """Fake causal-LM model with the layer hierarchy expected by ``hooks.py``.

    ``hooks._get_layer_modules`` looks for ``model.model.layers`` and
    ``hooks.register_hooks`` attaches hooks on ``layer.self_attn`` /
    ``layer.mlp``.  This class satisfies both without any real weights.
    """

    def __init__(self, num_layers: int = NUM_FAKE_LAYERS) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_FakeLayer() for _ in range(num_layers)]
        )
        self.model.embed_tokens = torch.nn.Embedding(10, HIDDEN_DIM)
        self.device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_fake_generate(tokenizer_obj: AutoTokenizer):
    """Return a replacement for ``generation.generate_single_token``.

    Produces structurally valid outputs (logits of the right vocab size)
    without running a real model forward pass.
    """
    vocab_size = tokenizer_obj.vocab_size  # type: ignore[attr-defined]

    def _fake(model, tok, messages, enable_thinking=False, return_all_logits=False):
        logits = torch.randn(vocab_size) * 0.1
        # Mildly boost the " A" token so argmax is deterministic.
        ids = tok.encode(" A", add_special_tokens=False)
        if ids:
            logits[ids[0]] += 2.0
        token_id = int(logits.argmax().item())
        response = tok.decode([token_id], skip_special_tokens=True).strip()
        return response, logits, token_id, None

    return _fake


@pytest.fixture(scope="session")
def tokenizer():
    """Load a real Qwen3 tokenizer (lightweight, no GPU needed)."""
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture
def fake_model():
    """Fresh fake model per test (hooks are registered/removed on it)."""
    return FakeModel()


@pytest.fixture
def sentences_file(tmp_path):
    """Temporary file with 10 short English sentences."""
    sentences = [
        "The cat sat quietly on the warm comfortable mat by the fire",
        "A large brown dog ran quickly through the green park today",
        "She walked along the beautiful sandy beach watching the waves",
        "The old library had many interesting and dusty forgotten books",
        "He drove his red car carefully through the busy city streets",
        "The mountain trail wound through dense ancient forests above",
        "Fresh bread from the local bakery smelled absolutely wonderful",
        "The concert hall was completely filled with excited music fans",
        "Students gathered in the courtyard before their morning classes",
        "The sunset painted the sky with brilliant orange and purple",
    ]
    p = tmp_path / "sentences.txt"
    p.write_text("\n".join(sentences))
    return str(p)


# ---------------------------------------------------------------------------
# Individual mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_generate(tokenizer):
    """Patch ``generation.generate_single_token`` with a fake."""
    with patch(
        "spe.generation.generate_single_token",
        side_effect=_make_fake_generate(tokenizer),
    ):
        yield


@pytest.fixture
def tmp_output_dir(monkeypatch, tmp_path):
    """Change to a temporary directory so evaluation plot files are written there."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Composite mock fixtures (one per experiment type)
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_mocks(mock_generate, tmp_output_dir):
    """All mocks needed by the classification experiment."""


@pytest.fixture
def localization_mocks(mock_generate, tmp_output_dir):
    """All mocks needed by the localization experiment."""


@pytest.fixture
def icl_mocks(mock_generate, tmp_output_dir):
    """All mocks needed by the ICL teaching experiment."""
