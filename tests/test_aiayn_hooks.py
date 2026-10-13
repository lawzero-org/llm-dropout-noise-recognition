"""AIAYN-style dropout (appendix): hooks at the four positions, only on the DROPOUT token range."""

import typing
from unittest.mock import patch

import pytest
import torch

from spe import aiayn_hooks

DIM, SEQ, LAYERS = 16, 10, 3


class _Attn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.v = torch.nn.Linear(DIM, DIM)
        self.o_proj = torch.nn.Linear(DIM, DIM)

    def forward(self, x):
        return self.o_proj(self.v(x))


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = torch.nn.Linear(DIM, 4 * DIM)
        self.down_proj = torch.nn.Linear(4 * DIM, DIM)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up(x)))


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class _Body(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(SEQ, DIM)
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(LAYERS))


class _TinyLM(torch.nn.Module):
    """Per-token model (no mixing across positions) with the HF module layout."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Body()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, ids):
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x)
        return x


ENTRIES = [
    {"type": "DROPOUT", "first_token": 2, "last_token": 4, "dropout_rate": 0.5},
    {"type": "NOTHING", "first_token": 6, "last_token": 8},
]


def _run(model, positions):
    ids = torch.arange(SEQ).unsqueeze(0)
    seen = {}

    def fake_generate(m, tok, messages, enable_thinking=False):
        seen["hooks"] = sum(
            len(mod._forward_hooks) + len(mod._forward_pre_hooks) for mod in m.modules()
        )
        seen["out"] = m(ids)
        return "A", torch.zeros(3), 0, None

    with patch("spe.generation.generate_single_token", side_effect=fake_generate):
        aiayn_hooks.run_single_sample_multi(
            model,
            # the tokenizer is only used by the (patched) generate_single_token
            typing.cast(typing.Any, None),
            [],
            perturbation_entries=ENTRIES,
            hook_target="both",
            first_layer=0,
            last_layer=-1,
            aiayn_positions=positions,
        )
    return seen


@pytest.mark.parametrize(
    ("positions", "expected_hooks"),
    [
        (None, 1 + LAYERS + LAYERS + 2 * LAYERS),  # all four positions
        (
            {
                "embedding": True,
                "attn_output": False,
                "ffn_internal": False,
                "post_sublayer": False,
            },
            1,
        ),
        (
            {
                "embedding": False,
                "attn_output": True,
                "ffn_internal": False,
                "post_sublayer": False,
            },
            LAYERS,
        ),
        (
            {
                "embedding": False,
                "attn_output": False,
                "ffn_internal": True,
                "post_sublayer": False,
            },
            LAYERS,
        ),
        (
            {
                "embedding": False,
                "attn_output": False,
                "ffn_internal": False,
                "post_sublayer": True,
            },
            2 * LAYERS,
        ),
    ],
)
def test_hooks_per_position_and_removed(positions, expected_hooks) -> None:
    torch.manual_seed(0)
    model = _TinyLM()
    seen = _run(model, positions)
    assert seen["hooks"] == expected_hooks
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
        == 0
    )


def test_only_the_dropout_range_changes() -> None:
    torch.manual_seed(0)
    model = _TinyLM().eval()
    with torch.no_grad():
        clean = model(torch.arange(SEQ).unsqueeze(0))
        out = _run(model, None)["out"]
    changed = (out - clean).abs().amax(dim=-1)[0] > 0
    assert changed[2:5].all()
    assert not changed[:2].any() and not changed[5:].any()  # incl. the NOTHING range
