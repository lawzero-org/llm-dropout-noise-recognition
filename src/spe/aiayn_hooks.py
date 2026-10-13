"""AIAYN-style dropout with configurable positions.

Implements four dropout positions from the "Attention Is All You Need" paper:

- **(i)  embedding**: on the output of ``embed_tokens``, before the first
  transformer layer.
- **(ii) attn_output**: on the attention output before ``o_proj``.  This is the
  closest hookable approximation to attention-weight dropout (the softmax output
  is inline in HuggingFace and not hookable as a submodule).
- **(iii) ffn_internal**: on the ``d_ff``-dimensional gated intermediate inside
  the FFN, before ``down_proj``.
- **(iv) post_sublayer**: on the output of each sublayer (self_attn / mlp),
  before the residual connection.  This reuses the existing hooks from
  ``hooks.py``.

Each position is individually togglable via ``aiayn_positions``.  The default
when ``dropout_style=aiayn`` is all four enabled (the paper's appendix setting);
used by the localization runner.
"""

import typing

import omegaconf
import torch
import torch.nn.functional as F
import transformers

from spe import generation, hooks

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_POSITIONS: dict[str, bool] = {
    "embedding": True,
    "attn_output": True,
    "ffn_internal": True,
    "post_sublayer": True,
}


def resolve_positions(cfg: omegaconf.DictConfig) -> dict[str, bool]:
    """Read ``aiayn_positions`` from config, defaulting to all enabled."""
    positions_cfg = cfg.perturbation.get("aiayn_positions", {})
    return {
        key: positions_cfg.get(key, default)
        for key, default in DEFAULT_POSITIONS.items()
    }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _get_embed_module(model: hooks.ModelType) -> torch.nn.Module:
    """Locate the token embedding module inside a HuggingFace model (multimodal wrappers included)."""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()  # type: ignore[union-attr]
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens  # type: ignore[attr-defined]
    raise ValueError("Could not find embed_tokens in model")


def _get_attn_module(layer: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    """Return the attention submodule of a transformer layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn  # type: ignore[attr-defined]
    if hasattr(layer, "linear_attn"):
        return layer.linear_attn  # type: ignore[attr-defined]
    raise AttributeError(
        f"Layer {layer_idx} ({type(layer).__name__}) has no known attention submodule"
    )


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _resolve_dropout_ranges(
    perturbations: list[dict[str, typing.Any]],
) -> list[tuple[int, int, float]]:
    """Extract and validate DROPOUT entries into (first, end, rate) tuples."""
    result: list[tuple[int, int, float]] = []
    for entry in perturbations:
        if entry["type"] != "DROPOUT":
            continue
        if entry["last_token"] == -1:
            raise ValueError(
                "last_token=-1 is not allowed in multi-range AIAYN hooks. "
                "Compute the concrete token index before constructing the entry."
            )
        result.append(
            (
                entry["first_token"],
                entry["last_token"] + 1,
                entry.get("dropout_rate", 0.0),
            )
        )
    return result


def _create_multi_forward_hook_dropout(
    perturbations: list[dict[str, typing.Any]],
) -> typing.Callable | None:
    """Compound forward hook (post) for multiple DROPOUT ranges."""
    resolved = _resolve_dropout_ranges(perturbations)
    if not resolved:
        return None

    def hook(
        module: torch.nn.Module,
        input: typing.Any,
        output: typing.Any,
    ) -> typing.Any:
        h = output[0] if isinstance(output, tuple) else output
        noisy = h.clone()
        for first, end_idx, rate in resolved:
            noisy[:, first:end_idx, :] = F.dropout(
                h[:, first:end_idx, :],
                p=rate,
                training=True,
            )
        if isinstance(output, tuple):
            return (noisy,) + output[1:]
        return noisy

    return hook


def _create_multi_pre_hook_dropout(
    perturbations: list[dict[str, typing.Any]],
) -> typing.Callable | None:
    """Compound forward pre-hook for multiple DROPOUT ranges."""
    resolved = _resolve_dropout_ranges(perturbations)
    if not resolved:
        return None

    def hook(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        h = args[0]
        noisy = h.clone()
        for first, end_idx, rate in resolved:
            noisy[:, first:end_idx, :] = F.dropout(
                h[:, first:end_idx, :],
                p=rate,
                training=True,
            )
        return (noisy,) + args[1:]

    return hook


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def _register_embedding_hook(
    model: hooks.ModelType,
    hook_fn: typing.Callable,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register a forward hook on ``embed_tokens``."""
    embed = _get_embed_module(model)
    return [embed.register_forward_hook(hook_fn)]


def _register_attn_output_hooks(
    model: hooks.ModelType,
    hook_fn: typing.Callable,
    first_layer: int,
    last_layer: int,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register a pre-hook on ``self_attn.o_proj`` for each layer in range."""
    resolved_first, resolved_last = hooks.resolve_layer_range(
        model,
        first_layer,
        last_layer,
    )
    layers = hooks._get_layer_modules(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for i, layer in enumerate(layers):
        if i < resolved_first or i > resolved_last:
            continue
        attn = _get_attn_module(layer, i)
        # linear-attention layers of hybrid models (e.g. Qwen3.5+) name the output projection out_proj
        proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
        if proj is None:
            raise AttributeError(
                f"Layer {i} attention module ({type(attn).__name__}) "
                f"has no 'o_proj' / 'out_proj' submodule"
            )
        handles.append(proj.register_forward_pre_hook(hook_fn))

    return handles


def _register_ffn_internal_hooks(
    model: hooks.ModelType,
    hook_fn: typing.Callable,
    first_layer: int,
    last_layer: int,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register a pre-hook on ``mlp.down_proj`` for each layer in range."""
    resolved_first, resolved_last = hooks.resolve_layer_range(
        model,
        first_layer,
        last_layer,
    )
    layers = hooks._get_layer_modules(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for i, layer in enumerate(layers):
        if i < resolved_first or i > resolved_last:
            continue
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
            raise AttributeError(
                f"Layer {i} ({type(layer).__name__}) has no 'mlp.down_proj' submodule"
            )
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook_fn))  # type: ignore[attr-defined]

    return handles


# ---------------------------------------------------------------------------
# Run helper (mirrors core.run_single_sample)
# ---------------------------------------------------------------------------


def run_single_sample_multi(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    perturbation_entries: list[dict[str, typing.Any]],
    hook_target: str,
    first_layer: int,
    last_layer: int,
    aiayn_positions: dict[str, bool] | None = None,
    enable_thinking: bool = False,
) -> tuple[str, torch.Tensor, int]:
    """AIAYN dropout on the DROPOUT token ranges of a localization sample.

    Registers hooks at the enabled positions for every DROPOUT entry and
    the standard multi-perturbation post-sublayer hook for all entries.

    Args:
        model: The language model.
        tokenizer: Matching tokenizer.
        messages: Chat messages for generation.
        perturbation_entries: List of perturbation dicts (same format as
            ``hooks.create_multi_perturbation_hook``).
        hook_target: ``"attn"``, ``"mlp"``, or ``"both"`` (for post-sublayer).
        first_layer: First layer to hook (inclusive).
        last_layer: Last layer to hook (inclusive, ``-1`` = last).
        aiayn_positions: Which positions to enable.  Defaults to all four.
        enable_thinking: Whether to allow thinking tokens.

    Returns:
        Tuple of (decoded_response, logits, generated_token_id).
    """
    pos = aiayn_positions or DEFAULT_POSITIONS
    handles: list[torch.utils.hooks.RemovableHandle] = []

    if pos.get("embedding", True):
        hook = _create_multi_forward_hook_dropout(perturbation_entries)
        if hook is not None:
            handles += _register_embedding_hook(model, hook)

    if pos.get("attn_output", True):
        hook = _create_multi_pre_hook_dropout(perturbation_entries)
        if hook is not None:
            handles += _register_attn_output_hooks(
                model,
                hook,
                first_layer=first_layer,
                last_layer=last_layer,
            )

    if pos.get("ffn_internal", True):
        hook = _create_multi_pre_hook_dropout(perturbation_entries)
        if hook is not None:
            handles += _register_ffn_internal_hooks(
                model,
                hook,
                first_layer=first_layer,
                last_layer=last_layer,
            )

    if pos.get("post_sublayer", True):
        hook = hooks.create_multi_perturbation_hook(perturbation_entries)
        handles += hooks.register_hooks(
            model,
            hook,
            target=hook_target,
            first_layer=first_layer,
            last_layer=last_layer,
        )

    try:
        response, logits, token_id, _all_logits = generation.generate_single_token(
            model,
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
        )
    finally:
        hooks.remove_hooks(handles)

    return response, logits, token_id
