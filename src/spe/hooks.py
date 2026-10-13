"""Forward hooks for activation perturbation (dropout and noise)."""

import typing

import torch
import torch.nn.functional as F
import transformers

ModelType = torch.nn.Module | transformers.AutoModelForCausalLM


def create_dropout_hook(
    dropout_rate: float,
    first_token: int = 0,
    last_token: int = -1,
) -> typing.Callable:
    """Create a forward hook that applies dropout to activations.

    Args:
        dropout_rate: Probability of zeroing each element.
        first_token: Index of the first token to perturb (inclusive).
        last_token: Index of the last token to perturb (inclusive).
            Negative values are supported with Python semantics:
            ``-1`` means the last token, ``-2`` means the second to
            last, etc.

    Returns:
        Hook function compatible with ``register_forward_hook``.
    """
    end = last_token + 1 if last_token != -1 else None

    def hook(
        module: torch.nn.Module, input: typing.Any, output: typing.Any
    ) -> typing.Any:
        h = output[0] if isinstance(output, tuple) else output
        noisy = h.clone()
        noisy[:, first_token:end, :] = F.dropout(
            h[:, first_token:end, :], p=dropout_rate, training=True
        )
        if isinstance(output, tuple):
            return (noisy,) + output[1:]
        return noisy

    return hook


def create_noise_hook(
    noise_std: float,
    first_token: int = 0,
    last_token: int = -1,
) -> typing.Callable:
    """Create a forward hook that adds Gaussian noise to activations.

    Args:
        noise_std: Standard deviation of the additive noise.
        first_token: Index of the first token to perturb (inclusive).
        last_token: Index of the last token to perturb (inclusive).
            Negative values are supported with Python semantics:
            ``-1`` means the last token, ``-2`` means the second to
            last, etc.

    Returns:
        Hook function compatible with ``register_forward_hook``.
    """
    end = last_token + 1 if last_token != -1 else None

    def hook(
        module: torch.nn.Module, input: typing.Any, output: typing.Any
    ) -> typing.Any:
        h = output[0] if isinstance(output, tuple) else output
        noisy = h.clone()
        noisy[:, first_token:end, :] = h[:, first_token:end, :] + (
            torch.randn_like(h[:, first_token:end, :]) * noise_std
        )
        if isinstance(output, tuple):
            return (noisy,) + output[1:]
        return noisy

    return hook


def create_multi_perturbation_hook(
    perturbations: list[dict[str, typing.Any]],
) -> typing.Callable:
    """Create a forward hook that applies multiple perturbations to non overlapping token ranges.

    Instead of registering N separate hooks (each cloning the full tensor),
    this creates one hook that clones once and applies all perturbations.
    Entries with ``type == "NOTHING"`` are skipped (no perturbation on that range).

    Args:
        perturbations: List of dicts, each with:
            - ``type``: ``"DROPOUT"``, ``"NOISE"``, or ``"NOTHING"``
            - ``first_token``: Inclusive start index of the token range
            - ``last_token``: Inclusive end index of the token range
            - ``dropout_rate``: Dropout probability (used when type is ``"DROPOUT"``)
            - ``noise_std``: Noise standard deviation (used when type is ``"NOISE"``)

    Returns:
        Hook function compatible with ``register_forward_hook``.

    Raises:
        ValueError: If token ranges overlap or if an unknown perturbation type is given.
    """
    valid_types = ("DROPOUT", "NOISE", "NOTHING")
    for entry in perturbations:
        if entry["type"] not in valid_types:
            raise ValueError(
                f"Unknown perturbation type: {entry['type']!r}. Expected one of {valid_types}"
            )
        if entry["last_token"] == -1:
            raise ValueError(
                "last_token=-1 (end of sequence) is not allowed in multi perturbation hook. "
                "Compute the concrete token index before constructing the entry."
            )
        if entry["first_token"] < 0:
            raise ValueError(f"first_token must be >= 0, got {entry['first_token']}")
        if entry["last_token"] < entry["first_token"]:
            raise ValueError(
                f"Inverted token range: first_token={entry['first_token']} > last_token={entry['last_token']}"
            )

    # Filter out NOTHING entries (they need no perturbation)
    active = [e for e in perturbations if e["type"] != "NOTHING"]

    # Validate non overlapping ranges: sort by first_token and check each pair
    sorted_active = sorted(active, key=lambda e: e["first_token"])
    for i in range(len(sorted_active) - 1):
        curr = sorted_active[i]
        nxt = sorted_active[i + 1]
        if curr["last_token"] >= nxt["first_token"]:
            raise ValueError(
                f"Overlapping token ranges: entry ending at {curr['last_token']} "
                f"overlaps with entry starting at {nxt['first_token']}"
            )

    # Pre compute end indices (exclusive) for slicing.
    # last_token=-1 is already rejected above, so all values are concrete.
    resolved: list[tuple[str, int, int, float, float]] = []
    for entry in active:
        end = entry["last_token"] + 1
        resolved.append(
            (
                entry["type"],
                entry["first_token"],
                end,
                entry.get("dropout_rate", 0.0),
                entry.get("noise_std", 0.0),
            )
        )

    def hook(
        module: torch.nn.Module, input: typing.Any, output: typing.Any
    ) -> typing.Any:
        if not resolved:
            return output

        h = output[0] if isinstance(output, tuple) else output
        noisy = h.clone()

        for ptype, first, end_idx, dropout_rate, noise_std in resolved:
            if ptype == "DROPOUT":
                noisy[:, first:end_idx, :] = F.dropout(
                    h[:, first:end_idx, :], p=dropout_rate, training=True
                )
            elif ptype == "NOISE":
                noisy[:, first:end_idx, :] = h[:, first:end_idx, :] + (
                    torch.randn_like(h[:, first:end_idx, :]) * noise_std
                )

        if isinstance(output, tuple):
            return (noisy,) + output[1:]
        return noisy

    return hook


def _get_layer_modules(model: ModelType) -> torch.nn.ModuleList:
    """Locate the transformer layer list inside a HuggingFace model.

    Args:
        model: A HuggingFace causal LM.

    Returns:
        The ``ModuleList`` of transformer layers.

    Raises:
        ValueError: If layers cannot be found.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):  # type: ignore[attr-defined]
        return model.model.layers  # type: ignore[attr-defined]
    if (  # Gemma 4 multimodal: top.model.language_model.layers
        hasattr(model, "model")
        and hasattr(model.model, "language_model")  # type: ignore[attr-defined]
        and hasattr(model.model.language_model, "layers")  # type: ignore[attr-defined]
    ):
        return model.model.language_model.layers  # type: ignore[attr-defined]
    if hasattr(model, "layers"):
        return model.layers  # type: ignore[attr-defined]
    raise ValueError("Could not find layer modules in model")


def resolve_layer_range(
    model: ModelType,
    first_layer: int = 0,
    last_layer: int = -1,
) -> tuple[int, int]:
    """Validate and resolve layer range indices against the model.

    Only ``last_layer=-1`` is accepted as a negative value (meaning
    "through the last layer").  All other negative values raise
    ``ValueError``.

    Args:
        model: The transformer model.
        first_layer: Index of the first layer (inclusive). Must be >= 0.
        last_layer: Index of the last layer (inclusive). Use ``-1`` for
            the last layer in the model.

    Returns:
        Tuple of (resolved_first, resolved_last) as inclusive indices.

    Raises:
        ValueError: If the range is invalid or out of bounds.
    """
    num_layers = len(_get_layer_modules(model))

    if first_layer < 0:
        raise ValueError(f"first_layer must be >= 0, got {first_layer}")
    if last_layer == -1:
        resolved_last = num_layers - 1
    else:
        if last_layer < 0:
            raise ValueError(
                f"last_layer must be >= 0 or -1 (all layers), got {last_layer}"
            )
        resolved_last = last_layer
    if first_layer >= num_layers:
        raise ValueError(
            f"first_layer={first_layer} is out of range, model has {num_layers} layers (0..{num_layers - 1})"
        )
    if resolved_last >= num_layers:
        raise ValueError(
            f"last_layer={last_layer} is out of range, model has {num_layers} layers (0..{num_layers - 1})"
        )
    if first_layer > resolved_last:
        raise ValueError(
            f"first_layer={first_layer} > last_layer={resolved_last} (from {last_layer}), empty layer range"
        )

    return first_layer, resolved_last


def register_hooks(
    model: ModelType,
    hook_fn: typing.Callable,
    target: str = "both",
    first_layer: int = 0,
    last_layer: int = -1,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register a forward hook on attention and/or MLP sub layers.

    The caller decides which hook function (dropout, noise, compound) to pass in.

    Args:
        model: The transformer model.
        hook_fn: A callable returned by ``create_dropout_hook`` or
            ``create_noise_hook`` (or any custom hook).
        target: ``"attn"``, ``"mlp"``, or ``"both"``.
        first_layer: Index of the first layer to hook (inclusive).
            Must be >= 0.
        last_layer: Index of the last layer to hook (inclusive).
            Use ``-1`` to mean the last layer in the model.
            No other negative values are allowed.

    Returns:
        List of hook handles.  Pass to ``remove_hooks`` after generation.

    Raises:
        ValueError: If ``target`` is not one of ``"attn"``, ``"mlp"``,
            ``"both"``, or if the layer range is invalid.
    """
    valid_targets = ("attn", "mlp", "both")
    if target not in valid_targets:
        raise ValueError(f"target must be one of {valid_targets}, got {target!r}")

    resolved_first, resolved_last = resolve_layer_range(model, first_layer, last_layer)
    end = resolved_last + 1

    handles: list[torch.utils.hooks.RemovableHandle] = []
    layer_modules = _get_layer_modules(model)

    for i, layer in enumerate(layer_modules):
        if i < resolved_first or i >= end:
            continue
        # Gemma 2/3/4 and OLMo 2/3 apply an extra RMSNorm to the sub-layer
        # output before adding it to the residual stream (detected by
        # post_feedforward_layernorm). Hooking the raw
        # self_attn / mlp output would have its magnitude normalized
        # away. Hook the post-norm output instead so the perturbation
        # lands on the actual contribution to the residual — matching
        # the effective perturbation site in standard pre-norm models.
        gemma_style = hasattr(layer, "post_feedforward_layernorm")
        if target in ("attn", "both"):
            if gemma_style:
                handles.append(
                    layer.post_attention_layernorm.register_forward_hook(hook_fn)  # type: ignore[attr-defined]
                )
            elif hasattr(layer, "self_attn"):
                handles.append(layer.self_attn.register_forward_hook(hook_fn))  # type: ignore[attr-defined]
            elif hasattr(layer, "linear_attn"):
                handles.append(layer.linear_attn.register_forward_hook(hook_fn))  # type: ignore[attr-defined]
            else:
                raise AttributeError(
                    f"Layer {i} ({type(layer).__name__}) has no known attention submodule "
                    f"(expected 'self_attn' or 'linear_attn')"
                )
        if target in ("mlp", "both"):
            if gemma_style:
                handles.append(
                    layer.post_feedforward_layernorm.register_forward_hook(hook_fn)  # type: ignore[attr-defined]
                )
            elif hasattr(layer, "mlp"):
                handles.append(layer.mlp.register_forward_hook(hook_fn))  # type: ignore[attr-defined]
            else:
                raise AttributeError(
                    f"Layer {i} ({type(layer).__name__}) has no 'mlp' submodule"
                )

    return handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove all registered hooks.

    Args:
        handles: List of hook handles returned by ``register_hooks``.
    """
    for h in handles:
        h.remove()
