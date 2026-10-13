"""Shared experiment primitives: inner loop body and config helpers."""

import typing

import torch
import transformers

from spe import generation, hooks


def run_single_sample(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    hook_fn: typing.Callable | None,
    hook_target: str,
    first_layer: int,
    last_layer: int,
    enable_thinking: bool = False,
) -> tuple[str, torch.Tensor, int]:
    """Register hooks, generate one token, remove hooks.

    Args:
        model: The language model.
        tokenizer: Matching tokenizer.
        messages: Chat messages for generation.
        hook_fn: Hook function to register, or ``None`` for
            baseline (no perturbation).
        hook_target: ``"attn"``, ``"mlp"``, or ``"both"``.
        first_layer: First layer to hook (inclusive).
        last_layer: Last layer to hook (inclusive, -1 = last).
        enable_thinking: Whether to allow thinking tokens.

    Returns:
        Tuple of (decoded_response, logits, generated_token_id).
    """
    if hook_fn is not None:
        handles = hooks.register_hooks(
            model,
            hook_fn,
            target=hook_target,
            first_layer=first_layer,
            last_layer=last_layer,
        )
    else:
        handles = []

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


def describe_option(name: str) -> str:
    """Return a parenthetical description for a class name."""
    descriptions = {
        "DROPOUT": "(random neuron dropping)",
        "NOISE": "(gaussian noise injection)",
    }
    return descriptions.get(name, "")
