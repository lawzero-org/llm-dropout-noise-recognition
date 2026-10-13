"""Model and tokenizer loading, hardware diagnostics."""

import os
import platform
import subprocess
import typing
from pathlib import Path

import huggingface_hub
import torch
import transformers

# Fallback chat template used when the tokenizer does not ship its own
# (e.g. base / pretraining checkpoints). Concatenates message contents
# separated by blank lines, no role markers, keeping the multi-turn
# structure of the introspection prompts intact.
_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['content'] }}"
    "{% if not loop.last %}\n\n{% endif %}"
    "{% endfor %}"
)


def _patch_apply_chat_template_for_gemma4(tokenizer) -> None:
    """Make ``continue_final_message=True`` work for Gemma 4.

    Gemma 4's chat template strips channel-control tags from assistant
    content, so we cannot prepend the empty thought channel to a
    prefilled assistant message directly. Instead, when asked to
    continue a final assistant message, render messages up to (but not
    including) that message with ``add_generation_prompt=True`` so the
    template injects the empty thought channel, then concatenate the
    prefill text raw. The model sees the same prompt shape it was
    trained on, and the prefill appears in the position where the
    answer should follow. Single-token measurement is preserved.

    No-op for non-Gemma-4 tokenizers.
    """
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "gemma-4" not in name:
        return

    original = tokenizer.apply_chat_template

    def patched(messages, **kwargs):
        if (
            kwargs.get("continue_final_message")
            and messages
            and messages[-1].get("role") == "assistant"
        ):
            prefill_text = messages[-1].get("content", "")
            tokenize = kwargs.pop("tokenize", True)
            return_tensors = kwargs.pop("return_tensors", None)
            kwargs.pop("add_generation_prompt", None)
            kwargs.pop("continue_final_message", None)
            base = original(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            full = base + prefill_text
            if not tokenize:
                return full
            return tokenizer(full, return_tensors=return_tensors).input_ids
        return original(messages, **kwargs)

    tokenizer.apply_chat_template = patched  # type: ignore[method-assign]


def print_hardware_info() -> None:
    """Print GPU / CUDA availability and memory usage."""
    print("=" * 50)
    print("HARDWARE CHECK")
    print("=" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, will run on CPU (slow!)")
    print("=" * 50)


_AUTO_CLASSES = {
    "causal_lm": transformers.AutoModelForCausalLM,
    "image_text_to_text": transformers.AutoModelForImageTextToText,
}


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    revision: str | None = None,
    auto_class: str = "causal_lm",
) -> tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier (e.g. ``Qwen/Qwen3-32B``).
        dtype: Torch dtype string.  Converted via ``getattr(torch, dtype)``.
        device_map: Device map strategy passed to ``from_pretrained``.
        revision: Optional git revision (branch / tag / commit sha) on the
            HF repo. Used to pin to a specific training checkpoint, e.g.
            ``stage1-step100000`` on ``allenai/Olmo-3-1125-32B``.
        auto_class: Which Auto* loader the checkpoint registers under.
            Multimodal checkpoints (e.g. Mistral 3.x) only register under
            ``image_text_to_text``; we still use them text-only.

    Returns:
        Tuple of (model, tokenizer).
    """
    # "auto" lets the checkpoint keep its native format; otherwise resolve a
    # concrete torch dtype (e.g. bfloat16).
    resolved_dtype = dtype if dtype == "auto" else getattr(torch, dtype)

    print(f"Loading model: {model_name} (revision={revision})")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, revision=revision
    )
    if getattr(tokenizer, "chat_template", None) is None:
        print("Tokenizer has no chat_template; applying plain-concatenation fallback.")
        tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE
    _patch_apply_chat_template_for_gemma4(tokenizer)
    model = _AUTO_CLASSES[auto_class].from_pretrained(
        model_name,
        dtype=resolved_dtype,
        device_map=device_map,
        revision=revision,
    )

    print(f"Model loaded on device: {model.device}")  # type: ignore[attr-defined]
    print(f"Model dtype: {model.dtype}")  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        print(
            f"GPU memory after model load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )

    return typing.cast(transformers.AutoModelForCausalLM, model), typing.cast(
        transformers.AutoTokenizer, tokenizer
    )


def runtime_info(model_name: str, revision: str | None) -> dict[str, typing.Any]:
    """What actually ran: resolved HF commit, code commit, library versions, hardware.

    Logged to W&B config (not part of the seed hash, which only uses the requested config).
    Best effort: outside a git checkout (e.g. an unpacked archive) the code fields are
    ``None``, and so is ``hf_commit`` if the model is not in the local HF cache.
    """
    repo_root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(  # noqa: S603
                ["git", "-C", str(repo_root), *args],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    try:
        snapshot = huggingface_hub.snapshot_download(
            model_name, revision=revision, local_files_only=True
        )
        hf_commit = Path(snapshot).name
    except Exception:  # noqa: BLE001 - diagnostics only, never fail a run over them
        hf_commit = None
    status = git("status", "--porcelain")
    return {
        "hf_commit": hf_commit,
        "code_commit": git("rev-parse", "HEAD"),
        "code_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
