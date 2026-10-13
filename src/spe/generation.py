"""Single token generation and probability extraction."""

import typing

import torch
import torch.nn.functional as F
import transformers


def letter_variants(letter: str) -> list[str]:
    """Generate token variants for any single letter (e.g. "C" -> ["C", " C", "c", " c"])."""
    return [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]


def argmax_to_label(
    argmax_token_id: int,
    variant_token_ids: dict[str, int],
    letters: list[str],
) -> str:
    """Map an argmax token ID to its letter label, or ``"OTHER"``."""
    tid_to_letter: dict[int, str] = {}
    for letter in letters:
        for variant in letter_variants(letter):
            if variant in variant_token_ids:
                tid_to_letter[variant_token_ids[variant]] = letter
    return tid_to_letter.get(argmax_token_id, "OTHER")


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Compute entropy (in nats) of the softmax distribution over logits."""
    log_probs = logits.float() - torch.logsumexp(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum().item()


def generate_single_token(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    enable_thinking: bool = False,
    return_all_logits: bool = False,
) -> tuple[str, torch.Tensor, int, torch.Tensor | None]:
    """Generate exactly one token and return the decoded text, logits and token id.

    Args:
        model: The causal language model.
        tokenizer: The matching tokenizer.
        messages: Chat messages (system / user / assistant dicts).
        enable_thinking: Whether to allow the model to produce thinking
            tokens before the answer.  Defaults to ``False`` so the first
            generated token is the actual answer (needed for probability
            extraction).
        return_all_logits: If True, also return the full logits tensor
            of shape ``[seq_len, vocab_size]`` for all positions. Used
            to compute per token log probabilities for sentence ranges.

    Returns:
        Tuple of (decoded_response, first_token_logits, generated_token_id, all_logits).
        ``first_token_logits`` has shape ``[vocab_size]``.
        ``generated_token_id`` is the vocabulary index of the token the model produced.
        ``all_logits`` is ``None`` unless ``return_all_logits=True``.
    """
    is_prefill = messages[-1]["role"] == "assistant"

    text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=not is_prefill,
        continue_final_message=is_prefill,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)  # type: ignore[call-overload, attr-defined]

    with torch.no_grad():
        outputs = model(**inputs)  # type: ignore[call-overload]

    last_pos_logits = outputs.logits[0, -1, :]  # [vocab_size]
    generated_token_id = int(last_pos_logits.argmax().item())
    response = tokenizer.decode([generated_token_id], skip_special_tokens=True)  # type: ignore[attr-defined]
    all_logits = outputs.logits[0] if return_all_logits else None

    return (
        response.strip(),
        last_pos_logits,
        generated_token_id,
        all_logits,
    )


def compute_token_range_log_prob(
    all_logits: torch.Tensor,
    input_ids: torch.Tensor,
    first_token: int,
    last_token: int,
) -> float:
    """Compute the sum of log probabilities for tokens in a range.

    For each position t in [first_token, last_token], computes
    log P(input_ids[t] | context up to t) using the logits at
    position t-1. The first token in the range uses the logits
    from the preceding position as context.

    Args:
        all_logits: Full logits tensor of shape ``[seq_len, vocab_size]``.
        input_ids: Token ids of shape ``[seq_len]``.
        first_token: Inclusive start of the range.
        last_token: Inclusive end of the range.

    Returns:
        Sum of log probabilities (a negative number; closer to 0 = more
        probable).
    """
    log_probs = torch.nn.functional.log_softmax(all_logits.float(), dim=-1)
    total = 0.0
    for t in range(first_token, last_token + 1):
        if t == 0:
            continue
        total += log_probs[t - 1, input_ids[t]].item()
    return total


def resolve_token_ids_safe(
    tokenizer: transformers.AutoTokenizer,
    labels: typing.Sequence[str],
) -> dict[str, int]:
    """Encode each label string, keeping only those that map to a single token.

    Labels that encode to multiple tokens are skipped with a warning.

    Args:
        tokenizer: The tokenizer to use.
        labels: Strings to encode (e.g. ``["YES", "Yes", "yes", "NO", "No", "no"]``).

    Returns:
        Dict mapping each single token label to its token id.
    """
    token_ids: dict[str, int] = {}
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)  # type: ignore[attr-defined]
        if len(encoded) != 1:
            print(
                f"WARNING: label '{label}' encoded to {len(encoded)} tokens {encoded}, skipping."
            )
            continue
        token_ids[label] = encoded[0]
    return token_ids


def extract_n_letter_variant_data(
    logits: torch.Tensor,
    variant_token_ids: dict[str, int],
    tokenizer: transformers.AutoTokenizer,
    letters: list[str],
) -> dict[str, typing.Any]:
    """Extract argmax, variant logits/probs, and letter group aggregates for N letters.

    It works for any number of letters (2 to 26) by using
    ``letter_variants()`` to build variant groups dynamically.

    Args:
        logits: Raw logits of shape ``[vocab_size]``.
        variant_token_ids: Mapping from variant label (e.g. ``"A"``,
            ``" A"``, ``"b"``) to vocabulary index. Only resolved
            (single token) variants.
        tokenizer: Used to decode the argmax token id.
        letters: Ordered list of uppercase letters to compute groups
            for (e.g. ``["A", "B", "C", "D", "E"]``).

    Returns:
        Dict with:
        - ``argmax_token_id``, ``argmax_token``, ``argmax_logit``, ``argmax_prob``
        - ``v_logit_<variant>`` and ``v_prob_<variant>`` for each resolved variant
        - ``sum_prob_<L>``, ``logsumexp_<L>`` for each letter L
        - ``primary_prob_<L>``, ``primary_logit_<L>`` for each letter L
            (primary variant is ``" <L>"``, i.e. space + uppercase letter)
    """
    logits = logits.float()
    probs = F.softmax(logits, dim=-1)
    log_norm = torch.logsumexp(logits, dim=-1).item()

    argmax_id = int(torch.argmax(logits).item())
    argmax_logit = logits[argmax_id].item()
    argmax_prob = probs[argmax_id].item()
    argmax_token = repr(tokenizer.decode([argmax_id]))  # type: ignore[attr-defined]

    log_probs_all = logits - log_norm
    ent = -(probs * log_probs_all).sum().item()

    data: dict[str, typing.Any] = {
        "entropy": ent,
        "argmax_token_id": argmax_id,
        "argmax_token": argmax_token,
        "argmax_logit": argmax_logit,
        "argmax_prob": argmax_prob,
    }

    # Variant level values
    for variant, tid in variant_token_ids.items():
        data[f"v_logit_{variant}"] = logits[tid].item()
        data[f"v_prob_{variant}"] = probs[tid].item()

    # Group sums, logsumexp, and primary variant for each letter
    for letter in letters:
        variants = letter_variants(letter)
        primary = f" {letter}"

        letter_probs = [
            probs[variant_token_ids[v]].item()
            for v in variants
            if v in variant_token_ids
        ]
        letter_logits_t = torch.tensor(
            [
                logits[variant_token_ids[v]].item()
                for v in variants
                if v in variant_token_ids
            ]
        )

        data[f"sum_prob_{letter}"] = sum(letter_probs) if letter_probs else 0.0
        logsumexp_letter = (
            torch.logsumexp(letter_logits_t, dim=0).item()
            if len(letter_logits_t) > 0
            else float("-inf")
        )
        data[f"logsumexp_{letter}"] = logsumexp_letter

        # The primary variant " L" must be a single token, otherwise the
        # primary metrics are meaningless: fail loudly (KeyError).
        tid = variant_token_ids[primary]
        data[f"primary_prob_{letter}"] = probs[tid].item()
        data[f"primary_logit_{letter}"] = logits[tid].item()

        # Log-probabilities: logit - logsumexp(all vocab) = log P(token)
        data[f"primary_log_prob_{letter}"] = data[f"primary_logit_{letter}"] - log_norm
        data[f"aggregate_log_prob_{letter}"] = logsumexp_letter - log_norm

    return data
