"""Shared sentence sampling: from a pre-built sentence pool or the sentence generator."""

import random

import transformers

from spe import sentence_generator


def sample_or_generate_sentences(
    needed: int,
    same_sentence: bool,
    sentences_pool: list[str] | None = None,
    sentence_n_tokens: int | None = None,
    tokenizer: transformers.AutoTokenizer | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """Sample from a pool or dynamically generate sentences.

    When ``same_sentence`` is ``True``, a single sentence is chosen
    and repeated ``needed`` times.  Otherwise, all returned sentences
    are unique.

    Exactly one of ``sentences_pool`` or ``sentence_n_tokens`` must
    be provided.

    Args:
        needed: Number of sentences to return.
        same_sentence: Repeat one sentence ``needed`` times.
        sentences_pool: Pre-loaded sentence pool (from file).
        sentence_n_tokens: Token count for dynamic generation.
            Used only when ``sentences_pool`` is ``None``.
        tokenizer: Tokenizer for the sentence generator. Required
            when using dynamic generation.
        rng: RNG for the sentence generator. Required when using
            dynamic generation.

    Returns:
        List of ``needed`` sentences.

    Raises:
        ValueError: If neither source is provided.
        RuntimeError: If dynamic generation cannot produce enough
            unique sentences.
    """
    if sentences_pool is not None:
        if same_sentence:
            s = random.choice(sentences_pool)
            return [s] * needed
        return random.sample(sentences_pool, needed)

    if sentence_n_tokens is None:
        raise ValueError("Either sentences_pool or sentence_n_tokens must be provided")
    if tokenizer is None:
        raise ValueError("tokenizer is required for dynamic generation")

    if same_sentence:
        s = sentence_generator.sample_sentence(sentence_n_tokens, tokenizer, rng)
        return [s] * needed

    seen: set[str] = set()
    result: list[str] = []
    for _ in range(needed):
        for _attempt in range(sentence_generator.MAX_ATTEMPTS):
            s = sentence_generator.sample_sentence(sentence_n_tokens, tokenizer, rng)
            if s not in seen:
                seen.add(s)
                result.append(s)
                break
        else:
            raise RuntimeError(
                f"Could not generate {needed} unique sentences "
                f"with {sentence_n_tokens} tokens"
            )
    return result
