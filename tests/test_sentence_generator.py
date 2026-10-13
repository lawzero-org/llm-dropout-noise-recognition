"""The sentence generator (paper footnote 3) returns sentences of exactly n tokens."""

import random

import pytest
from transformers import AutoTokenizer

from spe import sentence_generator


@pytest.fixture(scope="module")
def tokenizer():
    """A Qwen3 tokenizer (the one the paper's token counts refer to)."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")


@pytest.mark.parametrize("n_tokens", sentence_generator.VALID_LENGTHS)
def test_exact_token_count(tokenizer, n_tokens: int) -> None:
    rng = random.Random(0)
    for _ in range(3):
        s = sentence_generator.sample_sentence(n_tokens, tokenizer, rng)
        assert len(tokenizer.encode(s, add_special_tokens=False)) == n_tokens


def test_reproducible_with_seed(tokenizer) -> None:
    a = sentence_generator.sample_sentence(15, tokenizer, random.Random(7))
    b = sentence_generator.sample_sentence(15, tokenizer, random.Random(7))
    assert a == b


def test_unsupported_length(tokenizer) -> None:
    with pytest.raises(ValueError):
        sentence_generator.sample_sentence(4, tokenizer, random.Random(0))
