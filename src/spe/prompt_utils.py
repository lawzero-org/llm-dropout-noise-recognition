"""Shared prompt-building and token-range utilities."""

import pathlib
import typing

import omegaconf
import transformers
import yaml

# Resolved from the package, so prompt variants load from any working directory.
_PROMPT_POOL_SEARCH_PATH = (
    pathlib.Path(__file__).resolve().parent / "conf/prompts/turns"
)


def load_prompt_pool(
    pool_names: list[str],
    class_names: list[str] | None = None,
    labels: list[str] | None = None,
    required_template_fields: tuple[str, ...] | None = None,
    extra_fields: list[str] | None = None,
) -> list[dict[str, typing.Any]]:
    """Load and validate prompt pool variant configs from YAML files.

    Args:
        pool_names: List of config names (e.g.
            ``["classification/main_variants/v00", ...]``).
        class_names: If provided, validate that each variant's
            ``class_names`` matches.
        labels: If provided, validate that each variant's ``labels``
            matches.
        required_template_fields: If provided, validate all fields
            are present and return them in a ``"templates"`` key
            (used by ICL).  When ``None``, the raw ``"turns"`` list
            is returned instead.
        extra_fields: Optional list of additional YAML keys to copy
            into the entry dict if present (e.g.
            ``["content_correct_group"]`` for localization).

    Returns:
        List of dicts, one per variant.  Each dict has at minimum a
        ``"name"`` key plus either ``"turns"`` or ``"templates"``.

    Raises:
        FileNotFoundError: If a variant YAML file does not exist.
        ValueError: If a variant fails schema validation.
    """
    pool: list[dict[str, typing.Any]] = []
    for pname in pool_names:
        ppath = _PROMPT_POOL_SEARCH_PATH / f"{pname}.yaml"
        if not ppath.exists():
            raise FileNotFoundError(
                f"Prompt pool config '{pname}' not found at {ppath}"
            )
        with open(ppath) as f:
            pcfg = yaml.safe_load(f)

        if class_names is not None and list(pcfg.get("class_names", [])) != class_names:
            raise ValueError(
                f"Prompt pool config '{pname}' has class_names="
                f"{pcfg.get('class_names')} but parent has {class_names}."
            )
        if labels is not None and list(pcfg.get("labels", [])) != labels:
            raise ValueError(
                f"Prompt pool config '{pname}' has labels="
                f"{pcfg.get('labels')} but parent has {labels}."
            )

        entry: dict[str, typing.Any] = {"name": pcfg.get("name", pname)}

        if required_template_fields is not None:
            missing = [f for f in required_template_fields if f not in pcfg]
            if missing:
                raise ValueError(
                    f"Prompt pool config '{pname}' is missing template "
                    f"fields: {missing}. Each variant must be a complete "
                    f"prompt definition."
                )
            entry["templates"] = {f: pcfg[f] for f in required_template_fields}
        else:
            entry["turns"] = pcfg["turns"]

        if extra_fields:
            for field in extra_fields:
                if field in pcfg:
                    entry[field] = pcfg[field]

        pool.append(entry)

    print(f"Prompt pool loaded: {len(pool)} variants")
    return pool


def build_messages(
    cfg: omegaconf.DictConfig,
    format_kwargs: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build a chat message list from the system prompt and prompt turns.

    Starts with the system message from ``cfg.prompts.system.content``,
    then appends each turn from ``cfg.prompts.turns.turns``.  Any turn whose
    content contains ``{placeholder}`` style markers is formatted with
    *format_kwargs*.

    Args:
        cfg: Resolved Hydra config (must have ``prompts.system`` and
            ``prompts.turns``).
        format_kwargs: Optional substitution values for turn content
            templates.

    Returns:
        List of message dicts ready for ``tokenizer.apply_chat_template``.
    """
    messages = [{"role": "system", "content": cfg.prompts.system.content}]

    for turn in cfg.prompts.turns.turns:
        content = turn.content
        if format_kwargs:
            content = content.format(**format_kwargs)
        messages.append({"role": turn.role, "content": content})

    return messages


def build_messages_from_turns(
    system_content: str,
    turns: list[dict[str, str]],
    format_kwargs: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build a chat message list from an explicit turns list.

    Same as ``build_messages`` but takes a raw turns list instead
    of reading from the Hydra config.  Used when the prompt pool
    selects a different turns list for each sample.

    Args:
        system_content: System prompt text.
        turns: List of dicts with ``role`` and ``content`` keys.
        format_kwargs: Optional substitution values for turn content
            templates.

    Returns:
        List of message dicts ready for ``tokenizer.apply_chat_template``.
    """
    messages = [{"role": "system", "content": system_content}]

    for turn in turns:
        content = turn["content"] if isinstance(turn, dict) else turn.content
        role = turn["role"] if isinstance(turn, dict) else turn.role
        if format_kwargs:
            content = content.format(**format_kwargs)
        messages.append({"role": role, "content": content})

    return messages


def validate_token_range(first_token: int, last_token: int) -> None:
    """Raise if the resolved token range is empty or inverted.

    Args:
        first_token: Inclusive start index (must be >= 0).
        last_token: Inclusive end index.  Negative values use Python
            semantics (``-1`` = last token, ``-2`` = second to last,
            etc.).

    Raises:
        ValueError: If the range is invalid.
    """
    if last_token >= 0 and last_token < first_token:
        raise ValueError(
            f"Resolved token range is empty or inverted: "
            f"first_token={first_token}, last_token={last_token}. "
            "Check that the sentence appears in the prompt."
        )


def print_token_map(
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    first_token: int,
    last_token: int,
    label: str = "DROPOUT",
    enable_thinking: bool = False,
) -> None:
    """Print each prompt token, marking the perturbation range.

    Tokens inside ``[first_token, last_token]`` are prefixed with
    ``>>`` and labeled with *label*.

    Args:
        tokenizer: The tokenizer (must support ``apply_chat_template``).
        messages: Chat messages used for this sample.
        first_token: Inclusive start index of the perturbation range.
        last_token: Inclusive end index (``-1`` means last token).
        label: Tag printed next to perturbed tokens.
        enable_thinking: Whether thinking tokens are enabled.
    """
    is_prefill = messages[-1]["role"] == "assistant"

    text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=not is_prefill,
        continue_final_message=is_prefill,
        enable_thinking=enable_thinking,
    )
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()  # type: ignore[call-overload]
    num_tokens = len(token_ids)

    start = first_token if first_token >= 0 else max(0, num_tokens + first_token)
    end = last_token if last_token >= 0 else num_tokens + last_token
    range_count = max(0, end - start + 1)

    print(f"\n{'=' * 90}")
    print(
        f"TOKEN MAP ({label})  |  total: {num_tokens}  |  "
        f"range: {range_count} tokens ({start}..{end})"
    )
    print(f"{'=' * 90}")

    for idx, tid in enumerate(token_ids):
        token_str = tokenizer.decode([tid])  # type: ignore[attr-defined]
        if start <= idx <= end:
            print(f">> {idx:>5}  {repr(token_str)}  <-- {label}")
        else:
            print(f"   {idx:>5}  {repr(token_str)}")

    print(f"{'=' * 90}")


def compute_substring_token_range(
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    substring: str,
    anchor: str | None = None,
    enable_thinking: bool = False,
) -> tuple[int, int]:
    """Find the inclusive token range covering *substring* in the prompt.

    Tokenizes the full chat prompt (using the same path as
    ``generation.generate_single_token``) and uses offset mapping to
    locate the tokens that correspond to the given substring.

    Args:
        tokenizer: Must support ``apply_chat_template`` and
            ``return_offsets_mapping``.
        messages: Full chat message list (system + all turns).
        substring: The text to locate (e.g. a sentence).
        anchor: Optional prefix used to disambiguate when *substring*
            appears more than once.  When provided, the function
            searches for ``anchor + substring`` in the prompt text,
            then returns the token range for *substring* only
            (excluding the anchor).
        enable_thinking: Must match the value used during generation.

    Returns:
        ``(first_token, last_token)`` inclusive indices into the
        tokenized prompt.

    Raises:
        ValueError: If the substring (or anchor + substring) is not
            found in the prompt text.
    """
    is_prefill = messages[-1]["role"] == "assistant"

    text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=not is_prefill,
        continue_final_message=is_prefill,
        enable_thinking=enable_thinking,
    )

    # Locate the character span of the substring
    search_string = (anchor + substring) if anchor else substring
    match_start = text.find(search_string)
    if match_start == -1:
        raise ValueError(f"Substring not found in prompt text: {search_string!r}")

    # If anchored, skip past the anchor to get the substring span
    if anchor:
        match_start += len(anchor)

    char_start = match_start
    char_end = char_start + len(substring)

    # Tokenize with offset mapping
    encoding = tokenizer(  # type: ignore[call-overload]
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoding.offset_mapping[0].tolist()  # list of (start, end)

    first_token = _find_first_overlapping(offsets, char_start)
    last_token = _find_last_overlapping(offsets, char_end)

    return first_token, last_token


def _find_first_overlapping(
    offsets: list[tuple[int, int]],
    char_start: int,
) -> int:
    """Return index of the first token whose span overlaps char_start."""
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end > char_start and tok_start < char_start + 1:
            return idx
    raise ValueError(f"No token overlaps character position {char_start}")


def _find_last_overlapping(
    offsets: list[tuple[int, int]],
    char_end: int,
) -> int:
    """Return index of the last token whose span overlaps char_end."""
    for idx in range(len(offsets) - 1, -1, -1):
        tok_start, tok_end = offsets[idx]
        if tok_start < char_end and tok_end > char_end - 1:
            return idx
    raise ValueError(f"No token overlaps character position {char_end}")


def compute_ordered_substring_token_ranges(
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    substrings: list[str],
    enable_thinking: bool = False,
) -> list[tuple[int, int]]:
    """Find token ranges for an ordered list of substrings.

    Tokenizes the prompt once, then locates each substring in order,
    advancing the search position after each match. This correctly
    handles duplicate substrings appearing in different turns.

    Args:
        tokenizer: Must support ``apply_chat_template`` and
            ``return_offsets_mapping``.
        messages: Full chat message list (system + all turns).
        substrings: Substrings to locate, in the order they appear
            in the prompt.
        enable_thinking: Must match the value used during generation.

    Returns:
        List of ``(first_token, last_token)`` inclusive token ranges,
        one per substring.
    """
    is_prefill = messages[-1]["role"] == "assistant"
    text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=not is_prefill,
        continue_final_message=is_prefill,
        enable_thinking=enable_thinking,
    )
    encoding = tokenizer(  # type: ignore[call-overload]
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoding.offset_mapping[0].tolist()

    ranges: list[tuple[int, int]] = []
    search_from = 0
    for sub in substrings:
        pos = text.find(sub, search_from)
        if pos == -1:
            raise ValueError(
                f"Substring not found (search_from={search_from}): {sub!r}"
            )
        char_end = pos + len(sub)
        first_tok = _find_first_overlapping(offsets, pos)
        last_tok = _find_last_overlapping(offsets, char_end)
        ranges.append((first_tok, last_tok))
        search_from = char_end
    return ranges


def print_multi_range_token_map(
    tokenizer: transformers.AutoTokenizer,
    messages: list[dict[str, str]],
    perturbation_entries: list[dict[str, typing.Any]],
    enable_thinking: bool = False,
) -> None:
    """Print token map with multiple perturbation ranges highlighted.

    Each perturbation entry marks its token range with the perturbation
    type and label.

    Args:
        tokenizer: The tokenizer (must support ``apply_chat_template``).
        messages: Chat messages.
        perturbation_entries: List of dicts with ``type``,
            ``first_token``, ``last_token``, and ``label``.
        enable_thinking: Whether thinking tokens are enabled.
    """
    is_prefill = messages[-1]["role"] == "assistant"

    text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=not is_prefill,
        continue_final_message=is_prefill,
        enable_thinking=enable_thinking,
    )
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()  # type: ignore[call-overload]
    num_tokens = len(token_ids)

    # Build a lookup: token_index -> (perturbation_type, label)
    range_lookup: dict[int, tuple[str, str]] = {}
    for entry in perturbation_entries:
        ft = entry["first_token"]
        lt = entry["last_token"]
        end = lt + 1 if lt >= 0 else num_tokens
        entry_label = entry.get("label", entry["type"])
        for idx in range(ft, min(end, num_tokens)):
            range_lookup[idx] = (entry["type"], entry_label)

    print(f"\n{'=' * 90}")
    print(
        f"TOKEN MAP  |  total: {num_tokens}  |  "
        f"{len(perturbation_entries)} perturbation ranges"
    )
    print(f"{'=' * 90}")

    for idx, tid in enumerate(token_ids):
        token_str = tokenizer.decode([tid])  # type: ignore[attr-defined]
        if idx in range_lookup:
            ptype, plabel = range_lookup[idx]
            print(f">> {idx:>5}  {repr(token_str)}  <-- {ptype} [{plabel}]")
        else:
            print(f"   {idx:>5}  {repr(token_str)}")

    print(f"{'=' * 90}")
