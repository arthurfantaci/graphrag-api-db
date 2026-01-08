"""Text processing utilities for chunking and embedding.

This module provides shared utilities for:
- Accurate token counting using tiktoken (OpenAI's tokenizer)
- CharInterval overlap detection for entity-chunk linkage
- Text normalization for embedding
- Chunk ID generation

These utilities are used by both the chunker and embedder modules.
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph_models import CharInterval


@lru_cache(maxsize=1)
def _get_tiktoken_encoding() -> Any:
    """Get or create tiktoken encoding (cached).

    Uses cl100k_base encoding which is compatible with:
    - text-embedding-3-small
    - text-embedding-3-large
    - gpt-4o
    - gpt-4-turbo

    Returns:
        The cl100k_base tiktoken encoding.

    Raises:
        ImportError: If tiktoken is not installed.
    """
    try:
        import tiktoken

        # cl100k_base is used by text-embedding-3-* and gpt-4*
        return tiktoken.get_encoding("cl100k_base")
    except ImportError as e:
        raise ImportError(
            "tiktoken is required for accurate token counting. "
            "Install it with: uv sync --group embedding"
        ) from e


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (OpenAI's tokenizer).

    This provides accurate token counts for OpenAI embedding models.
    Uses cl100k_base encoding (compatible with text-embedding-3-* and gpt-4*).

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens in the text.

    Raises:
        ImportError: If tiktoken is not installed.

    Example:
        >>> count_tokens("Hello, world!")
        4
    """
    encoding = _get_tiktoken_encoding()
    return len(encoding.encode(text))


def estimate_tokens_fast(text: str, chars_per_token: float = 4.0) -> int:
    """Fast token estimation without tiktoken dependency.

    Use this when tiktoken is not available or for quick estimates.
    Less accurate than count_tokens() but much faster.

    Args:
        text: Text to estimate tokens for.
        chars_per_token: Average characters per token (4.0 is typical for English).

    Returns:
        Estimated number of tokens.
    """
    return max(1, int(len(text) / chars_per_token))


@lru_cache(maxsize=128)
def _compile_whitespace_pattern() -> re.Pattern[str]:
    """Compile and cache whitespace normalization pattern."""
    return re.compile(r"\s+")


def normalize_text_for_embedding(text: str) -> str:
    """Normalize text for embedding generation.

    Performs:
    - Strip leading/trailing whitespace
    - Collapse multiple whitespace to single space
    - Remove null bytes and other control characters

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text suitable for embedding.
    """
    # Remove null bytes and control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse whitespace
    pattern = _compile_whitespace_pattern()
    text = pattern.sub(" ", text)

    return text.strip()


def find_overlapping_intervals(
    chunk_start: int,
    chunk_end: int,
    intervals: list[CharInterval],
) -> list[CharInterval]:
    """Find CharIntervals that overlap with a chunk's character range.

    Two intervals overlap if they share any characters. This is used to
    link entities (with char_interval) to chunks (with char_start/char_end).

    Overlap condition: interval.start < chunk_end AND interval.end > chunk_start

    Args:
        chunk_start: Start position of chunk in source text.
        chunk_end: End position of chunk in source text.
        intervals: List of CharInterval objects to check.

    Returns:
        List of CharIntervals that overlap with the chunk range.

    Example:
        >>> # Chunk covers chars 100-500
        >>> # Entity at chars 150-200 overlaps
        >>> # Entity at chars 50-99 does not overlap
        >>> overlapping = find_overlapping_intervals(100, 500, entity_intervals)
    """
    overlapping = []
    for interval in intervals:
        if interval is None:
            continue
        # Check for overlap: intervals overlap if they share any range
        if interval.start_pos < chunk_end and interval.end_pos > chunk_start:
            overlapping.append(interval)
    return overlapping


def find_entity_ids_in_range(
    chunk_start: int,
    chunk_end: int,
    entities: list,  # list[ExtractedEntity] but avoiding circular import
) -> list[str]:
    """Find entity IDs whose CharInterval overlaps the chunk range.

    This is the primary method for linking chunks to entities in GraphRAG.

    Args:
        chunk_start: Start position of chunk in source text.
        chunk_end: End position of chunk in source text.
        entities: List of ExtractedEntity objects with char_interval fields.

    Returns:
        List of entity_id strings for entities that overlap the chunk.
    """
    entity_ids = []
    for entity in entities:
        char_interval = getattr(entity, "char_interval", None)
        if char_interval is None:
            continue
        # Check for overlap
        if char_interval.start_pos < chunk_end and char_interval.end_pos > chunk_start:
            entity_ids.append(entity.entity_id)
    return entity_ids


def generate_chunk_id(
    article_id: str,
    chunk_type: str,
    index: int,
    sub_index: int | None = None,
) -> str:
    """Generate a consistent chunk ID.

    ID formats:
    - Article summary: "{article_id}-summary"
    - Section chunk: "{article_id}-sec{index}"
    - Sliding window: "{article_id}-sec{index}-sw{sub_index}"

    Args:
        article_id: Parent article ID (e.g., "ch1-art3").
        chunk_type: Type of chunk ("summary", "section", "sliding_window").
        index: Primary index (section number for section/sliding_window).
        sub_index: Secondary index (window number for sliding_window).

    Returns:
        Unique chunk ID string.

    Examples:
        >>> generate_chunk_id("ch1-art3", "summary", 0)
        'ch1-art3-summary'
        >>> generate_chunk_id("ch1-art3", "section", 2)
        'ch1-art3-sec2'
        >>> generate_chunk_id("ch1-art3", "sliding_window", 1, 3)
        'ch1-art3-sec1-sw3'
    """
    if chunk_type in {"summary", "article_summary"}:
        return f"{article_id}-summary"
    if chunk_type == "section":
        return f"{article_id}-sec{index}"
    if chunk_type == "sliding_window":
        if sub_index is None:
            raise ValueError("sub_index required for sliding_window chunks")
        return f"{article_id}-sec{index}-sw{sub_index}"
    msg = f"Unknown chunk_type: {chunk_type}"
    raise ValueError(msg)


def split_text_with_overlap(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks by token count.

    Used for Tier 3 sliding window chunking when sections exceed max_tokens.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to overlap between chunks.

    Returns:
        List of (chunk_text, char_start, char_end) tuples.

    Note:
        Character positions are approximate since we split by tokens.
        The overlap ensures context continuity across chunk boundaries.
    """
    encoding = _get_tiktoken_encoding()
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return [(text, 0, len(text))]

    chunks = []
    step = max_tokens - overlap_tokens
    if step <= 0:
        step = max_tokens // 2  # Fallback: at least half-step progress

    i = 0
    while i < len(tokens):
        end = min(i + max_tokens, len(tokens))
        chunk_tokens = tokens[i:end]
        chunk_text = encoding.decode(chunk_tokens)

        # Approximate character positions
        # This is an approximation since token→char mapping isn't 1:1
        if i == 0:
            char_start = 0
        else:
            # Find where this chunk starts in original text
            prefix_text = encoding.decode(tokens[:i])
            char_start = len(prefix_text)

        char_end = char_start + len(chunk_text)

        chunks.append((chunk_text, char_start, char_end))

        if end >= len(tokens):
            break
        i += step

    return chunks


def truncate_to_tokens(
    text: str,
    max_tokens: int,
) -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate.
        max_tokens: Maximum tokens to keep.

    Returns:
        Truncated text with at most max_tokens tokens.
    """
    encoding = _get_tiktoken_encoding()
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
