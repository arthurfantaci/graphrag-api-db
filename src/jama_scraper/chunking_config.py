"""Configuration for the chunking strategy.

This module provides a frozen dataclass configuration for controlling
how articles are chunked into GraphRAG-friendly segments.

The three-tier chunking strategy:
1. Tier 1 (Article Summary): Condensed overview, ~300 tokens
2. Tier 2 (Section): Natural heading-based boundaries, ~500-1500 tokens
3. Tier 3 (Sliding Window): Overlapping windows for large sections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Frozen configuration for chunking strategy.

    This configuration controls how articles are split into chunks
    for GraphRAG retrieval. The three-tier approach balances:
    - Broad context (article summaries)
    - Natural semantic boundaries (sections)
    - Fine-grained retrieval (sliding windows)

    Attributes:
        include_summaries: Whether to create Tier 1 article summary chunks.
        summary_max_tokens: Maximum tokens for summary chunks.
        section_max_tokens: Maximum tokens before applying sliding window.
        section_min_tokens: Minimum tokens to include a section (skip very short).
        sliding_window_size: Token size for Tier 3 sliding windows.
        sliding_window_overlap: Token overlap between consecutive windows.
        sliding_window_threshold: Apply sliding window when section exceeds this.
        link_entities: Enable CharInterval overlap detection for entity linkage.
        checkpoint_dir: Directory for chunking checkpoints (for resume).
    """

    # Tier 1: Article summaries
    include_summaries: bool = True
    summary_max_tokens: int = 300

    # Tier 2: Section-based chunks
    section_max_tokens: int = 1500
    section_min_tokens: int = 50  # Skip very short sections

    # Tier 3: Sliding window (for large sections)
    sliding_window_size: int = 512
    sliding_window_overlap: int = 64
    sliding_window_threshold: int = 1500  # Apply when section > this

    # Entity linkage
    link_entities: bool = True

    # Checkpoint directory (for resume capability)
    checkpoint_dir: Path = field(default_factory=lambda: Path(".chunking_cache"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if self.summary_max_tokens <= 0:
            raise ValueError("summary_max_tokens must be positive")

        if self.section_max_tokens <= 0:
            raise ValueError("section_max_tokens must be positive")

        if self.section_min_tokens < 0:
            raise ValueError("section_min_tokens must be non-negative")

        if self.sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive")

        if self.sliding_window_overlap < 0:
            raise ValueError("sliding_window_overlap must be non-negative")

        if self.sliding_window_overlap >= self.sliding_window_size:
            raise ValueError(
                "sliding_window_overlap must be less than sliding_window_size"
            )

        if self.sliding_window_threshold <= 0:
            raise ValueError("sliding_window_threshold must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "include_summaries": self.include_summaries,
            "summary_max_tokens": self.summary_max_tokens,
            "section_max_tokens": self.section_max_tokens,
            "section_min_tokens": self.section_min_tokens,
            "sliding_window_size": self.sliding_window_size,
            "sliding_window_overlap": self.sliding_window_overlap,
            "sliding_window_threshold": self.sliding_window_threshold,
            "link_entities": self.link_entities,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

    @classmethod
    def from_args(
        cls,
        summary_max_tokens: int | None = None,
        section_max_tokens: int | None = None,
        sliding_window_size: int | None = None,
        sliding_window_overlap: int | None = None,
        checkpoint_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> ChunkingConfig:
        """Create configuration from CLI arguments.

        Args:
            summary_max_tokens: Override default summary token limit.
            section_max_tokens: Override default section token limit.
            sliding_window_size: Override default window size.
            sliding_window_overlap: Override default window overlap.
            checkpoint_dir: Override default checkpoint directory.
            **kwargs: Additional configuration options.

        Returns:
            ChunkingConfig instance with specified overrides.
        """
        config_dict: dict[str, Any] = {}

        if summary_max_tokens is not None:
            config_dict["summary_max_tokens"] = summary_max_tokens

        if section_max_tokens is not None:
            config_dict["section_max_tokens"] = section_max_tokens

        if sliding_window_size is not None:
            config_dict["sliding_window_size"] = sliding_window_size

        if sliding_window_overlap is not None:
            config_dict["sliding_window_overlap"] = sliding_window_overlap

        if checkpoint_dir is not None:
            if isinstance(checkpoint_dir, str):
                checkpoint_dir = Path(checkpoint_dir)
            config_dict["checkpoint_dir"] = checkpoint_dir

        # Merge any additional kwargs
        for key, value in kwargs.items():
            if hasattr(cls, key):
                config_dict[key] = value

        return cls(**config_dict)


# Default configuration instance
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()
