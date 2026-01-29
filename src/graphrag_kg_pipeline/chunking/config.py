"""Configuration for hierarchical HTML chunking.

This module defines the HierarchicalChunkingConfig frozen dataclass that
configures the LangChain-based chunking strategy.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HierarchicalChunkingConfig:
    """Configuration for hierarchical HTML chunking.

    This frozen dataclass provides immutable configuration for the
    two-stage chunking strategy:

    1. HTML Header Stage: Split by heading elements (h1, h2, h3)
    2. Character Stage: Further split large sections with overlapping windows

    Attributes:
        headers_to_split_on: List of (tag, metadata_key) tuples for header splitting.
        sliding_window_size: Target chunk size in characters for the char splitter.
        sliding_window_overlap: Character overlap between sliding window chunks.
        sliding_window_threshold: Character threshold above which to apply char splitting.
        min_chunk_size: Minimum chunk size to keep (filters tiny fragments).
        separators: Separator patterns for RecursiveCharacterTextSplitter.
        keep_separator: Whether to keep separators in chunk boundaries.
    """

    # HTML Header splitting configuration
    headers_to_split_on: tuple[tuple[str, str], ...] = field(
        default_factory=lambda: (
            ("h1", "article_title"),
            ("h2", "section"),
            ("h3", "subsection"),
        )
    )

    # Sliding window configuration (for large sections)
    sliding_window_size: int = 512
    sliding_window_overlap: int = 64
    sliding_window_threshold: int = 1500

    # Filtering configuration
    min_chunk_size: int = 50

    # RecursiveCharacterTextSplitter configuration
    separators: tuple[str, ...] = field(
        default_factory=lambda: (
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            "? ",  # Question endings
            "! ",  # Exclamation endings
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Words
            "",  # Characters (last resort)
        )
    )
    keep_separator: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.sliding_window_size <= 0:
            msg = "sliding_window_size must be positive"
            raise ValueError(msg)
        if self.sliding_window_overlap >= self.sliding_window_size:
            msg = "sliding_window_overlap must be less than sliding_window_size"
            raise ValueError(msg)
        if self.sliding_window_threshold <= 0:
            msg = "sliding_window_threshold must be positive"
            raise ValueError(msg)
        if self.min_chunk_size < 0:
            msg = "min_chunk_size must be non-negative"
            raise ValueError(msg)

    @classmethod
    def for_rag(cls) -> "HierarchicalChunkingConfig":
        """Create configuration optimized for RAG retrieval.

        Returns a configuration with smaller chunks and more overlap,
        suitable for semantic search and retrieval.

        Returns:
            Configuration optimized for RAG use cases.
        """
        return cls(
            sliding_window_size=400,
            sliding_window_overlap=80,
            sliding_window_threshold=1200,
            min_chunk_size=30,
        )

    @classmethod
    def for_extraction(cls) -> "HierarchicalChunkingConfig":
        """Create configuration optimized for entity extraction.

        Returns a configuration with larger chunks for better context
        during LLM entity extraction.

        Returns:
            Configuration optimized for entity extraction.
        """
        return cls(
            sliding_window_size=1024,
            sliding_window_overlap=128,
            sliding_window_threshold=2500,
            min_chunk_size=100,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "headers_to_split_on": list(self.headers_to_split_on),
            "sliding_window_size": self.sliding_window_size,
            "sliding_window_overlap": self.sliding_window_overlap,
            "sliding_window_threshold": self.sliding_window_threshold,
            "min_chunk_size": self.min_chunk_size,
            "separators": list(self.separators),
            "keep_separator": self.keep_separator,
        }
