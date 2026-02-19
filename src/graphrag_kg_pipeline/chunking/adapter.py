"""Adapter for neo4j_graphrag TextSplitter integration.

This module provides the LangChainTextSplitterAdapter that bridges
our HierarchicalHTMLSplitter to neo4j_graphrag's expected interface.
"""

from typing import TYPE_CHECKING

from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig
from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
    HierarchicalHTMLSplitter,
    MarkdownSplitter,
)

if TYPE_CHECKING:
    from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter


def create_text_splitter_adapter(
    config: HierarchicalChunkingConfig | None = None,
    use_markdown: bool = False,
) -> "TextSplitter":
    """Create a neo4j_graphrag compatible text splitter.

    Factory function that creates an adapter wrapping our hierarchical
    splitter for use with neo4j_graphrag's SimpleKGPipeline.

    Args:
        config: Chunking configuration. Uses defaults if not provided.
        use_markdown: If True, use MarkdownSplitter instead of HTML splitter.

    Returns:
        A TextSplitter compatible with neo4j_graphrag pipelines.

    Example:
        >>> from graphrag_kg_pipeline.chunking import create_text_splitter_adapter
        >>> splitter = create_text_splitter_adapter(HierarchicalChunkingConfig.for_rag())
        >>> # Use with SimpleKGPipeline
        >>> pipeline = SimpleKGPipeline(..., text_splitter=splitter)
    """
    from neo4j_graphrag.experimental.components.text_splitters.langchain import (
        LangChainTextSplitterAdapter,
    )

    config = config or HierarchicalChunkingConfig()

    splitter = MarkdownSplitter(config) if use_markdown else HierarchicalHTMLSplitter(config)

    return LangChainTextSplitterAdapter(splitter)
