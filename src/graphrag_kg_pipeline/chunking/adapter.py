"""Adapter for neo4j_graphrag TextSplitter integration.

This module provides the LangChainTextSplitterAdapter that bridges
our HierarchicalHTMLSplitter to neo4j_graphrag's expected interface.
"""

from typing import TYPE_CHECKING, Any

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


class ChunkMetadata:
    """Helper for extracting and enriching chunk metadata.

    Provides utilities for working with chunk metadata from
    LangChain documents during pipeline processing.
    """

    @staticmethod
    def extract_heading_hierarchy(metadata: dict[str, Any]) -> list[str]:
        """Extract heading hierarchy from chunk metadata.

        Args:
            metadata: Document metadata dict from LangChain.

        Returns:
            List of headings from h1 to deepest level.
        """
        headings = []
        for key in ["article_title", "section", "subsection", "h1", "h2", "h3"]:
            value = metadata.get(key)
            if value:
                headings.append(value)
        return headings

    @staticmethod
    def get_section_path(metadata: dict[str, Any]) -> str:
        """Get dot-separated section path from metadata.

        Args:
            metadata: Document metadata dict.

        Returns:
            Section path like "Article Title.Section.Subsection".
        """
        headings = ChunkMetadata.extract_heading_hierarchy(metadata)
        return ".".join(headings) if headings else "root"

    @staticmethod
    def enrich_with_article_context(
        metadata: dict[str, Any],
        article_id: str,
        chapter_number: int,
        article_title: str,
    ) -> dict[str, Any]:
        """Add article context to chunk metadata.

        Args:
            metadata: Original chunk metadata.
            article_id: Source article ID.
            chapter_number: Chapter number.
            article_title: Article title.

        Returns:
            Enriched metadata dict.
        """
        return {
            **metadata,
            "article_id": article_id,
            "chapter_number": chapter_number,
            "article_title": article_title,
            "section_path": ChunkMetadata.get_section_path(metadata),
        }
