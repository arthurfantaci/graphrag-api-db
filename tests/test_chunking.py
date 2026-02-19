"""Tests for the chunking module.

This module tests the LangChain-based hierarchical chunking functionality,
including the HTMLHeaderTextSplitter integration, Chonkie semantic chunking,
and configuration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig


class TestHierarchicalChunkingConfig:
    """Tests for HierarchicalChunkingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default configuration values are sensible."""
        config = HierarchicalChunkingConfig()

        assert config.sliding_window_size == 512
        assert config.sliding_window_overlap == 64
        assert config.sliding_window_threshold == 1500
        assert config.min_chunk_size == 100  # Raised from 50 to prevent degenerate chunks
        assert len(config.headers_to_split_on) == 3

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = HierarchicalChunkingConfig(
            sliding_window_size=1024,
            sliding_window_overlap=128,
            sliding_window_threshold=2000,
        )

        assert config.sliding_window_size == 1024
        assert config.sliding_window_overlap == 128
        assert config.sliding_window_threshold == 2000

    def test_frozen_immutability(self) -> None:
        """Test that config is immutable (frozen dataclass)."""
        config = HierarchicalChunkingConfig()

        with pytest.raises(AttributeError):
            config.sliding_window_size = 1024  # type: ignore[misc]

    def test_headers_to_split_on_default(self) -> None:
        """Test default header splitting configuration."""
        config = HierarchicalChunkingConfig()

        # Should include h1, h2, h3 by default
        header_tags = [h[0] for h in config.headers_to_split_on]
        assert "h1" in header_tags
        assert "h2" in header_tags
        assert "h3" in header_tags


class TestHierarchicalHTMLSplitter:
    """Tests for HierarchicalHTMLSplitter class."""

    def test_split_simple_html(self, sample_article_html_with_headers: str) -> None:
        """Test splitting HTML with multiple headers."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        config = HierarchicalChunkingConfig(
            sliding_window_size=512,
            sliding_window_overlap=64,
        )
        splitter = HierarchicalHTMLSplitter(config)

        chunks = splitter.split_text(sample_article_html_with_headers)

        # Should produce multiple chunks based on headers
        assert len(chunks) > 0

        # Each chunk should be a string
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_split_returns_documents(self, sample_article_html_with_headers: str) -> None:
        """Test that split_text_as_documents returns Document objects."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        config = HierarchicalChunkingConfig()
        splitter = HierarchicalHTMLSplitter(config)

        documents = splitter.split_text_as_documents(sample_article_html_with_headers)

        # Should produce Document objects
        assert len(documents) > 0

        for doc in documents:
            assert hasattr(doc, "page_content")
            assert len(doc.page_content) > 0

    def test_large_section_splitting(self) -> None:
        """Test that large sections are further split by character splitter."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        # Create HTML with a very large section
        large_content = "This is test content. " * 500  # ~11000 characters
        html = f"<h1>Title</h1><p>{large_content}</p>"

        config = HierarchicalChunkingConfig(
            sliding_window_size=512,
            sliding_window_overlap=64,
            sliding_window_threshold=1000,  # Low threshold to force splitting
        )
        splitter = HierarchicalHTMLSplitter(config)

        chunks = splitter.split_text(html)

        # Should produce multiple chunks from the large section
        assert len(chunks) > 1

    def test_empty_html(self) -> None:
        """Test handling of empty HTML."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        splitter = HierarchicalHTMLSplitter()

        chunks = splitter.split_text("")

        # Should handle gracefully
        assert isinstance(chunks, list)
        assert len(chunks) == 0


class TestHeaderMetadataPrepending:
    """Tests for header metadata prepending to chunk text."""

    def test_prepend_article_metadata(self) -> None:
        """Test that article metadata is prepended to chunk content."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        html = (
            "<h1>Guide Title</h1>"
            "<h2>Requirements Traceability</h2>"
            "<p>" + "Traceability is the ability to trace requirements. " * 10 + "</p>"
            "<h2>Impact Analysis</h2>"
            "<p>" + "Impact analysis helps assess change effects. " * 10 + "</p>"
        )

        config = HierarchicalChunkingConfig(min_chunk_size=50)
        splitter = HierarchicalHTMLSplitter(config)

        docs = splitter.split_text_as_documents(
            html,
            article_metadata={"article_title": "Best Practices"},
        )

        assert len(docs) > 0
        # Chunks should start with the article context prefix
        assert docs[0].page_content.startswith("Article: Best Practices")

    def test_no_metadata_no_prefix(self) -> None:
        """Test that chunks are unchanged without article_metadata."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        html = (
            "<h2>Section Title</h2><p>" + "Some meaningful content here for testing. " * 10 + "</p>"
        )

        config = HierarchicalChunkingConfig(min_chunk_size=50)
        splitter = HierarchicalHTMLSplitter(config)

        docs = splitter.split_text_as_documents(html)

        assert len(docs) > 0
        # No prefix should be added
        assert not docs[0].page_content.startswith("Article:")


class TestMarkdownSplitter:
    """Tests for the MarkdownSplitter class."""

    def test_split_markdown(self) -> None:
        """Test splitting markdown content."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import MarkdownSplitter

        markdown = """# Title

Introduction paragraph with enough content to pass the minimum chunk size filter.

## Section One

This is section one content with enough text to be meaningful for entity extraction and retrieval purposes. Requirements traceability is the ability to trace requirements through their lifecycle.

## Section Two

This is section two content with more text about impact analysis. Impact analysis helps teams understand how changes to one requirement affect other parts of the system.
"""

        splitter = MarkdownSplitter()
        chunks = splitter.split_text(markdown)

        assert len(chunks) > 0

    def test_empty_markdown(self) -> None:
        """Test handling of empty markdown."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import MarkdownSplitter

        splitter = MarkdownSplitter()
        chunks = splitter.split_text("")

        assert isinstance(chunks, list)
        assert len(chunks) == 0


class TestTextSplitterAdapter:
    """Tests for the LangChain text splitter adapter."""

    def test_create_adapter(self) -> None:
        """Test creating adapter from config."""
        from graphrag_kg_pipeline.chunking.adapter import create_text_splitter_adapter
        from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig

        config = HierarchicalChunkingConfig()
        adapter = create_text_splitter_adapter(config)

        assert adapter is not None

    def test_adapter_has_splitter(self) -> None:
        """Test that adapter wraps a splitter correctly."""
        from graphrag_kg_pipeline.chunking.adapter import create_text_splitter_adapter
        from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig

        config = HierarchicalChunkingConfig()
        adapter = create_text_splitter_adapter(config)

        # Adapter should have the expected interface
        assert hasattr(adapter, "text_splitter")


class TestSemanticChunkingConfig:
    """Tests for semantic chunking configuration."""

    def test_semantic_chunking_defaults(self) -> None:
        """Verify semantic chunking is disabled by default."""
        config = HierarchicalChunkingConfig()

        assert config.use_semantic_chunking is False
        assert config.semantic_threshold == 0.5

    def test_semantic_chunking_enabled(self) -> None:
        """Verify semantic chunking can be enabled via config."""
        config = HierarchicalChunkingConfig(
            use_semantic_chunking=True,
            semantic_threshold=0.7,
        )

        assert config.use_semantic_chunking is True
        assert config.semantic_threshold == 0.7


class TestSemanticChunking:
    """Tests for Chonkie SemanticChunker integration."""

    def _make_mock_chunk(self, text: str) -> MagicMock:
        """Create a mock Chonkie SemanticChunk."""
        chunk = MagicMock()
        chunk.text = text
        chunk.token_count = len(text.split())
        return chunk

    def test_semantic_split_used_when_enabled(self) -> None:
        """Verify SemanticChunker is called for large sections when enabled."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        config = HierarchicalChunkingConfig(
            use_semantic_chunking=True,
            semantic_threshold=0.5,
            sliding_window_threshold=500,
            min_chunk_size=50,
        )
        splitter = HierarchicalHTMLSplitter(config)

        # Mock the _semantic_split to return valid docs
        from langchain_core.documents import Document

        mock_docs = [
            Document(
                page_content="First semantic chunk with enough content for testing.", metadata={}
            ),
            Document(
                page_content="Second semantic chunk also with enough content here.", metadata={}
            ),
        ]
        splitter._semantic_split = MagicMock(return_value=mock_docs)

        # Create HTML with a large section that exceeds threshold
        large_text = "This is test content about requirements management. " * 30
        html = f"<h1>Title</h1><p>{large_text}</p>"

        docs = splitter.split_text_as_documents(html)

        # Should have called _semantic_split
        splitter._semantic_split.assert_called()
        assert len(docs) >= 2

    def test_rcts_fallback_when_semantic_disabled(self) -> None:
        """Verify RCTS is used when semantic chunking is disabled."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        config = HierarchicalChunkingConfig(
            use_semantic_chunking=False,
            sliding_window_threshold=500,
            sliding_window_size=256,
            min_chunk_size=50,
        )
        splitter = HierarchicalHTMLSplitter(config)

        large_text = "This is test content about requirements management. " * 30
        html = f"<h1>Title</h1><p>{large_text}</p>"

        docs = splitter.split_text_as_documents(html)

        # Should produce chunks via RCTS (no _semantic_chunker attribute)
        assert not hasattr(splitter, "_semantic_chunker")
        assert len(docs) > 0

    def test_rcts_fallback_on_empty_semantic_output(self) -> None:
        """Verify fallback to RCTS when SemanticChunker returns empty."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        config = HierarchicalChunkingConfig(
            use_semantic_chunking=True,
            sliding_window_threshold=500,
            sliding_window_size=256,
            min_chunk_size=50,
        )
        splitter = HierarchicalHTMLSplitter(config)

        # Mock _semantic_split to return None (trigger fallback)
        splitter._semantic_split = MagicMock(return_value=None)

        large_text = "This is test content about requirements management. " * 30
        html = f"<h1>Title</h1><p>{large_text}</p>"

        docs = splitter.split_text_as_documents(html)

        # Should still produce chunks via RCTS fallback
        assert len(docs) > 0

    def test_metadata_preserved_in_semantic_split(self) -> None:
        """Verify metadata is preserved through semantic splitting."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import (
            HierarchicalHTMLSplitter,
        )

        config = HierarchicalChunkingConfig(
            use_semantic_chunking=True,
            sliding_window_threshold=500,
            min_chunk_size=50,
        )
        splitter = HierarchicalHTMLSplitter(config)

        # Mock _semantic_split to return docs with metadata
        from langchain_core.documents import Document

        mock_docs = [
            Document(
                page_content="Chunk with enough text for testing purposes.",
                metadata={"section": "Requirements Traceability"},
            ),
        ]
        splitter._semantic_split = MagicMock(return_value=mock_docs)

        large_text = "Requirements traceability content. " * 30
        html = f"<h2>Requirements Traceability</h2><p>{large_text}</p>"

        docs = splitter.split_text_as_documents(html)

        # Should contain our mock docs with metadata
        found = any("section" in doc.metadata for doc in docs)
        assert found

    def test_markdown_splitter_semantic_support(self) -> None:
        """Verify MarkdownSplitter also supports semantic chunking."""
        from graphrag_kg_pipeline.chunking.hierarchical_chunker import MarkdownSplitter

        config = HierarchicalChunkingConfig(
            use_semantic_chunking=True,
            sliding_window_threshold=500,
            min_chunk_size=50,
        )
        splitter = MarkdownSplitter(config)

        # Mock _semantic_split to return None (fallback to RCTS)
        splitter._semantic_split = MagicMock(return_value=None)

        large_text = "This is long markdown content. " * 30
        markdown = f"# Title\n\n{large_text}"

        docs = splitter.split_text_as_documents(markdown)

        # Should produce chunks via RCTS fallback
        assert len(docs) > 0
