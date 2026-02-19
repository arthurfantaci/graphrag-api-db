"""Tests for the loaders module.

This module tests the custom DataLoader implementations for
loading HTML content into the neo4j_graphrag pipeline.
"""

from __future__ import annotations

import pytest


class TestArticleIndex:
    """Tests for the ArticleIndex dataclass."""

    def test_article_index_creation(self) -> None:
        """Test creating an empty ArticleIndex."""
        from graphrag_kg_pipeline.loaders.index_builder import ArticleIndex

        index = ArticleIndex()

        assert index.total_articles == 0
        assert len(index.by_id) == 0
        assert len(index.by_chapter) == 0
        assert len(index.by_url) == 0

    def test_article_index_get_article(self) -> None:
        """Test getting article by ID."""
        from graphrag_kg_pipeline.loaders.index_builder import ArticleIndex

        index = ArticleIndex()
        index.by_id["ch1-art1"] = {
            "title": "Test Article",
            "url": "https://example.com",
        }

        article = index.get_article("ch1-art1")
        assert article is not None
        assert article["title"] == "Test Article"

        # Test missing article
        assert index.get_article("nonexistent") is None

    def test_article_index_get_chapter_articles(self) -> None:
        """Test getting articles by chapter."""
        from graphrag_kg_pipeline.loaders.index_builder import ArticleIndex

        index = ArticleIndex()
        index.by_id["ch1-art1"] = {"title": "Article 1"}
        index.by_id["ch1-art2"] = {"title": "Article 2"}
        index.by_chapter[1] = ["ch1-art1", "ch1-art2"]

        articles = index.get_chapter_articles(1)
        assert len(articles) == 2

        # Test missing chapter
        assert len(index.get_chapter_articles(99)) == 0

    def test_article_index_contains(self) -> None:
        """Test membership check."""
        from graphrag_kg_pipeline.loaders.index_builder import ArticleIndex

        index = ArticleIndex()
        index.by_id["ch1-art1"] = {"title": "Test"}

        assert "ch1-art1" in index
        assert "nonexistent" not in index


class TestBuildArticleIndex:
    """Tests for the build_article_index function."""

    def test_build_empty_guide(self) -> None:
        """Test building index from empty guide."""
        from graphrag_kg_pipeline.loaders.index_builder import build_article_index
        from graphrag_kg_pipeline.models_core import (
            Glossary,
            GuideMetadata,
            RequirementsManagementGuide,
        )

        guide = RequirementsManagementGuide(
            metadata=GuideMetadata(
                title="Test Guide",
                publisher="Test",
                base_url="https://example.com",
                total_chapters=0,
            ),
            chapters=[],
            glossary=Glossary(url="https://example.com/glossary", terms=[]),
        )

        index = build_article_index(guide)

        assert index.total_articles == 0

    def test_build_index_with_content_flag(self) -> None:
        """Test building index with include_content=False."""
        from graphrag_kg_pipeline.loaders.index_builder import build_article_index
        from graphrag_kg_pipeline.models_core import (
            Article,
            Chapter,
            ContentType,
            Glossary,
            GuideMetadata,
            RequirementsManagementGuide,
        )

        article = Article(
            article_id="ch1-art1",
            chapter_number=1,
            article_number=1,
            title="Test Article",
            url="https://example.com/article",
            content_type=ContentType.ARTICLE,
            markdown_content="# Test\n\nContent here.",
        )

        guide = RequirementsManagementGuide(
            metadata=GuideMetadata(
                title="Test Guide",
                publisher="Test",
                base_url="https://example.com",
                total_chapters=1,
            ),
            chapters=[
                Chapter(
                    chapter_number=1,
                    title="Chapter 1",
                    overview_url="https://example.com/ch1",
                    articles=[article],
                )
            ],
            glossary=Glossary(url="https://example.com/glossary", terms=[]),
        )

        # With content
        index_with = build_article_index(guide, include_content=True)
        assert "markdown_content" in index_with.by_id["ch1-art1"]

        # Without content
        index_without = build_article_index(guide, include_content=False)
        assert "markdown_content" not in index_without.by_id["ch1-art1"]


class TestJamaHTMLLoader:
    """Tests for the JamaHTMLLoader class."""

    def test_loader_initialization(self) -> None:
        """Test that loader initializes correctly."""
        from graphrag_kg_pipeline.loaders.html_loader import JamaHTMLLoader

        loader = JamaHTMLLoader()
        assert loader is not None
        assert loader.article_index == {}
        assert loader.preprocess_html is True

    def test_loader_with_article_index(self) -> None:
        """Test loader initialization with article index."""
        from graphrag_kg_pipeline.loaders.html_loader import JamaHTMLLoader

        index = {
            "ch1-art1": {
                "title": "Test Article",
                "url": "https://example.com",
                "markdown_content": "# Test\n\nContent here.",
            }
        }

        loader = JamaHTMLLoader(article_index=index)

        assert loader.article_index == index
        assert "ch1-art1" in loader.article_index

    def test_loader_preprocess_flag(self) -> None:
        """Test loader with preprocess flag disabled."""
        from graphrag_kg_pipeline.loaders.html_loader import JamaHTMLLoader

        loader = JamaHTMLLoader(preprocess_html=False)
        assert loader.preprocess_html is False

    @pytest.mark.asyncio
    async def test_loader_run_with_article_id(self) -> None:
        """Test loading article by ID from index."""
        from graphrag_kg_pipeline.loaders.html_loader import JamaHTMLLoader

        index = {
            "ch1-art1": {
                "title": "Test Article",
                "url": "https://example.com/article",
                "chapter_number": 1,
                "markdown_content": "# Test Article\n\nThis is test content.",
            }
        }

        loader = JamaHTMLLoader(article_index=index)
        result = await loader.run("ch1-art1")

        assert result is not None
        # Result should be a PdfDocument with text
        assert hasattr(result, "text")
        assert "Test Article" in result.text or "test content" in result.text

    @pytest.mark.asyncio
    async def test_loader_returns_metadata(self) -> None:
        """Test that loader returns document with metadata via document_info."""
        from graphrag_kg_pipeline.loaders.html_loader import JamaHTMLLoader

        index = {
            "ch1-art1": {
                "title": "Test Article",
                "url": "https://example.com/article",
                "chapter_number": 1,
                "markdown_content": "# Test\n\nContent.",
            }
        }

        loader = JamaHTMLLoader(article_index=index)
        result = await loader.run("ch1-art1")

        # Should have document_info with metadata about the article
        assert hasattr(result, "document_info")
        assert result.document_info.metadata is not None
        assert "article_id" in result.document_info.metadata
        assert result.document_info.metadata["article_id"] == "ch1-art1"

    @pytest.mark.asyncio
    async def test_loader_missing_article_raises(self) -> None:
        """Test that missing article raises error."""
        from graphrag_kg_pipeline.loaders.html_loader import JamaHTMLLoader

        loader = JamaHTMLLoader(article_index={})

        with pytest.raises((ValueError, KeyError)):
            await loader.run("nonexistent-article")
