"""Tests for data models.

This module tests the Pydantic models used throughout the scraper,
including resource models for the knowledge graph.
"""

from __future__ import annotations


class TestResourceModels:
    """Tests for resource model classes."""

    def test_resource_type_enum(self) -> None:
        """Test ResourceType enum values."""
        from graphrag_kg_pipeline.models.resource import ResourceType

        assert ResourceType.IMAGE.value == "image"
        assert ResourceType.VIDEO.value == "video"
        assert ResourceType.WEBINAR.value == "webinar"
        assert ResourceType.DEFINITION.value == "definition"

    def test_image_resource_creation(self) -> None:
        """Test creating an ImageResource."""
        from graphrag_kg_pipeline.models.resource import ImageResource

        image = ImageResource(
            resource_id="img-001",
            source_article_id="ch1-art1",
            url="https://example.com/image.png",
            alt_text="Test image",
            caption="A test image caption",
        )

        assert image.url == "https://example.com/image.png"
        assert image.alt_text == "Test image"
        assert image.resource_type.value == "image"
        assert image.has_caption is True

    def test_image_resource_without_caption(self) -> None:
        """Test ImageResource without caption."""
        from graphrag_kg_pipeline.models.resource import ImageResource

        image = ImageResource(
            resource_id="img-002",
            source_article_id="ch1-art1",
            url="https://example.com/image.png",
        )

        assert image.has_caption is False

    def test_video_resource_creation(self) -> None:
        """Test creating a VideoResource."""
        from graphrag_kg_pipeline.models.resource import VideoResource

        video = VideoResource(
            resource_id="vid-001",
            source_article_id="ch1-art1",
            url="https://youtube.com/watch?v=123",
            video_id="123",
            embed_url="https://www.youtube.com/embed/123",
            title="Test Video",
            platform="youtube",
        )

        assert video.url == "https://youtube.com/watch?v=123"
        assert video.platform == "youtube"
        assert video.video_id == "123"
        assert video.resource_type.value == "video"


class TestCoreModels:
    """Tests for core scraper models."""

    def test_article_model(self) -> None:
        """Test Article model validation."""
        from graphrag_kg_pipeline.models.content import Article, ContentType

        article = Article(
            article_id="ch1-art1",
            chapter_number=1,
            article_number=1,
            title="What is Requirements Management?",
            url="https://www.jamasoftware.com/requirements-management-guide/chapter-1/article-1",
            content_type=ContentType.ARTICLE,
            markdown_content="# Requirements Management\n\nRequirements management is...",
        )

        assert article.article_id == "ch1-art1"
        assert article.title == "What is Requirements Management?"

    def test_article_computed_fields(self) -> None:
        """Test that Article has computed word/char counts."""
        from graphrag_kg_pipeline.models.content import Article, ContentType

        content = "This is test content with multiple words for counting purposes."
        article = Article(
            article_id="test",
            chapter_number=1,
            article_number=1,
            title="Test",
            url="https://example.com",
            content_type=ContentType.ARTICLE,
            markdown_content=content,
        )

        # Word count should be computed from markdown_content
        assert article.word_count > 0
        assert article.char_count > 0

    def test_chapter_model(self) -> None:
        """Test Chapter model."""
        from graphrag_kg_pipeline.models.content import Article, Chapter, ContentType

        article = Article(
            article_id="ch1-art1",
            chapter_number=1,
            article_number=1,
            title="Test Article",
            url="https://example.com/article",
            content_type=ContentType.ARTICLE,
            markdown_content="# Test",
        )

        chapter = Chapter(
            chapter_number=1,
            title="Requirements Management",
            overview_url="https://example.com/chapter-1",
            articles=[article],
        )

        assert chapter.chapter_number == 1
        assert len(chapter.articles) == 1

    def test_glossary_term_model(self) -> None:
        """Test GlossaryTerm model."""
        from graphrag_kg_pipeline.models.content import GlossaryTerm

        term = GlossaryTerm(
            term="Traceability",
            definition="The ability to trace requirements.",
        )

        assert term.term == "Traceability"
        assert "trace" in term.definition.lower()

    def test_guide_metadata_model(self) -> None:
        """Test GuideMetadata model."""
        from graphrag_kg_pipeline.models.content import GuideMetadata

        metadata = GuideMetadata(
            title="The Essential Guide",
            publisher="Jama Software",
            base_url="https://example.com",
            total_chapters=15,
        )

        assert metadata.total_chapters == 15

    def test_content_type_enum(self) -> None:
        """Test ContentType enum values."""
        from graphrag_kg_pipeline.models.content import ContentType

        assert ContentType.ARTICLE.value == "article"
        assert ContentType.CHAPTER_OVERVIEW.value == "chapter_overview"
        assert ContentType.GLOSSARY.value == "glossary"
        assert ContentType.GLOSSARY_TERM.value == "glossary_term"


class TestModelSerialization:
    """Tests for model serialization."""

    def test_article_json_serialization(self) -> None:
        """Test that Article serializes to JSON correctly."""
        from graphrag_kg_pipeline.models.content import Article, ContentType

        article = Article(
            article_id="ch1-art1",
            chapter_number=1,
            article_number=1,
            title="Test Article",
            url="https://example.com",
            content_type=ContentType.ARTICLE,
            markdown_content="# Test",
        )

        json_str = article.model_dump_json()

        assert isinstance(json_str, str)
        assert "ch1-art1" in json_str

    def test_guide_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip for full guide."""
        import json

        from graphrag_kg_pipeline.models.content import (
            Article,
            Chapter,
            ContentType,
            Glossary,
            GlossaryTerm,
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
            markdown_content="# Test\n\nContent.",
        )

        guide = RequirementsManagementGuide(
            metadata=GuideMetadata(
                title="Test Guide",
                publisher="Test Publisher",
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
            glossary=Glossary(
                url="https://example.com/glossary",
                terms=[
                    GlossaryTerm(term="Test", definition="A test term."),
                ],
            ),
        )

        json_str = guide.model_dump_json()

        # Parse back
        parsed = json.loads(json_str)
        restored = RequirementsManagementGuide.model_validate(parsed)

        assert restored.metadata.title == guide.metadata.title
        assert len(restored.chapters) == len(guide.chapters)
        assert restored.chapters[0].articles[0].article_id == "ch1-art1"
