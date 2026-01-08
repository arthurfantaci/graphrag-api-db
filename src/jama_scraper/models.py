"""Pydantic models for Jama Requirements Management Guide content.

Designed for:
- LLM/Agent consumption
- RAG retrieval systems
- Knowledge graph construction
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class ContentType(str, Enum):
    """Type of content in the guide."""

    CHAPTER_OVERVIEW = "chapter_overview"
    ARTICLE = "article"
    GLOSSARY = "glossary"
    GLOSSARY_TERM = "glossary_term"


class CrossReference(BaseModel):
    """A cross-reference to another section or external resource."""

    text: str = Field(description="The anchor text of the link")
    url: str = Field(description="The target URL")
    is_internal: bool = Field(description="Whether this links to another guide section")
    target_section_id: str | None = Field(
        default=None, description="Section ID if internal link (e.g., 'ch1-art3')"
    )


class Section(BaseModel):
    """A section within an article (h2, h3 level content blocks)."""

    heading: str = Field(description="Section heading text")
    level: int = Field(description="Heading level (2 for h2, 3 for h3, etc.)")
    content: str = Field(description="Section content in markdown format")
    cross_references: list[CrossReference] = Field(
        default_factory=list, description="Links found within this section"
    )


class ImageReference(BaseModel):
    """A reference to an image/graphic in the article."""

    url: str = Field(description="Image source URL")
    alt_text: str = Field(default="", description="Alt text description of the image")
    title: str | None = Field(default=None, description="Title attribute if present")
    caption: str | None = Field(default=None, description="Figure caption if present")
    context: str | None = Field(
        default=None, description="Surrounding text context for the image"
    )


class VideoReference(BaseModel):
    """A reference to a video/webinar embedded in the article."""

    url: str = Field(description="Video URL (e.g., YouTube watch URL)")
    embed_url: str = Field(description="Embed URL for the video")
    video_id: str = Field(description="Video ID (e.g., YouTube video ID)")
    platform: str = Field(
        default="youtube", description="Video platform (youtube, vimeo, etc.)"
    )
    title: str | None = Field(default=None, description="Video title if available")
    context: str | None = Field(
        default=None, description="Surrounding text context for the video"
    )


class WebinarReference(BaseModel):
    """A reference to a webinar resource linked in the article.

    Captures webinars from various patterns:
    - Text links to webinar URLs
    - Image thumbnails linking to webinars
    - Featured webinar sections with "In This Webinar" descriptions
    """

    url: str = Field(description="Webinar URL")
    title: str = Field(description="Webinar title or link text")
    description: str | None = Field(
        default=None,
        description="Webinar description from 'In This Webinar' section if available",
    )
    thumbnail_url: str | None = Field(
        default=None, description="Thumbnail image URL if linked from an image"
    )
    context: str | None = Field(
        default=None, description="Surrounding heading context for the webinar"
    )


class RelatedArticle(BaseModel):
    """A related article callout within the content.

    These are explicitly marked with "RELATED ARTICLE:" in av_promobox elements.
    """

    url: str = Field(description="URL of the related article")
    title: str = Field(description="Title of the related article")
    source_type: str = Field(
        default="internal",
        description="Source type: 'internal' (guide), 'blog', 'resource', 'external'",
    )


class Article(BaseModel):
    """An individual article/page within a chapter."""

    # Identification
    article_id: str = Field(description="Unique article ID (e.g., 'ch1-art3')")
    chapter_number: int = Field(description="Parent chapter number (1-15)")
    article_number: int = Field(
        description="Article number within chapter (0 for overview)"
    )

    # Content
    title: str = Field(description="Article title")
    url: str = Field(description="Source URL")
    content_type: ContentType = Field(description="Type of content")

    # Full content
    raw_html: str | None = Field(
        default=None, description="Original HTML content (optional, for debugging)"
    )
    markdown_content: str = Field(description="Full article content in markdown")
    sections: list[Section] = Field(
        default_factory=list, description="Parsed sections with headings"
    )

    # Metadata for RAG
    summary: str | None = Field(
        default=None, description="Brief summary for retrieval (can be LLM-generated)"
    )
    key_concepts: list[str] = Field(
        default_factory=list, description="Key terms/concepts mentioned"
    )
    cross_references: list[CrossReference] = Field(
        default_factory=list, description="All cross-references in this article"
    )
    images: list[ImageReference] = Field(
        default_factory=list,
        description="Images and graphics referenced in this article",
    )
    videos: list[VideoReference] = Field(
        default_factory=list,
        description="Videos embedded in this article (e.g., YouTube)",
    )
    webinars: list[WebinarReference] = Field(
        default_factory=list,
        description="Webinar resources linked in this article",
    )
    related_articles: list[RelatedArticle] = Field(
        default_factory=list,
        description="Explicitly marked related article callouts",
    )

    # Timestamps
    scraped_at: datetime = Field(
        default_factory=_utc_now, description="When this content was scraped"
    )

    @computed_field
    @property
    def word_count(self) -> int:
        """Approximate word count of the article."""
        return len(self.markdown_content.split())

    @computed_field
    @property
    def char_count(self) -> int:
        """Character count of the article."""
        return len(self.markdown_content)


class Chapter(BaseModel):
    """A chapter containing multiple articles."""

    chapter_number: int = Field(description="Chapter number (1-15)")
    title: str = Field(description="Chapter title")
    overview_url: str = Field(description="URL to chapter overview page")
    description: str | None = Field(
        default=None, description="Brief description of chapter contents"
    )
    articles: list[Article] = Field(
        default_factory=list, description="All articles in this chapter"
    )

    @computed_field
    @property
    def article_count(self) -> int:
        """Number of articles in this chapter."""
        return len(self.articles)

    @computed_field
    @property
    def total_word_count(self) -> int:
        """Total words across all articles."""
        return sum(a.word_count for a in self.articles)


class GlossaryTerm(BaseModel):
    """A single glossary term and definition."""

    term: str = Field(description="The term being defined")
    acronym: str | None = Field(
        default=None, description="Acronym if applicable (e.g., 'AoA')"
    )
    definition: str = Field(description="The definition text")
    related_terms: list[str] = Field(
        default_factory=list, description="Related glossary terms"
    )
    related_chapters: list[int] = Field(
        default_factory=list, description="Chapter numbers where this term is relevant"
    )


class Glossary(BaseModel):
    """The complete glossary."""

    url: str = Field(description="Source URL for glossary")
    terms: list[GlossaryTerm] = Field(
        default_factory=list, description="All glossary terms"
    )
    scraped_at: datetime = Field(default_factory=_utc_now)

    @computed_field
    @property
    def term_count(self) -> int:
        """Number of terms in glossary."""
        return len(self.terms)


class GuideMetadata(BaseModel):
    """Metadata about the complete guide."""

    title: str = Field(
        default="The Essential Guide to Requirements Management and Traceability"
    )
    publisher: str = Field(default="Jama Software")
    base_url: str = Field(
        default="https://www.jamasoftware.com/requirements-management-guide/"
    )
    scraped_at: datetime = Field(default_factory=_utc_now)
    scraper_version: str = Field(default="0.1.0")
    total_chapters: int = Field(default=15)
    includes_glossary: bool = Field(default=True)


class RequirementsManagementGuide(BaseModel):
    """The complete Requirements Management Guide.

    This is the root model containing all chapters, articles, and the glossary.
    Designed for serialization to JSON/JSONL for LLM consumption.
    """

    metadata: GuideMetadata = Field(default_factory=GuideMetadata)
    chapters: list[Chapter] = Field(default_factory=list, description="All 15 chapters")
    glossary: Glossary | None = Field(default=None, description="The glossary section")

    @computed_field
    @property
    def total_articles(self) -> int:
        """Total number of articles across all chapters."""
        return sum(ch.article_count for ch in self.chapters)

    @computed_field
    @property
    def total_word_count(self) -> int:
        """Total words in the entire guide."""
        return sum(ch.total_word_count for ch in self.chapters)

    def to_jsonl_articles(self) -> list[dict]:
        """Export all articles as individual JSONL records.

        Each record is self-contained with chapter context,
        ideal for RAG chunking and retrieval.
        """
        records = []
        for chapter in self.chapters:
            for article in chapter.articles:
                record = {
                    "type": "article",
                    "guide_title": self.metadata.title,
                    "chapter_number": chapter.chapter_number,
                    "chapter_title": chapter.title,
                    **article.model_dump(exclude={"raw_html"}),
                }
                records.append(record)

        # Add glossary terms as individual records
        if self.glossary:
            for term in self.glossary.terms:
                record = {
                    "type": "glossary_term",
                    "guide_title": self.metadata.title,
                    **term.model_dump(),
                }
                records.append(record)

        return records
