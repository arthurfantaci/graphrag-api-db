"""Article index builder for the guide ETL pipeline.

This module provides utilities for building an article index that
maps article IDs to their metadata, enabling efficient lookup
during pipeline processing.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from graphrag_kg_pipeline.models import RequirementsManagementGuide

logger = structlog.get_logger(__name__)


@dataclass
class ArticleIndex:
    """Index of articles for efficient lookup.

    Provides multiple access patterns:
    - By article_id (primary key)
    - By chapter number
    - By URL

    Attributes:
        by_id: Mapping from article_id to article data.
        by_chapter: Mapping from chapter_number to list of article_ids.
        by_url: Mapping from URL to article_id.
        total_articles: Total number of indexed articles.
    """

    by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_chapter: dict[int, list[str]] = field(default_factory=dict)
    by_url: dict[str, str] = field(default_factory=dict)

    @property
    def total_articles(self) -> int:
        """Total number of articles in the index."""
        return len(self.by_id)

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        """Get article by ID.

        Args:
            article_id: Article identifier.

        Returns:
            Article data dict or None if not found.
        """
        return self.by_id.get(article_id)

    def get_chapter_articles(self, chapter_number: int) -> list[dict[str, Any]]:
        """Get all articles in a chapter.

        Args:
            chapter_number: Chapter number (1-15).

        Returns:
            List of article data dicts.
        """
        article_ids = self.by_chapter.get(chapter_number, [])
        return [self.by_id[aid] for aid in article_ids if aid in self.by_id]

    def get_article_by_url(self, url: str) -> dict[str, Any] | None:
        """Get article by URL.

        Args:
            url: Article URL.

        Returns:
            Article data dict or None if not found.
        """
        article_id = self.by_url.get(url)
        if article_id:
            return self.by_id.get(article_id)
        return None

    def article_ids(self) -> list[str]:
        """Get all article IDs.

        Returns:
            List of all article IDs.
        """
        return list(self.by_id.keys())

    def __contains__(self, article_id: str) -> bool:
        """Check if article exists in index.

        Args:
            article_id: Article identifier.

        Returns:
            True if article exists.
        """
        return article_id in self.by_id


def build_article_index(
    guide: "RequirementsManagementGuide",
    include_content: bool = True,
) -> ArticleIndex:
    """Build article index from scraped guide.

    Creates an ArticleIndex with multiple access patterns for
    efficient lookup during pipeline processing.

    Args:
        guide: The scraped RequirementsManagementGuide.
        include_content: Whether to include markdown_content in index.

    Returns:
        ArticleIndex with all articles indexed.

    Example:
        >>> guide = await scraper.scrape_all()
        >>> index = build_article_index(guide)
        >>> article = index.get_article("ch1-art3")
        >>> print(article["title"])
    """
    index = ArticleIndex()

    for chapter in guide.chapters:
        chapter_articles = []

        for article in chapter.articles:
            article_data = {
                "article_id": article.article_id,
                "chapter_number": chapter.chapter_number,
                "chapter_title": chapter.title,
                "article_number": article.article_number,
                "title": article.title,
                "url": article.url,
                "content_type": article.content_type.value,
                "word_count": article.word_count,
                "char_count": article.char_count,
                "section_count": len(article.sections),
                "image_count": len(article.images),
                "video_count": len(article.videos),
                "webinar_count": len(article.webinars),
                "cross_reference_count": len(article.cross_references),
            }

            if include_content:
                article_data["markdown_content"] = article.markdown_content
                article_data["sections"] = [
                    {
                        "heading": s.heading,
                        "level": s.level,
                        "content": s.content,
                    }
                    for s in article.sections
                ]

            # Index by ID
            index.by_id[article.article_id] = article_data

            # Index by URL
            index.by_url[article.url] = article.article_id

            # Track for chapter index
            chapter_articles.append(article.article_id)

        # Index by chapter
        index.by_chapter[chapter.chapter_number] = chapter_articles

    logger.info(
        "Built article index",
        total_articles=index.total_articles,
        chapters=len(index.by_chapter),
    )

    return index
