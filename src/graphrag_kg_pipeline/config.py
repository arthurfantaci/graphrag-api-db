"""Configuration and URL mappings for the Jama Requirements Management Guide.

All URLs extracted from the guide's table of contents.
"""

from dataclasses import dataclass, field

BASE_URL = "https://www.jamasoftware.com/requirements-management-guide"


@dataclass
class ArticleConfig:
    """Configuration for a single article."""

    number: int  # 0 for overview
    title: str
    slug: str  # URL path segment
    url: str | None = None  # Full URL from TOC; used instead of slug-based construction


@dataclass
class ChapterConfig:
    """Configuration for a chapter."""

    number: int
    title: str
    slug: str
    articles: list[ArticleConfig] = field(default_factory=list)

    @property
    def overview_url(self) -> str:
        """Return the URL for this chapter's overview page."""
        return f"{BASE_URL}/{self.slug}/"

    def get_article_url(self, article: ArticleConfig) -> str:
        """Return the full URL for an article within this chapter."""
        if article.url:
            return article.url
        return self.overview_url


GLOSSARY_URL = f"{BASE_URL}/rm-glossary/"

# Rate limiting configuration
RATE_LIMIT_DELAY_SECONDS = 1.0  # Delay between requests to be respectful
MAX_CONCURRENT_REQUESTS = 3  # Max parallel requests
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 3
