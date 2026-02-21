"""Tests for scraper: TOC discovery, OG image extraction, and thumbnail enrichment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest

from graphrag_kg_pipeline.exceptions import ScraperError
from graphrag_kg_pipeline.models.content import (
    Article,
    Chapter,
    ContentType,
    RequirementsManagementGuide,
    WebinarReference,
)
from graphrag_kg_pipeline.parser import HTMLParser
from graphrag_kg_pipeline.scraper import GuideScraper

if TYPE_CHECKING:
    from types import TracebackType

# Representative HTML fixture modeled on the live Jama guide's #chapter-menu.
# Includes: 3 chapters, a cross-chapter URL (Ch3 art1 → Ch2 path), overview
# articles (empty span.border), a glossary entry, and &nbsp; in a title.
CHAPTER_MENU_HTML = """\
<html><body>
<div id="chapter-menu" class="expand"><h6>Chapters</h6>
<div class="menu-container"><ul>
    <li id="requirements-management" class="expand ">
    <strong>1.</strong> Requirements Management
    </li><div id="requirements-management" class="expand-list" style="display:none;"><ul>
        <li id="requirements-management" class="">
            <span class="border"></span>
            <a href="https://www.jamasoftware.com/requirements-management-guide/requirements-management/">Overview</a>
        </li>
        <li id="what-is-requirements-management" class="">
            <span class="border">1</span>
            <a href="https://www.jamasoftware.com/requirements-management-guide/requirements-management/what-is-requirements-management/">\xa0What is Requirements Management?</a>
        </li>
        <li id="why-do-you-need-requirements-management" class="">
            <span class="border">2</span>
            <a href="https://www.jamasoftware.com/requirements-management-guide/requirements-management/why-do-you-need-requirements-management/"> Why do you need Requirements Management?</a>
        </li>
    </ul></div>
    <li id="writing-requirements" class="expand ">
    <strong>2.</strong> Writing Requirements
    </li><div id="writing-requirements" class="expand-list" style="display:none;"><ul>
        <li id="writing-requirements" class="">
            <span class="border"></span>
            <a href="https://www.jamasoftware.com/requirements-management-guide/writing-requirements/">Overview</a>
        </li>
        <li id="functional-requirements" class="">
            <span class="border">1</span>
            <a href="https://www.jamasoftware.com/requirements-management-guide/writing-requirements/functional-requirements/"> Functional requirements examples</a>
        </li>
    </ul></div>
    <li id="automotive-engineering" class="expand ">
    <strong>3.</strong> Automotive Development
    </li><div id="automotive-engineering" class="expand-list" style="display:none;"><ul>
        <li id="automotive-engineering" class="">
            <span class="border"></span>
            <a href="https://www.jamasoftware.com/requirements-management-guide/automotive-engineering/">Overview</a>
        </li>
        <li id="cross-chapter-article" class="">
            <span class="border">1</span>
            <a href="https://www.jamasoftware.com/requirements-management-guide/writing-requirements/cross-chapter-article/"> Cross-Chapter Article</a>
        </li>
    </ul></div>
    <li id="rm-glossary" class="glossary"><a href="https://www.jamasoftware.com/requirements-management-guide/rm-glossary/">Glossary</a></li>
</ul><div class="clear"></div></div></div>
</body></html>
"""


@pytest.fixture
def parser() -> HTMLParser:
    """Provide an HTMLParser instance."""
    return HTMLParser()


class TestParseChapterMenu:
    """Tests for HTMLParser.parse_chapter_menu()."""

    def test_structure(self, parser: HTMLParser) -> None:
        """Verify chapter count, article counts, and titles."""
        chapters = parser.parse_chapter_menu(CHAPTER_MENU_HTML)

        assert len(chapters) == 3

        ch1 = chapters[0]
        assert ch1.number == 1
        assert ch1.title == "Requirements Management"
        assert ch1.slug == "requirements-management"
        assert len(ch1.articles) == 3  # overview + 2 articles

        ch2 = chapters[1]
        assert ch2.number == 2
        assert ch2.title == "Writing Requirements"
        assert len(ch2.articles) == 2  # overview + 1 article

        ch3 = chapters[2]
        assert ch3.number == 3
        assert ch3.title == "Automotive Development"
        assert len(ch3.articles) == 2

    def test_article_urls_from_href(self, parser: HTMLParser) -> None:
        """Verify article URLs come directly from href, not slug construction."""
        chapters = parser.parse_chapter_menu(CHAPTER_MENU_HTML)
        ch1 = chapters[0]

        art1 = ch1.articles[1]  # "What is Requirements Management?"
        assert art1.url == (
            "https://www.jamasoftware.com/requirements-management-guide/"
            "requirements-management/what-is-requirements-management/"
        )

        # get_article_url should prefer the stored url
        assert ch1.get_article_url(art1) == art1.url

    def test_cross_chapter_url(self, parser: HTMLParser) -> None:
        """Verify a cross-chapter URL is captured verbatim from href.

        Chapter 3 (automotive-engineering) article 1 links to a
        writing-requirements path — the parser must not rewrite it.
        """
        chapters = parser.parse_chapter_menu(CHAPTER_MENU_HTML)
        ch3 = chapters[2]  # Automotive Development
        art1 = ch3.articles[1]

        assert art1.number == 1
        assert "writing-requirements" in art1.url
        assert "automotive-engineering" not in art1.url

    def test_overview_articles(self, parser: HTMLParser) -> None:
        """Verify overview articles (empty span.border) get number=0."""
        chapters = parser.parse_chapter_menu(CHAPTER_MENU_HTML)

        for chapter in chapters:
            overview = chapter.articles[0]
            assert overview.number == 0
            assert overview.title == "Overview"
            assert overview.url is not None

    def test_missing_div_raises(self, parser: HTMLParser) -> None:
        """Verify ScraperError when #chapter-menu is absent."""
        html = "<html><body><p>No menu here</p></body></html>"

        with pytest.raises(ScraperError, match="Could not find chapter menu"):
            parser.parse_chapter_menu(html)

    def test_skips_glossary(self, parser: HTMLParser) -> None:
        """Verify li.glossary is not included as a chapter."""
        chapters = parser.parse_chapter_menu(CHAPTER_MENU_HTML)

        chapter_titles = [ch.title for ch in chapters]
        assert "Glossary" not in chapter_titles
        assert all(ch.slug != "rm-glossary" for ch in chapters)

    def test_strips_nbsp(self, parser: HTMLParser) -> None:
        r"""Verify \xa0 (non-breaking space) in titles is cleaned."""
        chapters = parser.parse_chapter_menu(CHAPTER_MENU_HTML)
        ch1 = chapters[0]

        # The fixture has \xa0 before "What is Requirements Management?"
        art1 = ch1.articles[1]
        assert "\xa0" not in art1.title
        assert art1.title == "What is Requirements Management?"


# ---------------------------------------------------------------------------
# OG Image Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractOgImage:
    """Tests for HTMLParser.extract_og_image()."""

    def test_standard_og_image(self, parser: HTMLParser) -> None:
        """Extract URL from a valid og:image meta tag."""
        html = '<html><head><meta property="og:image" content="https://example.com/img.jpg"></head></html>'
        assert parser.extract_og_image(html) == "https://example.com/img.jpg"

    def test_missing_og_image(self, parser: HTMLParser) -> None:
        """Return None when no og:image tag is present."""
        html = "<html><head><title>No OG</title></head><body></body></html>"
        assert parser.extract_og_image(html) is None

    def test_empty_og_image_content(self, parser: HTMLParser) -> None:
        """Return None when the content attribute is empty."""
        html = '<html><head><meta property="og:image" content=""></head></html>'
        assert parser.extract_og_image(html) is None

    def test_whitespace_og_image_content(self, parser: HTMLParser) -> None:
        """Return None when the content attribute is whitespace-only."""
        html = '<html><head><meta property="og:image" content="   "></head></html>'
        assert parser.extract_og_image(html) is None

    def test_og_image_with_full_page(self, parser: HTMLParser) -> None:
        """Extract og:image from a realistic full-page HTML document."""
        html = """\
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Webinar | Jama Software</title>
            <meta property="og:type" content="website">
            <meta property="og:title" content="Requirements Webinar">
            <meta property="og:image" content="https://resources.jamasoftware.com/thumb.png">
            <meta property="og:url" content="https://resources.jamasoftware.com/webinar/example">
        </head>
        <body><h1>Webinar Page</h1></body>
        </html>
        """
        assert parser.extract_og_image(html) == "https://resources.jamasoftware.com/thumb.png"


# ---------------------------------------------------------------------------
# Webinar Thumbnail Enrichment Tests
# ---------------------------------------------------------------------------

WEBINAR_URL_A = "https://resources.jamasoftware.com/webinar/test-webinar-a"
WEBINAR_URL_B = "https://resources.jamasoftware.com/webinar/test-webinar-b"
OG_IMAGE_A = "https://resources.jamasoftware.com/og-a.jpg"
OG_IMAGE_B = "https://resources.jamasoftware.com/og-b.jpg"


class MockFetcher:
    """Minimal Fetcher protocol implementation for testing."""

    def __init__(self, responses: dict[str, str | None]) -> None:
        self.responses = responses
        self.fetch_count: dict[str, int] = {}

    async def fetch(self, url: str) -> str | None:
        """Return pre-configured HTML for the given URL."""
        self.fetch_count[url] = self.fetch_count.get(url, 0) + 1
        return self.responses.get(url)

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""


def _og_page(image_url: str) -> str:
    """Build a minimal HTML page with an og:image meta tag."""
    return f'<html><head><meta property="og:image" content="{image_url}"></head></html>'


def _make_guide(webinars_by_article: list[list[WebinarReference]]) -> RequirementsManagementGuide:
    """Build a minimal guide with articles containing the given webinars."""
    articles = []
    for i, webinars in enumerate(webinars_by_article):
        articles.append(
            Article(
                article_id=f"ch1-art{i}",
                chapter_number=1,
                article_number=i,
                title=f"Article {i}",
                url=f"https://example.com/art{i}",
                content_type=ContentType.ARTICLE,
                markdown_content="Content.",
                webinars=webinars,
            )
        )
    chapter = Chapter(
        chapter_number=1,
        title="Test Chapter",
        overview_url="https://example.com/overview",
        articles=articles,
    )
    return RequirementsManagementGuide(chapters=[chapter])


class TestEnrichWebinarThumbnails:
    """Tests for GuideScraper._enrich_webinar_thumbnails()."""

    @pytest.mark.asyncio
    async def test_enriches_null_thumbnails(self) -> None:
        """Propagate OG image to webinar refs with null thumbnail."""
        webinar = WebinarReference(url=WEBINAR_URL_A, title="Webinar A")
        guide = _make_guide([[webinar]])

        fetcher = MockFetcher({WEBINAR_URL_A: _og_page(OG_IMAGE_A)})
        scraper = GuideScraper()
        await scraper._enrich_webinar_thumbnails(guide, fetcher)

        assert webinar.thumbnail_url == OG_IMAGE_A

    @pytest.mark.asyncio
    async def test_deduplicates_urls(self) -> None:
        """Same URL in multiple articles triggers a single fetch."""
        w1 = WebinarReference(url=WEBINAR_URL_A, title="Webinar A copy 1")
        w2 = WebinarReference(url=WEBINAR_URL_A, title="Webinar A copy 2")
        guide = _make_guide([[w1], [w2]])

        fetcher = MockFetcher({WEBINAR_URL_A: _og_page(OG_IMAGE_A)})
        scraper = GuideScraper()
        await scraper._enrich_webinar_thumbnails(guide, fetcher)

        assert w1.thumbnail_url == OG_IMAGE_A
        assert w2.thumbnail_url == OG_IMAGE_A
        assert fetcher.fetch_count[WEBINAR_URL_A] == 1

    @pytest.mark.asyncio
    async def test_skips_existing_thumbnails(self) -> None:
        """Webinars with existing thumbnails are not re-fetched."""
        existing = WebinarReference(
            url=WEBINAR_URL_A, title="Already has thumb", thumbnail_url="https://existing.jpg"
        )
        guide = _make_guide([[existing]])

        fetcher = MockFetcher({WEBINAR_URL_A: _og_page(OG_IMAGE_A)})
        scraper = GuideScraper()
        await scraper._enrich_webinar_thumbnails(guide, fetcher)

        assert existing.thumbnail_url == "https://existing.jpg"
        assert WEBINAR_URL_A not in fetcher.fetch_count

    @pytest.mark.asyncio
    async def test_handles_fetch_failure(self) -> None:
        """Fetcher returning None does not crash; thumbnail stays null."""
        webinar = WebinarReference(url=WEBINAR_URL_B, title="Webinar B")
        guide = _make_guide([[webinar]])

        fetcher = MockFetcher({WEBINAR_URL_B: None})
        scraper = GuideScraper()
        await scraper._enrich_webinar_thumbnails(guide, fetcher)

        assert webinar.thumbnail_url is None
