"""Tests for dynamic guide structure discovery via #chapter-menu TOC parsing."""

from __future__ import annotations

import pytest

from graphrag_kg_pipeline.exceptions import ScraperError
from graphrag_kg_pipeline.parser import HTMLParser

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
