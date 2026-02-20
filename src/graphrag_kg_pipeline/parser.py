"""HTML parsing utilities for extracting and converting Jama guide content.

Converts HTML to clean Markdown and extracts metadata for RAG/knowledge graph use.
"""

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .config import BASE_URL, ArticleConfig, ChapterConfig
from .exceptions import ScraperError
from .models.content import (
    CrossReference,
    ImageReference,
    RelatedArticle,
    Section,
    VideoReference,
    WebinarReference,
)

# Constants for content extraction
MIN_CONCEPT_LENGTH = 2
MAX_CONCEPT_LENGTH = 50
MAX_KEY_CONCEPTS = 20
MIN_CONTENT_ELEMENTS = 2
MIN_TITLE_LENGTH = 10

# Tags to remove during HTML cleaning (contain no meaningful content)
# Note: iframe is NOT included - YouTube embeds are valuable content
TAGS_TO_REMOVE = frozenset(
    [
        "style",
        "script",
        "noscript",
        "svg",
        "object",
        "embed",
        "canvas",
        "map",
        "audio",
        "video",
        "source",
        "track",
        "template",
    ]
)

# Patterns for extracting video information
YOUTUBE_PATTERNS = [
    re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]+)"),
    re.compile(r"youtube\.com/watch\?v=([a-zA-Z0-9_-]+)"),
    re.compile(r"youtu\.be/([a-zA-Z0-9_-]+)"),
]

# CSS class patterns indicating promotional/CTA content (not guide content)
# Note: avia-button-no means "no button" - we only want actual button elements
PROMO_CLASS_PATTERNS = [
    r"avia-buttonrow",  # CTA button rows
    r"avia-button(?!-no)",  # CTA buttons, but NOT avia-button-no
]

# Link href patterns indicating promotional content
PROMO_LINK_PATTERNS = re.compile(
    r"(/trial/|/demo/|/pricing/|/contact/|/request/|#form)",
    re.IGNORECASE,
)

# Text patterns indicating promotional content (case-insensitive)
PROMO_TEXT_PATTERNS = re.compile(
    r"(free\s+\d+-day\s+trial|book\s+a\s+demo|request\s+a\s+demo|"
    r"ready\s+to\s+find\s+out\s+more|get\s+started\s+today|"
    r"schedule\s+a\s+demo|contact\s+us\s+today|learn\s+more\s+about\s+jama)",
    re.IGNORECASE,
)


class HTMLParser:
    """Parser for Jama guide HTML content."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        """Initialize the parser with the base URL for resolving relative links."""
        self.base_url = base_url

    def parse_article(self, html: str, source_url: str) -> dict:
        """Parse an article page and extract all content.

        Returns a dict with:
        - title: Article title
        - markdown_content: Full content as markdown
        - sections: List of Section objects
        - cross_references: List of CrossReference objects
        - key_concepts: Extracted key terms
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title = self._extract_title(soup)

        # Find main content area
        content_elem = self._find_content_element(soup)

        if not content_elem:
            return {
                "title": title,
                "markdown_content": "",
                "sections": [],
                "cross_references": [],
                "key_concepts": [],
                "images": [],
                "videos": [],
                "webinars": [],
                "related_articles": [],
            }

        # Clean HTML: remove style, script, and other non-content elements
        self._clean_html(content_elem)

        # Extract cross-references before converting to markdown
        cross_refs = self._extract_cross_references(content_elem, source_url)

        # Extract images, videos, webinars, and related articles
        images = self._extract_images(content_elem, source_url)
        videos = self._extract_videos(content_elem, source_url)
        webinars = self._extract_webinars(content_elem, source_url)
        related_articles = self._extract_related_articles(content_elem, source_url)

        # Inject webinar links into content before markdown conversion
        # This ensures webinar URLs appear inline where they belong
        self._inject_webinar_links(content_elem, webinars)

        # Convert to markdown (includes images and videos)
        markdown = self._html_to_markdown(content_elem, include_images=True)

        # Remove promotional text blocks from markdown
        markdown = self._remove_promo_text(markdown)

        # Parse into sections
        sections = self._parse_sections(content_elem, source_url)

        # Extract key concepts
        key_concepts = self._extract_key_concepts(content_elem, title)

        return {
            "title": title,
            "markdown_content": markdown,
            "sections": sections,
            "cross_references": cross_refs,
            "key_concepts": key_concepts,
            "images": images,
            "videos": videos,
            "webinars": webinars,
            "related_articles": related_articles,
        }

    def parse_glossary(self, html: str, _source_url: str) -> list[dict]:
        """Parse the glossary page and extract all terms.

        Args:
            html: Raw HTML content of the glossary page.
            _source_url: Source URL (unused, kept for API consistency).

        Returns:
            A list of dicts with 'term' and 'definition' keys.
        """
        soup = BeautifulSoup(html, "lxml")
        content_elem = self._find_content_element(soup)

        if not content_elem:
            return []

        # Clean HTML before processing
        self._clean_html(content_elem)

        terms = []

        # Glossaries can use tables, dt/dd, h3/p, or strong/text patterns.
        # Try strategies in order of reliability (most structured first).

        # Strategy 1: Table-based glossary (ACRONYM | TERM | DEFINITION columns)
        # Tables with explicit column headers are the most reliable format.
        for table in content_elem.find_all("table"):
            # Check if this looks like a glossary table
            header_row = table.find("tr", class_=re.compile(r"heading", re.IGNORECASE))
            if not header_row:
                header_row = table.find("tr")  # First row may be header

            if not header_row:
                continue

            # Get column headers
            headers = [th.get_text(strip=True).upper() for th in header_row.find_all(["th", "td"])]

            # Check for expected columns
            if "TERM" not in headers or "DEFINITION" not in headers:
                continue

            # Find column indices
            acronym_idx = headers.index("ACRONYM") if "ACRONYM" in headers else None
            term_idx = headers.index("TERM")
            definition_idx = headers.index("DEFINITION")

            # Parse data rows (skip header)
            for tr in table.find_all("tr")[1:]:  # Skip header row
                cells = tr.find_all("td")
                if len(cells) < max(term_idx, definition_idx) + 1:
                    continue

                term = cells[term_idx].get_text(strip=True)
                definition = cells[definition_idx].get_text(strip=True)
                acronym = (
                    cells[acronym_idx].get_text(strip=True)
                    if acronym_idx is not None and len(cells) > acronym_idx
                    else None
                )

                if term and definition:
                    terms.append(
                        {
                            "term": term,
                            "acronym": acronym if acronym else None,
                            "definition": definition,
                        }
                    )

        # Strategy 2: Definition lists
        if not terms:
            for dl in content_elem.find_all("dl"):
                dts = dl.find_all("dt")
                dds = dl.find_all("dd")
                for dt, dd in zip(dts, dds, strict=False):
                    terms.append(
                        {
                            "term": dt.get_text(strip=True),
                            "definition": dd.get_text(strip=True),
                        }
                    )

        # Strategy 3: Headings followed by paragraphs
        if not terms:
            current_term = None
            for elem in content_elem.find_all(["h2", "h3", "h4", "p"]):
                if elem.name in ["h2", "h3", "h4"]:
                    current_term = elem.get_text(strip=True)
                elif elem.name == "p" and current_term:
                    definition = elem.get_text(strip=True)
                    if definition:
                        terms.append(
                            {
                                "term": current_term,
                                "definition": definition,
                            }
                        )
                        current_term = None

        # Strategy 4: Strong tags for terms
        if not terms:
            for p in content_elem.find_all("p"):
                strong = p.find("strong")
                if strong:
                    term = strong.get_text(strip=True)
                    # Get text after the strong tag
                    definition_parts = []
                    for sibling in strong.next_siblings:
                        if isinstance(sibling, NavigableString):
                            definition_parts.append(str(sibling))
                        elif isinstance(sibling, Tag):
                            definition_parts.append(sibling.get_text())
                    definition = " ".join(definition_parts).strip()
                    definition = re.sub(
                        r"^[:\s\-]+", "", definition
                    )  # Remove leading colons/dashes
                    if term and definition:
                        terms.append(
                            {
                                "term": term,
                                "definition": definition,
                            }
                        )

        return terms

    def parse_chapter_menu(self, html: str) -> list[ChapterConfig]:
        """Parse the #chapter-menu TOC to discover all chapters and articles.

        Extracts the complete chapter/article structure from the guide's main
        page by parsing the ``div#chapter-menu`` element. Each chapter header
        is an ``li.expand`` with a ``<strong>N.</strong>`` number, and its
        articles are in the adjacent ``div.expand-list``. Glossary entries
        (``li.glossary``) are skipped.

        Args:
            html: Raw HTML of the guide's main page.

        Returns:
            List of fully-populated ChapterConfig objects.

        Raises:
            ScraperError: If ``#chapter-menu`` is not found in the HTML.
        """
        # Use html.parser (not lxml) because the source HTML has <div> elements
        # inside <ul>, which is invalid. lxml restructures the tree and breaks
        # sibling relationships; html.parser preserves the original nesting.
        soup = BeautifulSoup(html, "html.parser")

        menu = soup.select_one("div#chapter-menu")
        if not menu:
            msg = "Could not find chapter menu (#chapter-menu) on guide page"
            raise ScraperError(msg)

        chapters: list[ChapterConfig] = []

        for chapter_li in menu.select("li.expand"):
            strong = chapter_li.find("strong")
            if not strong:
                continue

            number_text = strong.get_text(strip=True).rstrip(".")
            try:
                chapter_number = int(number_text)
            except ValueError:
                continue

            # Title is the li's direct text content, excluding the <strong> tag
            title_parts = []
            for child in chapter_li.children:
                if isinstance(child, NavigableString):
                    text = str(child).strip()
                    if text:
                        title_parts.append(text)
            chapter_title = " ".join(title_parts).replace("\xa0", " ").strip()

            chapter_slug = chapter_li.get("id", "")

            # Find the article list div sharing the same id
            expand_list = menu.find("div", class_="expand-list", id=chapter_slug)
            if not expand_list:
                expand_list = chapter_li.find_next("div", class_="expand-list")

            articles: list[ArticleConfig] = []
            if expand_list:
                for article_li in expand_list.select("li"):
                    if "glossary" in (article_li.get("class") or []):
                        continue

                    link = article_li.find("a", href=True)
                    if not link:
                        continue

                    border_span = article_li.select_one("span.border")
                    article_number = 0
                    if border_span:
                        border_text = border_span.get_text(strip=True)
                        article_number = int(border_text) if border_text else 0

                    article_url = link["href"]
                    article_title = link.get_text(strip=True).replace("\xa0", " ").strip()
                    article_slug = article_li.get("id", "")

                    articles.append(
                        ArticleConfig(
                            number=article_number,
                            title=article_title,
                            slug=article_slug,
                            url=article_url,
                        )
                    )

            chapters.append(
                ChapterConfig(
                    number=chapter_number,
                    title=chapter_title,
                    slug=chapter_slug,
                    articles=articles,
                )
            )

        return chapters

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title."""
        # Try h1 first
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        # Fall back to title tag
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Remove common suffixes
            for suffix in [" | Jama Software", " - Jama Software"]:
                if title.endswith(suffix):
                    title = title[: -len(suffix)]
            return title

        return "Untitled"

    def _find_content_element(self, soup: BeautifulSoup) -> Tag | None:
        """Find the main content container.

        Targets the flex_cell_inner div which contains article content,
        selectively excluding final sections that contain CTA content.
        """
        # Jama's site uses Enfold/Avia theme with flex_cell layout
        # The second flex_cell_inner contains the main article content
        flex_cell_inners = soup.select(".flex_cell_inner")
        if len(flex_cell_inners) >= MIN_CONTENT_ELEMENTS:
            content_elem = flex_cell_inners[1]

            # Only remove final section if it contains CTA/promotional content
            sections = content_elem.find_all("section", recursive=False)
            if sections and self._is_cta_section(sections[-1]):
                sections[-1].decompose()

            return content_elem

        # Fallback: try original flex_cell approach
        flex_cells = soup.select(".flex_cell")
        if len(flex_cells) >= MIN_CONTENT_ELEMENTS:
            return flex_cells[1]

        # Fallback: common content selectors
        selectors = [
            "article",
            ".post-content",
            ".entry-content",
            ".content-area",
            "main",
            "#main-content",
            ".main-content",
        ]

        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem

        # Fall back to body
        return soup.find("body")

    def _is_cta_section(self, section: Tag) -> bool:
        """Check if a section contains CTA/promotional content.

        Args:
            section: BeautifulSoup Tag to check.

        Returns:
            True if the section appears to contain CTA content.
        """
        # Preserve sections with valuable cross-reference content
        text = section.get_text(strip=True)
        if "RELATED ARTICLE" in text.upper():
            return False  # Keep this section - it has cross-references

        # Check for CTA button classes (not avia-button-no which means "no button")
        if section.find(class_=re.compile(r"avia-button(?!-no)", re.IGNORECASE)):
            return True

        # Check for CTA link patterns (but not blog links which are informational)
        for link in section.find_all("a", href=True):
            href = str(link.get("href", ""))
            # Skip blog links - they're informational, not CTAs
            if "/blog/" in href:
                continue
            if PROMO_LINK_PATTERNS.search(href):
                return True

        # Check text content for CTA patterns
        if PROMO_TEXT_PATTERNS.search(text):
            return True

        # Check for "Ready to Find Out More" or similar headings
        for heading in section.find_all(["h2", "h3", "h4"]):
            heading_text = heading.get_text(strip=True).lower()
            if any(
                pattern in heading_text
                for pattern in ["find out more", "book a demo", "get started"]
            ):
                return True

        return False

    def _clean_html(self, elem: Tag) -> None:
        """Remove non-content elements from HTML in place.

        Removes style tags, scripts, promotional content, and other elements
        that don't contain meaningful article content. This prevents CSS/JS
        and marketing CTAs from leaking into the markdown output.

        Args:
            elem: BeautifulSoup Tag to clean (modified in place).
        """
        # Remove non-content tags entirely
        for tag in elem.find_all(TAGS_TO_REMOVE):
            tag.decompose()

        # Remove HTML comments
        for comment in elem.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove elements with hidden display (often used for CSS-in-JS)
        hidden_pattern = re.compile(r"display:\s*none", re.IGNORECASE)
        for hidden in elem.find_all(style=hidden_pattern):
            hidden.decompose()

        # Remove promotional/CTA elements by class patterns
        self._remove_promotional_content(elem)

    def _remove_promotional_content(self, elem: Tag) -> None:
        """Remove promotional and CTA content from HTML.

        Identifies and removes marketing blocks like trial buttons, demo CTAs,
        and "Ready to find out more?" sections that aren't part of the guide.

        Args:
            elem: BeautifulSoup Tag to clean (modified in place).
        """
        # Remove elements with promotional CSS classes (CTA buttons)
        for pattern in PROMO_CLASS_PATTERNS:
            for tag in elem.find_all(class_=re.compile(pattern, re.IGNORECASE)):
                tag.decompose()

        # Remove specific promotional link buttons (not regular article links)
        for link in elem.find_all("a", href=PROMO_LINK_PATTERNS):
            # Only remove if it's a CTA button (has button-like classes)
            link_classes = link.get("class", [])
            class_str = " ".join(link_classes) if link_classes else ""
            if "button" in class_str.lower() or "cta" in class_str.lower():
                link.decompose()

    def _remove_promo_text(self, markdown: str) -> str:
        """Remove promotional text blocks from markdown content.

        Removes specific CTA sections like "Ready to Find Out More?" and
        promotional paragraphs that aren't part of the guide content.

        Args:
            markdown: Markdown content to clean.

        Returns:
            Cleaned markdown with promotional text removed.
        """
        lines = markdown.split("\n")
        cleaned_lines = []
        skip_until_next_section = False

        for line in lines:
            # Check if this line starts a promotional section
            if re.match(r"^#{1,4}\s*Ready to Find Out More", line, re.IGNORECASE):
                skip_until_next_section = True
                continue

            if re.match(r"^#{1,4}\s*Book a Demo", line, re.IGNORECASE):
                skip_until_next_section = True
                continue

            # Stop skipping when we hit a new non-promo heading
            is_heading = re.match(r"^#{1,4}\s+", line)
            is_promo_heading = re.search(r"(demo|trial|contact|pricing)", line, re.IGNORECASE)
            if skip_until_next_section and is_heading and not is_promo_heading:
                skip_until_next_section = False

            if skip_until_next_section:
                continue

            # Skip individual promotional lines
            if PROMO_TEXT_PATTERNS.search(line):
                continue

            cleaned_lines.append(line)

        # Clean up multiple blank lines
        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _extract_cross_references(self, elem: Tag, source_url: str) -> list[CrossReference]:
        """Extract all links as cross-references.

        Handles both text links and image links (links wrapping images).
        For image links, uses the image's title or alt text as the link text.
        """
        refs = []
        source_domain = urlparse(source_url).netloc

        for link in elem.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)

            # If no text, check if this is an image link
            if not text:
                img = link.find("img")
                if img:
                    # Use image title or alt as link text
                    text = img.get("title", "") or img.get("alt", "") or ""
                    text = text.strip()

            if not text or not href:
                continue

            # Skip anchor links and javascript
            if href.startswith(("#", "javascript:")):
                continue

            # Normalize URL
            full_url = urljoin(source_url, href)
            parsed = urlparse(full_url)

            # Check if internal
            is_internal = parsed.netloc == source_domain or "jamasoftware.com" in parsed.netloc

            # Try to extract section ID for internal links
            section_id = None
            if is_internal and "/requirements-management-guide/" in full_url:
                match = re.search(r"/requirements-management-guide/([^/]+)/([^/]+)?", full_url)
                if match:
                    chapter_slug = match.group(1)
                    article_slug = match.group(2) or "overview"
                    section_id = f"{chapter_slug}/{article_slug}"

            refs.append(
                CrossReference(
                    text=text,
                    url=full_url,
                    is_internal=is_internal,
                    target_section_id=section_id,
                )
            )

        return refs

    def _extract_images(self, elem: Tag, source_url: str) -> list[ImageReference]:
        """Extract all images from the content as ImageReference objects."""
        images = []
        seen_urls = set()

        for img in elem.find_all("img"):
            # Get the actual image URL (skip lazy-loading placeholders)
            src = img.get("src", "")

            # Skip data: URLs (lazy loading placeholders)
            if src.startswith("data:"):
                # Try to get from data-src or parent noscript
                src = img.get("data-src", "") or img.get("data-lazy-src", "")

            # Also check noscript siblings for actual URL
            if not src or src.startswith("data:"):
                noscript = img.find_next_sibling("noscript")
                if noscript:
                    noscript_img = BeautifulSoup(str(noscript), "lxml").find("img")
                    if noscript_img:
                        src = noscript_img.get("src", "")

            # Skip if no valid URL or already seen
            if not src or src.startswith("data:") or src in seen_urls:
                continue

            seen_urls.add(src)

            # Normalize URL
            full_url = urljoin(source_url, src)

            # Get alt text and title
            alt_text = img.get("alt", "")
            title = img.get("title") or None

            # Look for figcaption
            caption = None
            figure = img.find_parent("figure")
            if figure:
                figcaption = figure.find("figcaption")
                if figcaption:
                    caption = figcaption.get_text(strip=True)

            # Get surrounding context (previous heading or paragraph)
            context = None
            prev_heading = img.find_previous(["h2", "h3", "h4"])
            if prev_heading:
                context = prev_heading.get_text(strip=True)

            images.append(
                ImageReference(
                    url=full_url,
                    alt_text=alt_text,
                    title=title,
                    caption=caption,
                    context=context,
                )
            )

        return images

    def _extract_videos(self, elem: Tag, _source_url: str) -> list[VideoReference]:
        """Extract all videos from iframes as VideoReference objects.

        Args:
            elem: BeautifulSoup Tag containing the content.
            _source_url: Source URL (unused, kept for API consistency).

        Returns:
            List of VideoReference objects for embedded videos.
        """
        videos = []
        seen_ids = set()

        for iframe in elem.find_all("iframe"):
            src = iframe.get("src", "") or iframe.get("data-src", "")
            if not src:
                continue

            # Extract YouTube video ID
            video_id = None
            for pattern in YOUTUBE_PATTERNS:
                match = pattern.search(src)
                if match:
                    video_id = match.group(1)
                    break

            if not video_id or video_id in seen_ids:
                continue

            seen_ids.add(video_id)

            # Build URLs
            embed_url = f"https://www.youtube.com/embed/{video_id}"
            watch_url = f"https://www.youtube.com/watch?v={video_id}"

            # Get title from iframe if available
            title = iframe.get("title") or None

            # Get surrounding context
            context = None
            prev_heading = iframe.find_previous(["h2", "h3", "h4"])
            if prev_heading:
                context = prev_heading.get_text(strip=True)

            videos.append(
                VideoReference(
                    url=watch_url,
                    embed_url=embed_url,
                    video_id=video_id,
                    platform="youtube",
                    title=title,
                    context=context,
                )
            )

        return videos

    def _inject_webinar_links(self, elem: Tag, webinars: list[WebinarReference]) -> None:
        """Inject webinar links into the HTML before markdown conversion.

        This ensures webinar URLs appear inline in the content where they belong,
        not just in a separate extracted list. Handles two patterns:

        1. Image links: `<a href="webinar"><img/></a>` -> replaced with text link
        2. Description headings: "In This Webinar..." -> wrapped with webinar link

        Args:
            elem: BeautifulSoup Tag to modify in-place.
            webinars: List of extracted WebinarReference objects.
        """
        if not webinars:
            return

        # Build URL -> webinar mapping for quick lookup
        webinar_map = {w.url: w for w in webinars}

        # Pattern 1: Replace image links with text links
        # Find all avia-image-container divs with webinar links
        container_pattern = re.compile(r"avia-image-container")
        image_containers = elem.find_all("div", class_=container_pattern)
        for container in image_containers:
            link = container.find("a", href=True)
            if not link:
                continue

            href = link.get("href", "")
            # Check if this links to a webinar
            if "/webinar/" not in href:
                continue

            # Find the webinar reference
            webinar = None
            for url, w in webinar_map.items():
                if href in url or url in href:
                    webinar = w
                    break

            if webinar:
                # Create a markdown-style text link to replace the image container
                link_text = f"**📹 [{webinar.title}]({webinar.url})**"
                # Replace the container with our text
                new_tag = NavigableString(f"\n\n{link_text}\n\n")
                container.replace_with(new_tag)

        # Pattern 2: Make "In This Webinar" headings clickable
        # Find all text containing "In This Webinar"
        in_this_texts = elem.find_all(string=re.compile(r"In This Webinar", re.IGNORECASE))
        for text in in_this_texts:
            # Get parent heading element
            parent = text.find_parent(["h2", "h3", "h4", "h5", "h6", "p", "div"])
            if not parent:
                continue

            # Find the associated webinar URL from sibling column
            desc_container = text.find_parent("div", class_=re.compile(r"flex_column"))
            if not desc_container:
                continue

            # Look in previous sibling for the webinar link
            prev_sib = desc_container.find_previous_sibling(
                "div", class_=re.compile(r"flex_column")
            )
            if not prev_sib:
                continue

            webinar_link = prev_sib.find("a", href=re.compile(r"/webinar/", re.IGNORECASE))
            if not webinar_link:
                continue

            href = webinar_link.get("href", "")

            # Find the webinar reference
            webinar = None
            for url, w in webinar_map.items():
                if href in url or url in href:
                    webinar = w
                    break

            if webinar:
                # Replace the heading text with a linked version
                description = parent.get_text(strip=True)
                link_md = f"**[{description}]({webinar.url})**"
                new_tag = NavigableString(f"\n\n{link_md}\n\n")
                parent.replace_with(new_tag)

    def _extract_webinars(self, elem: Tag, source_url: str) -> list[WebinarReference]:
        """Extract webinar links from the content.

        Identifies links to Jama's webinar resources including:
        - Featured webinar sections with "In This Webinar" descriptions
        - Image thumbnails linking to webinars
        - Text links to webinars

        Args:
            elem: BeautifulSoup Tag containing the content.
            source_url: Source URL for resolving relative links.

        Returns:
            List of WebinarReference objects for webinar links found.
        """
        webinars = []
        seen_urls: set[str] = set()

        # Pattern to identify webinar URLs
        webinar_pattern = re.compile(
            r"(resources\.jamasoftware\.com/webinar/|jamasoftware\.com/webinar/)",
            re.IGNORECASE,
        )

        # First pass: Find "In This Webinar" descriptions
        webinar_descriptions = self._find_webinar_descriptions(elem, source_url, webinar_pattern)

        # Second pass: Extract all webinar links
        for link in elem.find_all("a", href=True):
            href = link["href"]
            if not webinar_pattern.search(href):
                continue

            full_url = urljoin(source_url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            webinar = self._create_webinar_reference(
                link, full_url, href, source_url, webinar_descriptions
            )
            webinars.append(webinar)

        return webinars

    def _find_webinar_descriptions(
        self, elem: Tag, source_url: str, webinar_pattern: re.Pattern[str]
    ) -> dict[str, str]:
        """Find "In This Webinar" descriptions and map them to URLs.

        These are in two-column layouts: av_three_fifth (image) + av_two_fifth (desc).
        """
        descriptions: dict[str, str] = {}

        in_this_texts = elem.find_all(string=re.compile(r"In This Webinar", re.IGNORECASE))
        for text in in_this_texts:
            desc_container = text.find_parent("div", class_=re.compile(r"flex_column"))
            if not desc_container:
                continue

            # Get description from parent element
            description = text.strip()
            parent_tag = text.find_parent(["h3", "h4", "p", "div"])
            if parent_tag:
                description = parent_tag.get_text(strip=True)

            # Find webinar link in previous sibling column
            prev_sib = desc_container.find_previous_sibling(
                "div", class_=re.compile(r"flex_column")
            )
            if prev_sib:
                webinar_link = prev_sib.find("a", href=webinar_pattern)
                if webinar_link:
                    href = webinar_link.get("href", "")
                    full_url = urljoin(source_url, href)
                    descriptions[full_url] = description

        return descriptions

    def _create_webinar_reference(
        self,
        link: Tag,
        full_url: str,
        href: str,
        source_url: str,
        descriptions: dict[str, str],
    ) -> WebinarReference:
        """Create a WebinarReference from a link element."""
        text = link.get_text(strip=True)
        thumbnail_url = None

        # Handle image links
        img = link.find("img")
        if img:
            if not text:
                # Only accept img title/alt if substantive (>10 chars)
                img_title = (img.get("title", "") or "").strip()
                img_alt = (img.get("alt", "") or "").strip()
                candidate = img_title or img_alt
                if len(candidate) > MIN_TITLE_LENGTH:
                    text = candidate

            # Get thumbnail (prefer actual URL over lazy-load placeholder)
            thumb = img.get("data-lazy-src") or img.get("data-src") or img.get("src")
            if thumb and not thumb.startswith("data:"):
                thumbnail_url = urljoin(source_url, thumb)

        # Extract title from URL slug if no text
        if not text:
            match = re.search(r"/webinar/([^/?#]+)", href)
            if match:
                text = "Webinar: " + match.group(1).replace("-", " ").title()

        # Fallback: check nearest sibling <p> text
        if not text:
            next_p = link.find_next_sibling("p")
            if next_p:
                p_text = next_p.get_text(strip=True)
                if len(p_text) > MIN_TITLE_LENGTH:
                    # Use first sentence as title
                    text = p_text.split(".")[0].strip()

        # Last resort
        if not text:
            text = "Webinar"

        # Get context (nearest heading, but not "In This Webinar")
        context = None
        prev_heading = link.find_previous(["h2", "h3", "h4"])
        if prev_heading:
            heading_text = prev_heading.get_text(strip=True)
            if "In This Webinar" not in heading_text:
                context = heading_text

        return WebinarReference(
            url=full_url,
            title=text,
            description=descriptions.get(full_url),
            thumbnail_url=thumbnail_url,
            context=context,
        )

    def _extract_related_articles(self, elem: Tag, source_url: str) -> list[RelatedArticle]:
        """Extract explicitly marked related article callouts.

        Identifies "RELATED ARTICLE:" patterns in av_promobox elements,
        which are editorial callouts to related content.

        Args:
            elem: BeautifulSoup Tag containing the content.
            source_url: Source URL for resolving relative links.

        Returns:
            List of RelatedArticle objects for related article callouts found.
        """
        related = []
        seen_urls: set[str] = set()

        # Find all promobox elements (these contain RELATED ARTICLE callouts)
        promoboxes = elem.find_all("div", class_=re.compile(r"av_promobox"))

        for box in promoboxes:
            text = box.get_text(strip=True)

            # Check if this is a RELATED ARTICLE callout
            if "RELATED ARTICLE" not in text.upper():
                continue

            # Find the link
            link = box.find("a", href=True)
            if not link:
                continue

            href = link.get("href", "")
            if not href:
                continue

            # Normalize URL
            full_url = urljoin(source_url, href)

            # Skip duplicates
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Get link text (article title)
            title = link.get_text(strip=True)
            if not title:
                continue

            # Determine source type based on URL
            source_type = "external"
            if "jamasoftware.com" in full_url:
                if "/blog/" in full_url:
                    source_type = "blog"
                elif "/requirements-management-guide/" in full_url:
                    source_type = "internal"
                elif "resources.jamasoftware.com" in full_url:
                    source_type = "resource"
                else:
                    source_type = "jama"

            related.append(
                RelatedArticle(
                    url=full_url,
                    title=title,
                    source_type=source_type,
                )
            )

        return related

    def _html_to_markdown(self, elem: Tag, include_images: bool = False) -> str:
        """Convert HTML element to markdown."""
        lines = []

        for child in elem.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    lines.append(text)
            elif isinstance(child, Tag):
                lines.append(self._tag_to_markdown(child, include_images))

        # Clean up the result
        markdown = "\n\n".join(line for line in lines if line.strip())
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)  # Limit consecutive newlines
        return markdown.strip()

    def _tag_to_markdown(self, tag: Tag, include_images: bool = False) -> str:
        """Convert a single tag to markdown."""
        name = tag.name

        # Skip non-content tags (safety net if _clean_html missed any)
        if name in TAGS_TO_REMOVE:
            return ""

        # Handle images
        if name == "img" and include_images:
            return self._img_to_markdown(tag)

        # Handle iframes (YouTube embeds)
        if name == "iframe":
            return self._iframe_to_markdown(tag)

        # Headings
        if name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(name[1])
            text = tag.get_text(strip=True)
            return f"{'#' * level} {text}"

        # Paragraphs
        if name == "p":
            return self._inline_to_markdown(tag)

        # Lists
        if name == "ul":
            items = []
            for li in tag.find_all("li", recursive=False):
                items.append(f"- {self._inline_to_markdown(li)}")
            return "\n".join(items)

        if name == "ol":
            items = []
            for i, li in enumerate(tag.find_all("li", recursive=False), 1):
                items.append(f"{i}. {self._inline_to_markdown(li)}")
            return "\n".join(items)

        # Blockquotes
        if name == "blockquote":
            text = self._inline_to_markdown(tag)
            return "\n".join(f"> {line}" for line in text.split("\n"))

        # Code blocks
        if name == "pre":
            code = tag.find("code")
            if code:
                lang = ""
                if code.get("class"):
                    for cls in code["class"]:
                        if cls.startswith("language-"):
                            lang = cls[9:]
                            break
                return f"```{lang}\n{code.get_text()}\n```"
            return f"```\n{tag.get_text()}\n```"

        # Divs and other containers - recurse
        if name in ["div", "section", "article", "main", "aside"]:
            return self._html_to_markdown(tag, include_images)

        # Tables
        if name == "table":
            return self._table_to_markdown(tag)

        # Default: just get text
        return tag.get_text(strip=True)

    def _img_to_markdown(self, img: Tag) -> str:
        """Convert an image tag to markdown with description."""
        # Get the actual image URL (skip lazy-loading placeholders)
        src = img.get("src", "")

        # Skip data: URLs - try alternatives
        if src.startswith("data:"):
            src = img.get("data-src", "") or img.get("data-lazy-src", "")

        # Check noscript for actual URL
        if not src or src.startswith("data:"):
            noscript = img.find_next_sibling("noscript")
            if noscript:
                noscript_img = BeautifulSoup(str(noscript), "lxml").find("img")
                if noscript_img:
                    src = noscript_img.get("src", "")

        if not src or src.startswith("data:"):
            return ""

        alt_text = img.get("alt", "")
        title = img.get("title", "")

        # Build description from available metadata
        description = alt_text or title or "Image"

        # Return markdown image with description
        return f"![{description}]({src})"

    def _iframe_to_markdown(self, iframe: Tag) -> str:
        """Convert an iframe to markdown, handling YouTube embeds.

        Args:
            iframe: BeautifulSoup iframe Tag.

        Returns:
            Markdown representation of the video, or empty string if not supported.
        """
        src = iframe.get("src", "") or iframe.get("data-src", "")
        if not src:
            return ""

        # Extract YouTube video ID
        video_id = None
        for pattern in YOUTUBE_PATTERNS:
            match = pattern.search(src)
            if match:
                video_id = match.group(1)
                break

        if not video_id:
            return ""  # Non-YouTube iframes are not rendered

        # Build watch URL
        watch_url = f"https://www.youtube.com/watch?v={video_id}"

        # Get title from iframe or use default
        title = iframe.get("title", "").strip()
        if not title or title.lower() == "youtube video player":
            title = "YouTube Video"

        # Return as a markdown link with video indicator
        return f"🎬 [{title}]({watch_url})"

    def _inline_to_markdown(self, tag: Tag) -> str:
        """Convert inline content to markdown, preserving nested links."""
        parts = []

        for child in tag.children:
            if isinstance(child, NavigableString):
                # Skip comments
                if isinstance(child, Comment):
                    continue
                parts.append(str(child))
            elif isinstance(child, Tag):
                # Skip non-content tags
                if child.name in TAGS_TO_REMOVE:
                    continue
                if child.name in {"strong", "b"}:
                    # Recursively process to preserve nested links
                    inner = self._inline_to_markdown(child)
                    parts.append(f"**{inner}**")
                elif child.name in {"em", "i"}:
                    inner = self._inline_to_markdown(child)
                    parts.append(f"*{inner}*")
                elif child.name == "u":
                    # Underline - just process contents (no markdown equivalent)
                    parts.append(self._inline_to_markdown(child))
                elif child.name == "code":
                    parts.append(f"`{child.get_text()}`")
                elif child.name == "a":
                    href = child.get("href", "")
                    # Check if this link wraps an image (image link)
                    img = child.find("img")
                    if img:
                        # For image links, use image title/alt as link text
                        text = img.get("title", "") or img.get("alt", "") or "Link"
                        text = text.strip()
                    else:
                        # Recursively process link text for regular links
                        text = self._inline_to_markdown(child)

                    if href and text:
                        parts.append(f"[{text}]({href})")
                    elif text:
                        parts.append(text)
                elif child.name == "br":
                    parts.append("\n")
                elif child.name == "img":
                    parts.append(self._img_to_markdown(child))
                elif child.name == "span":
                    # Process span contents
                    parts.append(self._inline_to_markdown(child))
                else:
                    parts.append(child.get_text())

        return "".join(parts).strip()

    def _table_to_markdown(self, table: Tag) -> str:
        """Convert HTML table to markdown table."""
        rows = []

        # Headers
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
                rows.append("| " + " | ".join(headers) + " |")
                rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Body
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                # Add header separator if we didn't have thead
                if len(rows) == 0:
                    rows.append("| " + " | ".join(cells) + " |")
                    rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
                else:
                    rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)

    def _parse_sections(self, elem: Tag, source_url: str) -> list[Section]:
        """Parse content into sections based on headings."""
        sections = []
        current_heading = None
        current_level = 0
        current_content = []

        for child in elem.descendants:
            if isinstance(child, Tag) and child.name in ["h2", "h3", "h4"]:
                # Save previous section
                if current_heading:
                    sections.append(
                        Section(
                            heading=current_heading,
                            level=current_level,
                            content="\n\n".join(current_content).strip(),
                            cross_references=self._extract_cross_references(
                                BeautifulSoup("".join(str(c) for c in current_content), "lxml"),
                                source_url,
                            )
                            if current_content
                            else [],
                        )
                    )

                # Start new section
                current_heading = child.get_text(strip=True)
                current_level = int(child.name[1])
                current_content = []

            elif isinstance(child, Tag) and child.name == "p" and current_heading:
                current_content.append(self._inline_to_markdown(child))

        # Don't forget the last section
        if current_heading:
            sections.append(
                Section(
                    heading=current_heading,
                    level=current_level,
                    content="\n\n".join(current_content).strip(),
                    cross_references=[],
                )
            )

        return sections

    def _extract_key_concepts(self, elem: Tag, title: str) -> list[str]:
        """Extract key concepts/terms from the content."""
        concepts = set()

        # Add title words (excluding common words)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "how",
            "what",
            "why",
            "when",
            "where",
            "who",
            "which",
            "whom",
            "whose",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "you",
            "your",
            "we",
            "our",
            "they",
            "their",
            "i",
            "my",
            "me",
            "he",
            "she",
            "him",
            "her",
            "his",
            "with",
            "from",
            "by",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        for word in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", title):
            if word.lower() not in stop_words and len(word) > MIN_CONCEPT_LENGTH:
                concepts.add(word)

        # Look for emphasized terms
        for tag in elem.find_all(["strong", "b", "em"]):
            text = tag.get_text(strip=True)
            if (
                MIN_CONCEPT_LENGTH < len(text) < MAX_CONCEPT_LENGTH
                and text.lower() not in stop_words
            ):
                concepts.add(text)

        # Look for definition-like patterns
        for text in elem.stripped_strings:
            # Pattern: "Term is ..." or "Term refers to ..."
            match = re.match(r"^([A-Z][a-zA-Z\s]+?)\s+(?:is|are|refers?\s+to|means?)\s+", text)
            if match:
                term = match.group(1).strip()
                if len(term) < MAX_CONCEPT_LENGTH:
                    concepts.add(term)

        return sorted(concepts)[:MAX_KEY_CONCEPTS]
