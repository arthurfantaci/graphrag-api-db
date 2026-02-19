"""Custom DataLoader for Jama Guide HTML content.

This module implements the neo4j_graphrag DataLoader interface for
loading and preprocessing Jama Guide articles.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup, NavigableString, Tag
import structlog

if TYPE_CHECKING:
    from neo4j_graphrag.experimental.components.pdf_loader import PdfDocument

logger = structlog.get_logger(__name__)


class JamaHTMLLoader:
    """DataLoader for Jama Guide HTML and markdown content.

    Implements the neo4j_graphrag DataLoader pattern for loading
    article content. Supports both file-based loading and direct
    content loading via article index.

    Attributes:
        article_index: Mapping from article_id to metadata dict.
        preprocess_html: Whether to clean HTML before processing.

    Example:
        >>> index = build_article_index(guide)
        >>> loader = JamaHTMLLoader(article_index=index)
        >>> doc = await loader.run("ch1-art3")
        >>> print(doc.text[:100])
    """

    def __init__(
        self,
        article_index: dict[str, dict[str, Any]] | None = None,
        preprocess_html: bool = True,
    ) -> None:
        """Initialize the loader.

        Args:
            article_index: Optional mapping of article_id to metadata.
            preprocess_html: Whether to clean HTML (remove nav, scripts, etc.).
        """
        self.article_index = article_index or {}
        self.preprocess_html = preprocess_html

    async def run(
        self,
        source: str | Path,
    ) -> "PdfDocument":
        """Load content from a source.

        Supports multiple source types:
        - article_id string (looks up in article_index)
        - Path to HTML file
        - Path to markdown file

        Args:
            source: Article ID or file path.

        Returns:
            PdfDocument with text content and metadata.

        Raises:
            ValueError: If source cannot be resolved.
        """
        from neo4j_graphrag.experimental.components.pdf_loader import PdfDocument

        # Check if source is an article_id
        if isinstance(source, str) and source in self.article_index:
            from neo4j_graphrag.experimental.components.types import DocumentInfo

            article = self.article_index[source]
            return PdfDocument(
                text=article.get("markdown_content", ""),
                document_info=DocumentInfo(
                    path=article.get("url", source),
                    metadata={
                        "article_id": source,
                        "title": article.get("title", ""),
                        "chapter_number": str(article.get("chapter_number", 0)),
                        "source_type": "article_index",
                    },
                ),
            )

        # Check if source is a file path
        filepath = Path(source) if isinstance(source, str) else source

        if not filepath.exists():
            msg = f"Source not found: {source}"
            raise ValueError(msg)

        # Load based on file extension
        suffix = filepath.suffix.lower()

        if suffix in {".html", ".htm"}:
            content = await self._load_html(filepath)
        elif suffix in {".md", ".markdown"}:
            content = await self._load_markdown(filepath)
        else:
            msg = f"Unsupported file type: {suffix}"
            raise ValueError(msg)

        from neo4j_graphrag.experimental.components.types import DocumentInfo

        return PdfDocument(
            text=content,
            document_info=DocumentInfo(
                path=str(filepath),
                metadata={
                    "filename": filepath.name,
                    "source_type": "file",
                },
            ),
        )

    async def _load_html(self, filepath: Path) -> str:
        """Load and preprocess HTML file.

        Args:
            filepath: Path to HTML file.

        Returns:
            Extracted text content.
        """
        html_content = filepath.read_text(encoding="utf-8")

        if self.preprocess_html:
            html_content = self._preprocess_html(html_content)

        # Extract text from HTML
        soup = BeautifulSoup(html_content, "lxml")
        return self._extract_text(soup)

    async def _load_markdown(self, filepath: Path) -> str:
        """Load markdown file.

        Args:
            filepath: Path to markdown file.

        Returns:
            Markdown content.
        """
        return filepath.read_text(encoding="utf-8")

    def _preprocess_html(self, html: str) -> str:
        """Preprocess HTML by removing navigation, scripts, etc.

        Args:
            html: Raw HTML content.

        Returns:
            Cleaned HTML.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove unwanted elements
        unwanted_selectors = [
            "nav",
            "header",
            "footer",
            "script",
            "style",
            "noscript",
            "aside",
            ".navigation",
            ".sidebar",
            ".menu",
            ".advertisement",
            "#cookie-banner",
        ]

        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()

        # Try to find main content area
        main_content = (
            soup.select_one(".flex_cell_inner")
            or soup.select_one("main")
            or soup.select_one("article")
            or soup.select_one(".content")
            or soup
        )

        return str(main_content)

    def _extract_text(self, soup: BeautifulSoup | Tag) -> str:
        """Extract readable text from soup.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Extracted text with whitespace normalized.
        """
        # Get text with some structure preserved
        lines = []
        for element in soup.descendants:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    lines.append(text)
            elif isinstance(element, Tag) and element.name in {"p", "div", "br", "li"}:
                lines.append("\n")

        text = " ".join(lines)
        # Normalize whitespace
        text = " ".join(text.split())
        return text
