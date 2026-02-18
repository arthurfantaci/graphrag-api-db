"""Hierarchical HTML chunking using LangChain text splitters.

This module provides a two-stage chunking strategy:
1. HTMLHeaderTextSplitter: Split by HTML heading elements
2. RecursiveCharacterTextSplitter: Further split large sections

This approach preserves document structure while ensuring chunks stay
within size limits for embedding and LLM context windows.
"""

from typing import TYPE_CHECKING

from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig

if TYPE_CHECKING:
    from langchain_core.documents import Document


class HierarchicalHTMLSplitter:
    """Two-stage hierarchical HTML splitter.

    First stage uses HTMLHeaderTextSplitter to split by heading elements,
    preserving heading metadata. Second stage uses RecursiveCharacterTextSplitter
    to further split sections exceeding the threshold.

    Attributes:
        config: Chunking configuration.
        html_splitter: LangChain HTMLHeaderTextSplitter instance.
        char_splitter: LangChain RecursiveCharacterTextSplitter instance.

    Example:
        >>> config = HierarchicalChunkingConfig(
        ...     sliding_window_size=512,
        ...     sliding_window_overlap=64,
        ...     sliding_window_threshold=1500,
        ... )
        >>> splitter = HierarchicalHTMLSplitter(config)
        >>> chunks = splitter.split_text("<h1>Title</h1><p>Content...</p>")
        >>> len(chunks)
        3
    """

    def __init__(self, config: HierarchicalChunkingConfig | None = None) -> None:
        """Initialize the hierarchical splitter.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        # Lazy import to avoid requiring langchain at module load
        from langchain_text_splitters import (
            HTMLHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        self.config = config or HierarchicalChunkingConfig()

        # Stage 1: HTML header-based splitting
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=list(self.config.headers_to_split_on),
            return_each_element=False,
        )

        # Stage 2: Character-based splitting for large sections
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.sliding_window_size,
            chunk_overlap=self.config.sliding_window_overlap,
            separators=list(self.config.separators),
            keep_separator=self.config.keep_separator,
        )

    def split_text(self, html_content: str) -> list[str]:
        """Split HTML content into hierarchical chunks.

        First splits by HTML headers, then applies character splitting
        to any sections exceeding the threshold.

        Args:
            html_content: HTML string to split.

        Returns:
            List of text strings (chunk content).
        """
        docs = self.split_text_as_documents(html_content)
        return [doc.page_content for doc in docs]

    def split_text_as_documents(self, html_content: str) -> list["Document"]:
        """Split HTML content into hierarchical chunks with metadata.

        First splits by HTML headers, then applies character splitting
        to any sections exceeding the threshold.

        Args:
            html_content: HTML string to split.

        Returns:
            List of Document objects with metadata from headers.
        """
        if not html_content or not html_content.strip():
            return []

        # Stage 1: Split by HTML headers
        header_splits = self.html_splitter.split_text(html_content)

        if not header_splits:
            return []

        # Stage 2: Further split large sections
        result = []
        for doc in header_splits:
            if len(doc.page_content) > self.config.sliding_window_threshold:
                # Large section - apply character splitting
                sub_chunks = self.char_splitter.split_documents([doc])
                result.extend(sub_chunks)
            elif len(doc.page_content) >= self.config.min_chunk_size:
                # Section within limits - keep as is
                result.append(doc)
            # Skip tiny fragments below min_chunk_size

        return result

    def split_text_with_positions(self, html_content: str) -> list[tuple["Document", int, int]]:
        """Split HTML content and track character positions.

        Like split_text but also returns start/end character positions
        in the original content, useful for entity linking.

        Args:
            html_content: HTML string to split.

        Returns:
            List of (Document, start_pos, end_pos) tuples.
        """
        chunks = self.split_text_as_documents(html_content)
        result = []

        # Track positions by finding each chunk's content in original
        search_start = 0
        for chunk in chunks:
            content = chunk.page_content
            # Find position in remaining content
            pos = html_content.find(content, search_start)
            if pos >= 0:
                result.append((chunk, pos, pos + len(content)))
                search_start = pos + len(content)
            else:
                # Fallback if exact match not found (due to splitting transformations)
                result.append((chunk, -1, -1))

        return result


class MarkdownSplitter:
    """Simple markdown splitter without HTML preprocessing.

    Use this when content is already in markdown format
    (post HTML-to-markdown conversion).
    """

    def __init__(self, config: HierarchicalChunkingConfig | None = None) -> None:
        """Initialize the markdown splitter.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        self.config = config or HierarchicalChunkingConfig()

        # Markdown header patterns
        markdown_headers = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=markdown_headers,
            return_each_line=False,
        )

        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.sliding_window_size,
            chunk_overlap=self.config.sliding_window_overlap,
            separators=list(self.config.separators),
            keep_separator=self.config.keep_separator,
        )

    def split_text(self, markdown_content: str) -> list[str]:
        """Split markdown content into hierarchical chunks.

        Args:
            markdown_content: Markdown string to split.

        Returns:
            List of text strings (chunk content).
        """
        docs = self.split_text_as_documents(markdown_content)
        return [doc.page_content for doc in docs]

    def split_text_as_documents(self, markdown_content: str) -> list["Document"]:
        """Split markdown content into hierarchical chunks with metadata.

        Args:
            markdown_content: Markdown string to split.

        Returns:
            List of Document objects with header metadata.
        """
        if not markdown_content or not markdown_content.strip():
            return []

        # Stage 1: Split by markdown headers
        header_splits = self.md_splitter.split_text(markdown_content)

        if not header_splits:
            return []

        # Stage 2: Further split large sections
        result = []
        for doc in header_splits:
            if len(doc.page_content) > self.config.sliding_window_threshold:
                sub_chunks = self.char_splitter.split_documents([doc])
                result.extend(sub_chunks)
            elif len(doc.page_content) >= self.config.min_chunk_size:
                result.append(doc)

        return result
