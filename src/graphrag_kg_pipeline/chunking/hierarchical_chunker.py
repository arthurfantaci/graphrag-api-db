"""Hierarchical HTML chunking using LangChain text splitters.

This module provides a two-stage chunking strategy:
1. HTMLHeaderTextSplitter: Split by HTML heading elements
2. RecursiveCharacterTextSplitter or Chonkie SemanticChunker: Further split large sections

This approach preserves document structure while ensuring chunks stay
within size limits for embedding and LLM context windows.
"""

from typing import TYPE_CHECKING

import structlog

from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


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

    def split_text_as_documents(
        self,
        html_content: str,
        article_metadata: dict[str, str] | None = None,
    ) -> list["Document"]:
        """Split HTML content into hierarchical chunks with metadata.

        First splits by HTML headers, then prepends section heading context
        to each chunk, then applies character splitting to large sections.

        Args:
            html_content: HTML string to split.
            article_metadata: Optional article context (article_title, etc.)
                to prepend to chunks for improved extraction and retrieval.

        Returns:
            List of Document objects with metadata from headers.
        """
        if not html_content or not html_content.strip():
            return []

        # Stage 1: Split by HTML headers
        header_splits = self.html_splitter.split_text(html_content)

        if not header_splits:
            return []

        # Stage 1.5: Prepend header metadata to chunk text for context
        # This helps both extraction (LLM sees section context) and
        # retrieval (embeddings capture hierarchical position)
        if article_metadata:
            for doc in header_splits:
                prefix_parts = []
                article_title = article_metadata.get("article_title")
                if article_title:
                    prefix_parts.append(f"Article: {article_title}")
                for key in ["article_title", "section", "subsection"]:
                    if key in doc.metadata:
                        prefix_parts.append(doc.metadata[key])
                if prefix_parts:
                    doc.page_content = " > ".join(prefix_parts) + "\n\n" + doc.page_content

        # Stage 2: Further split large sections
        semantic_splits = 0
        rcts_fallbacks = 0
        result = []
        for doc in header_splits:
            if len(doc.page_content) > self.config.sliding_window_threshold:
                if self.config.use_semantic_chunking:
                    sub_docs = self._semantic_split(doc)
                    if sub_docs:
                        result.extend(sub_docs)
                        semantic_splits += 1
                        continue
                    rcts_fallbacks += 1
                # Fallback or default: character splitting
                sub_chunks = self.char_splitter.split_documents([doc])
                result.extend(sub_chunks)
            elif len(doc.page_content) >= self.config.min_chunk_size:
                result.append(doc)
            # Skip tiny fragments below min_chunk_size

        if self.config.use_semantic_chunking:
            logger.info(
                "Semantic chunking stats",
                semantic_splits=semantic_splits,
                rcts_fallbacks=rcts_fallbacks,
            )

        return result

    def _semantic_split(self, doc: "Document") -> list["Document"] | None:
        """Split a document using Chonkie's SemanticChunker.

        Returns a list of Documents on success, or None to signal
        fallback to RecursiveCharacterTextSplitter.

        Args:
            doc: LangChain Document to split semantically.

        Returns:
            List of sub-documents, or None if fallback is needed.
        """
        try:
            from chonkie import SemanticChunker
            from langchain_core.documents import Document as LCDocument

            if not hasattr(self, "_semantic_chunker"):
                self._semantic_chunker = SemanticChunker(
                    threshold=self.config.semantic_threshold,
                    chunk_size=self.config.sliding_window_size,
                )

            chunks = self._semantic_chunker.chunk(doc.page_content)

            if not chunks:
                logger.info("Semantic chunker returned empty output, falling back to RCTS")
                return None

            # Reject if any chunk exceeds 2x the threshold size
            oversized = [
                c for c in chunks if len(c.text) > 2 * self.config.sliding_window_threshold
            ]
            if oversized:
                logger.info(
                    "Semantic chunker produced oversized chunks, falling back to RCTS",
                    oversized_count=len(oversized),
                )
                return None

            # Convert Chonkie chunks to LangChain Documents, preserving metadata
            sub_docs = []
            for chunk in chunks:
                if len(chunk.text) >= self.config.min_chunk_size:
                    sub_docs.append(
                        LCDocument(page_content=chunk.text, metadata=dict(doc.metadata))
                    )

            return sub_docs if sub_docs else None

        except Exception:
            logger.warning("Semantic chunking failed, falling back to RCTS", exc_info=True)
            return None


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

    def split_text_as_documents(
        self,
        markdown_content: str,
        article_metadata: dict[str, str] | None = None,
    ) -> list["Document"]:
        """Split markdown content into hierarchical chunks with metadata.

        Args:
            markdown_content: Markdown string to split.
            article_metadata: Optional article context to prepend to chunks.

        Returns:
            List of Document objects with header metadata.
        """
        if not markdown_content or not markdown_content.strip():
            return []

        # Stage 1: Split by markdown headers
        header_splits = self.md_splitter.split_text(markdown_content)

        if not header_splits:
            return []

        # Stage 1.5: Prepend header metadata to chunk text for context
        if article_metadata:
            for doc in header_splits:
                prefix_parts = []
                article_title = article_metadata.get("article_title")
                if article_title:
                    prefix_parts.append(f"Article: {article_title}")
                for key in ["h1", "h2", "h3"]:
                    if key in doc.metadata:
                        prefix_parts.append(doc.metadata[key])
                if prefix_parts:
                    doc.page_content = " > ".join(prefix_parts) + "\n\n" + doc.page_content

        # Stage 2: Further split large sections
        semantic_splits = 0
        rcts_fallbacks = 0
        result = []
        for doc in header_splits:
            if len(doc.page_content) > self.config.sliding_window_threshold:
                if self.config.use_semantic_chunking:
                    sub_docs = self._semantic_split(doc)
                    if sub_docs:
                        result.extend(sub_docs)
                        semantic_splits += 1
                        continue
                    rcts_fallbacks += 1
                sub_chunks = self.char_splitter.split_documents([doc])
                result.extend(sub_chunks)
            elif len(doc.page_content) >= self.config.min_chunk_size:
                result.append(doc)

        if self.config.use_semantic_chunking:
            logger.info(
                "Semantic chunking stats",
                semantic_splits=semantic_splits,
                rcts_fallbacks=rcts_fallbacks,
            )

        return result

    def _semantic_split(self, doc: "Document") -> list["Document"] | None:
        """Split a document using Chonkie's SemanticChunker.

        Returns a list of Documents on success, or None to signal
        fallback to RecursiveCharacterTextSplitter.

        Args:
            doc: LangChain Document to split semantically.

        Returns:
            List of sub-documents, or None if fallback is needed.
        """
        try:
            from chonkie import SemanticChunker
            from langchain_core.documents import Document as LCDocument

            if not hasattr(self, "_semantic_chunker"):
                self._semantic_chunker = SemanticChunker(
                    threshold=self.config.semantic_threshold,
                    chunk_size=self.config.sliding_window_size,
                )

            chunks = self._semantic_chunker.chunk(doc.page_content)

            if not chunks:
                logger.info("Semantic chunker returned empty output, falling back to RCTS")
                return None

            oversized = [
                c for c in chunks if len(c.text) > 2 * self.config.sliding_window_threshold
            ]
            if oversized:
                logger.info(
                    "Semantic chunker produced oversized chunks, falling back to RCTS",
                    oversized_count=len(oversized),
                )
                return None

            sub_docs = []
            for chunk in chunks:
                if len(chunk.text) >= self.config.min_chunk_size:
                    sub_docs.append(
                        LCDocument(page_content=chunk.text, metadata=dict(doc.metadata))
                    )

            return sub_docs if sub_docs else None

        except Exception:
            logger.warning("Semantic chunking failed, falling back to RCTS", exc_info=True)
            return None
