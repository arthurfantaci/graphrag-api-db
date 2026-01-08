"""Pydantic models for text chunks and embeddings.

This module defines the data structures for:
- Chunk: A text segment with entity linkage metadata
- ChunkedGuide: Collection of all chunks with indexing
- EmbeddedChunk: A chunk with its embedding vector

These models support the three-tier chunking strategy:
1. Article summaries (~300 tokens)
2. Section-based chunks (natural boundaries)
3. Sliding window chunks (for large sections)
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ChunkType(str, Enum):
    """Types of chunks in the three-tier hierarchy."""

    ARTICLE_SUMMARY = "article_summary"  # Tier 1: condensed article overview
    SECTION = "section"  # Tier 2: natural heading-based boundaries
    SLIDING_WINDOW = "sliding_window"  # Tier 3: overlapping windows for large sections


class Chunk(BaseModel):
    """A text chunk with entity linkage metadata for GraphRAG.

    Each chunk is a segment of an article's markdown content, with:
    - Precise character positions in the source article
    - Links to entities that appear in this chunk (via CharInterval overlap)
    - Token count for embedding cost estimation

    Attributes:
        chunk_id: Unique identifier (e.g., "ch1-art3-sec2", "ch1-art3-sw0").
        chunk_type: Type of chunk (summary, section, or sliding_window).
        source_article_id: Parent article ID (e.g., "ch1-art3").
        source_section_index: Section index for section/sliding_window chunks.
        text: The chunk's text content.
        char_start: Start position in article's markdown_content.
        char_end: End position in article's markdown_content.
        entity_ids: IDs of entities overlapping this chunk (via CharInterval).
        heading: Section heading if applicable (for context in retrieval).
        chapter_number: Chapter number for filtering.
        token_count: Number of tokens (via tiktoken).
    """

    chunk_id: str = Field(description="Unique chunk identifier")
    chunk_type: ChunkType = Field(description="Type of chunk in 3-tier hierarchy")
    source_article_id: str = Field(description="Parent article ID")
    source_section_index: int | None = Field(
        default=None,
        description="Section index (0-based) for section/sliding_window chunks",
    )

    # Content
    text: str = Field(description="Chunk text content")
    char_start: int = Field(description="Start position in article markdown_content")
    char_end: int = Field(description="End position in article markdown_content")

    # Entity linkage (populated via CharInterval overlap detection)
    entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity IDs that overlap this chunk's char range",
    )

    # Metadata for retrieval
    heading: str | None = Field(
        default=None, description="Section heading for context"
    )
    chapter_number: int | None = Field(
        default=None, description="Chapter number for filtering"
    )
    article_title: str | None = Field(
        default=None, description="Article title for context"
    )

    # Token count (for cost estimation)
    token_count: int = Field(description="Number of tokens (via tiktoken)")

    @computed_field
    @property
    def char_count(self) -> int:
        """Character count of chunk text."""
        return len(self.text)

    @computed_field
    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())

    @computed_field
    @property
    def entity_count(self) -> int:
        """Number of linked entities."""
        return len(self.entity_ids)


class EmbeddedChunk(BaseModel):
    """A chunk with its embedding vector.

    This model is used to store embeddings separately from chunks,
    allowing for different embedding models and incremental updates.

    Attributes:
        chunk_id: Reference to the Chunk this embedding belongs to.
        embedding: The embedding vector (dimension varies by model).
        model_id: Which embedding model generated this vector.
        embedded_at: Timestamp of embedding generation.
    """

    chunk_id: str = Field(description="Chunk ID this embedding belongs to")
    embedding: list[float] = Field(description="Embedding vector")
    model_id: str = Field(
        description="Embedding model ID (e.g., text-embedding-3-small)"
    )
    embedded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the embedding was generated",
    )

    @computed_field
    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        return len(self.embedding)


class ChunkingStats(BaseModel):
    """Statistics about the chunking process."""

    total_chunks: int = Field(description="Total number of chunks created")
    summary_chunks: int = Field(description="Tier 1: article summary chunks")
    section_chunks: int = Field(description="Tier 2: section-based chunks")
    sliding_window_chunks: int = Field(description="Tier 3: sliding window chunks")
    total_tokens: int = Field(description="Total tokens across all chunks")
    total_entities_linked: int = Field(description="Total entity-chunk links")
    articles_processed: int = Field(description="Number of articles chunked")


class ChunkedGuide(BaseModel):
    """Collection of all chunks from the guide with indexing for GraphRAG.

    This model provides:
    - All chunks keyed by chunk_id for lookup
    - Forward index: article_id → chunk_ids
    - Reverse index: entity_id → chunk_ids (critical for GraphRAG)

    The reverse index (entity_to_chunks) enables hybrid retrieval:
    1. Vector search finds semantically similar chunks
    2. Graph traversal finds related entities
    3. Reverse index expands to chunks mentioning those entities

    Attributes:
        chunks: All chunks keyed by chunk_id.
        article_to_chunks: Forward index from article to its chunks.
        entity_to_chunks: Reverse index from entity to chunks mentioning it.
        chunking_config: Configuration used for chunking.
        stats: Statistics about the chunking process.
        created_at: When this chunked guide was created.
    """

    chunks: dict[str, Chunk] = Field(
        default_factory=dict, description="All chunks keyed by chunk_id"
    )

    article_to_chunks: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Forward index: article_id → list of chunk_ids",
    )

    entity_to_chunks: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Reverse index: entity_id → list of chunk_ids (for GraphRAG)",
    )

    chunking_config: dict[str, Any] = Field(
        default_factory=dict, description="Configuration used for chunking"
    )

    stats: ChunkingStats | None = Field(
        default=None, description="Statistics about chunking"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this chunked guide was created",
    )

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk and update indices.

        Args:
            chunk: Chunk to add.
        """
        # Add to main chunks dict
        self.chunks[chunk.chunk_id] = chunk

        # Update forward index (article → chunks)
        if chunk.source_article_id not in self.article_to_chunks:
            self.article_to_chunks[chunk.source_article_id] = []
        self.article_to_chunks[chunk.source_article_id].append(chunk.chunk_id)

        # Update reverse index (entity → chunks)
        for entity_id in chunk.entity_ids:
            if entity_id not in self.entity_to_chunks:
                self.entity_to_chunks[entity_id] = []
            self.entity_to_chunks[entity_id].append(chunk.chunk_id)

    def get_chunks_for_article(self, article_id: str) -> list[Chunk]:
        """Get all chunks for an article.

        Args:
            article_id: Article ID to look up.

        Returns:
            List of chunks belonging to the article.
        """
        chunk_ids = self.article_to_chunks.get(article_id, [])
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def get_chunks_for_entity(self, entity_id: str) -> list[Chunk]:
        """Get all chunks mentioning an entity (GraphRAG reverse lookup).

        Args:
            entity_id: Entity ID to look up.

        Returns:
            List of chunks that mention this entity.
        """
        chunk_ids = self.entity_to_chunks.get(entity_id, [])
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def compute_stats(self) -> ChunkingStats:
        """Compute and set statistics about the chunking."""
        summary_count = 0
        section_count = 0
        window_count = 0
        total_tokens = 0
        total_entity_links = 0
        articles = set()

        for chunk in self.chunks.values():
            if chunk.chunk_type == ChunkType.ARTICLE_SUMMARY:
                summary_count += 1
            elif chunk.chunk_type == ChunkType.SECTION:
                section_count += 1
            elif chunk.chunk_type == ChunkType.SLIDING_WINDOW:
                window_count += 1

            total_tokens += chunk.token_count
            total_entity_links += len(chunk.entity_ids)
            articles.add(chunk.source_article_id)

        self.stats = ChunkingStats(
            total_chunks=len(self.chunks),
            summary_chunks=summary_count,
            section_chunks=section_count,
            sliding_window_chunks=window_count,
            total_tokens=total_tokens,
            total_entities_linked=total_entity_links,
            articles_processed=len(articles),
        )
        return self.stats


class EmbeddedGuideChunks(BaseModel):
    """Collection of embeddings for all chunks.

    Stored separately from ChunkedGuide to allow:
    - Different embedding models
    - Incremental embedding updates
    - Separate storage/caching

    Attributes:
        embeddings: All embeddings keyed by chunk_id.
        model_id: Embedding model used.
        total_tokens_embedded: Total tokens processed.
        created_at: When embeddings were generated.
    """

    embeddings: dict[str, EmbeddedChunk] = Field(
        default_factory=dict, description="Embeddings keyed by chunk_id"
    )

    model_id: str = Field(description="Embedding model ID")

    total_tokens_embedded: int = Field(
        default=0, description="Total tokens embedded"
    )

    estimated_cost_usd: float = Field(
        default=0.0, description="Estimated API cost in USD"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When embeddings were generated",
    )

    def add_embedding(self, embedded: EmbeddedChunk) -> None:
        """Add an embedding to the collection.

        Args:
            embedded: EmbeddedChunk to add.
        """
        self.embeddings[embedded.chunk_id] = embedded

    def get_embedding(self, chunk_id: str) -> list[float] | None:
        """Get embedding vector for a chunk.

        Args:
            chunk_id: Chunk ID to look up.

        Returns:
            Embedding vector or None if not found.
        """
        embedded = self.embeddings.get(chunk_id)
        return embedded.embedding if embedded else None

    @computed_field
    @property
    def total_embeddings(self) -> int:
        """Total number of embeddings."""
        return len(self.embeddings)
