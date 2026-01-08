"""Three-tier text chunking for GraphRAG retrieval.

This module implements a three-tier chunking strategy:

1. **Tier 1 - Article Summaries** (~300 tokens):
   - Condensed overview of each article
   - Uses LLM-generated summary if available (from enrichment)
   - Falls back to first N characters of content

2. **Tier 2 - Section Chunks** (natural boundaries):
   - Uses existing Article.sections from HTML parser
   - Respects heading hierarchy (h2/h3 boundaries)
   - Skips very short sections

3. **Tier 3 - Sliding Window** (for large sections):
   - Applied when sections exceed threshold
   - Overlapping windows ensure context continuity
   - Enables fine-grained retrieval

All chunks are linked to entities via CharInterval overlap detection,
enabling hybrid GraphRAG retrieval (vector + graph).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .chunk_models import Chunk, ChunkedGuide, ChunkType
from .chunking_config import DEFAULT_CHUNKING_CONFIG, ChunkingConfig
from .text_utils import (
    count_tokens,
    find_entity_ids_in_range,
    generate_chunk_id,
    normalize_text_for_embedding,
    split_text_with_overlap,
    truncate_to_tokens,
)

if TYPE_CHECKING:
    from .graph_models import ArticleEnrichment, EnrichedGuide, ExtractedEntity
    from .models import Article, RequirementsManagementGuide

console = Console()


class JamaChunker:
    """Chunk articles into GraphRAG-friendly segments.

    Implements three-tier chunking:
    1. Article summaries (Tier 1) - condensed overview
    2. Section-based chunks (Tier 2) - natural boundaries
    3. Sliding window (Tier 3) - for large sections

    Links chunks to entities via CharInterval overlap detection.

    Example:
        >>> config = ChunkingConfig()
        >>> chunker = JamaChunker(config)
        >>> chunked_guide = chunker.chunk_guide(guide, enriched_guide)
        >>> print(f"Created {len(chunked_guide.chunks)} chunks")
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the chunker.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or DEFAULT_CHUNKING_CONFIG

    def chunk_guide(
        self,
        guide: RequirementsManagementGuide,
        enriched: EnrichedGuide | None = None,
    ) -> ChunkedGuide:
        """Chunk all articles in the guide.

        Args:
            guide: The scraped guide with all articles.
            enriched: Optional enriched guide with entity data and summaries.
                      If provided, enables entity linkage and uses LLM summaries.

        Returns:
            ChunkedGuide with all chunks and indices.
        """
        chunked_guide = ChunkedGuide(chunking_config=self.config.to_dict())

        # Collect all articles
        all_articles: list[Article] = []
        for chapter in guide.chapters:
            all_articles.extend(chapter.articles)

        console.print(f"\n[bold cyan]Chunking {len(all_articles)} articles...[/]")

        # Get enrichment data if available
        article_enrichments: dict[str, ArticleEnrichment] = {}
        if enriched:
            article_enrichments = enriched.article_enrichments

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking articles...", total=len(all_articles))

            for article in all_articles:
                progress.update(
                    task,
                    description=f"Chunking: {article.title[:40]}...",
                )

                # Get enrichment for this article (if available)
                enrichment = article_enrichments.get(article.article_id)

                # Get entities for this article (for linking)
                article_entities: list[ExtractedEntity] = []
                if enrichment and self.config.link_entities:
                    article_entities = enrichment.entities

                # Chunk the article
                chunks = self._chunk_article(article, enrichment, article_entities)

                # Add chunks to guide
                for chunk in chunks:
                    chunked_guide.add_chunk(chunk)

                progress.advance(task)

        # Compute statistics
        stats = chunked_guide.compute_stats()

        console.print("\n[green]Chunking complete![/]")
        console.print(f"  Total chunks: {stats.total_chunks}")
        console.print(f"    Summary chunks (Tier 1): {stats.summary_chunks}")
        console.print(f"    Section chunks (Tier 2): {stats.section_chunks}")
        console.print(f"    Sliding window (Tier 3): {stats.sliding_window_chunks}")
        console.print(f"  Total tokens: {stats.total_tokens:,}")
        console.print(f"  Entity-chunk links: {stats.total_entities_linked:,}")

        return chunked_guide

    def _chunk_article(
        self,
        article: Article,
        enrichment: ArticleEnrichment | None,
        entities: list[ExtractedEntity],
    ) -> list[Chunk]:
        """Chunk a single article using three-tier strategy.

        Args:
            article: Article to chunk.
            enrichment: Optional enrichment with summary and entities.
            entities: Entities from this article (for linking).

        Returns:
            List of chunks from this article.
        """
        chunks: list[Chunk] = []

        # Tier 1: Article summary
        if self.config.include_summaries:
            summary_chunk = self._create_summary_chunk(article, enrichment, entities)
            if summary_chunk:
                chunks.append(summary_chunk)

        # Tier 2 & 3: Section-based chunks (with sliding window for large sections)
        section_chunks = self._create_section_chunks(article, entities)
        chunks.extend(section_chunks)

        return chunks

    def _create_summary_chunk(
        self,
        article: Article,
        enrichment: ArticleEnrichment | None,
        entities: list[ExtractedEntity],
    ) -> Chunk | None:
        """Create Tier 1 article summary chunk.

        Uses LLM-generated summary from enrichment if available,
        otherwise uses first N tokens of article content.

        Args:
            article: Article to summarize.
            enrichment: Optional enrichment with LLM summary.
            entities: Entities for linking.

        Returns:
            Summary chunk or None if article too short.
        """
        # Prefer LLM-generated summary from enrichment
        if enrichment and enrichment.summary:
            summary_text = enrichment.summary
        else:
            # Fall back to truncated content
            summary_text = truncate_to_tokens(
                article.markdown_content,
                self.config.summary_max_tokens,
            )

        # Normalize for embedding
        summary_text = normalize_text_for_embedding(summary_text)

        if not summary_text:
            return None

        # Count tokens
        token_count = count_tokens(summary_text)

        # For summary, we link to all entities in the article
        # (since summary represents whole article)
        entity_ids = [e.entity_id for e in entities] if entities else []

        return Chunk(
            chunk_id=generate_chunk_id(article.article_id, "summary", 0),
            chunk_type=ChunkType.ARTICLE_SUMMARY,
            source_article_id=article.article_id,
            source_section_index=None,
            text=summary_text,
            char_start=0,
            char_end=len(summary_text),
            entity_ids=entity_ids,
            heading=article.title,
            chapter_number=article.chapter_number,
            article_title=article.title,
            token_count=token_count,
        )

    def _create_section_chunks(
        self,
        article: Article,
        entities: list[ExtractedEntity],
    ) -> list[Chunk]:
        """Create Tier 2 and Tier 3 chunks from article sections.

        - Tier 2: Section-based chunks using natural boundaries
        - Tier 3: Sliding window applied to large sections

        Args:
            article: Article with sections.
            entities: Entities for linking.

        Returns:
            List of section and/or sliding window chunks.
        """
        chunks: list[Chunk] = []

        # Track character position in markdown_content
        # Note: sections.content may not map 1:1 to markdown_content positions
        # We'll use content matching to find approximate positions
        markdown_content = article.markdown_content
        current_pos = 0

        for section_idx, section in enumerate(article.sections):
            section_text = normalize_text_for_embedding(section.content)

            if not section_text:
                continue

            # Count tokens in section
            section_tokens = count_tokens(section_text)

            # Skip very short sections
            if section_tokens < self.config.section_min_tokens:
                continue

            # Find approximate position of section in markdown
            # (sections don't have char positions, so we search)
            section_start = markdown_content.find(section.content[:100], current_pos)
            if section_start == -1:
                section_start = current_pos
            section_end = section_start + len(section.content)
            current_pos = section_end

            # Decide: Tier 2 (section) or Tier 3 (sliding window)
            if section_tokens <= self.config.sliding_window_threshold:
                # Tier 2: Single section chunk
                entity_ids = find_entity_ids_in_range(
                    section_start, section_end, entities
                ) if entities else []

                chunk_id = generate_chunk_id(
                    article.article_id, "section", section_idx
                )
                chunk = Chunk(
                    chunk_id=chunk_id,
                    chunk_type=ChunkType.SECTION,
                    source_article_id=article.article_id,
                    source_section_index=section_idx,
                    text=section_text,
                    char_start=section_start,
                    char_end=section_end,
                    entity_ids=entity_ids,
                    heading=section.heading,
                    chapter_number=article.chapter_number,
                    article_title=article.title,
                    token_count=section_tokens,
                )
                chunks.append(chunk)
            else:
                # Tier 3: Apply sliding window to large section
                windows = split_text_with_overlap(
                    section_text,
                    max_tokens=self.config.sliding_window_size,
                    overlap_tokens=self.config.sliding_window_overlap,
                )

                for window_idx, (window_text, rel_start, rel_end) in enumerate(windows):
                    # Calculate absolute positions
                    abs_start = section_start + rel_start
                    abs_end = section_start + rel_end

                    # Find entities in this window
                    entity_ids = find_entity_ids_in_range(
                        abs_start, abs_end, entities
                    ) if entities else []

                    window_tokens = count_tokens(window_text)

                    chunk_id = generate_chunk_id(
                        article.article_id,
                        "sliding_window",
                        section_idx,
                        window_idx,
                    )
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        chunk_type=ChunkType.SLIDING_WINDOW,
                        source_article_id=article.article_id,
                        source_section_index=section_idx,
                        text=window_text,
                        char_start=abs_start,
                        char_end=abs_end,
                        entity_ids=entity_ids,
                        heading=f"{section.heading} (part {window_idx + 1})",
                        chapter_number=article.chapter_number,
                        article_title=article.title,
                        token_count=window_tokens,
                    )
                    chunks.append(chunk)

        return chunks

    def chunk_article(
        self,
        article: Article,
        enrichment: ArticleEnrichment | None = None,
    ) -> list[Chunk]:
        """Chunk a single article (public API).

        Convenience method for chunking individual articles without
        processing the entire guide.

        Args:
            article: Article to chunk.
            enrichment: Optional enrichment with summary and entities.

        Returns:
            List of chunks from the article.
        """
        entities = enrichment.entities if enrichment else []
        return self._chunk_article(article, enrichment, entities)


def check_chunking_available() -> bool:
    """Check if chunking dependencies are available.

    Returns:
        True if tiktoken is installed, False otherwise.
    """
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        return False
    return True


class ChunkingNotAvailableError(Exception):
    """Raised when chunking dependencies are not installed."""

    def __init__(self) -> None:
        """Initialize the error with installation instructions."""
        super().__init__(
            "Chunking requires tiktoken. Install with: uv sync --group embedding"
        )
