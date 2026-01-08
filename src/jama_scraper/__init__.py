"""Jama Requirements Management Guide Scraper.

A tool to scrape and consolidate Jama Software's "Essential Guide to
Requirements Management and Traceability" into LLM-friendly formats.

Usage:
    from jama_scraper import JamaGuideScraper, run_scraper
    import asyncio

    # Quick run
    guide = asyncio.run(run_scraper())

    # Or with more control
    scraper = JamaGuideScraper()
    guide = asyncio.run(scraper.scrape_all())

    # For JavaScript-rendered content (e.g., YouTube embeds):
    scraper = JamaGuideScraper(use_browser=True)
    guide = asyncio.run(scraper.scrape_all())

    # With LLM enrichment for GraphRAG (requires langextract):
    guide = asyncio.run(run_scraper(enrich=True, export_neo4j=True))

    # Full GraphRAG pipeline with chunking and embedding:
    guide = asyncio.run(run_scraper(
        enrich=True,
        chunk=True,
        embed=True,
        export_neo4j=True,
    ))
"""

from .chunk_export import ChunkExporter
from .chunk_models import (
    Chunk,
    ChunkedGuide,
    ChunkType,
    EmbeddedChunk,
    EmbeddedGuideChunks,
)
from .chunker import ChunkingNotAvailableError, JamaChunker, check_chunking_available
from .chunking_config import ChunkingConfig
from .embedder import (
    EmbeddingNotAvailableError,
    JamaEmbedder,
    check_embedding_available,
)
from .embedding_config import EmbeddingConfig, EmbeddingProvider
from .enrichment_config import EnrichmentConfig, LLMProvider, create_config_from_args
from .exceptions import (
    BrowserNotInstalledError,
    FetchError,
    PlaywrightNotAvailableError,
    ScraperError,
)
from .extractor import (
    JamaExtractor,
    LangExtractNotAvailableError,
    check_langextract_available,
)
from .fetcher import (
    Fetcher,
    FetcherConfig,
    HttpxFetcher,
    PlaywrightFetcher,
    create_fetcher,
)
from .graph_export import Neo4jExporter, generate_import_command
from .graph_models import (
    ArticleEnrichment,
    CharInterval,
    EnrichedGuide,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    GlossaryEnrichment,
    RelationshipType,
)
from .models import (
    Article,
    Chapter,
    ContentType,
    CrossReference,
    Glossary,
    GlossaryTerm,
    GuideMetadata,
    ImageReference,
    RelatedArticle,
    RequirementsManagementGuide,
    Section,
    VideoReference,
    WebinarReference,
)
from .scraper import JamaGuideScraper, run_scraper

__version__ = "0.1.0"

__all__ = [
    # Core Models
    "Article",
    "Chapter",
    "ContentType",
    "CrossReference",
    "Glossary",
    "GlossaryTerm",
    "GuideMetadata",
    "ImageReference",
    "RelatedArticle",
    "RequirementsManagementGuide",
    "Section",
    "VideoReference",
    "WebinarReference",
    # Scraper
    "JamaGuideScraper",
    "run_scraper",
    # Fetchers
    "Fetcher",
    "FetcherConfig",
    "HttpxFetcher",
    "PlaywrightFetcher",
    "create_fetcher",
    # Enrichment
    "EnrichmentConfig",
    "LLMProvider",
    "create_config_from_args",
    "JamaExtractor",
    "check_langextract_available",
    # Graph Models (for Neo4j/GraphRAG)
    "ArticleEnrichment",
    "CharInterval",
    "EnrichedGuide",
    "EntityType",
    "ExtractedEntity",
    "ExtractedRelationship",
    "GlossaryEnrichment",
    "RelationshipType",
    # Graph Export
    "Neo4jExporter",
    "generate_import_command",
    # Chunking (for GraphRAG retrieval)
    "Chunk",
    "ChunkedGuide",
    "ChunkType",
    "ChunkingConfig",
    "JamaChunker",
    "check_chunking_available",
    "ChunkExporter",
    # Embedding (for vector search)
    "EmbeddedChunk",
    "EmbeddedGuideChunks",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "JamaEmbedder",
    "check_embedding_available",
    # Exceptions
    "BrowserNotInstalledError",
    "ChunkingNotAvailableError",
    "EmbeddingNotAvailableError",
    "FetchError",
    "LangExtractNotAvailableError",
    "PlaywrightNotAvailableError",
    "ScraperError",
]
