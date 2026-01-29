"""Jama Requirements Management Guide Scraper.

A tool to scrape and consolidate Jama Software's "Essential Guide to
Requirements Management and Traceability" into a Neo4j knowledge graph
using neo4j_graphrag's SimpleKGPipeline.

Usage:
    from graphrag_kg_pipeline import JamaGuideScraper, run_scraper
    import asyncio

    # Quick run (scrape + Neo4j pipeline)
    guide = asyncio.run(run_scraper())

    # Scrape only (no Neo4j)
    guide = asyncio.run(run_scraper(scrape_only=True))

    # Or with more control
    scraper = JamaGuideScraper()
    guide = asyncio.run(scraper.scrape_all())

    # For JavaScript-rendered content (e.g., YouTube embeds):
    scraper = JamaGuideScraper(use_browser=True)
    guide = asyncio.run(scraper.scrape_all())

    # Full pipeline with validation:
    guide = asyncio.run(run_scraper(run_validation=True))
"""

# =============================================================================
# CORE MODELS (always available)
# =============================================================================
# Chunking
from .chunking import (
    HierarchicalChunkingConfig,
    HierarchicalHTMLSplitter,
    create_text_splitter_adapter,
)

# =============================================================================
# EXCEPTIONS (always available)
# =============================================================================
from .exceptions import (
    BrowserNotInstalledError,
    FetchError,
    Neo4jConfigError,
    PlaywrightNotAvailableError,
    ScraperError,
)

# Extraction
from .extraction import (
    NODE_TYPES,
    PATTERNS,
    RELATIONSHIP_TYPES,
    JamaKGPipelineConfig,
    create_extraction_template,
    create_jama_kg_pipeline,
    get_schema_for_pipeline,
    process_guide_with_pipeline,
)

# =============================================================================
# FETCHERS (always available)
# =============================================================================
from .fetcher import (
    Fetcher,
    FetcherConfig,
    HttpxFetcher,
    PlaywrightFetcher,
    create_fetcher,
)

# Graph building
from .graph import (
    ConstraintManager,
    SupplementaryGraphBuilder,
    create_all_constraints,
    create_article_relationships,
    create_chapter_structure,
    create_glossary_structure,
    create_resource_nodes,
    create_vector_index,
)

# Loaders
from .loaders import (
    ArticleIndex,
    JamaHTMLLoader,
    build_article_index,
)

# =============================================================================
# NEW PIPELINE MODULES (neo4j_graphrag based)
# =============================================================================
# Resource models
from .models.resource import (
    BaseResource,
    DefinitionResource,
    ExternalLinkResource,
    ResourceType,
)
from .models.resource import (
    ImageResource as ImageResourceModel,
)
from .models.resource import (
    VideoResource as VideoResourceModel,
)
from .models.resource import (
    WebinarResource as WebinarResourceModel,
)
from .models_core import (
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

# Post-processing
from .postprocessing import (
    INDUSTRY_TAXONOMY,
    EntityNormalizer,
    GlossaryConceptLinker,
    IndustryNormalizer,
    link_glossary_to_concepts,
    normalize_entity_name,
    normalize_industry,
)

# =============================================================================
# SCRAPER (always available)
# =============================================================================
from .scraper import JamaGuideScraper, run_scraper

# Validation
from .validation import (
    ValidationQueries,
    ValidationReport,
    ValidationReporter,
    generate_validation_report,
    run_all_validations,
)

__version__ = "0.1.0"

__all__ = [
    # ==========================================================================
    # CORE MODELS
    # ==========================================================================
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
    # ==========================================================================
    # SCRAPER
    # ==========================================================================
    "JamaGuideScraper",
    "run_scraper",
    # ==========================================================================
    # FETCHERS
    # ==========================================================================
    "Fetcher",
    "FetcherConfig",
    "HttpxFetcher",
    "PlaywrightFetcher",
    "create_fetcher",
    # ==========================================================================
    # NEW PIPELINE (neo4j_graphrag)
    # ==========================================================================
    # Resource models
    "ResourceType",
    "BaseResource",
    "ImageResourceModel",
    "VideoResourceModel",
    "WebinarResourceModel",
    "DefinitionResource",
    "ExternalLinkResource",
    # Chunking
    "HierarchicalChunkingConfig",
    "HierarchicalHTMLSplitter",
    "create_text_splitter_adapter",
    # Extraction
    "JamaKGPipelineConfig",
    "NODE_TYPES",
    "RELATIONSHIP_TYPES",
    "PATTERNS",
    "get_schema_for_pipeline",
    "create_extraction_template",
    "create_jama_kg_pipeline",
    "process_guide_with_pipeline",
    # Loaders
    "JamaHTMLLoader",
    "ArticleIndex",
    "build_article_index",
    # Post-processing
    "INDUSTRY_TAXONOMY",
    "IndustryNormalizer",
    "normalize_industry",
    "EntityNormalizer",
    "normalize_entity_name",
    "GlossaryConceptLinker",
    "link_glossary_to_concepts",
    # Graph building
    "SupplementaryGraphBuilder",
    "create_chapter_structure",
    "create_resource_nodes",
    "create_glossary_structure",
    "create_article_relationships",
    "ConstraintManager",
    "create_all_constraints",
    "create_vector_index",
    # Validation
    "ValidationQueries",
    "run_all_validations",
    "ValidationReport",
    "ValidationReporter",
    "generate_validation_report",
    # ==========================================================================
    # EXCEPTIONS
    # ==========================================================================
    "BrowserNotInstalledError",
    "FetchError",
    "Neo4jConfigError",
    "PlaywrightNotAvailableError",
    "ScraperError",
]
