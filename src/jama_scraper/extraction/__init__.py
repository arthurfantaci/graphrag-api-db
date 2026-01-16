"""Entity and relationship extraction for the Jama Guide knowledge graph.

This package provides:
- Schema definitions for node types and relationship types
- Domain-specific extraction prompts
- SimpleKGPipeline factory and configuration
"""

from jama_scraper.extraction.pipeline import (
    JamaKGPipelineConfig,
    create_jama_kg_pipeline,
    process_guide_with_pipeline,
)
from jama_scraper.extraction.prompts import (
    REQUIREMENTS_DOMAIN_INSTRUCTIONS,
    create_extraction_template,
)
from jama_scraper.extraction.schema import (
    NODE_TYPES,
    PATTERNS,
    RELATIONSHIP_TYPES,
    get_schema_for_pipeline,
)

__all__ = [
    # Schema
    "NODE_TYPES",
    "RELATIONSHIP_TYPES",
    "PATTERNS",
    "get_schema_for_pipeline",
    # Prompts
    "REQUIREMENTS_DOMAIN_INSTRUCTIONS",
    "create_extraction_template",
    # Pipeline
    "JamaKGPipelineConfig",
    "create_jama_kg_pipeline",
    "process_guide_with_pipeline",
]
