"""Entity and relationship extraction for the requirements guide knowledge graph.

This package provides:
- Schema definitions for node types and relationship types
- Domain-specific extraction prompts
- SimpleKGPipeline factory and configuration
"""

from graphrag_kg_pipeline.extraction.pipeline import (
    KGPipelineConfig,
    create_async_neo4j_driver,
    create_kg_pipeline,
    create_neo4j_driver,
    process_guide_with_pipeline,
)
from graphrag_kg_pipeline.extraction.prompts import (
    REQUIREMENTS_DOMAIN_INSTRUCTIONS,
    create_extraction_template,
)
from graphrag_kg_pipeline.extraction.schema import (
    LLM_EXTRACTED_ENTITY_LABELS,
    NODE_TYPES,
    PATTERNS,
    RELATIONSHIP_TYPES,
    get_schema_for_pipeline,
)

__all__ = [
    # Schema
    "LLM_EXTRACTED_ENTITY_LABELS",
    "NODE_TYPES",
    "RELATIONSHIP_TYPES",
    "PATTERNS",
    "get_schema_for_pipeline",
    # Prompts
    "REQUIREMENTS_DOMAIN_INSTRUCTIONS",
    "create_extraction_template",
    # Pipeline
    "KGPipelineConfig",
    "create_kg_pipeline",
    "create_neo4j_driver",
    "create_async_neo4j_driver",
    "process_guide_with_pipeline",
]
