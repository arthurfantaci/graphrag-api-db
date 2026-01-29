"""Validation utilities for the knowledge graph.

This package provides:
- Cypher validation queries
- Report generation for graph quality
- Data repair utilities for common issues
"""

from graphrag_kg_pipeline.validation.fixes import (
    ValidationFixer,
    fix_generic_entities,
    fix_missing_chunk_ids,
    fix_plural_entities,
    format_fix_preview,
)
from graphrag_kg_pipeline.validation.queries import (
    LLM_EXTRACTED_ENTITY_LABELS,
    ValidationQueries,
    run_all_validations,
)
from graphrag_kg_pipeline.validation.reporter import (
    ValidationReport,
    ValidationReporter,
    generate_validation_report,
)

__all__ = [
    # Queries
    "LLM_EXTRACTED_ENTITY_LABELS",
    "ValidationQueries",
    "run_all_validations",
    # Reporter
    "ValidationReport",
    "ValidationReporter",
    "generate_validation_report",
    # Fixes
    "ValidationFixer",
    "fix_missing_chunk_ids",
    "fix_plural_entities",
    "fix_generic_entities",
    "format_fix_preview",
]
