"""Validation utilities for the knowledge graph.

This package provides:
- Cypher validation queries for graph quality checks
- Report generation for validation results
- Data repair utilities for common issues
"""

from graphrag_kg_pipeline.validation.fixes import (
    ValidationFixer,
)
from graphrag_kg_pipeline.validation.queries import (
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
    "ValidationQueries",
    "run_all_validations",
    # Reporter
    "ValidationReport",
    "ValidationReporter",
    "generate_validation_report",
    # Fixes
    "ValidationFixer",
]
