"""Graph building and constraint management for Neo4j.

This package provides:
- Supplementary graph structure (Chapter, Resource nodes)
- Database constraints and indexes
"""

from graphrag_kg_pipeline.graph.constraints import (
    ConstraintManager,
    create_all_constraints,
    create_vector_index,
)
from graphrag_kg_pipeline.graph.supplementary import (
    SupplementaryGraphBuilder,
    create_article_relationships,
    create_chapter_structure,
    create_glossary_structure,
    create_resource_nodes,
)

__all__ = [
    # Supplementary
    "SupplementaryGraphBuilder",
    "create_chapter_structure",
    "create_resource_nodes",
    "create_glossary_structure",
    "create_article_relationships",
    # Constraints
    "ConstraintManager",
    "create_all_constraints",
    "create_vector_index",
]
