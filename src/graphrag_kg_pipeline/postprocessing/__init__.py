"""Post-processing for entity normalization and consolidation.

This package provides:
- Industry taxonomy mapping (100+ variants → 18 canonical names)
- Entity name normalization and deduplication
- Entity cleanup (generic terms, plurals, mislabeled challenges)
- Glossary-to-concept linking
- MENTIONED_IN and APPLIES_TO relationship backfill
- Entity description summarization
"""

from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
    EntityCleanupNormalizer,
)
from graphrag_kg_pipeline.postprocessing.entity_summarizer import EntitySummarizer
from graphrag_kg_pipeline.postprocessing.glossary_linker import (
    GlossaryConceptLinker,
    link_glossary_to_concepts,
)
from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
    INDUSTRY_TAXONOMY,
    IndustryNormalizer,
    normalize_industry,
)
from graphrag_kg_pipeline.postprocessing.mentioned_in_backfill import (
    MentionedInBackfiller,
)
from graphrag_kg_pipeline.postprocessing.normalizer import (
    EntityNormalizer,
    normalize_entity_name,
)

__all__ = [
    # Industry
    "INDUSTRY_TAXONOMY",
    "IndustryNormalizer",
    "normalize_industry",
    # Entity normalization
    "EntityNormalizer",
    "normalize_entity_name",
    # Entity cleanup
    "EntityCleanupNormalizer",
    # Glossary
    "GlossaryConceptLinker",
    "link_glossary_to_concepts",
    # Backfill
    "MentionedInBackfiller",
    # Summarization
    "EntitySummarizer",
]
