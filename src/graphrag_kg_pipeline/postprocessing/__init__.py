"""Post-processing for entity normalization and consolidation.

This package provides:
- Industry taxonomy mapping (96→19 canonical names)
- Entity name normalization
- Entity cleanup (generic terms, plurals, mislabeled challenges)
- Glossary-to-concept linking
- MENTIONED_IN and APPLIES_TO relationship backfill
- Entity description summarization
"""

from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
    GENERIC_TERMS_TO_DELETE,
    LLM_EXTRACTED_ENTITY_LABELS,
    PLURAL_TO_SINGULAR,
    POSITIVE_OUTCOME_WORDS,
    EntityCleanupClassifier,
    EntityCleanupNormalizer,
    classify_entity_for_cleanup,
    is_generic_term,
    is_potentially_mislabeled_challenge,
    normalize_to_singular,
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
    STANDARD_INDUSTRY_MAP,
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
    # Entity cleanup
    "GENERIC_TERMS_TO_DELETE",
    "LLM_EXTRACTED_ENTITY_LABELS",
    "PLURAL_TO_SINGULAR",
    "POSITIVE_OUTCOME_WORDS",
    "EntityCleanupClassifier",
    "EntityCleanupNormalizer",
    "classify_entity_for_cleanup",
    "is_generic_term",
    "is_potentially_mislabeled_challenge",
    "normalize_to_singular",
    # Entity normalization
    "EntityNormalizer",
    "normalize_entity_name",
    # Glossary
    "GlossaryConceptLinker",
    "link_glossary_to_concepts",
    # Backfill
    "STANDARD_INDUSTRY_MAP",
    "MentionedInBackfiller",
    # Summarization
    "EntitySummarizer",
]
