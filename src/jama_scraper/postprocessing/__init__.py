"""Post-processing for entity normalization and consolidation.

This package provides:
- Industry taxonomy mapping (96→19 canonical names)
- Entity name normalization
- Glossary-to-concept linking
"""

from jama_scraper.postprocessing.glossary_linker import (
    GlossaryConceptLinker,
    link_glossary_to_concepts,
)
from jama_scraper.postprocessing.industry_taxonomy import (
    INDUSTRY_TAXONOMY,
    IndustryNormalizer,
    normalize_industry,
)
from jama_scraper.postprocessing.normalizer import (
    EntityNormalizer,
    normalize_entity_name,
)

__all__ = [
    # Industry
    "INDUSTRY_TAXONOMY",
    "IndustryNormalizer",
    "normalize_industry",
    # Entity
    "EntityNormalizer",
    "normalize_entity_name",
    # Glossary
    "GlossaryConceptLinker",
    "link_glossary_to_concepts",
]
