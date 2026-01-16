"""Pydantic models for the Jama Guide ETL pipeline.

This package contains:
- Core content models (article, chapter, glossary) - from parent models.py
- Resource models (images, videos, webinars, definitions, external links)
"""

# Re-export core models from the legacy models.py (in parent directory)
# This allows both `from jama_scraper.models import Article` and
# `from jama_scraper import Article` to work
# New resource models
from jama_scraper.models.resource import (
    BaseResource,
    DefinitionResource,
    ExternalLinkResource,
    ImageResource,
    ResourceType,
    VideoResource,
    WebinarResource,
)
from jama_scraper.models_core import (
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

__all__ = [
    # Core models
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
    # Resource types
    "ResourceType",
    "BaseResource",
    "ImageResource",
    "VideoResource",
    "WebinarResource",
    "DefinitionResource",
    "ExternalLinkResource",
]
