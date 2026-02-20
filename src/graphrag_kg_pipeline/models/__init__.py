"""Pydantic models for the Jama Guide ETL pipeline.

This package contains:
- Core content models (article, chapter, glossary)
- Resource models (images, videos, webinars, definitions, external links)
"""

from graphrag_kg_pipeline.models.content import (
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
from graphrag_kg_pipeline.models.resource import (
    BaseResource,
    DefinitionResource,
    ExternalLinkResource,
    ImageResource,
    ResourceType,
    VideoResource,
    WebinarResource,
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
