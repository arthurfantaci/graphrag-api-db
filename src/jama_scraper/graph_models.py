"""Pydantic models for LangExtract entities and Neo4j graph export.

Designed for:
- LangExtract extraction results with source grounding
- Neo4j graph database schema
- GraphRAG chatbot consumption

These models complement the existing models.py, adding semantic
extraction capabilities without modifying the core Article model.
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, computed_field


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


class EntityType(str, Enum):
    """Types of entities to extract for the knowledge graph.

    Each type maps to a Neo4j node label and has specific attributes
    that are meaningful for that entity type.
    """

    CONCEPT = "concept"
    """Technical concepts and terminology (e.g., 'requirements traceability')."""

    METHODOLOGY = "methodology"
    """Development methodologies and frameworks (e.g., 'Agile', 'MBSE', 'V-Model')."""

    TOOL = "tool"
    """Software tools and platforms (e.g., 'Jama Connect', 'DOORS')."""

    STANDARD = "standard"
    """Industry standards and regulations (e.g., 'ISO 13485', 'DO-178C')."""

    CHALLENGE = "challenge"
    """Problems and challenges teams face (e.g., 'scope creep')."""

    BEST_PRACTICE = "best_practice"
    """Recommended practices and solutions (e.g., 'live traceability')."""

    ROLE = "role"
    """Job roles and stakeholders (e.g., 'product manager')."""

    INDUSTRY = "industry"
    """Industry verticals (e.g., 'medical device', 'aerospace')."""

    ARTIFACT = "artifact"
    """Documents and work products (e.g., 'SRS', 'PRD', 'RTM')."""

    PROCESS_STAGE = "process_stage"
    """Development lifecycle stages (e.g., 'elicitation', 'verification')."""


class RelationshipType(str, Enum):
    """Types of relationships between entities.

    These map directly to Neo4j relationship types for graph traversal.
    """

    DEFINES = "DEFINES"
    """The text provides a definition for a concept."""

    MENTIONS = "MENTIONS"
    """Entity is mentioned in the article context."""

    PREREQUISITE_FOR = "PREREQUISITE_FOR"
    """Concept A must be understood/done before B."""

    COMPONENT_OF = "COMPONENT_OF"
    """Part-whole relationships (e.g., 'verification is part of V&V')."""

    ADDRESSES = "ADDRESSES"
    """A practice/tool addresses a challenge."""

    DEMONSTRATES = "DEMONSTRATES"
    """Example demonstrates a practice or concept."""

    REQUIRES = "REQUIRES"
    """A standard/process requires something."""

    RELATED_TO = "RELATED_TO"
    """General semantic relationship."""

    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    """Comparing alternatives (e.g., 'live vs after-the-fact traceability')."""

    USED_BY = "USED_BY"
    """Tool/practice used by an industry or role."""

    APPLIES_TO = "APPLIES_TO"
    """Standard applies to an industry."""

    PRODUCES = "PRODUCES"
    """Process produces an artifact."""


class CharInterval(BaseModel):
    """Character interval for source grounding.

    LangExtract provides exact character positions where entities were
    found in the source text, enabling precise attribution and
    visualization.
    """

    start_pos: int = Field(description="Start character position in source text")
    end_pos: int = Field(description="End character position in source text")

    @computed_field
    @property
    def length(self) -> int:
        """Length of the interval in characters."""
        return self.end_pos - self.start_pos


class ExtractedEntity(BaseModel):
    """An entity extracted by LangExtract with source grounding.

    Each entity has a stable ID generated from its type and canonical name,
    enabling deduplication across articles while preserving source attribution.
    """

    # Core identification
    entity_id: str = Field(description="Unique entity ID (e.g., 'concept-abc123')")
    entity_type: EntityType = Field(description="Type classification of the entity")
    name: str = Field(description="Canonical name of the entity")

    # Source grounding (from LangExtract)
    source_text: str = Field(description="Exact text from source document")
    char_interval: CharInterval | None = Field(
        default=None,
        description="Character positions in source article markdown",
    )
    source_article_id: str = Field(
        description="Article where entity was extracted (e.g., 'ch1-art3')"
    )

    # Semantic attributes (flexible, type-dependent)
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Type-specific attributes. Examples by type:\n"
            "- concept: 'definition', 'synonyms'\n"
            "- tool: 'vendor', 'category'\n"
            "- standard: 'organization', 'domain'\n"
            "- challenge: 'impact', 'symptoms'\n"
            "- best_practice: 'benefit', 'context'"
        ),
    )

    # Extraction metadata
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (1.0 = exact match)",
    )
    alignment_status: str = Field(
        default="exact",
        description="LangExtract alignment: 'exact', 'fuzzy', 'no_match'",
    )
    extracted_at: datetime = Field(default_factory=_utc_now)


class ExtractedRelationship(BaseModel):
    """A relationship between two entities.

    Relationships are directional and include evidence text from the
    source article that supports the relationship.
    """

    relationship_id: str = Field(description="Unique relationship ID")
    relationship_type: RelationshipType = Field(
        description="Type of relationship (maps to Neo4j relationship type)"
    )

    # Source and target entities
    source_entity_id: str = Field(description="Source entity ID (the 'from' node)")
    target_entity_id: str = Field(description="Target entity ID (the 'to' node)")

    # Source grounding
    source_text: str | None = Field(
        default=None,
        description="Text passage that evidences this relationship",
    )
    source_article_id: str = Field(
        description="Article where relationship was inferred"
    )

    # Relationship attributes
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Relationship-specific attributes (e.g., 'strength', 'context')",
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the relationship",
    )
    extracted_at: datetime = Field(default_factory=_utc_now)


class ArticleEnrichment(BaseModel):
    """Enrichment data for a single article.

    Contains all LLM-extracted content for one article, including
    the summary (which populates the empty Article.summary field),
    entities, and relationships.
    """

    article_id: str = Field(
        description="Reference to original article (e.g., 'ch1-art3')"
    )

    # LLM-generated summary (fills empty Article.summary)
    summary: str = Field(description="LLM-generated article summary (2-3 sentences)")

    # Extracted entities and relationships
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Entities extracted from this article",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="Relationships inferred from this article",
    )

    # Processing metadata
    model_used: str = Field(
        description="LLM model used for extraction (e.g., 'gpt-4o')"
    )
    extraction_passes: int = Field(
        default=1,
        description="Number of LangExtract extraction passes",
    )
    processing_time_seconds: float = Field(
        default=0.0,
        description="Time taken to process this article",
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token counts: {'input': N, 'output': M}",
    )

    @computed_field
    @property
    def entity_count(self) -> int:
        """Number of entities extracted from this article."""
        return len(self.entities)

    @computed_field
    @property
    def relationship_count(self) -> int:
        """Number of relationships extracted from this article."""
        return len(self.relationships)


class GlossaryEnrichment(BaseModel):
    """Enrichment for glossary terms.

    Populates the previously-empty related_terms and related_chapters
    fields in GlossaryTerm by using LLM inference.
    """

    term: str = Field(description="Original glossary term")

    # Inferred relationships (were always empty in original model)
    related_terms: list[str] = Field(
        default_factory=list,
        description="Semantically related glossary terms",
    )
    related_chapters: list[int] = Field(
        default_factory=list,
        description="Chapters where this term is most relevant",
    )
    related_entity_ids: list[str] = Field(
        default_factory=list,
        description="Extracted entities that relate to this term",
    )


class EnrichedGuide(BaseModel):
    """The complete enriched guide with entities and relationships.

    This is the top-level model for LangExtract output, containing:
    - Per-article enrichments (summaries, entities, relationships)
    - Global deduplicated entity registry
    - All relationships across the guide
    - Glossary enrichments
    """

    # Enrichments keyed by article_id
    article_enrichments: dict[str, ArticleEnrichment] = Field(
        default_factory=dict,
        description="Enrichment data keyed by article_id",
    )

    # Glossary enrichments
    glossary_enrichments: list[GlossaryEnrichment] = Field(
        default_factory=list,
        description="Enriched glossary terms",
    )

    # Global deduplicated entity registry
    entities: dict[str, ExtractedEntity] = Field(
        default_factory=dict,
        description="Global entity registry keyed by entity_id (deduplicated)",
    )

    # All relationships across the guide
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="All relationships from all articles",
    )

    # Extraction metadata
    extraction_config: dict = Field(
        default_factory=dict,
        description="Configuration used for extraction",
    )
    total_processing_time_seconds: float = Field(
        default=0.0,
        description="Total processing time for all articles",
    )
    total_token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Aggregate token usage: {'input': N, 'output': M}",
    )
    extracted_at: datetime = Field(default_factory=_utc_now)

    @computed_field
    @property
    def entity_count(self) -> int:
        """Total unique entities across all articles."""
        return len(self.entities)

    @computed_field
    @property
    def relationship_count(self) -> int:
        """Total relationships across all articles."""
        return len(self.relationships)

    @computed_field
    @property
    def article_count(self) -> int:
        """Number of articles enriched."""
        return len(self.article_enrichments)

    def get_entities_by_type(self, entity_type: EntityType) -> list[ExtractedEntity]:
        """Get all entities of a specific type.

        Args:
            entity_type: The type of entities to filter.

        Returns:
            List of entities matching the specified type.
        """
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_relationships_by_type(
        self, relationship_type: RelationshipType
    ) -> list[ExtractedRelationship]:
        """Get all relationships of a specific type.

        Args:
            relationship_type: The type of relationships to filter.

        Returns:
            List of relationships matching the specified type.
        """
        return [
            r for r in self.relationships if r.relationship_type == relationship_type
        ]

    def get_entity_relationships(
        self, entity_id: str
    ) -> tuple[list[ExtractedRelationship], list[ExtractedRelationship]]:
        """Get all relationships involving an entity.

        Args:
            entity_id: The entity ID to find relationships for.

        Returns:
            Tuple of (outgoing_relationships, incoming_relationships).
        """
        outgoing = [r for r in self.relationships if r.source_entity_id == entity_id]
        incoming = [r for r in self.relationships if r.target_entity_id == entity_id]
        return outgoing, incoming
