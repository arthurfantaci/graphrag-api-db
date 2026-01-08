"""LangExtract-based semantic extraction for requirements management content.

Provides entity and relationship extraction with:
- Configurable LLM backends (OpenAI, Gemini, Ollama)
- Batch processing with checkpointing for resume capability
- Entity deduplication and merging across articles
- Source grounding preservation from LangExtract

Example:
    config = EnrichmentConfig(provider=LLMProvider.OPENAI)
    extractor = JamaExtractor(config)
    enriched = await extractor.enrich_guide(guide)
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .extraction_schemas import (
    ENTITY_EXTRACTION_PROMPT,
    RELATIONSHIP_EXTRACTION_PROMPT,
    SUMMARY_PROMPT,
    get_entity_examples,
    get_relationship_examples,
    get_summary_examples,
)
from .graph_models import (
    ArticleEnrichment,
    CharInterval,
    EnrichedGuide,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    GlossaryEnrichment,
    RelationshipType,
)

if TYPE_CHECKING:
    from .enrichment_config import EnrichmentConfig
    from .models import Article, Glossary, RequirementsManagementGuide

console = Console()


class LangExtractNotAvailableError(Exception):
    """LangExtract package is not installed."""

    def __init__(self) -> None:
        """Initialize the error with installation instructions."""
        super().__init__(
            "LangExtract not installed. Install with: uv sync --group enrichment"
        )


class JamaExtractor:
    """Extract entities and relationships from scraped guide content.

    Uses LangExtract to perform semantic extraction with LLMs, extracting
    domain-specific entities (concepts, methodologies, tools, etc.) and
    relationships between them for Neo4j graph construction.

    Features:
    - Checkpoint-based resume capability
    - Entity deduplication across articles
    - Parallel batch processing
    - Rich progress display

    Attributes:
        config: Enrichment configuration settings.
    """

    def __init__(self, config: EnrichmentConfig) -> None:
        """Initialize the extractor with configuration.

        Args:
            config: EnrichmentConfig with LLM provider settings.
        """
        self.config = config
        self._entity_cache: dict[str, ExtractedEntity] = {}
        self._checkpoint_dir = config.checkpoint_dir

        # Validate config before proceeding
        config.validate()

    async def enrich_guide(
        self,
        guide: RequirementsManagementGuide,
        resume: bool = True,
    ) -> EnrichedGuide:
        """Process all articles and extract entities/relationships.

        Args:
            guide: The scraped RequirementsManagementGuide to enrich.
            resume: If True, resume from checkpoint if available.

        Returns:
            EnrichedGuide containing all extracted entities and relationships.
        """
        # Ensure checkpoint directory exists
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing progress if resuming
        completed_ids: set[str] = set()
        enrichments: dict[str, ArticleEnrichment] = {}

        if resume:
            completed_ids, enrichments = self._load_checkpoints()
            if completed_ids:
                console.print(
                    f"[cyan]Resuming from checkpoint: "
                    f"{len(completed_ids)} articles already processed[/]"
                )

        # Get articles to process
        all_articles = [art for ch in guide.chapters for art in ch.articles]
        pending_articles = [
            art for art in all_articles if art.article_id not in completed_ids
        ]

        console.print(
            f"[cyan]Enriching {len(pending_articles)} articles "
            f"(of {len(all_articles)} total)...[/]"
        )
        console.print(
            f"[dim]Using {self.config.provider.value} / "
            f"{self.config.effective_model_id}[/]"
        )

        start_time = time.monotonic()

        # Process articles with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Extracting entities...",
                total=len(pending_articles),
            )

            for article in pending_articles:
                progress.update(
                    task,
                    description=f"Processing: {article.title[:40]}...",
                )

                enrichment = self._extract_article(article)
                enrichments[article.article_id] = enrichment

                # Merge entities into global cache
                for entity in enrichment.entities:
                    self._merge_entity(entity)

                # Save checkpoint
                self._save_checkpoint(article.article_id, enrichment)

                progress.advance(task)

        # Process glossary enrichments
        glossary_enrichments = []
        if guide.glossary:
            glossary_enrichments = self._enrich_glossary(guide.glossary)

        # Build final EnrichedGuide
        total_time = time.monotonic() - start_time

        enriched = EnrichedGuide(
            article_enrichments=enrichments,
            glossary_enrichments=glossary_enrichments,
            entities=self._entity_cache,
            relationships=self._collect_relationships(enrichments),
            extraction_config={
                "provider": self.config.provider.value,
                "model_id": self.config.effective_model_id,
                "extraction_passes": self.config.extraction_passes,
            },
            total_processing_time_seconds=total_time,
        )

        console.print("\n[green]Enrichment complete![/]")
        console.print(f"  Articles processed: {len(enrichments)}")
        console.print(f"  Unique entities: {enriched.entity_count}")
        console.print(f"  Relationships: {enriched.relationship_count}")
        console.print(f"  Processing time: {total_time:.1f}s")

        return enriched

    def _extract_article(self, article: Article) -> ArticleEnrichment:
        """Extract entities and relationships from a single article.

        Args:
            article: Article to process.

        Returns:
            ArticleEnrichment with extracted data.
        """
        import langextract as lx

        start_time = time.monotonic()
        params = self.config.to_langextract_params()

        # 1. Generate summary
        summary = self._generate_summary(article.markdown_content, params)

        # 2. Extract entities
        entity_result = lx.extract(
            text_or_documents=article.markdown_content,
            prompt_description=ENTITY_EXTRACTION_PROMPT,
            examples=get_entity_examples(),
            **params,
        )
        entities = self._convert_entities(entity_result, article.article_id)

        # 3. Extract relationships (with reduced params for efficiency)
        rel_params = {**params, "extraction_passes": 1, "max_char_buffer": 2000}
        relationship_result = lx.extract(
            text_or_documents=article.markdown_content,
            prompt_description=RELATIONSHIP_EXTRACTION_PROMPT,
            examples=get_relationship_examples(),
            **rel_params,
        )
        relationships = self._convert_relationships(
            relationship_result,
            article.article_id,
            entities,
        )

        processing_time = time.monotonic() - start_time

        return ArticleEnrichment(
            article_id=article.article_id,
            summary=summary,
            entities=entities,
            relationships=relationships,
            model_used=self.config.effective_model_id,
            extraction_passes=self.config.extraction_passes,
            processing_time_seconds=processing_time,
        )

    def _generate_summary(self, content: str, params: dict) -> str:
        """Generate article summary using LangExtract.

        Args:
            content: Article markdown content.
            params: LangExtract parameters.

        Returns:
            Generated summary string.
        """
        import langextract as lx

        # Use single pass for summary
        summary_params = {**params, "extraction_passes": 1}

        result = lx.extract(
            text_or_documents=content[:8000],  # Limit context for summary
            prompt_description=SUMMARY_PROMPT,
            examples=get_summary_examples(),
            **summary_params,
        )

        # Extract summary from result
        for extraction in result.extractions:
            if extraction.extraction_class == "summary":
                return extraction.extraction_text

        # Fallback: use first 200 chars of content
        return content[:200].strip() + "..."

    def _convert_entities(
        self,
        result: object,
        article_id: str,
    ) -> list[ExtractedEntity]:
        """Convert LangExtract results to ExtractedEntity objects.

        Args:
            result: LangExtract extraction result.
            article_id: Source article ID.

        Returns:
            List of ExtractedEntity objects.
        """
        entities = []

        for extraction in result.extractions:
            # Map extraction class to EntityType
            try:
                entity_type = EntityType(extraction.extraction_class.lower())
            except ValueError:
                continue  # Skip unknown entity types

            # Generate stable entity ID
            entity_id = self._generate_entity_id(
                entity_type.value,
                extraction.extraction_text,
            )

            # Build char interval if available
            char_interval = None
            if hasattr(extraction, "char_interval") and extraction.char_interval:
                char_interval = CharInterval(
                    start_pos=extraction.char_interval.start_pos,
                    end_pos=extraction.char_interval.end_pos,
                )

            # Determine confidence based on alignment status
            confidence = 1.0
            alignment_status = "exact"
            if hasattr(extraction, "alignment_status") and extraction.alignment_status:
                alignment_status = str(extraction.alignment_status.value)
                if "fuzzy" in alignment_status.lower():
                    confidence = 0.8
                elif "no_match" in alignment_status.lower():
                    confidence = 0.5

            # Convert attribute values to strings (LangExtract may return lists)
            raw_attrs = extraction.attributes or {}
            str_attrs = {
                k: ", ".join(v) if isinstance(v, list) else str(v)
                for k, v in raw_attrs.items()
            }

            entity = ExtractedEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=extraction.extraction_text,
                source_text=extraction.extraction_text,
                char_interval=char_interval,
                source_article_id=article_id,
                attributes=str_attrs,
                confidence=confidence,
                alignment_status=alignment_status,
            )

            entities.append(entity)

        return entities

    def _convert_relationships(
        self,
        result: object,
        article_id: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelationship]:
        """Convert LangExtract relationship results.

        Args:
            result: LangExtract extraction result.
            article_id: Source article ID.
            entities: Entities extracted from the same article.

        Returns:
            List of ExtractedRelationship objects.
        """
        relationships = []
        entity_map = {e.name.lower(): e for e in entities}

        for extraction in result.extractions:
            # Map extraction class to RelationshipType
            try:
                rel_type = RelationshipType(extraction.extraction_class.upper())
            except ValueError:
                continue  # Skip unknown relationship types

            # Get source and target from attributes
            attrs = extraction.attributes or {}
            source_name = attrs.get("source", "")
            target_name = attrs.get("target", "")

            if not source_name or not target_name:
                continue

            # Find matching entities
            source_entity = entity_map.get(source_name.lower())
            target_entity = entity_map.get(target_name.lower())

            # Generate entity IDs even if not in local entities
            # (they might be in another article)
            source_id = (
                source_entity.entity_id
                if source_entity
                else self._generate_entity_id("concept", source_name)
            )
            target_id = (
                target_entity.entity_id
                if target_entity
                else self._generate_entity_id("concept", target_name)
            )

            rel_hash = hashlib.md5(  # noqa: S324
                f"{source_id}-{rel_type.value}-{target_id}".encode()
            ).hexdigest()[:12]
            rel_id = f"rel-{rel_hash}"

            relationship = ExtractedRelationship(
                relationship_id=rel_id,
                relationship_type=rel_type,
                source_entity_id=source_id,
                target_entity_id=target_id,
                source_text=extraction.extraction_text,
                source_article_id=article_id,
                confidence=1.0,
            )

            relationships.append(relationship)

        return relationships

    def _generate_entity_id(self, entity_type: str, name: str) -> str:
        """Generate stable entity ID from type and name.

        Uses MD5 hash of normalized name for consistent IDs across runs.

        Args:
            entity_type: Type of entity.
            name: Entity name.

        Returns:
            Stable entity ID string.
        """
        normalized = name.lower().strip()
        hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:8]  # noqa: S324
        return f"{entity_type}-{hash_suffix}"

    def _merge_entity(self, entity: ExtractedEntity) -> None:
        """Merge entity into global cache, handling duplicates.

        If an entity with the same ID exists, updates attributes
        while keeping the first occurrence's core data.

        Args:
            entity: Entity to merge into cache.
        """
        if entity.entity_id not in self._entity_cache:
            self._entity_cache[entity.entity_id] = entity
        else:
            # Merge attributes from new occurrence
            existing = self._entity_cache[entity.entity_id]
            merged_attrs = {**existing.attributes, **entity.attributes}
            # Create new entity with merged attributes
            self._entity_cache[entity.entity_id] = ExtractedEntity(
                entity_id=existing.entity_id,
                entity_type=existing.entity_type,
                name=existing.name,
                source_text=existing.source_text,
                char_interval=existing.char_interval,
                source_article_id=existing.source_article_id,
                attributes=merged_attrs,
                confidence=max(existing.confidence, entity.confidence),
                alignment_status=existing.alignment_status,
            )

    def _collect_relationships(
        self,
        enrichments: dict[str, ArticleEnrichment],
    ) -> list[ExtractedRelationship]:
        """Collect all relationships from enrichments.

        Args:
            enrichments: Dictionary of article enrichments.

        Returns:
            Deduplicated list of all relationships.
        """
        seen_ids: set[str] = set()
        relationships: list[ExtractedRelationship] = []

        for enrichment in enrichments.values():
            for rel in enrichment.relationships:
                if rel.relationship_id not in seen_ids:
                    seen_ids.add(rel.relationship_id)
                    relationships.append(rel)

        return relationships

    def _enrich_glossary(self, glossary: Glossary) -> list[GlossaryEnrichment]:
        """Enrich glossary terms with related entities and chapters.

        Uses extracted entities to find connections between glossary
        terms and the knowledge graph.

        Args:
            glossary: Glossary to enrich.

        Returns:
            List of GlossaryEnrichment objects.
        """
        enrichments = []

        for term in glossary.terms:
            # Find entities that mention this term
            related_entity_ids = []
            term_lower = term.term.lower()

            for entity_id, entity in self._entity_cache.items():
                if (
                    term_lower in entity.name.lower()
                    or term_lower in entity.source_text.lower()
                ):
                    related_entity_ids.append(entity_id)

            # Find related glossary terms (simple string matching)
            related_terms = []
            for other_term in glossary.terms:
                if other_term.term != term.term and (
                    term_lower in other_term.definition.lower()
                    or other_term.term.lower() in term.definition.lower()
                ):
                    related_terms.append(other_term.term)

            # Determine relevant chapters from entity sources
            related_chapters: set[int] = set()
            for entity_id in related_entity_ids:
                entity = self._entity_cache.get(entity_id)
                if entity:
                    # Extract chapter number from article_id (e.g., "ch5-art2")
                    try:
                        ch_num = int(entity.source_article_id.split("-")[0][2:])
                        related_chapters.add(ch_num)
                    except (IndexError, ValueError):
                        pass

            enrichments.append(
                GlossaryEnrichment(
                    term=term.term,
                    related_terms=related_terms[:5],  # Limit to top 5
                    related_chapters=sorted(related_chapters)[:5],
                    related_entity_ids=related_entity_ids[:10],
                )
            )

        return enrichments

    def _save_checkpoint(
        self,
        article_id: str,
        enrichment: ArticleEnrichment,
    ) -> None:
        """Save enrichment checkpoint for resume capability.

        Args:
            article_id: Article ID being checkpointed.
            enrichment: Enrichment data to save.
        """
        checkpoint_file = self._checkpoint_dir / f"{article_id}.json"
        checkpoint_file.write_text(
            enrichment.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _load_checkpoints(
        self,
    ) -> tuple[set[str], dict[str, ArticleEnrichment]]:
        """Load existing checkpoints for resume.

        Returns:
            Tuple of (completed article IDs, enrichment dictionary).
        """
        completed_ids: set[str] = set()
        enrichments: dict[str, ArticleEnrichment] = {}

        if not self._checkpoint_dir.exists():
            return completed_ids, enrichments

        for checkpoint_file in self._checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                enrichment = ArticleEnrichment.model_validate(data)
                enrichments[enrichment.article_id] = enrichment
                completed_ids.add(enrichment.article_id)

                # Rebuild entity cache from checkpoint
                for entity in enrichment.entities:
                    self._merge_entity(entity)

            except (json.JSONDecodeError, ValueError) as e:
                console.print(
                    f"[yellow]Warning: Invalid checkpoint {checkpoint_file}: {e}[/]"
                )

        return completed_ids, enrichments

    def clear_checkpoints(self) -> int:
        """Clear all checkpoint files.

        Returns:
            Number of checkpoint files removed.
        """
        if not self._checkpoint_dir.exists():
            return 0

        count = 0
        for checkpoint_file in self._checkpoint_dir.glob("*.json"):
            checkpoint_file.unlink()
            count += 1

        return count


def check_langextract_available() -> bool:
    """Check if LangExtract is installed and available.

    Returns:
        True if LangExtract can be imported, False otherwise.
    """
    try:
        import langextract  # noqa: F401
    except ImportError:
        return False
    else:
        return True
