"""LangExtract augmentation for entity source grounding.

Runs Google's LangExtract as a post-processing step to discover missed entities
and add source grounding (exact character spans) to existing ones. Uses the
same OPENAI_API_KEY already required by the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import tenacity

from graphrag_kg_pipeline.utils.retry import langextract_retry

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)

# Map our schema node types to LangExtract extraction classes
_EXTRACTION_CLASSES: list[str] = [
    "Concept",
    "Challenge",
    "Artifact",
    "Bestpractice",
    "Processstage",
    "Role",
    "Standard",
    "Tool",
    "Methodology",
    "Industry",
    "Organization",
    "Outcome",
]


class LangExtractAugmenter:
    """Post-extraction augmentation with source grounding via LangExtract.

    Queries Neo4j for chunk text, runs LangExtract entity extraction, then
    MERGEs new entities into the graph and attaches source spans to existing
    entities for provenance tracking.

    Attributes:
        driver: Async Neo4j driver.
        database: Neo4j database name.
        openai_api_key: OpenAI API key for LangExtract.
        model: LLM model name (default: gpt-4o).
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str = "neo4j",
        openai_api_key: str = "",
        model: str = "gpt-4o",
    ) -> None:
        """Initialize the LangExtract augmenter.

        Args:
            driver: Async Neo4j driver.
            database: Neo4j database name.
            openai_api_key: OpenAI API key for LangExtract.
            model: LLM model to use.
        """
        self.driver = driver
        self.database = database
        self.openai_api_key = openai_api_key
        self.model = model
        self.new_count = 0
        self.grounded_count = 0

    async def augment(self, *, max_chunks: int = 0) -> dict[str, int]:
        """Run LangExtract augmentation on all chunks in the graph.

        Fetches chunks from Neo4j, extracts entities with LangExtract,
        then merges new entities and attaches source spans.

        Args:
            max_chunks: Maximum chunks to process (0 = all).

        Returns:
            Statistics dict with new_entities and grounded_entities counts.
        """
        import langextract as lx

        chunks = await self._get_chunks(max_chunks)
        if not chunks:
            logger.info("No chunks found for LangExtract augmentation")
            return {"new_entities": 0, "grounded_entities": 0}

        logger.info("Starting LangExtract augmentation", chunk_count=len(chunks))

        # Build prompt and examples for requirements management domain
        prompt = (
            "Extract entities from this requirements management text. "
            "Entity types: Concept, Challenge, Artifact, Bestpractice, "
            "Processstage, Role, Standard, Tool, Methodology, Industry, "
            "Organization, Outcome. "
            "Use exact text from the input. Extract in order of appearance."
        )
        examples = self._build_examples()

        # Concatenate chunk texts into a single document for extraction
        # (LangExtract handles long texts by splitting internally)
        for chunk in chunks:
            chunk_id = chunk["id"]
            chunk_text = chunk["text"]

            if not chunk_text or len(chunk_text.strip()) < 50:
                continue

            try:
                result = self._extract_with_retry(
                    lx=lx,
                    chunk_text=chunk_text,
                    prompt=prompt,
                    examples=examples,
                )

                for entity in result.extractions:
                    if entity.extraction_class not in _EXTRACTION_CLASSES:
                        continue

                    normalized_name = entity.extraction_text.lower().strip()
                    if len(normalized_name) < 2:
                        continue

                    # Build source span info
                    source_span = None
                    if entity.char_interval:
                        source_span = (
                            f"{entity.char_interval.start_pos}-{entity.char_interval.end_pos}"
                        )

                    exists = await self._entity_exists(normalized_name)
                    if not exists:
                        await self._create_entity(
                            name=normalized_name,
                            label=entity.extraction_class,
                            source_span=source_span,
                            chunk_id=chunk_id,
                        )
                        self.new_count += 1
                    elif source_span:
                        await self._add_source_span(
                            name=normalized_name,
                            source_span=source_span,
                            chunk_id=chunk_id,
                        )
                        self.grounded_count += 1

            except tenacity.RetryError:
                logger.warning(
                    "API rate limit exhausted after retries for LangExtract",
                    chunk_id=chunk_id,
                    exc_info=True,
                )
            except Exception:
                logger.warning(
                    "LangExtract failed for chunk, skipping",
                    chunk_id=chunk_id,
                    exc_info=True,
                )

        logger.info(
            "LangExtract augmentation complete",
            new_entities=self.new_count,
            grounded_entities=self.grounded_count,
        )

        return {
            "new_entities": self.new_count,
            "grounded_entities": self.grounded_count,
        }

    @langextract_retry
    def _extract_with_retry(
        self, *, lx: Any, chunk_text: str, prompt: str, examples: list[Any]
    ) -> Any:
        """Call langextract with retry on wrapped rate limit errors.

        langextract wraps OpenAI errors in ``InferenceRuntimeError(original=e)``,
        so we use a custom predicate to unwrap and match retryable errors.

        Args:
            lx: The langextract module.
            chunk_text: Text to extract entities from.
            prompt: Extraction prompt description.
            examples: Few-shot examples.

        Returns:
            The langextract extraction result.
        """
        return lx.extract(
            text_or_documents=chunk_text,
            prompt_description=prompt,
            examples=examples,
            model_id=self.model,
            api_key=self.openai_api_key,
            fence_output=True,
            use_schema_constraints=False,
        )

    def _build_examples(self) -> list[Any]:
        """Build few-shot examples for LangExtract.

        Returns:
            List of ExampleData for the requirements management domain.
        """
        import langextract as lx

        return [
            lx.data.ExampleData(
                text=(
                    "Requirements traceability is essential for ISO 26262 compliance "
                    "in the automotive industry. Teams use Jama Connect to manage "
                    "bidirectional traceability matrices."
                ),
                extractions=[
                    lx.data.Extraction(
                        extraction_class="Concept",
                        extraction_text="Requirements traceability",
                    ),
                    lx.data.Extraction(
                        extraction_class="Standard",
                        extraction_text="ISO 26262",
                    ),
                    lx.data.Extraction(
                        extraction_class="Industry",
                        extraction_text="automotive",
                    ),
                    lx.data.Extraction(
                        extraction_class="Tool",
                        extraction_text="Jama Connect",
                    ),
                    lx.data.Extraction(
                        extraction_class="Artifact",
                        extraction_text="bidirectional traceability matrices",
                    ),
                ],
            )
        ]

    async def _get_chunks(self, max_chunks: int = 0) -> list[dict[str, str]]:
        """Fetch chunk nodes from Neo4j.

        Args:
            max_chunks: Maximum chunks to return (0 = all).

        Returns:
            List of dicts with 'id' and 'text' keys.
        """
        limit_clause = f" LIMIT {max_chunks}" if max_chunks > 0 else ""
        query = f"""
            MATCH (c:Chunk)
            WHERE c.text IS NOT NULL AND size(c.text) > 50
            RETURN elementId(c) AS id, c.text AS text
            ORDER BY c.index
            {limit_clause}
        """
        chunks = []
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            async for record in result:
                chunks.append({"id": record["id"], "text": record["text"]})
        return chunks

    async def _entity_exists(self, name: str) -> bool:
        """Check if an entity with this normalized name already exists.

        Args:
            name: Lowercase normalized entity name.

        Returns:
            True if entity exists in the graph.
        """
        query = """
            MATCH (n)
            WHERE n.name = $name
              AND any(lbl IN labels(n) WHERE lbl IN $labels)
            RETURN count(n) > 0 AS exists
        """
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, name=name, labels=_EXTRACTION_CLASSES)
            record = await result.single()
            return record["exists"] if record else False

    async def _create_entity(
        self,
        name: str,
        label: str,
        source_span: str | None,
        chunk_id: str,
    ) -> None:
        """Create a new entity node with optional source span.

        Args:
            name: Normalized entity name.
            label: Entity type label.
            source_span: Character span in source text.
            chunk_id: Element ID of the source chunk.
        """
        props: dict[str, Any] = {
            "name": name,
            "display_name": name.title(),
            "source": "langextract",
        }
        if source_span:
            props["source_span"] = source_span

        query = f"""
            MERGE (n:{label} {{name: $name}})
            ON CREATE SET n += $props, n:__Entity__:__KGBuilder__
            WITH n
            MATCH (c:Chunk) WHERE elementId(c) = $chunk_id
            MERGE (n)-[:MENTIONED_IN]->(c)
        """
        async with self.driver.session(database=self.database) as session:
            await session.run(query, name=name, props=props, chunk_id=chunk_id)

    async def _add_source_span(
        self,
        name: str,
        source_span: str,
        chunk_id: str,
    ) -> None:
        """Add source grounding metadata to an existing entity.

        Appends the source span to a list property and ensures a
        MENTIONED_IN relationship to the source chunk.

        Args:
            name: Normalized entity name.
            source_span: Character span in source text.
            chunk_id: Element ID of the source chunk.
        """
        query = """
            MATCH (n)
            WHERE n.name = $name
              AND any(lbl IN labels(n) WHERE lbl IN $labels)
            SET n.source_spans = coalesce(n.source_spans, []) + $span
            WITH n
            MATCH (c:Chunk) WHERE elementId(c) = $chunk_id
            MERGE (n)-[:MENTIONED_IN]->(c)
        """
        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                name=name,
                labels=_EXTRACTION_CLASSES,
                span=source_span,
                chunk_id=chunk_id,
            )
