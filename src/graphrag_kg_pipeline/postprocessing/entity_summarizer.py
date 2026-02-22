"""Entity description summarization for knowledge graph quality.

When the same entity is extracted from multiple chunks, each extraction
produces a separate description fragment. This module consolidates
multiple descriptions into a single coherent summary using an LLM.

Based on Essential GraphRAG Ch7 §7.2.3 (pp 96-99).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)

SUMMARIZATION_PROMPT = """You are given multiple description fragments for the same entity
in a knowledge graph about requirements management.

Entity: {entity_name} (type: {entity_label})

Description fragments:
{descriptions}

Synthesize these fragments into a single coherent description (1-3 sentences).
Preserve all unique information. Do not add information not present in the fragments.
Return ONLY the consolidated description text, nothing else."""


class EntitySummarizer:
    """Consolidates multiple entity descriptions into coherent summaries.

    After extraction from multiple chunks, entities may accumulate
    fragmented descriptions. This class queries for entities with
    multiple description fragments and uses an LLM to consolidate them.

    Attributes:
        driver: Neo4j async driver.
        database: Neo4j database name.
        openai_api_key: OpenAI API key.
        model: LLM model for summarization.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str = "neo4j",
        openai_api_key: str = "",
        model: str = "gpt-4o",
    ) -> None:
        """Initialize the summarizer.

        Args:
            driver: Neo4j async driver.
            database: Database name.
            openai_api_key: OpenAI API key.
            model: LLM model for summarization (default: gpt-4o for quality).
        """
        self.driver = driver
        self.database = database
        self.openai_api_key = openai_api_key
        self.model = model

    async def find_entities_with_fragments(self) -> list[dict[str, Any]]:
        """Find entities that have description stored as a JSON array.

        The neo4j_graphrag pipeline may store descriptions as arrays when
        multiple chunks contribute descriptions. This finds those entities.

        Returns:
            List of entities with fragmented descriptions.
        """
        query = """
        MATCH (e)
        WHERE e.description IS NOT NULL
        AND e.name IS NOT NULL
        AND any(lbl IN labels(e) WHERE lbl IN [
            'Concept', 'Challenge', 'Artifact', 'Bestpractice',
            'Processstage', 'Role', 'Standard', 'Tool',
            'Methodology', 'Industry', 'Organization', 'Outcome'
        ])
        WITH e, labels(e)[0] AS label, e.description AS desc
        WHERE size(desc) > 200
        RETURN elementId(e) AS element_id,
               e.name AS name,
               label,
               desc AS description
        ORDER BY size(desc) DESC
        LIMIT 100
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            records = []
            async for record in result:
                records.append(
                    {
                        "element_id": record["element_id"],
                        "name": record["name"],
                        "label": record["label"],
                        "description": record["description"],
                    }
                )
            return records

    async def summarize(self) -> dict[str, Any]:
        """Summarize fragmented entity descriptions.

        Returns:
            Statistics about summarization operations.
        """
        stats: dict[str, int] = {
            "entities_found": 0,
            "entities_summarized": 0,
            "errors": 0,
        }

        entities = await self.find_entities_with_fragments()
        stats["entities_found"] = len(entities)

        if not entities:
            logger.info("No entities with fragmented descriptions found")
            return stats

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        for entity in entities:
            try:
                # Parse description fragments
                desc = entity["description"]
                fragments = self._parse_fragments(desc)

                if len(fragments) < 2:
                    continue

                # Build summarization prompt
                descriptions_text = "\n".join(f"- {frag}" for frag in fragments)
                prompt = SUMMARIZATION_PROMPT.format(
                    entity_name=entity["name"],
                    entity_label=entity["label"],
                    descriptions=descriptions_text,
                )

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=300,
                )

                summary = (response.choices[0].message.content or "").strip()

                if summary:
                    await self._update_description(
                        element_id=entity["element_id"],
                        summary=summary,
                    )
                    stats["entities_summarized"] += 1

            except Exception:
                stats["errors"] += 1
                logger.warning(
                    "Failed to summarize entity",
                    entity_name=entity["name"],
                    exc_info=True,
                )

        logger.info(
            "Entity summarization complete",
            found=stats["entities_found"],
            summarized=stats["entities_summarized"],
        )

        return stats

    def _parse_fragments(self, description: str) -> list[str]:
        """Parse description into fragments.

        Handles both JSON arrays and plain text with delimiters.

        Args:
            description: Raw description string (may be JSON array or text).

        Returns:
            List of description fragments.
        """
        # Try JSON array first
        try:
            parsed = json.loads(description)
            if isinstance(parsed, list):
                return [str(f).strip() for f in parsed if str(f).strip()]
        except (json.JSONDecodeError, TypeError):
            pass

        # Split on common delimiters
        if " | " in description:
            return [f.strip() for f in description.split(" | ") if f.strip()]

        # Single description — no fragments to merge
        return [description]

    async def _update_description(
        self,
        element_id: str,
        summary: str,
    ) -> None:
        """Update an entity's description with the consolidated summary.

        Args:
            element_id: Neo4j element ID of the entity.
            summary: Consolidated description text.
        """
        query = """
        MATCH (e) WHERE elementId(e) = $element_id
        SET e.description = $summary
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(query, element_id=element_id, summary=summary)
