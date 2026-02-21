"""Multi-pass extraction gleaning for improved entity recall.

Implements the Microsoft GraphRAG gleaning concept: after initial extraction,
re-scans each chunk with already-extracted entities and asks the LLM to
identify anything that was missed. This catches 20-30% of entities that
single-pass extraction misses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)

GLEANING_PROMPT = """You previously extracted these entities and relationships from the text below:

{existing_entities}

Text:
{chunk_text}

Review the text again carefully. Are there additional entities or relationships
that were missed? Focus on:
- Definitions or descriptions of concepts not yet captured
- Standards, tools, or methodologies mentioned in passing
- Implicit relationships between already-extracted entities
- Industry sectors referenced but not explicitly extracted

Return ONLY the NEW entities and relationships not already listed above.
Return JSON with format: {{"nodes": [...], "relationships": [...]}}
If nothing was missed, return: {{"nodes": [], "relationships": []}}"""


class ExtractionGleaner:
    """Post-extraction gleaning to improve entity recall.

    After SimpleKGPipeline processes an article, queries Neo4j for
    chunks + their extracted entities, then runs a second LLM pass
    per chunk to catch missed entities/relationships.

    Attributes:
        driver: Neo4j async driver.
        database: Neo4j database name.
        openai_api_key: OpenAI API key.
        model: LLM model for gleaning.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str,
        openai_api_key: str,
        model: str = "gpt-4o",
    ) -> None:
        """Initialize the gleaner.

        Args:
            driver: Neo4j async driver.
            database: Database name.
            openai_api_key: OpenAI API key.
            model: LLM model for gleaning (default: same as primary extraction).
        """
        self.driver = driver
        self.database = database
        self.openai_api_key = openai_api_key
        self.model = model

    async def glean_article(self, article_id: str) -> dict[str, Any]:
        """Run gleaning pass for all chunks of an article.

        Args:
            article_id: The article ID to glean.

        Returns:
            Statistics about gleaned entities/relationships.
        """
        stats = {"chunks_processed": 0, "new_entities": 0, "new_relationships": 0, "errors": 0}

        # Query chunks and their linked entities for this article
        chunks_with_entities = await self._get_chunks_with_entities(article_id)

        if not chunks_with_entities:
            logger.info("No chunks found for gleaning", article_id=article_id)
            return stats

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        for chunk_data in chunks_with_entities:
            try:
                chunk_text = chunk_data["text"]
                existing = chunk_data["entities"]

                # Format existing entities for the prompt
                entities_str = json.dumps(existing, indent=2) if existing else "None extracted"

                prompt = GLEANING_PROMPT.format(
                    existing_entities=entities_str,
                    chunk_text=chunk_text,
                )

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content or "{}"
                # Strip markdown code fences if present (fallback)
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                result = json.loads(content)

                new_nodes = result.get("nodes", [])
                new_rels = result.get("relationships", [])

                if new_nodes or new_rels:
                    await self._merge_gleaned_results(
                        chunk_element_id=chunk_data["element_id"],
                        nodes=new_nodes,
                        relationships=new_rels,
                    )
                    stats["new_entities"] += len(new_nodes)
                    stats["new_relationships"] += len(new_rels)

                stats["chunks_processed"] += 1

            except (json.JSONDecodeError, KeyError):
                stats["errors"] += 1
                logger.warning(
                    "Failed to parse gleaning result",
                    article_id=article_id,
                    exc_info=True,
                )
            except Exception:
                stats["errors"] += 1
                logger.warning(
                    "Gleaning error for chunk",
                    article_id=article_id,
                    exc_info=True,
                )

        logger.info(
            "Gleaning complete",
            article_id=article_id,
            chunks=stats["chunks_processed"],
            new_entities=stats["new_entities"],
            new_relationships=stats["new_relationships"],
        )

        return stats

    async def _get_chunks_with_entities(self, article_id: str) -> list[dict[str, Any]]:
        """Query chunks and their linked entities for an article.

        Args:
            article_id: Article identifier.

        Returns:
            List of dicts with chunk text, element_id, and linked entities.
        """
        query = """
        MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article {article_id: $article_id})
        OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c)
        WITH c, collect(DISTINCT {
            name: e.name,
            label: labels(e)[0],
            definition: e.definition
        }) AS entities
        RETURN elementId(c) AS element_id,
               c.text AS text,
               entities
        ORDER BY c.index
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, article_id=article_id)
            records = []
            async for record in result:
                entities = [e for e in record["entities"] if e.get("name")]
                records.append(
                    {
                        "element_id": record["element_id"],
                        "text": record["text"],
                        "entities": entities,
                    }
                )
            return records

    async def _merge_gleaned_results(
        self,
        chunk_element_id: str,
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        """Merge gleaned entities and relationships into Neo4j.

        Uses MERGE (idempotent) to avoid duplicates.

        Args:
            chunk_element_id: Element ID of the source chunk.
            nodes: New entity nodes from gleaning.
            relationships: New relationships from gleaning.
        """
        from graphrag_kg_pipeline.extraction.schema import NODE_TYPES, RELATIONSHIP_TYPES

        allowed_labels = set(NODE_TYPES.keys())
        allowed_rel_types = set(RELATIONSHIP_TYPES.keys()) | {"MENTIONED_IN", "RELATED_TO"}

        for node in nodes:
            label = node.get("label", "Concept")
            name = node.get("name", "").lower().strip()
            if not name:
                continue

            # Validate label against schema to prevent Cypher injection
            if label not in allowed_labels:
                logger.warning(
                    "Skipping gleaned entity with invalid label",
                    name=name,
                    label=label,
                    allowed=list(allowed_labels),
                )
                continue

            # Build properties
            props = {"name": name}
            if node.get("display_name"):
                props["display_name"] = node["display_name"]
            if node.get("definition"):
                props["definition"] = node["definition"]

            # MERGE entity with __Entity__ + __KGBuilder__ labels (matching
            # neo4j_graphrag's label stack) and MENTIONED_IN relationship.
            # Without __Entity__, gleaned nodes are invisible to entity
            # resolution, cross-label dedup, and downstream queries.
            query = f"""
            MERGE (e:__Entity__:{label} {{name: $name}})
            ON CREATE SET e += $props, e:__KGBuilder__
            WITH e
            MATCH (c) WHERE elementId(c) = $chunk_id
            MERGE (e)-[:MENTIONED_IN]->(c)
            """

            async with self.driver.session(database=self.database) as session:
                await session.run(
                    query,
                    name=name,
                    props=props,
                    chunk_id=chunk_element_id,
                )

        # Merge relationships between entities
        for rel in relationships:
            source = rel.get("start_node_id") or rel.get("source", "")
            target = rel.get("end_node_id") or rel.get("target", "")
            rel_type = rel.get("type", "RELATED_TO")

            if not source or not target:
                continue

            # Validate relationship type against schema to prevent Cypher injection
            if rel_type not in allowed_rel_types:
                logger.warning(
                    "Skipping gleaned relationship with invalid type",
                    source=source,
                    target=target,
                    rel_type=rel_type,
                    allowed=list(allowed_rel_types),
                )
                continue

            query = f"""
            MATCH (a {{name: $source}})
            MATCH (b {{name: $target}})
            MERGE (a)-[:{rel_type}]->(b)
            """

            async with self.driver.session(database=self.database) as session:
                await session.run(query, source=source.lower(), target=target.lower())
