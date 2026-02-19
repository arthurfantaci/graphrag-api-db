"""MENTIONED_IN and APPLIES_TO backfill for improved graph connectivity.

After LLM extraction, many Standard and Industry entities are missing
MENTIONED_IN relationships to chunks where they appear in text. This
module uses text matching with word-boundary awareness to backfill
those relationships. It also creates well-known Standard→Industry
APPLIES_TO relationships from domain knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)

# Minimum entity name length to avoid false-positive substring matches
_MIN_ENTITY_NAME_LENGTH = 4

# Well-known Standard → Industry relationships from domain knowledge
STANDARD_INDUSTRY_MAP: dict[str, str] = {
    "iso 26262": "automotive",
    "iso 21434": "automotive",
    "a-spice": "automotive",
    "do-178c": "aerospace",
    "do-254": "aerospace",
    "do-178b": "aerospace",
    "arp 4754a": "aerospace",
    "arp 4761": "aerospace",
    "iec 62304": "medical devices",
    "iso 13485": "medical devices",
    "iso 14971": "medical devices",
    "iec 61508": "industrial automation",
    "iec 61511": "industrial automation",
    "iso 15288": "systems engineering",
    "iso/iec 12207": "software development",
    "ieee 830": "software development",
    "ieee 29148": "software development",
    "en 50128": "rail",
    "en 50129": "rail",
    "ecss-e-st-40c": "space",
    "ecss-q-st-80c": "space",
}


class MentionedInBackfiller:
    """Backfills MENTIONED_IN relationships via text matching.

    After LLM extraction, scans chunk text for entity names using
    word-boundary matching. Only processes Standard and Industry
    labels to avoid false positives from generic terms.

    Attributes:
        driver: Neo4j async driver.
        database: Neo4j database name.
    """

    def __init__(self, driver: AsyncDriver, database: str = "neo4j") -> None:
        """Initialize the backfiller.

        Args:
            driver: Neo4j async driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database

    async def backfill_mentioned_in(self) -> dict[str, Any]:
        """Backfill MENTIONED_IN relationships for Standard and Industry entities.

        Uses word-boundary matching: wraps both entity name and chunk text
        with spaces, then checks containment. This prevents false positives
        like "rail" matching "traceability".

        Returns:
            Statistics about created relationships.
        """
        query = """
        MATCH (e)
        WHERE (e:Standard OR e:Industry)
        AND e.name IS NOT NULL
        AND size(e.name) >= $min_length
        WITH e, e.name AS entity_name

        MATCH (c:Chunk)
        WHERE c.text IS NOT NULL
        AND toLower(' ' + c.text + ' ') CONTAINS toLower(' ' + entity_name + ' ')
        AND NOT EXISTS { MATCH (e)-[:MENTIONED_IN]->(c) }

        MERGE (e)-[:MENTIONED_IN]->(c)
        RETURN count(*) AS created
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, min_length=_MIN_ENTITY_NAME_LENGTH)
            record = await result.single()
            created = record["created"] if record else 0

        logger.info("MENTIONED_IN backfill complete", relationships_created=created)
        return {"mentioned_in_created": created}

    async def backfill_applies_to(self) -> dict[str, Any]:
        """Create APPLIES_TO relationships from well-known Standard→Industry mappings.

        Uses MERGE to avoid duplicates. Creates Industry nodes if they
        don't exist (they should already exist from extraction).

        Returns:
            Statistics about created relationships.
        """
        created = 0
        for standard_name, industry_name in STANDARD_INDUSTRY_MAP.items():
            query = """
            MATCH (s:Standard {name: $standard_name})
            MERGE (i:Industry {name: $industry_name})
            MERGE (s)-[:APPLIES_TO]->(i)
            RETURN count(*) AS created
            """

            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    query,
                    standard_name=standard_name,
                    industry_name=industry_name,
                )
                record = await result.single()
                if record and record["created"] > 0:
                    created += 1

        logger.info("APPLIES_TO backfill complete", relationships_created=created)
        return {"applies_to_created": created}

    async def backfill(self) -> dict[str, Any]:
        """Run all backfill operations.

        Returns:
            Combined statistics from all backfill operations.
        """
        mentioned_stats = await self.backfill_mentioned_in()
        applies_stats = await self.backfill_applies_to()

        return {**mentioned_stats, **applies_stats}
