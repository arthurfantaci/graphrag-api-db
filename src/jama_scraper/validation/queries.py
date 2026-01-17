"""Validation queries for knowledge graph quality.

This module provides Cypher queries to validate the knowledge graph
after loading, checking for orphans, duplicates, and missing data.
"""

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import Driver

logger = structlog.get_logger(__name__)


class ValidationQueries:
    """Collection of validation queries for the knowledge graph.

    Provides queries to check for common data quality issues:
    - Orphan nodes (disconnected from the graph)
    - Duplicate entities
    - Missing required properties
    - Invalid relationship patterns

    Example:
        >>> queries = ValidationQueries(driver)
        >>> orphans = await queries.find_orphan_chunks()
        >>> print(f"Found {orphans} orphan chunks")
    """

    def __init__(self, driver: "Driver", database: str = "neo4j") -> None:
        """Initialize with Neo4j driver.

        Args:
            driver: Neo4j driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database

    async def find_orphan_chunks(self) -> int:
        """Find chunks not connected to any article.

        Returns:
            Count of orphan chunks.
        """
        query = """
        MATCH (c:Chunk)
        WHERE NOT (c)-[:FROM_ARTICLE]->()
        RETURN count(c) AS orphan_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["orphan_count"] if record else 0

    async def find_orphan_entities(self) -> list[dict]:
        """Find entities not connected to any chunk or article.

        Returns:
            List of orphan entity details.
        """
        query = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
        AND NOT (n)--()
        RETURN labels(n)[0] AS label, n.name AS name, elementId(n) AS element_id
        LIMIT 100
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_duplicate_entities(self) -> list[dict]:
        """Find entities with duplicate names within the same label.

        Returns:
            List of duplicate entity groups.
        """
        query = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
        WITH labels(n)[0] AS label, n.name AS name, count(n) AS cnt
        WHERE cnt > 1
        RETURN label, name, cnt
        ORDER BY cnt DESC
        LIMIT 50
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_missing_embeddings(self) -> int:
        """Find chunks without embeddings.

        Returns:
            Count of chunks missing embeddings.
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NULL
        RETURN count(c) AS missing_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["missing_count"] if record else 0

    async def count_industries(self) -> int:
        """Count total Industry nodes.

        Target is ≤19 after consolidation.

        Returns:
            Industry node count.
        """
        query = """
        MATCH (i:Industry)
        RETURN count(i) AS industry_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["industry_count"] if record else 0

    async def get_entity_stats(self) -> dict[str, int]:
        """Get counts for each entity type.

        Returns:
            Mapping from label to count.
        """
        query = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Article', 'Chunk', 'Chapter', 'Image', 'Video', 'Webinar', 'Definition'])
        WITH labels(n)[0] AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return {record["label"]: record["count"] async for record in result}

    async def find_invalid_patterns(self) -> list[dict]:
        """Find relationships that don't match defined patterns.

        Uses the actual PATTERNS from the extraction schema to ensure
        we don't report false positives.

        Returns:
            List of invalid relationship patterns.
        """
        # Import the actual valid patterns from schema
        from jama_scraper.extraction.schema import PATTERNS

        # Build exclusion condition from all valid patterns
        exclusions = []
        for source, rel, target in PATTERNS:
            exclusions.append(
                f"(labels(a)[0] = '{source}' AND type(r) = '{rel}' AND labels(b)[0] = '{target}')"
            )

        exclusion_clause = " OR ".join(exclusions)

        query = f"""
        MATCH (a)-[r]->(b)
        WHERE any(label IN labels(a) WHERE label IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
        AND any(label IN labels(b) WHERE label IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
        AND NOT ({exclusion_clause})
        RETURN labels(a)[0] AS source_label, type(r) AS rel_type,
               labels(b)[0] AS target_label, count(*) AS count
        ORDER BY count DESC
        LIMIT 20
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def check_article_coverage(self) -> dict:
        """Check that all expected articles exist.

        Returns:
            Coverage statistics.
        """
        query = """
        MATCH (a:Article)
        WITH count(a) AS total_articles,
             count(DISTINCT a.chapter_number) AS chapters_with_articles
        RETURN total_articles, chapters_with_articles
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                return {
                    "total_articles": record["total_articles"],
                    "chapters_with_articles": record["chapters_with_articles"],
                }
            return {"total_articles": 0, "chapters_with_articles": 0}


async def run_all_validations(
    driver: "Driver",
    database: str = "neo4j",
) -> dict[str, Any]:
    """Run all validation queries and return results.

    Args:
        driver: Neo4j driver.
        database: Database name.

    Returns:
        Dictionary of all validation results.
    """
    queries = ValidationQueries(driver, database)

    results = {
        "orphan_chunks": await queries.find_orphan_chunks(),
        "orphan_entities": await queries.find_orphan_entities(),
        "duplicate_entities": await queries.find_duplicate_entities(),
        "missing_embeddings": await queries.find_missing_embeddings(),
        "industry_count": await queries.count_industries(),
        "entity_stats": await queries.get_entity_stats(),
        "invalid_patterns": await queries.find_invalid_patterns(),
        "article_coverage": await queries.check_article_coverage(),
    }

    # Compute summary
    results["summary"] = {
        "has_orphan_chunks": results["orphan_chunks"] > 0,
        "has_orphan_entities": len(results["orphan_entities"]) > 0,
        "has_duplicates": len(results["duplicate_entities"]) > 0,
        "has_missing_embeddings": results["missing_embeddings"] > 0,
        "industry_count_ok": results["industry_count"] <= 19,
        "has_invalid_patterns": len(results["invalid_patterns"]) > 0,
    }

    # Overall status
    results["validation_passed"] = all([
        not results["summary"]["has_orphan_chunks"],
        not results["summary"]["has_duplicates"],
        results["summary"]["industry_count_ok"],
    ])

    logger.info(
        "Validation complete",
        passed=results["validation_passed"],
        orphan_chunks=results["orphan_chunks"],
        duplicates=len(results["duplicate_entities"]),
        industries=results["industry_count"],
    )

    return results
