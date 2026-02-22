"""Validation queries for knowledge graph quality.

This module provides Cypher queries to validate the knowledge graph
after loading, checking for orphans, duplicates, and missing data.
"""

from typing import TYPE_CHECKING, Any

import structlog

from graphrag_kg_pipeline.extraction.schema import LLM_EXTRACTED_ENTITY_LABELS

if TYPE_CHECKING:
    from neo4j import AsyncDriver, Driver

    # Accept either sync or async driver
    AnyDriver = Driver | AsyncDriver

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

    def __init__(self, driver: "AnyDriver", database: str = "neo4j") -> None:
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
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Organization', 'Outcome'])
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
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Organization', 'Outcome'])
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
        from graphrag_kg_pipeline.extraction.schema import PATTERNS

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
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Organization', 'Outcome'])
        AND any(label IN labels(b) WHERE label IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Organization', 'Outcome'])
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

    async def find_missing_chunk_ids(self) -> int:
        """Find chunks without chunk_id property.

        Returns:
            Count of chunks missing chunk_id.
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.chunk_id IS NULL
        RETURN count(c) AS missing_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["missing_count"] if record else 0

    async def find_plural_singular_duplicates(
        self,
        labels: list[str] | None = None,
    ) -> list[dict]:
        """Find entity pairs that differ only by plural suffix.

        Detects cases like "requirement" vs "requirements" that should
        be merged. Only checks LLM-extracted entity labels by default.

        Args:
            labels: Entity labels to check. Defaults to LLM_EXTRACTED_ENTITY_LABELS.

        Returns:
            List of plural/singular pairs with relationship counts.
        """
        labels = labels or LLM_EXTRACTED_ENTITY_LABELS
        label_list = "[" + ", ".join(f"'{lbl}'" for lbl in labels) + "]"

        query = f"""
        MATCH (singular)
        WHERE any(label IN labels(singular) WHERE label IN {label_list})
        AND singular.name IS NOT NULL
        AND NOT singular.name ENDS WITH 's'

        MATCH (plural)
        WHERE any(label IN labels(plural) WHERE label IN {label_list})
        AND plural.name = singular.name + 's'
        AND labels(singular)[0] = labels(plural)[0]

        // Count relationships for impact assessment
        OPTIONAL MATCH (singular)-[r1]-()
        OPTIONAL MATCH (plural)-[r2]-()

        WITH singular, plural,
             labels(singular)[0] AS label,
             count(DISTINCT r1) AS singular_rels,
             count(DISTINCT r2) AS plural_rels
        RETURN label,
               singular.name AS singular_name,
               plural.name AS plural_name,
               singular_rels,
               plural_rels,
               elementId(singular) AS singular_id,
               elementId(plural) AS plural_id
        ORDER BY label, singular_name
        LIMIT 100
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_generic_entities(
        self,
        generic_terms: set[str] | None = None,
        labels: list[str] | None = None,
    ) -> list[dict]:
        """Find overly generic entity names that should be removed.

        Args:
            generic_terms: Set of generic term names to find.
                If None, uses a default set of common generic terms.
            labels: Entity labels to check. Defaults to LLM_EXTRACTED_ENTITY_LABELS.

        Returns:
            List of generic entities with relationship counts.
        """
        # Default generic terms if none provided
        if generic_terms is None:
            generic_terms = {
                "tool",
                "tools",
                "software",
                "solution",
                "solutions",
                "platform",
                "system",
                "systems",
                "method",
                "methods",
                "process",
                "processes",
                "approach",
                "technique",
                "techniques",
                "document",
                "documents",
            }

        labels = labels or LLM_EXTRACTED_ENTITY_LABELS
        label_list = "[" + ", ".join(f"'{lbl}'" for lbl in labels) + "]"
        term_list = "[" + ", ".join(f"'{term}'" for term in generic_terms) + "]"

        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN {label_list})
        AND toLower(n.name) IN {term_list}

        // Count relationships for impact assessment
        OPTIONAL MATCH (n)-[r]-()
        WITH n, labels(n)[0] AS label, count(DISTINCT r) AS relationship_count
        RETURN label,
               n.name AS name,
               relationship_count,
               elementId(n) AS element_id
        ORDER BY relationship_count DESC, label, name
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def get_entity_relationship_counts(
        self,
        names: list[str],
        label: str,
    ) -> dict[str, int]:
        """Get relationship counts for specific entities.

        Useful for assessing impact before deletion or merge.

        Args:
            names: List of entity names to check.
            label: Entity label to filter by.

        Returns:
            Mapping from entity name to relationship count.
        """
        name_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"

        query = f"""
        MATCH (n:{label})
        WHERE n.name IN {name_list}
        OPTIONAL MATCH (n)-[r]-()
        WITH n.name AS name, count(DISTINCT r) AS relationship_count
        RETURN name, relationship_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return {record["name"]: record["relationship_count"] async for record in result}

    async def find_missing_chunk_index(self) -> int:
        """Find chunks where index is NULL.

        Returns:
            Count of chunks missing index property.
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.index IS NULL
        RETURN count(c) AS missing_count
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["missing_count"] if record else 0

    async def find_degenerate_chunks(self, min_length: int = 100) -> list[dict]:
        """Find chunks with very short text and no entity relationships.

        Args:
            min_length: Minimum text length threshold.

        Returns:
            List of degenerate chunk details.
        """
        query = """
        MATCH (c:Chunk)
        WHERE c.text IS NOT NULL AND size(c.text) < $min_length
        AND NOT EXISTS { MATCH ()-[:MENTIONED_IN]->(c) }
        RETURN elementId(c) AS element_id,
               c.text AS text,
               size(c.text) AS text_length
        ORDER BY text_length
        LIMIT 50
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, min_length=min_length)
            return [dict(record) async for record in result]

    async def find_truncated_webinar_titles(self) -> list[dict]:
        """Find webinar nodes with truncated or missing titles.

        Returns:
            List of webinars with short or missing titles.
        """
        query = """
        MATCH (w:Webinar)
        WHERE w.title IS NULL OR size(w.title) < 15 OR w.title = 'Webinar'
        RETURN elementId(w) AS element_id,
               w.title AS title,
               w.url AS url
        LIMIT 50
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_entities_without_mentioned_in(
        self,
        labels: list[str] | None = None,
    ) -> list[dict]:
        """Find entities with no MENTIONED_IN relationship to any chunk.

        Args:
            labels: Entity labels to check. Defaults to LLM_EXTRACTED_ENTITY_LABELS.

        Returns:
            List of entities without MENTIONED_IN relationships.
        """
        labels = labels or LLM_EXTRACTED_ENTITY_LABELS
        label_list = "[" + ", ".join(f"'{lbl}'" for lbl in labels) + "]"

        query = f"""
        MATCH (e)
        WHERE any(lbl IN labels(e) WHERE lbl IN {label_list})
        AND NOT EXISTS {{ MATCH (e)-[:MENTIONED_IN]->(:Chunk) }}
        RETURN labels(e)[0] AS label, e.name AS name, elementId(e) AS element_id
        ORDER BY label, name
        LIMIT 100
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_entities_without_semantic_relationships(self) -> list[dict]:
        """Find entities that have MENTIONED_IN but no outbound semantic relationships.

        These are "ghost" entities that appear in chunks but have no
        connections to other entities in the knowledge graph.

        Returns:
            List of entities with only MENTIONED_IN relationships.
        """
        query = """
        MATCH (e)-[:MENTIONED_IN]->(:Chunk)
        WHERE any(lbl IN labels(e) WHERE lbl IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Organization', 'Outcome'])
        AND NOT EXISTS {
            MATCH (e)-[r]->()
            WHERE type(r) <> 'MENTIONED_IN'
        }
        AND NOT EXISTS {
            MATCH ()-[r]->(e)
            WHERE type(r) <> 'MENTIONED_IN'
        }
        RETURN labels(e)[0] AS label, e.name AS name, elementId(e) AS element_id
        ORDER BY label, name
        LIMIT 100
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_potentially_mislabeled_entities(self) -> list[dict]:
        """Find Challenge nodes with names suggesting positive outcomes.

        Returns:
            List of potentially mislabeled Challenge entities.
        """
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            POSITIVE_OUTCOME_WORDS,
        )

        # Build Cypher list of positive outcome words
        word_list = "[" + ", ".join(f"'{w}'" for w in POSITIVE_OUTCOME_WORDS) + "]"

        query = f"""
        MATCH (c:Challenge)
        WHERE c.name IS NOT NULL
        WITH c, split(toLower(c.name), ' ') AS words
        WHERE any(word IN words WHERE word IN {word_list})
        RETURN c.name AS name, elementId(c) AS element_id
        ORDER BY c.name
        LIMIT 50
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_near_duplicate_entities(self) -> list[dict]:
        """Find entities where one name is a substring of another.

        Returns:
            List of near-duplicate entity pairs.
        """
        query = """
        MATCH (a), (b)
        WHERE any(lbl IN labels(a) WHERE lbl IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Organization', 'Outcome'])
        AND labels(a)[0] = labels(b)[0]
        AND a.name IS NOT NULL AND b.name IS NOT NULL
        AND a.name <> b.name
        AND size(a.name) > 4
        AND b.name CONTAINS a.name
        AND size(b.name) - size(a.name) <= 5
        AND elementId(a) < elementId(b)
        RETURN labels(a)[0] AS label,
               a.name AS shorter_name,
               b.name AS longer_name
        ORDER BY label, a.name
        LIMIT 50
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def find_missing_definitions(self) -> list[dict]:
        """Find entities without definition property, grouped by label.

        Returns:
            List of label/count pairs for entities missing definitions.
        """
        query = """
        MATCH (e)
        WHERE any(lbl IN labels(e) WHERE lbl IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry',
             'Organization', 'Outcome'])
        AND (e.definition IS NULL OR e.definition = '')
        WITH labels(e)[0] AS label, count(e) AS count
        RETURN label, count
        ORDER BY count DESC
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def get_chunk_article_mapping(self, limit: int = 100) -> list[dict]:
        """Get chunk to article mapping for chunk_id generation.

        Returns chunks with their article_id and index for generating
        chunk_id values.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of chunk/article mappings.
        """
        query = """
        MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article)
        WHERE c.chunk_id IS NULL
        RETURN elementId(c) AS chunk_element_id,
               a.article_id AS article_id,
               c.index AS chunk_index
        ORDER BY a.article_id, c.index
        LIMIT $limit
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, limit=limit)
            return [dict(record) async for record in result]


async def run_all_validations(
    driver: "AnyDriver",
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
        # Chunk quality checks
        "missing_chunk_ids": await queries.find_missing_chunk_ids(),
        "missing_chunk_index": await queries.find_missing_chunk_index(),
        "degenerate_chunks": await queries.find_degenerate_chunks(),
        # Entity quality checks
        "plural_singular_duplicates": await queries.find_plural_singular_duplicates(),
        "generic_entities": await queries.find_generic_entities(),
        "entities_without_mentioned_in": await queries.find_entities_without_mentioned_in(),
        "entities_without_semantic_rels": await queries.find_entities_without_semantic_relationships(),
        "potentially_mislabeled": await queries.find_potentially_mislabeled_entities(),
        "near_duplicates": await queries.find_near_duplicate_entities(),
        "missing_definitions": await queries.find_missing_definitions(),
        # Webinar quality
        "truncated_webinar_titles": await queries.find_truncated_webinar_titles(),
    }

    # Compute summary
    results["summary"] = {
        "has_orphan_chunks": results["orphan_chunks"] > 0,
        "has_orphan_entities": len(results["orphan_entities"]) > 0,
        "has_duplicates": len(results["duplicate_entities"]) > 0,
        "has_missing_embeddings": results["missing_embeddings"] > 0,
        "industry_count_ok": results["industry_count"] <= 19,
        "has_invalid_patterns": len(results["invalid_patterns"]) > 0,
        # Chunk summary flags
        "has_missing_chunk_ids": results["missing_chunk_ids"] > 0,
        "has_missing_chunk_index": results["missing_chunk_index"] > 0,
        "has_degenerate_chunks": len(results["degenerate_chunks"]) > 0,
        # Entity summary flags
        "has_plural_duplicates": len(results["plural_singular_duplicates"]) > 0,
        "has_generic_entities": len(results["generic_entities"]) > 0,
        "has_entities_without_mentioned_in": len(results["entities_without_mentioned_in"]) > 0,
        "has_entities_without_semantic_rels": len(results["entities_without_semantic_rels"]) > 0,
        "has_potentially_mislabeled": len(results["potentially_mislabeled"]) > 0,
        "has_near_duplicates": len(results["near_duplicates"]) > 0,
        "has_missing_definitions": len(results["missing_definitions"]) > 0,
        "has_truncated_webinar_titles": len(results["truncated_webinar_titles"]) > 0,
    }

    # Overall status — pass/fail includes chunk_index (critical) + existing checks
    # All other new checks are advisory only
    results["validation_passed"] = all(
        [
            not results["summary"]["has_orphan_chunks"],
            not results["summary"]["has_duplicates"],
            results["summary"]["industry_count_ok"],
            not results["summary"]["has_missing_chunk_ids"],
            not results["summary"]["has_missing_chunk_index"],
            not results["summary"]["has_plural_duplicates"],
        ]
    )

    logger.info(
        "Validation complete",
        passed=results["validation_passed"],
        orphan_chunks=results["orphan_chunks"],
        duplicates=len(results["duplicate_entities"]),
        industries=results["industry_count"],
        missing_chunk_ids=results["missing_chunk_ids"],
        missing_chunk_index=results["missing_chunk_index"],
        plural_duplicates=len(results["plural_singular_duplicates"]),
        generic_entities=len(results["generic_entities"]),
    )

    return results
