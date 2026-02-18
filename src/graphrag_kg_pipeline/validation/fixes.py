"""Data repair utilities for knowledge graph validation.

This module provides fix functions for common data quality issues
detected by validation queries. Each fix can be previewed before
applying to understand the impact.

Supported fixes:
- fix_degenerate_chunks(): Delete short chunks with no entity relationships
- fix_missing_chunk_index(): Assign sequential index per article
- fix_missing_chunk_ids(): Generate chunk_id for all Chunk nodes
- fix_truncated_webinar_titles(): Extract title from description
- fix_mislabeled_entities(): Relabel Challenge nodes with positive-outcome names
- fix_missing_mentioned_in(): Backfill MENTIONED_IN relationships
- fix_missing_definitions(): Backfill definitions from glossary
- fix_plural_entities(): Merge plural variants into singulars
- fix_generic_entities(): Remove overly generic entity nodes
"""

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver, Driver

    # Accept either sync or async driver
    AnyDriver = Driver | AsyncDriver

logger = structlog.get_logger(__name__)

# Minimum text length for a chunk to be considered valid
MIN_CHUNK_TEXT_LENGTH = 100


async def fix_degenerate_chunks(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
    min_length: int = MIN_CHUNK_TEXT_LENGTH,
) -> dict[str, Any]:
    """Delete chunks with very short text and no entity relationships.

    Removes heading-only chunks and other degenerate fragments that
    provide no semantic value for retrieval.

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be deleted.
        min_length: Minimum text length threshold.

    Returns:
        Statistics about the fix operation.
    """
    stats: dict[str, Any] = {
        "total_found": 0,
        "deleted": 0,
        "dry_run": dry_run,
    }

    count_query = """
    MATCH (c:Chunk)
    WHERE c.text IS NOT NULL AND size(c.text) < $min_length
    AND NOT EXISTS { MATCH ()-[:MENTIONED_IN]->(c) }
    RETURN count(c) AS found_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(count_query, min_length=min_length)
        record = await result.single()
        stats["total_found"] = record["found_count"] if record else 0

    if stats["total_found"] == 0 or dry_run:
        return stats

    delete_query = """
    MATCH (c:Chunk)
    WHERE c.text IS NOT NULL AND size(c.text) < $min_length
    AND NOT EXISTS { MATCH ()-[:MENTIONED_IN]->(c) }
    DETACH DELETE c
    RETURN count(c) AS deleted_count
    """

    async with driver.session(database=database) as session:
        try:
            result = await session.run(delete_query, min_length=min_length)
            record = await result.single()
            stats["deleted"] = record["deleted_count"] if record else 0
            logger.info("Deleted degenerate chunks", count=stats["deleted"])
        except Exception as e:
            stats["errors"] = [f"Failed to delete degenerate chunks: {e}"]
            logger.exception("Failed to delete degenerate chunks")

    return stats


async def fix_missing_chunk_index(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Assign sequential index to chunks missing the index property.

    Orders chunks by elementId within each article and assigns
    sequential indices. Note: elementId ordering is best-effort —
    it correlates with insertion order but is not guaranteed.

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be fixed.

    Returns:
        Statistics about the fix operation.
    """
    stats: dict[str, Any] = {
        "total_missing": 0,
        "fixed": 0,
        "dry_run": dry_run,
    }

    count_query = """
    MATCH (c:Chunk)
    WHERE c.index IS NULL
    RETURN count(c) AS missing_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(count_query)
        record = await result.single()
        stats["total_missing"] = record["missing_count"] if record else 0

    if stats["total_missing"] == 0 or dry_run:
        return stats

    # Assign sequential index per article, ordered by elementId (best-effort)
    fix_query = """
    MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article)
    WHERE c.index IS NULL
    WITH a, c ORDER BY elementId(c)
    WITH a, collect(c) AS chunks
    UNWIND range(0, size(chunks) - 1) AS idx
    WITH chunks[idx] AS chunk, idx
    SET chunk.index = idx
    RETURN count(chunk) AS fixed_count
    """

    async with driver.session(database=database) as session:
        try:
            result = await session.run(fix_query)
            record = await result.single()
            stats["fixed"] = record["fixed_count"] if record else 0
            logger.info("Fixed missing chunk indices", count=stats["fixed"])
        except Exception as e:
            stats["errors"] = [f"Failed to fix chunk indices: {e}"]
            logger.exception("Failed to fix chunk indices")

    return stats


async def fix_truncated_webinar_titles(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fix webinar nodes with truncated or missing titles.

    Extracts the first sentence of the description as the title,
    or uses a cleaned URL slug as fallback.

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be fixed.

    Returns:
        Statistics about the fix operation.
    """
    stats: dict[str, Any] = {
        "total_found": 0,
        "fixed": 0,
        "dry_run": dry_run,
    }

    count_query = """
    MATCH (w:Webinar)
    WHERE w.title IS NULL OR size(w.title) < 15 OR w.title = 'Webinar'
    RETURN count(w) AS found_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(count_query)
        record = await result.single()
        stats["total_found"] = record["found_count"] if record else 0

    if stats["total_found"] == 0 or dry_run:
        return stats

    # Use description's first sentence as title, URL slug as fallback
    fix_query = """
    MATCH (w:Webinar)
    WHERE w.title IS NULL OR size(w.title) < 15 OR w.title = 'Webinar'
    WITH w,
         CASE
           WHEN w.description IS NOT NULL AND size(w.description) > 15
           THEN CASE
             WHEN w.description CONTAINS '.'
             THEN left(w.description, apoc.text.indexOf(w.description, '.') + 1)
             ELSE left(w.description, 100)
           END
           ELSE 'Webinar: ' + replace(
             last(split(coalesce(w.url, 'unknown'), '/')),
             '-', ' '
           )
         END AS new_title
    SET w.title = new_title
    RETURN count(w) AS fixed_count
    """

    async with driver.session(database=database) as session:
        try:
            result = await session.run(fix_query)
            record = await result.single()
            stats["fixed"] = record["fixed_count"] if record else 0
            logger.info("Fixed truncated webinar titles", count=stats["fixed"])
        except Exception as e:
            stats["errors"] = [f"Failed to fix webinar titles: {e}"]
            logger.exception("Failed to fix webinar titles")

    return stats


async def fix_mislabeled_entities(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Relabel Challenge nodes that have positive-outcome names.

    Moves entities like "High-Quality Products" from :Challenge to
    :Concept label. Checks for name collisions before relabeling.

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be fixed.

    Returns:
        Statistics about the fix operation.
    """
    from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
        POSITIVE_OUTCOME_WORDS,
    )

    stats: dict[str, Any] = {
        "total_found": 0,
        "relabeled": 0,
        "skipped_collision": 0,
        "dry_run": dry_run,
    }

    word_list = "[" + ", ".join(f"'{w}'" for w in POSITIVE_OUTCOME_WORDS) + "]"

    # Find mislabeled challenges
    find_query = f"""
    MATCH (c:Challenge)
    WHERE c.name IS NOT NULL
    WITH c, split(toLower(c.name), ' ') AS words
    WHERE any(word IN words WHERE word IN {word_list})
    RETURN elementId(c) AS element_id, c.name AS name
    """

    async with driver.session(database=database) as session:
        result = await session.run(find_query)
        entities = [dict(record) async for record in result]
        stats["total_found"] = len(entities)

    if stats["total_found"] == 0 or dry_run:
        return stats

    # Relabel each entity, checking for name collisions
    for entity in entities:
        relabel_query = """
        MATCH (c:Challenge) WHERE elementId(c) = $element_id
        // Check no Concept exists with same name
        WHERE NOT EXISTS {
            MATCH (existing:Concept {name: c.name})
        }
        REMOVE c:Challenge
        SET c:Concept
        RETURN count(c) AS relabeled
        """

        async with driver.session(database=database) as session:
            try:
                result = await session.run(
                    relabel_query, element_id=entity["element_id"]
                )
                record = await result.single()
                if record and record["relabeled"] > 0:
                    stats["relabeled"] += 1
                else:
                    stats["skipped_collision"] += 1
            except Exception:
                stats["skipped_collision"] += 1
                logger.warning(
                    "Failed to relabel entity",
                    entity_name=entity["name"],
                    exc_info=True,
                )

    logger.info(
        "Fixed mislabeled entities",
        relabeled=stats["relabeled"],
        skipped=stats["skipped_collision"],
    )
    return stats


async def fix_missing_mentioned_in(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill MENTIONED_IN relationships using text matching.

    Wraps the MentionedInBackfiller to create MENTIONED_IN relationships
    for Standard and Industry entities found in chunk text.

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be created.

    Returns:
        Statistics about the fix operation.
    """
    from graphrag_kg_pipeline.postprocessing.mentioned_in_backfill import (
        MentionedInBackfiller,
    )

    if dry_run:
        # Count entities without MENTIONED_IN as estimate
        count_query = """
        MATCH (e)
        WHERE any(lbl IN labels(e) WHERE lbl IN ['Standard', 'Industry'])
        AND NOT EXISTS { MATCH (e)-[:MENTIONED_IN]->(:Chunk) }
        RETURN count(e) AS estimate
        """
        async with driver.session(database=database) as session:
            result = await session.run(count_query)
            record = await result.single()
            estimate = record["estimate"] if record else 0

        return {
            "dry_run": True,
            "entities_without_mentioned_in": estimate,
        }

    backfiller = MentionedInBackfiller(driver, database)
    mentioned_in_stats = await backfiller.backfill_mentioned_in()
    applies_to_stats = await backfiller.backfill_applies_to()

    return {
        "dry_run": False,
        **mentioned_in_stats,
        **applies_to_stats,
    }


async def fix_missing_definitions(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill entity definitions from glossary Definition nodes.

    Matches entity names (case-insensitive) against Definition node
    terms and copies the definition text.

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be fixed.

    Returns:
        Statistics about the fix operation.
    """
    stats: dict[str, Any] = {
        "total_missing": 0,
        "backfilled": 0,
        "dry_run": dry_run,
    }

    count_query = """
    MATCH (e)
    WHERE any(lbl IN labels(e) WHERE lbl IN
        ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
         'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
    AND (e.definition IS NULL OR e.definition = '')
    RETURN count(e) AS missing_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(count_query)
        record = await result.single()
        stats["total_missing"] = record["missing_count"] if record else 0

    if stats["total_missing"] == 0 or dry_run:
        return stats

    # Match against glossary definitions using case-insensitive comparison
    fix_query = """
    MATCH (e), (d:Definition)
    WHERE any(lbl IN labels(e) WHERE lbl IN
        ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
         'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
    AND (e.definition IS NULL OR e.definition = '')
    AND d.definition IS NOT NULL
    AND toLower(e.name) = toLower(d.term)
    SET e.definition = d.definition
    RETURN count(e) AS backfilled_count
    """

    async with driver.session(database=database) as session:
        try:
            result = await session.run(fix_query)
            record = await result.single()
            stats["backfilled"] = record["backfilled_count"] if record else 0
            logger.info("Backfilled definitions", count=stats["backfilled"])
        except Exception as e:
            stats["errors"] = [f"Failed to backfill definitions: {e}"]
            logger.exception("Failed to backfill definitions")

    return stats


async def fix_missing_chunk_ids(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate chunk_id for all Chunk nodes that don't have one.

    chunk_id format: {article_id}_chunk_{index:04d}
    Example: "ch01-art01_chunk_0003"

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be fixed.

    Returns:
        Statistics about the fix operation.
    """
    stats = {
        "total_missing": 0,
        "fixed": 0,
        "errors": [],
        "dry_run": dry_run,
    }

    # First, count missing
    count_query = """
    MATCH (c:Chunk)
    WHERE c.chunk_id IS NULL
    RETURN count(c) AS missing_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(count_query)
        record = await result.single()
        stats["total_missing"] = record["missing_count"] if record else 0

    if stats["total_missing"] == 0:
        logger.info("No missing chunk_ids found")
        return stats

    if dry_run:
        logger.info(
            "Dry run: would fix chunk_ids",
            missing=stats["total_missing"],
        )
        return stats

    # Fix the missing chunk_ids
    # Using right() to pad the index with zeros
    fix_query = """
    MATCH (c:Chunk)-[:FROM_ARTICLE]->(a:Article)
    WHERE c.chunk_id IS NULL
    WITH c, a.article_id AS article_id, c.index AS idx
    SET c.chunk_id = article_id + '_chunk_' + right('000' + toString(coalesce(idx, 0)), 4)
    RETURN count(c) AS fixed_count
    """

    async with driver.session(database=database) as session:
        try:
            result = await session.run(fix_query)
            record = await result.single()
            stats["fixed"] = record["fixed_count"] if record else 0
            logger.info("Fixed missing chunk_ids", count=stats["fixed"])
        except Exception as e:
            error_msg = f"Failed to fix chunk_ids: {e}"
            stats["errors"].append(error_msg)
            logger.exception(error_msg)

    return stats


async def fix_plural_entities(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Merge plural entity variants into their singular forms.

    Uses the EntityCleanupNormalizer to find and merge plural/singular
    pairs (e.g., "requirements" -> "requirement").

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be fixed.

    Returns:
        Statistics about the fix operation.
    """
    from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
        EntityCleanupNormalizer,
    )

    normalizer = EntityCleanupNormalizer(driver, database)

    if dry_run:
        preview = await normalizer.preview_cleanup()
        return {
            "dry_run": True,
            "would_merge": preview["to_normalize_count"],
            "entities": preview["to_normalize"],
            "relationship_impact": preview["normalize_relationship_impact"],
        }

    merged = await normalizer.merge_plural_to_singular()
    return {
        "dry_run": False,
        "merged": merged,
    }


async def fix_generic_entities(
    driver: "AnyDriver",
    database: str = "neo4j",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete overly generic entity nodes.

    Removes entities with names like "tool", "software", "process"
    that provide no semantic value.

    Args:
        driver: Neo4j driver instance.
        database: Database name.
        dry_run: If True, only report what would be deleted.

    Returns:
        Statistics about the fix operation.
    """
    from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
        EntityCleanupNormalizer,
    )

    normalizer = EntityCleanupNormalizer(driver, database)

    if dry_run:
        preview = await normalizer.preview_cleanup()
        return {
            "dry_run": True,
            "would_delete": preview["to_delete_count"],
            "entities": preview["to_delete"],
            "relationship_impact": preview["delete_relationship_impact"],
        }

    deleted = await normalizer.delete_generic_entities()
    return {
        "dry_run": False,
        "deleted": deleted,
    }


class ValidationFixer:
    """Orchestrates all validation fix operations.

    Provides a unified interface for running all fixes with
    preview/dry-run support.

    Example:
        >>> fixer = ValidationFixer(driver)
        >>> preview = await fixer.preview_all_fixes()
        >>> if user_confirms(preview):
        ...     results = await fixer.apply_all_fixes()
    """

    def __init__(self, driver: "AnyDriver", database: str = "neo4j") -> None:
        """Initialize the fixer.

        Args:
            driver: Neo4j driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database

    async def preview_all_fixes(self) -> dict[str, Any]:
        """Preview all available fixes without applying them.

        Returns:
            Dictionary with preview of all fix operations.
        """
        results: dict[str, Any] = {}

        results["degenerate_chunks"] = await fix_degenerate_chunks(
            self.driver, self.database, dry_run=True
        )
        results["chunk_index"] = await fix_missing_chunk_index(
            self.driver, self.database, dry_run=True
        )
        results["chunk_ids"] = await fix_missing_chunk_ids(
            self.driver, self.database, dry_run=True
        )
        results["webinar_titles"] = await fix_truncated_webinar_titles(
            self.driver, self.database, dry_run=True
        )
        results["mislabeled_entities"] = await fix_mislabeled_entities(
            self.driver, self.database, dry_run=True
        )
        results["mentioned_in"] = await fix_missing_mentioned_in(
            self.driver, self.database, dry_run=True
        )
        results["definitions"] = await fix_missing_definitions(
            self.driver, self.database, dry_run=True
        )
        results["plural_entities"] = await fix_plural_entities(
            self.driver, self.database, dry_run=True
        )
        results["generic_entities"] = await fix_generic_entities(
            self.driver, self.database, dry_run=True
        )

        # Calculate totals
        results["summary"] = {
            "degenerate_chunks_to_delete": results["degenerate_chunks"]["total_found"],
            "chunk_indices_to_fix": results["chunk_index"]["total_missing"],
            "chunk_ids_to_fix": results["chunk_ids"]["total_missing"],
            "webinar_titles_to_fix": results["webinar_titles"]["total_found"],
            "mislabeled_to_fix": results["mislabeled_entities"]["total_found"],
            "mentioned_in_estimate": results["mentioned_in"].get(
                "entities_without_mentioned_in", 0
            ),
            "definitions_to_backfill": results["definitions"]["total_missing"],
            "entities_to_merge": results["plural_entities"]["would_merge"],
            "entities_to_delete": results["generic_entities"]["would_delete"],
        }

        return results

    async def apply_all_fixes(self) -> dict[str, Any]:
        """Apply all validation fixes.

        Order of operations:
        1. Delete degenerate chunks (garbage first)
        2. Re-index remaining chunks
        3. Generate chunk_ids from indices
        4. Fix truncated webinar titles
        5. Relabel mislabeled entities (before backfill)
        6. Backfill MENTIONED_IN relationships
        7. Backfill missing definitions
        8. Delete generic entities
        9. Merge plural entities

        Returns:
            Statistics from all fix operations.
        """
        results: dict[str, Any] = {}

        logger.info("Starting validation fixes")

        # Step 1: Delete degenerate chunks first
        results["degenerate_chunks"] = await fix_degenerate_chunks(
            self.driver, self.database, dry_run=False
        )

        # Step 2: Re-index remaining chunks
        results["chunk_index"] = await fix_missing_chunk_index(
            self.driver, self.database, dry_run=False
        )

        # Step 3: Generate chunk_ids from indices
        results["chunk_ids"] = await fix_missing_chunk_ids(
            self.driver, self.database, dry_run=False
        )

        # Step 4: Fix truncated webinar titles
        results["webinar_titles"] = await fix_truncated_webinar_titles(
            self.driver, self.database, dry_run=False
        )

        # Step 5: Relabel mislabeled entities
        results["mislabeled_entities"] = await fix_mislabeled_entities(
            self.driver, self.database, dry_run=False
        )

        # Step 6: Backfill MENTIONED_IN
        results["mentioned_in"] = await fix_missing_mentioned_in(
            self.driver, self.database, dry_run=False
        )

        # Step 7: Backfill definitions
        results["definitions"] = await fix_missing_definitions(
            self.driver, self.database, dry_run=False
        )

        # Step 8: Delete generic entities
        results["generic_entities"] = await fix_generic_entities(
            self.driver, self.database, dry_run=False
        )

        # Step 9: Merge plural entities
        results["plural_entities"] = await fix_plural_entities(
            self.driver, self.database, dry_run=False
        )

        logger.info("Validation fixes complete")

        return results

    async def apply_chunk_id_fix_only(self) -> dict[str, Any]:
        """Apply only the chunk_id fix.

        This is a safe operation that only adds properties.

        Returns:
            Statistics from the fix operation.
        """
        return await fix_missing_chunk_ids(self.driver, self.database, dry_run=False)

    async def apply_entity_cleanup_only(self) -> dict[str, Any]:
        """Apply only entity cleanup (delete generic + merge plurals).

        Returns:
            Statistics from the fix operations.
        """
        results: dict[str, Any] = {}

        results["generic_entities"] = await fix_generic_entities(
            self.driver, self.database, dry_run=False
        )

        results["plural_entities"] = await fix_plural_entities(
            self.driver, self.database, dry_run=False
        )

        return results


def format_fix_preview(preview: dict[str, Any]) -> str:
    """Format a fix preview as human-readable text.

    Args:
        preview: Preview dictionary from ValidationFixer.preview_all_fixes().

    Returns:
        Formatted string for display.
    """
    summary = preview["summary"]
    lines = [
        "=== Validation Fix Preview ===",
        "",
        f"Degenerate chunks to delete: {summary.get('degenerate_chunks_to_delete', 0)}",
        f"Chunk indices to fix: {summary.get('chunk_indices_to_fix', 0)}",
        f"Chunk IDs to generate: {summary.get('chunk_ids_to_fix', 0)}",
        f"Webinar titles to fix: {summary.get('webinar_titles_to_fix', 0)}",
        f"Mislabeled entities to relabel: {summary.get('mislabeled_to_fix', 0)}",
        f"MENTIONED_IN to backfill (est.): {summary.get('mentioned_in_estimate', 0)}",
        f"Definitions to backfill: {summary.get('definitions_to_backfill', 0)}",
        f"Generic entities to delete: {summary.get('entities_to_delete', 0)}",
        f"Plural entities to merge: {summary.get('entities_to_merge', 0)}",
        "",
    ]

    # Detail generic entities
    if preview.get("generic_entities", {}).get("would_delete", 0) > 0:
        lines.append("Generic entities to delete:")
        for entity in preview["generic_entities"]["entities"][:10]:
            rels = entity.get("relationship_count", 0)
            lines.append(f"  - {entity['label']}: {entity['name']} ({rels} relationships)")
        if preview["generic_entities"]["would_delete"] > 10:
            remaining = preview["generic_entities"]["would_delete"] - 10
            lines.append(f"  ... and {remaining} more")
        lines.append("")

    # Detail plural entities
    if preview.get("plural_entities", {}).get("would_merge", 0) > 0:
        lines.append("Plural entities to merge:")
        for entity in preview["plural_entities"]["entities"][:10]:
            normalized = entity.get("normalized_name", "?")
            rels = entity.get("relationship_count", 0)
            lines.append(
                f"  - {entity['label']}: {entity['name']} -> {normalized} ({rels} rels)"
            )
        if preview["plural_entities"]["would_merge"] > 10:
            remaining = preview["plural_entities"]["would_merge"] - 10
            lines.append(f"  ... and {remaining} more")
        lines.append("")

    return "\n".join(lines)
