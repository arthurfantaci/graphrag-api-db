"""Data repair utilities for knowledge graph validation.

This module provides fix functions for common data quality issues
detected by validation queries. Each fix can be previewed before
applying to understand the impact.

Supported fixes:
- fix_missing_chunk_ids(): Generate chunk_id for all Chunk nodes
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
    pairs (e.g., "requirements" → "requirement").

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
        results = {}

        # Preview chunk_id fix
        results["chunk_ids"] = await fix_missing_chunk_ids(self.driver, self.database, dry_run=True)

        # Preview plural entity fix
        results["plural_entities"] = await fix_plural_entities(
            self.driver, self.database, dry_run=True
        )

        # Preview generic entity fix
        results["generic_entities"] = await fix_generic_entities(
            self.driver, self.database, dry_run=True
        )

        # Calculate totals
        results["summary"] = {
            "chunk_ids_to_fix": results["chunk_ids"]["total_missing"],
            "entities_to_merge": results["plural_entities"]["would_merge"],
            "entities_to_delete": results["generic_entities"]["would_delete"],
            "total_changes": (
                results["chunk_ids"]["total_missing"]
                + results["plural_entities"]["would_merge"]
                + results["generic_entities"]["would_delete"]
            ),
        }

        return results

    async def apply_all_fixes(self) -> dict[str, Any]:
        """Apply all validation fixes.

        Order of operations:
        1. Fix chunk_ids (safe, additive)
        2. Delete generic entities (before merge to reduce noise)
        3. Merge plural entities (after delete to avoid merging into generic)

        Returns:
            Statistics from all fix operations.
        """
        results = {}

        logger.info("Starting validation fixes")

        # Step 1: Fix chunk_ids
        results["chunk_ids"] = await fix_missing_chunk_ids(
            self.driver, self.database, dry_run=False
        )

        # Step 2: Delete generic entities first
        results["generic_entities"] = await fix_generic_entities(
            self.driver, self.database, dry_run=False
        )

        # Step 3: Merge plural entities
        results["plural_entities"] = await fix_plural_entities(
            self.driver, self.database, dry_run=False
        )

        # Calculate totals
        results["summary"] = {
            "chunk_ids_fixed": results["chunk_ids"]["fixed"],
            "entities_deleted": results["generic_entities"]["deleted"],
            "entities_merged": results["plural_entities"]["merged"],
            "total_changes": (
                results["chunk_ids"]["fixed"]
                + results["generic_entities"]["deleted"]
                + results["plural_entities"]["merged"]
            ),
        }

        logger.info(
            "Validation fixes complete",
            chunk_ids=results["chunk_ids"]["fixed"],
            deleted=results["generic_entities"]["deleted"],
            merged=results["plural_entities"]["merged"],
        )

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
        results = {}

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
    lines = [
        "=== Validation Fix Preview ===",
        "",
        f"Chunk IDs to generate: {preview['summary']['chunk_ids_to_fix']}",
        f"Generic entities to delete: {preview['summary']['entities_to_delete']}",
        f"Plural entities to merge: {preview['summary']['entities_to_merge']}",
        "",
        f"Total changes: {preview['summary']['total_changes']}",
        "",
    ]

    # Detail generic entities
    if preview["generic_entities"]["would_delete"] > 0:
        lines.append("Generic entities to delete:")
        for entity in preview["generic_entities"]["entities"][:10]:
            rels = entity.get("relationship_count", 0)
            lines.append(f"  - {entity['label']}: {entity['name']} ({rels} relationships)")
        if preview["generic_entities"]["would_delete"] > 10:
            remaining = preview["generic_entities"]["would_delete"] - 10
            lines.append(f"  ... and {remaining} more")
        lines.append("")

    # Detail plural entities
    if preview["plural_entities"]["would_merge"] > 0:
        lines.append("Plural entities to merge:")
        for entity in preview["plural_entities"]["entities"][:10]:
            normalized = entity.get("normalized_name", "?")
            rels = entity.get("relationship_count", 0)
            lines.append(f"  - {entity['label']}: {entity['name']} → {normalized} ({rels} rels)")
        if preview["plural_entities"]["would_merge"] > 10:
            remaining = preview["plural_entities"]["would_merge"] - 10
            lines.append(f"  ... and {remaining} more")
        lines.append("")

    return "\n".join(lines)
