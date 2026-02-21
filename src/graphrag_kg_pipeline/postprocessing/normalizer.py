"""Entity name normalization utilities.

This module provides utilities for normalizing entity names
and deduplicating entities in the knowledge graph, including
cross-label deduplication for entities with the same name
but different type labels.
"""

import re
from typing import TYPE_CHECKING, Any, ClassVar

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name to lowercase, trimmed form.

    Applies standard normalization:
    - Lowercase
    - Trim whitespace
    - Collapse multiple spaces
    - Remove leading/trailing punctuation

    Args:
        name: Raw entity name.

    Returns:
        Normalized name.

    Example:
        >>> normalize_entity_name("  Requirements Traceability  ")
        'requirements traceability'
        >>> normalize_entity_name("ISO-26262")
        'iso-26262'
    """
    if not name:
        return ""

    # Lowercase and strip
    normalized = name.lower().strip()

    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)

    # Remove leading/trailing punctuation (but keep internal)
    normalized = re.sub(r"^[^\w]+|[^\w]+$", "", normalized)

    return normalized


def names_are_equivalent(name1: str, name2: str) -> bool:
    """Check if two names are equivalent after normalization.

    Args:
        name1: First name.
        name2: Second name.

    Returns:
        True if names are equivalent.
    """
    return normalize_entity_name(name1) == normalize_entity_name(name2)


class EntityNormalizer:
    """Normalizer for entity names in Neo4j.

    Provides methods to normalize entity names, deduplicate
    nodes with equivalent names (same-label and cross-label).

    Example:
        >>> normalizer = EntityNormalizer(driver)
        >>> stats = await normalizer.normalize_all_entities()
        >>> print(f"Updated {stats['updated']} entity names")
    """

    # Label priority ranking for cross-label deduplication.
    # Intrinsic types (Standard, Tool, Industry, Role) are highly specific —
    # the LLM is almost always correct when it assigns these.
    # Concept is the safe generic default.
    # Contextual types (Artifact, Processstage, Bestpractice, Challenge) are
    # framing-dependent and most often misclassifications.
    LABEL_PRIORITY: ClassVar[list[str]] = [
        "Standard",
        "Tool",
        "Industry",
        "Role",
        "Methodology",
        "Concept",
        "Artifact",
        "Processstage",
        "Bestpractice",
        "Challenge",
    ]

    _SYSTEM_LABELS: ClassVar[frozenset[str]] = frozenset({"__Entity__", "__KGBuilder__"})

    def __init__(self, driver: "AsyncDriver", database: str = "neo4j") -> None:
        """Initialize the normalizer.

        Args:
            driver: Neo4j async driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database

        # Labels to normalize
        self.entity_labels = [
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
        ]

    async def normalize_all_entities(self) -> dict:
        """Normalize names for all entity types.

        Updates the `name` property to lowercase normalized form
        and preserves original in `display_name` if not already set.

        Returns:
            Statistics about the normalization.
        """
        stats = {
            "total_processed": 0,
            "updated": 0,
            "by_label": {},
        }

        for label in self.entity_labels:
            count = await self._normalize_label(label)
            stats["by_label"][label] = count
            stats["updated"] += count

        # Get total processed
        stats["total_processed"] = await self._count_entities()

        logger.info(
            "Entity normalization complete",
            updated=stats["updated"],
            total=stats["total_processed"],
        )

        return stats

    async def _normalize_label(self, label: str) -> int:
        """Normalize entities of a specific label.

        Args:
            label: Node label to normalize.

        Returns:
            Number of nodes updated.
        """
        query = f"""
        MATCH (n:{label})
        WHERE n.name IS NOT NULL AND n.name <> toLower(trim(n.name))
        WITH n, n.name AS original_name
        SET n.display_name = CASE
            WHEN n.display_name IS NULL THEN original_name
            ELSE n.display_name
        END,
        n.name = toLower(trim(n.name))
        RETURN count(n) AS updated
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["updated"] if record else 0

    async def _count_entities(self) -> int:
        """Count total entities.

        Returns:
            Total entity count.
        """
        labels_list = ", ".join(f"'{lbl}'" for lbl in self.entity_labels)
        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN [{labels_list}])
        RETURN count(n) AS total
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["total"] if record else 0

    # ─── Same-label deduplication ─────────────────────────────────────────

    async def deduplicate_by_name(self) -> dict:
        """Deduplicate entities with identical normalized names.

        Merges duplicate nodes, preserving relationships.

        Returns:
            Statistics about deduplication.
        """
        stats = {
            "duplicates_found": 0,
            "merged": 0,
            "by_label": {},
        }

        for label in self.entity_labels:
            merged = await self._deduplicate_label(label)
            stats["by_label"][label] = merged
            stats["merged"] += merged

        logger.info(
            "Entity deduplication complete",
            merged=stats["merged"],
        )

        return stats

    async def _deduplicate_label(self, label: str) -> int:
        """Deduplicate entities of a specific label.

        Args:
            label: Node label to deduplicate.

        Returns:
            Number of nodes merged.
        """
        # Find duplicates
        find_query = f"""
        MATCH (n:{label})
        WITH n.name AS name, collect(n) AS nodes, count(n) AS cnt
        WHERE cnt > 1
        RETURN name, [node IN nodes | elementId(node)] AS node_ids, cnt
        """

        merged_count = 0

        async with self.driver.session(database=self.database) as session:
            result = await session.run(find_query)
            duplicates = [dict(record) async for record in result]

        for dup in duplicates:
            node_ids = dup["node_ids"]
            if len(node_ids) > 1:
                # Keep first, merge others into it
                primary_id = node_ids[0]
                for dup_id in node_ids[1:]:
                    await self._merge_duplicate(label, primary_id, dup_id)
                    merged_count += 1

        return merged_count

    async def _merge_duplicate(
        self,
        label: str,
        primary_id: str,
        duplicate_id: str,
    ) -> None:
        """Merge a duplicate node into the primary.

        Args:
            label: Node label.
            primary_id: Element ID of primary node.
            duplicate_id: Element ID of duplicate node.
        """
        query = f"""
        MATCH (primary:{label}) WHERE elementId(primary) = $primary_id
        MATCH (dup:{label}) WHERE elementId(dup) = $duplicate_id

        // Transfer outgoing relationships
        CALL {{
            WITH primary, dup
            MATCH (dup)-[r]->()
            WITH primary, r, endNode(r) AS target, type(r) AS rel_type
            CALL apoc.merge.relationship(primary, rel_type, {{}}, properties(r), target, {{}})
            YIELD rel
            DELETE r
            RETURN count(*) AS out_count
        }}

        // Transfer incoming relationships
        CALL {{
            WITH primary, dup
            MATCH ()-[r]->(dup)
            WITH primary, r, startNode(r) AS source, type(r) AS rel_type
            CALL apoc.merge.relationship(source, rel_type, {{}}, properties(r), primary, {{}})
            YIELD rel
            DELETE r
            RETURN count(*) AS in_count
        }}

        // Merge properties (keep primary's if both have)
        SET primary += dup

        // Delete duplicate
        DELETE dup
        """

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(query, primary_id=primary_id, duplicate_id=duplicate_id)
        except Exception as e:
            # Fallback without APOC if not available
            logger.warning(
                "APOC merge failed, using basic merge",
                error=str(e),
                label=label,
            )
            await self._merge_duplicate_basic(label, primary_id, duplicate_id)

    async def _merge_duplicate_basic(
        self,
        label: str,
        primary_id: str,
        duplicate_id: str,
    ) -> None:
        """Basic merge without APOC.

        Args:
            label: Node label.
            primary_id: Element ID of primary node.
            duplicate_id: Element ID of duplicate node.
        """
        query = f"""
        MATCH (primary:{label}) WHERE elementId(primary) = $primary_id
        MATCH (dup:{label}) WHERE elementId(dup) = $duplicate_id

        // Note: This simple version may lose some relationships
        // if there are already relationships between primary and targets

        // Delete duplicate
        DETACH DELETE dup
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(query, primary_id=primary_id, duplicate_id=duplicate_id)

    # ─── Cross-label deduplication ────────────────────────────────────────

    async def deduplicate_cross_label(self) -> dict[str, Any]:
        """Deduplicate entities with the same name but different type labels.

        When the LLM extractor encounters the same real-world entity across
        multiple articles, it may classify it under different labels (e.g.,
        "traceability" as both Concept and Challenge). This method merges
        those cross-label duplicates using a priority ranking to choose the
        canonical label.

        Returns:
            Statistics: cross_label_merged, by_winning_label, replaced_labels.
        """
        stats: dict[str, Any] = {
            "cross_label_merged": 0,
            "by_winning_label": {},
            "replaced_labels": {},
        }

        # Find entities sharing the same name across different labels
        find_query = """
        MATCH (e:__Entity__)
        WHERE e.name IS NOT NULL
        WITH e.name AS name, collect(e) AS nodes
        WHERE size(nodes) > 1
        RETURN name,
               [n IN nodes | elementId(n)] AS node_ids,
               [n IN nodes | labels(n)] AS all_labels
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(find_query)
            groups = [dict(record) async for record in result]

        for group in groups:
            name = group["name"]
            node_ids = group["node_ids"]
            all_labels = group["all_labels"]

            # Extract type labels (exclude system labels)
            type_labels = []
            for label_set in all_labels:
                type_only = [lbl for lbl in label_set if lbl not in self._SYSTEM_LABELS]
                type_labels.append(type_only)

            # Only process if there are genuinely different type labels
            unique_types = {lbl for grp in type_labels for lbl in grp}
            if len(unique_types) <= 1:
                continue

            # Pick the winning label
            winning_label = self._resolve_winning_label(type_labels)

            # Find the primary node (first node that has the winning label)
            primary_idx = None
            for idx, labels in enumerate(type_labels):
                if winning_label in labels:
                    primary_idx = idx
                    break

            # Fallback: use the first node if none has the exact winning label
            if primary_idx is None:
                primary_idx = 0

            primary_id = node_ids[primary_idx]

            # Merge all other nodes into primary
            for idx, dup_id in enumerate(node_ids):
                if idx == primary_idx:
                    continue

                replaced_type = type_labels[idx][0] if type_labels[idx] else "unknown"

                try:
                    await self._merge_cross_label_duplicate(primary_id, dup_id)
                    stats["cross_label_merged"] += 1

                    # Track winning label distribution
                    stats["by_winning_label"][winning_label] = (
                        stats["by_winning_label"].get(winning_label, 0) + 1
                    )

                    # Track which labels were replaced
                    stats["replaced_labels"][replaced_type] = (
                        stats["replaced_labels"].get(replaced_type, 0) + 1
                    )

                    logger.debug(
                        "Merged cross-label duplicate",
                        name=name,
                        winning_label=winning_label,
                        replaced_label=replaced_type,
                    )

                except Exception:
                    logger.warning(
                        "Failed to merge cross-label duplicate",
                        name=name,
                        primary_id=primary_id,
                        duplicate_id=dup_id,
                        exc_info=True,
                    )

        logger.info(
            "Cross-label deduplication complete",
            merged=stats["cross_label_merged"],
            by_winning_label=stats["by_winning_label"],
            replaced_labels=stats["replaced_labels"],
        )

        return stats

    def _resolve_winning_label(self, labels_list: list[list[str]]) -> str:
        """Pick the canonical type label from a set of cross-label duplicates.

        Uses LABEL_PRIORITY ranking: intrinsic types (Standard, Tool, Industry,
        Role) win over generic (Concept) which wins over contextual types
        (Artifact, Processstage, Bestpractice, Challenge).

        Args:
            labels_list: List of label sets, one per node in the duplicate group.

        Returns:
            The highest-priority label found, or "Concept" as default.
        """
        # Collect all type labels across nodes (excluding system labels)
        all_type_labels: set[str] = set()
        for labels in labels_list:
            for label in labels:
                if label not in self._SYSTEM_LABELS:
                    all_type_labels.add(label)

        if not all_type_labels:
            return "Concept"

        # Return the label with the lowest index in LABEL_PRIORITY (= highest priority)
        best_priority = len(self.LABEL_PRIORITY)
        best_label = "Concept"

        for label in all_type_labels:
            try:
                idx = self.LABEL_PRIORITY.index(label)
            except ValueError:
                continue
            if idx < best_priority:
                best_priority = idx
                best_label = label

        return best_label

    async def _merge_cross_label_duplicate(
        self,
        primary_id: str,
        duplicate_id: str,
    ) -> None:
        """Merge a cross-label duplicate node into the primary.

        Uses APOC to transfer relationships and explicit list-append
        for collection properties (source_spans, aliases) to avoid the
        data-loss bug in ``SET primary += dup`` for list properties.

        Follows the APOC pattern from industry_taxonomy.py:502-549.

        Args:
            primary_id: Element ID of the primary (winning) node.
            duplicate_id: Element ID of the duplicate node to absorb.
        """
        query = """
        MATCH (primary) WHERE elementId(primary) = $primary_id
        MATCH (dup) WHERE elementId(dup) = $duplicate_id

        // Transfer outgoing relationships
        CALL (primary, dup) {
            MATCH (dup)-[r]->()
            WITH primary, r, endNode(r) AS target, type(r) AS rel_type
            CALL apoc.merge.relationship(primary, rel_type, {}, properties(r), target, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS out_count
        }

        // Transfer incoming relationships
        CALL (primary, dup) {
            MATCH ()-[r]->(dup)
            WITH primary, r, startNode(r) AS source, type(r) AS rel_type
            CALL apoc.merge.relationship(source, rel_type, {}, properties(r), primary, {})
            YIELD rel
            DELETE r
            RETURN count(*) AS in_count
        }

        // Merge list properties (append + deduplicate)
        SET primary.source_spans = apoc.coll.toSet(
            coalesce(primary.source_spans, []) + coalesce(dup.source_spans, [])
        )
        SET primary.aliases = apoc.coll.toSet(
            coalesce(primary.aliases, []) + coalesce(dup.aliases, [])
        )

        // Scalar properties: keep longer definition
        SET primary.definition = CASE
            WHEN primary.definition IS NOT NULL
                 AND size(primary.definition) >= size(coalesce(dup.definition, ''))
            THEN primary.definition
            ELSE coalesce(dup.definition, primary.definition)
        END

        // Delete duplicate
        DELETE dup
        """

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(query, primary_id=primary_id, duplicate_id=duplicate_id)
        except Exception as e:
            logger.warning(
                "APOC cross-label merge failed, using DETACH DELETE fallback",
                error=str(e),
            )
            fallback_query = """
            MATCH (dup) WHERE elementId(dup) = $duplicate_id
            DETACH DELETE dup
            """
            async with self.driver.session(database=self.database) as session:
                await session.run(fallback_query, duplicate_id=duplicate_id)


async def get_entity_name_stats(driver: "AsyncDriver", database: str = "neo4j") -> dict:
    """Get statistics about entity names.

    Args:
        driver: Neo4j driver.
        database: Database name.

    Returns:
        Statistics about entity naming.
    """
    query = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN
        ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
         'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
    WITH labels(n)[0] AS label,
         n.name AS name,
         CASE WHEN n.name = toLower(trim(n.name)) THEN 1 ELSE 0 END AS is_normalized
    RETURN label,
           count(*) AS total,
           sum(is_normalized) AS normalized_count,
           count(*) - sum(is_normalized) AS needs_normalization
    ORDER BY label
    """

    async with driver.session(database=database) as session:
        result = await session.run(query)
        return {
            record["label"]: {
                "total": record["total"],
                "normalized": record["normalized_count"],
                "needs_normalization": record["needs_normalization"],
            }
            async for record in result
        }
