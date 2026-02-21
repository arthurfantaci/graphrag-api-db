"""Neo4j constraint and index management.

This module provides utilities for creating database constraints
and indexes to ensure data integrity and query performance.
"""

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from neo4j import Driver

logger = structlog.get_logger(__name__)


# =============================================================================
# CONSTRAINT DEFINITIONS
# =============================================================================

# Uniqueness constraints for node types
#
# NOTE: Entity type constraints (Concept, Challenge, etc.) are intentionally
# omitted. neo4j_graphrag 1.13+ uses CREATE with __KGBuilder__ label followed
# by apoc.create.addLabels() to add type labels. Per-type uniqueness constraints
# cause IndexEntryConflictException when the same entity name appears across
# multiple extraction batches, rolling back entire batch transactions.
# Entity deduplication is handled by neo4j_graphrag's entity resolution step.
UNIQUENESS_CONSTRAINTS = [
    # Lexical graph nodes
    ("Article", "article_id"),
    ("Chunk", "chunk_id"),
    # Supplementary nodes
    ("Chapter", "chapter_number"),
    ("Image", "resource_id"),
    ("Video", "resource_id"),
    ("Webinar", "resource_id"),
    ("Definition", "term_id"),
]

# Property existence constraints
EXISTENCE_CONSTRAINTS = [
    ("Article", "title"),
    ("Article", "url"),
    ("Chunk", "text"),
    ("Chapter", "title"),
]

# Indexes for common query patterns
INDEXES = [
    # Entity lookups
    ("Concept", "display_name"),
    ("Standard", "organization"),
    ("Tool", "vendor"),
    ("Industry", "regulated"),
    # Article lookups
    ("Article", "chapter_number"),
    ("Article", "url"),
    ("Chunk", "source_article_id"),
    # Resource lookups
    ("Image", "source_article_id"),
    ("Video", "source_article_id"),
    ("Video", "platform"),
    ("Webinar", "source_article_id"),
]


class ConstraintManager:
    """Manager for Neo4j constraints and indexes.

    Provides methods for creating and verifying database
    constraints and indexes.

    Example:
        >>> manager = ConstraintManager(driver)
        >>> await manager.create_all()
        >>> status = await manager.verify_all()
    """

    def __init__(self, driver: "Driver", database: str = "neo4j") -> None:
        """Initialize the manager.

        Args:
            driver: Neo4j driver instance.
            database: Database name.
        """
        self.driver = driver
        self.database = database

    async def create_all(self) -> dict:
        """Create all constraints and indexes.

        Returns:
            Statistics about created objects.
        """
        stats = {
            "uniqueness_constraints": 0,
            "existence_constraints": 0,
            "indexes": 0,
            "errors": [],
        }

        # Create uniqueness constraints
        for label, prop in UNIQUENESS_CONSTRAINTS:
            try:
                await self._create_uniqueness_constraint(label, prop)
                stats["uniqueness_constraints"] += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    stats["errors"].append(f"{label}.{prop}: {e}")

        # Create existence constraints (Neo4j Enterprise only)
        for label, prop in EXISTENCE_CONSTRAINTS:
            try:
                await self._create_existence_constraint(label, prop)
                stats["existence_constraints"] += 1
            except Exception as e:
                # Existence constraints require Enterprise edition
                if "enterprise" not in str(e).lower():
                    if "already exists" not in str(e).lower():
                        stats["errors"].append(f"{label}.{prop}: {e}")

        # Create indexes
        for label, prop in INDEXES:
            try:
                await self._create_index(label, prop)
                stats["indexes"] += 1
            except Exception as e:
                if "already exists" not in str(e).lower():
                    stats["errors"].append(f"{label}.{prop}: {e}")

        logger.info(
            "Created constraints and indexes",
            uniqueness=stats["uniqueness_constraints"],
            existence=stats["existence_constraints"],
            indexes=stats["indexes"],
            errors=len(stats["errors"]),
        )

        return stats

    async def _create_uniqueness_constraint(
        self,
        label: str,
        property_name: str,
    ) -> None:
        """Create a uniqueness constraint.

        Args:
            label: Node label.
            property_name: Property to constrain.
        """
        constraint_name = f"unique_{label.lower()}_{property_name}"
        query = f"""
        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
        FOR (n:{label})
        REQUIRE n.{property_name} IS UNIQUE
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(query)

    async def _create_existence_constraint(
        self,
        label: str,
        property_name: str,
    ) -> None:
        """Create an existence constraint (Enterprise only).

        Args:
            label: Node label.
            property_name: Property to constrain.
        """
        constraint_name = f"exists_{label.lower()}_{property_name}"
        query = f"""
        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
        FOR (n:{label})
        REQUIRE n.{property_name} IS NOT NULL
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(query)

    async def _create_index(
        self,
        label: str,
        property_name: str,
    ) -> None:
        """Create a property index.

        Args:
            label: Node label.
            property_name: Property to index.
        """
        index_name = f"idx_{label.lower()}_{property_name}"
        query = f"""
        CREATE INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON (n.{property_name})
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(query)

    async def verify_all(self) -> dict:
        """Verify all constraints and indexes exist.

        Returns:
            Verification status for each constraint/index.
        """
        status = {
            "constraints": [],
            "indexes": [],
            "missing_constraints": [],
            "missing_indexes": [],
        }

        # Get existing constraints
        constraint_query = "SHOW CONSTRAINTS"
        async with self.driver.session(database=self.database) as session:
            result = await session.run(constraint_query)
            constraints = [dict(record) async for record in result]
            status["constraints"] = [c.get("name", "") for c in constraints]

        # Get existing indexes
        index_query = "SHOW INDEXES"
        async with self.driver.session(database=self.database) as session:
            result = await session.run(index_query)
            indexes = [dict(record) async for record in result]
            status["indexes"] = [i.get("name", "") for i in indexes]

        # Check for missing
        for label, prop in UNIQUENESS_CONSTRAINTS:
            expected = f"unique_{label.lower()}_{prop}"
            if expected not in status["constraints"]:
                status["missing_constraints"].append(f"{label}.{prop}")

        for label, prop in INDEXES:
            expected = f"idx_{label.lower()}_{prop}"
            if expected not in status["indexes"]:
                status["missing_indexes"].append(f"{label}.{prop}")

        return status


async def create_all_constraints(
    driver: "Driver",
    database: str = "neo4j",
) -> dict:
    """Convenience function to create all constraints.

    Args:
        driver: Neo4j driver.
        database: Database name.

    Returns:
        Creation statistics.
    """
    manager = ConstraintManager(driver, database)
    return await manager.create_all()


async def create_fulltext_index(
    driver: "Driver",
    database: str = "neo4j",
    index_name: str = "chunk_text_fulltext",
) -> None:
    """Create full-text index for BM25 hybrid search on Chunk text.

    Enables keyword-based retrieval alongside vector search for exact terms
    like standard names ("ISO 26262"), acronyms ("RTM"), and technical terms.

    Args:
        driver: Neo4j driver.
        database: Database name.
        index_name: Name for the full-text index.
    """
    query = f"""
    CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
    FOR (c:Chunk) ON EACH [c.text]
    """

    async with driver.session(database=database) as session:
        await session.run(query)

    logger.info("Created full-text index", index_name=index_name)


async def create_vector_index(
    driver: "Driver",
    database: str = "neo4j",
    index_name: str = "chunk_embeddings",
    dimensions: int = 1536,
    similarity_function: str = "cosine",
) -> None:
    """Create vector index for chunk embeddings.

    Args:
        driver: Neo4j driver.
        database: Database name.
        index_name: Name for the vector index.
        dimensions: Embedding dimensions.
        similarity_function: Similarity function (cosine, euclidean).
    """
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (c:Chunk)
    ON c.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}
    }}
    """

    async with driver.session(database=database) as session:
        await session.run(query)

    logger.info(
        "Created vector index",
        index_name=index_name,
        dimensions=dimensions,
        similarity=similarity_function,
    )


async def create_community_vector_index(
    driver: "Driver",
    database: str = "neo4j",
    index_name: str = "community_summary_embeddings",
    dimensions: int = 1024,
    similarity_function: str = "cosine",
) -> None:
    """Create vector index for community summary embeddings.

    Args:
        driver: Neo4j driver.
        database: Database name.
        index_name: Name for the vector index.
        dimensions: Embedding dimensions (Voyage AI voyage-4 = 1024).
        similarity_function: Similarity function (cosine, euclidean).
    """
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (c:Community)
    ON c.summary_embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}
    }}
    """

    async with driver.session(database=database) as session:
        await session.run(query)

    logger.info(
        "Created community vector index",
        index_name=index_name,
        dimensions=dimensions,
        similarity=similarity_function,
    )


async def drop_all_constraints(
    driver: "Driver",
    database: str = "neo4j",
) -> int:
    """Drop all constraints (useful for reset).

    Args:
        driver: Neo4j driver.
        database: Database name.

    Returns:
        Number of constraints dropped.
    """
    dropped = 0

    # Get all constraints
    query = "SHOW CONSTRAINTS"
    async with driver.session(database=database) as session:
        result = await session.run(query)
        constraints = [dict(record) async for record in result]

    # Drop each constraint
    for constraint in constraints:
        name = constraint.get("name", "")
        if name:
            drop_query = f"DROP CONSTRAINT {name} IF EXISTS"
            async with driver.session(database=database) as session:
                await session.run(drop_query)
                dropped += 1

    logger.info("Dropped constraints", count=dropped)
    return dropped
