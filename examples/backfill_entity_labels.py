"""Backfill __Entity__ and __KGBuilder__ labels on gleaned-only nodes.

Fixes Issue #37: Concept node count anomaly. Gleaning created entity nodes
without the standard __Entity__/__KGBuilder__ labels, making them invisible
to entity resolution and cross-label deduplication.

This script adds the missing labels to all entity-type nodes that lack them.
Safe to run multiple times (idempotent).

Usage:
    uv run python examples/backfill_entity_labels.py
    uv run python examples/backfill_entity_labels.py --dry-run

Requires: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env or environment.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

ENTITY_LABELS = [
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


async def main() -> None:
    """Backfill missing __Entity__ and __KGBuilder__ labels."""
    parser = argparse.ArgumentParser(
        description="Backfill entity labels on gleaned nodes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count affected nodes without modifying",
    )
    args = parser.parse_args()

    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not neo4j_password:
        print("ERROR: NEO4J_PASSWORD environment variable required")
        sys.exit(1)

    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    db = neo4j_database

    # Count affected nodes
    count_query = """
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN $entity_labels)
        AND NOT n:__Entity__
        RETURN count(n) AS affected_count
    """

    async with driver.session(database=db) as session:
        result = await session.run(count_query, entity_labels=ENTITY_LABELS)
        record = await result.single()
        affected = record["affected_count"] if record else 0

    print(f"Found {affected} entity-type nodes missing __Entity__ label")

    if affected == 0:
        print("Nothing to backfill.")
        await driver.close()
        return

    if args.dry_run:
        print("Dry run — no changes made.")
        await driver.close()
        return

    # Backfill labels
    backfill_query = """
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN $entity_labels)
        AND NOT n:__Entity__
        SET n:__Entity__, n:__KGBuilder__
        RETURN count(n) AS updated
    """

    async with driver.session(database=db) as session:
        result = await session.run(backfill_query, entity_labels=ENTITY_LABELS)
        record = await result.single()
        updated = record["updated"] if record else 0

    print(f"Backfilled {updated} nodes with __Entity__ + __KGBuilder__ labels")

    # Verify
    async with driver.session(database=db) as session:
        result = await session.run(count_query, entity_labels=ENTITY_LABELS)
        record = await result.single()
        remaining = record["affected_count"] if record else 0

    print(f"Remaining nodes without __Entity__: {remaining}")

    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
