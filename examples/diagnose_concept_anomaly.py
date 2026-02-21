"""Diagnostic script for Issue #37: Concept node count anomaly.

Runs read-only Cypher queries against the live Neo4j Aura database to
characterize why MATCH (n:Concept) RETURN count(n) exceeds
MATCH (n:__Entity__) RETURN count(n).

Usage:
    uv run python examples/diagnose_concept_anomaly.py

Requires: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env or environment.
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase


async def run_query(driver, database: str, description: str, query: str) -> list[dict]:
    """Run a Cypher query and print results."""
    print(f"\n{'─' * 60}")
    print(f"  {description}")
    print(f"{'─' * 60}")
    print(f"  Cypher: {query.strip()[:120]}...")
    async with driver.session(database=database) as session:
        result = await session.run(query)
        records = [dict(r) async for r in result]
    for r in records:
        print(f"  → {r}")
    if not records:
        print("  → (no results)")
    return records


async def main() -> None:
    """Run all diagnostic queries."""
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

    print("=" * 60)
    print("  Issue #37: Concept Node Count Anomaly — Diagnostics")
    print("=" * 60)
    print(f"  Database: {neo4j_uri}")

    # ── Step 1: Characterize the anomaly ──────────────────────
    print("\n\n>>> STEP 1: Characterize the anomaly")

    await run_query(driver, db, "1a. Total Concept nodes", """
        MATCH (n:Concept) RETURN count(n) AS concept_count
    """)

    await run_query(driver, db, "1b. Concept nodes WITH __Entity__", """
        MATCH (n:Concept) WHERE n:__Entity__ RETURN count(n) AS with_entity
    """)

    await run_query(driver, db, "1c. Concept nodes WITHOUT __Entity__", """
        MATCH (n:Concept) WHERE NOT n:__Entity__ RETURN count(n) AS without_entity
    """)

    await run_query(driver, db, "1d. Total __Entity__ nodes", """
        MATCH (n:__Entity__) RETURN count(n) AS entity_count
    """)

    await run_query(driver, db, "1e. Concept nodes WITHOUT __KGBuilder__", """
        MATCH (n:Concept) WHERE NOT n:__KGBuilder__ RETURN count(n) AS without_kgbuilder
    """)

    # ── Step 2: Check multi-labeling issues ───────────────────
    print("\n\n>>> STEP 2: Check multi-labeling issues")

    await run_query(driver, db, "2a. Concept nodes with extra type labels", """
        MATCH (n:Concept)
        WITH n, [l IN labels(n) WHERE NOT l STARTS WITH '__' AND l <> 'Concept'] AS extras
        WHERE size(extras) > 0
        RETURN extras, count(n) AS cnt
        ORDER BY cnt DESC
    """)

    await run_query(driver, db, "2b. Label combos of Concept nodes WITHOUT __Entity__", """
        MATCH (n:Concept) WHERE NOT n:__Entity__
        RETURN labels(n) AS all_labels, count(n) AS cnt
        ORDER BY cnt DESC LIMIT 10
    """)

    # ── Step 3: Sample orphaned/suspicious nodes ──────────────
    print("\n\n>>> STEP 3: Sample orphaned/suspicious nodes")

    await run_query(driver, db, "3a. Sample Concept nodes without __Entity__", """
        MATCH (n:Concept) WHERE NOT n:__Entity__
        RETURN n.name, labels(n) AS all_labels, keys(n) AS props
        LIMIT 15
    """)

    await run_query(driver, db, "3b. Disconnected Concept nodes without __Entity__", """
        MATCH (n:Concept) WHERE NOT n:__Entity__ AND NOT (n)--()
        RETURN count(n) AS disconnected_count
    """)

    await run_query(driver, db, "3c. Concept nodes w/o __Entity__ by relationship type", """
        MATCH (n:Concept)-[r]-()
        WHERE NOT n:__Entity__
        RETURN type(r) AS rel_type, count(r) AS cnt
        ORDER BY cnt DESC LIMIT 10
    """)

    # ── Step 4: Check if gleaning is the source ──────────────
    print("\n\n>>> STEP 4: Check gleaning hypothesis")

    await run_query(driver, db, "4a. Concept w/o __Entity__ that have MENTIONED_IN rels", """
        MATCH (n:Concept)-[:MENTIONED_IN]->(c:Chunk)
        WHERE NOT n:__Entity__
        RETURN count(DISTINCT n) AS gleaned_with_mentions
    """)

    await run_query(driver, db, "4b. ALL entity-type nodes without __Entity__", """
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
        AND NOT n:__Entity__
        WITH [l IN labels(n) WHERE NOT l STARTS WITH '__'] AS type_labels, count(n) AS cnt
        RETURN type_labels, cnt
        ORDER BY cnt DESC
    """)

    await run_query(driver, db, "4c. Concept w/o __Entity__ that overlap __Entity__ names", """
        MATCH (n:Concept) WHERE NOT n:__Entity__
        WITH n.name AS name, count(n) AS orphan_count
        OPTIONAL MATCH (e:__Entity__ {name: name})
        RETURN
            count(CASE WHEN e IS NOT NULL THEN 1 END) AS has_entity_counterpart,
            count(CASE WHEN e IS NULL THEN 1 END) AS no_entity_counterpart
    """)

    # ── Step 5: Summary counts for all entity labels ──────────
    print("\n\n>>> STEP 5: Full entity label breakdown")

    await run_query(driver, db, "5a. Per-label counts WITH and WITHOUT __Entity__", """
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN
            ['Concept', 'Challenge', 'Artifact', 'Bestpractice', 'Processstage',
             'Role', 'Standard', 'Tool', 'Methodology', 'Industry'])
        WITH [l IN labels(n) WHERE NOT l STARTS WITH '__'][0] AS type_label,
             CASE WHEN n:__Entity__ THEN 'yes' ELSE 'no' END AS has_entity
        RETURN type_label, has_entity, count(*) AS cnt
        ORDER BY type_label, has_entity
    """)

    await driver.close()

    print("\n\n" + "=" * 60)
    print("  Diagnostics complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
