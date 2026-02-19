"""Leiden community detection using leidenalg + igraph.

Exports the entity graph from Neo4j, runs Leiden community detection
locally, and writes community IDs back to entity nodes. Uses only
semantic relationship types from the extraction schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from graphrag_kg_pipeline.extraction.schema import RELATIONSHIP_TYPES

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)

# Semantic relationship types to include in community detection
_SEMANTIC_REL_TYPES = list(RELATIONSHIP_TYPES.keys())


class CommunityDetector:
    """Leiden community detection on the entity graph.

    Exports semantic edges from Neo4j, builds an igraph Graph,
    runs the Leiden algorithm, and writes community IDs back.

    Attributes:
        driver: Async Neo4j driver.
        database: Neo4j database name.
    """

    def __init__(self, driver: AsyncDriver, database: str = "neo4j") -> None:
        """Initialize the community detector.

        Args:
            driver: Async Neo4j driver.
            database: Neo4j database name.
        """
        self.driver = driver
        self.database = database

    async def detect_communities(
        self,
        *,
        gamma: float = 1.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run Leiden community detection on the entity graph.

        Steps:
        1. Export semantic edges from Neo4j
        2. Build igraph Graph from edges
        3. Run Leiden with ModularityVertexPartition
        4. Write community IDs back to Neo4j
        5. Return statistics

        Args:
            gamma: Resolution parameter (higher = smaller communities).
            seed: Random seed for reproducibility.

        Returns:
            Statistics dict with community_count, modularity, and node_count.
        """
        import igraph as ig
        import leidenalg as la

        # Step 1: Export semantic edges
        edges, node_names = await self._export_semantic_edges()

        if not edges:
            logger.info("No semantic edges found, skipping community detection")
            return {"community_count": 0, "modularity": 0.0, "node_count": 0}

        logger.info(
            "Building community graph",
            edge_count=len(edges),
            node_count=len(node_names),
        )

        # Step 2: Build igraph Graph
        graph = ig.Graph.TupleList(edges, directed=False)

        # Step 3: Run Leiden (gamma controls resolution: higher = smaller communities)
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            resolution_parameter=gamma,
            seed=seed,
        )

        community_count = len(partition)
        modularity = partition.modularity

        logger.info(
            "Leiden community detection complete",
            community_count=community_count,
            modularity=f"{modularity:.4f}",
            node_count=graph.vcount(),
        )

        # Step 4: Write community IDs back to Neo4j
        assignments = {}
        for community_id, members in enumerate(partition):
            for node_idx in members:
                node_name = graph.vs[node_idx]["name"]
                assignments[node_name] = community_id

        await self._write_community_ids(assignments)

        return {
            "community_count": community_count,
            "modularity": modularity,
            "node_count": graph.vcount(),
        }

    async def _export_semantic_edges(self) -> tuple[list[tuple[str, str]], set[str]]:
        """Export semantic relationship edges from Neo4j.

        Only includes relationships defined in the extraction schema
        (ADDRESSES, REQUIRES, COMPONENT_OF, etc.), excluding structural
        relationships like FROM_ARTICLE, MENTIONED_IN, HAS_CHAPTER.

        Returns:
            Tuple of (edge_list, node_names) where edges are (source, target) tuples.
        """
        rel_type_filter = "|".join(_SEMANTIC_REL_TYPES)
        query = f"""
            MATCH (a)-[r:{rel_type_filter}]->(b)
            WHERE a.name IS NOT NULL AND b.name IS NOT NULL
            RETURN DISTINCT a.name AS source, b.name AS target
        """
        edges = []
        node_names: set[str] = set()

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            async for record in result:
                source = record["source"]
                target = record["target"]
                edges.append((source, target))
                node_names.add(source)
                node_names.add(target)

        logger.info(
            "Exported semantic edges",
            edge_count=len(edges),
            node_count=len(node_names),
            rel_types=_SEMANTIC_REL_TYPES,
        )
        return edges, node_names

    async def _write_community_ids(self, assignments: dict[str, int]) -> None:
        """Write community IDs to entity nodes in Neo4j.

        Args:
            assignments: Mapping of entity name to community ID.
        """
        query = """
            UNWIND $assignments AS assignment
            MATCH (n)
            WHERE n.name = assignment.name
            SET n.communityId = assignment.communityId
        """
        # Convert dict to list of dicts for UNWIND
        assignment_list = [{"name": name, "communityId": cid} for name, cid in assignments.items()]

        # Batch in groups of 500
        batch_size = 500
        for i in range(0, len(assignment_list), batch_size):
            batch = assignment_list[i : i + batch_size]
            async with self.driver.session(database=self.database) as session:
                await session.run(query, assignments=batch)

        logger.info(
            "Written community IDs to Neo4j",
            total_assignments=len(assignments),
        )
