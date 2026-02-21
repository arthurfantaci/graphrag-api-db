"""Voyage AI embeddings for community summaries.

Embeds Community node summaries using Voyage AI voyage-4 (1024d) so that
communities are discoverable via vector similarity search alongside chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)

# Voyage AI supports up to 128 texts per embed() call
_VOYAGE_BATCH_SIZE = 128


class CommunityEmbedder:
    """Embed community summaries with Voyage AI for vector retrieval.

    Queries Community nodes that have a summary but no summary_embedding,
    batches them through Voyage AI, and writes the vectors back to Neo4j.

    Attributes:
        driver: Async Neo4j driver.
        database: Neo4j database name.
        model: Voyage AI model name.
        dimensions: Embedding output dimensions.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str = "neo4j",
        model: str = "voyage-4",
        dimensions: int = 1024,
    ) -> None:
        """Initialize the community embedder.

        Args:
            driver: Async Neo4j driver.
            database: Neo4j database name.
            model: Voyage AI model name.
            dimensions: Embedding output dimensions.
        """
        self.driver = driver
        self.database = database
        self.model = model
        self.dimensions = dimensions

    async def embed_community_summaries(self) -> dict[str, Any]:
        """Embed all community summaries that lack embeddings.

        Fetches communities missing summary_embedding, embeds in batches
        via Voyage AI, and writes vectors back to Neo4j.

        Returns:
            Statistics dict with embedded and skipped counts.
        """
        import voyageai

        client = voyageai.AsyncClient()

        # Fetch communities needing embeddings
        communities = await self._get_communities_without_embeddings()

        if not communities:
            logger.info("All communities already have embeddings")
            return {"embedded": 0, "skipped": 0, "total": 0}

        total_embedded = 0

        # Process in batches
        for batch_start in range(0, len(communities), _VOYAGE_BATCH_SIZE):
            batch = communities[batch_start : batch_start + _VOYAGE_BATCH_SIZE]
            texts = [c["summary"] for c in batch]
            community_ids = [c["communityId"] for c in batch]

            result = await client.embed(
                texts,
                model=self.model,
                input_type="document",
                output_dimension=self.dimensions,
            )

            # Write embeddings back to Neo4j
            for cid, embedding in zip(community_ids, result.embeddings, strict=True):
                await self._set_embedding(cid, embedding)

            total_embedded += len(batch)

        logger.info(
            "Community summary embedding complete",
            embedded=total_embedded,
            already_had_embeddings=0,
        )

        return {
            "embedded": total_embedded,
            "skipped": 0,
            "total": total_embedded,
        }

    async def _get_communities_without_embeddings(self) -> list[dict[str, Any]]:
        """Fetch Community nodes that have a summary but no embedding.

        Returns:
            List of dicts with communityId and summary.
        """
        query = """
            MATCH (c:Community)
            WHERE c.summary IS NOT NULL AND c.summary_embedding IS NULL
            RETURN c.communityId AS communityId, c.summary AS summary
            ORDER BY c.communityId
        """
        communities: list[dict[str, Any]] = []
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            async for record in result:
                communities.append(
                    {
                        "communityId": record["communityId"],
                        "summary": record["summary"],
                    }
                )
        return communities

    async def _set_embedding(
        self,
        community_id: int,
        embedding: list[float],
    ) -> None:
        """Write a summary_embedding to a Community node.

        Args:
            community_id: The community ID.
            embedding: The embedding vector.
        """
        query = """
            MATCH (c:Community {communityId: $community_id})
            SET c.summary_embedding = $embedding
        """
        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                community_id=community_id,
                embedding=embedding,
            )
