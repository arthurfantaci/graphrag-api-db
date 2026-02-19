"""LLM-generated community summaries.

After Leiden community detection, this module generates natural language
summaries for each community using gpt-4o-mini, stored as Community nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = structlog.get_logger(__name__)


class CommunitySummarizer:
    """Generate LLM summaries for entity communities.

    Queries each community's members, builds a prompt, and creates
    a Community node with the generated summary.

    Attributes:
        driver: Async Neo4j driver.
        database: Neo4j database name.
        openai_api_key: OpenAI API key.
        model: LLM model name for summarization.
        min_community_size: Minimum members to summarize.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str = "neo4j",
        openai_api_key: str = "",
        model: str = "gpt-4o-mini",
        min_community_size: int = 3,
    ) -> None:
        """Initialize the community summarizer.

        Args:
            driver: Async Neo4j driver.
            database: Neo4j database name.
            openai_api_key: OpenAI API key.
            model: LLM model for summarization.
            min_community_size: Skip communities smaller than this.
        """
        self.driver = driver
        self.database = database
        self.openai_api_key = openai_api_key
        self.model = model
        self.min_community_size = min_community_size

    async def summarize_communities(self) -> dict[str, Any]:
        """Generate summaries for all communities.

        Queries community members, generates LLM summaries, and creates
        Community nodes linked to member entities.

        Returns:
            Statistics dict with communities_summarized and skipped counts.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        communities = await self._get_communities()
        summarized = 0
        skipped = 0

        for community_id, members in communities.items():
            if len(members) < self.min_community_size:
                skipped += 1
                continue

            member_descriptions = []
            for m in members[:20]:  # Cap at 20 members for prompt length
                desc = f"- {m['name']} ({m['label']})"
                if m.get("description"):
                    desc += f": {m['description']}"
                member_descriptions.append(desc)

            prompt = (
                "You are summarizing a community of related entities from a "
                "requirements management knowledge graph. Based on the entity "
                "names and types below, write a 2-3 sentence summary describing "
                "what this community represents and how the entities relate.\n\n"
                "Community members:\n" + "\n".join(member_descriptions) + "\n\nSummary:"
            )

            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                )
                summary = response.choices[0].message.content.strip()

                await self._create_community_node(community_id, summary, members)
                summarized += 1

            except Exception:
                logger.warning(
                    "Failed to summarize community",
                    community_id=community_id,
                    exc_info=True,
                )

        logger.info(
            "Community summarization complete",
            summarized=summarized,
            skipped=skipped,
        )
        return {"communities_summarized": summarized, "communities_skipped": skipped}

    async def _get_communities(self) -> dict[int, list[dict[str, str]]]:
        """Query Neo4j for community assignments.

        Returns:
            Dict mapping community_id to list of member dicts (name, label, description).
        """
        query = """
            MATCH (n)
            WHERE n.communityId IS NOT NULL AND n.name IS NOT NULL
            RETURN n.communityId AS communityId,
                   n.name AS name,
                   head(labels(n)) AS label,
                   n.description AS description
            ORDER BY n.communityId
        """
        communities: dict[int, list[dict[str, str]]] = {}
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            async for record in result:
                cid = record["communityId"]
                if cid not in communities:
                    communities[cid] = []
                communities[cid].append(
                    {
                        "name": record["name"],
                        "label": record["label"],
                        "description": record["description"] or "",
                    }
                )
        return communities

    async def _create_community_node(
        self,
        community_id: int,
        summary: str,
        members: list[dict[str, str]],
    ) -> None:
        """Create a Community node and link to member entities.

        Args:
            community_id: The community ID.
            summary: LLM-generated summary text.
            members: List of member entity dicts.
        """
        query = """
            MERGE (c:Community {communityId: $community_id})
            SET c.summary = $summary,
                c.member_count = $member_count
            WITH c
            UNWIND $member_names AS member_name
            MATCH (n) WHERE n.name = member_name AND n.communityId = $community_id
            MERGE (n)-[:IN_COMMUNITY]->(c)
        """
        member_names = [m["name"] for m in members]
        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                community_id=community_id,
                summary=summary,
                member_count=len(members),
                member_names=member_names,
            )
