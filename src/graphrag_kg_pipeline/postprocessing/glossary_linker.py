"""Glossary to concept linking utilities.

This module provides utilities for linking glossary definitions
to Concept entities extracted from articles.
"""

from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz, process
import structlog

if TYPE_CHECKING:
    from neo4j import Driver

    from graphrag_kg_pipeline.models import Glossary

logger = structlog.get_logger(__name__)


class GlossaryConceptLinker:
    """Links glossary definitions to extracted Concept entities.

    Creates DEFINES relationships between Definition nodes and
    Concept nodes based on term matching.

    Example:
        >>> linker = GlossaryConceptLinker(driver)
        >>> stats = await linker.link_definitions()
        >>> print(f"Created {stats['links_created']} DEFINES relationships")
    """

    def __init__(
        self,
        driver: "Driver",
        database: str = "neo4j",
        match_threshold: int = 85,
    ) -> None:
        """Initialize the linker.

        Args:
            driver: Neo4j driver instance.
            database: Database name.
            match_threshold: Minimum fuzzy match score (0-100).
        """
        self.driver = driver
        self.database = database
        self.match_threshold = match_threshold

    async def link_definitions(self) -> dict[str, Any]:
        """Link Definition nodes to matching Concept nodes.

        Creates (Definition)-[DEFINES_CONCEPT]->(Concept) relationships
        where the definition term matches the concept name.

        Returns:
            Statistics about linking.
        """
        stats = {
            "definitions_processed": 0,
            "concepts_found": 0,
            "links_created": 0,
            "unmatched": [],
        }

        # Get all definitions
        definitions = await self._get_definitions()
        stats["definitions_processed"] = len(definitions)

        # Get all concepts
        concepts = await self._get_concepts()
        concept_names = {c["name"]: c["element_id"] for c in concepts}

        # Link each definition to matching concepts
        for definition in definitions:
            term = definition["term"].lower()

            # Try exact match first
            if term in concept_names:
                await self._create_link(
                    definition["element_id"],
                    concept_names[term],
                )
                stats["links_created"] += 1
                stats["concepts_found"] += 1
                continue

            # Try fuzzy match
            match = process.extractOne(
                term,
                concept_names.keys(),
                scorer=fuzz.ratio,
            )

            if match and match[1] >= self.match_threshold:
                await self._create_link(
                    definition["element_id"],
                    concept_names[match[0]],
                )
                stats["links_created"] += 1
                stats["concepts_found"] += 1
            else:
                stats["unmatched"].append(term)

        logger.info(
            "Glossary linking complete",
            definitions=stats["definitions_processed"],
            links=stats["links_created"],
            unmatched=len(stats["unmatched"]),
        )

        return stats

    async def _get_definitions(self) -> list[dict]:
        """Get all Definition nodes.

        Returns:
            List of definition properties.
        """
        query = """
        MATCH (d:Definition)
        RETURN d.term AS term, d.definition AS definition,
               elementId(d) AS element_id
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def _get_concepts(self) -> list[dict]:
        """Get all Concept nodes.

        Returns:
            List of concept properties.
        """
        query = """
        MATCH (c:Concept)
        RETURN c.name AS name, c.display_name AS display_name,
               elementId(c) AS element_id
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(query)
            return [dict(record) async for record in result]

    async def _create_link(
        self,
        definition_id: str,
        concept_id: str,
    ) -> None:
        """Create DEFINES_CONCEPT relationship.

        Args:
            definition_id: Element ID of Definition node.
            concept_id: Element ID of Concept node.
        """
        query = """
        MATCH (d:Definition) WHERE elementId(d) = $definition_id
        MATCH (c:Concept) WHERE elementId(c) = $concept_id
        MERGE (d)-[:DEFINES_CONCEPT]->(c)
        """

        async with self.driver.session(database=self.database) as session:
            await session.run(
                query,
                definition_id=definition_id,
                concept_id=concept_id,
            )


async def link_glossary_to_concepts(
    driver: "Driver",
    database: str = "neo4j",
    match_threshold: int = 85,
) -> dict[str, Any]:
    """Convenience function for glossary linking.

    Args:
        driver: Neo4j driver.
        database: Database name.
        match_threshold: Minimum fuzzy match score.

    Returns:
        Linking statistics.
    """
    linker = GlossaryConceptLinker(
        driver=driver,
        database=database,
        match_threshold=match_threshold,
    )
    return await linker.link_definitions()


def find_concept_matches_for_glossary(
    glossary: "Glossary",
    extracted_concepts: list[str],
    threshold: int = 85,
) -> dict[str, list[str]]:
    """Find concept matches for glossary terms (offline).

    Utility function for matching glossary terms to extracted
    concept names without Neo4j.

    Args:
        glossary: Glossary with terms.
        extracted_concepts: List of extracted concept names.
        threshold: Minimum match score.

    Returns:
        Mapping from term to list of matching concepts.
    """
    matches: dict[str, list[str]] = {}

    for term_obj in glossary.terms:
        term = term_obj.term.lower()
        matches[term] = []

        # Exact match
        if term in extracted_concepts:
            matches[term].append(term)
            continue

        # Fuzzy matches
        results = process.extract(
            term,
            extracted_concepts,
            scorer=fuzz.ratio,
            limit=3,
        )

        for match_result in results:
            if match_result[1] >= threshold:
                matches[term].append(match_result[0])

    return matches
