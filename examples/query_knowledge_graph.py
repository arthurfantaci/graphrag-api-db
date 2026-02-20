#!/usr/bin/env python3
"""Query the Jama Guide knowledge graph with four retrieval strategies.

Demonstrates four approaches to querying a Neo4j knowledge graph built by
the graphrag-kg pipeline:

1. **Vector similarity search** — semantic matching on chunk embeddings
2. **Chunk-to-entity traversal** — graph traversal from retrieved chunks to entities
3. **Direct entity search** — find entities by name pattern
4. **Relationship exploration** — show connections for a specific entity

Usage:
    uv run python examples/query_knowledge_graph.py
    uv run python examples/query_knowledge_graph.py "What is impact analysis?"
    uv run python examples/query_knowledge_graph.py "What is impact analysis?" --search traceability

Requires:
    - A populated Neo4j database (run ``graphrag-kg`` first)
    - Environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
    - Either VOYAGE_API_KEY (preferred) or OPENAI_API_KEY for embeddings
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import GraphDatabase
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.embeddings.base import Embedder
    from neo4j_graphrag.types import RetrieverResult

load_dotenv()

console = Console()

DEFAULT_QUERY = "What can you tell me about Requirements Tracing?"
DEFAULT_SEARCH_TERM = "traceability"


def get_driver() -> Driver:
    """Create a Neo4j driver from environment variables.

    Returns:
        A Neo4j driver instance configured from NEO4J_URI, NEO4J_USERNAME,
        and NEO4J_PASSWORD environment variables.
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(username, password))


def create_embedder() -> Embedder:
    """Create an embedder based on available API keys.

    Prefers Voyage AI (voyage-4, 1024d) when VOYAGE_API_KEY is set,
    falling back to OpenAI text-embedding-3-small.

    Returns:
        An Embedder instance for vector similarity search.

    Raises:
        RuntimeError: If neither VOYAGE_API_KEY nor OPENAI_API_KEY is set.
    """
    if os.getenv("VOYAGE_API_KEY"):
        from graphrag_kg_pipeline.embeddings.voyage import VoyageAIEmbeddings

        return VoyageAIEmbeddings(input_type="query")

    if os.getenv("OPENAI_API_KEY"):
        from neo4j_graphrag.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    msg = "Set VOYAGE_API_KEY or OPENAI_API_KEY in your .env file"
    raise RuntimeError(msg)


def vector_search(driver: Driver, query: str, top_k: int = 5) -> RetrieverResult:
    """Perform vector similarity search on chunk embeddings.

    Args:
        driver: Neo4j driver instance.
        query: Natural language query string.
        top_k: Number of results to return.

    Returns:
        Retriever result containing matched chunks with similarity scores.
    """
    from neo4j_graphrag.retrievers import VectorRetriever

    embedder = create_embedder()
    retriever = VectorRetriever(
        driver=driver,
        index_name="chunk_embeddings",
        embedder=embedder,
        return_properties=["text"],
    )
    return retriever.search(query_text=query, top_k=top_k)


def get_entities_from_chunks(driver: Driver, chunk_ids: list[str]) -> list[dict]:
    """Find entities mentioned in the retrieved chunks.

    Traverses MENTIONED_IN relationships from chunks to discover which
    domain entities (Concepts, Challenges, etc.) appear in the results.

    Args:
        driver: Neo4j driver instance.
        chunk_ids: List of Neo4j element IDs for retrieved chunks.

    Returns:
        List of entity dicts with name, label, definition, and mention count.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Chunk)<-[:MENTIONED_IN]-(entity)
            WHERE elementId(c) IN $chunk_ids
            WITH entity, labels(entity)[0] AS label, count(*) AS mentions
            RETURN label, entity.name AS name, entity.display_name AS display_name,
                   entity.definition AS definition, mentions
            ORDER BY mentions DESC, label, name
            LIMIT 20
            """,
            chunk_ids=chunk_ids,
        )
        return [dict(record) for record in result]


def search_entities_by_name(driver: Driver, search_term: str) -> list[dict]:
    """Search for entities whose name contains the given term.

    Searches across all LLM-extracted entity types (Concept, Challenge,
    BestPractice, Standard, Methodology, Artifact, Tool) using a
    case-sensitive CONTAINS match on the lowercased name.

    Args:
        driver: Neo4j driver instance.
        search_term: Substring to search for in entity names.

    Returns:
        List of entity dicts with name, label, definition, and connection count.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE (n:Concept OR n:Challenge OR n:Bestpractice OR n:Standard
                   OR n:Methodology OR n:Artifact OR n:Tool)
                  AND (n.name CONTAINS $term OR n.display_name CONTAINS $term)
            WITH n, labels(n)[0] AS label
            OPTIONAL MATCH (n)-[r]-(related)
            WITH n, label, count(DISTINCT related) AS connections
            RETURN label, n.name AS name, n.display_name AS display_name,
                   n.definition AS definition, connections
            ORDER BY connections DESC
            LIMIT 10
            """,
            term=search_term.lower(),
        )
        return [dict(record) for record in result]


def get_related_entities(driver: Driver, entity_name: str) -> list[dict]:
    """Get all entities related to a specific entity via any relationship.

    Args:
        driver: Neo4j driver instance.
        entity_name: Lowercase entity name to look up.

    Returns:
        List of relationship dicts with type, direction, and related entity info.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n {name: $name})-[r]-(related)
            WITH type(r) AS rel_type,
                 labels(related)[0] AS related_label,
                 related.name AS related_name,
                 related.display_name AS related_display,
                 startNode(r) = n AS outgoing
            RETURN rel_type,
                   CASE WHEN outgoing THEN '->' ELSE '<-' END AS direction,
                   related_label, related_name, related_display
            ORDER BY rel_type, related_label
            """,
            name=entity_name.lower(),
        )
        return [dict(record) for record in result]


def display_vector_results(results: RetrieverResult) -> list[str | None]:
    """Display vector search results and return chunk element IDs.

    Args:
        results: Retriever result from vector similarity search.

    Returns:
        List of chunk element IDs (some may be None if metadata is missing).
    """
    console.print("\n[bold green]1. Vector Similarity Search[/] (semantic match)")
    console.print("-" * 60)

    chunk_ids = []
    for i, item in enumerate(results.items, 1):
        chunk_ids.append(item.metadata.get("element_id") if item.metadata else None)
        text = (
            item.content.get("text", "")[:300]
            if isinstance(item.content, dict)
            else str(item.content)[:300]
        )
        score = item.metadata.get("score", 0.0) if item.metadata else 0.0
        console.print(f"\n[yellow]Result {i}[/] (score: {score:.3f})")
        console.print(f"{text}...")
    return chunk_ids


def display_entity_table(entities: list[dict], title: str, count_col: str) -> None:
    """Display entities in a formatted Rich table.

    Args:
        entities: List of entity dicts from a query result.
        title: Section title displayed above the table.
        count_col: Header label for the numeric column (e.g., "Mentions").
    """
    console.print(f"\n\n[bold green]{title}[/]")
    console.print("-" * 60)

    if not entities:
        console.print("[dim]No entities found[/]")
        return

    table = Table(show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column(count_col, justify="right")
    table.add_column("Definition", max_width=50)

    for record in entities:
        definition = record.get("definition") or ""
        table.add_row(
            record["label"],
            record.get("display_name") or record["name"],
            str(record.get("mentions") or record.get("connections", 0)),
            f"{definition[:50]}..." if definition else "-",
        )
    console.print(table)


def display_relationships(relationships: list[dict], entity_name: str) -> None:
    """Display relationships for an entity in a formatted Rich table.

    Args:
        relationships: List of relationship dicts from get_related_entities.
        entity_name: The entity name used as the query center.
    """
    console.print(f"\n[bold green]4. Relationships for '{entity_name}'[/]")
    console.print("-" * 60)

    if not relationships:
        console.print("[dim]No relationships found[/]")
        return

    table = Table(show_header=True)
    table.add_column("Direction", justify="center")
    table.add_column("Relationship", style="magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Related Entity", style="green")

    for rel in relationships:
        table.add_row(
            rel["direction"],
            rel["rel_type"],
            rel["related_label"],
            rel.get("related_display") or rel["related_name"],
        )
    console.print(table)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with query and search_term attributes.
    """
    parser = argparse.ArgumentParser(
        description="Query the Jama Guide knowledge graph",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Natural language query for vector search (default: %(default)s)",
    )
    parser.add_argument(
        "--search",
        default=DEFAULT_SEARCH_TERM,
        help="Term for direct entity name search (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the knowledge graph query demonstration."""
    args = parse_args()

    console.print(Panel(f"[bold cyan]Query:[/] {args.query}", title="Jama Guide Knowledge Graph"))

    driver = get_driver()
    try:
        # 1. Vector similarity search
        results = vector_search(driver, args.query)
        chunk_ids = display_vector_results(results)

        # 2. Entities from retrieved chunks
        valid_chunk_ids = [cid for cid in chunk_ids if cid]
        entities = get_entities_from_chunks(driver, valid_chunk_ids) if valid_chunk_ids else []
        display_entity_table(entities, "2. Entities Mentioned in Retrieved Chunks", "Mentions")

        # 3. Direct entity search
        direct_results = search_entities_by_name(driver, args.search)
        display_entity_table(
            direct_results,
            f"3. Direct Entity Search (name contains '{args.search}')",
            "Connections",
        )

        # 4. Relationships for top result
        if direct_results:
            top_entity = direct_results[0]["name"]
            relationships = get_related_entities(driver, top_entity)
            display_relationships(relationships, top_entity)

        console.print("\n[bold green]Query complete.[/]")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
