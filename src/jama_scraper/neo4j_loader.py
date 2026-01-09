"""Neo4j database loader for the Jama Guide data.

This module provides functionality to load scraped and enriched data
directly into a Neo4j database, supporting both AuraDB (cloud) and
self-hosted deployments.

Features:
- MERGE-based upserts for incremental updates
- Batch processing for performance
- Vector index creation for similarity search
- Support for Neo4j 5.x
"""

from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass, field
import json
import os
from typing import TYPE_CHECKING, Any

from neo4j import GraphDatabase
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from pathlib import Path

console = Console()

# Batch size for bulk operations
BATCH_SIZE = 500

# Vector index configuration
VECTOR_DIMENSIONS = 1536  # text-embedding-3-small
VECTOR_SIMILARITY = "cosine"


@dataclass
class LoadResult:
    """Results from a Neo4j load operation."""

    nodes_created: dict[str, int] = field(default_factory=dict)
    relationships_created: dict[str, int] = field(default_factory=dict)
    vector_index_created: bool = False
    embeddings_loaded: int = 0


class Neo4jLoader:
    """Load exported data into Neo4j database.

    Supports both Neo4j AuraDB (cloud) and self-hosted deployments.
    Uses MERGE for idempotent upserts, enabling incremental updates.

    Example:
        loader = Neo4jLoader(
            uri="neo4j+s://xxx.databases.neo4j.io",
            username="neo4j",
            password="password"
        )
        result = loader.load_all(Path("output"))
        loader.close()
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        """Initialize the Neo4j loader.

        Args:
            uri: Neo4j connection URI (bolt:// or neo4j+s://).
            username: Database username.
            password: Database password.
        """
        self.uri = uri
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify the database connection is working."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            console.print(f"[green]Connected to Neo4j: {self.uri}[/]")
        except Exception as e:
            console.print(f"[red]Failed to connect to Neo4j: {e}[/]")
            raise

    def close(self) -> None:
        """Close the database connection."""
        self.driver.close()

    def load_all(self, output_dir: Path) -> LoadResult:
        """Load all exported data into Neo4j.

        Args:
            output_dir: Directory containing exported files.

        Returns:
            LoadResult with counts of created nodes and relationships.
        """
        result = LoadResult()
        neo4j_dir = output_dir / "neo4j"

        if not neo4j_dir.exists():
            msg = f"Neo4j export directory not found: {neo4j_dir}"
            raise FileNotFoundError(msg)

        console.print("\n[bold cyan]Loading data into Neo4j...[/]")

        # Step 1: Create constraints and indexes
        self._create_constraints()

        # Step 2: Load nodes
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Chapters
            task = progress.add_task("Loading chapters...", total=None)
            count = self._load_nodes_from_csv(
                neo4j_dir / "nodes_chapters.csv", "Chapter", "chapter_number"
            )
            result.nodes_created["Chapter"] = count
            progress.update(task, description=f"Loaded {count} chapters")

            # Articles
            task = progress.add_task("Loading articles...", total=None)
            count = self._load_nodes_from_csv(
                neo4j_dir / "nodes_articles.csv", "Article", "article_id"
            )
            result.nodes_created["Article"] = count
            progress.update(task, description=f"Loaded {count} articles")

            # Entities (with type labels)
            task = progress.add_task("Loading entities...", total=None)
            count = self._load_entity_nodes(neo4j_dir / "nodes_entities.csv")
            result.nodes_created["Entity"] = count
            progress.update(task, description=f"Loaded {count} entities")

            # Glossary terms
            task = progress.add_task("Loading glossary...", total=None)
            count = self._load_nodes_from_csv(
                neo4j_dir / "nodes_glossary.csv", "GlossaryTerm", "term"
            )
            result.nodes_created["GlossaryTerm"] = count
            progress.update(task, description=f"Loaded {count} glossary terms")

            # Chunks (if available)
            chunks_csv = neo4j_dir / "nodes_chunks.csv"
            if chunks_csv.exists():
                task = progress.add_task("Loading chunks...", total=None)
                count = self._load_nodes_from_csv(chunks_csv, "Chunk", "chunk_id")
                result.nodes_created["Chunk"] = count
                progress.update(task, description=f"Loaded {count} chunks")

        # Step 3: Load relationships
        console.print("\n[cyan]Loading relationships...[/]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Chapter -> Article
            task = progress.add_task("Loading CONTAINS...", total=None)
            count = self._load_relationships_from_csv(
                neo4j_dir / "rels_contains.csv",
                "Chapter",
                "chapter_number",
                "Article",
                "article_id",
                "CONTAINS",
            )
            result.relationships_created["CONTAINS"] = count
            progress.update(task, description=f"Loaded {count} CONTAINS")

            # Article -> Entity (MENTIONS)
            task = progress.add_task("Loading MENTIONS...", total=None)
            count = self._load_relationships_from_csv(
                neo4j_dir / "rels_mentions.csv",
                "Article",
                "article_id",
                "Entity",
                "entity_id",
                "MENTIONS",
                properties=["confidence"],
            )
            result.relationships_created["MENTIONS"] = count
            progress.update(task, description=f"Loaded {count} MENTIONS")

            # Entity -> Entity (semantic relationships)
            rels_semantic = neo4j_dir / "rels_semantic.csv"
            if rels_semantic.exists():
                task = progress.add_task("Loading semantic rels...", total=None)
                count = self._load_semantic_relationships(rels_semantic)
                result.relationships_created["semantic"] = count
                progress.update(task, description=f"Loaded {count} semantic rels")

            # Article -> Chunk (HAS_CHUNK)
            rels_chunks = neo4j_dir / "rels_article_chunk.csv"
            if rels_chunks.exists():
                task = progress.add_task("Loading HAS_CHUNK...", total=None)
                count = self._load_relationships_from_csv(
                    rels_chunks,
                    "Article",
                    "article_id",
                    "Chunk",
                    "chunk_id",
                    "HAS_CHUNK",
                )
                result.relationships_created["HAS_CHUNK"] = count
                progress.update(task, description=f"Loaded {count} HAS_CHUNK")

            # Chunk -> Entity (MENTIONS_ENTITY)
            rels_chunk_entity = neo4j_dir / "rels_chunk_entity.csv"
            if rels_chunk_entity.exists():
                task = progress.add_task("Loading MENTIONS_ENTITY...", total=None)
                count = self._load_relationships_from_csv(
                    rels_chunk_entity,
                    "Chunk",
                    "chunk_id",
                    "Entity",
                    "entity_id",
                    "MENTIONS_ENTITY",
                )
                result.relationships_created["MENTIONS_ENTITY"] = count
                progress.update(task, description=f"Loaded {count} MENTIONS_ENTITY")

        # Step 4: Create vector index and load embeddings
        embeddings_path = output_dir / "embeddings.jsonl"
        if embeddings_path.exists():
            console.print("\n[cyan]Setting up vector search...[/]")
            self._create_vector_index()
            result.vector_index_created = True
            result.embeddings_loaded = self._load_embeddings(embeddings_path)

        # Print summary
        self._print_summary(result)
        return result

    def _create_constraints(self) -> None:
        """Create uniqueness constraints and indexes."""
        console.print("[cyan]Creating constraints and indexes...[/]")

        # fmt: off
        constraints = [
            "CREATE CONSTRAINT chapter_id IF NOT EXISTS "
            "FOR (c:Chapter) REQUIRE c.chapter_number IS UNIQUE",
            "CREATE CONSTRAINT article_id IF NOT EXISTS "
            "FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT glossary_term IF NOT EXISTS "
            "FOR (g:GlossaryTerm) REQUIRE g.term IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX entity_type_idx IF NOT EXISTS "
            "FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_name_idx IF NOT EXISTS "
            "FOR (e:Entity) ON (e.name)",
            "CREATE INDEX article_title_idx IF NOT EXISTS "
            "FOR (a:Article) ON (a.title)",
        ]
        # fmt: on

        with self.driver.session() as session:
            for constraint in constraints:
                with contextlib.suppress(Exception):
                    session.run(constraint)

            for index in indexes:
                with contextlib.suppress(Exception):
                    session.run(index)

    def _load_nodes_from_csv(
        self, csv_path: Path, label: str, id_property: str
    ) -> int:
        """Load nodes from CSV using MERGE.

        Args:
            csv_path: Path to the CSV file.
            label: Node label.
            id_property: Property to use as unique identifier (clean name).

        Returns:
            Number of nodes loaded.
        """
        if not csv_path.exists():
            return 0

        rows = self._read_csv(csv_path)
        if not rows:
            return 0

        # Transform rows to use clean property names
        # CSV headers like "chapter_number:ID(Chapter)" -> "chapter_number"
        clean_rows = []
        for row in rows:
            clean_row = {}
            for key, value in row.items():
                if key.startswith(":"):
                    continue  # Skip Neo4j format columns like :LABEL
                # Extract clean name: "prop:ID(X)" -> "prop", "prop:int" -> "prop"
                clean_key = key.split(":")[0]
                clean_row[clean_key] = value
            clean_rows.append(clean_row)

        if not clean_rows:
            return 0

        # Build MERGE query with clean property names
        sample = clean_rows[0]
        other_props = [p for p in sample if p != id_property]

        set_clause = ", ".join(f"n.{p} = row.{p}" for p in other_props)

        query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{{id_property}: row.{id_property}}})
        SET {set_clause}
        """

        return self._execute_batch(query, clean_rows)

    def _load_entity_nodes(self, csv_path: Path) -> int:
        """Load entity nodes with dual labels (Entity + type label).

        Args:
            csv_path: Path to the entities CSV.

        Returns:
            Number of entities loaded.
        """
        if not csv_path.exists():
            return 0

        rows = self._read_csv(csv_path)
        if not rows:
            return 0

        # Group by entity type for efficient label assignment
        count = 0
        with self.driver.session() as session:
            for batch in self._chunk_list(rows, BATCH_SIZE):
                result = session.execute_write(self._merge_entities, batch)
                count += result

        return count

    @staticmethod
    def _merge_entities(tx: Any, rows: list[dict]) -> int:
        """Merge entity nodes with type-specific labels."""
        count = 0
        for row in rows:
            # Extract entity type for secondary label
            entity_type = row.get("entity_type", "").lower()
            type_label = entity_type.replace("_", "").title() if entity_type else ""

            # Build properties, excluding Neo4j format columns
            props = {
                k.split(":")[0]: v
                for k, v in row.items()
                if not k.startswith(":") and v
            }

            # MERGE with Entity label, then add type label
            query = """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e += $props
            """
            tx.run(query, entity_id=props.get("entity_id"), props=props)

            # Add type-specific label if present
            if type_label:
                label_query = (
                    f"MATCH (e:Entity {{entity_id: $entity_id}}) SET e:{type_label}"
                )
                tx.run(label_query, entity_id=props.get("entity_id"))

            count += 1

        return count

    def _load_relationships_from_csv(
        self,
        csv_path: Path,
        start_label: str,
        start_id_prop: str,
        end_label: str,
        end_id_prop: str,
        rel_type: str,
        properties: list[str] | None = None,
    ) -> int:
        """Load relationships from CSV using MERGE.

        Args:
            csv_path: Path to the CSV file.
            start_label: Label of the start node.
            start_id_prop: ID property of start node.
            end_label: Label of the end node.
            end_id_prop: ID property of end node.
            rel_type: Relationship type.
            properties: Optional list of relationship properties.

        Returns:
            Number of relationships loaded.
        """
        if not csv_path.exists():
            return 0

        rows = self._read_csv(csv_path)
        if not rows:
            return 0

        # Build property SET clause if properties specified
        prop_clause = ""
        if properties:
            prop_clause = "SET " + ", ".join(f"r.{p} = row.{p}" for p in properties)

        query = f"""
        UNWIND $rows AS row
        MATCH (a:{start_label} {{{start_id_prop}: row.start_id}})
        MATCH (b:{end_label} {{{end_id_prop}: row.end_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        {prop_clause}
        """

        # Transform CSV columns to expected format
        transformed = []
        for row in rows:
            new_row = {
                "start_id": row.get(":START_ID(Chapter)")
                or row.get(":START_ID(Article)")
                or row.get(":START_ID(Entity)")
                or row.get(":START_ID(Chunk)")
                or row.get("start_id"),
                "end_id": row.get(":END_ID(Article)")
                or row.get(":END_ID(Entity)")
                or row.get(":END_ID(Chunk)")
                or row.get("end_id"),
            }
            # Add properties
            if properties:
                for prop in properties:
                    # Handle typed property names like confidence:float
                    for key in row:
                        if key.startswith(prop):
                            new_row[prop] = row[key]
                            break
            transformed.append(new_row)

        return self._execute_batch(query, transformed)

    def _load_semantic_relationships(self, csv_path: Path) -> int:
        """Load semantic Entity->Entity relationships.

        These have dynamic relationship types based on the :TYPE column.

        Args:
            csv_path: Path to the semantic relationships CSV.

        Returns:
            Number of relationships loaded.
        """
        if not csv_path.exists():
            return 0

        rows = self._read_csv(csv_path)
        if not rows:
            return 0

        count = 0
        with self.driver.session() as session:
            for batch in self._chunk_list(rows, BATCH_SIZE):
                for row in batch:
                    rel_type = row.get(":TYPE", "RELATED_TO")
                    start_id = row.get(":START_ID(Entity)")
                    end_id = row.get(":END_ID(Entity)")
                    confidence = row.get("confidence:float", row.get("confidence"))
                    evidence = row.get("evidence", "")

                    query = f"""
                    MATCH (a:Entity {{entity_id: $start_id}})
                    MATCH (b:Entity {{entity_id: $end_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.confidence = $confidence, r.evidence = $evidence
                    """
                    session.run(
                        query,
                        start_id=start_id,
                        end_id=end_id,
                        confidence=float(confidence) if confidence else 0.0,
                        evidence=evidence,
                    )
                    count += 1

        return count

    def _create_vector_index(self) -> None:
        """Create vector index for chunk embeddings."""
        console.print("  Creating vector index...")

        # Check Neo4j version for vector support
        with self.driver.session() as session:
            try:
                # Neo4j 5.11+ vector index syntax
                session.run(
                    f"""
                    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.embedding
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {VECTOR_DIMENSIONS},
                        `vector.similarity_function`: '{VECTOR_SIMILARITY}'
                    }}}}
                    """
                )
                console.print("  [green]Vector index created[/]")
            except Exception as e:
                if "vector" in str(e).lower():
                    console.print(
                        "  [yellow]Vector indexes require Neo4j 5.11+. "
                        "Embeddings will be stored but not indexed.[/]"
                    )
                else:
                    console.print(f"  [yellow]Could not create vector index: {e}[/]")

    def _load_embeddings(self, embeddings_path: Path) -> int:
        """Load embeddings from JSONL file into Chunk nodes.

        Args:
            embeddings_path: Path to embeddings.jsonl.

        Returns:
            Number of embeddings loaded.
        """
        console.print("  Loading embeddings...")

        embeddings = []
        with embeddings_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    embeddings.append(json.loads(line))

        count = 0
        # Smaller batches for embeddings (100 vs 500)
        with self.driver.session() as session:
            for batch in self._chunk_list(embeddings, 100):
                result = session.execute_write(self._set_embeddings, batch)
                count += result

        console.print(f"  [green]Loaded {count} embeddings[/]")
        return count

    @staticmethod
    def _set_embeddings(tx: Any, embeddings: list[dict]) -> int:
        """Set embedding vectors on Chunk nodes."""
        count = 0
        for emb in embeddings:
            tx.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                SET c.embedding = $embedding
                """,
                chunk_id=emb["chunk_id"],
                embedding=emb["embedding"],
            )
            count += 1
        return count

    def _execute_batch(self, query: str, rows: list[dict]) -> int:
        """Execute a query in batches.

        Args:
            query: Cypher query with $rows parameter.
            rows: List of row dictionaries.

        Returns:
            Total number of rows processed.
        """
        count = 0
        with self.driver.session() as session:
            for batch in self._chunk_list(rows, BATCH_SIZE):
                session.run(query, rows=batch)
                count += len(batch)
        return count

    def _read_csv(self, path: Path) -> list[dict]:
        """Read CSV file into list of dictionaries.

        Args:
            path: Path to CSV file.

        Returns:
            List of row dictionaries.
        """
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def _chunk_list(lst: list, size: int) -> list[list]:
        """Split list into chunks of specified size."""
        return [lst[i : i + size] for i in range(0, len(lst), size)]

    def _print_summary(self, result: LoadResult) -> None:
        """Print load summary."""
        console.print("\n[bold green]Neo4j load complete![/]")
        console.print("\nNodes loaded:")
        for label, count in result.nodes_created.items():
            console.print(f"  {label}: {count}")

        console.print("\nRelationships loaded:")
        for rel_type, count in result.relationships_created.items():
            console.print(f"  {rel_type}: {count}")

        if result.vector_index_created:
            emb_count = result.embeddings_loaded
            console.print(f"\nVector index: created ({emb_count} embeddings)")


def get_neo4j_config() -> tuple[str, str, str] | None:
    """Get Neo4j configuration from environment variables.

    Returns:
        Tuple of (uri, username, password) or None if not configured.
    """
    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        return None

    return uri, username, password


def check_neo4j_config() -> bool:
    """Check if Neo4j configuration is available.

    Returns:
        True if all required environment variables are set.
    """
    return get_neo4j_config() is not None
