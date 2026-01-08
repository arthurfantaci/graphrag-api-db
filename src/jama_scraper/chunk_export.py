"""Export chunks and embeddings to various formats.

This module provides export functionality for:
- chunks.jsonl: All chunks with metadata for RAG systems
- embeddings.jsonl: Chunk IDs with embedding vectors
- Neo4j CSV: nodes_chunks.csv, relationship CSVs
- Neo4j Cypher: Chunk node creation statements

The exports are designed for integration with:
- Vector databases (Pinecone, Weaviate, Neo4j vector index)
- Graph databases (Neo4j)
- RAG systems (LangChain, LlamaIndex)
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from .chunk_models import ChunkedGuide, EmbeddedGuideChunks

console = Console()


def _escape_cypher(value: str) -> str:
    """Escape a string for use in Cypher queries.

    Args:
        value: String to escape.

    Returns:
        Escaped string safe for Cypher.
    """
    if not value:
        return ""
    # Escape backslashes first, then quotes
    return value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


class ChunkExporter:
    """Export chunks and embeddings to various formats.

    Output formats:
    - chunks.jsonl: All chunks with full metadata
    - embeddings.jsonl: Chunk IDs with embedding vectors
    - Neo4j CSV: Bulk import files for chunks
    - Neo4j Cypher: CREATE statements for chunks

    Example:
        >>> exporter = ChunkExporter(Path("output"))
        >>> paths = exporter.export_all(chunked_guide, embeddings)
        >>> print(paths)
        {'chunks': Path('output/chunks.jsonl'), ...}
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize the exporter.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = output_dir
        self._neo4j_dir = output_dir / "neo4j"

    def export_all(
        self,
        chunked_guide: ChunkedGuide,
        embeddings: EmbeddedGuideChunks | None = None,
    ) -> dict[str, Path]:
        """Export to all formats.

        Args:
            chunked_guide: ChunkedGuide with chunks to export.
            embeddings: Optional embeddings to export.

        Returns:
            Dictionary mapping format names to file paths.
        """
        paths: dict[str, Path] = {}

        # Export chunks.jsonl
        paths["chunks"] = self.export_chunks_jsonl(chunked_guide)

        # Export embeddings.jsonl if available
        if embeddings:
            paths["embeddings"] = self.export_embeddings_jsonl(embeddings)

        # Export Neo4j files
        neo4j_paths = self.export_neo4j(chunked_guide)
        paths.update(neo4j_paths)

        console.print(f"\n[green]Chunk export complete: {len(paths)} files[/]")
        for name, path in paths.items():
            console.print(f"  {name}: {path.name}")

        return paths

    def export_chunks_jsonl(self, chunked_guide: ChunkedGuide) -> Path:
        """Export chunks to JSONL format.

        Each line is a self-contained JSON object with:
        - chunk_id, chunk_type, source_article_id
        - text content
        - entity_ids for GraphRAG linkage
        - metadata (heading, chapter, token_count)

        Args:
            chunked_guide: ChunkedGuide with chunks.

        Returns:
            Path to the exported file.
        """
        path = self.output_dir / "chunks.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for chunk in chunked_guide.chunks.values():
                record = {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type.value,
                    "source_article_id": chunk.source_article_id,
                    "source_section_index": chunk.source_section_index,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "entity_ids": chunk.entity_ids,
                    "heading": chunk.heading,
                    "chapter_number": chunk.chapter_number,
                    "article_title": chunk.article_title,
                    "token_count": chunk.token_count,
                    "word_count": chunk.word_count,
                    "entity_count": chunk.entity_count,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        console.print(
            f"Saved chunks to: {path} ({len(chunked_guide.chunks)} records)"
        )
        return path

    def export_embeddings_jsonl(
        self,
        embeddings: EmbeddedGuideChunks,
    ) -> Path:
        """Export embeddings to JSONL format.

        Each line contains:
        - chunk_id: Reference to chunk
        - embedding: Vector as list of floats
        - model_id: Embedding model used
        - embedded_at: Timestamp

        Args:
            embeddings: EmbeddedGuideChunks with vectors.

        Returns:
            Path to the exported file.
        """
        path = self.output_dir / "embeddings.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for embedded in embeddings.embeddings.values():
                record = {
                    "chunk_id": embedded.chunk_id,
                    "embedding": embedded.embedding,
                    "model_id": embedded.model_id,
                    "embedded_at": embedded.embedded_at.isoformat(),
                    "dimensions": embedded.dimensions,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        console.print(
            f"Saved embeddings to: {path} ({len(embeddings.embeddings)} records)"
        )
        return path

    def export_neo4j(self, chunked_guide: ChunkedGuide) -> dict[str, Path]:
        """Export to Neo4j CSV and Cypher formats.

        Creates:
        - nodes_chunks.csv: Chunk nodes
        - rels_article_chunk.csv: Article -[:HAS_CHUNK]-> Chunk
        - rels_chunk_entity.csv: Chunk -[:MENTIONS_ENTITY]-> Entity
        - chunks_import.cypher: Cypher CREATE statements

        Args:
            chunked_guide: ChunkedGuide with chunks.

        Returns:
            Dictionary of format names to file paths.
        """
        self._neo4j_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # CSV exports
        paths["csv_chunks"] = self._write_chunks_csv(chunked_guide)
        paths["csv_article_chunk"] = self._write_article_chunk_rels_csv(chunked_guide)
        paths["csv_chunk_entity"] = self._write_chunk_entity_rels_csv(chunked_guide)

        # Cypher export
        paths["cypher_chunks"] = self._write_chunks_cypher(chunked_guide)

        return paths

    def _write_chunks_csv(self, chunked_guide: ChunkedGuide) -> Path:
        """Write chunks node CSV for Neo4j bulk import.

        Args:
            chunked_guide: ChunkedGuide with chunks.

        Returns:
            Path to CSV file.
        """
        path = self._neo4j_dir / "nodes_chunks.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "chunk_id:ID(Chunk)",
                "chunk_type",
                "source_article_id",
                "heading",
                "chapter_number:int",
                "token_count:int",
                "word_count:int",
                "char_start:int",
                "char_end:int",
                "entity_count:int",
                ":LABEL",
            ])

            for chunk in chunked_guide.chunks.values():
                writer.writerow([
                    chunk.chunk_id,
                    chunk.chunk_type.value,
                    chunk.source_article_id,
                    chunk.heading or "",
                    chunk.chapter_number or 0,
                    chunk.token_count,
                    chunk.word_count,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.entity_count,
                    "Chunk",
                ])

        return path

    def _write_article_chunk_rels_csv(self, chunked_guide: ChunkedGuide) -> Path:
        """Write Article->Chunk relationships CSV.

        Args:
            chunked_guide: ChunkedGuide with chunks.

        Returns:
            Path to CSV file.
        """
        path = self._neo4j_dir / "rels_article_chunk.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ":START_ID(Article)",
                ":END_ID(Chunk)",
                ":TYPE",
            ])

            for article_id, chunk_ids in chunked_guide.article_to_chunks.items():
                for chunk_id in chunk_ids:
                    writer.writerow([
                        article_id,
                        chunk_id,
                        "HAS_CHUNK",
                    ])

        return path

    def _write_chunk_entity_rels_csv(self, chunked_guide: ChunkedGuide) -> Path:
        """Write Chunk->Entity relationships CSV.

        Args:
            chunked_guide: ChunkedGuide with chunks.

        Returns:
            Path to CSV file.
        """
        path = self._neo4j_dir / "rels_chunk_entity.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ":START_ID(Chunk)",
                ":END_ID(Entity)",
                ":TYPE",
            ])

            for chunk in chunked_guide.chunks.values():
                for entity_id in chunk.entity_ids:
                    writer.writerow([
                        chunk.chunk_id,
                        entity_id,
                        "MENTIONS_ENTITY",
                    ])

        return path

    def _write_chunks_cypher(self, chunked_guide: ChunkedGuide) -> Path:
        """Write Cypher CREATE statements for chunks.

        Args:
            chunked_guide: ChunkedGuide with chunks.

        Returns:
            Path to Cypher file.
        """
        path = self._neo4j_dir / "chunks_import.cypher"

        lines = [
            "// =================================================================",
            "// Chunk Import Script for GraphRAG",
            f"// Generated: {datetime.now(UTC).isoformat()}",
            "// =================================================================",
            "",
            "// Constraint for Chunk nodes",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS",
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;",
            "",
            "// =================================================================",
            "// CHUNK NODES",
            "// =================================================================",
            "",
        ]

        for chunk in chunked_guide.chunks.values():
            heading_escaped = _escape_cypher(chunk.heading or "")
            # Don't include full text in Cypher (too long), just reference
            lines.append(
                f"CREATE (:Chunk {{"
                f"chunk_id: '{chunk.chunk_id}', "
                f"chunk_type: '{chunk.chunk_type.value}', "
                f"source_article_id: '{chunk.source_article_id}', "
                f"heading: '{heading_escaped}', "
                f"chapter_number: {chunk.chapter_number or 0}, "
                f"token_count: {chunk.token_count}, "
                f"word_count: {chunk.word_count}, "
                f"char_start: {chunk.char_start}, "
                f"char_end: {chunk.char_end}, "
                f"entity_count: {chunk.entity_count}"
                f"}});"
            )

        lines.extend([
            "",
            "// =================================================================",
            "// ARTICLE -> CHUNK RELATIONSHIPS",
            "// =================================================================",
            "",
        ])

        for article_id, chunk_ids in chunked_guide.article_to_chunks.items():
            for chunk_id in chunk_ids:
                lines.append(
                    f"MATCH (a:Article {{article_id: '{article_id}'}})\n"
                    f"MATCH (c:Chunk {{chunk_id: '{chunk_id}'}})\n"
                    f"CREATE (a)-[:HAS_CHUNK]->(c);"
                )

        lines.extend([
            "",
            "// =================================================================",
            "// CHUNK -> ENTITY RELATIONSHIPS",
            "// =================================================================",
            "",
        ])

        for chunk in chunked_guide.chunks.values():
            for entity_id in chunk.entity_ids:
                lines.append(
                    f"MATCH (c:Chunk {{chunk_id: '{chunk.chunk_id}'}})\n"
                    f"MATCH (e:Entity {{entity_id: '{entity_id}'}})\n"
                    f"CREATE (c)-[:MENTIONS_ENTITY]->(e);"
                )

        path.write_text("\n".join(lines), encoding="utf-8")
        return path
