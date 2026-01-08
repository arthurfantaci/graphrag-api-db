"""Neo4j export utilities for enriched guide content.

Generates both Cypher scripts and CSV files for Neo4j import:
- Cypher scripts: Direct CREATE statements for small datasets
- CSV files: Bulk import format for large datasets using neo4j-admin import

Example:
    exporter = Neo4jExporter(output_dir=Path("./neo4j"))
    exporter.export_all(guide, enriched_guide)
"""

from __future__ import annotations

import csv
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

    from .graph_models import (
        EnrichedGuide,
        ExtractedEntity,
        ExtractedRelationship,
    )
    from .models import RequirementsManagementGuide

console = Console()


class Neo4jExporter:
    """Export enriched guide data to Neo4j import formats.

    Generates both Cypher scripts for direct execution and CSV files
    for bulk import using neo4j-admin import tool.

    Attributes:
        output_dir: Directory to write export files.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize exporter with output directory.

        Args:
            output_dir: Directory path for export files.
        """
        self.output_dir = output_dir
        self._neo4j_dir = output_dir / "neo4j"

    def export_all(
        self,
        guide: RequirementsManagementGuide,
        enriched: EnrichedGuide,
    ) -> dict[str, Path]:
        """Export all data to both Cypher and CSV formats.

        Args:
            guide: Original scraped guide data.
            enriched: LangExtract-enriched guide data.

        Returns:
            Dictionary of format names to output file paths.
        """
        self._neo4j_dir.mkdir(parents=True, exist_ok=True)

        console.print("[cyan]Exporting to Neo4j formats...[/]")

        paths = {}

        # Export Cypher script
        cypher_path = self._export_cypher(guide, enriched)
        paths["cypher"] = cypher_path

        # Export CSV files
        csv_paths = self._export_csv(guide, enriched)
        paths.update(csv_paths)

        console.print(f"[green]Neo4j export complete: {len(paths)} files[/]")
        for name, path in paths.items():
            console.print(f"  {name}: {path.name}")

        return paths

    def _export_cypher(
        self,
        guide: RequirementsManagementGuide,
        enriched: EnrichedGuide,
    ) -> Path:
        """Generate Cypher script with CREATE statements.

        Args:
            guide: Original guide data.
            enriched: Enriched guide data.

        Returns:
            Path to generated Cypher script.
        """
        lines: list[str] = []

        # Header
        lines.extend(
            [
                "// =================================================================",
                "// Neo4j Import Script for Jama Requirements Management Guide",
                f"// Generated: {datetime.now().isoformat()}",
                "// =================================================================",
                "",
                "// Drop existing data (optional - uncomment if needed)",
                "// MATCH (n) DETACH DELETE n;",
                "",
            ]
        )

        # Constraints
        lines.extend(self._generate_constraints())

        # Indexes
        lines.extend(self._generate_indexes())

        # Chapter nodes
        lines.extend(self._generate_chapter_nodes(guide))

        # Article nodes
        lines.extend(self._generate_article_nodes(guide, enriched))

        # Entity nodes
        lines.extend(self._generate_entity_nodes(enriched))

        # Glossary nodes
        lines.extend(self._generate_glossary_nodes(guide))

        # Relationships
        lines.extend(self._generate_relationships(guide, enriched))

        # Write script
        cypher_path = self._neo4j_dir / "neo4j_import.cypher"
        cypher_path.write_text("\n".join(lines), encoding="utf-8")

        return cypher_path

    def _generate_constraints(self) -> list[str]:
        """Generate uniqueness constraints."""
        return [
            "// =================================================================",
            "// CONSTRAINTS",
            "// =================================================================",
            "",
            "CREATE CONSTRAINT chapter_id IF NOT EXISTS",
            "FOR (c:Chapter) REQUIRE c.chapter_number IS UNIQUE;",
            "",
            "CREATE CONSTRAINT article_id IF NOT EXISTS",
            "FOR (a:Article) REQUIRE a.article_id IS UNIQUE;",
            "",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS",
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;",
            "",
            "CREATE CONSTRAINT glossary_term IF NOT EXISTS",
            "FOR (g:GlossaryTerm) REQUIRE g.term IS UNIQUE;",
            "",
        ]

    def _generate_indexes(self) -> list[str]:
        """Generate indexes for common queries."""
        return [
            "// =================================================================",
            "// INDEXES",
            "// =================================================================",
            "",
            "CREATE INDEX entity_type_idx IF NOT EXISTS",
            "FOR (e:Entity) ON (e.entity_type);",
            "",
            "CREATE INDEX entity_name_idx IF NOT EXISTS",
            "FOR (e:Entity) ON (e.name);",
            "",
            "CREATE INDEX article_title_idx IF NOT EXISTS",
            "FOR (a:Article) ON (a.title);",
            "",
            "CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS",
            "FOR (a:Article) ON EACH [a.summary];",
            "",
        ]

    def _generate_chapter_nodes(
        self,
        guide: RequirementsManagementGuide,
    ) -> list[str]:
        """Generate CREATE statements for chapter nodes."""
        lines = [
            "// =================================================================",
            "// CHAPTER NODES",
            "// =================================================================",
            "",
        ]

        for chapter in guide.chapters:
            title_escaped = _escape_cypher(chapter.title)
            url_escaped = _escape_cypher(chapter.overview_url)

            lines.append(
                f"CREATE (:Chapter {{"
                f"chapter_number: {chapter.chapter_number}, "
                f"title: '{title_escaped}', "
                f"url: '{url_escaped}'"
                f"}});"
            )

        lines.append("")
        return lines

    def _generate_article_nodes(
        self,
        guide: RequirementsManagementGuide,
        enriched: EnrichedGuide,
    ) -> list[str]:
        """Generate CREATE statements for article nodes."""
        lines = [
            "// =================================================================",
            "// ARTICLE NODES",
            "// =================================================================",
            "",
        ]

        for chapter in guide.chapters:
            for article in chapter.articles:
                title_escaped = _escape_cypher(article.title)
                url_escaped = _escape_cypher(article.url)

                # Get enriched summary if available
                enrichment = enriched.article_enrichments.get(article.article_id)
                summary = enrichment.summary if enrichment else article.summary or ""
                summary_escaped = _escape_cypher(summary[:500])  # Limit length

                lines.append(
                    f"CREATE (:Article {{"
                    f"article_id: '{article.article_id}', "
                    f"title: '{title_escaped}', "
                    f"url: '{url_escaped}', "
                    f"summary: '{summary_escaped}', "
                    f"word_count: {article.word_count}, "
                    f"chapter_number: {chapter.chapter_number}"
                    f"}});"
                )

        lines.append("")
        return lines

    def _generate_entity_nodes(self, enriched: EnrichedGuide) -> list[str]:
        """Generate CREATE statements for entity nodes."""
        lines = [
            "// =================================================================",
            "// ENTITY NODES",
            "// =================================================================",
            "",
        ]

        for entity in enriched.entities.values():
            lines.append(self._entity_to_cypher(entity))

        lines.append("")
        return lines

    def _entity_to_cypher(self, entity: ExtractedEntity) -> str:
        """Convert entity to Cypher CREATE statement."""
        # Entity type label (capitalized)
        type_label = entity.entity_type.value.replace("_", "").title()
        name_escaped = _escape_cypher(entity.name)
        source_escaped = _escape_cypher(entity.source_text[:200])

        # Build attributes string
        attrs = [
            f"entity_id: '{entity.entity_id}'",
            f"name: '{name_escaped}'",
            f"entity_type: '{entity.entity_type.value}'",
            f"source_text: '{source_escaped}'",
            f"source_article_id: '{entity.source_article_id}'",
            f"confidence: {entity.confidence}",
        ]

        # Add type-specific attributes
        for key, value in entity.attributes.items():
            value_escaped = _escape_cypher(str(value)[:200])
            attrs.append(f"{key}: '{value_escaped}'")

        attrs_str = ", ".join(attrs)
        return f"CREATE (:Entity:{type_label} {{{attrs_str}}});"

    def _generate_glossary_nodes(
        self,
        guide: RequirementsManagementGuide,
    ) -> list[str]:
        """Generate CREATE statements for glossary term nodes."""
        lines = [
            "// =================================================================",
            "// GLOSSARY TERM NODES",
            "// =================================================================",
            "",
        ]

        if guide.glossary:
            for term in guide.glossary.terms:
                term_escaped = _escape_cypher(term.term)
                definition_escaped = _escape_cypher(term.definition[:500])

                lines.append(
                    f"CREATE (:GlossaryTerm {{"
                    f"term: '{term_escaped}', "
                    f"definition: '{definition_escaped}'"
                    f"}});"
                )

        lines.append("")
        return lines

    def _generate_relationships(
        self,
        guide: RequirementsManagementGuide,
        enriched: EnrichedGuide,
    ) -> list[str]:
        """Generate MATCH/CREATE statements for relationships."""
        lines = [
            "// =================================================================",
            "// RELATIONSHIPS",
            "// =================================================================",
            "",
            "// Chapter -> Article relationships",
        ]

        # Chapter CONTAINS Article
        for chapter in guide.chapters:
            for article in chapter.articles:
                lines.append(
                    f"MATCH (c:Chapter {{chapter_number: {chapter.chapter_number}}})\n"
                    f"MATCH (a:Article {{article_id: '{article.article_id}'}})\n"
                    f"CREATE (c)-[:CONTAINS]->(a);"
                )

        lines.extend(["", "// Article -> Entity MENTIONS relationships"])

        # Article MENTIONS Entity
        for enrichment in enriched.article_enrichments.values():
            for entity in enrichment.entities:
                lines.append(
                    f"MATCH (a:Article {{article_id: '{enrichment.article_id}'}})\n"
                    f"MATCH (e:Entity {{entity_id: '{entity.entity_id}'}})\n"
                    f"CREATE (a)-[:MENTIONS {{confidence: {entity.confidence}}}]->(e);"
                )

        lines.extend(["", "// Semantic Entity -> Entity relationships"])

        # Entity relationships
        for rel in enriched.relationships:
            lines.append(self._relationship_to_cypher(rel))

        lines.append("")
        return lines

    def _relationship_to_cypher(self, rel: ExtractedRelationship) -> str:
        """Convert relationship to Cypher MATCH/CREATE statement."""
        props = [f"confidence: {rel.confidence}"]

        if rel.source_text:
            source_escaped = _escape_cypher(rel.source_text[:200])
            props.append(f"evidence: '{source_escaped}'")

        props_str = ", ".join(props)

        return (
            f"MATCH (s:Entity {{entity_id: '{rel.source_entity_id}'}})\n"
            f"MATCH (t:Entity {{entity_id: '{rel.target_entity_id}'}})\n"
            f"CREATE (s)-[:{rel.relationship_type.value} {{{props_str}}}]->(t);"
        )

    def _export_csv(
        self,
        guide: RequirementsManagementGuide,
        enriched: EnrichedGuide,
    ) -> dict[str, Path]:
        """Export to CSV files for neo4j-admin import.

        Args:
            guide: Original guide data.
            enriched: Enriched guide data.

        Returns:
            Dictionary of CSV type names to file paths.
        """
        paths = {}

        # Chapters CSV
        paths["csv_chapters"] = self._write_chapters_csv(guide)

        # Articles CSV
        paths["csv_articles"] = self._write_articles_csv(guide, enriched)

        # Entities CSV
        paths["csv_entities"] = self._write_entities_csv(enriched)

        # Glossary CSV
        if guide.glossary:
            paths["csv_glossary"] = self._write_glossary_csv(guide)

        # Relationship CSVs
        paths["csv_contains"] = self._write_contains_rels_csv(guide)
        paths["csv_mentions"] = self._write_mentions_rels_csv(enriched)
        paths["csv_semantic"] = self._write_semantic_rels_csv(enriched)

        return paths

    def _write_chapters_csv(
        self,
        guide: RequirementsManagementGuide,
    ) -> Path:
        """Write chapters node CSV."""
        path = self._neo4j_dir / "nodes_chapters.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "chapter_number:ID(Chapter)",
                    "title",
                    "url",
                    ":LABEL",
                ]
            )

            for chapter in guide.chapters:
                writer.writerow(
                    [
                        chapter.chapter_number,
                        chapter.title,
                        chapter.overview_url,
                        "Chapter",
                    ]
                )

        return path

    def _write_articles_csv(
        self,
        guide: RequirementsManagementGuide,
        enriched: EnrichedGuide,
    ) -> Path:
        """Write articles node CSV."""
        path = self._neo4j_dir / "nodes_articles.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "article_id:ID(Article)",
                    "title",
                    "url",
                    "summary",
                    "word_count:int",
                    "chapter_number:int",
                    ":LABEL",
                ]
            )

            for chapter in guide.chapters:
                for article in chapter.articles:
                    enrichment = enriched.article_enrichments.get(article.article_id)
                    summary = (
                        enrichment.summary if enrichment else (article.summary or "")
                    )

                    writer.writerow(
                        [
                            article.article_id,
                            article.title,
                            article.url,
                            summary[:500],
                            article.word_count,
                            chapter.chapter_number,
                            "Article",
                        ]
                    )

        return path

    def _write_entities_csv(self, enriched: EnrichedGuide) -> Path:
        """Write entities node CSV."""
        path = self._neo4j_dir / "nodes_entities.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "entity_id:ID(Entity)",
                    "name",
                    "entity_type",
                    "source_text",
                    "source_article_id",
                    "confidence:float",
                    "definition",
                    "vendor",
                    "organization",
                    "impact",
                    "benefit",
                    ":LABEL",
                ]
            )

            for entity in enriched.entities.values():
                # Type label for dual-label nodes
                type_label = entity.entity_type.value.replace("_", "").title()
                labels = f"Entity;{type_label}"

                writer.writerow(
                    [
                        entity.entity_id,
                        entity.name,
                        entity.entity_type.value,
                        entity.source_text[:200],
                        entity.source_article_id,
                        entity.confidence,
                        entity.attributes.get("definition", ""),
                        entity.attributes.get("vendor", ""),
                        entity.attributes.get("organization", ""),
                        entity.attributes.get("impact", ""),
                        entity.attributes.get("benefit", ""),
                        labels,
                    ]
                )

        return path

    def _write_glossary_csv(
        self,
        guide: RequirementsManagementGuide,
    ) -> Path:
        """Write glossary terms node CSV."""
        path = self._neo4j_dir / "nodes_glossary.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "term:ID(GlossaryTerm)",
                    "definition",
                    ":LABEL",
                ]
            )

            if guide.glossary:
                for term in guide.glossary.terms:
                    writer.writerow(
                        [
                            term.term,
                            term.definition[:500],
                            "GlossaryTerm",
                        ]
                    )

        return path

    def _write_contains_rels_csv(
        self,
        guide: RequirementsManagementGuide,
    ) -> Path:
        """Write Chapter-CONTAINS->Article relationships CSV."""
        path = self._neo4j_dir / "rels_contains.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ":START_ID(Chapter)",
                    ":END_ID(Article)",
                    ":TYPE",
                ]
            )

            for chapter in guide.chapters:
                for article in chapter.articles:
                    writer.writerow(
                        [
                            chapter.chapter_number,
                            article.article_id,
                            "CONTAINS",
                        ]
                    )

        return path

    def _write_mentions_rels_csv(self, enriched: EnrichedGuide) -> Path:
        """Write Article-MENTIONS->Entity relationships CSV."""
        path = self._neo4j_dir / "rels_mentions.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ":START_ID(Article)",
                    ":END_ID(Entity)",
                    "confidence:float",
                    ":TYPE",
                ]
            )

            for enrichment in enriched.article_enrichments.values():
                for entity in enrichment.entities:
                    writer.writerow(
                        [
                            enrichment.article_id,
                            entity.entity_id,
                            entity.confidence,
                            "MENTIONS",
                        ]
                    )

        return path

    def _write_semantic_rels_csv(self, enriched: EnrichedGuide) -> Path:
        """Write Entity-[TYPE]->Entity semantic relationships CSV."""
        path = self._neo4j_dir / "rels_semantic.csv"

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ":START_ID(Entity)",
                    ":END_ID(Entity)",
                    "confidence:float",
                    "evidence",
                    ":TYPE",
                ]
            )

            for rel in enriched.relationships:
                writer.writerow(
                    [
                        rel.source_entity_id,
                        rel.target_entity_id,
                        rel.confidence,
                        (rel.source_text or "")[:200],
                        rel.relationship_type.value,
                    ]
                )

        return path


def _escape_cypher(text: str) -> str:
    """Escape text for Cypher string literals.

    Handles single quotes and backslashes.

    Args:
        text: Text to escape.

    Returns:
        Escaped text safe for Cypher.
    """
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")


def generate_import_command(neo4j_dir: Path) -> str:
    """Generate neo4j-admin import command for CSV files.

    Args:
        neo4j_dir: Directory containing CSV files.

    Returns:
        Command string for neo4j-admin import.
    """
    return f"""
# Neo4j Admin Import Command
# Run this from the Neo4j installation directory

neo4j-admin database import full \\
  --nodes=Chapter={neo4j_dir}/nodes_chapters.csv \\
  --nodes=Article={neo4j_dir}/nodes_articles.csv \\
  --nodes=Entity={neo4j_dir}/nodes_entities.csv \\
  --nodes=GlossaryTerm={neo4j_dir}/nodes_glossary.csv \\
  --relationships=CONTAINS={neo4j_dir}/rels_contains.csv \\
  --relationships=MENTIONS={neo4j_dir}/rels_mentions.csv \\
  --relationships={neo4j_dir}/rels_semantic.csv \\
  --overwrite-destination \\
  neo4j
""".strip()
