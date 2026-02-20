"""Validation report generation.

This module provides utilities for generating human-readable
validation reports from query results.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from neo4j import AsyncDriver, Driver

    # Accept either sync or async driver
    AnyDriver = Driver | AsyncDriver

logger = structlog.get_logger(__name__)


@dataclass
class ValidationReport:
    """Structured validation report.

    Attributes:
        timestamp: When the validation was run.
        validation_passed: Overall pass/fail status.
        summary: High-level summary statistics.
        details: Detailed validation results.
        recommendations: Suggested fixes for issues.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    validation_passed: bool = False
    summary: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report.

        Returns:
            Markdown-formatted report string.
        """
        lines = [
            "# Knowledge Graph Validation Report",
            "",
            f"**Generated:** {self.timestamp.isoformat()}",
            f"**Status:** {'✅ PASSED' if self.validation_passed else '❌ FAILED'}",
            "",
            "## Summary",
            "",
        ]

        # Summary table
        lines.append("| Check | Status |")
        lines.append("|-------|--------|")

        for check, passed in self.summary.items():
            status = (
                "✅" if not passed else "❌"
            )  # Note: summary flags are "has_X" so True = problem
            if check == "industry_count_ok":
                status = "✅" if passed else "❌"  # This one is positive
            lines.append(f"| {check.replace('_', ' ').title()} | {status} |")

        lines.append("")
        lines.append("## Details")
        lines.append("")

        # Entity statistics
        if "entity_stats" in self.details:
            lines.append("### Entity Counts")
            lines.append("")
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for label, count in sorted(
                self.details["entity_stats"].items(),
                key=lambda x: -x[1],
            ):
                lines.append(f"| {label} | {count} |")
            lines.append("")

        # Orphan chunks
        if self.details.get("orphan_chunks", 0) > 0:
            lines.append("### ⚠️ Orphan Chunks")
            lines.append("")
            lines.append(
                f"Found **{self.details['orphan_chunks']}** chunks not connected to any article."
            )
            lines.append("")

        # Duplicate entities
        if self.details.get("duplicate_entities"):
            lines.append("### ⚠️ Duplicate Entities")
            lines.append("")
            lines.append("| Label | Name | Count |")
            lines.append("|-------|------|-------|")
            for dup in self.details["duplicate_entities"][:10]:
                lines.append(f"| {dup['label']} | {dup['name']} | {dup['cnt']} |")
            lines.append("")

        # Missing embeddings
        if self.details.get("missing_embeddings", 0) > 0:
            lines.append("### ⚠️ Missing Embeddings")
            lines.append("")
            lines.append(
                f"Found **{self.details['missing_embeddings']}** chunks without embeddings."
            )
            lines.append("")

        # Missing chunk indices
        if self.details.get("missing_chunk_index", 0) > 0:
            lines.append("### ⚠️ Missing Chunk Index")
            lines.append("")
            lines.append(
                f"Found **{self.details['missing_chunk_index']}** chunks without index property."
            )
            lines.append("")

        # Missing chunk_ids
        if self.details.get("missing_chunk_ids", 0) > 0:
            lines.append("### ⚠️ Missing Chunk IDs")
            lines.append("")
            lines.append(
                f"Found **{self.details['missing_chunk_ids']}** chunks without chunk_id property."
            )
            lines.append("")
            lines.append("Run `graphrag-kg validate --fix` to generate chunk_ids.")
            lines.append("")

        # Degenerate chunks
        if self.details.get("degenerate_chunks"):
            lines.append("### ⚠️ Degenerate Chunks")
            lines.append("")
            lines.append(
                f"Found **{len(self.details['degenerate_chunks'])}** chunks "
                "with very short text and no entity relationships:"
            )
            lines.append("")
            for chunk in self.details["degenerate_chunks"][:5]:
                text_preview = (chunk.get("text") or "")[:60]
                lines.append(f"  - ({chunk.get('text_length', 0)} chars) `{text_preview}...`")
            if len(self.details["degenerate_chunks"]) > 5:
                remaining = len(self.details["degenerate_chunks"]) - 5
                lines.append(f"  - ... and {remaining} more")
            lines.append("")

        # Plural/singular duplicates
        if self.details.get("plural_singular_duplicates"):
            lines.append("### ⚠️ Plural/Singular Duplicates")
            lines.append("")
            lines.append(
                "Found entity pairs that differ only by plural suffix "
                "(e.g., 'requirement' vs 'requirements'):"
            )
            lines.append("")
            lines.append("| Label | Singular | Plural | Singular Rels | Plural Rels |")
            lines.append("|-------|----------|--------|---------------|-------------|")
            for dup in self.details["plural_singular_duplicates"][:15]:
                lines.append(
                    f"| {dup['label']} | {dup['singular_name']} | "
                    f"{dup['plural_name']} | {dup['singular_rels']} | "
                    f"{dup['plural_rels']} |"
                )
            if len(self.details["plural_singular_duplicates"]) > 15:
                remaining = len(self.details["plural_singular_duplicates"]) - 15
                lines.append(f"| ... | {remaining} more pairs | ... | ... | ... |")
            lines.append("")
            lines.append("Run `graphrag-kg validate --fix` to merge plurals into singulars.")
            lines.append("")

        # Generic entities
        if self.details.get("generic_entities"):
            lines.append("### ⚠️ Generic Entities")
            lines.append("")
            lines.append("Found overly generic entity names that provide no semantic value:")
            lines.append("")
            lines.append("| Label | Name | Relationships |")
            lines.append("|-------|------|---------------|")
            for entity in self.details["generic_entities"][:15]:
                lines.append(
                    f"| {entity['label']} | {entity['name']} | {entity['relationship_count']} |"
                )
            if len(self.details["generic_entities"]) > 15:
                remaining = len(self.details["generic_entities"]) - 15
                lines.append(f"| ... | {remaining} more | ... |")
            lines.append("")
            lines.append("Run `graphrag-kg validate --fix` to remove generic entities.")
            lines.append("")

        # Potentially mislabeled entities
        if self.details.get("potentially_mislabeled"):
            lines.append("### ⚠️ Potentially Mislabeled Challenges")
            lines.append("")
            lines.append(
                "Found Challenge nodes with positive-outcome names "
                "(may be goals/outcomes, not challenges):"
            )
            lines.append("")
            for entity in self.details["potentially_mislabeled"][:10]:
                lines.append(f"  - {entity['name']}")
            if len(self.details["potentially_mislabeled"]) > 10:
                remaining = len(self.details["potentially_mislabeled"]) - 10
                lines.append(f"  - ... and {remaining} more")
            lines.append("")

        # Entities without MENTIONED_IN
        if self.details.get("entities_without_mentioned_in"):
            count = len(self.details["entities_without_mentioned_in"])
            lines.append("### ⚠️ Entities Without MENTIONED_IN")
            lines.append("")
            lines.append(f"Found **{count}** entities not linked to any chunk via MENTIONED_IN:")
            lines.append("")
            # Group by label for compact display
            by_label: dict[str, int] = {}
            for entity in self.details["entities_without_mentioned_in"]:
                lbl = entity.get("label", "Unknown")
                by_label[lbl] = by_label.get(lbl, 0) + 1
            for lbl, cnt in sorted(by_label.items(), key=lambda x: -x[1]):
                lines.append(f"  - {lbl}: {cnt}")
            lines.append("")

        # Ghost entities (MENTIONED_IN only, no semantic rels)
        if self.details.get("entities_without_semantic_rels"):
            count = len(self.details["entities_without_semantic_rels"])
            lines.append("### ⚠️ Ghost Entities (No Semantic Relationships)")
            lines.append("")
            lines.append(
                f"Found **{count}** entities with MENTIONED_IN but no "
                "outbound semantic relationships:"
            )
            lines.append("")
            for entity in self.details["entities_without_semantic_rels"][:10]:
                lines.append(f"  - {entity.get('label', '?')}: {entity['name']}")
            if count > 10:
                lines.append(f"  - ... and {count - 10} more")
            lines.append("")

        # Near-duplicate entities
        if self.details.get("near_duplicates"):
            lines.append("### ⚠️ Near-Duplicate Entities")
            lines.append("")
            lines.append("| Label | Shorter | Longer |")
            lines.append("|-------|---------|--------|")
            for pair in self.details["near_duplicates"][:15]:
                lines.append(
                    f"| {pair['label']} | {pair['shorter_name']} | {pair['longer_name']} |"
                )
            if len(self.details["near_duplicates"]) > 15:
                remaining = len(self.details["near_duplicates"]) - 15
                lines.append(f"| ... | {remaining} more pairs | ... |")
            lines.append("")

        # Missing definitions
        if self.details.get("missing_definitions"):
            lines.append("### ⚠️ Missing Definitions")
            lines.append("")
            lines.append("| Label | Count |")
            lines.append("|-------|-------|")
            for item in self.details["missing_definitions"]:
                lines.append(f"| {item['label']} | {item['count']} |")
            lines.append("")

        # Truncated webinar titles
        if self.details.get("truncated_webinar_titles"):
            count = len(self.details["truncated_webinar_titles"])
            lines.append("### ⚠️ Truncated Webinar Titles")
            lines.append("")
            lines.append(f"Found **{count}** webinars with truncated or missing titles.")
            lines.append("")

        # Industry count
        lines.append("### Industry Count")
        lines.append("")
        industry_count = self.details.get("industry_count", 0)
        if industry_count <= 19:
            lines.append(f"✅ **{industry_count}** industries (target: ≤19)")
        else:
            lines.append(f"❌ **{industry_count}** industries (target: ≤19)")
        lines.append("")

        # Invalid patterns
        if self.details.get("invalid_patterns"):
            lines.append("### ⚠️ Invalid Relationship Patterns")
            lines.append("")
            lines.append("| Source | Relationship | Target | Count |")
            lines.append("|--------|--------------|--------|-------|")
            for pattern in self.details["invalid_patterns"][:10]:
                lines.append(
                    f"| {pattern['source_label']} | {pattern['rel_type']} | "
                    f"{pattern['target_label']} | {pattern['count']} |"
                )
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)

    def save(self, filepath: Path) -> None:
        """Save report to file.

        Archives any existing report with an ISO 8601 timestamp before
        writing the new one, so the latest report is always at the
        canonical path and previous versions are preserved.

        Args:
            filepath: Path to save the report.
        """
        if filepath.exists():
            self._archive_existing(filepath)
        filepath.write_text(self.to_markdown())
        logger.info("Saved validation report", path=str(filepath))

    @staticmethod
    def _archive_existing(filepath: Path) -> Path:
        """Rename an existing report with its modification timestamp.

        Args:
            filepath: Path to the existing report file.

        Returns:
            Path to the archived file.
        """
        mtime = filepath.stat().st_mtime
        ts = datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%dT%H%M%S")
        archived = filepath.with_name(f"{filepath.stem}_{ts}{filepath.suffix}")
        filepath.rename(archived)
        logger.info("Archived previous report", path=str(archived))
        return archived


class ValidationReporter:
    """Generator for validation reports.

    Creates structured reports from validation query results.

    Example:
        >>> reporter = ValidationReporter(driver)
        >>> report = await reporter.generate()
        >>> report.save(Path("validation_report.md"))
    """

    def __init__(self, driver: "AnyDriver", database: str = "neo4j") -> None:
        """Initialize the reporter.

        Args:
            driver: Neo4j driver.
            database: Database name.
        """
        self.driver = driver
        self.database = database

    async def generate(self) -> ValidationReport:
        """Generate a validation report.

        Returns:
            ValidationReport with all findings.
        """
        from graphrag_kg_pipeline.validation.queries import run_all_validations

        # Run all validations
        results = await run_all_validations(self.driver, self.database)

        # Build report
        report = ValidationReport(
            validation_passed=results["validation_passed"],
            summary=results["summary"],
            details={
                "orphan_chunks": results["orphan_chunks"],
                "orphan_entities": results["orphan_entities"],
                "duplicate_entities": results["duplicate_entities"],
                "missing_embeddings": results["missing_embeddings"],
                "industry_count": results["industry_count"],
                "entity_stats": results["entity_stats"],
                "invalid_patterns": results["invalid_patterns"],
                "article_coverage": results["article_coverage"],
                # Chunk quality
                "missing_chunk_ids": results.get("missing_chunk_ids", 0),
                "missing_chunk_index": results.get("missing_chunk_index", 0),
                "degenerate_chunks": results.get("degenerate_chunks", []),
                # Entity quality
                "plural_singular_duplicates": results.get("plural_singular_duplicates", []),
                "generic_entities": results.get("generic_entities", []),
                "entities_without_mentioned_in": results.get("entities_without_mentioned_in", []),
                "entities_without_semantic_rels": results.get("entities_without_semantic_rels", []),
                "potentially_mislabeled": results.get("potentially_mislabeled", []),
                "near_duplicates": results.get("near_duplicates", []),
                "missing_definitions": results.get("missing_definitions", []),
                # Webinar quality
                "truncated_webinar_titles": results.get("truncated_webinar_titles", []),
            },
            recommendations=self._generate_recommendations(results),
        )

        return report

    def _generate_recommendations(self, results: dict) -> list[str]:
        """Generate recommendations based on validation results.

        Args:
            results: Validation results.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if results["orphan_chunks"] > 0:
            recommendations.append(
                f"Run chunk-article linking to connect {results['orphan_chunks']} orphan chunks"
            )

        if len(results["duplicate_entities"]) > 0:
            recommendations.append(
                f"Run entity deduplication to merge {len(results['duplicate_entities'])} duplicate entity groups"
            )

        if results["missing_embeddings"] > 0:
            recommendations.append(
                f"Generate embeddings for {results['missing_embeddings']} chunks"
            )

        if results["industry_count"] > 19:
            recommendations.append(
                f"Run industry consolidation to reduce {results['industry_count']} industries to ≤19"
            )

        if len(results["invalid_patterns"]) > 0:
            recommendations.append(
                "Review and fix invalid relationship patterns (may indicate extraction issues)"
            )

        # Chunk quality recommendations
        if results.get("missing_chunk_index", 0) > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to assign {results['missing_chunk_index']} missing chunk indices"
            )

        if results.get("missing_chunk_ids", 0) > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to generate {results['missing_chunk_ids']} missing chunk_ids"
            )

        degenerate_count = len(results.get("degenerate_chunks", []))
        if degenerate_count > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to remove {degenerate_count} degenerate chunks"
            )

        # Entity quality recommendations
        plural_count = len(results.get("plural_singular_duplicates", []))
        if plural_count > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to merge {plural_count} plural/singular entity pairs"
            )

        generic_count = len(results.get("generic_entities", []))
        if generic_count > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to remove {generic_count} overly generic entities"
            )

        mislabeled_count = len(results.get("potentially_mislabeled", []))
        if mislabeled_count > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to relabel {mislabeled_count} potentially mislabeled Challenge entities"
            )

        no_mentioned_in = len(results.get("entities_without_mentioned_in", []))
        if no_mentioned_in > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to backfill MENTIONED_IN for {no_mentioned_in} disconnected entities"
            )

        missing_def_total = sum(
            item.get("count", 0) for item in results.get("missing_definitions", [])
        )
        if missing_def_total > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to backfill {missing_def_total} missing definitions from glossary"
            )

        webinar_count = len(results.get("truncated_webinar_titles", []))
        if webinar_count > 0:
            recommendations.append(
                f"Run `graphrag-kg validate --fix` to fix {webinar_count} truncated webinar titles"
            )

        # Advisory-only (no fix available)
        ghost_count = len(results.get("entities_without_semantic_rels", []))
        if ghost_count > 0:
            recommendations.append(
                f"Review {ghost_count} ghost entities with no semantic relationships (advisory)"
            )

        near_dup_count = len(results.get("near_duplicates", []))
        if near_dup_count > 0:
            recommendations.append(
                f"Review {near_dup_count} near-duplicate entity pairs for manual merge (advisory)"
            )

        if not recommendations:
            recommendations.append("No issues found - graph looks healthy!")

        return recommendations


async def generate_validation_report(
    driver: "AnyDriver",
    database: str = "neo4j",
    output_path: Path | None = None,
) -> ValidationReport:
    """Convenience function to generate and optionally save a report.

    Args:
        driver: Neo4j driver.
        database: Database name.
        output_path: Optional path to save report.

    Returns:
        Generated ValidationReport.
    """
    reporter = ValidationReporter(driver, database)
    report = await reporter.generate()

    if output_path:
        report.save(output_path)

    return report
