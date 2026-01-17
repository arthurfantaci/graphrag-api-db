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
    from neo4j import Driver

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

        Args:
            filepath: Path to save the report.
        """
        filepath.write_text(self.to_markdown())
        logger.info("Saved validation report", path=str(filepath))


class ValidationReporter:
    """Generator for validation reports.

    Creates structured reports from validation query results.

    Example:
        >>> reporter = ValidationReporter(driver)
        >>> report = await reporter.generate()
        >>> report.save(Path("validation_report.md"))
    """

    def __init__(self, driver: "Driver", database: str = "neo4j") -> None:
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
        from jama_scraper.validation.queries import run_all_validations

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

        if not recommendations:
            recommendations.append("No issues found - graph looks healthy!")

        return recommendations


async def generate_validation_report(
    driver: "Driver",
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
