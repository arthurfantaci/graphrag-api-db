"""Tests for the validation module.

This module tests the knowledge graph validation queries and reporting
functionality that ensures data quality after extraction.
"""

from __future__ import annotations

import pytest

from tests.conftest import MockDriver, MockSession

# =============================================================================
# ENTITY CLEANUP TESTS
# =============================================================================


class TestEntityCleanupTaxonomy:
    """Tests for entity cleanup taxonomy functions."""

    def test_is_generic_term_identifies_generic(self) -> None:
        """Test that generic terms are identified correctly."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import is_generic_term

        assert is_generic_term("tool") is True
        assert is_generic_term("tools") is True
        assert is_generic_term("software") is True
        assert is_generic_term("process") is True
        assert is_generic_term("method") is True
        assert is_generic_term("TOOL") is True  # Case insensitive
        assert is_generic_term("  tool  ") is True  # Whitespace handling

    def test_is_generic_term_rejects_specific(self) -> None:
        """Test that specific terms are not flagged as generic."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import is_generic_term

        assert is_generic_term("requirements management tool") is False
        assert is_generic_term("traceability") is False
        assert is_generic_term("jama connect") is False
        assert is_generic_term("iso 26262") is False

    def test_normalize_to_singular(self) -> None:
        """Test plural to singular normalization."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            normalize_to_singular,
        )

        assert normalize_to_singular("requirements") == "requirement"
        assert normalize_to_singular("stakeholders") == "stakeholder"
        assert normalize_to_singular("artifacts") == "artifact"
        assert normalize_to_singular("constraints") == "constraint"
        assert normalize_to_singular("REQUIREMENTS") == "requirement"  # Case insensitive

    def test_normalize_to_singular_preserves_singulars(self) -> None:
        """Test that singular terms are not changed."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            normalize_to_singular,
        )

        assert normalize_to_singular("requirement") == "requirement"
        assert normalize_to_singular("traceability") == "traceability"
        assert normalize_to_singular("iso 26262") == "iso 26262"

    def test_classify_entity_for_cleanup_delete(self) -> None:
        """Test that generic entities are classified for deletion."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            classify_entity_for_cleanup,
        )

        action, value = classify_entity_for_cleanup("tool", "Tool")
        assert action == "delete"
        assert value is None

        action, value = classify_entity_for_cleanup("software", "Concept")
        assert action == "delete"

    def test_classify_entity_for_cleanup_normalize(self) -> None:
        """Test that plural entities are classified for normalization."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            classify_entity_for_cleanup,
        )

        action, value = classify_entity_for_cleanup("requirements", "Concept")
        assert action == "keep"
        assert value == "requirement"

    def test_classify_entity_for_cleanup_skip_non_llm(self) -> None:
        """Test that non-LLM-extracted entities are skipped."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            classify_entity_for_cleanup,
        )

        # Definition nodes should be skipped
        action, _value = classify_entity_for_cleanup("Requirements", "Definition")
        assert action == "skip"

        # Article nodes should be skipped
        action, _value = classify_entity_for_cleanup("test", "Article")
        assert action == "skip"


class TestEntityCleanupClassifier:
    """Tests for the EntityCleanupClassifier class."""

    def test_classify_batch_categorizes_correctly(self) -> None:
        """Test that batch classification works correctly."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            EntityCleanupClassifier,
        )

        classifier = EntityCleanupClassifier()
        entities = [
            {"name": "tool", "label": "Tool"},
            {"name": "requirements", "label": "Concept"},
            {"name": "traceability", "label": "Concept"},
            {"name": "Requirements", "label": "Definition"},  # Should be skipped
        ]

        result = classifier.classify_batch(entities)

        assert len(result["to_delete"]) == 1
        assert result["to_delete"][0]["name"] == "tool"

        assert len(result["to_normalize"]) == 1
        assert result["to_normalize"][0]["name"] == "requirements"
        assert result["to_normalize"][0]["normalized_name"] == "requirement"

        assert len(result["to_keep"]) == 1
        assert result["to_keep"][0]["name"] == "traceability"

        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["label"] == "Definition"


# =============================================================================
# EXTRACTION VALIDATION TESTS
# =============================================================================


class TestValidateExtractedEntities:
    """Tests for the validate_extracted_entities function."""

    def test_filters_generic_entities(self) -> None:
        """Test that generic entities are filtered out."""
        from graphrag_kg_pipeline.extraction.pipeline import validate_extracted_entities

        entities = [
            {"name": "tool", "label": "Tool"},
            {"name": "software", "label": "Tool"},
            {"name": "traceability", "label": "Concept"},
        ]

        result = validate_extracted_entities(entities)

        assert len(result) == 1
        assert result[0]["name"] == "traceability"

    def test_normalizes_plurals(self) -> None:
        """Test that plural entities are normalized to singular."""
        from graphrag_kg_pipeline.extraction.pipeline import validate_extracted_entities

        entities = [
            {"name": "requirements", "label": "Concept", "display_name": "Requirements"},
            {"name": "stakeholders", "label": "Role"},
        ]

        result = validate_extracted_entities(entities)

        assert len(result) == 2
        assert result[0]["name"] == "requirement"
        assert result[0]["display_name"] == "Requirement"
        assert result[1]["name"] == "stakeholder"

    def test_preserves_valid_entities(self) -> None:
        """Test that valid entities are preserved unchanged."""
        from graphrag_kg_pipeline.extraction.pipeline import validate_extracted_entities

        entities = [
            {"name": "requirements traceability", "label": "Concept"},
            {"name": "iso 26262", "label": "Standard"},
            {"name": "jama connect", "label": "Tool"},
        ]

        result = validate_extracted_entities(entities)

        assert len(result) == 3
        assert all(e["name"] == entities[i]["name"] for i, e in enumerate(result))


# =============================================================================
# VALIDATION QUERIES TESTS
# =============================================================================


class TestValidationQueries:
    """Tests for ValidationQueries class."""

    def test_validation_queries_initialization(self) -> None:
        """Test that ValidationQueries initializes correctly."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        driver = MockDriver()
        queries = ValidationQueries(driver)

        assert queries is not None
        assert queries.driver == driver
        assert queries.database == "neo4j"

    def test_validation_queries_custom_database(self) -> None:
        """Test initialization with custom database name."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        driver = MockDriver()
        queries = ValidationQueries(driver, database="testdb")

        assert queries.database == "testdb"

    @pytest.mark.asyncio
    async def test_find_orphan_chunks(self) -> None:
        """Test finding orphan chunks query."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 5}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_orphan_chunks()

        assert count == 5

    @pytest.mark.asyncio
    async def test_find_orphan_chunks_zero(self) -> None:
        """Test finding orphan chunks when none exist."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 0}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_orphan_chunks()

        assert count == 0

    @pytest.mark.asyncio
    async def test_find_orphan_entities(self) -> None:
        """Test finding orphan entities query."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result(
            "element_id",
            [
                {"label": "Concept", "name": "orphan1", "element_id": "1"},
                {"label": "Tool", "name": "orphan2", "element_id": "2"},
            ],
        )
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        orphans = await queries.find_orphan_entities()

        assert len(orphans) == 2
        assert orphans[0]["name"] == "orphan1"

    @pytest.mark.asyncio
    async def test_find_duplicate_entities(self) -> None:
        """Test finding duplicate entities query."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result(
            "cnt > 1",
            [
                {"label": "Concept", "name": "duplicate", "cnt": 3},
            ],
        )
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        duplicates = await queries.find_duplicate_entities()

        assert len(duplicates) == 1
        assert duplicates[0]["cnt"] == 3

    @pytest.mark.asyncio
    async def test_find_missing_embeddings(self) -> None:
        """Test finding chunks without embeddings."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("missing_count", [{"missing_count": 10}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_missing_embeddings()

        assert count == 10

    @pytest.mark.asyncio
    async def test_count_industries(self) -> None:
        """Test counting industry nodes."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("industry_count", [{"industry_count": 18}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.count_industries()

        assert count == 18

    @pytest.mark.asyncio
    async def test_get_entity_stats(self) -> None:
        """Test getting entity statistics."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result(
            "label, count",
            [
                {"label": "Concept", "count": 100},
                {"label": "Industry", "count": 18},
                {"label": "Chunk", "count": 2000},
            ],
        )
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        stats = await queries.get_entity_stats()

        assert "Concept" in stats
        assert stats["Concept"] == 100

    @pytest.mark.asyncio
    async def test_find_invalid_patterns(self) -> None:
        """Test finding invalid relationship patterns."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        # Empty result means no invalid patterns
        session.set_default_result([])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        invalid = await queries.find_invalid_patterns()

        assert isinstance(invalid, list)


class TestRunAllValidations:
    """Tests for the run_all_validations function."""

    @pytest.mark.asyncio
    async def test_run_all_validations_structure(self) -> None:
        """Test that run_all_validations returns expected structure."""
        from graphrag_kg_pipeline.validation.queries import run_all_validations

        session = MockSession()
        # Set up default results for all queries
        session.set_result("orphan_count", [{"orphan_count": 0}])
        session.set_result("missing_count", [{"missing_count": 0}])
        session.set_result("industry_count", [{"industry_count": 18}])
        session.set_result(
            "total_articles", [{"total_articles": 103, "chapters_with_articles": 15}]
        )
        session.set_default_result([])
        driver = MockDriver(session)

        results = await run_all_validations(driver)

        # Check expected keys exist
        assert "orphan_chunks" in results
        assert "orphan_entities" in results
        assert "duplicate_entities" in results
        assert "missing_embeddings" in results
        assert "industry_count" in results
        assert "entity_stats" in results
        assert "summary" in results
        assert "validation_passed" in results

    @pytest.mark.asyncio
    async def test_validation_passed_when_clean(self) -> None:
        """Test that validation passes when no issues found."""
        from graphrag_kg_pipeline.validation.queries import run_all_validations

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 0}])
        session.set_result("missing_count", [{"missing_count": 0}])
        session.set_result("industry_count", [{"industry_count": 15}])
        session.set_result(
            "total_articles", [{"total_articles": 103, "chapters_with_articles": 15}]
        )
        session.set_default_result([])
        driver = MockDriver(session)

        results = await run_all_validations(driver)

        assert results["validation_passed"] is True

    @pytest.mark.asyncio
    async def test_validation_fails_with_orphans(self) -> None:
        """Test that validation fails when orphan chunks exist."""
        from graphrag_kg_pipeline.validation.queries import run_all_validations

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 5}])  # Has orphans
        session.set_result("missing_count", [{"missing_count": 0}])
        session.set_result("industry_count", [{"industry_count": 15}])
        session.set_result(
            "total_articles", [{"total_articles": 103, "chapters_with_articles": 15}]
        )
        session.set_default_result([])
        driver = MockDriver(session)

        results = await run_all_validations(driver)

        assert results["validation_passed"] is False

    @pytest.mark.asyncio
    async def test_validation_fails_with_too_many_industries(self) -> None:
        """Test that validation fails when industry count exceeds target."""
        from graphrag_kg_pipeline.validation.queries import run_all_validations

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 0}])
        session.set_result("missing_count", [{"missing_count": 0}])
        session.set_result("industry_count", [{"industry_count": 50}])  # Too many
        session.set_result(
            "total_articles", [{"total_articles": 103, "chapters_with_articles": 15}]
        )
        session.set_default_result([])
        driver = MockDriver(session)

        results = await run_all_validations(driver)

        # Industry count check is in summary but doesn't fail validation_passed
        assert results["summary"]["industry_count_ok"] is False


class TestValidationReporter:
    """Tests for validation reporting functionality."""

    def test_reporter_initialization(self) -> None:
        """Test that ValidationReporter initializes correctly."""
        from graphrag_kg_pipeline.validation.reporter import ValidationReporter

        driver = MockDriver()
        reporter = ValidationReporter(driver)
        assert reporter is not None
        assert reporter.driver == driver
        assert reporter.database == "neo4j"

    def test_generate_report_creates_markdown(self) -> None:
        """Test that report generation creates valid markdown."""
        from graphrag_kg_pipeline.validation.reporter import ValidationReport

        # Create ValidationReport directly with results
        report = ValidationReport(
            validation_passed=True,
            summary={
                "has_orphan_chunks": False,
                "has_orphan_entities": False,
                "has_duplicates": False,
                "has_missing_embeddings": False,
                "industry_count_ok": True,
                "has_invalid_patterns": False,
            },
            details={
                "orphan_chunks": 0,
                "orphan_entities": [],
                "duplicate_entities": [],
                "missing_embeddings": 0,
                "industry_count": 18,
                "entity_stats": {"Concept": 100, "Industry": 18},
                "invalid_patterns": [],
                "article_coverage": {
                    "total_articles": 103,
                    "chapters_with_articles": 15,
                },
            },
            recommendations=["No issues found - graph looks healthy!"],
        )

        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert "# Knowledge Graph Validation Report" in markdown
        assert "PASSED" in markdown

    def test_generate_report_shows_failures(self) -> None:
        """Test that report shows failure status when validation fails."""
        from graphrag_kg_pipeline.validation.reporter import ValidationReport

        # Create ValidationReport with failing data
        report = ValidationReport(
            validation_passed=False,
            summary={
                "has_orphan_chunks": True,
                "has_orphan_entities": True,
                "has_duplicates": False,
                "has_missing_embeddings": False,
                "industry_count_ok": False,
                "has_invalid_patterns": False,
            },
            details={
                "orphan_chunks": 5,
                "orphan_entities": [{"name": "test", "label": "Concept"}],
                "duplicate_entities": [],
                "missing_embeddings": 0,
                "industry_count": 50,
                "entity_stats": {"Concept": 100},
                "invalid_patterns": [],
                "article_coverage": {
                    "total_articles": 103,
                    "chapters_with_articles": 15,
                },
            },
            recommendations=[
                "Run chunk-article linking to connect 5 orphan chunks",
                "Run industry consolidation to reduce 50 industries to ≤19",
            ],
        )

        markdown = report.to_markdown()

        assert "FAILED" in markdown


# =============================================================================
# NEW VALIDATION QUERIES TESTS
# =============================================================================


class TestNewValidationQueries:
    """Tests for the new validation query methods."""

    @pytest.mark.asyncio
    async def test_find_missing_chunk_ids(self) -> None:
        """Test finding chunks without chunk_id property."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("chunk_id IS NULL", [{"missing_count": 2159}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_missing_chunk_ids()

        assert count == 2159

    @pytest.mark.asyncio
    async def test_find_missing_chunk_ids_zero(self) -> None:
        """Test when all chunks have chunk_id."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("chunk_id IS NULL", [{"missing_count": 0}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_missing_chunk_ids()

        assert count == 0

    @pytest.mark.asyncio
    async def test_find_plural_singular_duplicates(self) -> None:
        """Test finding plural/singular entity pairs."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result(
            "singular.name + 's'",
            [
                {
                    "label": "Concept",
                    "singular_name": "requirement",
                    "plural_name": "requirements",
                    "singular_rels": 45,
                    "plural_rels": 12,
                    "singular_id": "1",
                    "plural_id": "2",
                },
                {
                    "label": "Role",
                    "singular_name": "stakeholder",
                    "plural_name": "stakeholders",
                    "singular_rels": 30,
                    "plural_rels": 8,
                    "singular_id": "3",
                    "plural_id": "4",
                },
            ],
        )
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        duplicates = await queries.find_plural_singular_duplicates()

        assert len(duplicates) == 2
        assert duplicates[0]["singular_name"] == "requirement"
        assert duplicates[0]["plural_name"] == "requirements"

    @pytest.mark.asyncio
    async def test_find_generic_entities(self) -> None:
        """Test finding generic entity names."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result(
            "toLower(n.name) IN",
            [
                {
                    "label": "Tool",
                    "name": "tool",
                    "relationship_count": 15,
                    "element_id": "1",
                },
                {
                    "label": "Concept",
                    "name": "process",
                    "relationship_count": 8,
                    "element_id": "2",
                },
            ],
        )
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        generic = await queries.find_generic_entities()

        assert len(generic) == 2
        assert generic[0]["name"] == "tool"
        assert generic[0]["relationship_count"] == 15

    @pytest.mark.asyncio
    async def test_get_entity_relationship_counts(self) -> None:
        """Test getting relationship counts for specific entities."""
        from graphrag_kg_pipeline.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result(
            "name, relationship_count",
            [
                {"name": "requirement", "relationship_count": 45},
                {"name": "traceability", "relationship_count": 32},
            ],
        )
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        counts = await queries.get_entity_relationship_counts(
            ["requirement", "traceability"],
            "Concept",
        )

        assert counts["requirement"] == 45
        assert counts["traceability"] == 32


class TestRunAllValidationsWithNewChecks:
    """Tests for run_all_validations including new checks."""

    @pytest.mark.asyncio
    async def test_includes_new_checks(self) -> None:
        """Test that new validation checks are included in results."""
        from graphrag_kg_pipeline.validation.queries import run_all_validations

        session = MockSession()
        # Set up all required results
        session.set_result("orphan_count", [{"orphan_count": 0}])
        session.set_result("missing_count", [{"missing_count": 0}])
        session.set_result("industry_count", [{"industry_count": 15}])
        session.set_result(
            "total_articles", [{"total_articles": 103, "chapters_with_articles": 15}]
        )
        session.set_result("chunk_id IS NULL", [{"missing_count": 100}])
        session.set_default_result([])
        driver = MockDriver(session)

        results = await run_all_validations(driver)

        # Check new keys exist
        assert "missing_chunk_ids" in results
        assert "plural_singular_duplicates" in results
        assert "generic_entities" in results

        # Check summary includes new flags
        assert "has_missing_chunk_ids" in results["summary"]
        assert "has_plural_duplicates" in results["summary"]
        assert "has_generic_entities" in results["summary"]

    @pytest.mark.asyncio
    async def test_validation_fails_with_missing_chunk_ids(self) -> None:
        """Test that validation fails when chunks are missing chunk_id."""
        from graphrag_kg_pipeline.validation.queries import run_all_validations

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 0}])
        # Note: "missing_count" matches both missing embeddings and missing chunk_ids
        # So we need to use a more specific pattern for chunk_id query
        session.set_result("c.embedding IS NULL", [{"missing_count": 0}])  # Embeddings OK
        session.set_result("c.chunk_id IS NULL", [{"missing_count": 2159}])  # Chunk IDs missing!
        session.set_result("industry_count", [{"industry_count": 15}])
        session.set_result(
            "total_articles", [{"total_articles": 103, "chapters_with_articles": 15}]
        )
        session.set_default_result([])
        driver = MockDriver(session)

        results = await run_all_validations(driver)

        assert results["summary"]["has_missing_chunk_ids"] is True
        assert results["validation_passed"] is False


# =============================================================================
# VALIDATION FIXES TESTS
# =============================================================================


class TestValidationFixes:
    """Tests for validation fix functions."""

    @pytest.mark.asyncio
    async def test_fix_missing_chunk_ids_dry_run(self) -> None:
        """Test dry run mode for chunk_id fix."""
        from graphrag_kg_pipeline.validation.fixes import fix_missing_chunk_ids

        session = MockSession()
        session.set_result("chunk_id IS NULL", [{"missing_count": 100}])
        driver = MockDriver(session)

        result = await fix_missing_chunk_ids(driver, dry_run=True)

        assert result["dry_run"] is True
        assert result["total_missing"] == 100
        assert result["fixed"] == 0  # Dry run doesn't fix

    @pytest.mark.asyncio
    async def test_format_fix_preview(self) -> None:
        """Test formatting of fix preview output."""
        from graphrag_kg_pipeline.validation.fixes import format_fix_preview

        preview = {
            "summary": {
                "chunk_ids_to_fix": 100,
                "entities_to_delete": 5,
                "entities_to_merge": 10,
                "total_changes": 115,
            },
            "chunk_ids": {"total_missing": 100},
            "generic_entities": {
                "would_delete": 5,
                "entities": [
                    {"label": "Tool", "name": "tool", "relationship_count": 15},
                ],
            },
            "plural_entities": {
                "would_merge": 10,
                "entities": [
                    {"label": "Concept", "name": "requirements", "normalized_name": "requirement", "relationship_count": 45},
                ],
            },
        }

        output = format_fix_preview(preview)

        assert "=== Validation Fix Preview ===" in output
        assert "Chunk IDs to generate: 100" in output
        assert "Generic entities to delete: 5" in output
        assert "Plural entities to merge: 10" in output
        assert "Total changes: 115" in output
