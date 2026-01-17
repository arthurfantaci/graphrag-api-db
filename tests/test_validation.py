"""Tests for the validation module.

This module tests the knowledge graph validation queries and reporting
functionality that ensures data quality after extraction.
"""

from __future__ import annotations

import pytest

from tests.conftest import MockDriver, MockSession


class TestValidationQueries:
    """Tests for ValidationQueries class."""

    def test_validation_queries_initialization(self) -> None:
        """Test that ValidationQueries initializes correctly."""
        from jama_scraper.validation.queries import ValidationQueries

        driver = MockDriver()
        queries = ValidationQueries(driver)

        assert queries is not None
        assert queries.driver == driver
        assert queries.database == "neo4j"

    def test_validation_queries_custom_database(self) -> None:
        """Test initialization with custom database name."""
        from jama_scraper.validation.queries import ValidationQueries

        driver = MockDriver()
        queries = ValidationQueries(driver, database="testdb")

        assert queries.database == "testdb"

    @pytest.mark.asyncio
    async def test_find_orphan_chunks(self) -> None:
        """Test finding orphan chunks query."""
        from jama_scraper.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 5}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_orphan_chunks()

        assert count == 5

    @pytest.mark.asyncio
    async def test_find_orphan_chunks_zero(self) -> None:
        """Test finding orphan chunks when none exist."""
        from jama_scraper.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("orphan_count", [{"orphan_count": 0}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_orphan_chunks()

        assert count == 0

    @pytest.mark.asyncio
    async def test_find_orphan_entities(self) -> None:
        """Test finding orphan entities query."""
        from jama_scraper.validation.queries import ValidationQueries

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
        from jama_scraper.validation.queries import ValidationQueries

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
        from jama_scraper.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("missing_count", [{"missing_count": 10}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.find_missing_embeddings()

        assert count == 10

    @pytest.mark.asyncio
    async def test_count_industries(self) -> None:
        """Test counting industry nodes."""
        from jama_scraper.validation.queries import ValidationQueries

        session = MockSession()
        session.set_result("industry_count", [{"industry_count": 18}])
        driver = MockDriver(session)

        queries = ValidationQueries(driver)
        count = await queries.count_industries()

        assert count == 18

    @pytest.mark.asyncio
    async def test_get_entity_stats(self) -> None:
        """Test getting entity statistics."""
        from jama_scraper.validation.queries import ValidationQueries

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
        from jama_scraper.validation.queries import ValidationQueries

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
        from jama_scraper.validation.queries import run_all_validations

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
        from jama_scraper.validation.queries import run_all_validations

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
        from jama_scraper.validation.queries import run_all_validations

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
        from jama_scraper.validation.queries import run_all_validations

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
        from jama_scraper.validation.reporter import ValidationReporter

        driver = MockDriver()
        reporter = ValidationReporter(driver)
        assert reporter is not None
        assert reporter.driver == driver
        assert reporter.database == "neo4j"

    def test_generate_report_creates_markdown(self) -> None:
        """Test that report generation creates valid markdown."""
        from jama_scraper.validation.reporter import ValidationReport

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
        from jama_scraper.validation.reporter import ValidationReport

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
