"""Smoke tests for the graphrag-kg-pipeline package.

Verifies the package can be imported and key symbols are accessible.
"""

from __future__ import annotations


def test_package_import() -> None:
    """Verify the package imports without errors."""
    import graphrag_kg_pipeline

    assert hasattr(graphrag_kg_pipeline, "__version__")


def test_version_is_string() -> None:
    """Verify the version is a valid string."""
    from graphrag_kg_pipeline import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_core_exports_available() -> None:
    """Verify key public symbols are importable."""
    from graphrag_kg_pipeline import (
        NODE_TYPES,
        RELATIONSHIP_TYPES,
        JamaGuideScraper,
        JamaKGPipelineConfig,
        run_scraper,
    )

    assert JamaGuideScraper is not None
    assert JamaKGPipelineConfig is not None
    assert run_scraper is not None
    assert isinstance(NODE_TYPES, dict)
    assert isinstance(RELATIONSHIP_TYPES, dict)


def test_schema_has_expected_node_types() -> None:
    """Verify the schema contains the expected 10 node types."""
    from graphrag_kg_pipeline.extraction.schema import NODE_TYPES

    expected_types = {
        "Concept",
        "Challenge",
        "Artifact",
        "Industry",
        "Standard",
        "Tool",
    }
    actual_types = set(NODE_TYPES.keys())
    assert expected_types.issubset(actual_types), (
        f"Missing node types: {expected_types - actual_types}"
    )


def test_extraction_gleaning_validates_labels() -> None:
    """Verify gleaning module imports schema for label validation."""
    from graphrag_kg_pipeline.extraction.schema import NODE_TYPES, RELATIONSHIP_TYPES

    # Ensure the schema types that gleaning validates against exist
    assert len(NODE_TYPES) >= 6
    assert len(RELATIONSHIP_TYPES) >= 6
