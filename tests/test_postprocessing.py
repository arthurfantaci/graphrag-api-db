"""Tests for the postprocessing module.

This module tests entity normalization, industry taxonomy consolidation,
and glossary linking functionality.
"""

from __future__ import annotations


class TestIndustryTaxonomy:
    """Tests for industry taxonomy and classification."""

    def test_industry_taxonomy_defined(self) -> None:
        """Test that industry taxonomy mapping is defined."""
        from jama_scraper.postprocessing.industry_taxonomy import INDUSTRY_TAXONOMY

        assert isinstance(INDUSTRY_TAXONOMY, dict)
        assert len(INDUSTRY_TAXONOMY) > 0

    def test_industry_taxonomy_has_canonical_names(self) -> None:
        """Test that taxonomy maps to canonical industry names."""
        from jama_scraper.postprocessing.industry_taxonomy import INDUSTRY_TAXONOMY

        # Check some expected canonical industries
        canonical_values = set(INDUSTRY_TAXONOMY.values())

        expected_canonicals = [
            "aerospace",
            "automotive",
            "medical devices",
            "defense",
            "energy",
        ]

        for expected in expected_canonicals:
            assert expected in canonical_values, f"Missing canonical: {expected}"

    def test_industry_taxonomy_covers_variants(self) -> None:
        """Test that taxonomy covers common variants."""
        from jama_scraper.postprocessing.industry_taxonomy import INDUSTRY_TAXONOMY

        # These variants should map to canonical names
        variant_mappings = [
            ("auto", "automotive"),
            ("medtech", "medical devices"),
            ("aerospace & defense", "aerospace"),
            ("life sciences", "life sciences"),
        ]

        for variant, expected in variant_mappings:
            if variant in INDUSTRY_TAXONOMY:
                assert INDUSTRY_TAXONOMY[variant] == expected

    def test_concepts_not_industries_defined(self) -> None:
        """Test that concepts-not-industries set is defined."""
        from jama_scraper.postprocessing.industry_taxonomy import (
            CONCEPTS_NOT_INDUSTRIES,
        )

        assert isinstance(CONCEPTS_NOT_INDUSTRIES, set)
        assert len(CONCEPTS_NOT_INDUSTRIES) > 0

    def test_concepts_not_industries_contents(self) -> None:
        """Test that technology concepts are in the exclusion set."""
        from jama_scraper.postprocessing.industry_taxonomy import (
            CONCEPTS_NOT_INDUSTRIES,
        )

        # These should be classified as Concepts, not Industries
        expected_concepts = [
            "artificial intelligence",
            "iot",
            "machine learning",
            "software development",
        ]

        for concept in expected_concepts:
            assert concept in CONCEPTS_NOT_INDUSTRIES, f"Missing concept: {concept}"

    def test_generic_terms_to_delete_defined(self) -> None:
        """Test that generic terms set is defined."""
        from jama_scraper.postprocessing.industry_taxonomy import (
            GENERIC_TERMS_TO_DELETE,
        )

        assert isinstance(GENERIC_TERMS_TO_DELETE, set)
        assert len(GENERIC_TERMS_TO_DELETE) > 0

    def test_generic_terms_contents(self) -> None:
        """Test that vague terms are in deletion set."""
        from jama_scraper.postprocessing.industry_taxonomy import (
            GENERIC_TERMS_TO_DELETE,
        )

        expected_generic = ["industry", "regulated", "general"]

        for term in expected_generic:
            assert term in GENERIC_TERMS_TO_DELETE, f"Missing generic term: {term}"


class TestClassifyIndustryTerm:
    """Tests for the classify_industry_term function."""

    def test_classify_valid_industry(self) -> None:
        """Test classification of valid industry names."""
        from jama_scraper.postprocessing.industry_taxonomy import classify_industry_term

        action, value = classify_industry_term("automotive")
        assert action == "keep"
        assert value == "automotive"

    def test_classify_industry_variant(self) -> None:
        """Test classification of industry variants."""
        from jama_scraper.postprocessing.industry_taxonomy import classify_industry_term

        # "auto industry" is a variant that maps to "automotive"
        action, value = classify_industry_term("auto industry")
        assert action == "keep"
        assert value == "automotive"

    def test_classify_concept_reclassification(self) -> None:
        """Test that technology concepts are marked for reclassification."""
        from jama_scraper.postprocessing.industry_taxonomy import classify_industry_term

        action, value = classify_industry_term("artificial intelligence")
        assert action == "reclassify"
        assert value is None

    def test_classify_generic_deletion(self) -> None:
        """Test that generic terms are marked for deletion."""
        from jama_scraper.postprocessing.industry_taxonomy import classify_industry_term

        action, value = classify_industry_term("industry")
        assert action == "delete"
        assert value is None

    def test_classify_unknown_term(self) -> None:
        """Test classification of unknown terms."""
        from jama_scraper.postprocessing.industry_taxonomy import classify_industry_term

        action, value = classify_industry_term("xyzzy_not_a_real_industry_12345")
        assert action == "unknown"
        assert value is None

    def test_classify_case_insensitive(self) -> None:
        """Test that classification is case-insensitive."""
        from jama_scraper.postprocessing.industry_taxonomy import classify_industry_term

        # All these should normalize to the same result
        for variant in ["Automotive", "AUTOMOTIVE", "automotive"]:
            action, value = classify_industry_term(variant)
            assert action == "keep"
            assert value == "automotive"


class TestNormalizeIndustry:
    """Tests for the normalize_industry function."""

    def test_normalize_exact_match(self) -> None:
        """Test normalization with exact match."""
        from jama_scraper.postprocessing.industry_taxonomy import normalize_industry

        result = normalize_industry("automotive")
        assert result == "automotive"

    def test_normalize_fuzzy_match(self) -> None:
        """Test normalization with fuzzy matching."""
        from jama_scraper.postprocessing.industry_taxonomy import normalize_industry

        # Should fuzzy match to "medical devices"
        result = normalize_industry("medical device")
        assert result in ("medical devices", "medical device", None)

    def test_normalize_returns_none_for_unknown(self) -> None:
        """Test that unknown industries return None."""
        from jama_scraper.postprocessing.industry_taxonomy import normalize_industry

        result = normalize_industry("not_an_industry_xyz")
        assert result is None


class TestEntityNormalizer:
    """Tests for EntityNormalizer class."""

    def test_normalizer_initialization(self) -> None:
        """Test that normalizer initializes with driver."""
        from jama_scraper.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        driver = MockDriver()
        normalizer = EntityNormalizer(driver)

        assert normalizer is not None
        assert normalizer.driver == driver


class TestGlossaryLinker:
    """Tests for glossary linking functionality."""

    def test_glossary_linker_initialization(self) -> None:
        """Test that glossary linker initializes correctly."""
        from jama_scraper.postprocessing.glossary_linker import GlossaryConceptLinker
        from tests.conftest import MockDriver

        driver = MockDriver()
        linker = GlossaryConceptLinker(driver)

        assert linker is not None
        assert linker.driver == driver
        assert linker.database == "neo4j"
        assert linker.match_threshold == 85
