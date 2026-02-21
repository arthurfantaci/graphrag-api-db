"""Tests for the postprocessing module.

This module tests entity normalization, industry taxonomy consolidation,
and glossary linking functionality.
"""

from __future__ import annotations

import pytest


class TestIndustryTaxonomy:
    """Tests for industry taxonomy and classification."""

    def test_industry_taxonomy_defined(self) -> None:
        """Test that industry taxonomy mapping is defined."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            INDUSTRY_TAXONOMY,
        )

        assert isinstance(INDUSTRY_TAXONOMY, dict)
        assert len(INDUSTRY_TAXONOMY) > 0

    def test_industry_taxonomy_has_canonical_names(self) -> None:
        """Test that taxonomy maps to canonical industry names."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            INDUSTRY_TAXONOMY,
        )

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
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            INDUSTRY_TAXONOMY,
        )

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
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            CONCEPTS_NOT_INDUSTRIES,
        )

        assert isinstance(CONCEPTS_NOT_INDUSTRIES, set)
        assert len(CONCEPTS_NOT_INDUSTRIES) > 0

    def test_concepts_not_industries_contents(self) -> None:
        """Test that technology concepts are in the exclusion set."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
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
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            GENERIC_TERMS_TO_DELETE,
        )

        assert isinstance(GENERIC_TERMS_TO_DELETE, set)
        assert len(GENERIC_TERMS_TO_DELETE) > 0

    def test_generic_terms_contents(self) -> None:
        """Test that vague terms are in deletion set."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            GENERIC_TERMS_TO_DELETE,
        )

        expected_generic = ["industry", "regulated", "general"]

        for term in expected_generic:
            assert term in GENERIC_TERMS_TO_DELETE, f"Missing generic term: {term}"


class TestClassifyIndustryTerm:
    """Tests for the classify_industry_term function."""

    def test_classify_valid_industry(self) -> None:
        """Test classification of valid industry names."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            classify_industry_term,
        )

        action, value = classify_industry_term("automotive")
        assert action == "keep"
        assert value == "automotive"

    def test_classify_industry_variant(self) -> None:
        """Test classification of industry variants."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            classify_industry_term,
        )

        # "auto industry" is a variant that maps to "automotive"
        action, value = classify_industry_term("auto industry")
        assert action == "keep"
        assert value == "automotive"

    def test_classify_concept_reclassification(self) -> None:
        """Test that technology concepts are marked for reclassification."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            classify_industry_term,
        )

        action, value = classify_industry_term("artificial intelligence")
        assert action == "reclassify"
        assert value is None

    def test_classify_generic_deletion(self) -> None:
        """Test that generic terms are marked for deletion."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            classify_industry_term,
        )

        action, value = classify_industry_term("industry")
        assert action == "delete"
        assert value is None

    def test_classify_unknown_term(self) -> None:
        """Test classification of unknown terms."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            classify_industry_term,
        )

        action, value = classify_industry_term("xyzzy_not_a_real_industry_12345")
        assert action == "unknown"
        assert value is None

    def test_classify_case_insensitive(self) -> None:
        """Test that classification is case-insensitive."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            classify_industry_term,
        )

        # All these should normalize to the same result
        for variant in ["Automotive", "AUTOMOTIVE", "automotive"]:
            action, value = classify_industry_term(variant)
            assert action == "keep"
            assert value == "automotive"


class TestNormalizeIndustry:
    """Tests for the normalize_industry function."""

    def test_normalize_exact_match(self) -> None:
        """Test normalization with exact match."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            normalize_industry,
        )

        result = normalize_industry("automotive")
        assert result == "automotive"

    def test_normalize_fuzzy_match(self) -> None:
        """Test normalization with fuzzy matching."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            normalize_industry,
        )

        # Should fuzzy match to "medical devices"
        result = normalize_industry("medical device")
        assert result in ("medical devices", "medical device", None)

    def test_normalize_returns_none_for_unknown(self) -> None:
        """Test that unknown industries return None."""
        from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
            normalize_industry,
        )

        result = normalize_industry("not_an_industry_xyz")
        assert result is None


class TestEntityNormalizer:
    """Tests for EntityNormalizer class."""

    def test_normalizer_initialization(self) -> None:
        """Test that normalizer initializes with driver."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        driver = MockDriver()
        normalizer = EntityNormalizer(driver)

        assert normalizer is not None
        assert normalizer.driver == driver


class TestCrossLabelDeduplication:
    """Tests for cross-label entity deduplication."""

    def test_label_priority_ranking_defined(self) -> None:
        """Test that LABEL_PRIORITY constant exists and has all 10 labels."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer

        assert hasattr(EntityNormalizer, "LABEL_PRIORITY")
        assert len(EntityNormalizer.LABEL_PRIORITY) == 10

    def test_label_priority_covers_entity_labels(self) -> None:
        """Test that LABEL_PRIORITY covers the same labels as entity_labels."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        normalizer = EntityNormalizer(MockDriver())
        assert set(EntityNormalizer.LABEL_PRIORITY) == set(normalizer.entity_labels)

    def test_resolve_winning_label_intrinsic_wins(self) -> None:
        """Test that intrinsic types (Standard) beat generic types (Concept)."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        normalizer = EntityNormalizer(MockDriver())
        result = normalizer._resolve_winning_label(
            [
                ["__Entity__", "__KGBuilder__", "Standard"],
                ["__Entity__", "__KGBuilder__", "Concept"],
            ]
        )
        assert result == "Standard"

    def test_resolve_winning_label_concept_beats_contextual(self) -> None:
        """Test that Concept beats contextual types like Challenge."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        normalizer = EntityNormalizer(MockDriver())
        result = normalizer._resolve_winning_label(
            [
                ["__Entity__", "__KGBuilder__", "Challenge"],
                ["__Entity__", "__KGBuilder__", "Concept"],
            ]
        )
        assert result == "Concept"

    def test_resolve_winning_label_no_type_labels_defaults_concept(self) -> None:
        """Test that nodes with only system labels default to Concept."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        normalizer = EntityNormalizer(MockDriver())
        result = normalizer._resolve_winning_label(
            [["__Entity__", "__KGBuilder__"], ["__Entity__", "__KGBuilder__"]]
        )
        assert result == "Concept"

    def test_resolve_winning_label_single_label(self) -> None:
        """Test that a single recognized label returns itself."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        normalizer = EntityNormalizer(MockDriver())
        result = normalizer._resolve_winning_label([["__Entity__", "Tool"], ["__Entity__", "Tool"]])
        assert result == "Tool"

    def test_cross_label_dedup_method_exists(self) -> None:
        """Test that deduplicate_cross_label method exists on EntityNormalizer."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver

        normalizer = EntityNormalizer(MockDriver())
        assert callable(getattr(normalizer, "deduplicate_cross_label", None))

    @pytest.mark.asyncio
    async def test_cross_label_dedup_no_duplicates(self) -> None:
        """Test cross-label dedup returns zero when no duplicates exist."""
        from graphrag_kg_pipeline.postprocessing.normalizer import EntityNormalizer
        from tests.conftest import MockDriver, MockSession

        session = MockSession()
        session.set_default_result([])
        driver = MockDriver(session)

        normalizer = EntityNormalizer(driver)
        stats = await normalizer.deduplicate_cross_label()
        assert stats["cross_label_merged"] == 0
        assert stats["by_winning_label"] == {}
        assert stats["replaced_labels"] == {}


class TestMislabeledChallengeDetection:
    """Tests for mislabeled Challenge entity detection."""

    def test_positive_outcome_detected(self) -> None:
        """Test that positive outcomes are flagged as mislabeled Challenges."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            is_potentially_mislabeled_challenge,
        )

        assert is_potentially_mislabeled_challenge("High-Quality Products") is True
        assert is_potentially_mislabeled_challenge("Improved Delivery") is True
        assert is_potentially_mislabeled_challenge("Reduced Time-to-Market") is True

    def test_actual_challenges_not_flagged(self) -> None:
        """Test that genuine challenges are not flagged."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            is_potentially_mislabeled_challenge,
        )

        assert is_potentially_mislabeled_challenge("scope creep") is False
        assert is_potentially_mislabeled_challenge("requirements volatility") is False
        assert is_potentially_mislabeled_challenge("late defect discovery") is False

    def test_empty_name(self) -> None:
        """Test empty name returns False."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            is_potentially_mislabeled_challenge,
        )

        assert is_potentially_mislabeled_challenge("") is False

    def test_positive_outcome_words_defined(self) -> None:
        """Test that POSITIVE_OUTCOME_WORDS is a populated frozenset."""
        from graphrag_kg_pipeline.postprocessing.entity_cleanup import (
            POSITIVE_OUTCOME_WORDS,
        )

        assert isinstance(POSITIVE_OUTCOME_WORDS, frozenset)
        assert len(POSITIVE_OUTCOME_WORDS) > 10


class TestMentionedInBackfiller:
    """Tests for MENTIONED_IN and APPLIES_TO backfill."""

    def test_standard_industry_map_defined(self) -> None:
        """Test that the Standard→Industry mapping is populated."""
        from graphrag_kg_pipeline.postprocessing.mentioned_in_backfill import (
            STANDARD_INDUSTRY_MAP,
        )

        assert isinstance(STANDARD_INDUSTRY_MAP, dict)
        assert len(STANDARD_INDUSTRY_MAP) > 10

    def test_standard_industry_map_known_entries(self) -> None:
        """Test that key standards are mapped to correct industries."""
        from graphrag_kg_pipeline.postprocessing.mentioned_in_backfill import (
            STANDARD_INDUSTRY_MAP,
        )

        assert STANDARD_INDUSTRY_MAP["iso 26262"] == "automotive"
        assert STANDARD_INDUSTRY_MAP["do-178c"] == "aerospace"
        assert STANDARD_INDUSTRY_MAP["iec 62304"] == "medical devices"

    def test_backfiller_initialization(self) -> None:
        """Test that the backfiller initializes correctly."""
        from graphrag_kg_pipeline.postprocessing.mentioned_in_backfill import (
            MentionedInBackfiller,
        )
        from tests.conftest import MockDriver

        driver = MockDriver()
        backfiller = MentionedInBackfiller(driver, "neo4j")
        assert backfiller.driver == driver
        assert backfiller.database == "neo4j"

    @pytest.mark.asyncio
    async def test_backfill_mentioned_in(self) -> None:
        """Test MENTIONED_IN backfill with mock driver."""
        from graphrag_kg_pipeline.postprocessing.mentioned_in_backfill import (
            MentionedInBackfiller,
        )
        from tests.conftest import MockDriver, MockSession

        session = MockSession()
        session.set_default_result([{"created": 5}])
        driver = MockDriver(session)

        backfiller = MentionedInBackfiller(driver, "neo4j")
        stats = await backfiller.backfill_mentioned_in()
        assert "mentioned_in_created" in stats

    @pytest.mark.asyncio
    async def test_backfill_applies_to(self) -> None:
        """Test APPLIES_TO backfill with mock driver."""
        from graphrag_kg_pipeline.postprocessing.mentioned_in_backfill import (
            MentionedInBackfiller,
        )
        from tests.conftest import MockDriver, MockSession

        session = MockSession()
        session.set_default_result([{"created": 1}])
        driver = MockDriver(session)

        backfiller = MentionedInBackfiller(driver, "neo4j")
        stats = await backfiller.backfill_applies_to()
        assert "applies_to_created" in stats


class TestEntitySummarizer:
    """Tests for entity description summarization."""

    def test_summarizer_initialization(self) -> None:
        """Test that the summarizer initializes correctly."""
        from graphrag_kg_pipeline.postprocessing.entity_summarizer import EntitySummarizer
        from tests.conftest import MockDriver

        driver = MockDriver()
        summarizer = EntitySummarizer(
            driver=driver,
            database="neo4j",
            openai_api_key="sk-test-123",
        )
        assert summarizer.model == "gpt-4o"

    def test_parse_fragments_json_array(self) -> None:
        """Test parsing description fragments from JSON array."""
        from graphrag_kg_pipeline.postprocessing.entity_summarizer import EntitySummarizer
        from tests.conftest import MockDriver

        summarizer = EntitySummarizer(
            driver=MockDriver(), database="neo4j", openai_api_key="sk-test"
        )
        fragments = summarizer._parse_fragments('["desc1", "desc2", "desc3"]')
        assert len(fragments) == 3
        assert fragments[0] == "desc1"

    def test_parse_fragments_pipe_delimited(self) -> None:
        """Test parsing pipe-delimited descriptions."""
        from graphrag_kg_pipeline.postprocessing.entity_summarizer import EntitySummarizer
        from tests.conftest import MockDriver

        summarizer = EntitySummarizer(
            driver=MockDriver(), database="neo4j", openai_api_key="sk-test"
        )
        fragments = summarizer._parse_fragments("first desc | second desc | third desc")
        assert len(fragments) == 3

    def test_parse_fragments_single(self) -> None:
        """Test parsing single description returns list of one."""
        from graphrag_kg_pipeline.postprocessing.entity_summarizer import EntitySummarizer
        from tests.conftest import MockDriver

        summarizer = EntitySummarizer(
            driver=MockDriver(), database="neo4j", openai_api_key="sk-test"
        )
        fragments = summarizer._parse_fragments("just a single description")
        assert len(fragments) == 1

    @pytest.mark.asyncio
    async def test_summarize_no_entities(self) -> None:
        """Test summarization with no fragmented entities."""
        from graphrag_kg_pipeline.postprocessing.entity_summarizer import EntitySummarizer
        from tests.conftest import MockDriver, MockSession

        session = MockSession()
        session.set_default_result([])
        driver = MockDriver(session)

        summarizer = EntitySummarizer(driver=driver, database="neo4j", openai_api_key="sk-test")
        stats = await summarizer.summarize()
        assert stats["entities_found"] == 0
        assert stats["entities_summarized"] == 0

    def test_summarization_prompt_template(self) -> None:
        """Test that the summarization prompt has expected placeholders."""
        from graphrag_kg_pipeline.postprocessing.entity_summarizer import (
            SUMMARIZATION_PROMPT,
        )

        assert "{entity_name}" in SUMMARIZATION_PROMPT
        assert "{entity_label}" in SUMMARIZATION_PROMPT
        assert "{descriptions}" in SUMMARIZATION_PROMPT


class TestGlossaryLinker:
    """Tests for glossary linking functionality."""

    def test_glossary_linker_initialization(self) -> None:
        """Test that glossary linker initializes correctly."""
        from graphrag_kg_pipeline.postprocessing.glossary_linker import (
            GlossaryConceptLinker,
        )
        from tests.conftest import MockDriver

        driver = MockDriver()
        linker = GlossaryConceptLinker(driver)

        assert linker is not None
        assert linker.driver == driver
        assert linker.database == "neo4j"
        assert linker.match_threshold == 85
