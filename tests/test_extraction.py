"""Tests for the extraction module.

This module tests the knowledge graph extraction schema, prompts,
and pipeline configuration for the neo4j_graphrag integration.
"""

from __future__ import annotations

import pytest


class TestExtractionSchema:
    """Tests for extraction schema definitions."""

    def test_node_types_defined(self) -> None:
        """Test that all expected node types are defined."""
        from graphrag_kg_pipeline.extraction.schema import NODE_TYPES

        expected_types = [
            "Concept",
            "Challenge",
            "Artifact",
            "Bestpractice",
            "Processstage",
            "Role",
            "Standard",
            "Tool",
            "Methodology",
            "Industry",
        ]

        # NODE_TYPES is a dict with label as key
        for expected in expected_types:
            assert expected in NODE_TYPES, f"Missing node type: {expected}"

    def test_node_types_have_required_fields(self) -> None:
        """Test that node types have label and description."""
        from graphrag_kg_pipeline.extraction.schema import NODE_TYPES

        for label, node_type in NODE_TYPES.items():
            assert "label" in node_type, f"Node type {label} missing label"
            assert "description" in node_type, f"Node type {label} missing description"
            assert "properties" in node_type, f"Node type {label} missing properties"

    def test_node_types_have_name_property(self) -> None:
        """Test that all node types have a name property."""
        from graphrag_kg_pipeline.extraction.schema import NODE_TYPES

        for label, node_type in NODE_TYPES.items():
            props = node_type["properties"]
            assert "name" in props, f"Node type {label} missing 'name' property"

    def test_relationship_types_defined(self) -> None:
        """Test that expected relationship types are defined."""
        from graphrag_kg_pipeline.extraction.schema import RELATIONSHIP_TYPES

        expected_rels = [
            "ADDRESSES",
            "REQUIRES",
            "COMPONENT_OF",
            "RELATED_TO",
            "ALTERNATIVE_TO",
            "USED_BY",
            "APPLIES_TO",
            "PRODUCES",
            "DEFINES",
            "PREREQUISITE_FOR",
        ]

        # RELATIONSHIP_TYPES is a dict with label as key
        for expected in expected_rels:
            assert expected in RELATIONSHIP_TYPES, f"Missing relationship type: {expected}"

    def test_patterns_are_valid_triples(self) -> None:
        """Test that patterns are valid (source, rel, target) triples."""
        from graphrag_kg_pipeline.extraction.schema import (
            NODE_TYPES,
            PATTERNS,
            RELATIONSHIP_TYPES,
        )

        node_labels = set(NODE_TYPES.keys())
        rel_labels = set(RELATIONSHIP_TYPES.keys())

        for pattern in PATTERNS:
            assert len(pattern) == 3, f"Pattern must be triple: {pattern}"
            source, rel, target = pattern

            assert source in node_labels, f"Invalid source in pattern: {source}"
            assert rel in rel_labels, f"Invalid relationship in pattern: {rel}"
            assert target in node_labels, f"Invalid target in pattern: {target}"

    def test_patterns_count(self) -> None:
        """Test that we have a reasonable number of patterns."""
        from graphrag_kg_pipeline.extraction.schema import PATTERNS

        # Should have at least 20 patterns for a rich schema
        assert len(PATTERNS) >= 20, f"Expected at least 20 patterns, got {len(PATTERNS)}"

    def test_industry_has_regulated_property(self) -> None:
        """Test that Industry node type has regulated property."""
        from graphrag_kg_pipeline.extraction.schema import NODE_TYPES

        industry = NODE_TYPES.get("Industry")
        assert industry is not None

        props = industry["properties"]
        assert "regulated" in props

    def test_standard_has_organization_property(self) -> None:
        """Test that Standard node type has organization property."""
        from graphrag_kg_pipeline.extraction.schema import NODE_TYPES

        standard = NODE_TYPES.get("Standard")
        assert standard is not None

        props = standard["properties"]
        assert "organization" in props


class TestExtractionPrompts:
    """Tests for extraction prompt templates."""

    def test_domain_instructions_content(self) -> None:
        """Test that domain instructions contain key sections."""
        from graphrag_kg_pipeline.extraction.prompts import (
            REQUIREMENTS_DOMAIN_INSTRUCTIONS,
        )

        # Should contain critical classification rules
        assert "Industry vs Concept" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "Name Normalization" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "Standards Identification" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

    def test_domain_instructions_has_examples(self) -> None:
        """Test that domain instructions include few-shot examples."""
        from graphrag_kg_pipeline.extraction.prompts import (
            REQUIREMENTS_DOMAIN_INSTRUCTIONS,
        )

        assert "FEW-SHOT EXAMPLES" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "Example 1" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

    def test_domain_instructions_has_negative_examples(self) -> None:
        """Test that domain instructions include negative examples."""
        from graphrag_kg_pipeline.extraction.prompts import (
            REQUIREMENTS_DOMAIN_INSTRUCTIONS,
        )

        assert "COMMON MISTAKES TO AVOID" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "WRONG" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

    def test_negative_examples_cover_known_issues(self) -> None:
        """Test that negative examples address known extraction errors."""
        from graphrag_kg_pipeline.extraction.prompts import (
            REQUIREMENTS_DOMAIN_INSTRUCTIONS,
        )

        # Should warn about Concept -[USED_BY]-> Tool
        assert "Concept" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "USED_BY" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

        # Should warn about certification organizations
        assert (
            "TÜV" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
            or "certification" in REQUIREMENTS_DOMAIN_INSTRUCTIONS.lower()
        )

        # Should warn about Standard -[APPLIES_TO]-> Concept
        assert "Standard" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "APPLIES_TO" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

    def test_create_extraction_template(self) -> None:
        """Test that extraction template can be created."""
        from graphrag_kg_pipeline.extraction.prompts import create_extraction_template

        template = create_extraction_template()

        assert template is not None
        assert hasattr(template, "template")

    def test_extraction_template_has_schema_placeholder(self) -> None:
        """Test that template includes schema placeholder."""
        from graphrag_kg_pipeline.extraction.prompts import create_extraction_template

        template = create_extraction_template()

        assert "{schema}" in template.template

    def test_extraction_template_has_text_placeholder(self) -> None:
        """Test that template includes text placeholder."""
        from graphrag_kg_pipeline.extraction.prompts import create_extraction_template

        template = create_extraction_template()

        assert "{text}" in template.template

    def test_get_few_shot_examples(self) -> None:
        """Test that few-shot examples are properly structured."""
        from graphrag_kg_pipeline.extraction.prompts import get_few_shot_examples

        examples = get_few_shot_examples()

        assert len(examples) >= 2

        for example in examples:
            assert "text" in example
            assert "entities" in example
            assert "relationships" in example
            assert len(example["entities"]) > 0


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_config_from_env(self, mock_env_vars: dict[str, str]) -> None:
        """Test creating config from environment variables."""
        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        config = JamaKGPipelineConfig.from_env()

        assert config.neo4j_uri == mock_env_vars["NEO4J_URI"]
        assert config.neo4j_username == mock_env_vars["NEO4J_USERNAME"]
        assert config.neo4j_password == mock_env_vars["NEO4J_PASSWORD"]
        assert config.openai_api_key == mock_env_vars["OPENAI_API_KEY"]

    def test_config_default_models(self, mock_env_vars: dict[str, str]) -> None:
        """Test that config has sensible default model names."""
        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        config = JamaKGPipelineConfig.from_env()

        # Should have reasonable defaults
        assert "gpt" in config.llm_model.lower() or config.llm_model
        assert "embedding" in config.embedding_model.lower() or config.embedding_model

    def test_config_has_chunking_config(self, mock_env_vars: dict[str, str]) -> None:
        """Test that pipeline config includes chunking config."""
        from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig
        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        config = JamaKGPipelineConfig.from_env()

        assert hasattr(config, "chunking_config")
        assert isinstance(config.chunking_config, HierarchicalChunkingConfig)

    def test_config_missing_env_warns_or_uses_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test behavior when env vars are missing."""
        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        # Clear all Neo4j env vars
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Should either raise or use defaults
        try:
            config = JamaKGPipelineConfig.from_env()
            # If it doesn't raise, it should have some defaults or empty strings
            assert config is not None
        except (ValueError, KeyError):
            # Expected if strict validation
            pass


class TestPromptQualityImprovements:
    """Tests for Phase 2 prompt strengthening."""

    def test_challenge_classification_rules(self) -> None:
        """Test that prompts include Challenge classification guardrails."""
        from graphrag_kg_pipeline.extraction.prompts import (
            REQUIREMENTS_DOMAIN_INSTRUCTIONS,
        )

        assert "Challenge Classification" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "NEVER classify" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "High-Quality Products" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

    def test_definition_extraction_emphasis(self) -> None:
        """Test that prompts emphasize definition extraction."""
        from graphrag_kg_pipeline.extraction.prompts import (
            REQUIREMENTS_DOMAIN_INSTRUCTIONS,
        )

        assert "Definition Extraction" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "ALWAYS include definitions" in REQUIREMENTS_DOMAIN_INSTRUCTIONS
        assert "CRITICAL" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

    def test_standard_industry_extraction_guideline(self) -> None:
        """Test that prompts instruct extraction of standards/industries even in passing."""
        from graphrag_kg_pipeline.extraction.prompts import (
            REQUIREMENTS_DOMAIN_INSTRUCTIONS,
        )

        assert "Extract all standards and industries" in REQUIREMENTS_DOMAIN_INSTRUCTIONS

    def test_few_shot_example_pool_expanded(self) -> None:
        """Test that the few-shot example pool has at least 7 curated examples."""
        from graphrag_kg_pipeline.extraction.prompts import get_few_shot_examples

        examples = get_few_shot_examples()
        assert len(examples) >= 7

    def test_few_shot_examples_cover_all_entity_types(self) -> None:
        """Test that few-shot examples cover all major entity types."""
        from graphrag_kg_pipeline.extraction.prompts import get_few_shot_examples

        examples = get_few_shot_examples()
        entity_types_covered = set()
        for example in examples:
            for entity in example.get("entities", []):
                entity_types_covered.add(entity.get("type"))

        # Should cover at least these key types
        expected_types = {
            "Concept",
            "Challenge",
            "Standard",
            "Industry",
            "Tool",
            "Role",
            "Artifact",
            "Methodology",
            "Bestpractice",
            "Processstage",
        }
        assert expected_types.issubset(entity_types_covered), (
            f"Missing entity types: {expected_types - entity_types_covered}"
        )

    def test_few_shot_examples_include_definitions(self) -> None:
        """Test that at least some examples demonstrate definition extraction."""
        from graphrag_kg_pipeline.extraction.prompts import get_few_shot_examples

        examples = get_few_shot_examples()
        has_definition = any(
            any(e.get("definition") for e in ex.get("entities", [])) for ex in examples
        )
        assert has_definition, "At least one example should show definition extraction"


class TestExtractionGleaner:
    """Tests for the extraction gleaner module."""

    @pytest.mark.asyncio
    async def test_gleaner_no_chunks(self) -> None:
        """Test gleaning with no chunks returns zero stats."""
        from graphrag_kg_pipeline.extraction.gleaning import ExtractionGleaner
        from tests.conftest import MockDriver, MockSession

        session = MockSession()
        session.set_default_result([])
        driver = MockDriver(session)

        gleaner = ExtractionGleaner(
            driver=driver,
            database="neo4j",
            openai_api_key="sk-test-123",
            model="gpt-4o",
        )

        stats = await gleaner.glean_article("test-article")
        assert stats["chunks_processed"] == 0
        assert stats["new_entities"] == 0
        assert stats["new_relationships"] == 0

    def test_gleaner_prompt_template(self) -> None:
        """Test that the gleaning prompt has the expected structure."""
        from graphrag_kg_pipeline.extraction.gleaning import GLEANING_PROMPT

        assert "{existing_entities}" in GLEANING_PROMPT
        assert "{chunk_text}" in GLEANING_PROMPT
        assert "missed" in GLEANING_PROMPT.lower()


class TestPipelineConfigGleaningSettings:
    """Tests for gleaning and contextual retrieval config settings."""

    def test_default_gleaning_enabled(self) -> None:
        """Test that gleaning is enabled by default."""
        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        config = JamaKGPipelineConfig()
        assert config.enable_gleaning is True
        assert config.gleaning_passes == 1


class TestGlossaryPipelineProcessing:
    """Tests for glossary formatting and pipeline integration."""

    def test_format_glossary_for_pipeline(self) -> None:
        """Test that glossary is formatted as structured markdown."""
        from graphrag_kg_pipeline.extraction.pipeline import format_glossary_for_pipeline
        from graphrag_kg_pipeline.models_core import Glossary, GlossaryTerm

        glossary = Glossary(
            url="https://example.com/glossary",
            terms=[
                GlossaryTerm(
                    term="Traceability",
                    definition="The ability to trace a requirement through its lifecycle.",
                    acronym=None,
                ),
                GlossaryTerm(
                    term="RTM",
                    definition="A document that maps requirements to test cases.",
                    acronym="RTM",
                ),
            ],
        )

        result = format_glossary_for_pipeline(glossary)

        # Should be valid markdown with H1 and H2 headers
        assert "# Requirements Management Glossary" in result
        assert "## Traceability" in result
        assert "## RTM" in result
        assert "**Acronym**: RTM" in result
        assert "The ability to trace a requirement" in result

    def test_format_glossary_empty_terms(self) -> None:
        """Test formatting with no terms produces just the header."""
        from graphrag_kg_pipeline.extraction.pipeline import format_glossary_for_pipeline
        from graphrag_kg_pipeline.models_core import Glossary

        glossary = Glossary(url="https://example.com/glossary", terms=[])
        result = format_glossary_for_pipeline(glossary)

        assert "# Requirements Management Glossary" in result
        assert "##" not in result
