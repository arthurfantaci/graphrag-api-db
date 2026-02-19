"""Tests for LangExtract augmentation module.

Tests the LangExtractAugmenter class with mocked Neo4j driver and
LangExtract library, verifying entity creation, dedup checks, and
source grounding.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLangExtractAugmenter:
    """Tests for the LangExtractAugmenter class."""

    def _make_augmenter(self, mock_driver=None):
        from graphrag_kg_pipeline.postprocessing.langextract_augmenter import (
            LangExtractAugmenter,
        )

        driver = mock_driver or AsyncMock()
        return LangExtractAugmenter(
            driver=driver,
            database="neo4j",
            openai_api_key="sk-test",
            model="gpt-4o",
        )

    def test_initialization(self) -> None:
        augmenter = self._make_augmenter()

        assert augmenter.database == "neo4j"
        assert augmenter.openai_api_key == "sk-test"
        assert augmenter.model == "gpt-4o"
        assert augmenter.new_count == 0
        assert augmenter.grounded_count == 0

    def test_build_examples(self) -> None:
        augmenter = self._make_augmenter()
        examples = augmenter._build_examples()

        assert len(examples) == 1
        assert len(examples[0].extractions) == 5
        classes = {e.extraction_class for e in examples[0].extractions}
        assert "Concept" in classes
        assert "Standard" in classes
        assert "Industry" in classes
        assert "Tool" in classes
        assert "Artifact" in classes

    @pytest.mark.asyncio
    async def test_augment_no_chunks(self) -> None:
        augmenter = self._make_augmenter()
        augmenter._get_chunks = AsyncMock(return_value=[])

        stats = await augmenter.augment()

        assert stats["new_entities"] == 0
        assert stats["grounded_entities"] == 0

    @pytest.mark.asyncio
    @patch("langextract.extract")
    async def test_augment_creates_new_entity(self, mock_extract) -> None:
        augmenter = self._make_augmenter()

        augmenter._get_chunks = AsyncMock(
            return_value=[
                {
                    "id": "chunk-1",
                    "text": "Requirements traceability is essential for managing complex systems and ensuring compliance.",
                }
            ]
        )
        augmenter._entity_exists = AsyncMock(return_value=False)
        augmenter._create_entity = AsyncMock()
        augmenter._add_source_span = AsyncMock()

        mock_extraction = MagicMock()
        mock_extraction.extraction_class = "Concept"
        mock_extraction.extraction_text = "Requirements Traceability"
        mock_extraction.char_interval = MagicMock()
        mock_extraction.char_interval.start_pos = 0
        mock_extraction.char_interval.end_pos = 26

        mock_result = MagicMock()
        mock_result.extractions = [mock_extraction]
        mock_extract.return_value = mock_result

        stats = await augmenter.augment()

        augmenter._create_entity.assert_called_once()
        call_kwargs = augmenter._create_entity.call_args[1]
        assert call_kwargs["name"] == "requirements traceability"
        assert call_kwargs["label"] == "Concept"
        assert call_kwargs["source_span"] == "0-26"
        assert stats["new_entities"] == 1

    @pytest.mark.asyncio
    @patch("langextract.extract")
    async def test_augment_grounds_existing_entity(self, mock_extract) -> None:
        augmenter = self._make_augmenter()

        augmenter._get_chunks = AsyncMock(
            return_value=[
                {
                    "id": "chunk-1",
                    "text": "Requirements traceability is essential for managing complex systems and ensuring compliance.",
                }
            ]
        )
        augmenter._entity_exists = AsyncMock(return_value=True)
        augmenter._create_entity = AsyncMock()
        augmenter._add_source_span = AsyncMock()

        mock_extraction = MagicMock()
        mock_extraction.extraction_class = "Concept"
        mock_extraction.extraction_text = "Requirements Traceability"
        mock_extraction.char_interval = MagicMock()
        mock_extraction.char_interval.start_pos = 0
        mock_extraction.char_interval.end_pos = 26

        mock_result = MagicMock()
        mock_result.extractions = [mock_extraction]
        mock_extract.return_value = mock_result

        stats = await augmenter.augment()

        augmenter._create_entity.assert_not_called()
        augmenter._add_source_span.assert_called_once()
        assert stats["grounded_entities"] == 1

    @pytest.mark.asyncio
    @patch("langextract.extract")
    async def test_augment_skips_invalid_classes(self, mock_extract) -> None:
        augmenter = self._make_augmenter()

        augmenter._get_chunks = AsyncMock(
            return_value=[
                {"id": "chunk-1", "text": "Some text with enough content for processing."}
            ]
        )
        augmenter._entity_exists = AsyncMock(return_value=False)
        augmenter._create_entity = AsyncMock()

        mock_extraction = MagicMock()
        mock_extraction.extraction_class = "InvalidType"
        mock_extraction.extraction_text = "something"
        mock_extraction.char_interval = None

        mock_result = MagicMock()
        mock_result.extractions = [mock_extraction]
        mock_extract.return_value = mock_result

        stats = await augmenter.augment()

        augmenter._create_entity.assert_not_called()
        assert stats["new_entities"] == 0

    @pytest.mark.asyncio
    @patch("langextract.extract")
    async def test_augment_skips_short_names(self, mock_extract) -> None:
        augmenter = self._make_augmenter()

        augmenter._get_chunks = AsyncMock(
            return_value=[
                {"id": "chunk-1", "text": "Some text with enough content for processing."}
            ]
        )
        augmenter._entity_exists = AsyncMock(return_value=False)
        augmenter._create_entity = AsyncMock()

        mock_extraction = MagicMock()
        mock_extraction.extraction_class = "Concept"
        mock_extraction.extraction_text = "a"  # Too short
        mock_extraction.char_interval = None

        mock_result = MagicMock()
        mock_result.extractions = [mock_extraction]
        mock_extract.return_value = mock_result

        stats = await augmenter.augment()

        augmenter._create_entity.assert_not_called()
        assert stats["new_entities"] == 0

    def test_extraction_classes_match_schema(self) -> None:
        from graphrag_kg_pipeline.extraction.schema import NODE_TYPES
        from graphrag_kg_pipeline.postprocessing.langextract_augmenter import (
            _EXTRACTION_CLASSES,
        )

        schema_labels = set(NODE_TYPES.keys())
        augmenter_classes = set(_EXTRACTION_CLASSES)

        assert augmenter_classes == schema_labels
