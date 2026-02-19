"""Tests for Leiden community detection and summarization.

Tests the CommunityDetector and CommunitySummarizer with mocked
Neo4j driver, using real igraph/leidenalg for community detection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCommunityDetector:
    """Tests for the CommunityDetector class."""

    def _make_detector(self, mock_driver=None):
        from graphrag_kg_pipeline.graph.community_detection import CommunityDetector

        driver = mock_driver or AsyncMock()
        return CommunityDetector(driver=driver, database="neo4j")

    def test_initialization(self) -> None:
        detector = self._make_detector()
        assert detector.database == "neo4j"

    def test_semantic_rel_types_from_schema(self) -> None:
        from graphrag_kg_pipeline.extraction.schema import RELATIONSHIP_TYPES
        from graphrag_kg_pipeline.graph.community_detection import _SEMANTIC_REL_TYPES

        assert set(_SEMANTIC_REL_TYPES) == set(RELATIONSHIP_TYPES.keys())
        assert "ADDRESSES" in _SEMANTIC_REL_TYPES
        assert "REQUIRES" in _SEMANTIC_REL_TYPES
        # Should NOT include structural relationships
        assert "FROM_ARTICLE" not in _SEMANTIC_REL_TYPES
        assert "MENTIONED_IN" not in _SEMANTIC_REL_TYPES

    @pytest.mark.asyncio
    async def test_detect_no_edges(self) -> None:
        detector = self._make_detector()
        detector._export_semantic_edges = AsyncMock(return_value=([], set()))

        stats = await detector.detect_communities()

        assert stats["community_count"] == 0
        assert stats["modularity"] == 0.0
        assert stats["node_count"] == 0

    @pytest.mark.asyncio
    async def test_detect_with_edges(self) -> None:
        """Test Leiden with real igraph/leidenalg on mock edges."""
        detector = self._make_detector()

        # Two clusters of connected nodes
        edges = [
            ("traceability", "scope creep"),
            ("traceability", "requirements elicitation"),
            ("scope creep", "requirements elicitation"),
            ("iso 26262", "automotive"),
            ("iso 26262", "functional safety"),
            ("automotive", "functional safety"),
        ]
        node_names = {
            "traceability",
            "scope creep",
            "requirements elicitation",
            "iso 26262",
            "automotive",
            "functional safety",
        }
        detector._export_semantic_edges = AsyncMock(return_value=(edges, node_names))
        detector._write_community_ids = AsyncMock()

        stats = await detector.detect_communities()

        # Should find at least 1 community
        assert stats["community_count"] >= 1
        assert stats["node_count"] == 6
        assert isinstance(stats["modularity"], float)
        detector._write_community_ids.assert_called_once()

        # Verify the assignments dict has all nodes
        assignments = detector._write_community_ids.call_args[0][0]
        assert len(assignments) == 6

    @pytest.mark.asyncio
    async def test_detect_single_component(self) -> None:
        """A fully connected graph should form one community."""
        detector = self._make_detector()

        edges = [
            ("a", "b"),
            ("b", "c"),
            ("a", "c"),
        ]
        detector._export_semantic_edges = AsyncMock(return_value=(edges, {"a", "b", "c"}))
        detector._write_community_ids = AsyncMock()

        stats = await detector.detect_communities()

        assert stats["community_count"] >= 1
        assert stats["node_count"] == 3


class TestCommunitySummarizer:
    """Tests for the CommunitySummarizer class."""

    def _make_summarizer(self, mock_driver=None):
        from graphrag_kg_pipeline.graph.community_summarizer import CommunitySummarizer

        driver = mock_driver or AsyncMock()
        return CommunitySummarizer(
            driver=driver,
            database="neo4j",
            openai_api_key="sk-test",
            model="gpt-4o-mini",
            min_community_size=3,
        )

    def test_initialization(self) -> None:
        summarizer = self._make_summarizer()
        assert summarizer.min_community_size == 3
        assert summarizer.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_skip_small_communities(self) -> None:
        summarizer = self._make_summarizer()

        # Community with only 2 members (below min_community_size=3)
        summarizer._get_communities = AsyncMock(
            return_value={
                0: [
                    {"name": "a", "label": "Concept", "description": ""},
                    {"name": "b", "label": "Concept", "description": ""},
                ],
            }
        )

        # Mock the OpenAI client import inside summarize_communities
        mock_client = AsyncMock()
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            stats = await summarizer.summarize_communities()

        assert stats["communities_summarized"] == 0
        assert stats["communities_skipped"] == 1

    @pytest.mark.asyncio
    async def test_summarize_large_community(self) -> None:
        summarizer = self._make_summarizer()

        summarizer._get_communities = AsyncMock(
            return_value={
                0: [
                    {
                        "name": "traceability",
                        "label": "Concept",
                        "description": "Tracking requirements",
                    },
                    {
                        "name": "scope creep",
                        "label": "Challenge",
                        "description": "Uncontrolled growth",
                    },
                    {
                        "name": "requirements elicitation",
                        "label": "Concept",
                        "description": "Gathering requirements",
                    },
                ],
            }
        )
        summarizer._create_community_node = AsyncMock()

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "This community covers requirements traceability."

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            stats = await summarizer.summarize_communities()

        assert stats["communities_summarized"] == 1
        summarizer._create_community_node.assert_called_once()
