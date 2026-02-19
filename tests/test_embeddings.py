"""Tests for embedding providers.

Tests the VoyageAIEmbeddings class and the auto-detection logic
in the pipeline factory.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestVoyageAIEmbeddings:
    """Tests for the Voyage AI embeddings class."""

    @patch("voyageai.Client")
    @patch("voyageai.AsyncClient")
    def test_embed_query_passes_input_type(self, mock_async_client_cls, mock_client_cls) -> None:
        """Verify input_type is passed to the Voyage API."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2, 0.3]]
        mock_client.embed.return_value = mock_result
        mock_client_cls.return_value = mock_client

        from graphrag_kg_pipeline.embeddings.voyage import VoyageAIEmbeddings

        embedder = VoyageAIEmbeddings(
            model="voyage-4",
            input_type="document",
            dimensions=1536,
        )
        result = embedder.embed_query("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embed.assert_called_once_with(
            ["test text"],
            model="voyage-4",
            input_type="document",
            output_dimension=1536,
        )

    @patch("voyageai.Client")
    @patch("voyageai.AsyncClient")
    def test_embed_query_with_query_input_type(
        self, mock_async_client_cls, mock_client_cls
    ) -> None:
        """Verify query input_type works for search embeddings."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.4, 0.5, 0.6]]
        mock_client.embed.return_value = mock_result
        mock_client_cls.return_value = mock_client

        from graphrag_kg_pipeline.embeddings.voyage import VoyageAIEmbeddings

        embedder = VoyageAIEmbeddings(input_type="query")
        result = embedder.embed_query("search query")

        assert result == [0.4, 0.5, 0.6]
        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args
        assert call_kwargs[1]["input_type"] == "query"

    @patch("voyageai.Client")
    @patch("voyageai.AsyncClient")
    def test_embed_query_raises_on_error(self, mock_async_client_cls, mock_client_cls) -> None:
        """Verify errors are wrapped in EmbeddingsGenerationError."""
        mock_client = MagicMock()
        mock_client.embed.side_effect = RuntimeError("API error")
        mock_client_cls.return_value = mock_client

        from neo4j_graphrag.exceptions import EmbeddingsGenerationError

        from graphrag_kg_pipeline.embeddings.voyage import VoyageAIEmbeddings

        embedder = VoyageAIEmbeddings()

        with pytest.raises(EmbeddingsGenerationError, match="API error"):
            embedder.embed_query("test")

    @pytest.mark.asyncio
    @patch("voyageai.Client")
    @patch("voyageai.AsyncClient")
    async def test_async_embed_query(self, mock_async_client_cls, mock_client_cls) -> None:
        """Verify async embedding works."""
        mock_async_client = MagicMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.7, 0.8, 0.9]]
        mock_async_client.embed = MagicMock(return_value=mock_result)

        # Make it a coroutine
        async def async_embed(*_args, **_kwargs):
            return mock_result

        mock_async_client.embed = async_embed
        mock_async_client_cls.return_value = mock_async_client

        from graphrag_kg_pipeline.embeddings.voyage import VoyageAIEmbeddings

        embedder = VoyageAIEmbeddings()
        result = await embedder.async_embed_query("async test")

        assert result == [0.7, 0.8, 0.9]


class TestEmbedderAutoDetection:
    """Tests for the embedder auto-detection in pipeline config."""

    def test_voyage_api_key_from_env(self, mock_env_vars, monkeypatch) -> None:
        """Verify VOYAGE_API_KEY is loaded from environment."""
        monkeypatch.setenv("VOYAGE_API_KEY", "voy-test-key")

        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        config = JamaKGPipelineConfig.from_env()
        assert config.voyage_api_key == "voy-test-key"

    def test_no_voyage_key_defaults_empty(self, mock_env_vars) -> None:
        """Verify missing VOYAGE_API_KEY defaults to empty string."""
        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        config = JamaKGPipelineConfig.from_env()
        assert config.voyage_api_key == ""

    def test_config_has_voyage_fields(self) -> None:
        """Verify config dataclass has Voyage AI fields."""
        from graphrag_kg_pipeline.extraction.pipeline import JamaKGPipelineConfig

        config = JamaKGPipelineConfig()
        assert config.voyage_model == "voyage-4"
        assert config.voyage_api_key == ""
