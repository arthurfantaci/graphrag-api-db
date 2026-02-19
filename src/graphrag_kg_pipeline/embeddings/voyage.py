"""Voyage AI embeddings for asymmetric document/query retrieval.

Voyage-4 provides state-of-the-art retrieval performance with separate
input_type for documents ("document") vs queries ("query"), enabling
asymmetric embeddings that improve RAG accuracy.
"""

from __future__ import annotations

from typing import Any

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
import structlog

logger = structlog.get_logger(__name__)


class VoyageAIEmbeddings(Embedder):
    """Voyage AI embeddings implementing the neo4j_graphrag Embedder interface.

    Uses Voyage AI's asymmetric embedding model for improved retrieval.
    Documents are embedded with input_type="document" and queries with
    input_type="query" for optimal RAG performance.

    Attributes:
        model: Voyage AI model name (default: voyage-4).
        input_type: Embedding input type ("document" for indexing, "query" for search).
        dimensions: Output embedding dimensions (default: 1024).
    """

    def __init__(
        self,
        model: str = "voyage-4",
        input_type: str = "document",
        dimensions: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Initialize the Voyage AI embedder.

        Args:
            model: Model name (default: voyage-4).
            input_type: Either "document" (for indexing) or "query" (for search).
            dimensions: Output vector dimensions (default: 1024).
            **kwargs: Additional arguments passed to Voyage AI client.
        """
        super().__init__()
        import voyageai  # noqa: PLC0415

        self.model = model
        self.input_type = input_type
        self.dimensions = dimensions
        self.client = voyageai.Client(**kwargs)
        self.async_client = voyageai.AsyncClient(**kwargs)

        logger.info(
            "Initialized Voyage AI embeddings",
            model=model,
            input_type=input_type,
            dimensions=dimensions,
        )

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.
            **kwargs: Additional arguments for the Voyage API.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingsGenerationError: If the API call fails.
        """
        try:
            result = self.client.embed(
                [text],
                model=self.model,
                input_type=self.input_type,
                output_dimension=self.dimensions,
                **kwargs,
            )
            return result.embeddings[0]
        except EmbeddingsGenerationError:
            raise
        except Exception as e:
            msg = f"Voyage AI embedding failed: {e}"
            raise EmbeddingsGenerationError(msg) from e

    async def async_embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Asynchronously embed a single text string.

        Args:
            text: Text to embed.
            **kwargs: Additional arguments for the Voyage API.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingsGenerationError: If the API call fails.
        """
        try:
            result = await self.async_client.embed(
                [text],
                model=self.model,
                input_type=self.input_type,
                output_dimension=self.dimensions,
                **kwargs,
            )
            return result.embeddings[0]
        except EmbeddingsGenerationError:
            raise
        except Exception as e:
            msg = f"Voyage AI async embedding failed: {e}"
            raise EmbeddingsGenerationError(msg) from e
