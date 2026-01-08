"""Configuration for embedding generation.

This module provides configuration for embedding providers, with
OpenAI as the initial supported provider. The design allows for
future extension to other providers (Ollama, Voyage, Cohere).

Key features:
- Frozen dataclass for immutable configuration
- Cost estimation for API usage
- API key management (from environment or explicit)
- Model-specific dimension and cost information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Any


class EmbeddingProvider(str, Enum):
    """Supported embedding providers.

    Currently only OpenAI is supported. Future providers can be added:
    - OLLAMA = "ollama"  # nomic-embed-text (free, local)
    - VOYAGE = "voyage"  # voyage-3, voyage-3-lite
    - COHERE = "cohere"  # embed-english-v3.0
    """

    OPENAI = "openai"


# Default embedding models per provider
DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
}

# Embedding dimensions per model
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    # Future models:
    # "nomic-embed-text": 768,
    # "voyage-3": 1024,
    # "voyage-3-lite": 512,
    # "embed-english-v3.0": 1024,
}

# Cost per 1 million tokens (USD)
EMBEDDING_COSTS_PER_MILLION: dict[str, float] = {
    "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
    # Ollama models are free (local)
}

# Environment variable names for API keys
API_KEY_ENV_VARS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OPENAI: "OPENAI_API_KEY",
}


@dataclass(frozen=True)
class EmbeddingConfig:
    """Frozen configuration for embedding generation.

    This configuration controls how embeddings are generated and cached.
    Uses OpenAI's text-embedding-3-small by default for cost efficiency.

    Attributes:
        provider: Embedding provider (currently only OPENAI).
        model_id: Specific model ID (uses provider default if None).
        api_key: API key (from environment if None).
        batch_size: Number of chunks per API batch request.
        max_concurrent_batches: Maximum parallel batch requests.
        checkpoint_dir: Directory for embedding checkpoints (resume).
        checkpoint_frequency: Save checkpoint every N chunks.
    """

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model_id: str | None = None
    api_key: str | None = None

    # Batching configuration
    batch_size: int = 100  # OpenAI supports up to 2048 inputs
    max_concurrent_batches: int = 3

    # Checkpoint configuration
    checkpoint_dir: Path = field(default_factory=lambda: Path(".embedding_cache"))
    checkpoint_frequency: int = 50  # Save every N chunks

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validation is deferred to validate() method
        # to allow creation of config before API key is set

    @property
    def effective_model_id(self) -> str:
        """Get the effective model ID (uses default if not specified).

        Returns:
            Model ID string.
        """
        if self.model_id:
            return self.model_id
        return DEFAULT_EMBEDDING_MODELS[self.provider]

    @property
    def effective_api_key(self) -> str | None:
        """Get the effective API key (from env if not specified).

        Returns:
            API key string or None if not available.
        """
        if self.api_key:
            return self.api_key
        env_var = API_KEY_ENV_VARS.get(self.provider)
        if env_var:
            return os.environ.get(env_var)
        return None

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions for the model.

        Returns:
            Number of dimensions in embedding vectors.
        """
        return EMBEDDING_DIMENSIONS.get(self.effective_model_id, 1536)

    @property
    def cost_per_million_tokens(self) -> float:
        """Get the cost per million tokens for the model.

        Returns:
            Cost in USD per million tokens, or 0.0 if free/unknown.
        """
        return EMBEDDING_COSTS_PER_MILLION.get(self.effective_model_id, 0.0)

    def estimate_cost(self, total_tokens: int) -> float:
        """Estimate API cost for embedding a given number of tokens.

        Args:
            total_tokens: Total tokens to embed.

        Returns:
            Estimated cost in USD.
        """
        cost_per_token = self.cost_per_million_tokens / 1_000_000
        return total_tokens * cost_per_token

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.provider not in EmbeddingProvider:
            msg = f"Unsupported provider: {self.provider}"
            raise ValueError(msg)

        if not self.effective_api_key:
            env_var = API_KEY_ENV_VARS.get(self.provider, "API_KEY")
            msg = (
                f"No API key available for {self.provider.value}. "
                f"Set {env_var} environment variable or pass api_key parameter."
            )
            raise ValueError(msg)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.max_concurrent_batches <= 0:
            raise ValueError("max_concurrent_batches must be positive")

        if self.checkpoint_frequency <= 0:
            raise ValueError("checkpoint_frequency must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "provider": self.provider.value,
            "model_id": self.effective_model_id,
            "batch_size": self.batch_size,
            "max_concurrent_batches": self.max_concurrent_batches,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_frequency": self.checkpoint_frequency,
            "dimensions": self.dimensions,
            "cost_per_million_tokens": self.cost_per_million_tokens,
        }

    @classmethod
    def from_args(
        cls,
        provider: str | EmbeddingProvider = "openai",
        model_id: str | None = None,
        api_key: str | None = None,
        checkpoint_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> EmbeddingConfig:
        """Create configuration from CLI arguments.

        Args:
            provider: Provider name or enum.
            model_id: Specific model ID.
            api_key: API key (optional, falls back to environment).
            checkpoint_dir: Checkpoint directory path.
            **kwargs: Additional configuration options.

        Returns:
            EmbeddingConfig instance.
        """
        # Convert string provider to enum
        if isinstance(provider, str):
            provider = EmbeddingProvider(provider.lower())

        config_dict: dict[str, Any] = {
            "provider": provider,
        }

        if model_id is not None:
            config_dict["model_id"] = model_id

        if api_key is not None:
            config_dict["api_key"] = api_key

        if checkpoint_dir is not None:
            if isinstance(checkpoint_dir, str):
                checkpoint_dir = Path(checkpoint_dir)
            config_dict["checkpoint_dir"] = checkpoint_dir

        # Merge additional kwargs
        for key in ["batch_size", "max_concurrent_batches", "checkpoint_frequency"]:
            if key in kwargs and kwargs[key] is not None:
                config_dict[key] = kwargs[key]

        return cls(**config_dict)


# Default configuration instance
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
