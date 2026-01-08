"""Configuration for LangExtract-based semantic enrichment.

Manages LLM provider settings, API keys, and extraction parameters.
Supports OpenAI (default), Google Gemini, and local Ollama inference.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path


class LLMProvider(str, Enum):
    """Supported LLM providers for LangExtract."""

    OPENAI = "openai"
    """OpenAI GPT models (gpt-4o, gpt-4o-mini)."""

    GEMINI = "gemini"
    """Google Gemini models (gemini-2.5-flash, gemini-2.5-pro)."""

    OLLAMA = "ollama"
    """Local Ollama inference (gemma2, llama3, etc.)."""


# Default model IDs for each provider
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OLLAMA: "gemma2:2b",
}

# API key environment variable names
API_KEY_ENV_VARS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.GEMINI: "LANGEXTRACT_API_KEY",  # Or GOOGLE_API_KEY
    LLMProvider.OLLAMA: "",  # No API key needed for local
}


@dataclass(frozen=True)
class EnrichmentConfig:
    """Configuration for LangExtract semantic enrichment.

    This dataclass is immutable (frozen) to ensure configuration
    doesn't change during extraction.

    Attributes:
        provider: LLM provider to use (openai, gemini, ollama).
        model_id: Specific model identifier (e.g., 'gpt-4o').
        api_key: API key for the provider (auto-loaded from env if None).
        extraction_passes: Number of LangExtract passes (higher = better recall).
        max_workers: Parallel workers for batch processing.
        max_char_buffer: Context window size for chunking.
        temperature: LLM temperature (lower = more deterministic).
        checkpoint_dir: Directory for saving progress checkpoints.
        ollama_url: URL for local Ollama server.

    Example:
        config = EnrichmentConfig(provider=LLMProvider.OPENAI)
        # API key auto-loaded from OPENAI_API_KEY env var
    """

    # Provider settings
    provider: LLMProvider = LLMProvider.OPENAI
    model_id: str | None = None  # Uses default for provider if None
    api_key: str | None = None  # Auto-loaded from environment if None

    # Extraction settings
    extraction_passes: int = 2
    """Number of extraction passes. Higher values improve recall but cost more."""

    max_workers: int = 10
    """Parallel workers for processing articles."""

    max_char_buffer: int = 1500
    """Context window size for text chunking (smaller = more accurate)."""

    temperature: float = 0.3
    """LLM temperature. Lower values produce more deterministic output."""

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path(".enrichment_cache"))
    """Directory for saving extraction progress (enables resume)."""

    # Ollama-specific
    ollama_url: str = "http://localhost:11434"
    """URL for local Ollama server."""

    def __post_init__(self) -> None:
        """Validate configuration and load API key from environment if needed."""
        # Convert checkpoint_dir to Path if string
        if isinstance(self.checkpoint_dir, str):
            object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

    @property
    def effective_model_id(self) -> str:
        """Get the model ID, using provider default if not specified."""
        if self.model_id:
            return self.model_id
        return DEFAULT_MODELS[self.provider]

    @property
    def effective_api_key(self) -> str | None:
        """Get API key from config or environment variable."""
        if self.api_key:
            return self.api_key

        env_var = API_KEY_ENV_VARS.get(self.provider, "")
        if env_var:
            return os.environ.get(env_var)

        return None

    def validate(self) -> None:
        """Validate the configuration, raising errors for missing requirements.

        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        # Check API key for cloud providers
        if self.provider != LLMProvider.OLLAMA and not self.effective_api_key:
            env_var = API_KEY_ENV_VARS[self.provider]
            msg = (
                f"API key required for {self.provider.value}. "
                f"Set {env_var} environment variable or pass api_key parameter."
            )
            raise ValueError(msg)

        # Validate extraction_passes
        if self.extraction_passes < 1:
            msg = "extraction_passes must be at least 1"
            raise ValueError(msg)

        # Validate max_workers
        if self.max_workers < 1:
            msg = "max_workers must be at least 1"
            raise ValueError(msg)

        # Validate temperature (LLM temperature range)
        max_temperature = 2.0
        if not 0.0 <= self.temperature <= max_temperature:
            msg = f"temperature must be between 0.0 and {max_temperature}"
            raise ValueError(msg)

    def to_langextract_params(self) -> dict:
        """Convert to parameters for langextract.extract() call.

        Returns:
            Dictionary of parameters for LangExtract.
        """
        params = {
            "model_id": self.effective_model_id,
            "extraction_passes": self.extraction_passes,
            "max_workers": self.max_workers,
            "max_char_buffer": self.max_char_buffer,
            "temperature": self.temperature,
            "show_progress": False,
        }

        # Add API key for cloud providers
        if self.provider != LLMProvider.OLLAMA:
            params["api_key"] = self.effective_api_key

        # Add Ollama-specific settings
        if self.provider == LLMProvider.OLLAMA:
            params["model_url"] = self.ollama_url
            params["fence_output"] = False
            params["use_schema_constraints"] = False

        # OpenAI-specific settings
        if self.provider == LLMProvider.OPENAI:
            params["fence_output"] = True
            params["use_schema_constraints"] = False

        return params


def create_config_from_args(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    extraction_passes: int = 2,
    max_workers: int = 10,
) -> EnrichmentConfig:
    """Create EnrichmentConfig from CLI arguments.

    Args:
        provider: Provider name ('openai', 'gemini', 'ollama').
        model: Model identifier (uses provider default if None).
        api_key: API key (loaded from environment if None).
        extraction_passes: Number of extraction passes.
        max_workers: Parallel workers for batch processing.

    Returns:
        Configured EnrichmentConfig instance.

    Example:
        config = create_config_from_args(provider="openai", model="gpt-4o-mini")
    """
    # Parse provider
    llm_provider = LLMProvider.OPENAI
    if provider:
        try:
            llm_provider = LLMProvider(provider.lower())
        except ValueError:
            valid = ", ".join(p.value for p in LLMProvider)
            msg = f"Invalid provider '{provider}'. Must be one of: {valid}"
            raise ValueError(msg) from None

    return EnrichmentConfig(
        provider=llm_provider,
        model_id=model,
        api_key=api_key,
        extraction_passes=extraction_passes,
        max_workers=max_workers,
    )
