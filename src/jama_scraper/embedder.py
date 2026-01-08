"""Embedding generation with batch processing and checkpoint/resume.

This module provides embedding generation for text chunks using OpenAI's
embedding API. Key features:

- Batch processing for efficient API usage
- Checkpoint-based resume for long-running jobs
- Cost estimation before running
- Progress tracking with Rich console

The embedder follows the same checkpoint pattern as extractor.py,
enabling resilient processing of large chunk collections.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .chunk_models import EmbeddedChunk, EmbeddedGuideChunks
from .embedding_config import EmbeddingConfig

if TYPE_CHECKING:
    from .chunk_models import ChunkedGuide

console = Console()


class JamaEmbedder:
    """Generate embeddings for text chunks.

    Features:
    - Batch processing for efficient API usage
    - Checkpoint-based resume for long-running jobs
    - Cost estimation before running
    - Progress tracking

    Example:
        >>> config = EmbeddingConfig()
        >>> embedder = JamaEmbedder(config)
        >>> cost_estimate = embedder.estimate_cost(chunked_guide)
        >>> print(f"Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")
        >>> embeddings = await embedder.embed_chunks(chunked_guide)
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the embedder.

        Args:
            config: Embedding configuration.
        """
        self.config = config
        self._client = None  # Lazy initialization
        self._checkpoint_dir = config.checkpoint_dir

    async def _ensure_client(self) -> None:
        """Ensure OpenAI client is initialized.

        Raises:
            ImportError: If openai package is not installed.
        """
        if self._client is not None:
            return

        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI package required for embedding. "
                "Install with: uv sync --group embedding"
            ) from e

        self._client = AsyncOpenAI(api_key=self.config.effective_api_key)

    def estimate_cost(self, chunked_guide: ChunkedGuide) -> dict:
        """Estimate tokens and cost before embedding.

        Args:
            chunked_guide: ChunkedGuide with chunks to embed.

        Returns:
            Dictionary with cost estimation details.
        """
        total_chunks = len(chunked_guide.chunks)
        total_tokens = sum(chunk.token_count for chunk in chunked_guide.chunks.values())
        estimated_cost = self.config.estimate_cost(total_tokens)

        return {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.config.effective_model_id,
            "dimensions": self.config.dimensions,
            "cost_per_million_tokens": self.config.cost_per_million_tokens,
        }

    async def embed_chunks(
        self,
        chunked_guide: ChunkedGuide,
        resume: bool = True,
    ) -> EmbeddedGuideChunks:
        """Generate embeddings for all chunks.

        Args:
            chunked_guide: ChunkedGuide with chunks to embed.
            resume: Whether to resume from checkpoint (skip already embedded).

        Returns:
            EmbeddedGuideChunks with all embeddings.
        """
        # Validate configuration
        self.config.validate()

        # Initialize client
        await self._ensure_client()

        # Create checkpoint directory
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoints if resuming
        completed_ids: set[str] = set()
        embedded_chunks: dict[str, EmbeddedChunk] = {}

        if resume:
            completed_ids, embedded_chunks = self._load_checkpoints()
            if completed_ids:
                console.print(
                    f"[cyan]Resuming from checkpoint: "
                    f"{len(completed_ids)} chunks already embedded[/]"
                )

        # Get pending chunks
        all_chunks = list(chunked_guide.chunks.values())
        pending_chunks = [c for c in all_chunks if c.chunk_id not in completed_ids]

        console.print(f"\n[bold cyan]Embedding {len(pending_chunks)} chunks...[/]")
        model_info = f"{self.config.provider.value} / {self.config.effective_model_id}"
        console.print(f"Using {model_info}")

        if not pending_chunks:
            console.print("[green]All chunks already embedded![/]")
            return self._build_result(embedded_chunks, chunked_guide)

        # Estimate cost
        pending_tokens = sum(c.token_count for c in pending_chunks)
        estimated_cost = self.config.estimate_cost(pending_tokens)
        console.print(
            f"Estimated cost: ${estimated_cost:.4f} "
            f"({pending_tokens:,} tokens)"
        )

        # Process in batches
        total_embedded = 0
        batch_size = self.config.batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding...", total=len(pending_chunks))

            for batch_start in range(0, len(pending_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(pending_chunks))
                batch = pending_chunks[batch_start:batch_end]

                progress.update(
                    task,
                    description=f"Embedding batch {batch_start // batch_size + 1}...",
                )

                # Get texts for batch
                texts = [chunk.text for chunk in batch]

                # Call OpenAI API
                embeddings = await self._embed_batch(texts)

                # Create EmbeddedChunk objects and save checkpoints
                for chunk, embedding in zip(batch, embeddings, strict=True):
                    embedded = EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        embedding=embedding,
                        model_id=self.config.effective_model_id,
                    )
                    embedded_chunks[chunk.chunk_id] = embedded
                    total_embedded += 1

                    # Save checkpoint periodically
                    if total_embedded % self.config.checkpoint_frequency == 0:
                        self._save_checkpoints(embedded_chunks)

                progress.advance(task, advance=len(batch))

        # Save final checkpoints
        self._save_checkpoints(embedded_chunks)

        console.print("\n[green]Embedding complete![/]")
        console.print(f"  Chunks embedded: {len(embedded_chunks)}")

        return self._build_result(embedded_chunks, chunked_guide)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        response = await self._client.embeddings.create(
            model=self.config.effective_model_id,
            input=texts,
        )

        # Extract embeddings in order
        return [item.embedding for item in response.data]

    def _build_result(
        self,
        embedded_chunks: dict[str, EmbeddedChunk],
        chunked_guide: ChunkedGuide,
    ) -> EmbeddedGuideChunks:
        """Build the final EmbeddedGuideChunks result.

        Args:
            embedded_chunks: Dictionary of embedded chunks.
            chunked_guide: Original chunked guide for token counting.

        Returns:
            EmbeddedGuideChunks with all embeddings.
        """
        total_tokens = sum(
            chunked_guide.chunks[cid].token_count
            for cid in embedded_chunks
            if cid in chunked_guide.chunks
        )

        return EmbeddedGuideChunks(
            embeddings=embedded_chunks,
            model_id=self.config.effective_model_id,
            total_tokens_embedded=total_tokens,
            estimated_cost_usd=self.config.estimate_cost(total_tokens),
        )

    def _save_checkpoints(self, embedded_chunks: dict[str, EmbeddedChunk]) -> None:
        """Save all embeddings to checkpoint files.

        Saves one JSON file per chunk for granular resume capability.

        Args:
            embedded_chunks: Embeddings to save.
        """
        for chunk_id, embedded in embedded_chunks.items():
            checkpoint_file = self._checkpoint_dir / f"{chunk_id}.json"
            checkpoint_file.write_text(
                embedded.model_dump_json(),
                encoding="utf-8",
            )

    def _load_checkpoints(self) -> tuple[set[str], dict[str, EmbeddedChunk]]:
        """Load existing checkpoints for resume.

        Returns:
            Tuple of (completed chunk IDs, embedded chunks dict).
        """
        completed_ids: set[str] = set()
        embedded_chunks: dict[str, EmbeddedChunk] = {}

        if not self._checkpoint_dir.exists():
            return completed_ids, embedded_chunks

        for checkpoint_file in self._checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                embedded = EmbeddedChunk.model_validate(data)
                embedded_chunks[embedded.chunk_id] = embedded
                completed_ids.add(embedded.chunk_id)
            except (json.JSONDecodeError, ValueError) as e:
                console.print(
                    f"[yellow]Warning: Could not load checkpoint "
                    f"{checkpoint_file.name}: {e}[/]"
                )

        return completed_ids, embedded_chunks

    def clear_checkpoints(self) -> int:
        """Clear all checkpoint files.

        Returns:
            Number of checkpoint files deleted.
        """
        if not self._checkpoint_dir.exists():
            return 0

        count = 0
        for checkpoint_file in self._checkpoint_dir.glob("*.json"):
            checkpoint_file.unlink()
            count += 1

        return count


def check_embedding_available() -> bool:
    """Check if embedding dependencies are available.

    Returns:
        True if openai and tiktoken are installed, False otherwise.
    """
    try:
        import openai  # noqa: F401
        import tiktoken  # noqa: F401
    except ImportError:
        return False
    return True


class EmbeddingNotAvailableError(Exception):
    """Raised when embedding dependencies are not installed."""

    def __init__(self) -> None:
        """Initialize the error with installation instructions."""
        super().__init__(
            "Embedding requires openai and tiktoken packages. "
            "Install with: uv sync --group embedding"
        )
