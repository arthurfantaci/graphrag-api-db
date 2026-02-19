"""Contextual chunk enrichment using LLM-generated context prefixes.

Implements the Anthropic Contextual Retrieval approach: for each chunk,
generates a brief context prefix that situates it within the full document.
This prefix is prepended before embedding (not before extraction) to
improve retrieval accuracy by up to 67%.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

CONTEXT_PROMPT = """Here is the full article:
<article>
{article_text}
</article>

Here is a chunk from that article:
<chunk>
{chunk_text}
</chunk>

Give a short succinct context (1-2 sentences) to situate this chunk within the
overall article for improving search retrieval. Focus on what section this is from
and what topic it covers. Answer only with the context, nothing else."""


class ContextualChunkEnricher:
    """Enriches chunks with LLM-generated context prefixes.

    For each chunk, sends the article text + chunk to an LLM to generate
    a brief contextual prefix. This prefix is stored in chunk metadata
    and prepended to page_content before embedding.

    Attributes:
        openai_api_key: API key for OpenAI.
        model: Model to use for context generation.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the enricher.

        Args:
            openai_api_key: OpenAI API key.
            model: Model name for context generation (default: gpt-4o-mini for cost).
        """
        self.openai_api_key = openai_api_key
        self.model = model

    async def enrich_chunks(
        self,
        chunks: list[Document],
        article_text: str,
    ) -> list[Document]:
        """Enrich chunks with contextual prefixes.

        Args:
            chunks: List of Document objects from chunking.
            article_text: Full article text for context.

        Returns:
            Enriched chunks with contextual_prefix in metadata and
            prefix prepended to page_content.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        # Truncate article text to avoid exceeding context limits
        max_article_chars = 12000
        truncated_article = article_text[:max_article_chars]
        if len(article_text) > max_article_chars:
            truncated_article += "\n[... truncated]"

        enriched = []
        for chunk in chunks:
            try:
                prompt = CONTEXT_PROMPT.format(
                    article_text=truncated_article,
                    chunk_text=chunk.page_content,
                )

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,
                )

                context_prefix = response.choices[0].message.content or ""
                context_prefix = context_prefix.strip()

                # Store in metadata and prepend to content
                chunk.metadata["contextual_prefix"] = context_prefix
                if context_prefix:
                    chunk.page_content = context_prefix + "\n\n" + chunk.page_content

                enriched.append(chunk)

            except Exception:
                logger.warning(
                    "Failed to enrich chunk, keeping original",
                    chunk_preview=chunk.page_content[:80],
                    exc_info=True,
                )
                enriched.append(chunk)

        logger.info(
            "Contextual enrichment complete",
            total_chunks=len(chunks),
            enriched=sum(1 for c in enriched if "contextual_prefix" in c.metadata),
        )

        return enriched

    def enrich_chunks_sync(
        self,
        chunks: list[Document],
        article_text: str,
    ) -> list[Document]:
        """Synchronous wrapper for enrich_chunks.

        Args:
            chunks: List of Document objects from chunking.
            article_text: Full article text for context.

        Returns:
            Enriched chunks.
        """
        import asyncio

        return asyncio.run(self.enrich_chunks(chunks, article_text))


def create_context_enricher(
    openai_api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> ContextualChunkEnricher | None:
    """Factory function to create a context enricher.

    Args:
        openai_api_key: OpenAI API key. If None, returns None (disabled).
        model: Model name for context generation.

    Returns:
        ContextualChunkEnricher instance, or None if disabled.
    """
    if not openai_api_key:
        return None
    return ContextualChunkEnricher(openai_api_key=openai_api_key, model=model)
