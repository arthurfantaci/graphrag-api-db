"""SimpleKGPipeline factory for Jama Guide extraction.

This module provides configuration and factory functions for creating
neo4j_graphrag SimpleKGPipeline instances configured for the
requirements management domain.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from jama_scraper.chunking.adapter import create_text_splitter_adapter
from jama_scraper.chunking.config import HierarchicalChunkingConfig
from jama_scraper.extraction.prompts import create_extraction_template
from jama_scraper.extraction.schema import get_schema_for_pipeline

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

    from jama_scraper.models import RequirementsManagementGuide

logger = structlog.get_logger(__name__)


@dataclass
class JamaKGPipelineConfig:
    """Configuration for the Jama Knowledge Graph pipeline.

    Consolidates all configuration for Neo4j connection, LLM providers,
    embedding models, and chunking settings.

    Attributes:
        neo4j_uri: Neo4j connection URI.
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        openai_api_key: OpenAI API key for LLM and embeddings.
        llm_model: LLM model name for extraction.
        embedding_model: Embedding model name.
        chunking_config: Hierarchical chunking configuration.
        batch_size: Number of chunks to process per batch.
        perform_entity_resolution: Whether to resolve duplicate entities.
        document_node_label: Label for document (article) nodes.
        chunk_node_label: Label for chunk nodes.
    """

    # Neo4j connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # OpenAI configuration
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Chunking configuration
    chunking_config: HierarchicalChunkingConfig = field(
        default_factory=HierarchicalChunkingConfig
    )

    # Pipeline settings
    batch_size: int = 10
    perform_entity_resolution: bool = True

    # Graph labels
    document_node_label: str = "Article"
    chunk_node_label: str = "Chunk"
    chunk_to_document_relationship: str = "FROM_ARTICLE"
    node_to_chunk_relationship: str = "MENTIONED_IN"

    @classmethod
    def from_env(cls) -> "JamaKGPipelineConfig":
        """Create configuration from environment variables.

        Reads from standard environment variables:
        - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
        - OPENAI_API_KEY

        Returns:
            Configuration populated from environment.

        Raises:
            ValueError: If required environment variables are missing.
        """
        import os

        from dotenv import load_dotenv

        load_dotenv()

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if not openai_api_key:
            msg = "OPENAI_API_KEY environment variable is required"
            raise ValueError(msg)

        return cls(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            openai_api_key=openai_api_key,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding sensitive values).

        Returns:
            Dictionary representation with passwords masked.
        """
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_username": self.neo4j_username,
            "neo4j_database": self.neo4j_database,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "batch_size": self.batch_size,
            "perform_entity_resolution": self.perform_entity_resolution,
            "document_node_label": self.document_node_label,
            "chunk_node_label": self.chunk_node_label,
        }


def create_neo4j_driver(config: JamaKGPipelineConfig) -> "Driver":
    """Create a Neo4j driver from configuration.

    Args:
        config: Pipeline configuration.

    Returns:
        Neo4j driver instance.
    """
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )


def create_async_neo4j_driver(config: JamaKGPipelineConfig) -> "Driver":
    """Create an async Neo4j driver from configuration.

    Args:
        config: Pipeline configuration.

    Returns:
        Async Neo4j driver instance for use with async context managers.
    """
    from neo4j import AsyncGraphDatabase

    return AsyncGraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )


def create_jama_kg_pipeline(
    config: JamaKGPipelineConfig,
) -> "SimpleKGPipeline":
    """Create a configured SimpleKGPipeline for Jama Guide extraction.

    Factory function that assembles all components:
    - Neo4j driver and connection
    - OpenAI LLM for extraction
    - OpenAI embeddings
    - LangChain text splitter adapter
    - Domain-specific extraction template
    - Requirements management schema

    Args:
        config: Pipeline configuration.

    Returns:
        Configured SimpleKGPipeline ready for processing.

    Example:
        >>> config = JamaKGPipelineConfig.from_env()
        >>> pipeline = create_jama_kg_pipeline(config)
        >>> await pipeline.run(text="...")
    """
    from neo4j_graphrag.embeddings import OpenAIEmbeddings
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.llm import OpenAILLM

    logger.info("Creating Jama KG pipeline", config=config.to_dict())

    # Create Neo4j driver
    driver = create_neo4j_driver(config)

    # Create LLM for extraction with JSON response format
    llm = OpenAILLM(
        model_name=config.llm_model,
        api_key=config.openai_api_key,
        model_params={
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
    )

    # Create embeddings
    embedder = OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.openai_api_key,
    )

    # Create text splitter adapter
    text_splitter = create_text_splitter_adapter(
        config=config.chunking_config,
        use_markdown=True,  # Content is already converted to markdown
    )

    # Create extraction template with domain instructions
    prompt_template = create_extraction_template()

    # Get schema formatted for SimpleKGPipeline
    schema = get_schema_for_pipeline()

    # Create lexical graph configuration
    from neo4j_graphrag.experimental.components.lexical_graph import (
        LexicalGraphConfig,
    )

    lexical_config = LexicalGraphConfig(
        document_node_label=config.document_node_label,
        chunk_node_label=config.chunk_node_label,
        chunk_to_document_relationship_type=config.chunk_to_document_relationship,
        node_to_chunk_relationship_type=config.node_to_chunk_relationship,
    )

    # Create the pipeline with schema parameter
    pipeline = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=text_splitter,
        prompt_template=prompt_template,
        schema=schema,
        lexical_graph_config=lexical_config,
        perform_entity_resolution=config.perform_entity_resolution,
        neo4j_database=config.neo4j_database,
        from_pdf=False,  # We're passing text directly, not PDF files
        on_error="IGNORE",  # Continue processing even if individual chunks fail
    )

    logger.info("Pipeline created successfully")
    return pipeline


async def process_article_with_pipeline(
    pipeline: "SimpleKGPipeline",
    article_id: str,
    markdown_content: str,
    article_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Process a single article through the pipeline.

    Args:
        pipeline: Configured SimpleKGPipeline.
        article_id: Article identifier.
        markdown_content: Article content in markdown format.
        article_metadata: Additional metadata to attach.

    Returns:
        Processing result with statistics.
    """
    logger.info("Processing article", article_id=article_id)

    try:
        # Run the pipeline
        result = await pipeline.run_async(
            text=markdown_content,
            document_metadata={
                "article_id": article_id,
                **article_metadata,
            },
        )

        return {
            "article_id": article_id,
            "status": "success",
            "result": result,
        }

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.error(
            "Failed to process article",
            article_id=article_id,
            error=str(e),
            error_type=type(e).__name__,
            traceback=tb,
        )
        return {
            "article_id": article_id,
            "status": "error",
            "error": str(e),
            "traceback": tb,
        }


async def process_guide_with_pipeline(
    guide: "RequirementsManagementGuide",
    config: JamaKGPipelineConfig,
    output_dir: Path | None = None,  # noqa: ARG001 - reserved for future use
) -> dict[str, Any]:
    """Process all articles in the guide through the pipeline.

    Main entry point for processing the complete guide. Creates the
    pipeline and processes each article, tracking statistics.

    Args:
        guide: The scraped guide with all chapters and articles.
        config: Pipeline configuration.
        output_dir: Optional directory for intermediate outputs (reserved for future).

    Returns:
        Processing statistics and results summary.
    """
    logger.info(
        "Starting guide processing",
        total_articles=guide.total_articles,
        total_chapters=len(guide.chapters),
    )

    # Create pipeline
    pipeline = create_jama_kg_pipeline(config)

    # Track statistics
    stats = {
        "total_articles": guide.total_articles,
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "errors": [],
    }

    try:
        # Process each article
        for chapter in guide.chapters:
            for article in chapter.articles:
                # Build metadata (neo4j_graphrag requires string values)
                metadata = {
                    "chapter_number": str(chapter.chapter_number),
                    "chapter_title": chapter.title,
                    "article_number": str(article.article_number),
                    "article_title": article.title,
                    "url": article.url,
                    "content_type": article.content_type.value,
                }

                # Process article
                result = await process_article_with_pipeline(
                    pipeline=pipeline,
                    article_id=article.article_id,
                    markdown_content=article.markdown_content,
                    article_metadata=metadata,
                )

                stats["processed"] += 1

                if result["status"] == "success":
                    stats["succeeded"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(
                        {
                            "article_id": article.article_id,
                            "error": result.get("error", "Unknown error"),
                        }
                    )

                logger.info(
                    "Article processed",
                    article_id=article.article_id,
                    status=result["status"],
                    progress=f"{stats['processed']}/{stats['total_articles']}",
                )

        logger.info(
            "Guide processing complete",
            succeeded=stats["succeeded"],
            failed=stats["failed"],
        )

    finally:
        # Close pipeline resources
        if hasattr(pipeline, "close"):
            await pipeline.close()

    return stats
