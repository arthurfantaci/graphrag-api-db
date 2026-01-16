"""Async web scraper for the Jama Requirements Management Guide.

Features:
- Pluggable fetcher abstraction (httpx or Playwright)
- Rate limiting to be respectful to the server
- Retry logic with exponential backoff
- Progress tracking with Rich
"""

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .config import (
    CHAPTERS,
    GLOSSARY_URL,
    MAX_CONCURRENT_REQUESTS,
    RATE_LIMIT_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    ArticleConfig,
    ChapterConfig,
)
from .fetcher import Fetcher, FetcherConfig, create_fetcher
from .models_core import (
    Article,
    Chapter,
    ContentType,
    Glossary,
    GlossaryTerm,
    GuideMetadata,
    RequirementsManagementGuide,
)
from .parser import HTMLParser

if TYPE_CHECKING:
    from rich.progress import TaskID

    from .chunk_models import ChunkedGuide, EmbeddedGuideChunks
    from .graph_models import EnrichedGuide

console = Console()


class JamaGuideScraper:
    """Scraper for the Jama Requirements Management Guide.

    Supports two fetching modes:
    - Default (httpx): Fast, lightweight, for static HTML content
    - Browser (Playwright): Slower but renders JavaScript for dynamic content

    Usage:
        scraper = JamaGuideScraper()
        guide = await scraper.scrape_all()
        scraper.save_json(guide, Path("output/guide.json"))
        scraper.save_jsonl(guide, Path("output/guide.jsonl"))

        # For JavaScript-rendered content (e.g., YouTube embeds):
        scraper = JamaGuideScraper(use_browser=True)
        guide = await scraper.scrape_all()
    """

    def __init__(
        self,
        rate_limit_delay: float = RATE_LIMIT_DELAY_SECONDS,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        include_raw_html: bool = False,
        use_browser: bool = False,
    ) -> None:
        """Initialize the scraper with rate limiting and concurrency settings.

        Args:
            rate_limit_delay: Minimum seconds between requests.
            max_concurrent: Maximum parallel requests.
            timeout: Request timeout in seconds.
            include_raw_html: Whether to include raw HTML in output.
            use_browser: If True, use Playwright for JS rendering.
        """
        self._config = FetcherConfig(
            rate_limit_delay=rate_limit_delay,
            max_concurrent=max_concurrent,
            timeout=timeout,
        )
        self._use_browser = use_browser
        self.include_raw_html = include_raw_html
        self.parser = HTMLParser()

    async def scrape_all(self) -> RequirementsManagementGuide:
        """Scrape the entire guide including all chapters and glossary."""
        console.print(
            "[bold blue]Starting Jama Requirements Management Guide Scraper[/]"
        )
        mode = "browser (Playwright)" if self._use_browser else "httpx"
        console.print(
            f"Rate limit: {self._config.rate_limit_delay}s delay, "
            f"{self._config.max_concurrent} concurrent requests"
        )
        console.print(f"Fetcher mode: {mode}")

        async with create_fetcher(self._use_browser, self._config) as fetcher:
            # First, discover any missing articles from chapter overviews
            chapters_config = await self._discover_all_articles(fetcher)

            # Scrape all chapters
            chapters = await self._scrape_all_chapters(fetcher, chapters_config)

            # Scrape glossary
            glossary = await self._scrape_glossary(fetcher)

            guide = RequirementsManagementGuide(
                metadata=GuideMetadata(scraped_at=datetime.now(UTC)),
                chapters=chapters,
                glossary=glossary,
            )

            console.print("\n[bold green]✓ Scraping complete![/]")
            console.print(f"  Chapters: {len(guide.chapters)}")
            console.print(f"  Total articles: {guide.total_articles}")
            console.print(f"  Total words: {guide.total_word_count:,}")
            if guide.glossary:
                console.print(f"  Glossary terms: {guide.glossary.term_count}")

            return guide

    async def _discover_all_articles(self, fetcher: Fetcher) -> list[ChapterConfig]:
        """Discover articles by scraping chapter overview pages.

        Args:
            fetcher: The fetcher to use for HTTP requests.

        Returns:
            List of chapter configurations with discovered articles.
        """
        console.print("\n[yellow]Discovering articles from chapter overviews...[/]")

        updated_chapters = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Discovering...", total=len(CHAPTERS))

            for chapter_config in CHAPTERS:
                # If chapter only has overview, try to discover more
                current_chapter = chapter_config
                if len(chapter_config.articles) <= 1:
                    html = await fetcher.fetch(chapter_config.overview_url)
                    if html:
                        discovered = self.parser.discover_articles(
                            html, chapter_config.slug
                        )
                        if discovered:
                            console.print(
                                f"  Found {len(discovered)} articles "
                                f"in Chapter {chapter_config.number}"
                            )
                            # Create new chapter config with discovered articles
                            new_articles = [ArticleConfig(0, "Overview", "")]
                            for i, art in enumerate(discovered, 1):
                                new_articles.append(
                                    ArticleConfig(i, art["title"], art["slug"])
                                )
                            current_chapter = replace(
                                chapter_config, articles=new_articles
                            )

                updated_chapters.append(current_chapter)
                progress.advance(task)

        return updated_chapters

    async def _scrape_all_chapters(
        self,
        fetcher: Fetcher,
        chapters_config: list[ChapterConfig],
    ) -> list[Chapter]:
        """Scrape all chapters.

        Args:
            fetcher: The fetcher to use for HTTP requests.
            chapters_config: List of chapter configurations.

        Returns:
            List of scraped Chapter objects.
        """
        console.print("\n[yellow]Scraping chapters...[/]")

        chapters = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            total_articles = sum(len(ch.articles) for ch in chapters_config)
            task = progress.add_task("Scraping articles...", total=total_articles)

            for chapter_config in chapters_config:
                chapter = await self._scrape_chapter(
                    fetcher, chapter_config, progress, task
                )
                chapters.append(chapter)

        return chapters

    async def _scrape_chapter(
        self,
        fetcher: Fetcher,
        config: ChapterConfig,
        progress: Progress,
        task: "TaskID",
    ) -> Chapter:
        """Scrape a single chapter.

        Args:
            fetcher: The fetcher to use for HTTP requests.
            config: Chapter configuration.
            progress: Rich progress bar.
            task: Progress task ID.

        Returns:
            Scraped Chapter object.
        """
        articles = []

        for article_config in config.articles:
            url = config.get_article_url(article_config)

            html = await fetcher.fetch(url)

            if html:
                parsed = self.parser.parse_article(html, url)

                content_type = (
                    ContentType.CHAPTER_OVERVIEW
                    if article_config.number == 0
                    else ContentType.ARTICLE
                )

                article = Article(
                    article_id=f"ch{config.number}-art{article_config.number}",
                    chapter_number=config.number,
                    article_number=article_config.number,
                    title=parsed["title"] or article_config.title,
                    url=url,
                    content_type=content_type,
                    raw_html=html if self.include_raw_html else None,
                    markdown_content=parsed["markdown_content"],
                    sections=parsed["sections"],
                    key_concepts=parsed["key_concepts"],
                    cross_references=parsed["cross_references"],
                    images=parsed["images"],
                    videos=parsed["videos"],
                    webinars=parsed["webinars"],
                    related_articles=parsed["related_articles"],
                )
                articles.append(article)
            else:
                console.print(f"[red]Failed to fetch: {url}[/]")

            progress.advance(task)

        return Chapter(
            chapter_number=config.number,
            title=config.title,
            overview_url=config.overview_url,
            articles=articles,
        )

    async def _scrape_glossary(self, fetcher: Fetcher) -> Glossary | None:
        """Scrape the glossary page.

        Args:
            fetcher: The fetcher to use for HTTP requests.

        Returns:
            Glossary object or None if fetch failed.
        """
        console.print("\n[yellow]Scraping glossary...[/]")

        html = await fetcher.fetch(GLOSSARY_URL)

        if not html:
            console.print("[red]Failed to fetch glossary[/]")
            return None

        terms_data = self.parser.parse_glossary(html, GLOSSARY_URL)

        terms = [
            GlossaryTerm(
                term=t["term"],
                acronym=t.get("acronym"),
                definition=t["definition"],
            )
            for t in terms_data
        ]

        console.print(f"  Found {len(terms)} glossary terms")

        return Glossary(
            url=GLOSSARY_URL,
            terms=terms,
        )

    def save_json(self, guide: RequirementsManagementGuide, path: Path) -> None:
        """Save the guide as a single JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                guide.model_dump(
                    exclude={"raw_html"} if not self.include_raw_html else None
                ),
                f,
                indent=2,
                default=str,  # Handle datetime
            )

        console.print(f"[green]Saved JSON to: {path}[/]")

    def save_jsonl(self, guide: RequirementsManagementGuide, path: Path) -> None:
        """Save the guide as JSONL (one record per article/term)."""
        path.parent.mkdir(parents=True, exist_ok=True)

        records = guide.to_jsonl_articles()

        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f, default=str)
                f.write("\n")

        console.print(f"[green]Saved JSONL to: {path} ({len(records)} records)[/]")

    def save_markdown(self, guide: RequirementsManagementGuide, path: Path) -> None:
        """Save the guide as a single consolidated Markdown file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# {guide.metadata.title}",
            "",
            f"*Published by {guide.metadata.publisher}*",
            f"*Scraped on {guide.metadata.scraped_at.strftime('%Y-%m-%d')}*",
            "",
            "---",
            "",
            "## Table of Contents",
            "",
        ]

        # TOC
        for chapter in guide.chapters:
            lines.append(f"- **Chapter {chapter.chapter_number}**: {chapter.title}")
            for article in chapter.articles:
                if article.article_number > 0:
                    lines.append(f"  - {article.title}")

        lines.extend(["", "---", ""])

        # Content
        for chapter in guide.chapters:
            lines.append(f"# Chapter {chapter.chapter_number}: {chapter.title}")
            lines.append("")

            for article in chapter.articles:
                if article.article_number == 0:
                    lines.append("## Overview")
                else:
                    lines.append(f"## {article.article_number}. {article.title}")

                lines.append("")
                lines.append(f"*Source: {article.url}*")
                lines.append("")
                lines.append(article.markdown_content)

                # Add webinars section if present
                if article.webinars:
                    lines.append("")
                    lines.append("### Webinars")
                    lines.append("")
                    for webinar in article.webinars:
                        lines.append(f"- **[{webinar.title}]({webinar.url})**")
                        if webinar.description:
                            lines.append(f"  - {webinar.description}")

                # Add related articles section if present
                if article.related_articles:
                    lines.append("")
                    lines.append("### Related Articles")
                    lines.append("")
                    for related in article.related_articles:
                        lines.append(f"- [{related.title}]({related.url})")

                # Add videos section if present
                if article.videos:
                    lines.append("")
                    lines.append("### Videos")
                    lines.append("")
                    for video in article.videos:
                        title = video.title or f"Video ({video.video_id})"
                        lines.append(f"- [{title}]({video.url})")

                lines.append("")
                lines.append("---")
                lines.append("")

        # Glossary
        if guide.glossary:
            lines.append("# Glossary")
            lines.append("")
            for term in guide.glossary.terms:
                lines.append(f"**{term.term}**: {term.definition}")
                lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        console.print(f"[green]Saved Markdown to: {path}[/]")


async def run_scraper(
    output_dir: Path = Path("output"),
    use_browser: bool = False,
    scrape_only: bool = False,
    skip_resources: bool = False,
    skip_supplementary: bool = False,
    run_validation: bool = False,
) -> RequirementsManagementGuide:
    """Run the complete neo4j_graphrag pipeline.

    This function executes the full pipeline:
    1. Scrape all articles and glossary
    2. Process through SimpleKGPipeline (chunking, extraction, embeddings)
    3. Apply entity normalization and industry consolidation
    4. Create supplementary structure (chapters, resources, glossary)
    5. Run validation checks (optional)

    Args:
        output_dir: Directory for output files.
        use_browser: If True, use Playwright for JS rendering.
        scrape_only: If True, only scrape articles (no Neo4j processing).
        skip_resources: If True, skip resource node creation.
        skip_supplementary: If True, skip all supplementary graph structure.
        run_validation: If True, run validation and generate report.

    Returns:
        The scraped guide data.
    """
    scraper = JamaGuideScraper(
        include_raw_html=False,
        use_browser=use_browser,
    )

    # Stage 1: Scrape
    guide = await scraper.scrape_all()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save base outputs (JSON for reference)
    scraper.save_json(guide, output_dir / "requirements_management_guide.json")
    scraper.save_jsonl(guide, output_dir / "requirements_management_guide.jsonl")

    if scrape_only:
        console.print("\n[yellow]Scrape-only mode: Skipping Neo4j pipeline[/]")
        return guide

    # Stage 2: Process through neo4j_graphrag pipeline
    pipeline_stats = await _run_neo4j_graphrag_pipeline(guide, output_dir)

    # Stage 3: Post-processing (entity normalization, industry consolidation)
    await _run_post_processing(output_dir)

    # Stage 4: Supplementary graph structure
    if not skip_supplementary:
        await _build_supplementary_structure(
            guide, output_dir, skip_resources=skip_resources
        )

    # Stage 5: Validation (optional)
    if run_validation:
        await _run_validation(output_dir)

    console.print("\n[bold green]✓ Pipeline complete![/]")
    console.print(f"  Articles processed: {pipeline_stats.get('processed', 0)}")
    console.print(f"  Succeeded: {pipeline_stats.get('succeeded', 0)}")
    if pipeline_stats.get('failed', 0) > 0:
        console.print(f"  [red]Failed: {pipeline_stats.get('failed', 0)}[/]")

    return guide


async def _run_neo4j_graphrag_pipeline(
    guide: RequirementsManagementGuide,
    output_dir: Path,
) -> dict:
    """Run the neo4j_graphrag SimpleKGPipeline.

    Args:
        guide: Scraped guide to process.
        output_dir: Directory for output files.

    Returns:
        Processing statistics.
    """
    from .extraction.pipeline import (
        JamaKGPipelineConfig,
        process_guide_with_pipeline,
    )

    console.print("\n[bold cyan]Starting neo4j_graphrag pipeline...[/]")

    # Load configuration from environment
    config = JamaKGPipelineConfig.from_env()

    console.print(f"  LLM model: {config.llm_model}")
    console.print(f"  Embedding model: {config.embedding_model}")
    console.print(f"  Neo4j: {config.neo4j_uri}")

    # Process guide
    stats = await process_guide_with_pipeline(guide, config, output_dir)

    console.print(f"\n[green]Pipeline processed {stats['processed']} articles[/]")

    return stats


async def _run_post_processing(_output_dir: Path) -> None:
    """Run post-processing: entity normalization and industry consolidation.

    Args:
        _output_dir: Directory for output files (reserved for future use).
    """
    from .extraction.pipeline import JamaKGPipelineConfig, create_neo4j_driver
    from .postprocessing.industry_taxonomy import IndustryNormalizer
    from .postprocessing.normalizer import EntityNormalizer

    console.print("\n[bold cyan]Running post-processing...[/]")

    config = JamaKGPipelineConfig.from_env()
    driver = create_neo4j_driver(config)

    try:
        # Entity normalization
        console.print("  Normalizing entity names...")
        normalizer = EntityNormalizer(driver, config.neo4j_database)
        norm_stats = await normalizer.normalize_all_entities()
        console.print(f"    Updated {norm_stats['updated']} entity names")

        # Entity deduplication
        console.print("  Deduplicating entities...")
        dedup_stats = await normalizer.deduplicate_by_name()
        console.print(f"    Merged {dedup_stats['merged']} duplicates")

        # Industry consolidation
        console.print("  Consolidating industries...")
        industry_normalizer = IndustryNormalizer(driver, config.neo4j_database)
        industry_stats = await industry_normalizer.consolidate_industries()
        console.print(
            f"    Consolidated {industry_stats['original_count']} → "
            f"{industry_stats['canonical_count']} industries"
        )

    finally:
        driver.close()


async def _build_supplementary_structure(
    guide: RequirementsManagementGuide,
    _output_dir: Path,
    skip_resources: bool = False,
) -> None:
    """Build supplementary graph structure.

    Args:
        guide: Scraped guide.
        _output_dir: Directory for output files (reserved for future use).
        skip_resources: If True, skip resource nodes.
    """
    from .extraction.pipeline import JamaKGPipelineConfig, create_neo4j_driver
    from .graph.constraints import create_all_constraints, create_vector_index
    from .graph.supplementary import SupplementaryGraphBuilder

    console.print("\n[bold cyan]Building supplementary graph structure...[/]")

    config = JamaKGPipelineConfig.from_env()
    driver = create_neo4j_driver(config)

    try:
        # Create constraints and indexes
        console.print("  Creating constraints and indexes...")
        await create_all_constraints(driver, config.neo4j_database)
        await create_vector_index(
            driver,
            config.neo4j_database,
            dimensions=config.embedding_dimensions,
        )

        # Build supplementary structure
        builder = SupplementaryGraphBuilder(driver, config.neo4j_database)

        if skip_resources:
            # Only create chapters and article relationships
            from .graph.supplementary import (
                create_article_relationships,
                create_chapter_structure,
            )

            console.print("  Creating chapter structure...")
            await create_chapter_structure(driver, guide, config.neo4j_database)

            console.print("  Creating article relationships...")
            await create_article_relationships(driver, guide, config.neo4j_database)
        else:
            console.print("  Creating all supplementary nodes...")
            stats = await builder.build_all(guide)
            console.print(f"    Chapters: {stats['chapters']}")
            console.print(f"    Images: {stats['images']}")
            console.print(f"    Videos: {stats['videos']}")
            console.print(f"    Webinars: {stats['webinars']}")
            console.print(f"    Definitions: {stats['definitions']}")

    finally:
        driver.close()


async def _run_validation(output_dir: Path) -> None:
    """Run validation and generate report.

    Args:
        output_dir: Directory for output files.
    """
    from .extraction.pipeline import JamaKGPipelineConfig, create_neo4j_driver
    from .validation.reporter import generate_validation_report

    console.print("\n[bold cyan]Running validation...[/]")

    config = JamaKGPipelineConfig.from_env()
    driver = create_neo4j_driver(config)

    try:
        report = await generate_validation_report(
            driver,
            config.neo4j_database,
            output_path=output_dir / "validation_report.md",
        )

        if report.validation_passed:
            console.print("[green]✓ Validation passed[/]")
        else:
            console.print("[yellow]⚠ Validation found issues[/]")
            for rec in report.recommendations[:3]:
                console.print(f"  - {rec}")

        console.print(f"\n  Full report: {output_dir / 'validation_report.md'}")

    finally:
        driver.close()


async def _run_enrichment(
    guide: RequirementsManagementGuide,
    output_dir: Path,
    llm_provider: str,
    resume: bool,
) -> "EnrichedGuide | None":
    """Run LangExtract semantic enrichment on the guide.

    Args:
        guide: Scraped guide to enrich.
        output_dir: Directory for output files.
        llm_provider: LLM provider name.
        resume: Whether to resume from checkpoint.

    Returns:
        EnrichedGuide or None if enrichment failed.
    """
    from .enrichment_config import EnrichmentConfig, LLMProvider
    from .extractor import JamaExtractor, check_langextract_available

    if not check_langextract_available():
        from .extractor import LangExtractNotAvailableError

        raise LangExtractNotAvailableError

    console.print("\n[bold cyan]Starting LLM enrichment...[/]")

    # Create configuration
    provider = LLMProvider(llm_provider.lower())
    config = EnrichmentConfig(
        provider=provider,
        checkpoint_dir=output_dir / ".enrichment_cache",
    )

    # Run extraction
    extractor = JamaExtractor(config)
    enriched_guide = await extractor.enrich_guide(guide, resume=resume)

    # Save enriched guide
    enriched_path = output_dir / "enriched_guide.json"
    enriched_path.write_text(
        enriched_guide.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]Saved enriched guide to: {enriched_path}[/]")

    return enriched_guide


def _run_chunking(
    guide: RequirementsManagementGuide,
    enriched_guide: "EnrichedGuide | None",
    output_dir: Path,
) -> "ChunkedGuide":
    """Chunk articles for GraphRAG retrieval.

    Args:
        guide: Scraped guide with articles.
        enriched_guide: Optional enriched guide for entity linkage.
        output_dir: Directory for output files.

    Returns:
        ChunkedGuide with all chunks.
    """
    from .chunk_export import ChunkExporter
    from .chunker import JamaChunker, check_chunking_available
    from .chunking_config import ChunkingConfig

    if not check_chunking_available():
        from .chunker import ChunkingNotAvailableError

        raise ChunkingNotAvailableError

    # Create chunker with default config
    config = ChunkingConfig(
        checkpoint_dir=output_dir / ".chunking_cache",
    )
    chunker = JamaChunker(config)

    # Chunk the guide
    chunked_guide = chunker.chunk_guide(guide, enriched_guide)

    # Export chunks
    exporter = ChunkExporter(output_dir)
    exporter.export_chunks_jsonl(chunked_guide)

    return chunked_guide


async def _run_embedding(
    chunked_guide: "ChunkedGuide",
    output_dir: Path,
    embedding_provider: str,
    estimate_cost: bool,
) -> "EmbeddedGuideChunks | None":
    """Generate embeddings for chunks.

    Args:
        chunked_guide: ChunkedGuide with chunks to embed.
        output_dir: Directory for output files.
        embedding_provider: Embedding provider name.
        estimate_cost: If True, only estimate cost and exit.

    Returns:
        EmbeddedGuideChunks or None if cost estimation only.
    """
    from .chunk_export import ChunkExporter
    from .embedder import JamaEmbedder, check_embedding_available
    from .embedding_config import EmbeddingConfig, EmbeddingProvider

    if not check_embedding_available():
        from .embedder import EmbeddingNotAvailableError

        raise EmbeddingNotAvailableError

    # Create configuration
    provider = EmbeddingProvider(embedding_provider.lower())
    config = EmbeddingConfig(
        provider=provider,
        checkpoint_dir=output_dir / ".embedding_cache",
    )

    embedder = JamaEmbedder(config)

    # Estimate cost if requested
    cost_info = embedder.estimate_cost(chunked_guide)
    console.print("\n[cyan]Embedding cost estimate:[/]")
    console.print(f"  Chunks: {cost_info['total_chunks']}")
    console.print(f"  Tokens: {cost_info['total_tokens']:,}")
    console.print(f"  Model: {cost_info['model']}")
    console.print(f"  Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")

    if estimate_cost:
        console.print("\n[yellow]Dry run: --estimate-cost specified, not embedding[/]")
        return None

    # Generate embeddings
    embedded_guide = await embedder.embed_chunks(chunked_guide, resume=True)

    # Export embeddings
    exporter = ChunkExporter(output_dir)
    exporter.export_embeddings_jsonl(embedded_guide)

    return embedded_guide


def _export_to_neo4j(
    guide: RequirementsManagementGuide,
    enriched_guide: "EnrichedGuide",
    output_dir: Path,
    chunked_guide: "ChunkedGuide | None" = None,
) -> None:
    """Export guide and enrichment to Neo4j formats.

    Args:
        guide: Original scraped guide.
        enriched_guide: Enriched guide with entities/relationships.
        output_dir: Directory for output files.
        chunked_guide: Optional chunked guide for chunk nodes.
    """
    from .graph_export import Neo4jExporter

    console.print("\n[bold cyan]Exporting to Neo4j formats...[/]")

    exporter = Neo4jExporter(output_dir)
    exporter.export_all(guide, enriched_guide)

    # Export chunk nodes if available
    if chunked_guide:
        from .chunk_export import ChunkExporter

        chunk_exporter = ChunkExporter(output_dir)
        chunk_exporter.export_neo4j(chunked_guide)


def _load_to_neo4j(output_dir: Path) -> None:
    """Load exported data into Neo4j database.

    Args:
        output_dir: Directory containing exported Neo4j files.

    Raises:
        Neo4jConfigError: If required environment variables are not set.
    """
    from .exceptions import Neo4jConfigError
    from .neo4j_loader import Neo4jLoader, get_neo4j_config

    console.print("\n[bold cyan]Loading data into Neo4j...[/]")

    # Get configuration from environment
    config = get_neo4j_config()
    if config is None:
        raise Neo4jConfigError

    uri, username, password = config

    # Load data
    loader = Neo4jLoader(uri=uri, username=username, password=password)
    try:
        loader.load_all(output_dir)
    finally:
        loader.close()
