"""Command-line interface for the Jama Guide Scraper."""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .chunker import ChunkingNotAvailableError
from .embedder import EmbeddingNotAvailableError
from .exceptions import BrowserNotInstalledError, PlaywrightNotAvailableError
from .extractor import LangExtractNotAvailableError
from .scraper import run_scraper

console = Console()


def main() -> None:
    """Run the Jama Guide Scraper CLI."""
    # Load .env file if present (for API keys like OPENAI_API_KEY)
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Scrape Jama Software's Requirements Management Guide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - outputs JSON and JSONL
  jama-scrape

  # Specify output directory
  jama-scrape --output ./data

  # Include all formats including Markdown
  jama-scrape --format json --format jsonl --format markdown

  # Include raw HTML for debugging
  jama-scrape --include-html

  # Use browser mode for JavaScript-rendered content (slower)
  jama-scrape --browser

  # Enable LLM enrichment with OpenAI (requires OPENAI_API_KEY)
  OPENAI_API_KEY=sk-... jama-scrape --enrich

  # Scrape + enrichment + Neo4j export
  OPENAI_API_KEY=sk-... jama-scrape --enrich --export-neo4j

  # Resume interrupted enrichment
  jama-scrape --enrich --resume

  # Enable chunking for GraphRAG
  jama-scrape --enrich --chunk

  # Full GraphRAG pipeline with embedding
  OPENAI_API_KEY=sk-... jama-scrape --enrich --chunk --embed --export-neo4j

  # Estimate embedding cost before running
  jama-scrape --enrich --chunk --embed --estimate-cost
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for scraped files (default: ./output)",
    )

    parser.add_argument(
        "-f",
        "--format",
        action="append",
        choices=["json", "jsonl", "markdown"],
        dest="formats",
        help="Output format(s). Can be specified multiple times. Default: json, jsonl",
    )

    parser.add_argument(
        "--include-html",
        action="store_true",
        help="Include raw HTML in output (increases file size significantly)",
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--browser",
        action="store_true",
        help=(
            "Use headless browser (Playwright) for JavaScript-rendered content. "
            "Slower but captures dynamic content like videos. "
            "Requires: uv sync --group browser && playwright install chromium"
        ),
    )

    # Enrichment options
    parser.add_argument(
        "--enrich",
        action="store_true",
        help=(
            "Enable LLM-powered semantic extraction using LangExtract. "
            "Extracts entities, relationships, and summaries for GraphRAG. "
            "Requires: uv sync --group enrichment and OPENAI_API_KEY env var"
        ),
    )

    parser.add_argument(
        "--export-neo4j",
        action="store_true",
        help="Generate Neo4j import files (Cypher scripts + CSV bulk import)",
    )

    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini", "ollama"],
        default="openai",
        help="LLM provider for enrichment (default: openai)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume enrichment from checkpoint (skips already processed articles)",
    )

    # Chunking options
    parser.add_argument(
        "--chunk",
        action="store_true",
        help=(
            "Chunk articles for GraphRAG retrieval (3-tier: summary, section, "
            "sliding window). Outputs chunks.jsonl. Best used with --enrich "
            "for entity linkage."
        ),
    )

    # Embedding options
    parser.add_argument(
        "--embed",
        action="store_true",
        help=(
            "Generate embeddings for chunks (requires --chunk). "
            "Outputs embeddings.jsonl. Requires: uv sync --group embedding"
        ),
    )

    parser.add_argument(
        "--embedding-provider",
        choices=["openai"],
        default="openai",
        help="Embedding provider (default: openai). Future: ollama, voyage, cohere",
    )

    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help=(
            "Estimate embedding cost without running "
            "(dry run, requires --chunk --embed)"
        ),
    )

    args = parser.parse_args()

    # Validate flag dependencies
    if args.embed and not args.chunk:
        parser.error("--embed requires --chunk")

    if args.estimate_cost and not (args.chunk and args.embed):
        parser.error("--estimate-cost requires --chunk --embed")

    # Default formats if none specified
    formats = args.formats or ["json", "jsonl"]

    console.print("[bold]Jama Requirements Management Guide Scraper[/]")
    console.print(f"Output directory: {args.output}")
    console.print(f"Formats: {', '.join(formats)}")
    if args.enrich:
        console.print(f"[cyan]Enrichment: enabled ({args.llm_provider})[/]")
    if args.chunk:
        console.print("[cyan]Chunking: enabled (3-tier)[/]")
        if not args.enrich:
            console.print(
                "[yellow]Warning: Running --chunk without --enrich. "
                "Entity linkage disabled.[/]"
            )
            console.print(
                "[yellow]For full GraphRAG capability: "
                "jama-scrape --enrich --chunk[/]"
            )
    if args.embed:
        console.print(f"[cyan]Embedding: enabled ({args.embedding_provider})[/]")
    if args.export_neo4j:
        console.print("[cyan]Neo4j export: enabled[/]")

    try:
        asyncio.run(
            run_scraper(
                output_dir=args.output,
                include_raw_html=args.include_html,
                formats=formats,
                use_browser=args.browser,
                enrich=args.enrich,
                export_neo4j=args.export_neo4j,
                llm_provider=args.llm_provider,
                resume_enrichment=args.resume,
                chunk=args.chunk,
                embed=args.embed,
                embedding_provider=args.embedding_provider,
                estimate_cost=args.estimate_cost,
            )
        )
    except LangExtractNotAvailableError:
        console.print("\n[red]Error: LangExtract not installed[/]")
        console.print("Install with: [cyan]uv sync --group enrichment[/]")
        raise SystemExit(1) from None
    except ChunkingNotAvailableError:
        console.print("\n[red]Error: Chunking dependencies not installed[/]")
        console.print("Install with: [cyan]uv sync --group embedding[/]")
        raise SystemExit(1) from None
    except EmbeddingNotAvailableError:
        console.print("\n[red]Error: Embedding dependencies not installed[/]")
        console.print("Install with: [cyan]uv sync --group embedding[/]")
        raise SystemExit(1) from None
    except PlaywrightNotAvailableError:
        console.print("\n[red]Error: Playwright not installed[/]")
        console.print("Install with: [cyan]uv sync --group browser[/]")
        raise SystemExit(1) from None
    except BrowserNotInstalledError:
        console.print("\n[red]Error: Browser binaries not installed[/]")
        console.print("Install with: [cyan]playwright install chromium[/]")
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Scraping interrupted by user[/]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
