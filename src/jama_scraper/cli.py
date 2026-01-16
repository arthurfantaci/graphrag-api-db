"""Command-line interface for the Jama Guide Scraper.

This CLI runs the complete Neo4j GraphRAG pipeline:
1. Scrape all articles and glossary
2. Process through neo4j_graphrag SimpleKGPipeline
3. Apply post-processing (entity normalization, industry consolidation)
4. Create supplementary graph structure (chapters, resources)
5. Run validation checks
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .exceptions import (
    BrowserNotInstalledError,
    Neo4jConfigError,
    PlaywrightNotAvailableError,
)
from .scraper import run_scraper

console = Console()


def main() -> None:
    """Run the Jama Guide Scraper CLI.

    Executes the complete neo4j_graphrag pipeline with entity extraction,
    chunking, embeddings, and knowledge graph construction.
    """
    # Load .env file for API keys
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Scrape Jama's Requirements Guide into Neo4j graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool runs the complete neo4j_graphrag pipeline:

  1. Scrape all 103 articles + glossary from Jama's guide
  2. Process through SimpleKGPipeline (chunking, extraction, embeddings)
  3. Apply entity normalization and industry consolidation
  4. Create supplementary structure (chapters, resources, glossary)
  5. Run validation checks

Required environment variables:
  OPENAI_API_KEY     - For entity extraction + embeddings

Required environment variables (for Neo4j):
  NEO4J_URI          - Database URI (e.g., bolt://localhost:7687)
  NEO4J_USERNAME     - Database username (default: neo4j)
  NEO4J_PASSWORD     - Database password

Examples:
  # Run full pipeline
  jama-scrape

  # Specify output directory for intermediate files
  jama-scrape -o ./output

  # Run with validation report at the end
  jama-scrape --validate

  # Skip resource node creation (faster)
  jama-scrape --skip-resources

  # Use browser for JavaScript-rendered content
  jama-scrape --browser
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for intermediate files (default: ./output)",
    )

    parser.add_argument(
        "--browser",
        action="store_true",
        help=(
            "Use headless browser for JavaScript-rendered content. "
            "Requires: uv sync --group browser && playwright install chromium"
        ),
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation queries and generate report after loading",
    )

    parser.add_argument(
        "--skip-resources",
        action="store_true",
        help="Skip creation of resource nodes (Image, Video, Webinar)",
    )

    parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Skip supplementary graph structure (chapters, resources)",
    )

    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape articles, skip neo4j_graphrag pipeline",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs and show what would be processed without running",
    )

    args = parser.parse_args()

    console.print("[bold cyan]Jama Guide → Neo4j GraphRAG Pipeline[/]")
    console.print(f"Output directory: {args.output}")
    console.print()

    if args.scrape_only:
        console.print("Mode: [yellow]Scrape only[/] (no Neo4j processing)")
        console.print()
        console.print("Pipeline stages:")
        console.print("  1. Scrape articles + glossary")
        console.print("  2. Save JSON/JSONL output")
    else:
        console.print("Pipeline stages:")
        console.print("  1. Scrape articles + glossary")
        console.print("  2. Process through SimpleKGPipeline")
        console.print("  3. Entity normalization & deduplication")
        if not args.skip_supplementary:
            console.print("  4. Create supplementary graph structure")
            if args.skip_resources:
                console.print("     (resources skipped)")
        if args.validate:
            console.print("  5. Run validation & generate report")
    console.print()

    if args.dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/]")
        console.print()
        # Show estimated processing
        console.print("Estimated processing:")
        console.print("  - ~103 articles")
        console.print("  - ~15 chapters")
        console.print("  - ~100 glossary terms")
        console.print("  - Embedding cost: ~$0.50-1.00 (text-embedding-3-small)")
        console.print("  - LLM extraction cost: ~$5-10 (gpt-4o)")
        return

    try:
        asyncio.run(
            run_scraper(
                output_dir=args.output,
                use_browser=args.browser,
                scrape_only=args.scrape_only,
                skip_resources=args.skip_resources,
                skip_supplementary=args.skip_supplementary,
                run_validation=args.validate,
            )
        )
    except PlaywrightNotAvailableError:
        console.print("\n[red]Error: Playwright not installed[/]")
        console.print("Install with: [cyan]uv sync --group browser[/]")
        raise SystemExit(1) from None
    except BrowserNotInstalledError:
        console.print("\n[red]Error: Browser binaries not installed[/]")
        console.print("Install with: [cyan]playwright install chromium[/]")
        raise SystemExit(1) from None
    except Neo4jConfigError:
        console.print("\n[red]Error: Neo4j configuration missing[/]")
        console.print("Set: [cyan]NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD[/]")
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
