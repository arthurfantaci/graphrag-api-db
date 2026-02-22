"""Command-line interface for the GraphRAG Knowledge Graph Pipeline.

This CLI provides two main commands:

1. `graphrag-kg` (default): Run the complete Neo4j GraphRAG pipeline
   - Scrape all articles and glossary from the guide
   - Process through neo4j_graphrag SimpleKGPipeline
   - Apply post-processing (entity normalization, industry consolidation)
   - Create supplementary graph structure (chapters, resources)
   - Optionally run validation checks

2. `graphrag-kg validate`: Validate and optionally fix data quality issues
   - Check for missing chunk_ids
   - Find plural/singular entity duplicates
   - Identify generic entities that should be removed
   - Generate validation reports
   - Apply fixes with --fix flag
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
from .preflight import PreflightError
from .scraper import run_scraper

console = Console()


def _create_scrape_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the scrape subcommand parser.

    Args:
        subparsers: Subparsers action to add the command to.
    """
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Run the complete scraping and graph building pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Run the complete neo4j_graphrag pipeline:

  1. Scrape all 103 articles + glossary from the guide
  2. Process through SimpleKGPipeline (chunking, extraction, embeddings)
  3. Apply entity normalization and industry consolidation
  4. Create supplementary structure (chapters, resources, glossary)
  5. Optionally run validation checks
        """,
    )

    scrape_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for intermediate files (default: ./output)",
    )

    scrape_parser.add_argument(
        "--browser",
        action="store_true",
        help=(
            "Use headless browser for JavaScript-rendered content. "
            "Requires: uv sync --group browser && playwright install chromium"
        ),
    )

    scrape_parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation queries and generate report after loading",
    )

    scrape_parser.add_argument(
        "--skip-resources",
        action="store_true",
        help="Skip creation of resource nodes (Image, Video, Webinar)",
    )

    scrape_parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Skip supplementary graph structure (chapters, resources)",
    )

    scrape_parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape articles, skip neo4j_graphrag pipeline",
    )

    scrape_parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run all pipeline stages in a single invocation: scrape → extract → "
            "chunk repair → entity creation → cleanup → enrichment → graph analysis → "
            "supplementary → validate + fix → report"
        ),
    )

    scrape_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs and show what would be processed without running",
    )


def _create_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the validate subcommand parser.

    Args:
        subparsers: Subparsers action to add the command to.
    """
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate and optionally fix knowledge graph data quality issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Validate the knowledge graph and optionally apply fixes.

Checks performed:
  - Orphan chunks (not connected to articles)
  - Missing chunk_id properties
  - Plural/singular entity duplicates (requirement vs requirements)
  - Overly generic entity names (tool, software, process)
  - Industry count (target: ≤19 canonical industries)
  - Invalid relationship patterns

Examples:
  # Run validation and show report
  graphrag-kg validate

  # Save report to file
  graphrag-kg validate -o validation_report.md

  # Preview what fixes would do
  graphrag-kg validate --fix --dry-run

  # Apply fixes
  graphrag-kg validate --fix

  # Only fix chunk_ids (safe operation)
  graphrag-kg validate --fix-chunk-ids

  # Only fix entity quality (merges and deletes)
  graphrag-kg validate --fix-entities
        """,
    )

    validate_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Save validation report to this file (markdown format)",
    )

    validate_parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply all available fixes (chunk_ids + entity cleanup)",
    )

    validate_parser.add_argument(
        "--fix-chunk-ids",
        action="store_true",
        help="Only fix missing chunk_id properties (safe, additive operation)",
    )

    validate_parser.add_argument(
        "--fix-entities",
        action="store_true",
        help="Only fix entity quality (delete generic + merge plurals)",
    )

    validate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what fixes would do without applying them",
    )


async def _run_validate_command(args: argparse.Namespace) -> None:
    """Run the validate subcommand.

    Args:
        args: Parsed command-line arguments.
    """
    import os

    from neo4j import AsyncGraphDatabase

    from .validation.fixes import ValidationFixer, format_fix_preview
    from .validation.reporter import generate_validation_report

    # Get Neo4j connection from environment
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not neo4j_password:
        console.print("[red]Error: NEO4J_PASSWORD environment variable required[/]")
        raise SystemExit(1)

    console.print("[bold cyan]Knowledge Graph Validation[/]")
    console.print(f"Database: {neo4j_uri}")
    console.print()

    driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_username, neo4j_password),
    )

    try:
        # Determine what to do
        apply_fixes = args.fix or args.fix_chunk_ids or args.fix_entities

        if apply_fixes:
            fixer = ValidationFixer(driver, neo4j_database)

            if args.dry_run:
                console.print("[yellow]Dry run mode - showing what would be fixed[/]")
                console.print()
                preview = await fixer.preview_all_fixes()
                console.print(format_fix_preview(preview))
            else:
                # Show preview first
                console.print("Analyzing data quality issues...")
                preview = await fixer.preview_all_fixes()

                if preview["summary"]["total_changes"] == 0:
                    console.print("[green]No fixes needed - data looks clean![/]")
                else:
                    console.print(format_fix_preview(preview))
                    console.print()

                    # Apply requested fixes
                    if args.fix:
                        console.print("[bold]Applying all fixes...[/]")
                        results = await fixer.apply_all_fixes()
                    elif args.fix_chunk_ids:
                        console.print("[bold]Fixing chunk_ids...[/]")
                        results = {"chunk_ids": await fixer.apply_chunk_id_fix_only()}
                    elif args.fix_entities:
                        console.print("[bold]Fixing entity quality...[/]")
                        results = await fixer.apply_entity_cleanup_only()

                    console.print()
                    console.print("[green]Fixes applied successfully![/]")
                    console.print()

                    # Show summary
                    if "summary" in results:
                        s = results["summary"]
                        chunk_ids = s.get("chunk_ids_fixed", 0)
                        deleted = s.get("entities_deleted", 0)
                        merged = s.get("entities_merged", 0)
                        console.print("Summary:")
                        console.print(f"  - Chunk IDs fixed: {chunk_ids}")
                        console.print(f"  - Entities deleted: {deleted}")
                        console.print(f"  - Entities merged: {merged}")

        # Always run validation report at the end
        console.print()
        console.print("[bold]Running validation checks...[/]")
        report = await generate_validation_report(
            driver,
            neo4j_database,
            output_path=args.output,
        )

        # Print report to console
        console.print()
        console.print(report.to_markdown())

        if args.output:
            console.print()
            console.print(f"[green]Report saved to: {args.output}[/]")

    finally:
        await driver.close()


def _run_scrape_command(args: argparse.Namespace) -> None:
    """Run the scrape subcommand.

    Args:
        args: Parsed command-line arguments.
    """
    console.print("[bold cyan]Requirements Guide → Neo4j GraphRAG Pipeline[/]")
    console.print(f"Output directory: {args.output}")
    console.print()

    if args.scrape_only:
        console.print("Mode: [yellow]Scrape only[/] (no Neo4j processing)")
        console.print()
        console.print("Pipeline stages:")
        console.print("  1. Scrape articles + glossary")
        console.print("  2. Save JSON/JSONL output")
    elif args.full:
        console.print("Mode: [bold green]Full pipeline[/] (all stages + fixes)")
        console.print()
        console.print("Pipeline stages:")
        console.print("  1. Scrape articles + glossary")
        console.print("  2. Process through SimpleKGPipeline")
        console.print("  3. Chunk repair (degenerate, index, chunk_ids)")
        console.print("  4. Entity creation (backfill, LangExtract)")
        console.print("  5. Entity cleanup (normalize, dedup, consolidate)")
        console.print("  6. Entity enrichment (summarize, definitions)")
        console.print("  7. Graph analysis (Leiden, communities, embeddings)")
        if not args.skip_supplementary:
            console.print("  8. Supplementary structure (chapters, resources)")
        console.print("  9. Validate + fix + re-validate + report")
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
        console.print("Estimated processing:")
        console.print("  - ~103 articles, ~15 chapters, ~100 glossary terms")
        console.print("  - Embedding cost: ~$0.10 (Voyage AI) or ~$0.50 (OpenAI)")
        console.print("  - LLM extraction cost: ~$5-10 (gpt-4o, 2 gleaning passes)")
        console.print("  - Entity summarization: ~$1-2 (gpt-4o)")
        console.print("  - LangExtract augmentation: ~$2-4 (gpt-4o)")
        console.print("  - Community summaries: ~$0.10 (gpt-4o-mini)")
        if args.full:
            console.print("  - Validation + fixes: ~10 seconds (Cypher only)")
        console.print("  - Estimated total: ~$9-17")
        console.print("  - Estimated time: ~1.5 hours")
        return

    try:
        asyncio.run(
            run_scraper(
                output_dir=args.output,
                use_browser=args.browser,
                scrape_only=args.scrape_only,
                skip_resources=args.skip_resources,
                skip_supplementary=args.skip_supplementary,
                run_validation=args.validate or args.full,
                run_full=args.full,
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
    except PreflightError:
        # Error details already printed by _run_preflight()
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        raise SystemExit(1) from None


def main() -> None:
    """Run the Guide Scraper CLI.

    Provides subcommands for scraping and validation.
    """
    # Load .env file for API keys
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="graphrag-kg",
        description="Scrape Requirements Guide into Neo4j graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  scrape     Run the complete scraping and graph building pipeline (default)
  validate   Validate and optionally fix knowledge graph data quality

Examples:
  graphrag-kg                           # Run full pipeline (same as 'scrape')
  graphrag-kg scrape --validate         # Run pipeline with validation
  graphrag-kg validate                  # Run validation only
  graphrag-kg validate --fix            # Run validation and apply fixes

Required environment variables:
  OPENAI_API_KEY     - For entity extraction + embeddings
  NEO4J_URI          - Database URI (e.g., bolt://localhost:7687)
  NEO4J_USERNAME     - Database username (default: neo4j)
  NEO4J_PASSWORD     - Database password
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create subcommand parsers
    _create_scrape_parser(subparsers)
    _create_validate_parser(subparsers)

    args = parser.parse_args()

    # Default to scrape if no command given
    if args.command is None or args.command == "scrape":
        # If no command, set defaults for scrape
        if args.command is None:
            args.output = Path("output")
            args.browser = False
            args.validate = False
            args.skip_resources = False
            args.skip_supplementary = False
            args.scrape_only = False
            args.dry_run = False
            args.full = False
        _run_scrape_command(args)
    elif args.command == "validate":
        try:
            asyncio.run(_run_validate_command(args))
        except KeyboardInterrupt:
            console.print("\n[yellow]Validation interrupted by user[/]")
            raise SystemExit(1) from None
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
