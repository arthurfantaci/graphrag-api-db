"""Compare the legacy fuzzy industry classifier with TypeSafe Choice judgments.

Reads every Industry node name from the target Neo4j database (point it at
staging), runs both classifiers side by side, and prints:

1. A row per name where the two disagree (or every row with ``--all``).
2. The full model output for each name the exact-match tables did not resolve:
   selected option, confidence, and the top probabilities. This is the
   before/after view for learning how a System One judgment differs from a
   lookup table plus a fuzzy-ratio threshold.

The script is read-only. It never writes to Neo4j.

Usage:
    eval $(./scripts/neo4j-staging.sh env)
    uv run python examples/compare_industry_classification.py
    uv run python examples/compare_industry_classification.py --all
    uv run python examples/compare_industry_classification.py --min-confidence 0.7
    uv run python examples/compare_industry_classification.py --terms retail pharmacy
    uv run python examples/compare_industry_classification.py --json judgments.json

Requires: TYPESAFE_API_KEY, and NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
unless ``--terms`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from graphrag_kg_pipeline.postprocessing.industry_taxonomy import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MIN_CONFIDENCE,
    IndustryJudgment,
    classify_by_table,
    classify_industry_term,
    classify_industry_terms,
)
from graphrag_kg_pipeline.utils.typesafe_client import create_typesafe_client

console = Console()

TOP_PROBABILITIES = 3


async def fetch_industry_names() -> list[str]:
    """Read every distinct Industry node name from Neo4j.

    Returns:
        Sorted list of names.
    """
    from neo4j import AsyncGraphDatabase

    missing = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not os.getenv(k)]
    if missing:
        msg = f"Missing {', '.join(missing)}; run eval $(./scripts/neo4j-staging.sh env) first"
        raise SystemExit(msg)
    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    driver = AsyncGraphDatabase.driver(uri, auth=auth)
    try:
        async with driver.session(database=database) as session:
            result = await session.run(
                "MATCH (i:Industry) RETURN DISTINCT i.name AS name ORDER BY name"
            )
            return [record["name"] async for record in result if record["name"]]
    finally:
        await driver.close()


def format_action(action: str, canonical: str | None) -> str:
    """Render an action tuple as ``keep→rail`` or ``delete``."""
    return f"{action}→{canonical}" if canonical else action


def top_probabilities(judgment: IndustryJudgment) -> str:
    """Render the highest-probability options as ``rail 0.52, none_of_these 0.41``."""
    ranked = sorted(judgment.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{name} {prob:.2f}" for name, prob in ranked[:TOP_PROBABILITIES])


async def run(
    terms: list[str],
    min_confidence: float,
    show_all: bool,
    json_path: Path | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Classify ``terms`` both ways and print the comparison.

    Args:
        terms: Industry names to classify.
        min_confidence: Confidence floor applied to the TypeSafe path.
        show_all: Print agreeing rows too, not just disagreements.
        json_path: Optional file to receive every judgment as JSON.
        batch_size: Terms per TypeSafe request.

    Returns:
        Process exit code.
    """
    api_key = os.getenv("TYPESAFE_API_KEY", "")
    unresolved = [term for term in terms if classify_by_table(term) is None]

    async with create_typesafe_client(api_key) as client:
        judgments = await classify_industry_terms(
            client, unresolved, batch_size=batch_size, min_confidence=min_confidence
        )
    by_term = {judgment.term: judgment for judgment in judgments}

    table = Table(title="Legacy fuzzy cascade vs TypeSafe Choice", show_lines=False)
    table.add_column("name", style="bold")
    table.add_column("table")
    table.add_column("legacy (fuzzy)")
    table.add_column("typesafe choice")
    table.add_column("conf", justify="right")
    table.add_column("typesafe action")
    table.add_column("agree", justify="center")

    disagreements = 0
    for term in terms:
        table_hit = classify_by_table(term)
        legacy = classify_industry_term(term)
        if table_hit is not None:
            new_action = table_hit
            choice, confidence = "(table)", ""
        else:
            judgment = by_term[term]
            new_action = (judgment.action, judgment.canonical)
            choice, confidence = judgment.choice, f"{judgment.confidence:.2f}"

        agree = legacy == new_action
        disagreements += not agree
        if show_all or not agree:
            table.add_row(
                term,
                format_action(*table_hit) if table_hit else "miss",
                format_action(*legacy),
                choice,
                confidence,
                format_action(*new_action),
                "[green]yes[/]" if agree else "[red]no[/]",
            )

    console.print(table)

    detail = Table(title="TypeSafe answers for names the tables missed")
    detail.add_column("name", style="bold")
    detail.add_column("choice")
    detail.add_column("conf", justify="right")
    detail.add_column("top probabilities")
    detail.add_column("action")
    for judgment in judgments:
        detail.add_row(
            judgment.term,
            judgment.choice,
            f"{judgment.confidence:.2f}",
            top_probabilities(judgment),
            format_action(judgment.action, judgment.canonical),
        )
    console.print(detail)

    confidences = [judgment.confidence for judgment in judgments]
    demoted = sum(1 for judgment in judgments if judgment.confidence < min_confidence)
    console.print(
        f"\nNames: {len(terms)}  table hits: {len(terms) - len(unresolved)}  "
        f"sent to TypeSafe: {len(unresolved)}  batch size: {batch_size}  "
        f"disagreements: {disagreements}"
    )
    if confidences:
        console.print(
            f"Confidence min/median/max: {min(confidences):.2f} / "
            f"{statistics.median(confidences):.2f} / {max(confidences):.2f}  "
            f"below floor {min_confidence:.2f}: {demoted}"
        )

    if json_path is not None:
        json_path.write_text(json.dumps([asdict(judgment) for judgment in judgments], indent=2))
        console.print(f"Wrote {len(judgments)} judgments to {json_path}")

    return 0


def main() -> int:
    """Parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--all", action="store_true", help="show agreeing rows too")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="terms per request"
    )
    parser.add_argument("--terms", nargs="+", help="classify these names instead of reading Neo4j")
    parser.add_argument("--json", type=Path, help="write every judgment to this JSON file")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("TYPESAFE_API_KEY"):
        console.print("[red]TYPESAFE_API_KEY is not set.[/]")
        return 1

    terms = args.terms or asyncio.run(fetch_industry_names())
    if not terms:
        console.print("[yellow]No Industry names found.[/]")
        return 0
    return asyncio.run(run(terms, args.min_confidence, args.all, args.json, args.batch_size))


if __name__ == "__main__":
    sys.exit(main())
