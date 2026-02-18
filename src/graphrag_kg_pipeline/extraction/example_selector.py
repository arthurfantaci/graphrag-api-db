"""Dynamic few-shot example selection for extraction prompts.

Selects the most relevant examples from a curated pool based on
text similarity, improving extraction quality for each chunk.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from graphrag_kg_pipeline.extraction.prompts import get_few_shot_examples

logger = structlog.get_logger(__name__)

# Minimum word length for keyword tokenization
_MIN_KEYWORD_LENGTH = 3


class FewShotExampleSelector:
    """Selects relevant few-shot examples for entity extraction prompts.

    Uses simple keyword overlap scoring to select the most relevant
    examples from the curated pool for each input chunk. This avoids
    the cost of embedding the example pool while still providing
    contextually appropriate examples.

    Attributes:
        examples: Pool of curated few-shot examples.
        num_examples: Number of examples to select per chunk.
    """

    def __init__(
        self,
        examples: list[dict[str, Any]] | None = None,
        num_examples: int = 3,
    ) -> None:
        """Initialize the selector.

        Args:
            examples: Pool of curated examples. If None, loads from prompts module.
            num_examples: Number of examples to select per chunk.
        """
        if examples is None:
            examples = get_few_shot_examples()

        self.examples = examples
        self.num_examples = num_examples

        # Pre-compute keyword sets for each example
        self._example_keywords: list[set[str]] = []
        for example in self.examples:
            keywords = set()
            # Extract keywords from text
            text_words = example["text"].lower().split()
            keywords.update(
                w.strip(".,;:!?()\"'") for w in text_words if len(w) > _MIN_KEYWORD_LENGTH
            )
            # Extract entity type names
            for entity in example.get("entities", []):
                keywords.add(entity.get("type", "").lower())
                keywords.add(entity.get("name", "").lower())
            self._example_keywords.append(keywords)

    def select_examples(self, input_text: str) -> list[dict[str, Any]]:
        """Select the most relevant examples for an input chunk.

        Uses keyword overlap scoring to rank examples by relevance.

        Args:
            input_text: The chunk text to find relevant examples for.

        Returns:
            List of the most relevant examples (up to num_examples).
        """
        if not input_text or not self.examples:
            return self.examples[: self.num_examples]

        # Tokenize input
        input_words = {
            w.strip(".,;:!?()\"'")
            for w in input_text.lower().split()
            if len(w) > _MIN_KEYWORD_LENGTH
        }

        # Score each example by keyword overlap
        scored = []
        for i, keywords in enumerate(self._example_keywords):
            overlap = len(input_words & keywords)
            scored.append((overlap, i))

        # Sort by overlap (descending), then by original order (ascending)
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Return top N examples
        selected = []
        for _score, idx in scored[: self.num_examples]:
            selected.append(self.examples[idx])

        return selected

    def format_examples_for_prompt(self, input_text: str) -> str:
        """Format selected examples as a string for the extraction prompt.

        Args:
            input_text: The chunk text to select examples for.

        Returns:
            Formatted string of examples for inclusion in the prompt.
        """
        selected = self.select_examples(input_text)
        if not selected:
            return ""

        parts = ["## SELECTED FEW-SHOT EXAMPLES:\n"]
        for i, example in enumerate(selected, 1):
            parts.append(f"### Dynamic Example {i}:")
            parts.append(f'Text: "{example["text"]}"')
            parts.append("Entities:")
            for entity in example.get("entities", []):
                entity_str = json.dumps(entity, indent=2)
                parts.append(f"  {entity_str}")
            if example.get("relationships"):
                parts.append("Relationships:")
                for rel in example["relationships"]:
                    parts.append(f"  ({rel['source']})-[{rel['type']}]->({rel['target']})")
            parts.append("")

        return "\n".join(parts)
