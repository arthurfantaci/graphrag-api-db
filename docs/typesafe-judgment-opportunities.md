# TypeSafe Judgment Opportunities — Handoff

**Date**: 2026-09-29
**Source**: Audit of `src/graphrag_kg_pipeline/` for places where code makes a semantic
judgment with brittle mechanisms (lookup tables, fuzzy-ratio thresholds, regex, word lists).
**Status**: Assessment only. No code changed. API key verified working.

---

## What TypeSafe is, in one paragraph

TypeSafe's model (Jev) does not generate text. It answers typed questions over a JSON
`state` and returns probabilities. Three question types:

| Primitive | Returns | Use for |
|---|---|---|
| `Choice` | one option from a defined set, plus a probability per option and a `confidence` | "which label / which candidate / which category" |
| `Noul` | a single probability 0–1 that a yes/no condition holds (no separate confidence) | "is this X?" |
| `Score` | a weighted position on 2–10 ordered levels, plus probabilities and `confidence` | "how strongly / which of these graded situations" |

Recommended pattern from the docs: **code finds candidates, the model picks, code acts.**
Questions over the same `state` run in parallel in one request, so batch them.
Docs index: https://docs.typesafe.ai/llms.txt (append `.md` to any page path for Markdown).

SDK facts (verified 2026-09-29 against the live docs):

- Install: `uv add typesafe-sdk`. Auth: `TYPESAFE_API_KEY` env var.
- Clients: `AsyncTypeSafeClient` (fits this async codebase) and `TypeSafeClient`.
- Call: `await client.system_one(state=..., questions={...}, model="jev-latest")`.
- Classes: `Noul(instructions, criteria=NoulCriteria(true=..., false=...))`,
  `Choice(instructions, criteria={option: description_or_None})`,
  `Score(instructions, criteria=[level0_desc, level1_desc, ...])`.
- Read answers: `response.answers[key].noul` / `.choice` / `.score`, plus `.probabilities`
  and `.confidence` on Choice and Score.
- HTTP: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`.
- Cost reference from the parallel-questions cookbook: about $0.0005 and 0.27 s for a
  13-question batched request.

Key check performed 2026-09-29: one Noul call ("Is `retail` the name of an industry
sector?") returned HTTP 200 and `noul: 0.95`. That is the exact input the current fuzzy
matcher maps to "rail".

---

## Where the pipeline judges with brittle code

The three audits found roughly 60 heuristic sites. Most are structural and fine. Eleven
make a semantic call that a Choice, Noul, or Score question would do better. Ranked by
payoff.

### 1. Cross-label dedup picks the winning label from a fixed list

- **Where**: `postprocessing/normalizer.py:87` (`LABEL_PRIORITY`), `:373-417`
  (`_resolve_winning_label`), `:465-501` (`deduplicate_cross_label`).
- **Current rule**: any two `__Entity__` nodes with the same lowercased `name` are merged,
  whatever their labels. The surviving label is the highest in a fixed list
  (Standard > Organization > Tool > Industry > Role > Methodology > Concept > Outcome >
  Artifact > Processstage > Bestpractice > Challenge).
- **Failure**: homographs (same name, different meaning) are merged. Labels not in the list
  default to Concept. `docs/cross-label-dedup-overview.md` already flags the homograph
  risk (the EARS note).
- **Replacement**: one request per duplicate group.
  - `state = {"name": ..., "candidates": [{"label", "definition", "sample_mention"}...],
    "label_definitions": {label: schema description}}`
  - `same_entity`: `Noul` — "Do all candidates describe one real-world entity, or are
    they distinct things that share a name?"
  - `best_label`: `Choice` over only the labels present in the group; criteria copied from
    `extraction/schema.py` descriptions.
  - Merge only when `same_entity` is above a threshold you set on real data, and use
    `best_label.confidence` to decide auto-merge vs. queue.
- **Ground truth available**: the 280 duplicate groups and 330 merges recorded in
  `docs/cross-label-dedup-results.md`.
- **Cookbook**: https://docs.typesafe.ai/cookbooks/entity_alignment.md

### 2. Industry classification runs a 106-entry table plus fuzzy matching at ratio 80

- **Where**: `postprocessing/industry_taxonomy.py:28-148` (`INDUSTRY_TAXONOMY`, 106
  variants → 23 canonical), `:155-238` (three exclusion sets), `:261-309`
  (`classify_industry_term` cascade with `process.extractOne` at `:280`, `:290`, `:300`).
- **Failure**: "retail" → "rail" at score 80. "pharmacy" → life sciences.
  "safety-critical systems" → reclassify as Concept at 78.9. Terms such as "chemical",
  "agriculture", "systems engineering", "industrial automation", "education", "banks"
  return "unknown" and are left untouched. Thresholds 80 and 75 are hardcoded.
- **Replacement**: a `Choice` per unresolved term.
  - Options: the 18 canonical industries, plus `organization` (a body, not a sector),
    `concept_not_industry`, `too_generic`, `none_of_these`.
  - Keep the exact-match table as a free fast path. Put every unresolved term in one
    `state` object and ask one Choice per term (`terms[i]`), so a full run is a handful of
    requests.
  - Map the answer to the existing action tuple `("keep", canonical) | ("reclassify_org",
    None) | ("reclassify", None) | ("delete", None) | ("unknown", None)` so the rest of
    `consolidate_industries()` is unchanged.
- **Cookbook**: https://docs.typesafe.ai/cookbooks/hierarchical_classification.md

### 3. Plural and near-duplicate detection uses a plus-s rule and substring length

- **Where**: `postprocessing/entity_cleanup.py:575` and `validation/queries.py:272`
  (`plural.name = singular.name + 's'`); `validation/queries.py:541-559` (substring within
  5 characters); `postprocessing/entity_cleanup.py:153-234` (72-entry
  `PLURAL_TO_SINGULAR`, unused by `run_cleanup`).
- **Failure**: catches "need/needs", "new/news"; misses "dependency/dependencies",
  "process/processes", "criterion/criteria". Near-duplicate check flags "test/testing"
  and misses "rtm" vs "requirements traceability matrix". The plural check gates
  `validation_passed`.
- **Replacement**: keep code rules but widen for recall (+s, -es, -ies, fuzz ratio ≥ 70,
  substring). Then ask a `Score` per candidate pair with three levels:
  "different things" / "closely related, may or may not be the same" / "one and the same".
  Merge at the top level, queue the middle, ignore the rest. Add `Noul` sub-questions
  ("same label?", "one is the plural of the other?") for explainability.
- **Cookbook**: https://docs.typesafe.ai/cookbooks/entity_alignment.md

### 4. Generic-term deletion has three disagreeing lists

- **Where**: `postprocessing/entity_cleanup.py:43-144` (95 terms, applied with
  `DETACH DELETE` at `:530-546`); `validation/queries.py:313-342` (17 terms);
  `extraction/prompts.py:237-251` (prompt list).
- **Failure**: the cleanup list includes "software", which is also a canonical industry, so
  an Industry node named "software" is deleted one step before industry consolidation
  tries to create it. Lists apply to all 12 labels. Report and fix disagree on counts.
- **Replacement**: a `Noul` — "Is this name too vague to be a useful node in a
  requirements-management knowledge graph?" with `state = {name, label, definition,
  mentions: [2 chunk sentences]}`. Deletion is destructive, so gate high (≥ 0.9) and keep a
  short hard blocklist for the obvious cases. Collapse the three lists into one.

### 5. Mislabeled Challenge detection matches any word from a positive-word list

- **Where**: `postprocessing/entity_cleanup.py:238-272` (`POSITIVE_OUTCOME_WORDS`, 31
  words), `validation/queries.py:519-528` and `validation/fixes.py:262-271` (any-word
  match), `validation/fixes.py:282-308` (relabels to Concept, not Outcome).
- **Failure**: flags "poor quality", "security vulnerability", "compliance gap",
  "reduced visibility" as outcomes. The helper `is_potentially_mislabeled_challenge`
  uses first-word only; the live queries use any-word. The fix relabels to Concept,
  contradicting the schema's Outcome type.
- **Replacement**: a `Choice` over `Challenge`, `Outcome`, `Concept` with the entity's
  name, definition, and a mention. Code relabels to whatever the answer is. Resolves both
  the false positives and the wrong target label.

### 6. Glossary linking uses character similarity at 85

- **Where**: `postprocessing/glossary_linker.py:88` (`fuzz.ratio ≥ 85`, Concept label
  only, not called by the pipeline); `validation/fixes.py:412-422` (definition backfill
  by exact `toLower(name) = toLower(term)`).
- **Failure**: "functional requirement" vs "non-functional requirement" scores about 92
  and would link. "risk analysis" vs "risk analyst" scores 88. Only Concept is searched.
  Backfill needs an exact match, so most terms never fill.
- **Replacement**: code pulls the top 5 fuzzy candidates across all 12 labels, then a
  `Choice` — "Which entity does this glossary term define?" — over those candidates plus
  `none`. The model can only pick a value code already found.
- **Cookbook**: https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md

### 7. Promo and call-to-action removal in the parser

- **Where**: `parser.py:463` (`_is_cta_section`), `parser.py:553`
  (`_remove_promo_text` state machine), `parser.py:49-75` (pattern lists).
- **Failure**: one matching link or phrase condemns a whole section. The markdown skipper
  keeps deleting until a heading that lacks "demo", "trial", "contact", or "pricing", so
  a heading like "Clinical Trial Requirements" swallows real content. A heading "Get
  Started with Traceability" is classified as a CTA.
- **Replacement**: keep the regexes as flaggers. For each flagged block ask a `Noul` —
  "Is this block marketing call-to-action rather than guide content?" with
  `state = {text, hrefs, heading}`. Only flagged blocks are sent.

### 8. Webinar titles come from a fallback chain

- **Where**: `parser.py:939` (`_create_webinar_reference`), `validation/queries.py:436-441`
  and `validation/fixes.py:200-217` (`fix_truncated_webinar_titles` cuts the description
  at the first period).
- **Failure**: accepts "Watch now" or "here" as the title. Slug-derived titles get mangled
  casing ("Iso 26262"). The first-period cut breaks on "e.g.", "2.0", domains.
- **Replacement**: collect every candidate in code (anchor text, img alt, img title,
  slug-title, first sentence of description, `og:title`) and ask a `Choice` — "Which of
  these is the webinar's real title?" — plus `none_usable`.

### Smaller wins, same shape

- **9. Glossary parsing strategy cascade** (`parser.py:177-279`): four strategies run
  all-or-nothing. Run all four, then a `Noul` per candidate pair — "Is this a real
  glossary term/definition pair?" — to drop letter dividers and "Note:" lines.
- **10. Degenerate chunk deletion** (`validation/fixes.py:76-82`, `MIN_CHUNK_TEXT_LENGTH
  = 100`; `chunking/hierarchical_chunker.py:136-150`, `min_chunk_size`): a `Noul` —
  "Does this short chunk carry standalone meaning?" — before deleting or dropping.
- **11. Gleaned relationship validity** (`extraction/gleaning.py:287-313`): no pattern
  check, endpoints matched by bare name. A `Noul` — "Does the chunk text support this
  (source, relation, target) claim?" — in the citation-check style.
  Cookbook: https://docs.typesafe.ai/cookbooks/citation_check.md

---

## What to leave alone

The gpt-4o extraction (SimpleKGPipeline), embeddings, Leiden, TOC parsing, and constraint
code are generation or pure structure. TypeSafe has no role there.

## Plain bugs found during the audit (not judgment problems)

Fix these in code; no model needed.

- `parser.py` `_parse_sections` (`:1302-1346`): `Section.cross_references` is always empty
  because markdown strings are re-parsed as HTML. The last section is hardcoded to `[]`.
- `labels(n)[0]` is used as "the entity type" in `extraction/gleaning.py:200`,
  `graph/community_summarizer.py:150`, and about 12 places in `validation/queries.py`.
  Entities also carry `__Entity__` and `__KGBuilder__`, and Neo4j does not guarantee label
  order. `postprocessing/normalizer.py:102` (`_SYSTEM_LABELS`) already has the filter.
- `loaders/html_loader.py:182-191` selects the first `.flex_cell_inner`; `parser.py:420-461`
  selects the second. One of them is wrong for this site.
- `loaders/html_loader.py:202-215` inserts newline markers then wipes them with
  `" ".join(text.split())`, so HTML-file input reaches the markdown splitter as one line.
- `extraction/gleaning.py:243-257` drops near-miss labels ("BestPractice", "ProcessStage")
  because the schema uses "Bestpractice" / "Processstage". Compare case-insensitively.
- `extraction/schema.py:518-588` `PATTERNS` is contradicted by prompt examples at
  `extraction/prompts.py:116`, `:348-352`, `:402`. `validate_pattern()` and
  `get_few_shot_examples()` have no callers.
- `postprocessing/mentioned_in_backfill.py:25-47` maps standards to "industrial
  automation", "systems engineering", "software development", none of which are canonical
  industries.
- `postprocessing/normalizer.py:280-311`: comment says "keep primary's if both have" but
  `SET primary += dup` overwrites the primary.

---

## How to start

1. Confirm `TYPESAFE_API_KEY` is set in the session (`echo ${#TYPESAFE_API_KEY}` should
   print 108). If not, the shell profile guard `_SHELL_SECRETS_LOADED` was inherited; run
   `unset _SHELL_SECRETS_LOADED && exec zsh` in the terminal and relaunch Claude Code.
2. `uv add typesafe-sdk`. Add a thin async wrapper in `utils/typesafe_client.py` that
   reuses the tenacity retry pattern in `utils/retry.py` (see `openai_retry`). Keep
   `_tenacity_logger` as `logging.getLogger()`.
3. Add a `TYPESAFE_API_KEY` presence check to `preflight.py`, advisory only.
4. Implement item 2 (industry) first. It is small, self-contained, and has a working test
   case. Then item 1 (cross-label dedup), which has 280 groups of ground truth.
5. Run against **staging** (`eval $(./scripts/neo4j-staging.sh env)`), diff decisions
   against the table-driven output, set thresholds from the disagreements.
6. Write tests that mock `system_one()` responses; do not call the API in CI.

Git workflow: Issue → Branch → PR for the code changes. Docs-only and memory-only edits
bundle into that PR.
