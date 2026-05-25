# Neo4j Sales Engineer — Technical Evaluation Prep

**Project:** `graphrag-api-db` — an end-to-end GraphRAG knowledge-graph pipeline built on Neo4j and `neo4j_graphrag`.

**How to use this doc:** The first section is your 5–6 bullet **elevator overview** — memorize the one-liners. Everything after is the **drill-down** for each point, following the pipeline stage by stage, with talking points, the Neo4j-product positioning, the deep technical detail you can pull from if pressed, and the likely interviewer questions. Lean sales-positioning, but every claim is anchored in real code (file:line references included so you can re-read the source).

> **Framing note:** This repo is the *ingestion / graph-construction* side. It builds the Neo4j graph and the vector/community indexes that **power** GraphRAG retrieval. A separate repo holds the Retrieval & Chat UI. When the interviewer asks "so how do you query it?", that's your cue to bridge to the retrieval repo — talking points for that bridge are in the final section.

---

## 1. Elevator Overview (the 5–6 bullets)

**1. End-to-end GraphRAG pipeline on Neo4j, built on `neo4j_graphrag`.**
I built a complete ETL-to-graph pipeline that ingests an unstructured technical guide (~101 articles + glossary) and turns it into a queryable Neo4j knowledge graph using Neo4j's first-party `neo4j_graphrag` library and `SimpleKGPipeline`. It demonstrates the full GraphRAG value proposition — vector search *plus* graph structure — on Neo4j-native tooling rather than a bolted-on stack.

**2. Schema-constrained LLM entity extraction (12 node types, 14 relationship types).**
Rather than letting the LLM emit an unconstrained graph, I defined a domain schema with ~50 validated (source, relationship, target) patterns. This prevents schema drift, keeps the graph queryable, and is exactly the governed-extraction story Neo4j positions against naive RAG.

**3. Hybrid retrieval foundation: vector embeddings + graph traversal + community summaries.**
The graph supports semantic retrieval via Neo4j native vector indexes (Voyage AI voyage-4, 1024-dim, OpenAI fallback) layered on explicit relationships and document/chunk structure. I added Leiden community detection with LLM-generated, separately-embedded community summaries — the Microsoft-GraphRAG global-vs-local retrieval pattern, running natively against Neo4j.

**4. Production-grade data quality and entity resolution.**
An 11-step, 3-phase post-processing layer handles normalization, same-name and cross-label deduplication, an industry taxonomy collapsing 100+ variants into ~18 canonical nodes, and a validation framework with dry-run repair. Reflects the real-world truth that GraphRAG quality lives or dies on entity resolution, not extraction.

**5. Deep Neo4j platform fluency (constraints, APOC, vector indexes, graph analytics).**
The project navigates non-obvious Neo4j behaviors — why entity-type uniqueness constraints cause silent batch rollbacks, the `__Entity__`/`__KGBuilder__` labeling entity resolution depends on, APOC-driven label operations, and preflight checks for connectivity/APOC/vector-index dimensions. Hands-on platform knowledge a customer will probe.

**6. Engineered like a product, with a safe staging/production workflow.**
Clean architecture (Protocol-based fetcher abstraction, dependency injection), strict tooling (Ruff, type-checking, pytest, CI/Codecov), cost-estimation dry-runs, and an environment-driven staging-vs-production switch with confirmation prompts before destructive operations against production Aura. Shows I can operationalize GraphRAG, not just prototype it.

---

## 2. Drill-down — Stage 1 & 2: Ingestion → Neo4j

**Arc to land:** *"I take unstructured web content and end up with a Neo4j graph queryable by both vectors and relationships — using Neo4j's own GraphRAG library."*

### Stage 1 — Scrape
**One-liner:** "I scrape a full technical guide — ~101 articles plus a glossary — dynamically discovering structure from the site's TOC, and normalize everything to Markdown before it touches Neo4j."

Talking points:
- **Dynamic discovery, not hard-coded** — chapters/articles come from the live `#chapter-menu` TOC in a single request, so the pipeline survives content changes (robust ingestion, not demo-ware).
- **Protocol-based fetcher abstraction** (httpx default; Playwright for JS-rendered content) via dependency injection — your "I architect for testability and extensibility" proof point. (`fetcher.py`, PEP 544 structural subtyping.)
- **Respectful scraping** — async with a concurrency semaphore (default 3), exponential-backoff retry, custom User-Agent (Cloudflare blocks the default httpx UA).

### Stage 2 — Extract & embed (the GraphRAG core)
**One-liner:** "I use `neo4j_graphrag`'s `SimpleKGPipeline` to do schema-constrained LLM entity extraction and vector embedding in one pass, writing both a *lexical graph* (documents → chunks) and a *domain graph* (entities + relationships) into Neo4j."

Neo4j product-positioning hooks:
- **First-party library.** `SimpleKGPipeline`, `LexicalGraphConfig`, `OpenAILLM`, and the `Embedder` interface are all `neo4j_graphrag` (`extraction/pipeline.py:204-284`). Message: *"Neo4j isn't just the database — its GraphRAG package gives you the whole construction pipeline."*
- **Dual-graph model = the differentiator.** `LexicalGraphConfig` (`pipeline.py:264-269`) wires `Article -[FROM_ARTICLE]- Chunk` and entities `-[MENTIONED_IN]-> Chunk`. Every extracted entity is traceable to its source chunk and article — *provenance/grounding* that vector-only RAG can't provide. Your strongest "why graph beats a plain vector DB" line.
- **Vectors live *in* Neo4j**, on chunk nodes, served by Neo4j's native vector index. *"One database does semantic search AND graph traversal — no separate vector store to keep in sync."*
- **Quality levers:** schema-constrained extraction (12 node / 14 rel / ~50 patterns) prevents drift; **gleaning** runs 2 LLM passes catching 20–30% more entities (`pipeline.py:347-365`); `perform_entity_resolution=True` uses the built-in resolver; `temperature=0` for determinism; `on_error="IGNORE"` so one bad chunk can't kill a ~1.5-hour run.
- **Chunking:** two-stage hierarchical splitter — LangChain `HTMLHeaderTextSplitter` preserves document structure, then `RecursiveCharacterTextSplitter` (or optional Chonkie semantic chunker) keeps chunks within embedding/context limits (`chunking/hierarchical_chunker.py`).

Likely questions:
- *"Why a graph instead of a vector DB?"* → provenance via the lexical graph + multi-hop traversal + community summaries. Vector similarity alone can't answer "what challenges does traceability address across industries."
- *"Why Neo4j specifically?"* → vectors + graph + graph analytics in one engine; first-party GraphRAG library; APOC.
- *"Hardest part of KG construction?"* → not extraction — **entity resolution and schema governance.** (Segue to Stage 3.)

---

## 3. Drill-down — Stage 3: Normalization, Entity Resolution & Community Detection

This is where the deepest Neo4j-product story lives. The pipeline is deliberately ordered into **3 phases** so all entity-creating steps run before any cleanup, and cleanup runs before graph analytics.

**Phase A — Entity creation:**
1. `MentionedInBackfiller.backfill()` — creates `MENTIONED_IN` + `APPLIES_TO` relationships.
2. `LangExtractAugmenter.augment()` — post-extraction augmentation with **source grounding** (text-span provenance).

**Phase B — Entity cleanup (runs after all entities exist):**
3. `EntityNormalizer.normalize_all_entities()` — lowercase + trim.
4. `deduplicate_by_name()` — merge same-name duplicates.
5. `deduplicate_cross_label()` — merge same-name entities with *different* type labels.
6. `EntityCleanupNormalizer.run_cleanup()` — drop generics, merge plural→singular.
7. `IndustryNormalizer.consolidate_industries()` — collapse 100+ variants → ~18 canonical industries (`postprocessing/industry_taxonomy.py`).
8. `EntitySummarizer.summarize()` — LLM-generated entity descriptions.

**Phase C — Graph analysis (on clean entities):**
9. `CommunityDetector.detect_communities()` — **Leiden** clustering.
10. `CommunitySummarizer.summarize_communities()` — LLM summaries → `Community` nodes via `IN_COMMUNITY`.
11. `CommunityEmbedder.embed_community_summaries()` — vector embeddings of summaries.

### Talking points
- **Entity resolution is the real work.** Normalization + same-name dedup + **cross-label dedup** (e.g., "FDA" extracted once as `Organization`, once as `Industry` → merged) is what makes the graph trustworthy. Lead with: *"Anyone can get an LLM to emit triples; the differentiator is resolving them into clean, canonical entities."*
- **Domain taxonomy.** `INDUSTRY_TAXONOMY` maps 100+ surface forms to ~18 canonical industries; `CANONICAL_INDUSTRIES` is derived (`industry_taxonomy.py:241`). An `ORGANIZATIONS_NOT_INDUSTRIES` set (`:190`) relabels NASA/FDA/IEEE from Industry → Organization. Shows you encode domain knowledge, not just generic NLP.
- **Phase ordering is intentional.** New entities (Phase A) must exist *before* cleanup (Phase B) so they get normalized too; analytics (Phase C) must run on *clean* entities or communities are noisy. This sequencing is a genuine engineering insight worth calling out.

### The Microsoft-GraphRAG pattern, on Neo4j (your headline for this stage)
- **Leiden community detection** via `leidenalg` + `igraph` on **semantic edges only** (structural edges like `FROM_ARTICLE` excluded), reproducible with a fixed seed and `gamma` resolution parameter (`graph/community_detection.py:46-70`). Exports the entity graph, runs Leiden locally, writes community IDs back to Neo4j.
- **Community summaries** — each cluster gets an LLM-generated summary stored as a `Community` node, linked `(:Entity)-[:IN_COMMUNITY]->(:Community)`.
- **Community embeddings** — summaries are embedded (voyage-4, 1024d) into a dedicated `community_summary_embeddings` vector index, cosine similarity (`graph/constraints.py:328-350`).
- **Why this matters for retrieval:** this is the *global* (thematic, "what are the big themes") vs *local* (entity-specific) retrieval split that Microsoft's GraphRAG popularized — and you implemented it on Neo4j primitives. Positioning: *"Neo4j gives you the storage, the vector index, AND the graph the community algorithm runs on — three roles, one engine."*
- **GDS angle:** I ran Leiden client-side via `leidenalg` for reference-implementation fidelity and reproducibility; the natural enterprise path is Neo4j **Graph Data Science**, which ships Leiden/Louvain as a native, scalable procedure. Be ready to say *"I'd move this into GDS for production scale"* — it shows you know the product roadmap.

Likely questions:
- *"What's Leiden and why not Louvain?"* → Leiden fixes Louvain's badly-connected-community defect, guarantees well-connected communities, converges better. Both are in Neo4j GDS.
- *"How do you handle duplicate entities?"* → built-in entity resolution at ingest + three explicit dedup passes (name, cross-label, plural) in post-processing.
- *"Resolution / number of communities?"* → `gamma` parameter; higher = more, smaller communities.

---

## 4. Drill-down — Stage 4: Supplementary Graph Structure

**One-liner:** "On top of the extracted knowledge graph, I add a navigational structure layer — Chapter nodes, Resource nodes (Image/Video/Webinar), and a glossary linked to the concepts it defines."

Talking points:
- `SupplementaryGraphBuilder` (`graph/supplementary.py`) adds `Chapter` nodes with article relationships, `Resource` nodes, and glossary structure.
- **Glossary-to-concept linking** uses fuzzy matching (`rapidfuzz`) so a defined term connects to the extracted `Concept` it describes — enriching retrieval with authoritative definitions.
- Positioning: *"This is where graph shines over vectors — I can navigate from a chapter to its articles to the entities they mention to the community those entities belong to, all as first-class relationships."* It's the multi-hop story made concrete.

---

## 5. Drill-down — Stage 5: Validation & Data Quality

**One-liner:** "I built a validation framework with Cypher-based quality checks and safe, dry-run-previewable repair operations — because a knowledge graph you can't trust is worse than no graph."

Talking points:
- `ValidationQueries` (`validation/queries.py`) runs checks: orphan chunks (chunks with no `FROM_ARTICLE`), duplicate entities, missing required properties, invalid relationship patterns.
- `ValidationFixer` applies repairs with a **dry-run preview** mode; fix ordering is deliberate (delete degenerate → re-index → chunk_ids → webinar titles → relabel → backfill `MENTIONED_IN` → definitions → generics → plurals).
- **Pass/fail gates:** orphan_chunks, duplicates, chunk_ids, chunk_index, plural_duplicates (industry count is advisory).
- Reports auto-archive prior versions with ISO-8601 timestamps before writing a new one.
- Positioning for an SE: *"This is the operational maturity question every customer eventually asks — how do you know the graph is correct, and how do you fix it safely in production?"*

---

## 6. Drill-down — Neo4j Platform Fluency (the "shows you actually know the product" section)

These are the non-obvious lessons that prove hands-on depth. Any one of them can win the room.

- **Entity-type uniqueness constraints are a trap.** `neo4j_graphrag` 1.13+ creates entities with `CREATE` + `apoc.create.addLabels()`, not `MERGE`. Per-type uniqueness constraints (e.g., `Concept.name`) cause `IndexEntryConflictException` when the same name recurs across extraction batches — **silently rolling back entire batch transactions.** So I deliberately keep uniqueness constraints **only** on structural nodes — Article, Chunk, Chapter, Image, Video, Webinar, Definition (`graph/constraints.py:21-39`) — and let `neo4j_graphrag`'s entity resolution handle dedup. *This is the single best "I learned this the hard way in production" anecdote.*
- **The `__Entity__` / `__KGBuilder__` labels matter.** Gleaning and the LangExtract augmenter `MERGE` entities with `:__Entity__:__KGBuilder__` labels (`extraction/gleaning.py:271-272`). Without `__Entity__`, gleaned/augmented nodes are **invisible** to entity resolution and cross-label dedup. Demonstrates you understand `neo4j_graphrag`'s internal labeling contract.
- **APOC dependency** — label operations rely on `apoc.create.addLabels()`; the pipeline preflight verifies APOC is installed.
- **Native vector indexes** — chunk embeddings and a separate `community_summary_embeddings` index (1024d, cosine) are created and dimension-checked (`constraints.py:294-350`). Talk to the dimension-match requirement (index dim must equal embedder dim).
- **Preflight validation** — checks Neo4j connectivity, APOC availability, and vector-index dimensions before a run (`preflight.py`). Shows you fail fast instead of 90 minutes into a load.
- **Direct OpenAI calls (gleaning) need `response_format={"type": "json_object"}`** (`gleaning.py:183`) for reliable JSON — a concrete LLM-integration gotcha.
- **Embedding provider abstraction** — `VoyageAIEmbeddings` implements `neo4j_graphrag.embeddings.base.Embedder`, with **asymmetric input types** ("document" at index time, "query" at search time) and OpenAI fallback. Shows you understand asymmetric embeddings and clean interface design.

---

## 7. Drill-down — Engineering & Ops Maturity

**One-liner:** "I built this like a product I'd hand to a customer, not a notebook."

- **Architecture patterns:** Protocol/structural subtyping (PEP 544) for the fetcher, Strategy pattern for swappable fetch backends, dependency injection, lazy initialization (Playwright only starts on first request).
- **Tooling:** Python 3.13, Ruff (broad rule set incl. security/bandit), `ty` type-checking, pytest, pre-commit, CI with Codecov patch-coverage gate.
- **Cost control:** `--dry-run` estimates LLM/embedding cost before committing to a ~1.5-hour run.
- **Staging vs production safety:** environment-driven targeting (shell-exported vars beat `.env`), a staging Neo4j Desktop DBMS, green "(staging)" / yellow "(production)" CLI labels, and **interactive confirmation prompts** before `--full` or `--fix` against production Aura. Promotion path dumps staging and uploads to Aura.
- Positioning: *"For a customer, the risk isn't 'can GraphRAG work' — it's 'can I run it safely against prod and know what it costs.' I built for both."*

---

## 8. Bridge to Retrieval & the Chat UI (the payoff)

When the interviewer asks "how do you actually query this?" — bridge to the separate Retrieval & Chat UI repo. Talking points you can give *from the graph you built here*, even before we review that repo together:

- **Three retrieval modes the graph supports:**
  1. **Local / vector** — embed the user query (voyage-4, `input_type="query"`), hit the Neo4j chunk vector index, return top-k chunks *with* their `Article`/`Chapter` provenance.
  2. **Graph-augmented** — from the retrieved entities, traverse relationships (`ADDRESSES`, `REQUIRES`, `APPLIES_TO`, etc.) to pull in connected context a pure vector search would miss (multi-hop).
  3. **Global / community** — for thematic questions, vector-search the `community_summary_embeddings` index and answer from `Community` summaries (the Microsoft-GraphRAG "global search").
- **Why this is the closing argument:** the construction repo isn't the point — it's that I built a graph deliberately shaped to make all three retrieval modes possible in one Neo4j instance. *"Vector-only RAG gives you one of these. The graph gives you all three, plus provenance."*

> **Next session action:** review the Retrieval & Chat UI repo directly so these bridge points can be made specific (actual retriever classes, Cypher, prompt assembly).

---

## 9. Rapid-fire Q&A cheat sheet

- **"What is GraphRAG?"** — RAG where retrieval is over a knowledge graph (entities + relationships + community summaries), not just a flat vector store. Adds provenance, multi-hop reasoning, and global/thematic retrieval.
- **"Why Neo4j over a vector DB?"** — One engine for vectors + graph traversal + graph analytics; native vector indexes; first-party `neo4j_graphrag` + GDS; provenance via the lexical graph.
- **"Vector index details?"** — Cosine, 1024-dim (voyage-4); separate indexes for chunks and community summaries; index dimension must equal embedder dimension.
- **"How big / how long?"** — ~101 articles + glossary; full re-ingestion ~1.5 hours (~1 article/min), dominated by LLM extraction + gleaning.
- **"Biggest lesson?"** — Entity-type uniqueness constraints silently roll back batches under `neo4j_graphrag`'s CREATE+addLabels pattern; keep uniqueness on structural nodes only and let entity resolution dedup the rest.
- **"What would you do for enterprise scale?"** — Move Leiden into Neo4j GDS, batch/parallelize extraction, consider structured-output extraction when `SimpleKGPipeline` supports it.
