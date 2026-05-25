# Handoff Brief → Retrieval & Chat UI Session

**Purpose:** Carry the goal, framing, and graph-contract knowledge from the `graphrag-api-db` (construction) prep session into a new Claude Code session rooted in the **Retrieval & Chat UI** repo. Paste the "Kickoff Prompt" (bottom of this file) into the new session to start with full context.

---

## 1. The goal (unchanged across sessions)

Prepare to articulate this GraphRAG work during a **Neo4j Sales Engineer technical evaluation**. Audience leans **sales-positioning grounded in Neo4j product detail** — i.e., always tie technical choices back to *why Neo4j / why GraphRAG beats naive vector RAG*. Output is interview talking points the candidate can expand on demand, not shipped features.

In the construction repo we produced `NEO4J_INTERVIEW_PREP.md` (PR #82): a 5–6 bullet overview + stage-by-stage drill-downs + a Neo4j platform-fluency section + a Q&A cheat sheet. **Section 8 of that doc is the retrieval bridge** — the new session's job is to make those bridge points concrete with real retriever code, Cypher, prompt assembly, and similarity scoring.

## 2. Working style that worked last session

- Ground every talking point in **actual source (file:line)**, not the README — the interviewer may drill.
- Lead with the one-liner the candidate says out loud, then layer the deeper detail "if pushed."
- For each topic, anticipate the **likely interviewer question** and the angle to answer it.
- Keep Neo4j-product positioning explicit: vectors + graph + analytics in one engine, provenance, multi-hop, global-vs-local retrieval.

## 3. The graph contract the retrieval repo queries against (verified facts)

The retrieval app reads a Neo4j graph built by `graphrag-api-db`. Key shape:

**Lexical graph (provenance backbone):**
- `(c:Chunk)-[:FROM_ARTICLE]->(a:Article)` — every chunk traces to its source article.
- `(e:<EntityType>)-[:MENTIONED_IN]->(c:Chunk)` — entities trace to the chunks that mention them.
- `Chunk` has `.text`, `.embedding`, `.chunk_id`, `.source_article_id`. `Article` has `.title`, `.url`, `.chapter_number`.

**Domain graph (entities + relationships):**
- 12 entity node types (Concept, Challenge, Artifact, Bestpractice, Processstage, Role, Standard, Tool, Methodology, Industry, Organization, Outcome), all also labeled `:__Entity__:__KGBuilder__`.
- Entity `.name` is lowercased/normalized; `.display_name` keeps original casing; many have LLM-generated `.summary`.
- 14 semantic relationship types for traversal: ADDRESSES, REQUIRES, COMPONENT_OF, APPLIES_TO, PUBLISHES, REGULATES, DEVELOPS, ACHIEVES, etc.

**Community layer (global / thematic retrieval):**
- `(e)-[:IN_COMMUNITY]->(:Community)`; `Community` nodes carry `.summary` and `.summary_embedding` (Leiden clustering + LLM summaries).

**Indexes the retrievers rely on:**
- `chunk_embeddings` — VECTOR index on `(c:Chunk).embedding`, cosine. ⚠️ **Verify actual dimension in the live graph**: the construction config uses Voyage `voyage-4` = **1024d**, but the index-creation helper defaults to 1536 (OpenAI). The retrieval query embedding MUST match the index dimension and the embedding model used at ingest.
- `chunk_text_fulltext` — FULL-TEXT index on `(c:Chunk).text` → enables **hybrid (vector + BM25) retrieval**.
- `community_summary_embeddings` — VECTOR index on `(c:Community).summary_embedding`, **1024d, cosine**.

**Embedding gotcha (asymmetric):** ingest embeds with Voyage `input_type="document"`; **retrieval must embed the query with `input_type="query"`** or relevance degrades. This is a strong "I understand asymmetric embeddings" talking point — check the retrieval repo actually does this.

## 4. Three retrieval modes to expect / verify in the repo

1. **Local / vector** — embed query → `chunk_embeddings` top-k → return chunks *with* Article/Chapter provenance.
2. **Graph-augmented** — from retrieved chunks/entities, traverse semantic relationships (ADDRESSES, REQUIRES, APPLIES_TO…) to pull connected context a flat vector search misses (multi-hop). In `neo4j_graphrag` this is typically a `VectorCypherRetriever` with a retrieval-query Cypher.
3. **Global / community** — vector-search `community_summary_embeddings`, answer from `Community` summaries (Microsoft-GraphRAG "global search" for thematic questions).

Plus possibly **hybrid** (vector + `chunk_text_fulltext` BM25) via `HybridRetriever` / `HybridCypherRetriever`, and **Text2Cypher** for structured questions.

## 5. What to extract/produce in the new session

Find and read (likely in the retrieval repo): retriever classes, the retrieval-query Cypher, the query-embedding call, the prompt-assembly/context-formatting code, similarity-score handling, and the chat/UI orchestration. Then produce concrete talking points covering:
- **Which `neo4j_graphrag` retriever(s)** are used (`VectorRetriever`, `VectorCypherRetriever`, `HybridRetriever`, `HybridCypherRetriever`, `Text2CypherRetriever`, `GraphRAG`) and why.
- **The actual Cypher** behind graph-augmented retrieval (the multi-hop expansion) — this is the money shot for "why graph."
- **Similarity scoring**: cosine, how top-k / thresholds are chosen, whether scores are surfaced/reranked, hybrid score fusion if present.
- **Prompt assembly**: how retrieved chunks + graph context + community summaries are formatted into the LLM prompt, and how provenance/citations are returned.
- **The closing argument**: vector-only RAG gives one retrieval mode; this graph gives local + graph-augmented + global, all in one Neo4j instance, with provenance.

Deliver as an addendum that mirrors the structure of `NEO4J_INTERVIEW_PREP.md` Section 8, so the two docs read as one guide.

---

## 6. Kickoff Prompt (paste this into the new session)

> I'm interviewing for a **Neo4j Sales Engineer** role and have a technical evaluation. I built a two-repo GraphRAG project: a construction/ingestion repo (`graphrag-api-db`) and **this repo**, the Retrieval & Chat UI. In a prior session we produced an interview prep guide for the construction side (committed as `NEO4J_INTERVIEW_PREP.md` on PR #82 of `graphrag-api-db`); its Section 8 sketches the retrieval story but without real code.
>
> Your job this session: review **this** Retrieval & Chat UI repo and produce concrete, source-grounded interview talking points (file:line references) for the retrieval/query layer, written to mirror that prep guide so the two read as one. Audience leans **sales-positioning grounded in Neo4j product detail** — always connect choices back to why Neo4j / why GraphRAG beats naive vector RAG. For each point: a spoken one-liner, deeper "if pushed" detail, and the likely interviewer question.
>
> Specifically dig into and explain: (1) which `neo4j_graphrag` retriever(s) are used and why; (2) the actual retrieval Cypher behind graph-augmented/multi-hop retrieval; (3) similarity scoring — cosine, top-k/threshold selection, score surfacing/reranking, hybrid fusion; (4) prompt assembly — how chunks + graph context + community summaries become the LLM prompt, and how provenance/citations are returned; (5) the three retrieval modes (local/vector, graph-augmented, global/community) and any hybrid (vector + BM25) or Text2Cypher paths.
>
> **The graph this app queries (built by the other repo), so you can connect retrieval to construction:**
> - Lexical: `(Chunk)-[:FROM_ARTICLE]->(Article)`, `(Entity)-[:MENTIONED_IN]->(Chunk)`. Chunk has `.text`, `.embedding`.
> - Domain: 12 entity types labeled `:__Entity__:__KGBuilder__`; `.name` lowercased, `.display_name` original; 14 semantic rel types (ADDRESSES, REQUIRES, COMPONENT_OF, APPLIES_TO, PUBLISHES, REGULATES, DEVELOPS, ACHIEVES…).
> - Community: `(Entity)-[:IN_COMMUNITY]->(Community)`; Community has `.summary`, `.summary_embedding`.
> - Indexes: `chunk_embeddings` (vector, cosine, on `Chunk.embedding`), `chunk_text_fulltext` (BM25 full-text on `Chunk.text`), `community_summary_embeddings` (vector, 1024d, cosine, on `Community.summary_embedding`).
> - Embeddings: Voyage `voyage-4`, **asymmetric** — ingest uses `input_type="document"`, so retrieval must embed the query with `input_type="query"`. Verify the index dimension matches the query-embedding dimension (ingest uses 1024d; confirm the live index isn't the 1536 default).
>
> Start by mapping the repo and locating the retriever + Cypher + prompt-assembly code, then write the addendum.

