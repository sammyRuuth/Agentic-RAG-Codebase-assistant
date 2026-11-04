# How This Project Works: The Complete Guide

This guide explains every part of the system: what each piece does, why it was built that way, what went wrong along the way, and how to talk about it in an interview. Read it alongside the code. Every section points to the file it describes.

---

## 1. The problem

Developers ask questions like *"how does this library decide to drop my auth header on a redirect?"* You can't answer that by keyword search alone, and pasting the whole repository into an LLM doesn't fit in the context window. A small local model (4B parameters) also can't be trusted to answer from memory.

**Retrieval-Augmented Generation (RAG)** solves this in two steps:
1. **Retrieve** the few pieces of code relevant to the question.
2. **Generate** an answer from *only* those pieces, citing them.

A plain RAG pipeline runs those two steps once, in a straight line. If retrieval returns the wrong code, the model answers from the wrong code. **Agentic, self-corrective RAG** adds checkpoints where the system judges its own intermediate results and loops back to fix them:

- Are these chunks actually relevant? If not, rewrite the query and search again.
- Is the answer actually supported by the chunks? If not, regenerate it, or go find better chunks.

## 2. Architecture and data flow

```
OFFLINE (once)                                   ONLINE (per question)
──────────────                                   ─────────────────────
data/requests/src/**/*.py                        question
        │                                            │
        ▼  ingest/chunker.py                         ▼
  AST parse → 277 chunks                       ┌─ retrieve ──────────────────────────────┐
  (function/class/method/module)               │   dense (Chroma) top-30  ─┐             │
        │                                      │   BM25 (in-memory) top-30 ┴─ RRF → top-8│
        ▼  ingest/index.py                     └─────────────────────────────────────────┘
  "search_document: " + header + code              │
        │  nomic-embed-text (768-dim)               ▼
        ▼                                     grade_documents (LLM yes/no per chunk)
  ChromaDB (persisted in ./chroma_db)              │ none relevant → rewrite_query → retrieve
        │                                          ▼
        └── at startup, all docs are read      generate (answer with [file:start-end] citations)
            back to build BM25 in memory           │
                                                   ▼
                                              check_grounding
                                                1. citations exist?          (regex)
                                                2. citations point at chunks? (deterministic)
                                                3. claims supported?          (LLM judge)
                                                   │ fail once → generate again with feedback
                                                   │ fail twice → rewrite_query → retrieve
                                                   ▼
                                                answer + sources
```

| Folder | Responsibility |
|---|---|
| [ingest/](../ingest/) | Turn source files into chunks, embed them, store them |
| [retrieval/](../retrieval/) | Find the best chunks for a query (dense, BM25, fusion) |
| [agent/](../agent/) | The LangGraph workflow, prompts, model setup, CLI |
| [eval/](../eval/) | Labeled queries and retrieval metrics |

---

## 3. AST-based chunking ([ingest/chunker.py](../ingest/chunker.py))

### Why not fixed-size chunks?

Most RAG tutorials split text every N characters or tokens. On code, that cuts functions in half: one chunk gets the signature and docstring, the next gets the body. Neither half answers the question on its own, and the embedding of each half is muddled.

Code already has natural units: **functions, classes and methods**. Python's built-in `ast` module parses a file into a tree, and each node knows its exact `lineno` and `end_lineno`. We cut along those boundaries.

### The chunking strategy

| Source construct | Chunks produced | Why |
|---|---|---|
| Top-level function | 1 `function` chunk | A self-contained unit |
| Small class (≤ 80 lines) | 1 `class` chunk with all its methods | Small classes are easier to understand whole |
| Big class | 1 `class_header` chunk + 1 `method` chunk per method | One chunk for a 900-line `Session` class would be useless |
| Top-level statements (imports, constants) | Grouped into contiguous `module` chunks | Constants like `DEFAULT_REDIRECT_LIMIT` live here |
| Any chunk > 150 lines | Split into 150-line windows overlapping by 20 lines | Very long vectors average away meaning |

Result on `requests`: **277 chunks from 19 files** (74 functions, 129 methods, 36 classes, 10 class headers, 28 module chunks). Median 13 lines, max 146.

### Details that matter (and make good interview answers)

- **Class header as a table of contents.** For a big class, the `class_header` chunk contains the `class X(...):` line, the docstring, the class attributes, and a generated list of every method signature. This chunk answers "what is `Session` and what can it do?" without loading 900 lines.
- **Decorators are included.** `_start_line()` takes the smallest line number among the node and its decorators, so `@property` stays attached to its method.
- **`@overload` stubs are merged.** Type-hint overload stubs are folded into the real implementation, so you get one chunk per real function, not three near-duplicates.
- **Property setters get unique IDs.** `@property def x` and `@x.setter def x` share a name, so the second one gets an `@L<line>` suffix. The `__main__` block asserts that IDs are unique.
- **Syntax errors don't drop files.** An unparseable file becomes one `<unparsed>` chunk instead of disappearing.
- **Stable, readable IDs.** For example, `src/requests/sessions.py::Session.send`. The eval labels refer to these IDs directly.

### The context header

Before embedding, each chunk gets a header:

```
# File: src/requests/sessions.py
# Type: method
# Class: Session
# Function: send
```

**Why:** there are several methods named `send` in `requests` (`Session.send`, `HTTPAdapter.send`, `BaseAdapter.send`). Their bodies look alike to an embedding model. The header tells the model *where* the code lives, and it also gives BM25 the file, class and function names as searchable tokens.

Two text fields per chunk:
- `text` = header + code. This is what gets **embedded and indexed**.
- `code` = raw code. This is what the **LLM sees and cites**. The header is synthetic, so the model is never shown it as if it were source.

---

## 4. Embeddings and ChromaDB ([ingest/index.py](../ingest/index.py))

### What an embedding is

An embedding model maps text to a vector (768 numbers for `nomic-embed-text`) so that texts with similar meaning end up close together. "How are redirects followed" lands near the code of `resolve_redirects` even though they share few exact words. That's **dense retrieval**: embed the query, return the nearest chunk vectors.

### Decisions

- **`nomic-embed-text` via Ollama.** Small (274 MB), local, good general-purpose quality.
- **Task prefixes.** Nomic was *trained* with `search_document: ` on documents and `search_query: ` on queries. Without them, queries and documents land in slightly different regions of the space. LangChain's `OllamaEmbeddings` doesn't add them, so `NomicEmbeddings` subclasses it and overrides `embed_documents` and `embed_query`. This kind of detail is easy to miss.
- **Cosine distance** (`hnsw:space: cosine`) instead of Chroma's default L2. Cosine compares vector direction (meaning) and ignores magnitude.
- **HNSW** (Hierarchical Navigable Small World) is the approximate nearest-neighbor index Chroma uses internally. With 277 chunks, brute force would also be instant. HNSW matters at millions of vectors.
- **Truncation check.** The longest chunk is about 1,091 tokens, under the 2,048-token context Ollama gives the embedder, so nothing gets silently truncated.
- **Chroma is the single source of truth.** BM25 is rebuilt at startup from `store.get()`, so the two retrievers always see the same corpus. You can't end up with a stale BM25 index that disagrees with the vector store.
- **Rebuild from scratch.** `build_index()` deletes `chroma_db/` first. Incremental updates (re-embedding only changed files) would be a natural next step.

---

## 5. BM25 and identifier-aware tokenization

### BM25 ([retrieval/bm25.py](../retrieval/bm25.py))

BM25 is the classic keyword ranking function used by search engines. For a query with terms *q₁…qₙ*, a document's score is:

```
score(D) = Σ  IDF(qᵢ) · f(qᵢ,D)·(k1+1) / ( f(qᵢ,D) + k1·(1 − b + b·|D|/avgdl) )
```

- **f(q, D)**: how often the term appears in the chunk. More is better, but with **diminishing returns** (controlled by `k1`, default 1.5). The 10th occurrence adds far less than the 1st.
- **IDF**: rare terms count more. `should_strip_auth` is rare and very informative. `request` appears everywhere and is nearly worthless.
- **Length normalization** (`b`, default 0.75): long chunks naturally contain more words, so they're penalized a bit.

It's "in-memory" because `rank_bm25` keeps the token lists and statistics in Python memory, which takes milliseconds for 277 chunks. Dense search finds meaning, BM25 finds exact names, and code questions need both.

### Why a special tokenizer ([retrieval/tokenizer.py](../retrieval/tokenizer.py))

A normal tokenizer turns `resolve_redirects` into one token. The query "how are redirects resolved" contains `redirects` and `resolved`, so it gets zero matches. The tokenizer emits **both** forms:

```
"Session.resolve_redirects" → session, resolve_redirects, resolv, redirect
"HTTPAdapter"               → httpadapter, http, adapt
"getProxies"                → getproxies, proxi
```

1. **The full identifier, unstemmed.** If the user types the exact name, it gets a strong, rare-term match.
2. **Sub-words.** Split on `_` and camelCase (`HTTPAdapter` → `HTTP` + `Adapter`, with acronyms kept together). These match natural-language descriptions.
3. **Stopwords removed.** English filler plus Python boilerplate (`self`, `def`, `return`, `none`), which appear in almost every chunk.
4. **Snowball stemming on sub-words.** `cookies`/`cookie` → `cooki`, and `redirects`/`redirect` → `redirect`.

### A bug worth telling in an interview

The first version used a hand-written plural rule: `-ies → -y`, otherwise strip `-s`. That turned `cookies` into `cooky` and left `cookie` unchanged, so **queries about cookies could never match code about a cookie**. I found it while debugging the worst query in the eval set. Switching to the Snowball stemmer (applied only to sub-words, never to full identifiers) fixed it: BM25 Recall@10 went from 0.723 to 0.750 and MRR from 0.793 to 0.908. The lesson: hand-rolled NLP shortcuts break on edge cases, and per-query failure analysis catches bugs that aggregate metrics hide.

---

## 6. Hybrid retrieval with Reciprocal Rank Fusion ([retrieval/hybrid.py](../retrieval/hybrid.py))

### Why combine

| Query | Dense | BM25 |
|---|---|---|
| "how does it retry failed connections" (no identifiers) | ✅ matches meaning | ❌ no shared words |
| "what does `should_strip_auth` do" | ⚠️ fuzzy on names | ✅ exact rare token |
| "how are query params **encoded**" | ❌ confused it with *character* encoding | ✅ found `prepare_url` |

They fail on *different* queries, so combining them is more robust than either one alone.

### Why RRF instead of adding scores

Cosine similarities fall around 0.4 to 0.8, and BM25 scores around 0 to 30. Adding them lets BM25 dominate, and normalizing them is fragile. **Reciprocal Rank Fusion** ignores the raw scores entirely and uses only each document's *rank*:

```
RRF(d) = Σ_retrievers  1 / (k + rank(d))      with k = 60
```

Worked example with k = 60:
- Chunk A: dense rank 1, missing from BM25 → 1/61 = **0.0164**
- Chunk B: dense rank 3, BM25 rank 2 → 1/63 + 1/62 = **0.0320**

B wins. A chunk that **both** retrievers like beats a chunk that only one retriever ranks first. `k` damps the advantage of the top ranks. 60 comes from the original paper (Cormack et al., 2009) and works well without tuning.

`fetch_k = 30`: each retriever contributes its top 30 candidates before fusion, then we keep the top 8 (agent) or top 10 (eval). Fusing deeper lists lets a chunk ranked #15 by both retrievers rise to the top.

---

## 7. The LangGraph agent ([agent/graph.py](../agent/graph.py))

### LangGraph concepts in five lines

- **State**: a typed dict (`AgentState`) passed through the graph.
- **Nodes**: Python functions that take the state and return a *partial update*.
- **Edges**: fixed transitions (`retrieve → grade_documents`).
- **Conditional edges**: functions that look at the state and return the *name* of the next node. This is how loops and branches work.
- **Reducers**: how an update is merged into the state. By default it overwrites. `trace: Annotated[list[str], operator.add]` *appends*, so every node adds a line to the trace without erasing earlier ones.

### The state

| Field | Meaning |
|---|---|
| `question` | Original question. Never changes. Grading and answering always use it. |
| `query` | Current search query. Changed by `rewrite_query`. |
| `candidates` | Raw output of the latest retrieval |
| `documents` | Chunks graded relevant, **accumulated across rounds** |
| `answer`, `grounded`, `feedback` | Latest answer, whether it passed, and why it failed |
| `rewrites`, `regenerations` | Loop counters |
| `trace` | Human-readable log of each step |

### The nodes

1. **`retrieve`**: hybrid search with the current `query`, top 8.
2. **`grade_documents`**: for each *new* candidate, ask the LLM "is this chunk relevant? yes/no". Chunks already accepted aren't re-graded. This is the **Corrective RAG (CRAG)** idea: filter noise before generation.
3. **`rewrite_query`**: tell the LLM the original question, the previous query, and which chunks came back, then ask for a better query that includes likely identifier names. This bridges the vocabulary gap ("saved" vs `extract_cookies_to_jar`).
4. **`generate`**: answer from the relevant chunks only, citing them with exact tags like `[src/requests/sessions.py:154-184]`. On a retry, the judge's feedback is inserted into the prompt.
5. **`check_grounding`**, the **Self-RAG** idea, with three checks from cheapest to most expensive:
   - Are there any citations at all? (regex)
   - Does every citation point *inside* a chunk the model was given? (`invalid_citations`, deterministic, catches made-up files and line numbers at zero LLM cost)
   - Is every claim supported? (LLM judge returns `VERDICT: yes/no` and `REASON: …`)
6. **`no_answer`**: an honest "couldn't find it", reached when nothing relevant is found after every rewrite.

### The routing (where self-correction lives)

```python
after_grading:   have relevant docs → generate
                 none, rewrites left → rewrite_query
                 none, no rewrites   → no_answer

after_grounding: grounded                 → END
                 regeneration left        → generate        # maybe the generator slipped
                 rewrites left            → rewrite_query   # the context is probably insufficient
                 otherwise                → END (with a ⚠ warning)
```

**Why two different recoveries?** A failed grounding check has two possible causes. If the *generator* made something up, regenerating with feedback can fix it. If the *context* lacks the answer, no amount of regenerating helps, and the agent needs to search again. The first design only regenerated. On "How are cookies set by a server response saved into the session?" it kept regenerating from a single weak chunk (`RequestsCookieJar.set`). After the redesign, the second rewrite pulled in `extract_cookies_to_jar`, which is the right function.

**Why accumulate documents?** If a rewrite finds new relevant chunks, the ones already found shouldn't be thrown away. Accumulating means a rewrite can only add context, never lose it.

### Bounded loops: two layers

1. **Explicit counters** (`MAX_REWRITES = 2`, `MAX_REGENERATIONS = 1`) drive the routing. The worst legitimate path is 16 node steps.
2. **`recursion_limit = 25`** is LangGraph's hard cap on steps per run. It should never trigger. If a routing bug ever created an infinite loop, it raises `GraphRecursionError` (caught in the CLI) instead of hanging forever. Counters are the policy, and the recursion limit is the safety net.

### Example trace (an unanswerable question)

```
· retrieve('How does the library compute the weather forecast?') -> 8 chunks
· grade_documents -> 0/8 relevant
· rewrite_query -> 'how does requests library handle weather data retrieval with session.get ...'
· retrieve(...) -> 8 chunks
· grade_documents -> 0/8 relevant
· rewrite_query -> ...
· retrieve(...) -> 8 chunks
· grade_documents -> 0/8 relevant
· no_answer
```

It searched, failed, tried twice more, and then **declined instead of making something up**. That's the behavior you want.

---

## 8. Running small models locally ([agent/llm.py](../agent/llm.py))

### The models

- **Qwen3-4B-Instruct-2507** (`qwen3:4b-instruct`): the default. Best grader in testing.
- **Gemma 3 4B** (`gemma3:4b`): switch with `--model gemma3:4b`. It grades more leniently (on the cookie query it called all 8 chunks relevant) but gives decent answers.
- Both use **Q4_K_M quantization**: weights stored at about 4 bits instead of 16, which brings a 4B model down to 2.5 to 3.4 GB so it fits on a 16 GB laptop with little quality loss.
- **`temperature=0`**: grading and judging should be deterministic.
- **`num_ctx=8192`**: Ollama's default context window is small, and 8 code chunks plus a prompt can exceed it. Overflow is silently truncated, which is a classic invisible bug.

### Three performance lessons (each one is a strong interview story)

**1. The "thinking" model trap.** Ollama's plain `qwen3:4b` tag is the *Thinking-2507* build. It always writes a long reasoning trace before answering, even with `reasoning=False`. The first full run took **4 min 39 s** and leaked `</think>` text into the answer. Running `ollama show qwen3:4b` revealed the thinking capability. Switching to the `qwen3:4b-instruct` build of the same model fixed both problems. A `split("</think>")` safety net stays in place.

**2. Structured output was 45× slower.** The first version used LangChain's `with_structured_output()` to get JSON like `{"reason": ..., "relevant": true}`. Ollama enforces JSON through **grammar-constrained decoding**, where every token is checked against a grammar. Ollama's own timing metadata showed:

| | prefill | decode | output tokens | total |
|---|---|---|---|---|
| JSON + reason | 0.3 s | 13.8 s | 140 | ~14 s |
| plain "yes"/"no" | 0.3 s | 0.05 s | 2 | **~0.3 s** |

On a test query, plain yes/no gave **the same verdicts** for all 8 chunks. So verdicts are now plain text parsed with lenient regexes (`says_yes`, `parse_verdict`), which tolerate `**Yes**.` and similar formatting. Lesson: measure where time goes (prefill vs decode) before guessing.

**3. Concurrency made things slower.** `llm.batch(..., max_concurrency=4)` took 205 s for 8 grades, versus about 9 s for one call at the time. A local Ollama server on one laptop GPU has no spare capacity, so parallel requests just compete for it. Grading now runs sequentially. On a GPU server with batching (vLLM), the opposite would be true.

Net effect: **about 4.5 min → 30 to 60 s per question** for typical queries (up to about 2 min when every correction loop runs), on an Apple Silicon Mac with 16 GB of memory.

### Known weakness: the small-model judge

The 4B grounding judge makes mistakes **in both directions**:
- **False pass**: an early answer said the default-port case *strips* the header when the code *keeps* it, and the judge passed it.
- **False fail**: a correct answer about `extract_cookies_to_jar` was rejected because the judge claimed the code "does not show `extract_cookies`", yet line 150 is `jar.extract_cookies(res, req)`.

That's why the deterministic citation check runs first. It's 100% reliable for what it checks. A stronger judge model, or majority voting over several judge calls, would improve this.

---

## 9. Evaluation

### Setup ([eval/queries.json](../eval/queries.json), [eval/evaluate.py](../eval/evaluate.py))

20 natural-language questions about `requests`, each labeled with the chunk IDs that are needed or useful to answer it (2 to 6 per query, 81 labels in total). Labels were written from the source code **before** running any retriever on these queries, so they aren't biased toward one method. **Review every label yourself.** You need to be able to defend each one.

### Metrics (averaged over queries)

- **Precision@10** = relevant results in the top 10 ÷ 10. "How much of what we returned is useful?"
- **Recall@10** = relevant results in the top 10 ÷ total relevant. "How much of what's useful did we find?"
- **Hit@10**: did at least one relevant chunk appear?
- **MRR** (Mean Reciprocal Rank): the average of 1 / (rank of the first relevant result). 1.0 means the first result was always relevant.

**The precision ceiling.** A query with 3 relevant chunks can score at most P@10 = 0.30, even when retrieval is perfect. With these labels, the **maximum achievable P@10 is 0.39**. That's why P@10 looks "low" and should always be read together with recall. The script prints this ceiling.

### Results

| Retriever | P@10 | R@10 | Hit@10 | MRR |
|---|---|---|---|---|
| Dense only | 0.245 | 0.645 | 0.90 | 0.779 |
| BM25 only | 0.295 | 0.750 | 1.00 | 0.908 |
| **Hybrid (RRF)** | 0.270 | 0.700 | 0.95 | 0.854 |

### How to interpret it honestly

1. **Hybrid beats dense-only on every metric**, which is the main claim.
2. **BM25 alone beats hybrid on this set.** The likely reason is that the queries were written by someone reading the code, so they reuse its vocabulary ("redirects", "proxies", "timeouts"). That's the best case for keyword search. Real users who don't know the codebase phrase things differently, and that's where dense search earns its place. Dense also solved queries BM25 struggled with (character encoding: dense 0.60 vs BM25 0.20 recall).
3. **Weighted RRF.** A quick sweep showed that down-weighting dense to 0.25 lifts hybrid to R@10 0.775. **I deliberately did not ship that**, because the weight was chosen by looking at the test set. With 20 queries, that's overfitting. The right approach is to tune on a separate development set and report on held-out queries.
4. **Limitations:** 20 queries is small (one query moves recall by 5 points), there's a single labeler, and there's one repository. This is a sanity check, not a benchmark.

---

## 10. Your resume bullets vs. reality

| Resume claim | Status |
|---|---|
| Self-corrective agentic RAG with LangGraph, file-level citations | ✅ Built. Citations include line ranges and are validated. |
| Local SLMs (Qwen3-4B, Gemma 3 4B) | ✅ Both work via `--model`. |
| AST chunking at function/class level with file/class/function metadata | ✅ |
| ChromaDB dense + in-memory BM25 + RRF + identifier-aware tokenization | ✅ |
| LLM relevance grading, query rewriting, grounding checks, recursion limits | ✅ |
| **40% Precision@10, 75% Recall@10, outperforming dense-only** | ❌ **Doesn't match.** Measured hybrid: **27% P@10, 70% R@10**. 40% P@10 is impossible with these labels (ceiling 39%). |

**Replace the last bullet with your real numbers.** Interviewers often ask "how did you measure that?", and you need to be able to rerun it on the spot. A suggested version:

> • Evaluated on 20 hand-labeled queries: hybrid retrieval reached 70% Recall@10 and 0.85 MRR vs. 65% / 0.78 for a dense-only baseline; cut per-query latency from ~4.5 min to under 1 min by replacing grammar-constrained JSON output with plain-text verdicts.

---

## 11. Interview questions you should be ready for

**Q: Why AST chunking instead of fixed-size?**
Fixed windows cut functions in half and produce incoherent embeddings. AST boundaries match how code is organized, give each chunk exact line ranges for citations, and supply structured metadata (class name, function name) for free.

**Q: How do you handle huge classes or functions?**
Big classes become a "table of contents" header chunk plus one chunk per method. Anything over 150 lines is split into overlapping windows, so context isn't lost at the boundaries.

**Q: Why hybrid retrieval?**
Dense finds meaning but is fuzzy on exact names. BM25 finds exact identifiers but has no idea about synonyms. They fail on different queries, so fusing them is more robust.

**Q: Explain RRF. Why k = 60?**
Score = Σ 1/(k + rank). It uses ranks only, so incompatible score scales don't matter. k damps the top-rank advantage so documents that both retrievers agree on rise to the top. 60 is the standard value from the original paper.

**Q: What's identifier-aware tokenization?**
Splitting `snake_case` and `camelCase` into sub-words, stemming those, and also keeping the full identifier. It matches both "resolve redirects" and `resolve_redirects`. Tell the cookie bug story here.

**Q: What makes this "agentic"?**
The LLM controls the flow: it decides chunk relevance, rewrites failed queries, and judges its own answers, and those decisions drive conditional edges in a state machine. It isn't a fixed pipeline.

**Q: How do you prevent infinite loops?**
State counters drive the routing, and LangGraph's `recursion_limit` is a hard backstop that raises `GraphRecursionError` instead of hanging.

**Q: How do you prevent hallucinated citations?**
Every citation is checked deterministically against the chunks provided, *before* the LLM judge runs. Made-up files or line numbers fail immediately.

**Q: What would you improve?**
A code-specific embedding model; a cross-encoder reranker after fusion; a stronger or majority-vote judge; incremental re-indexing; a larger eval set with a dev/test split for tuning the RRF weights; end-to-end answer-quality evaluation, not just retrieval; multi-language support with tree-sitter instead of Python's `ast`.

**Q: Why was BM25 better than hybrid in your eval?**
Query vocabulary bias (see §9). Be upfront about it. Explaining a surprising result well impresses interviewers more than a clean table does.

**Q: What was the hardest problem?**
Latency. Profiling with Ollama's timing metadata showed constrained JSON decoding was the bottleneck, not prefill. Switching to plain-text verdicts kept accuracy and cut grading from about 14 s to 0.3 s per chunk.

---

## 12. Things to try yourself (the best way to learn this)

1. `uv run python -m retrieval.hybrid "your query"` shows dense, BM25 and hybrid results side by side. Find queries where each one wins.
2. In `ingest/chunker.py`, set `SMALL_CLASS_LINES = 10_000` (never split classes), re-index, and re-run the eval. What happens to recall?
3. Remove the context header (embed `c.code` instead of `c.text`) and measure the effect.
4. Change `rrf_k` to 1 and to 1000 in `HybridRetriever` and watch how fusion behaves.
5. Write 10 *new* queries without looking at the code, the way a new user would, and label them. Does BM25 still win?
6. Set `MAX_REWRITES = 0` and ask the weather question. Then set `RECURSION_LIMIT = 5` and watch the backstop trigger.
7. Point it at another repo: clone it into `data/`, change `REPO_DIR` and the `include` glob, and re-index.
