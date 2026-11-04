# Agentic RAG Codebase Assistant

Ask questions about a Python codebase in plain English and get answers with file-and-line citations. Everything runs locally on small open models (Qwen3-4B, Gemma 3 4B) served by Ollama.

```
$ uv run python -m agent.cli "Where does the default User-Agent string come from?"
  · retrieve('Where does the default User-Agent string come from?') -> 8 chunks
  · grade_documents -> 2/8 new chunks relevant
  · generate
  · check_grounding -> grounded

The default User-Agent string comes from `default_user_agent()`, which returns
"python-requests/{__version__}" [src/requests/utils.py:942-948]. It is used as the
"User-Agent" header in `default_headers()` [src/requests/utils.py:951-962].
```

The indexed codebase is [`psf/requests`](https://github.com/psf/requests), cloned into `data/requests`. You can point it at any Python repo.

## How it works

```
 ingest/                      retrieval/                         agent/ (LangGraph)
 ───────                      ──────────                         ──────────────────
 .py files                    query ──┬─> ChromaDB (dense) ─┐     retrieve
   │ AST chunker                      │                     ├─RRF─> grade_documents ──(none)──> rewrite_query ─┐
   ▼ (function / class /              └─> BM25 (identifier- ┘       │                         ^              │
     method / module)                      aware tokens)            ▼                         │              │
 chunks + metadata ──> nomic-embed-text ──> ChromaDB              generate <─(retry)─┐        │      back to retrieve
                                                                    ▼                │        │
                                                                  check_grounding ───┴────────┘
                                                                    │ (citations valid + LLM judge)
                                                                    ▼
                                                                  answer with [file:line-line] citations
```

1. **AST chunking** ([ingest/chunker.py](ingest/chunker.py)) splits code at function, class and method boundaries. Each chunk keeps its file path, class name, function name and line range as metadata.
2. **Hybrid retrieval** ([retrieval/hybrid.py](retrieval/hybrid.py)) runs dense search in ChromaDB and BM25 side by side, then fuses the two rankings with Reciprocal Rank Fusion. BM25 uses a tokenizer that understands identifiers ([retrieval/tokenizer.py](retrieval/tokenizer.py)), so it matches `resolve_redirects` against "how are redirects resolved".
3. **Self-corrective agent** ([agent/graph.py](agent/graph.py)) is a LangGraph state machine. An LLM grades each retrieved chunk for relevance, and the agent rewrites the query when retrieval fails. Each answer then goes through a deterministic citation check and an LLM grounding judge. Explicit counters and LangGraph's `recursion_limit` keep every loop bounded.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
ollama pull nomic-embed-text qwen3:4b-instruct gemma3:4b
git clone --depth 1 https://github.com/psf/requests data/requests
uv sync
uv run python -m ingest.index          # chunk + embed into ./chroma_db (~40s)
```

## Usage

```bash
uv run python -m agent.cli "How does requests handle redirects?"
uv run python -m agent.cli --model gemma3:4b "How is basic auth encoded?"
uv run python -m retrieval.hybrid "proxy bypass"     # compare dense / bm25 / hybrid
uv run python -m eval.evaluate -v                    # retrieval metrics
```

## Retrieval evaluation

20 hand-labeled queries ([eval/queries.json](eval/queries.json)), each with 2 to 6 relevant chunks out of 277. Results at k=10:

| Retriever       | P@10  | R@10  | Hit@10 | MRR   |
|-----------------|-------|-------|--------|-------|
| Dense only      | 0.245 | 0.645 | 0.90   | 0.779 |
| BM25 only       | 0.295 | 0.750 | 1.00   | 0.908 |
| **Hybrid (RRF)**| 0.270 | 0.700 | 0.95   | 0.854 |

With about 4 relevant chunks per query, the best possible P@10 is 0.39. Hybrid beats dense-only on every metric. On this query set BM25 alone does best, because the queries share vocabulary with the code. See [docs/GUIDE.md](docs/GUIDE.md#9-evaluation) for the analysis.

## Layout

```
ingest/chunker.py     AST-based chunking
ingest/index.py       embeddings + ChromaDB index
retrieval/tokenizer.py  identifier-aware tokenizer for BM25
retrieval/bm25.py     in-memory BM25
retrieval/hybrid.py   dense + BM25 + Reciprocal Rank Fusion
agent/graph.py        LangGraph self-corrective workflow
agent/prompts.py      prompts for grading, rewriting, answering, judging
agent/llm.py          Ollama model factory
agent/cli.py          command-line entry point
eval/                 labeled queries + metrics
docs/GUIDE.md         design notes and every decision explained
```
