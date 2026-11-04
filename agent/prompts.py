"""Prompts for each LLM step. Kept short and literal: 4B models follow concrete
instructions far better than nuanced ones."""

GRADE = """You are checking whether a code chunk helps answer a question about the `requests` Python library.

Question: {question}

Code chunk ({source}):
```python
{code}
```

A chunk is relevant if it contains code, a docstring, or definitions that would help answer the question, even partially.
It is NOT relevant if it only shares a keyword with the question.

Answer with exactly one word: yes or no."""

REWRITE = """You rewrite questions into better search queries for a code search engine over the `requests` Python library.

Original question: {question}
Previous query: {query}
That query retrieved these chunks, which were not enough to answer the question:
{found}

Write ONE new search query. Include likely Python identifiers (function, class, or method names, e.g. `resolve_redirects`, `HTTPAdapter`) and technical keywords. Do not repeat the previous query. Reply with the query only, on one line."""

GENERATE = """You answer questions about the `requests` Python library using ONLY the code chunks below.

Rules:
- Base every claim on the chunks. If they don't contain the answer, say so.
- Cite the chunk for each claim using its exact citation tag, e.g. [src/requests/sessions.py:752-829].
- Refer to functions and classes by name. Be concise: a short explanation, then the key steps.
{feedback}
Code chunks:
{context}

Question: {question}

Answer:"""

GROUNDING = """You are verifying an answer about the `requests` Python library.

Code chunks the answer was supposed to use:
{context}

Answer:
{answer}

Is every factual claim in the answer supported by the code chunks above? Ignore style. An answer that claims something the chunks do not show is NOT grounded.

Reply in exactly this format:
VERDICT: yes or no
REASON: one sentence naming any unsupported claim"""
