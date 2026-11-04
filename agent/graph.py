"""Self-corrective agentic RAG as a LangGraph state machine.

    START -> retrieve -> grade_documents --(have relevant)--> generate -> check_grounding --(grounded)--> END
                ^               |                                  ^             |
                |         (none relevant,                          +-------------+ (not grounded,
                |          rewrites left)                                        |  regeneration left)
                +-------- rewrite_query <----------------------------------------+ (not grounded,
                                                                                    rewrites left)
              grade_documents --(none relevant, no rewrites left)--> no_answer -> END

Self-correction happens in two places:
  1. Retrieval side (Corrective RAG): an LLM grades each retrieved chunk; if
     nothing relevant came back, the query is rewritten and retrieval re-runs.
  2. Generation side (Self-RAG style): the answer's citations are validated
     deterministically, then an LLM judges whether the answer is grounded in
     the chunks. A failed check is first blamed on the generator (regenerate
     with the judge's feedback); if it fails again, the context itself is
     probably insufficient, so the agent goes back to rewrite + retrieve.
     Relevant chunks accumulate across retrieval rounds, so a rewrite can
     only add context, never lose what was already found.

Loops are bounded twice: explicit counters in the state (MAX_REWRITES,
MAX_REGENERATIONS) decide the routing, and LangGraph's recursion_limit is a
hard backstop that aborts the run if a bug ever made the graph cycle forever.
"""

import operator
import re
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agent import prompts
from retrieval.hybrid import HybridRetriever

TOP_K = 8
MAX_REWRITES = 2
MAX_REGENERATIONS = 1
RECURSION_LIMIT = 25   # worst legit path is 16 node steps; this only trips on a bug

CITATION = re.compile(r"\[([\w./-]+\.py):(\d+)-(\d+)\]")


class AgentState(TypedDict):
    question: str                     # original user question (never changes)
    query: str                        # current search query (rewritten on failure)
    candidates: list[Document]        # raw output of the latest retrieval
    documents: list[Document]         # chunks graded relevant, accumulated across rounds
    answer: str
    grounded: bool
    feedback: str                     # why the last answer failed the grounding check
    rewrites: int
    regenerations: int
    trace: Annotated[list[str], operator.add]   # reducer: each node appends, never overwrites


# Verdicts are parsed from plain text rather than requested as JSON via
# with_structured_output(). Ollama enforces JSON with grammar-constrained
# sampling, which decoded ~10 tok/s here vs ~45 tok/s unconstrained: a
# one-word "yes" grade takes ~0.3s, a JSON grade with a reason ~14s.
def says_yes(text: str) -> bool:
    """Lenient yes/no parse: small models add punctuation, markdown, prefixes."""
    m = re.search(r"\b(yes|no)\b", text.lower())
    return bool(m and m.group(1) == "yes")


def parse_verdict(text: str) -> tuple[bool, str]:
    verdict = re.search(r"verdict\W*(yes|no)", text, re.I)
    reason = re.search(r"reason\W*(.+)", text, re.I)
    ok = verdict.group(1).lower() == "yes" if verdict else says_yes(text)
    return ok, (reason.group(1).strip() if reason else text.strip()[:200])


def citation_tag(doc: Document) -> str:
    m = doc.metadata
    return f"[{m['file_path']}:{m['start_line']}-{m['end_line']}]"


def format_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        m = d.metadata
        name = ".".join(x for x in (m["class_name"], m["function_name"]) if x) or m["chunk_type"]
        parts.append(f"{citation_tag(d)} ({name})\n```python\n{m['code']}\n```")
    return "\n\n".join(parts)


def invalid_citations(answer: str, docs: list[Document]) -> list[str]:
    """Citations that don't point inside a chunk we actually gave the model.
    Cheap and deterministic: catches hallucinated files / line numbers
    before we spend an LLM call on the grounding judge."""
    bad = []
    for path, s, e in CITATION.findall(answer):
        s, e = int(s), int(e)
        if not any(d.metadata["file_path"] == path
                   and s >= d.metadata["start_line"] and e <= d.metadata["end_line"]
                   for d in docs):
            bad.append(f"[{path}:{s}-{e}]")
    return bad


def build_graph(retriever: HybridRetriever, llm: BaseChatModel):
    # ---- nodes: each takes the state and returns a partial update ----------

    def retrieve(state: AgentState) -> dict:
        docs = retriever.search(state["query"], k=TOP_K)
        return {"candidates": docs, "trace": [f"retrieve({state['query']!r}) -> {len(docs)} chunks"]}

    def grade_documents(state: AgentState) -> dict:
        have = {d.metadata["id"] for d in state["documents"]}
        docs = [d for d in state["candidates"] if d.metadata["id"] not in have]  # don't re-grade
        prompts_ = [
            prompts.GRADE.format(question=state["question"], source=citation_tag(d), code=d.metadata["code"])
            for d in docs
        ]
        # Sequential on purpose: a local Ollama server on a laptop has no spare
        # compute, so concurrent requests just contend and were slower overall.
        grades = [says_yes(m.content) for m in llm.batch(prompts_, config={"max_concurrency": 1})]
        kept = [d for d, ok in zip(docs, grades) if ok]
        return {"documents": state["documents"] + kept,
                "trace": [f"grade_documents -> {len(kept)}/{len(docs)} new chunks relevant"]}

    def rewrite_query(state: AgentState) -> dict:
        out = llm.invoke(prompts.REWRITE.format(
            question=state["question"], query=state["query"],
            found="\n".join(f"- {d.metadata['id']}" for d in state["candidates"]) or "- (nothing)",
        )).content
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        query = (lines[0] if lines else state["question"]).strip().strip("`\"'")
        # Clear the old answer/feedback: they were about the old context.
        return {"query": query, "rewrites": state["rewrites"] + 1, "answer": "", "feedback": "",
                "trace": [f"rewrite_query -> {query!r}"]}

    def generate(state: AgentState) -> dict:
        feedback = (f"\nYour previous answer was rejected: {state['feedback']}\n"
                    "Fix this: only state what the chunks show, and cite correctly.\n"
                    if state["feedback"] else "")
        msg = llm.invoke(prompts.GENERATE.format(
            context=format_context(state["documents"]), question=state["question"], feedback=feedback,
        ))
        n = state["regenerations"] + (1 if state["answer"] else 0)
        # Safety net in case a thinking-model build is used: drop its reasoning.
        answer = msg.content.split("</think>")[-1].strip()
        return {"answer": answer, "regenerations": n,
                "trace": ["generate" + (" (retry with judge feedback)" if feedback else "")]}

    def check_grounding(state: AgentState) -> dict:
        answer, docs = state["answer"], state["documents"]
        if not CITATION.search(answer):
            ok, why = False, "The answer contains no citations."
        elif bad := invalid_citations(answer, docs):
            ok, why = False, f"These citations do not match any provided chunk: {', '.join(bad)}."
        else:
            out = llm.invoke(prompts.GROUNDING.format(context=format_context(docs), answer=answer))
            ok, why = parse_verdict(out.content)
        return {"grounded": ok, "feedback": "" if ok else why,
                "trace": [f"check_grounding -> {'grounded' if ok else 'NOT grounded: ' + why}"]}

    def no_answer(state: AgentState) -> dict:
        return {"answer": "I couldn't find code in the repository relevant to this question.",
                "grounded": True, "trace": ["no_answer"]}

    # ---- conditional edges: pure functions of the state -> next node name --

    def after_grading(state: AgentState) -> str:
        if state["documents"]:
            return "generate"
        return "rewrite_query" if state["rewrites"] < MAX_REWRITES else "no_answer"

    def after_grounding(state: AgentState) -> str:
        if state["grounded"]:
            return END
        if state["regenerations"] < MAX_REGENERATIONS:
            return "generate"        # maybe the generator slipped: retry with feedback
        if state["rewrites"] < MAX_REWRITES:
            return "rewrite_query"   # still failing: the context is likely insufficient
        return END

    g = StateGraph(AgentState)
    for node in (retrieve, grade_documents, rewrite_query, generate, check_grounding, no_answer):
        g.add_node(node.__name__, node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", after_grading, ["generate", "rewrite_query", "no_answer"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges("check_grounding", after_grounding, ["generate", "rewrite_query", END])
    g.add_edge("no_answer", END)
    return g.compile()


def initial_state(question: str) -> AgentState:
    return AgentState(question=question, query=question, candidates=[], documents=[], answer="",
                      grounded=False, feedback="", rewrites=0, regenerations=0, trace=[])
