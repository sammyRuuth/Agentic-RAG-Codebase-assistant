"""Ask a question about the indexed codebase.

    uv run python -m agent.cli "How does requests handle redirects?"
    uv run python -m agent.cli --model gemma3:4b "Where is the User-Agent set?"
"""

import argparse

from langgraph.errors import GraphRecursionError

from agent.graph import RECURSION_LIMIT, build_graph, citation_tag, initial_state
from agent.llm import DEFAULT_MODEL, SUPPORTED, get_llm
from retrieval.hybrid import HybridRetriever


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model, e.g. {' / '.join(SUPPORTED)}")
    args = p.parse_args()

    graph = build_graph(HybridRetriever(), get_llm(args.model))
    final = None
    try:
        # stream_mode="values" yields the full state after every node, so we
        # can print the trace live as the agent moves through the graph.
        seen = 0
        for state in graph.stream(initial_state(args.question),
                                  config={"recursion_limit": RECURSION_LIMIT}, stream_mode="values"):
            for line in state["trace"][seen:]:
                print(f"  · {line}")
            seen = len(state["trace"])
            final = state
    except GraphRecursionError:
        print(f"Aborted: exceeded recursion limit of {RECURSION_LIMIT} steps.")
        return

    print("\n" + final["answer"])
    if not final["grounded"]:
        print("\n⚠  Answer failed the grounding check after retries; verify it against the sources.")
    if final["documents"]:
        print("\nSources:")
        for d in final["documents"]:
            print(f"  {citation_tag(d)}  {d.metadata['id'].split('::')[1]}")


if __name__ == "__main__":
    main()
