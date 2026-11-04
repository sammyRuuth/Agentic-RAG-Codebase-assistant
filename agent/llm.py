"""Local SLM factory. Models are served by Ollama (`ollama serve`)."""

import os

from langchain_ollama import ChatOllama

# Ollama's plain `qwen3:4b` tag is the *Thinking*-2507 build, which always
# reasons (minutes per answer on a laptop) regardless of the think flag. The
# instruct build is the same 4B model without the reasoning phase.
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct")
SUPPORTED = ("qwen3:4b-instruct", "gemma3:4b")


def get_llm(model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or DEFAULT_MODEL,
        temperature=0,       # deterministic grading/answers; no creativity wanted
        num_ctx=8192,        # Ollama defaults to a small window; 8 code chunks need more
        # Qwen3 "thinks" (emits <think>...</think>) by default. That's slow on a
        # laptop and pollutes structured output; Gemma ignores the flag.
        reasoning=False,
    )
