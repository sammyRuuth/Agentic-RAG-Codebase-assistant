"""AST-based chunking: split Python source into function/class-level chunks.

Strategy
--------
- Top-level function            -> 1 chunk ("function")
- Small class (<= SMALL_CLASS_LINES) -> 1 chunk ("class"), methods stay inside
- Big class                     -> 1 "class_header" chunk (docstring, attributes,
                                   method signatures) + 1 chunk per method ("method")
- Top-level non-def statements  -> grouped into contiguous "module" chunks
                                   (imports, constants, module docstring)
- Any chunk > MAX_CHUNK_LINES   -> split into overlapping line windows
"""

import ast
from dataclasses import dataclass, asdict, field
from pathlib import Path

SMALL_CLASS_LINES = 80
MAX_CHUNK_LINES = 150
WINDOW_OVERLAP = 20

FuncNode = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Chunk:
    id: str                 # stable, human-readable: "src/requests/sessions.py::Session.send"
    file_path: str          # relative to repo root
    chunk_type: str         # module | function | class | class_header | method
    class_name: str         # "" if not inside a class
    function_name: str      # "" if not a function/method
    start_line: int         # 1-indexed, inclusive
    end_line: int
    code: str               # raw source slice (what the LLM sees and cites)
    text: str = field(default="")  # code + context header (what gets embedded / BM25-indexed)

    def to_metadata(self) -> dict:
        """Flat dict for ChromaDB metadata (values must be str/int/float/bool)."""
        d = asdict(self)
        d.pop("code")
        d.pop("text")
        return d


def _start_line(node: ast.AST) -> int:
    # Decorators sit above the `def` line; include them in the chunk.
    decorators = getattr(node, "decorator_list", [])
    return min([d.lineno for d in decorators] + [node.lineno])


def _is_overload(node: ast.AST) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in getattr(node, "decorator_list", [])
    )


def _merge_overloads(nodes: list[ast.stmt]) -> list[tuple[ast.stmt, int]]:
    """Pair each node with its chunk start line, folding `@overload` stubs into
    the implementation that follows them (one chunk per real function)."""
    out, pending = [], {}
    for node in nodes:
        if isinstance(node, FuncNode) and _is_overload(node):
            pending.setdefault(node.name, _start_line(node))
            continue
        start = pending.pop(getattr(node, "name", None), None) or _start_line(node)
        out.append((node, start))
    return out


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {fn.name}({ast.unparse(fn.args)})"


def _context_header(file_path: str, class_name: str, function_name: str, chunk_type: str) -> str:
    # Prepended before embedding so a method like `send` still "knows" it lives
    # in Session inside sessions.py. Without this, 5 different `send` methods
    # look nearly identical to the embedding model.
    lines = [f"# File: {file_path}", f"# Type: {chunk_type}"]
    if class_name:
        lines.append(f"# Class: {class_name}")
    if function_name:
        lines.append(f"# Function: {function_name}")
    return "\n".join(lines)


class _FileChunker:
    def __init__(self, source: str, file_path: str):
        self.lines = source.splitlines()
        self.file_path = file_path
        self.chunks: list[Chunk] = []

    def slice(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1 : end])

    def add(self, name: str, chunk_type: str, class_name: str, function_name: str,
            start: int, end: int, code: str | None = None) -> None:
        code = code if code is not None else self.slice(start, end)
        if not code.strip():
            return
        base_id = f"{self.file_path}::{name}"
        n_lines = end - start + 1

        # Very long chunks get split into overlapping windows so no single
        # vector has to represent 300 lines of logic.
        if n_lines > MAX_CHUNK_LINES:
            step = MAX_CHUNK_LINES - WINDOW_OVERLAP
            part = 1
            for w_start in range(start, end + 1, step):
                w_end = min(w_start + MAX_CHUNK_LINES - 1, end)
                self._append(f"{base_id}#part{part}", chunk_type, class_name,
                             function_name, w_start, w_end, self.slice(w_start, w_end))
                part += 1
                if w_end == end:
                    break
            return
        self._append(base_id, chunk_type, class_name, function_name, start, end, code)

    def _append(self, chunk_id, chunk_type, class_name, function_name, start, end, code):
        header = _context_header(self.file_path, class_name, function_name, chunk_type)
        self.chunks.append(Chunk(
            id=chunk_id, file_path=self.file_path, chunk_type=chunk_type,
            class_name=class_name, function_name=function_name,
            start_line=start, end_line=end, code=code, text=f"{header}\n\n{code}",
        ))

    def run(self, tree: ast.Module) -> list[Chunk]:
        module_run: list[ast.stmt] = []  # contiguous non-def top-level statements

        def flush_module_run():
            if module_run:
                s, e = _start_line(module_run[0]), module_run[-1].end_lineno
                self.add(f"<module>@L{s}", "module", "", "", s, e)
                module_run.clear()

        for node, start in _merge_overloads(tree.body):
            if isinstance(node, FuncNode):
                flush_module_run()
                self.add(node.name, "function", "", node.name, start, node.end_lineno)
            elif isinstance(node, ast.ClassDef):
                flush_module_run()
                self._chunk_class(node)
            else:
                module_run.append(node)
        flush_module_run()
        return self.chunks

    def _chunk_class(self, cls: ast.ClassDef) -> None:
        start, end = _start_line(cls), cls.end_lineno
        methods = [(n, s) for n, s in _merge_overloads(cls.body) if isinstance(n, FuncNode)]

        if end - start + 1 <= SMALL_CLASS_LINES or not methods:
            self.add(cls.name, "class", cls.name, "", start, end)
            return

        # Header = class line + docstring + class attributes (everything that
        # isn't a method), followed by a list of method signatures as a "table
        # of contents". This chunk answers "what is Session / what can it do?".
        header_parts = [self.slice(start, cls.lineno)]  # decorators + `class X(...):`
        for node in cls.body:
            if not isinstance(node, FuncNode):
                header_parts.append(self.slice(_start_line(node), node.end_lineno))
        header_parts.append("    # Methods:")
        header_parts += [f"    #   {_signature(m)}" for m, _ in methods]
        header_end = methods[0][1] - 1
        self.add(cls.name, "class_header", cls.name, "", start, max(start, header_end),
                 code="\n".join(header_parts))

        seen: dict[str, int] = {}
        for m, m_start in methods:
            # @property + @x.setter produce two methods with the same name.
            seen[m.name] = seen.get(m.name, 0) + 1
            name = f"{cls.name}.{m.name}" + (f"@L{m.lineno}" if seen[m.name] > 1 else "")
            self.add(name, "method", cls.name, m.name, m_start, m.end_lineno)


def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    source = path.read_text(encoding="utf-8")
    rel = path.relative_to(repo_root).as_posix()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: one chunk for an unparseable file rather than dropping it.
        c = _FileChunker(source, rel)
        c.add("<unparsed>", "module", "", "", 1, len(c.lines))
        return c.chunks
    return _FileChunker(source, rel).run(tree)


def chunk_repo(repo_root: Path, include: str = "src/**/*.py") -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(repo_root.glob(include)):
        chunks.extend(chunk_file(path, repo_root))
    return chunks


if __name__ == "__main__":
    from collections import Counter

    repo = Path(__file__).resolve().parent.parent / "data" / "requests"
    chunks = chunk_repo(repo)
    sizes = [c.end_line - c.start_line + 1 for c in chunks]

    print(f"{len(chunks)} chunks from {len({c.file_path for c in chunks})} files")
    print("by type:", dict(Counter(c.chunk_type for c in chunks)))
    print(f"lines/chunk: min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")
    assert len({c.id for c in chunks}) == len(chunks), "duplicate chunk ids!"

    print("\nLargest 5:")
    for c in sorted(chunks, key=lambda c: c.start_line - c.end_line)[:5]:
        print(f"  {c.end_line - c.start_line + 1:>4} lines  {c.id}")

    sample = next(c for c in chunks if c.id.endswith("Session.send"))
    print(f"\n--- sample chunk: {sample.id} (L{sample.start_line}-{sample.end_line}) ---")
    print("\n".join(sample.text.splitlines()[:14]), "\n    ...")
