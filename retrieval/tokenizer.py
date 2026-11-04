"""Identifier-aware tokenization for BM25 over source code.

A plain whitespace/word tokenizer treats `resolve_redirects` and `HTTPAdapter`
as single opaque tokens, so a natural-language query like "how are redirects
resolved" or "http adapter" never matches them. We emit BOTH the full
identifier (exact-match signal for queries that name it) AND its sub-words
(fuzzy signal for queries that describe it):

    "Session.resolve_redirects"  -> session, resolve_redirects, resolv, redirect
    "HTTPAdapter"                -> httpadapter, http, adapt
    "getProxies"                 -> getproxies, proxi       ("get" is a stopword)

Sub-words are Snowball-stemmed so "cookies"/"cookie" and "redirects"/"redirect"
match; the full identifier is kept unstemmed so exact-name queries still win.
"""

import re

import snowballstemmer

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
# Splits camelCase / PascalCase, keeping acronyms together: HTTPAdapter -> HTTP, Adapter
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# English filler words plus Python keywords/boilerplate that appear in nearly
# every chunk and would only add noise.
STOPWORDS = frozenset("""
a an the and or of to in on for with by is are was were be been it its this that
these those as at from how what where which who why when does do did can could
should would will there their then than so if else not no yes into via use used
using about get gets set sets
self cls def return none true false class import pass elif try except finally
raise lambda yield while break continue global nonlocal assert del async await
""".split())


def split_identifier(ident: str) -> list[str]:
    parts = []
    for piece in ident.split("_"):
        parts.extend(_CAMEL.findall(piece))
    return [p.lower() for p in parts if p]


_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        full = ident.lower().strip("_")
        parts = split_identifier(ident)
        if len(parts) > 1 and full:
            tokens.append(full)          # exact identifier, e.g. "resolve_redirects"
        tokens.extend(_stemmer.stemWords([p for p in parts if p not in STOPWORDS and len(p) > 1]))
    return tokens
