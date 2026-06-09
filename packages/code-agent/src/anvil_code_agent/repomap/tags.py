"""Extract definition and reference symbols from Python source via tree-sitter.
API note (tree-sitter 0.25): Query(lang, src) + QueryCursor(q).captures(node) -> dict."""

from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Query, QueryCursor

_PY = Language(tspy.language())
_PARSER = Parser(_PY)
_DEFS = Query(
    _PY,
    "(function_definition name: (identifier) @d) (class_definition name: (identifier) @d)",
)
_REFS = Query(_PY, "(call function: (identifier) @r)")


@dataclass
class Tags:
    defs: set[str]
    refs: list[str]


def _names(query: Query, root, source: bytes) -> list[str]:
    caps = QueryCursor(query).captures(root)
    out: list[str] = []
    for nodes in caps.values():
        out.extend(source[n.start_byte : n.end_byte].decode("utf-8", "replace") for n in nodes)
    return out


def extract_tags(code: str) -> Tags:
    source = code.encode("utf-8")
    root = _PARSER.parse(source).root_node
    defs = set(_names(_DEFS, root, source))
    refs = _names(_REFS, root, source)
    return Tags(defs=defs, refs=refs)
