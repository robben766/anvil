"""Markdown-aware chunker with exact offset invariant.

Core guarantee: for every ChunkDraft c,
    text[c.start_offset:c.end_offset] == c.content

Algorithm
---------
1. _scan_blocks(text) → list of _Block (text or table), each carrying:
   - absolute start_offset / end_offset in the original text
   - header_path at the time the block is emitted
   - block_type: "text" | "table"

2. Table blocks → single ChunkDraft each.

3. Text blocks → _window(text, block, size, overlap):
   - Accumulate whole lines until adding the next line would exceed *size*.
   - Emit ChunkDraft with content = text[start:end] (slice, not join).
   - Next window start = line-aligned position after backing up *overlap* chars
     from end of last chunk.  If that would not advance the cursor, skip to
     the first line that starts strictly after the cursor (anti-deadlock).
   - A single line that alone exceeds *size* is emitted as an over-long chunk.

4. Assign seq numbers sequentially across all produced chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ChunkDraft:
    content: str        # == text[start_offset:end_offset]
    header_path: str    # "一级 > 二级"; "" when no headers
    start_offset: int
    end_offset: int
    seq: int


# ── Internal block representation ────────────────────────────────────────────

@dataclass
class _Block:
    block_type: Literal["text", "table"]
    start_offset: int
    end_offset: int
    header_path: str
    # For text blocks: list of (line_start_offset, line_text_including_newline)
    lines: list[tuple[int, str]]


# ── Header stack helpers ──────────────────────────────────────────────────────

def _parse_header(line: str) -> tuple[int, str] | None:
    """Return (level, title) if line is a markdown ATX header, else None."""
    stripped = line.rstrip("\n")
    for level in (3, 2, 1):
        prefix = "#" * level + " "
        if stripped.startswith(prefix) and not stripped.startswith("#" * (level + 1)):
            return level, stripped[len(prefix):]
    return None


def _make_header_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


# ── Block scanner ─────────────────────────────────────────────────────────────

def _scan_blocks(text: str) -> list[_Block]:
    """Scan text line by line and group into text/table _Block objects."""
    blocks: list[_Block] = []
    header_stack: list[tuple[int, str]] = []  # [(level, title), ...]

    # Pending text accumulation for the current section
    pending_lines: list[tuple[int, str]] = []  # (abs_offset, line_str)

    # Pending table accumulation
    table_lines: list[tuple[int, str]] = []
    table_start: int = 0

    def _flush_text() -> None:
        nonlocal pending_lines
        if pending_lines:
            start = pending_lines[0][0]
            end = pending_lines[-1][0] + len(pending_lines[-1][1])
            blocks.append(_Block(
                block_type="text",
                start_offset=start,
                end_offset=end,
                header_path=_make_header_path(header_stack),
                lines=list(pending_lines),
            ))
        pending_lines = []

    def _flush_table() -> None:
        nonlocal table_lines, table_start
        if table_lines:
            start = table_lines[0][0]
            end = table_lines[-1][0] + len(table_lines[-1][1])
            blocks.append(_Block(
                block_type="table",
                start_offset=start,
                end_offset=end,
                header_path=_make_header_path(header_stack),
                lines=list(table_lines),
            ))
        table_lines = []

    offset = 0
    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)

        hdr = _parse_header(line)
        if hdr is not None:
            level, title = hdr
            # Flush any pending table or text before changing section
            _flush_table()
            _flush_text()
            # Pop headers of same or deeper level
            header_stack = [(lvl, t) for lvl, t in header_stack if lvl < level]
            header_stack.append((level, title))
            # Header line itself is NOT added to any block
            continue

        is_table_line = line.startswith("|")
        if is_table_line:
            # If we were accumulating text, flush it first
            if pending_lines:
                _flush_text()
            if not table_lines:
                table_start = line_start
            table_lines.append((line_start, line))
        else:
            # Non-table content line
            if table_lines:
                _flush_table()
            pending_lines.append((line_start, line))

    # End of document – flush whatever remains
    _flush_table()
    _flush_text()

    return blocks


# ── Windowed chunker for text blocks ─────────────────────────────────────────

def _window(text: str, block: _Block, size: int, overlap: int) -> list[ChunkDraft]:
    """Slice a text block into overlapping window chunks aligned to line starts."""
    lines = block.lines  # [(abs_offset, line_str), ...]
    if not lines:
        return []

    chunks: list[ChunkDraft] = []
    n = len(lines)
    i = 0  # index into lines for the current window start

    while i < n:
        # Accumulate lines into a window (window start = lines[i][0])
        j = i
        accumulated = 0
        while j < n:
            line_len = len(lines[j][1])
            if accumulated > 0 and accumulated + line_len > size:
                # Adding this line would exceed size; stop here
                break
            accumulated += line_len
            j += 1
            # If we've accumulated at least one line, see if next fits
            # (loop continues to accumulate more)

        # j is exclusive end; lines[i:j] form the chunk
        # If j == i (no lines added, should not happen since we always take at least 1)
        if j == i:
            j = i + 1  # safety: take at least one line to avoid infinite loop

        chunk_start = lines[i][0]
        chunk_end = lines[j - 1][0] + len(lines[j - 1][1])
        chunks.append(ChunkDraft(
            content=text[chunk_start:chunk_end],
            header_path=block.header_path,
            start_offset=chunk_start,
            end_offset=chunk_end,
            seq=0,  # assigned later
        ))

        if j >= n:
            break

        # Determine next window start via overlap:
        # Back up `overlap` chars from chunk_end, find which line that falls in,
        # and start the next window at that line's start.
        overlap_target = chunk_end - overlap
        # Find the first line whose start is >= overlap_target
        # We want the line that CONTAINS overlap_target, then start window there.
        next_i = j  # default: no overlap, continue from where we left off
        for k in range(j - 1, i - 1, -1):
            if lines[k][0] <= overlap_target:
                # overlap_target falls in line k or after it
                # Start next window at line k
                candidate = k
                # Anti-deadlock: if candidate <= i, we haven't advanced
                if candidate <= i:
                    next_i = i + 1
                else:
                    next_i = candidate
                break
        else:
            # overlap_target is before the first line in current window
            next_i = i + 1

        i = next_i

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_markdown(
    text: str,
    *,
    size: int = 600,
    overlap: int = 100,
) -> list[ChunkDraft]:
    """Split *text* into ChunkDraft objects using markdown structure.

    Parameters
    ----------
    text:    raw markdown string (LF line endings expected)
    size:    soft maximum number of characters per chunk
    overlap: number of characters to back up when computing the start of the
             next window (aligned to line boundaries)
    """
    blocks = _scan_blocks(text)
    all_chunks: list[ChunkDraft] = []

    for block in blocks:
        if block.block_type == "table":
            # Atomic: the whole block is one chunk
            content = text[block.start_offset:block.end_offset]
            all_chunks.append(ChunkDraft(
                content=content,
                header_path=block.header_path,
                start_offset=block.start_offset,
                end_offset=block.end_offset,
                seq=0,
            ))
        else:
            all_chunks.extend(_window(text, block, size, overlap))

    # Filter out blank chunks (content that is entirely whitespace).
    # This can arise from markdown sections that contain only blank lines
    # (e.g. consecutive empty lines between headings).  The offset invariant
    # text[start_offset:end_offset] == content is unaffected because we drop
    # the chunk entirely rather than modifying its slice.
    all_chunks = [c for c in all_chunks if c.content.strip() != ""]

    # Assign monotonically increasing seq (re-numbered after blank filtering
    # so that seq is always a contiguous 0-based sequence).
    for seq, chunk in enumerate(all_chunks):
        chunk.seq = seq

    return all_chunks
