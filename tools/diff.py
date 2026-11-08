"""Unified-diff parsing shared by the miner (AIDev `patch` column) and the live reviewer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class Line:
    kind: str  # "+", "-", " "
    old_no: int | None
    new_no: int | None
    text: str


@dataclass
class Hunk:
    old_start: int
    new_start: int
    header: str
    lines: list[Line] = field(default_factory=list)


@dataclass
class FileDiff:
    path: str
    old_path: str | None = None
    status: str = "modified"  # added | modified | removed | renamed
    hunks: list[Hunk] = field(default_factory=list)
    content: str | None = None  # full post-change file content, when we have it
    binary: bool = False

    @property
    def additions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "+")

    @property
    def deletions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "-")

    def added_lines(self) -> list[tuple[int, str]]:
        return [(ln.new_no, ln.text) for h in self.hunks for ln in h.lines if ln.kind == "+"]

    def removed_lines(self) -> list[str]:
        return [ln.text for h in self.hunks for ln in h.lines if ln.kind == "-"]

    def changed_lines(self) -> set[int]:
        return {ln.new_no for h in self.hunks for ln in h.lines if ln.kind == "+" and ln.new_no}

    def hunk_lines(self) -> set[int]:
        """All new-side line numbers visible in the diff (added + context)."""
        return {ln.new_no for h in self.hunks for ln in h.lines if ln.new_no and ln.kind != "-"}

    def new_line_text(self, no: int) -> str | None:
        if self.content is not None:
            lines = self.content.splitlines()
            if 1 <= no <= len(lines):
                return lines[no - 1]
        for h in self.hunks:
            for ln in h.lines:
                if ln.new_no == no and ln.kind != "-":
                    return ln.text
        return None

    def snippet(self, no: int, radius: int = 2) -> list[tuple[int, str, str]]:
        """(line_no, kind, text) around a new-side line, for display."""
        out = []
        for h in self.hunks:
            for ln in h.lines:
                if ln.new_no and abs(ln.new_no - no) <= radius and ln.kind != "-":
                    out.append((ln.new_no, ln.kind, ln.text))
        if not out and self.content is not None:
            lines = self.content.splitlines()
            for i in range(max(1, no - radius), min(len(lines), no + radius) + 1):
                out.append((i, " ", lines[i - 1]))
        return out

    def patch_text(self) -> str:
        parts = []
        for h in self.hunks:
            parts.append(h.header)
            for ln in h.lines:
                parts.append(ln.kind + ln.text)
        return "\n".join(parts)

    def numbered_patch(self, max_lines: int = 400) -> str:
        """Patch with new-side line numbers, the format we show the LLM."""
        out = []
        for h in self.hunks:
            out.append(h.header)
            for ln in h.lines:
                no = ln.new_no if ln.kind != "-" else ln.old_no
                tag = "   " if ln.kind == " " else (" + " if ln.kind == "+" else " - ")
                out.append(f"{(no or 0):>5}{tag}{ln.text}")
                if len(out) >= max_lines:
                    out.append("      ... (truncated)")
                    return "\n".join(out)
        return "\n".join(out)


def parse_hunks(patch: str) -> list[Hunk]:
    """Parse the hunks of a single-file patch (GitHub / AIDev `patch` format)."""
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    old_no = new_no = 0
    for raw in (patch or "").splitlines():
        m = HUNK_RE.match(raw)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(3))
            cur = Hunk(old_no, new_no, raw)
            hunks.append(cur)
            continue
        if cur is None:
            continue
        if raw.startswith("\\"):  # "\ No newline at end of file"
            continue
        kind, text = (raw[:1] or " "), raw[1:]
        if kind == "+":
            cur.lines.append(Line("+", None, new_no, text))
            new_no += 1
        elif kind == "-":
            cur.lines.append(Line("-", old_no, None, text))
            old_no += 1
        else:
            cur.lines.append(Line(" ", old_no, new_no, text))
            old_no += 1
            new_no += 1
    return hunks


def parse_unified_diff(diff: str) -> list[FileDiff]:
    """Parse a multi-file `git diff` / `.diff` / `.patch`. Also accepts a bare single-file patch."""
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    buf: list[str] = []

    def flush():
        if cur is not None:
            cur.hunks = parse_hunks("\n".join(buf))
            files.append(cur)

    lines = (diff or "").splitlines()
    if lines and not any(ln.startswith(("diff --git", "--- ", "+++ ")) for ln in lines[:50]):
        # bare hunks with no file header
        fd = FileDiff(path="snippet.txt", hunks=parse_hunks(diff))
        return [fd] if fd.hunks else []

    for line in lines:
        if line.startswith("diff --git"):
            flush()
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            cur = FileDiff(path=m.group(2) if m else "unknown", old_path=m.group(1) if m else None)
            buf = []
        elif line.startswith("--- ") and (cur is None or not buf):
            if cur is None:
                cur = FileDiff(path="unknown")
                buf = []
            src = line[4:].strip()
            if src == "/dev/null":
                cur.status = "added"
            elif src.startswith("a/"):
                cur.old_path = src[2:]
        elif line.startswith("+++ ") and cur is not None and not buf:
            dst = line[4:].strip()
            if dst == "/dev/null":
                cur.status = "removed"
            else:
                cur.path = dst[2:] if dst.startswith("b/") else dst
        elif line.startswith(("new file mode",)) and cur is not None:
            cur.status = "added"
        elif line.startswith(("deleted file mode",)) and cur is not None:
            cur.status = "removed"
        elif line.startswith("rename to ") and cur is not None:
            cur.status = "renamed"
            cur.path = line[len("rename to ") :]
        elif line.startswith("Binary files") and cur is not None:
            cur.binary = True
        elif line.startswith(("index ", "similarity index", "rename from", "old mode", "new mode")):
            continue
        else:
            if cur is not None:
                buf.append(line)
    flush()
    for f in files:
        if f.status == "modified" and f.old_path and f.old_path != f.path:
            f.status = "renamed"
    return files


def to_unified_diff(files: list[FileDiff]) -> str:
    out = []
    for f in files:
        a = f.old_path or f.path
        out.append(f"diff --git a/{a} b/{f.path}")
        out.append("--- " + ("/dev/null" if f.status == "added" else f"a/{a}"))
        out.append("+++ " + ("/dev/null" if f.status == "removed" else f"b/{f.path}"))
        out.append(f.patch_text())
    return "\n".join(out) + "\n"
