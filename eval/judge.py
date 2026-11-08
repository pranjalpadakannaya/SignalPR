"""LLM judge: decide which review comments describe which known issues (one cached call per PR).

Martian's golden comments have no file or line, so they are matched by meaning (`--judge llm`). The judge
is whatever model the shared LLM client points at; keep it fixed across all systems so every system is
judged the same way.
"""

from __future__ import annotations

from agent import llm

SYSTEM = """You compare a code review against a list of known issues in the same pull request.
For each review comment decide which known issue it identifies (the same underlying problem, even if worded
differently), or none. Each known issue can be matched by at most one comment.
Reply with JSON only: {"matches": [{"comment": <comment number>, "issue": <issue number or null>}]}"""


def llm_match(comments: list[dict], issues: list[dict]) -> list[int | None]:
    if not comments or not issues:
        return [None] * len(comments)
    known = "\n".join(
        f"{i + 1}. [{g.get('severity') or '-'}] {g.get('file') or ''} {(g.get('body') or '')[:400]}"
        for i, g in enumerate(issues)
    )
    review = "\n".join(
        f"{j + 1}. {c.get('file')}:{c.get('line')} {c.get('title', '')}: {(c.get('body') or '')[:400]}"
        for j, c in enumerate(comments)
    )
    data, _ = llm.chat_json(
        SYSTEM,
        f"KNOWN ISSUES:\n{known}\n\nREVIEW COMMENTS:\n{review}",
        temperature=0.0,
        tag="eval/judge",
        max_tokens=1500,
    )
    out: list[int | None] = [None] * len(comments)
    used: set[int] = set()
    for m in (data or {}).get("matches", []) if isinstance(data, dict) else []:
        try:
            j, i = int(m.get("comment")) - 1, m.get("issue")
            i = None if i in (None, "", "null") else int(i) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= j < len(comments) and i is not None and 0 <= i < len(issues) and i not in used and out[j] is None:
            out[j] = i
            used.add(i)
    return out
