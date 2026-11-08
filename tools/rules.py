"""Built-in, dependency-free static checks that run on added lines (used when Semgrep is not installed,
and always as a cheap first pass).

These are always available (Semgrep / Ruff / ESLint / Lizard are used too when installed).
Every rule is a regex over a single added line, scoped to languages.
The same rules feed (a) the evidence collector and (b) the `security_hits` /
`smell_hits` features of the risk model, so training and inference agree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PY = {"python"}
JS = {"javascript", "typescript"}
ALL = None  # any language


@dataclass(frozen=True)
class Rule:
    id: str
    category: str  # security | correctness | design | performance
    severity: str  # critical | medium | low
    langs: set | None
    pattern: re.Pattern
    message: str


def R(id, cat, sev, langs, pat, msg, flags=0):
    return Rule(id, cat, sev, langs, re.compile(pat, flags), msg)


RULES: list[Rule] = [
    # ---------------- security ----------------
    R(
        "sec.eval",
        "security",
        "critical",
        PY | JS,
        r"(?<![\w.])eval\s*\(",
        "`eval` on dynamic input allows arbitrary code execution.",
    ),
    R(
        "sec.exec",
        "security",
        "critical",
        PY,
        r"(?<![\w.])exec\s*\(",
        "`exec` runs arbitrary code; avoid it on anything derived from input.",
    ),
    R(
        "sec.shell",
        "security",
        "critical",
        PY,
        r"subprocess\.\w+\(.*shell\s*=\s*True",
        "`shell=True` with interpolated arguments enables shell injection.",
    ),
    R(
        "sec.os_system",
        "security",
        "medium",
        PY,
        r"os\.(system|popen)\s*\(",
        "`os.system`/`os.popen` pass strings to a shell; prefer `subprocess.run([...])`.",
    ),
    R(
        "sec.pickle",
        "security",
        "critical",
        PY,
        r"pickle\.loads?\s*\(|cPickle\.loads?\s*\(",
        "Unpickling untrusted data can execute arbitrary code.",
    ),
    R(
        "sec.yaml",
        "security",
        "medium",
        PY,
        r"yaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.SafeLoader)(?![^)]*SafeLoader)",
        "`yaml.load` without `SafeLoader` can construct arbitrary objects; use `yaml.safe_load`.",
    ),
    R(
        "sec.sql_concat",
        "security",
        "critical",
        ALL,
        r"""(execute|query|raw|cursor\.\w+)\s*\(\s*(f["']|["'][^"']*\b(SELECT|INSERT|UPDATE|DELETE)\b[^"']*["']\s*(\+|%|\.format))""",
        "SQL built with string formatting is injectable; use parameterized queries.",
        re.I,
    ),
    R(
        "sec.sql_fstring",
        "security",
        "critical",
        ALL,
        r"""f["'][^"']*\b(SELECT|INSERT INTO|UPDATE|DELETE FROM)\b[^"']*\{""",
        "SQL assembled with an f-string; pass values as query parameters.",
        re.I,
    ),
    R(
        "sec.secret",
        "security",
        "critical",
        ALL,
        r"""(?i)\b(api[_-]?key|secret|passw(or)?d|token|private[_-]?key|aws_access_key_id)\b\s*[:=]\s*["'][A-Za-z0-9_\-/+=.]{8,}["']""",
        "Hard-coded credential in source; load it from the environment or a secret store.",
    ),
    R(
        "sec.aws_key",
        "security",
        "critical",
        ALL,
        r"AKIA[0-9A-Z]{16}",
        "Looks like an AWS access key ID committed to the repo.",
    ),
    R(
        "sec.tls_off",
        "security",
        "medium",
        ALL,
        r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true",
        "TLS certificate verification is disabled.",
    ),
    R(
        "sec.weak_hash",
        "security",
        "medium",
        ALL,
        r"\b(md5|sha1)\s*\(|hashlib\.(md5|sha1)\b|createHash\(\s*['\"](md5|sha1)",
        "MD5/SHA-1 are broken for security purposes (passwords, signatures, tokens).",
    ),
    R(
        "sec.random_token",
        "security",
        "medium",
        PY,
        r"random\.(random|randint|choice)\(.*(token|secret|password|otp|nonce)|(token|secret|otp|nonce)\w*\s*=\s*.*random\.(random|randint|choice)",
        "`random` is not cryptographically secure; use `secrets` for tokens.",
        re.I,
    ),
    R(
        "sec.innerhtml",
        "security",
        "medium",
        JS,
        r"\.innerHTML\s*=|dangerouslySetInnerHTML|document\.write\s*\(",
        "Assigning HTML from data enables XSS; use textContent or sanitize.",
    ),
    R(
        "sec.cors_any",
        "security",
        "medium",
        ALL,
        r"""Access-Control-Allow-Origin['"]?\s*[:,]\s*['"]\*|allow_origins\s*=\s*\[\s*['"]\*""",
        "CORS allows every origin.",
    ),
    R(
        "sec.debug_on",
        "security",
        "medium",
        PY,
        r"(DEBUG\s*=\s*True|app\.run\(.*debug\s*=\s*True)",
        "Debug mode enabled; it leaks stack traces and may allow code execution.",
    ),
    R(
        "sec.path_join_input",
        "security",
        "medium",
        PY,
        r"""open\(\s*(request\.|f["'].*\{.*(request|user|param|filename))""",
        "File path built from request data; check for path traversal.",
    ),
    R(
        "sec.jwt_none",
        "security",
        "critical",
        ALL,
        r"""algorithms?\s*=\s*\[?\s*['"]none['"]|verify_signature['"]?\s*:\s*False""",
        "JWT signature verification disabled.",
    ),
    # ---------------- correctness ----------------
    R(
        "cor.bare_except",
        "correctness",
        "medium",
        PY,
        r"^\s*except\s*:\s*(#.*)?$",
        "Bare `except:` also swallows KeyboardInterrupt/SystemExit and hides real bugs.",
    ),
    R(
        "cor.except_pass",
        "correctness",
        "medium",
        PY,
        r"except(\s+\w+(\s+as\s+\w+)?)?\s*:\s*pass\b",
        "Exception silently ignored.",
    ),
    R(
        "cor.mutable_default",
        "correctness",
        "medium",
        PY,
        r"def \w+\(.*=\s*(\[\]|\{\}|set\(\))\s*[,)]",
        "Mutable default argument is shared across calls.",
    ),
    R("cor.none_eq", "correctness", "low", PY, r"[!=]=\s*None\b", "Compare to `None` with `is` / `is not`."),
    R(
        "cor.empty_catch",
        "correctness",
        "medium",
        JS,
        r"catch\s*(\(\s*\w*\s*\))?\s*\{\s*\}",
        "Empty catch block swallows errors.",
    ),
    R(
        "cor.loose_eq",
        "correctness",
        "low",
        JS,
        r"[^=!]==[^=]|!=[^=]",
        "Loose equality (`==`) coerces types; use `===`.",
    ),
    R(
        "cor.float_money",
        "correctness",
        "low",
        ALL,
        r"(price|amount|total|balance|cost)\w*\s*=\s*(float|parseFloat)\(",
        "Money stored as float accumulates rounding error; use Decimal / integer cents.",
        re.I,
    ),
    R(
        "cor.assert_runtime",
        "correctness",
        "low",
        PY,
        r"^\s*assert\s+(?!.*#\s*noqa)",
        "`assert` is stripped with `python -O`; do not use it for runtime validation.",
    ),
    R(
        "cor.len_minus_one",
        "correctness",
        "low",
        PY,
        r"range\(\s*len\([^)]*\)\s*-\s*1\s*\)",
        "`range(len(x) - 1)` skips the last element; check for an off-by-one.",
    ),
    R(
        "cor.no_timeout",
        "correctness",
        "medium",
        PY,
        r"requests\.(get|post|put|delete|patch|request)\((?!.*timeout)",
        "HTTP call without a timeout can hang the worker forever.",
    ),
    R(
        "cor.open_no_with",
        "correctness",
        "low",
        PY,
        r"^\s*\w+\s*=\s*open\(",
        "File opened without `with`; the handle leaks if an exception is raised.",
    ),
    R(
        "cor.datetime_naive",
        "correctness",
        "low",
        PY,
        r"datetime\.(now|utcnow)\(\s*\)",
        "Naive datetime (no timezone); comparisons with aware datetimes will fail.",
    ),
    R(
        "cor.await_missing",
        "correctness",
        "medium",
        JS,
        r"^\s*(?!return|await|const|let|var)\w+\.(save|fetch|update|delete|create)\w*\(.*\);?\s*$",
        "Async call result is not awaited or returned; errors and ordering are lost.",
    ),
    # ---------------- design / maintainability ----------------
    R("des.todo", "design", "low", ALL, r"\b(TODO|FIXME|XXX|HACK)\b", "Unresolved TODO/FIXME left in the change."),
    R(
        "des.debug_print",
        "design",
        "low",
        ALL,
        r"^\s*(print\(|console\.(log|debug)\(|System\.out\.println|fmt\.Println\(\s*\"debug|debugger;)",
        "Debug output left in the change.",
    ),
    R(
        "des.star_import",
        "design",
        "low",
        PY,
        r"^\s*from\s+\S+\s+import\s+\*",
        "Wildcard import hides where names come from.",
    ),
    R("des.any_type", "design", "low", {"typescript"}, r":\s*any\b|as\s+any\b", "`any` disables type checking."),
    R(
        "des.ts_ignore",
        "design",
        "low",
        JS,
        r"@ts-ignore|eslint-disable",
        "Type/lint check suppressed instead of fixed.",
    ),
    # ---------------- performance ----------------
    R(
        "perf.query_in_loop",
        "performance",
        "medium",
        ALL,
        r"^\s+.*\.(objects\.(get|filter)|query\(|execute\(|find_one\(|findOne\(|fetch\()",
        "Database/network call inside an indented block; check for an N+1 pattern if this is in a loop.",
    ),
    R(
        "perf.sleep",
        "performance",
        "low",
        ALL,
        r"time\.sleep\(\s*\d|Thread\.sleep\(\s*\d|setTimeout\(.*,\s*\d{4,}",
        "Hard-coded sleep; prefer events, retries with backoff, or async waits.",
    ),
]

RULES_BY_ID = {r.id: r for r in RULES}


def run_rules(lang: str, added: list[tuple[int, str]]):
    """Yield (rule, line_no, line_text) for every hit on added lines."""
    prev_indent_loop = False
    for no, text in added:
        stripped = text.strip()
        if (
            not stripped
            or stripped.startswith(("#", "//", "*", "/*"))
            and "TODO" not in stripped
            and "FIXME" not in stripped
        ):
            continue
        is_loop = bool(re.match(r"\s*(for|while)\b", text))
        for r in RULES:
            if r.langs is not None and lang not in r.langs:
                continue
            if r.id == "perf.query_in_loop" and not prev_indent_loop:
                continue
            if r.pattern.search(text):
                yield r, no, text
        if is_loop:
            prev_indent_loop = True
        elif not text.startswith((" ", "\t")):
            prev_indent_loop = False
