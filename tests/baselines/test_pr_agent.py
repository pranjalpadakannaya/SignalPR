from agent.llm_client import LLMConfig
from agent.state import PRInput
from baselines.pr_agent import env_for, parse_review, severity_for

DIFF = """diff --git a/app/a.py b/app/a.py
--- a/app/a.py
+++ b/app/a.py
@@ -1,2 +1,3 @@
 x = 1
+y = 2
 z = 3
"""


def _pr():
    return PRInput.from_diff(DIFF)


def test_parse_review_maps_issues_to_comments():
    review = {
        "review": {
            "key_issues_to_review": [
                {"relevant_file": "app/a.py\n", "issue_header": "Possible Bug\n", "issue_content": "bad\n",
                 "start_line": 2, "end_line": 2, "existing_code": "y = 2\n", "improved_code": "y = 3\n"},
                {"relevant_file": "", "issue_header": "No file", "issue_content": "dropped"},
            ]
        }
    }
    (c,) = parse_review(review, _pr())
    assert (c.file, c.line, c.title, c.body, c.source) == ("app/a.py", 2, "Possible Bug", "bad", "baseline/pr_agent")
    assert c.severity == "medium" and c.quote == "y = 2"


def test_parse_review_tolerates_bad_shapes():
    assert parse_review({"review": {"key_issues_to_review": "No"}}, _pr()) == []
    assert parse_review({"review": {}}, _pr()) == []
    (c,) = parse_review({"review": {"key_issues_to_review": [{"relevant_file": "new.py", "start_line": "n/a"}]}}, _pr())
    assert c.line is None and c.file == "new.py"


def test_severity_for():
    assert severity_for("Security concern") == "critical"
    assert severity_for("Possible Bug") == "medium"
    assert severity_for("Style") == "medium"


def test_env_for_ollama_and_openai_compatible():
    e = env_for(LLMConfig.from_preset("qwen-7b", backend="ollama"))
    assert e["CONFIG__MODEL"] == "ollama/qwen2.5-coder:7b" and e["OLLAMA__API_BASE"] == "http://localhost:11434"
    e = env_for(LLMConfig.from_preset("qwen-32b", backend="vllm"))
    assert e["CONFIG__MODEL"] == "openai/Qwen/Qwen2.5-Coder-32B-Instruct"
    assert e["OPENAI__API_BASE"] == "http://localhost:8000/v1"
