import pytest

import benchmarks
from benchmarks import TestSetAccessError, check_access, get_benchmark, load_benchmark
from benchmarks.schema import (
    ChangedFile,
    KnownIssue,
    LinkedIssue,
    PRExample,
    norm_severity,
    read_jsonl,
    size_bucket,
    write_jsonl,
)


def make_example(**kw) -> PRExample:
    base = dict(
        id="martian/o_r_1",
        benchmark="martian",
        split="dev",
        repo="o/r",
        pr_number=1,
        pr_url="https://github.com/o/r/pull/1",
        clone_url="https://github.com/o/r.git",
        base_sha="a" * 40,
        head_sha="b" * 40,
        title="t",
        description="d",
        diff="",
        files=[ChangedFile(path="x.py", additions=3)],
        linked_issues=[LinkedIssue(number=7, title="Bug", body="it breaks")],
        known_issues=[
            KnownIssue(id="k0", body="real", severity="high", file="x.py", start_line=2, end_line=2),
            KnownIssue(id="k1", body="not real", label="fp"),
        ],
        metadata={"language": "Python"},
    )
    base.update(kw)
    return PRExample(**base)


def test_jsonl_roundtrip(tmp_path):
    ex = make_example()
    path = tmp_path / "prs.jsonl"
    assert write_jsonl([ex, ex], path) == 2
    back = read_jsonl(path)
    assert back[0] == ex and len(back) == 2
    assert isinstance(back[0].known_issues[0], KnownIssue) and isinstance(back[0].files[0], ChangedFile)


def test_from_dict_ignores_unknown_fields():
    d = make_example().to_dict()
    d["new_field"] = 1
    d["known_issues"][0]["future"] = "x"
    assert PRExample.from_dict(d).known_issues[0].body == "real"


def test_real_issues_and_linked_text():
    ex = make_example()
    assert [k.id for k in ex.real_issues] == ["k0"]
    assert ex.linked_issue_text.startswith("#7 Bug")


@pytest.mark.parametrize(
    "lines,bucket", [(0, "xs"), (50, "xs"), (51, "s"), (200, "s"), (500, "m"), (1000, "l"), (1001, "xl")]
)
def test_size_bucket(lines, bucket):
    assert size_bucket(lines) == bucket


def test_norm_severity():
    assert norm_severity("Critical") == "critical"
    assert norm_severity(" high ") == "high"
    assert norm_severity("blocker") is None and norm_severity(None) is None


def test_splits():
    assert get_benchmark("martian").split == "dev"
    assert get_benchmark("reviewbench").split == "test"
    with pytest.raises(KeyError):
        get_benchmark("nope")


def test_test_set_guard():
    assert check_access("martian", "tune").name == "martian"
    assert check_access("reviewbench", "eval").name == "reviewbench"
    with pytest.raises(TestSetAccessError):
        check_access("reviewbench", "tune")
    with pytest.raises(ValueError):
        check_access("martian", "train")


def test_load_refuses_test_set_before_touching_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmarks.registry, "DATA_DIR", tmp_path)
    with pytest.raises(TestSetAccessError):
        load_benchmark("reviewbench", purpose="tune")
    with pytest.raises(FileNotFoundError):
        load_benchmark("reviewbench", purpose="eval")


def test_load_and_subset(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmarks.registry, "DATA_DIR", tmp_path)
    rb = [
        make_example(id="reviewbench/a_1", benchmark="reviewbench", split="test", metadata={"in_test_subset": True}),
        make_example(id="reviewbench/a_2", benchmark="reviewbench", split="test", metadata={}),
    ]
    write_jsonl(rb, tmp_path / "reviewbench" / "prs.jsonl")
    assert len(load_benchmark("reviewbench", purpose="eval")) == 2
    assert [e.id for e in load_benchmark("reviewbench", purpose="eval", subset="test_subset")] == ["reviewbench/a_1"]
