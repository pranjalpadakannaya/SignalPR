import json

import pytest

from agent.llm_client import CacheMiss, LLMClient, LLMConfig, LLMError, extract_json
from tests.agent.fake_llm_server import FakeLLM

MSG = [{"role": "user", "content": "review this diff please"}]


def client(url, tmp_path, **kw):
    return LLMClient(
        LLMConfig(model="fake-coder", base_url=url, api_key="k", cache_dir=str(tmp_path), max_retries=3, **kw)
    )


def test_presets_and_env(monkeypatch):
    c = LLMConfig.from_preset("qwen-32b", backend="vllm")
    assert c.model == "Qwen/Qwen2.5-Coder-32B-Instruct" and c.base_url.endswith(":8000/v1")
    assert LLMConfig.from_preset("qwen-7b", "ollama").model == "qwen2.5-coder:7b"
    monkeypatch.setenv("SIGNALPR_LLM_PRESET", "qwen-32b")
    monkeypatch.setenv("SIGNALPR_LLM_BASE_URL", "http://sol-node:8000/v1")
    monkeypatch.setenv("SIGNALPR_LLM_BACKEND", "vllm")
    e = LLMConfig.from_env()
    assert e.base_url == "http://sol-node:8000/v1" and "32B" in e.model
    with pytest.raises(ValueError):
        LLMConfig.from_preset("qwen-32b-awq", backend="ollama")


def test_from_yaml(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("llm:\n  preset: qwen-7b\n  backend: ollama\n  temperature: 0.7\n  seed: 42\n")
    c = LLMConfig.from_yaml(p)
    assert (c.model, c.temperature, c.seed) == ("qwen2.5-coder:7b", 0.7, 42)


def test_cache_hit_and_settings_in_key(tmp_path):
    with FakeLLM() as srv:
        llm = client(srv.url, tmp_path)
        a = llm.chat(MSG)
        b = llm.chat(MSG)
        assert not a.cached and b.cached and a.text == b.text and len(srv.requests) == 1
        llm.chat(MSG, temperature=0.7)  # different settings -> new request
        llm.chat(MSG, max_tokens=50)
        assert len(srv.requests) == 3
        # a new client (new process) reads the same cache from disk
        assert client(srv.url, tmp_path).chat(MSG).cached


def test_samples_get_distinct_seeds_and_keys(tmp_path):
    with FakeLLM() as srv:
        llm = client(srv.url, tmp_path, seed=7)
        outs = [llm.chat(MSG, temperature=0.7, sample=k) for k in range(3)]
        assert [r["seed"] for r in srv.requests] == [7, 8, 9]
        assert len({o.key for o in outs}) == 3
        assert all(llm.chat(MSG, temperature=0.7, sample=k).cached for k in range(3))


def test_cache_modes(tmp_path):
    with FakeLLM() as srv:
        llm = client(srv.url, tmp_path, cache_mode="only")
        with pytest.raises(CacheMiss):
            llm.chat(MSG)
        client(srv.url, tmp_path).chat(MSG)  # fill the cache
        assert llm.chat(MSG).cached and len(srv.requests) == 1  # replay works
        client(srv.url, tmp_path, cache_mode="refresh").chat(MSG)
        client(srv.url, tmp_path, cache_mode="off").chat(MSG)
        assert len(srv.requests) == 3


def test_retries_on_429_then_succeeds(tmp_path):
    with FakeLLM(fail_first=2, status=429) as srv:
        r = client(srv.url, tmp_path).chat(MSG)
        assert r.text and len(srv.requests) == 3


def test_errors(tmp_path):
    with FakeLLM() as srv:
        bad = LLMClient(LLMConfig(model="m", base_url=srv.url, api_key="bad", cache_dir=str(tmp_path)))
        with pytest.raises(LLMError, match="401"):
            bad.chat(MSG)
    dead = LLMClient(LLMConfig(model="m", base_url="http://127.0.0.1:9/v1", cache_dir=str(tmp_path), max_retries=0))
    with pytest.raises(LLMError, match="cannot reach"):
        dead.chat(MSG)


def test_json_mode_and_extract(tmp_path):
    with FakeLLM() as srv:
        llm = client(srv.url, tmp_path)
        d = llm.chat_json([{"role": "user", "content": "fenced please"}])
        assert d["echo"].startswith("fenced") and srv.requests[-1]["response_format"] == {"type": "json_object"}
        assert llm.health()["models"] == ["fake-coder"]
    assert extract_json('blah {"a": [1, 2]} trailing') == {"a": [1, 2]}
    assert extract_json("no json") is None
    assert json.loads(json.dumps(llm.stats()))["calls"] == 1


def test_throughput_script(tmp_path, capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import llm_throughput

    with FakeLLM() as srv:
        llm_throughput.main(
            [
                "--backend",
                "vllm",
                "--base-url",
                srv.url,
                "--model",
                "fake-coder",
                "--n",
                "6",
                "--concurrency",
                "3",
                "--where",
                "test",
                "--out",
                str(tmp_path / "t.csv"),
            ]
        )
    out = capsys.readouterr().out
    assert "prs_per_hour" in out and "ReviewBench" in out
    assert (tmp_path / "t.csv").read_text().count("\n") == 2 and "| test |" in (tmp_path / "t.md").read_text()
