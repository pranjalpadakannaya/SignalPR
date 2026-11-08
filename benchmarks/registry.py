"""Which benchmarks exist, where their data lives, and the dev/test guard.

Martian is the DEV set: tune prompts and thresholds on it as much as you like.
ReviewBench is the TEST set: it may only be loaded with purpose="eval", so a tuning script that
accidentally points at it fails loudly instead of leaking the test set into the prompts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("benchmarks.yaml")
DATA_DIR = Path(os.environ.get("SIGNALPR_BENCH_DIR", ROOT / "data" / "benchmarks"))  # git-ignored
REPOS_DIR = Path(os.environ.get("SIGNALPR_REPOS_DIR", ROOT / "repos"))  # git-ignored checkouts

PURPOSES = ("tune", "eval")


class TestSetAccessError(RuntimeError):
    """Raised when a test-only benchmark is requested for anything other than final evaluation."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class Benchmark:
    name: str
    split: str
    config: dict[str, Any]

    @property
    def data_dir(self) -> Path:
        return DATA_DIR / self.name

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def prs_path(self) -> Path:
        return self.data_dir / "prs.jsonl"

    @property
    def is_test(self) -> bool:
        return self.split == "test"


@lru_cache(maxsize=1)
def _config() -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def get_benchmark(name: str) -> Benchmark:
    cfg = _config()
    if name not in cfg:
        raise KeyError(f"Unknown benchmark '{name}'. Known: {', '.join(cfg)}")
    split = cfg[name]["split"]
    if split not in ("dev", "test"):
        raise ValueError(f"{name}: split must be 'dev' or 'test', got {split!r}")
    return Benchmark(name=name, split=split, config=cfg[name])


def all_benchmarks() -> list[Benchmark]:
    return [get_benchmark(n) for n in _config()]


def check_access(name: str, purpose: str) -> Benchmark:
    """Return the benchmark if `purpose` is allowed for it, else raise TestSetAccessError."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    bench = get_benchmark(name)
    if bench.is_test and purpose != "eval":
        raise TestSetAccessError(
            f"'{name}' is the held-out TEST set and cannot be used for {purpose!r}. "
            "Tune on the dev set (martian) and load the test set only with purpose='eval' "
            "for the final, frozen configuration."
        )
    return bench
