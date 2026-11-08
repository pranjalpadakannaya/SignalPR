"""Read AIDev tables from data/raw as parquet (pyarrow) or csv(.gz), whichever is present."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk.paths import RAW_DIR


def table_path(name: str, raw_dir: Path = RAW_DIR) -> Path | None:
    for suffix in (".parquet", ".csv.gz", ".csv"):
        p = raw_dir / f"{name}{suffix}"
        if p.exists():
            return p
    return None


def has_table(name: str, raw_dir: Path = RAW_DIR) -> bool:
    return table_path(name, raw_dir) is not None


def read_table(name: str, columns: list[str] | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    p = table_path(name, raw_dir)
    if p is None:
        raise FileNotFoundError(
            f"Table '{name}' not found in {raw_dir}. Run `python -m risk download` (or prepare_aidev_upload.py) first."
        )
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(p).schema_arrow.names)
        cols = [c for c in columns if c in names] if columns else None
        return pq.read_table(p, columns=cols).to_pandas()
    head = pd.read_csv(p, nrows=0)
    cols = [c for c in columns if c in head.columns] if columns else None
    return pd.read_csv(p, usecols=cols, low_memory=False)
