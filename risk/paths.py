"""Where the risk model reads data and writes results. Override with env vars if needed."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SIGNALPR_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"  # AIDev tables: *.parquet (download) or *.csv.gz (prepare script)
PROCESSED_DIR = DATA_DIR / "processed"  # mined file-level dataset (not committed)
RESULTS_DIR = Path(os.environ.get("SIGNALPR_RISK_RESULTS", ROOT / "results" / "risk"))
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = Path(os.environ.get("SIGNALPR_MODEL_DIR", ROOT / "models"))  # git-ignored (repo policy); share the file
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.json"
INSIGHTS_PATH = RESULTS_DIR / "dataset_stats.json"

HF_REPO = os.environ.get("SIGNALPR_HF_REPO", "hao-li/AIDev")
HF_REVISION = os.environ.get("SIGNALPR_HF_REVISION", "v5")
