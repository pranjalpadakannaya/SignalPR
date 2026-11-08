"""python -m risk {download,mine,train,report,all}"""

import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(prog="risk", description="SignalPR risk model (issue #10)")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="AIDev-pop from Hugging Face → data/raw (parquet)")
    d.add_argument("--revision", default=None)
    d.add_argument("--with-patch", action="store_true", help="also compute patch features (streams 12 GB once)")
    m = sub.add_parser("mine", help="clean, label, split, build features")
    m.add_argument("--include-unmerged", action="store_true")
    t = sub.add_parser("train", help="train + evaluate the model")
    t.add_argument("--sample-frac", type=float, default=1.0)
    sub.add_parser("report", help="figures + summary.md for the paper")
    sub.add_parser("all", help="mine + train + report")
    a = p.parse_args(argv)
    if a.cmd == "download":
        from risk.download import download

        download(a.revision, a.with_patch)
    if a.cmd in ("mine", "all"):
        from risk.mine import mine

        mine(include_unmerged=getattr(a, "include_unmerged", False))
    if a.cmd in ("train", "all"):
        from risk.model import train

        r = train(sample_frac=getattr(a, "sample_frac", 1.0))
        print(json.dumps({k: r["test"][k] for k in ("pr_auc", "pr_auc_ci95", "roc_auc", "ece", "base_rate")}, indent=1))
    if a.cmd in ("report", "all"):
        from risk.report import make_figures

        print(json.dumps(make_figures(), indent=1))


if __name__ == "__main__":
    main()
