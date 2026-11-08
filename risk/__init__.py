"""Risk model: which changed files do human reviewers comment on? (issue #10)

python -m risk download [--with-patch]   # AIDev-pop from Hugging Face → data/raw
python -m risk mine                       # clean + label + split + features
python -m risk train                      # LightGBM → results/risk/
python -m risk report                     # figures for the paper → results/risk/figures

from risk import predict_risk
"""


def predict_risk(files, pr=None, model_path=None):
    from risk.predict import predict_risk as _p  # lazy: keeps `import risk` light

    return _p(files, pr, model_path)


__all__ = ["predict_risk"]
