from qdotml.data import build_dataset
from qdotml.models import train_line_detector, evaluate
import numpy as np

def test_supervised_runs():
    ds = build_dataset(seed=0, grid=48, n_devices=6, patches_per=12, patch_size=28)
    X, yl, dev = ds["X"], ds["y_line"], ds["device"]
    tr = dev < 4; te = dev >= 4
    clf = train_line_detector(X[tr], yl[tr], seed=0)
    m = evaluate(clf, X[te], yl[te])
    assert m["acc"] > 0.6
