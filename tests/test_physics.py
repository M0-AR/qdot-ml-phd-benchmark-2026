from qdotml.physics import generate_csd, sample_device
import numpy as np

def test_csd_shapes():
    rng = np.random.default_rng(0)
    csd = generate_csd(rng, grid=48, noise=1.0)
    assert csd["signal"].shape == (48, 48)
    assert csd["signal"].min() >= 0 and csd["signal"].max() <= 1
    assert csd["lines"].shape == (48, 48)

def test_device_sampling_varies():
    rng = np.random.default_rng(1)
    a = sample_device(rng); b = sample_device(rng)
    assert (a.a12, a.a21) != (b.a12, b.a21)
