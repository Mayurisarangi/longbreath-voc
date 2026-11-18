import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import sim

def test_simulator_shapes_and_missingness():
    for sc in ("ideal", "real"):
        d = sim.simulate(sc, seed=0)
        assert d["V"].shape == (1200, 5, 10)
        assert d["M"][:, 0].all()                      # baseline visit always observed
        assert (~d["M"]).sum(1).max() <= 2             # at most two missing visits
        assert np.isnan(d["V"]).mean() < 0.2
        assert np.bincount(d["y"]).tolist() == [480, 360, 360]

def test_seed_reproducible():
    a, b = sim.simulate("real", 0, 0.5), sim.simulate("real", 0, 0.5)
    assert np.allclose(np.nan_to_num(a["V"]), np.nan_to_num(b["V"]))

def test_model_sizes():
    import pytest; pytest.importorskip("torch"); import models
    assert 2.0e6 < models.nparams(models.make("HTAN", 30)) < 2.5e6
    assert models.nparams(models.make("HTAN-small", 30)) < 6e4
