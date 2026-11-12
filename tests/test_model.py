import copy

import numpy as np

from possession import config, model


def fake_prep(i, n=600, cols_dead=None, cols_team=12):
    rng = np.random.default_rng(i)
    nd = cols_dead or len(model.dead_columns(config.load_params()))
    gt = rng.choice(np.array(["H", "A", "D"]), n)
    return {"id": f"MET{i}", "gt": gt, "dead": rng.normal(size=(n, nd)).astype(np.float32),
            "own_H": rng.normal(size=(n, cols_team)).astype(np.float32),
            "own_A": rng.normal(size=(n, cols_team)).astype(np.float32)}


def small_params(size):
    p = copy.deepcopy(config.load_params())
    p["model"].update(max_iter=5, min_samples_leaf=5, row_step=2)
    p["model"]["ensemble"]["size"] = size
    return p


def test_size_one_is_the_plain_model():
    p = small_params(1)
    assert model.member_params(p, 0) == p


def test_ensemble_fits_and_predicts_with_own_rows():
    p = small_params(2)
    preps = [fake_prep(i) for i in range(3)]
    own = (preps[0]["dead"][:50], np.arange(50) % 2 == 0)
    members = model.fit_ensemble(preps, p, own=own)
    assert len(members) == 2 and all(len(m) == 4 for m in members)
    dead, team = model.predict_ensemble(members, preps[0], p)
    assert dead.shape == team.shape == (600,)
    one, _ = model.predict(members[0], preps[0], model.member_params(p, 0))
    assert not np.allclose(dead, one)          # members differ, so the mean differs
