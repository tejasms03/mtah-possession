import numpy as np

from possession import config, tcn


def _random_weights(rng, c=8, dilations=(1, 2, 4)):
    W = {"in_w": rng.normal(size=(c, tcn.N_IN)), "in_b": rng.normal(size=c),
         "em_w": rng.normal(size=(3, c)), "em_b": rng.normal(size=3),
         "to_w": rng.normal(size=(3, c)), "to_b": rng.normal(size=3), "dilations": np.array(dilations)}
    for k in range(len(dilations)):
        W[f"c{k}_w"], W[f"c{k}_b"] = 0.3 * rng.normal(size=(c, c, 3)), rng.normal(size=c)
        W[f"o{k}_w"], W[f"o{k}_b"] = 0.3 * rng.normal(size=(c, c)), rng.normal(size=c)
    return W


def _reference_forward(W, X):
    """Plain loop version of the dilated convolution stack."""
    n = len(X)
    x = np.zeros((n + 2 * tcn.PAD, X.shape[1]))
    x[tcn.PAD : tcn.PAD + n] = X
    h = x @ W["in_w"].T + W["in_b"]
    T = len(h)
    for k, d in enumerate(W["dilations"]):
        a = np.maximum(h, 0)
        z = np.tile(W[f"c{k}_b"], (T, 1))
        for t in range(T):
            for tap, off in enumerate((-d, 0, d)):
                if 0 <= t + off < T:
                    z[t] += W[f"c{k}_w"][:, :, tap] @ a[t + off]
        h = h + np.maximum(z, 0) @ W[f"o{k}_w"].T + W[f"o{k}_b"]
    a = np.maximum(h[tcn.PAD : tcn.PAD + n], 0)
    return a @ W["em_w"].T + W["em_b"]


def test_forward_matches_reference():
    rng = np.random.default_rng(0)
    W = _random_weights(rng)
    X = rng.normal(size=(40, tcn.N_IN)).astype(np.float32)
    e, _ = tcn._forward(W, X)
    np.testing.assert_allclose(e, _reference_forward(W, X), rtol=1e-9, atol=1e-9)


def test_team_blind():
    rng = np.random.default_rng(1)
    W = _random_weights(rng)
    X = rng.normal(size=(60, tcn.N_IN)).astype(np.float32)
    e1, t1 = tcn.predict(W, X)
    e2, t2 = tcn.predict(W, tcn.swap_ha(X))
    np.testing.assert_allclose(e1, e2[:, [1, 0, 2]], atol=1e-12)
    np.testing.assert_allclose(t1, t2[:, [0, 2, 1]], atol=1e-12)
    np.testing.assert_allclose(np.exp(e1).sum(1), 1.0, atol=1e-12)


def test_saved_weights_load():
    W = tcn.load(config.REPO_ROOT / config.load_params()["model"]["tcn"]["weights"])
    rng = np.random.default_rng(2)
    n = 300
    X = tcn.features(rng.normal(size=n), rng.normal(size=n), rng.normal(size=n), rng.random(n) > 0.3,
                     np.abs(rng.normal(size=n)) * 5, rng.random(n) * 30, rng.random(n) * 30,
                     rng.choice(["", "H", "A"], n))
    assert X.shape == (n, tcn.N_IN)
    e, t = tcn.predict(W, X)
    assert np.isfinite(e).all() and np.isfinite(t).all()
    np.testing.assert_allclose(np.exp(e).sum(1), 1.0, atol=1e-9)
