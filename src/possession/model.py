"""Per-frame H/A/D scores from a +/-2 s window of features.

LightGBM first (fixed seed, 1 thread). Temporal NN (1D CNN / GRU) as a drop-in experiment:
same inputs, same labels, same decoder. Training labels: degraded IDSSE + our confident labels,
validated on held-out whole minutes.
Owner: person 4.
"""


def train(windows, labels, params):
    raise NotImplementedError


def predict(model, windows):
    raise NotImplementedError

