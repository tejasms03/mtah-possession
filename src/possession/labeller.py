"""Deterministic labeller: confident labels only (NaN where unsure).

Hold through flight / contested / gaps; switch when an opponent controls for >= T_ctrl or makes
2+ consecutive touches, back-dated to the first touch. Also used, with the decoder, as the
fallback submission.
Owner: person 3.
"""


def confident_labels(features, dead, params):
    raise NotImplementedError

