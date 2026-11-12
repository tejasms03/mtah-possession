import numpy as np

from possession.decode import viterbi

FLAT = np.zeros((1, 3, 3))


def costs(T, ha=5.0, d=5.0):
    c = np.full((T, 3, 3), d)
    c[:, 0, 1] = c[:, 1, 0] = ha
    return c


def scores(labels, strength=1.0):
    e = np.zeros((len(labels), 3))
    for t, l in enumerate(labels):
        if l is not None:
            e[t, l] = strength
    return e


def test_clean_stretches_are_kept():
    lab = [0] * 50 + [1] * 50 + [2] * 50
    assert viterbi(scores(lab), (5, 5, 5), costs(150)).tolist() == lab


def test_one_frame_flicker_is_removed():
    lab = [0] * 40 + [1] + [0] * 40
    assert set(viterbi(scores(lab), (5, 5, 5), costs(81))) == {0}


def test_label_holds_through_frames_without_evidence():
    lab = [0] * 30 + [None] * 100 + [0] * 30
    assert set(viterbi(scores(lab), (5, 5, 5), costs(160))) == {0}


def test_minimum_length_respected():
    lab = [0] * 30 + [1] * 3 + [0] * 30          # strong but too short to become a possession
    path = viterbi(scores(lab, 10.0), (10, 10, 10), costs(63, ha=0.1))
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[-1], path, [-1]])) != 0))
    assert runs.min() >= 10 or set(path) == {0}


def test_cheap_switch_where_cost_is_low():
    lab = [0] * 30 + [1] * 30
    c = costs(60, ha=100.0)
    c[30, 0, 1] = 0.5                             # a touch at frame 30
    assert viterbi(scores(lab), (5, 5, 5), c).tolist() == lab
