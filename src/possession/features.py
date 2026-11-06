"""Stage 4: per-frame features (~35 columns), identical for our ATD and degraded IDSSE.

Groups: ball state, ball vs each team (distance, coupling, players within 5 m, touch),
team shape per team (centroid + velocity, spread, line height, speed, pressing),
whole game (mean speed, tracked count, set-piece tag), context (half, attacking direction).
Owner: person 1.
"""


def build_features(clean, ball, tags, params):
    raise NotImplementedError

