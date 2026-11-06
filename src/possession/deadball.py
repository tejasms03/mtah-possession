"""Dead-ball detection and restarts.

D score from collective speed drop, ball out of bounds (confirmed by stop/vanish), ball stationary
at a restart spot, SET PIECES tags. Restart moment = ball goes from still to moving. Possessor by
vote: team at the kick, restart-type rule, last toucher opponent, tag team (throw-ins are N/A).
Owner: person 2.
"""


def dead_ball_segments(features, tags, params):
    raise NotImplementedError


def restart_possessor(segment, features, tags, params):
    raise NotImplementedError

