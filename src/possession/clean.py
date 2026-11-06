"""Stage 2: clean the ATD before anything reads it.

- reject physically impossible frame-to-frame jumps (ball > ~35 m/s, players > ~11 m/s) -> NaN
- player velocities from positions (the ATD only gives speed)
- nearest-player chains that ignore slot ids (slots swap identity)
Owner: person 1.
"""


def clean_atd(df, slot_team, params):
    raise NotImplementedError

