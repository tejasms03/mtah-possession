"""IDSSE open data (Bassek et al., Scientific Data 2025; DFL; CC-BY 4.0).

7 Bundesliga matches, 25 fps, with per-frame ball possession and ball status. Downloaded by
script into external_data/ (gitignored), never committed. Attribution: name the DFL and cite
the paper in the technical document.

degrade(): make clean tracking look like our ATD using the noise measured in phase 0:
drop the ball in realistic gaps, fragment and shuffle track ids, add jitter, keep team only.
Owner: person 4.
"""


def download(dest):
    raise NotImplementedError


def load_match(match_id):
    raise NotImplementedError


def degrade(tracking, noise_params, seed):
    raise NotImplementedError

