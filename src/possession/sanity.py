"""Full-match checks that need no ground truth, plus the boundary audit on the public window.

- possession count (~300-380 expected), duration histogram
- agreement with Smart Tagging team on BUILD UP / PROGRESSION
- D->live transitions vs SET PIECES starts
- boundary audit: signed timing error, misses, false alarms per boundary type (public 5 min)
Owner: person 3.
"""


def sanity_panel(labels, tags):
    raise NotImplementedError


def boundary_audit(labels, gt):
    raise NotImplementedError

