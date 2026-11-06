"""HSMM / Viterbi over {H, A, D} for each half.

Switch penalties (lambda_HA, lambda_D) act as hysteresis, minimum durations stop flicker,
missing-ball frames carry flat evidence so the label holds. Switch offsets measured on the
public GT are applied here.
Owner: person 3.
"""


def decode(scores, half, params):
    raise NotImplementedError

