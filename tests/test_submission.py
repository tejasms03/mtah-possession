import numpy as np

from possession import config, submission


def test_written_file_is_valid(tmp_path):
    path = tmp_path / "sub.csv"
    submission.write(np.full(config.N_FRAMES, "H"), path)
    assert submission.validate(path) == []


def test_ungraded_frames_forced_to_x(tmp_path):
    path = tmp_path / "sub.csv"
    submission.write(np.full(config.N_FRAMES, "A"), path)
    lines = path.read_text().splitlines()
    assert lines[0] == "frame,label"
    assert lines[1] == "0,X"                      # before kickoff
    assert lines[11_185 + 1] == "11185,A"         # first graded frame
    assert lines[90_652 + 1] == "90652,X"         # halftime
    assert len(lines) == config.N_FRAMES + 1


def test_graded_frame_count():
    assert config.graded_mask().sum() == 158_857
