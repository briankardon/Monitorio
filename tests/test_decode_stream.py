"""Tests for decode_stream's testable helpers.

Full end-to-end coverage requires an RHD recording + matching tagged
video + calibration; that's exercised manually against the user's
test data. Here we cover the orchestration helpers: file-coverage
selection, file-range extension, playback-log parsing, and the
loader-side filename-timestamp parser.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from decode_stream import (
    LOADERS, _extend_file_range, _files_covering, _read_playback_log,
)
from loaders.rhd import parse_intan_filename_timestamp, RecordingBundle


def test_loader_registry_has_known_entries():
    assert "rhd" in LOADERS
    assert callable(LOADERS["rhd"]["load"])
    assert LOADERS["rhd"]["glob"] == "*.rhd"


def test_parse_intan_filename_timestamp_standard():
    p = Path("CUNY09_260501_160904.rhd")
    ts = parse_intan_filename_timestamp(p)
    assert ts == datetime.datetime(2026, 5, 1, 16, 9, 4)


def test_parse_intan_filename_timestamp_other_basename():
    # Different rig prefix, same convention -- still parses.
    p = Path("RIG_A_240101_000000.rhd")
    assert parse_intan_filename_timestamp(p) == datetime.datetime(2024, 1, 1, 0, 0, 0)


def test_parse_intan_filename_timestamp_nonconforming():
    # Anything that doesn't match the convention returns None.
    assert parse_intan_filename_timestamp(Path("recording.rhd")) is None
    assert parse_intan_filename_timestamp(Path("CUNY09_oldformat.rhd")) is None
    assert parse_intan_filename_timestamp(Path("CUNY09_260501_999999.rhd")) is None


def test_files_covering_picks_overlapping_range():
    # 6 files starting at 1-minute intervals.
    starts = [
        datetime.datetime(2026, 5, 1, 16, m, 0)
        for m in range(0, 6)
    ]
    paths = [Path(f"f{i}.rhd") for i in range(6)]
    # Window 16:02:00 -> 16:03:30 covers files 2 and 3 (file 2 spans
    # [16:02, 16:03), file 3 spans [16:03, 16:04)).
    cov = _files_covering(
        paths, starts,
        datetime.datetime(2026, 5, 1, 16, 2, 0),
        datetime.datetime(2026, 5, 1, 16, 3, 30),
    )
    assert cov == [2, 3]


def test_files_covering_empty_when_window_before_first():
    starts = [datetime.datetime(2026, 5, 1, 16, m, 0) for m in range(0, 3)]
    paths = [Path(f"f{i}.rhd") for i in range(3)]
    cov = _files_covering(
        paths, starts,
        datetime.datetime(2026, 5, 1, 15, 0, 0),
        datetime.datetime(2026, 5, 1, 15, 30, 0),
    )
    assert cov == []


def test_files_covering_last_file_extends_indefinitely():
    """The last file is treated as covering until +infinity (we
    don't know its actual duration without parsing it). This means a
    window after the last file's start is still covered."""
    starts = [datetime.datetime(2026, 5, 1, 16, m, 0) for m in range(0, 3)]
    paths = [Path(f"f{i}.rhd") for i in range(3)]
    cov = _files_covering(
        paths, starts,
        datetime.datetime(2026, 5, 1, 17, 0, 0),
        datetime.datetime(2026, 5, 1, 17, 30, 0),
    )
    assert cov == [2]   # only the last


def test_extend_file_range_before_and_after():
    assert _extend_file_range([3, 4], 10, "before") == [2, 3, 4]
    assert _extend_file_range([3, 4], 10, "after") == [3, 4, 5]


def test_extend_file_range_clamped_at_boundaries():
    # At the front: can't extend before file 0.
    assert _extend_file_range([0, 1], 10, "before") == [0, 1]
    # At the back: can't extend past the last file.
    assert _extend_file_range([8, 9], 10, "after") == [8, 9]


def test_read_playback_log_skips_comment_header(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text(
        "# session banner\n"
        "# config_path: cfg.toml\n"
        "play_index,start_time_iso,start_time_unix,video_path,"
        "duration_seconds,frames_shown,expected_frames,ivi_seconds,"
        "aborted,vlc_state,vlc_error\n"
        "1,2026-05-01T12:00:00,1000.0,/path/to/v1.mp4,10.5,,451,5.0,false,Ended,\n"
        "2,2026-05-01T12:00:30,1030.0,/path/to/v2.mp4,9.8,,451,3.5,false,Ended,\n"
    )
    rows = _read_playback_log(log)
    assert len(rows) == 2
    assert rows[0]["index"] == 1
    assert rows[0]["video"] == "/path/to/v1.mp4"
    assert rows[0]["started_unix"] == 1000.0
    assert rows[0]["expected_duration_s"] == 10.5
    assert rows[1]["index"] == 2


def test_read_playback_log_parses_real_session_format(tmp_path):
    """The log written by play_random.py has many comment lines
    (full config snapshot, hash, git commit, etc.) -- make sure they
    all get skipped cleanly."""
    log = tmp_path / "log.csv"
    log.write_text(
        "# ----------------------------------------------------------------------\n"
        "# Monitorio playback session\n"
        "# session_started_utc: 2026-05-01T23:16:40+00:00\n"
        "# config_path:         /path/to/cfg.toml\n"
        "# config_sha256_12:    abc123def456\n"
        "# script_git_hash:     1234567890ab\n"
        "# python:              3.13 (win32)\n"
        "# --- begin config file snapshot ---\n"
        "# videos = [\"v1.mp4\"]\n"
        "# [timing]\n"
        "# n_plays = 1\n"
        "# --- end config file snapshot ---\n"
        "# ----------------------------------------------------------------------\n"
        "play_index,start_time_iso,start_time_unix,video_path,"
        "duration_seconds,frames_shown,expected_frames,ivi_seconds,"
        "aborted,vlc_state,vlc_error\n"
        "1,2026-05-01T23:17:00,1745000000.0,/path/v.mp4,10.0,,451,5.0,false,Ended,\n"
    )
    rows = _read_playback_log(log)
    assert len(rows) == 1
    assert rows[0]["index"] == 1


def test_decode_one_playback_applies_clock_offset(monkeypatch, tmp_path):
    """clock_offset_s should shift the expected wall-clock time the
    matcher uses, without otherwise changing behavior. We verify by
    intercepting the inner _attempt_decode_window and checking the
    expected_start_dt that gets passed in."""
    import datetime
    from decode_stream import StreamResult, _decode_one_playback

    captured = {}

    def fake_attempt_decode_window(**kwargs):
        captured["expected_start_dt"] = kwargs["expected_start_dt"]
        return None, None, None

    monkeypatch.setattr(
        "decode_stream._attempt_decode_window",
        fake_attempt_decode_window,
    )

    play = {
        "index": 1,
        "video": "/v.mp4",
        "started_unix": 1000.0,
        "started_iso": "...",
        "expected_duration_s": 10.0,
    }
    # Two files an hour apart -- pick whichever happens to "cover" the
    # offsetted window; we don't care about the actual decode.
    files = [tmp_path / "f0.rhd", tmp_path / "f1.rhd"]
    for f in files:
        f.touch()
    file_starts = [
        datetime.datetime.fromtimestamp(900.0),    # 100 s before play
        datetime.datetime.fromtimestamp(60_000.0),   # well after play
    ]

    # No offset: expected_start_dt should be at started_unix.
    _decode_one_playback(
        play=play,
        all_files=files, file_starts=file_starts,
        load_func=lambda paths: None,
        pd_channels=[0],
        calibration_path=Path("/cal.json"),
        margin_s=5.0, drift_warn_s=1.0,
        sample_rate_for_units=1.0,
        clock_offset_s=0.0,
        output_csv=None,
    )
    assert captured["expected_start_dt"] == datetime.datetime.fromtimestamp(1000.0)

    # +30 offset: expected_start_dt should advance by 30 s.
    _decode_one_playback(
        play=play,
        all_files=files, file_starts=file_starts,
        load_func=lambda paths: None,
        pd_channels=[0],
        calibration_path=Path("/cal.json"),
        margin_s=5.0, drift_warn_s=1.0,
        sample_rate_for_units=1.0,
        clock_offset_s=30.0,
        output_csv=None,
    )
    assert captured["expected_start_dt"] == datetime.datetime.fromtimestamp(1030.0)

    # Negative offset works too.
    _decode_one_playback(
        play=play,
        all_files=files, file_starts=file_starts,
        load_func=lambda paths: None,
        pd_channels=[0],
        calibration_path=Path("/cal.json"),
        margin_s=5.0, drift_warn_s=1.0,
        sample_rate_for_units=1.0,
        clock_offset_s=-2.5,
        output_csv=None,
    )
    assert captured["expected_start_dt"] == datetime.datetime.fromtimestamp(997.5)


def test_recording_bundle_unpacks_as_4_tuple():
    """RecordingBundle should iterate as (samples, rate, names,
    boundaries) so existing 4-element tuple unpacking works."""
    import numpy as np
    from loaders.rhd import FileBoundary
    bundle = RecordingBundle(
        samples=np.zeros((3, 100)),
        sample_rate=20000.0,
        channel_names=["a", "b", "c"],
        file_boundaries=[FileBoundary(
            path=Path("x.rhd"),
            start_wall_clock=datetime.datetime(2026, 1, 1),
            n_samples=100, sample_offset_in_concat=0,
        )],
    )
    samples, rate, names, boundaries = bundle
    assert samples.shape == (3, 100)
    assert rate == 20000.0
    assert names == ["a", "b", "c"]
    assert len(boundaries) == 1
    # And attribute access still works.
    assert bundle.samples.shape == (3, 100)
    assert bundle.file_boundaries[0].path == Path("x.rhd")
