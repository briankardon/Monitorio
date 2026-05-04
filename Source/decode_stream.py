"""Batch-decode every playback in a session against the matching recording.

Pairs a playback log produced by `Source/playback/play_random.py` with
the recording files captured during that session. For each row in the
log:

  1. Identify the recording file(s) whose wall-clock coverage overlaps
     the playback's expected time window (using the loader's per-file
     start-time metadata + the playback's logged start_time_unix).
  2. Load that span; find sync-on segment(s) in the configured PD
     channels.
  3. Pick the segment closest to the expected timestamp (warn / extend
     the file range if no candidate falls within the initial window or
     the segment hits the loaded data's start/end -- means the
     timestamp pointed at a region that's actually in an adjacent
     file).
  4. Run decode_sync_tags on a tight window around that segment.
  5. Wrap the result + diagnostics in a StreamResult.

Returns a list of StreamResults (one per playback log entry) and
optionally writes a per-playback decode CSV plus a summary CSV.

Loader registry is open-ended: a loader is `(load_func, file_glob)`.
RHD is the only entry today; adding TDMS / OpenEphys / etc. is one
more registry line + a small helper to match the loader's interface
(must return something with a `samples` array, `sample_rate`,
`channel_names`, and `file_boundaries` with wall-clock starts and
sample offsets).

CLI:
    venv/Scripts/python Source/decode_stream.py \\
        playback_log.csv recording_dir/ \\
        --loader rhd --pd-channels 2,3,4 \\
        --calibration calibration.json \\
        --output-dir decoded/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Source/ on path so we can import siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_sync_tags import DecodeResult, decode_sync_tags
from loaders.rhd import (
    RecordingBundle, file_start_wall_clock,
    load_rhd_aux, load_rhd_board_adc,
)


# ----- loader registry ------------------------------------------------

LOADERS: dict[str, dict] = {
    "rhd": {
        "load": load_rhd_board_adc,
        "glob": "*.rhd",
        "describe": "Intan RHD2000 controller-box analog inputs",
    },
    "rhd_aux": {
        "load": load_rhd_aux,
        "glob": "*.rhd",
        "describe": "Intan RHD2000 headstage AUX inputs",
    },
}


@dataclass
class StreamResult:
    """One row of decode_stream's output -- one playback's outcome.

    status: 'decoded' (clean / soft warnings only)
            'failed'  (decoder raised, or no segment found)
            'missing' (recording doesn't cover this playback's time)
    """
    playback_index: int
    playback_video: Path
    playback_started_unix: float
    playback_started_iso: str
    expected_duration_s: float
    status: str = "pending"
    decode_result: DecodeResult | None = None
    decoded_csv_path: Path | None = None
    # Where in the concatenated samples the segment landed (None if
    # we never even tried decoding):
    segment_start_in_concat: int | None = None
    segment_end_in_concat: int | None = None
    # Wall-clock offset between expected and actual segment start.
    # Positive means the actual segment started later than expected.
    segment_drift_s: float | None = None
    # Files actually loaded for this playback, in load order.
    files_loaded: list[Path] = field(default_factory=list)
    reason: str = ""
    warnings_: list[str] = field(default_factory=list)


# ----- public API -----------------------------------------------------

def decode_stream(
    playback_log: Path,
    recording_dir: Path,
    *,
    loader: str = "rhd",
    pd_channels: list[int],
    calibration_path: Path,
    output_dir: Path | None = None,
    margin_s: float = 5.0,
    drift_warn_s: float = 1.0,
    sample_rate_for_units: float | str = 1.0,
    clock_offset_s: float = 0.0,
) -> list[StreamResult]:
    """Decode every playback in `playback_log` against recordings in
    `recording_dir`.

    pd_channels: which rows of the loaded `samples` array to pass to
                 the decoder, in the order the calibration JSON's
                 photodiode list expects (sync first if sync_bit is on,
                 then frame bits). Read out of the recording's full
                 channel list -- e.g. [2, 3, 4] for ANALOG-IN-4/5/6
                 when ANALOG-IN-1/2/3 are also enabled but not PDs.
    calibration_path: passed straight through to decode_sync_tags.
    output_dir: if given, per-playback CSVs (one per successful decode)
                and a summary CSV go here.
    margin_s: extra wall-clock margin on each side of the expected
              playback window when picking which files to load + when
              extracting the sub-window for decode_sync_tags.
    drift_warn_s: warn if the actual sync-on segment starts more than
                  this far from the expected wall-clock time (after
                  any clock_offset_s correction).
    sample_rate_for_units: passed to decode_sync_tags as `scale`. The
                  RHD loader returns samples in volts, so the default
                  scale=1.0 is right; this knob is here in case someone
                  wires a non-volt-returning loader into the registry.
    clock_offset_s: clock offset between the playback driver's host
                  and the recording controller's host, defined as
                  (recording_clock_seconds - playback_clock_seconds).
                  When the two run on the same machine (the common
                  case) this is 0. When they're on separate machines
                  whose system clocks aren't NTP-synced, set this to
                  the offset so the playback log's wall-clock
                  timestamps line up with the recording filenames'
                  wall-clock timestamps. Positive value: the
                  recording machine's clock runs ahead of the
                  playback machine's. Empirical recipe: run once with
                  clock_offset_s=0, look at the segment_drift_s
                  column of the summary CSV; if every row reports a
                  drift of about the same value, that's the
                  offset to plug in here. Drift after correction
                  should be tens to hundreds of ms (player startup
                  lag, filename-timestamp-vs-actual-sample-0 lag).
    """
    if loader not in LOADERS:
        raise ValueError(
            f"unknown loader {loader!r}; choose from {list(LOADERS)}"
        )
    loader_entry = LOADERS[loader]
    load_func = loader_entry["load"]
    glob_pattern = loader_entry["glob"]

    recording_dir = Path(recording_dir)
    if not recording_dir.is_dir():
        raise NotADirectoryError(f"recording_dir is not a directory: {recording_dir}")

    # All recording files in the dir, sorted by their parsed wall-clock
    # start. (Sort by start, not filename, so unconventional naming
    # still orders correctly.)
    all_files = sorted(
        recording_dir.glob(glob_pattern),
        key=lambda p: file_start_wall_clock(p),
    )
    if not all_files:
        raise FileNotFoundError(
            f"no {glob_pattern} files in {recording_dir}"
        )
    file_starts = [file_start_wall_clock(p) for p in all_files]
    print(
        f"[decode_stream] {len(all_files)} {glob_pattern} files in "
        f"{recording_dir}: {file_starts[0]} .. {file_starts[-1]}",
        file=sys.stderr,
    )

    # Parse the playback log.
    plays = _read_playback_log(playback_log)
    print(
        f"[decode_stream] {len(plays)} playback rows in {playback_log}",
        file=sys.stderr,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if clock_offset_s != 0:
        print(
            f"[decode_stream] applying clock_offset_s={clock_offset_s:+.4f} "
            f"(playback timestamps shifted to match the recording host's clock)",
            file=sys.stderr,
        )

    results: list[StreamResult] = []
    for play in plays:
        print(
            f"[decode_stream] playback #{play['index']}: "
            f"{Path(play['video']).name} at {play['started_iso']}",
            file=sys.stderr,
        )
        per_play_csv = (
            output_dir / f"play_{play['index']:03d}_decoded.csv"
            if output_dir is not None else None
        )
        result = _decode_one_playback(
            play=play,
            all_files=all_files,
            file_starts=file_starts,
            load_func=load_func,
            pd_channels=pd_channels,
            calibration_path=calibration_path,
            margin_s=margin_s,
            drift_warn_s=drift_warn_s,
            sample_rate_for_units=sample_rate_for_units,
            clock_offset_s=clock_offset_s,
            output_csv=per_play_csv,
        )
        _print_play_summary(result)
        results.append(result)

    if output_dir is not None:
        _write_summary(output_dir / "decode_stream_summary.csv", results)

    return results


# ----- per-playback driver --------------------------------------------

def _decode_one_playback(
    *,
    play: dict,
    all_files: list[Path],
    file_starts: list[datetime.datetime],
    load_func,
    pd_channels: list[int],
    calibration_path: Path,
    margin_s: float,
    drift_warn_s: float,
    sample_rate_for_units,
    clock_offset_s: float,
    output_csv: Path | None,
) -> StreamResult:
    base = StreamResult(
        playback_index=play["index"],
        playback_video=Path(play["video"]),
        playback_started_unix=play["started_unix"],
        playback_started_iso=play["started_iso"],
        expected_duration_s=play["expected_duration_s"],
    )

    # Convert the playback's wall-clock timestamp into the recording
    # host's clock by adding clock_offset_s. (Recording files' wall-
    # clock anchors come from their filename timestamps, written by
    # the recording controller's clock.)
    expected_start_dt = datetime.datetime.fromtimestamp(
        play["started_unix"] + clock_offset_s,
    )
    expected_end_dt = expected_start_dt + datetime.timedelta(
        seconds=play["expected_duration_s"],
    )
    window_start = expected_start_dt - datetime.timedelta(seconds=margin_s)
    window_end = expected_end_dt + datetime.timedelta(seconds=margin_s)

    # Pick the initial set of files: any whose wall-clock span overlaps
    # [window_start, window_end). Files are sorted; we know they're
    # contiguous in time (the recorder writes one after another), so a
    # contiguous index range works.
    initial_idx = _files_covering(
        all_files, file_starts, window_start, window_end,
    )
    if not initial_idx:
        base.status = "missing"
        base.reason = (
            f"no recording file covers the playback's expected window "
            f"[{window_start}, {window_end}). Recording probably "
            f"started after this playback ended (or vice versa)."
        )
        return base

    # Try the initial window. If the chosen segment hits a boundary of
    # the loaded data (sample 0 or n_samples-1), extend by one file in
    # that direction and retry. This handles "the playback's signal
    # actually lived in an adjacent file we didn't think we needed."
    used_idx = list(initial_idx)
    for _ in range(3):  # at most a few extension attempts
        bundle, picked_segment, extension_needed = _attempt_decode_window(
            all_files=all_files, used_idx=used_idx,
            load_func=load_func, pd_channels=pd_channels,
            expected_start_dt=expected_start_dt,
            expected_duration_s=play["expected_duration_s"],
            margin_s=margin_s,
        )
        if extension_needed is None:
            break
        # Try to extend in the requested direction.
        new_idx = _extend_file_range(
            used_idx, len(all_files), extension_needed,
        )
        if new_idx == used_idx:
            # Already at a boundary; can't extend any further.
            break
        used_idx = new_idx
    else:
        bundle = None
        picked_segment = None

    base.files_loaded = [all_files[i] for i in used_idx]
    if bundle is None or picked_segment is None:
        base.status = "missing"
        base.reason = (
            f"no sync-on segment found within the recording's coverage "
            f"of this playback's expected window. Either the playback "
            f"didn't make it to the recording (player crashed, monitor "
            f"off, etc.) or the timestamp drift is larger than the "
            f"file range we explored."
        )
        return base

    seg_start_in_concat, seg_end_in_concat = picked_segment
    base.segment_start_in_concat = seg_start_in_concat
    base.segment_end_in_concat = seg_end_in_concat

    # Wall-clock drift between expected and actual segment start.
    bundle_start_dt = bundle.file_boundaries[0].start_wall_clock
    actual_seg_start_dt = bundle_start_dt + datetime.timedelta(
        seconds=seg_start_in_concat / bundle.sample_rate,
    )
    drift = (actual_seg_start_dt - expected_start_dt).total_seconds()
    base.segment_drift_s = drift
    if abs(drift) > drift_warn_s:
        base.warnings_.append(
            f"segment_drift_s={drift:+.3f}: actual segment start is "
            f"{abs(drift):.3f} s {'after' if drift > 0 else 'before'} "
            f"the playback log's expected wall-clock time. The decode "
            f"is fine, but a drift of more than ~1 s suggests system-"
            f"clock differences between the playback driver and the "
            f"recording controller -- worth checking."
        )

    # Extract the segment + margin and hand it to decode_sync_tags.
    margin_n = int(round(margin_s * bundle.sample_rate))
    extract_lo = max(0, seg_start_in_concat - margin_n)
    extract_hi = min(bundle.samples.shape[1], seg_end_in_concat + margin_n)
    extracted = bundle.samples[pd_channels, extract_lo:extract_hi].copy()

    try:
        result = decode_sync_tags(
            extracted,
            sample_rate=bundle.sample_rate,
            video_path=base.playback_video,
            calibration_path=calibration_path,
            scale=sample_rate_for_units,
            output_path=output_csv,
            metadata=(
                f"play_index={play['index']} "
                f"playback_started_iso={play['started_iso']} "
                f"segment_drift_s={drift:+.3f}"
            ),
        )
    except Exception as e:
        base.status = "failed"
        base.reason = f"decode_sync_tags raised: {e!r}"
        return base

    base.status = "decoded"
    base.decode_result = result
    base.decoded_csv_path = output_csv
    base.warnings_.extend(result.warnings_)
    return base


# ----- file selection + segment picking -------------------------------

def _files_covering(
    paths: list[Path],
    file_starts: list[datetime.datetime],
    window_start: datetime.datetime,
    window_end: datetime.datetime,
) -> list[int]:
    """Return the indices of files whose wall-clock span overlaps
    [window_start, window_end). Each file is assumed to last from its
    own start until the next file's start (the typical Intan
    rolling-N-minute-files behavior); the last file is treated as
    extending forever (we don't know how long it actually lasts
    without parsing it). The result is a contiguous run of indices.
    """
    covering = []
    for i, start in enumerate(file_starts):
        next_start = (
            file_starts[i + 1] if i + 1 < len(file_starts)
            else datetime.datetime.max
        )
        if start < window_end and next_start > window_start:
            covering.append(i)
    return covering


def _extend_file_range(
    indices: list[int], n_total: int, direction: str,
) -> list[int]:
    """Return a new index list extended by one file in `direction`
    ('before' or 'after'), clamped to [0, n_total).
    """
    if not indices:
        return indices
    lo, hi = indices[0], indices[-1]
    if direction == "before" and lo > 0:
        return list(range(lo - 1, hi + 1))
    if direction == "after" and hi + 1 < n_total:
        return list(range(lo, hi + 2))
    return indices


def _attempt_decode_window(
    *, all_files, used_idx, load_func, pd_channels,
    expected_start_dt, expected_duration_s, margin_s,
):
    """Load the files in used_idx, find sync-on segments in the PD
    channels, and pick the best candidate for this playback.

    Returns (bundle, (seg_start, seg_end), extension_needed):
      - bundle is the loaded RecordingBundle (or None if we should
        give up)
      - (seg_start, seg_end) are concatenated-sample indices of the
        chosen segment, or None if no candidate
      - extension_needed is None (we're happy with this load),
        'before' (segment hit sample-0 boundary), or 'after' (hit
        the end-of-loaded-data boundary). Caller can extend the file
        range and retry.
    """
    paths = [all_files[i] for i in used_idx]
    bundle = load_func(paths)
    pd_samples = bundle.samples[pd_channels]
    sample_rate = bundle.sample_rate

    # Sync threshold: midpoint between channel min and max. Coarse but
    # plenty for "is sync currently on?"
    sync = pd_samples[0]
    sync_thr = (float(sync.min()) + float(sync.max())) / 2.0
    on = sync > sync_thr
    if not on.any():
        # No sync activity at all in this load.
        return bundle, None, None

    # Run-length-encode the sync-on signal; keep runs of at least
    # 0.25 s. Robustness layer is in decode_sync_tags itself; here
    # we just want candidate segment start/end pairs.
    diff = np.diff(on.astype(np.int8))
    rising = np.where(diff == 1)[0] + 1
    falling = np.where(diff == -1)[0] + 1
    if on[0]:
        rising = np.concatenate(([0], rising))
    if on[-1]:
        falling = np.concatenate((falling, [on.size]))
    keep = (falling - rising) >= int(0.25 * sample_rate)
    rising, falling = rising[keep], falling[keep]
    if rising.size == 0:
        return bundle, None, None

    # Pick the candidate whose start time is closest to the playback's
    # expected wall-clock time.
    bundle_start_dt = bundle.file_boundaries[0].start_wall_clock
    expected_offset_in_concat = (
        (expected_start_dt - bundle_start_dt).total_seconds() * sample_rate
    )
    distances = np.abs(rising.astype(np.int64) - int(expected_offset_in_concat))
    best = int(np.argmin(distances))
    seg_start = int(rising[best])
    seg_end = int(falling[best])

    # Boundary check: if the chosen segment is right at the start or
    # end of the loaded data, the actual signal probably extends into
    # an adjacent file.
    margin_n = int(margin_s * sample_rate)
    extension_needed = None
    if seg_start < margin_n:
        extension_needed = "before"
    elif seg_end > pd_samples.shape[1] - margin_n:
        extension_needed = "after"
    return bundle, (seg_start, seg_end), extension_needed


# ----- log parsing ----------------------------------------------------

def _read_playback_log(path: Path) -> list[dict]:
    """Parse a playback log (per-session CSV with a `#`-prefixed
    header) into a list of dicts."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            # First non-comment line is the column header; subsequent
            # lines are data.
            cols = line.split(",")
            if rows == [] and "play_index" in cols:
                # column header
                rows = []  # rest of loop will be data rows
                header = cols
                continue
            try:
                # All data rows at this point
                row = dict(zip(header, cols))
                rows.append({
                    "index": int(row["play_index"]),
                    "video": row["video_path"],
                    "started_unix": float(row["start_time_unix"]),
                    "started_iso": row["start_time_iso"],
                    "expected_duration_s": float(row.get("duration_seconds") or 0.0),
                })
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"could not parse playback log row {cols!r}: {e!r}"
                ) from e
    return rows


# ----- output helpers -------------------------------------------------

def _print_play_summary(r: StreamResult) -> None:
    if r.status == "decoded":
        n = r.decode_result.frame_table.shape[0] if r.decode_result is not None else 0
        flag = " (with warnings)" if r.warnings_ else ""
        print(
            f"  -> decoded {n} frames"
            f"{f', drift {r.segment_drift_s:+.3f} s' if r.segment_drift_s is not None else ''}"
            f"{flag}",
            file=sys.stderr,
        )
        for w in r.warnings_:
            print(f"     warning: {w[:200]}", file=sys.stderr)
    elif r.status == "missing":
        print(f"  -> missing: {r.reason[:200]}", file=sys.stderr)
    else:
        print(f"  -> FAILED: {r.reason[:200]}", file=sys.stderr)


def _write_summary(path: Path, results: list[StreamResult]) -> None:
    """One row per playback: index, status, frame count, drift, etc."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "play_index", "status", "playback_video", "playback_started_iso",
            "n_frames_decoded", "expected_duration_s",
            "segment_drift_s", "n_warnings", "decoded_csv", "reason",
        ])
        for r in results:
            n = (
                r.decode_result.frame_table.shape[0]
                if (r.decode_result is not None) else ""
            )
            w.writerow([
                r.playback_index, r.status, r.playback_video,
                r.playback_started_iso, n, f"{r.expected_duration_s:.4f}",
                f"{r.segment_drift_s:+.4f}" if r.segment_drift_s is not None else "",
                len(r.warnings_),
                r.decoded_csv_path or "",
                r.reason,
            ])


# ----- CLI ------------------------------------------------------------

def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("playback_log", type=Path,
                   help="per-session playback log CSV (from play_random.py)")
    p.add_argument("recording_dir", type=Path,
                   help="directory of recording files")
    p.add_argument("--loader", default="rhd", choices=sorted(LOADERS),
                   help="which loader to use (default: rhd)")
    p.add_argument("--pd-channels", type=_csv_ints, required=True,
                   help="comma-separated 0-based indices of the loaded "
                        "channels carrying the PD signals, in calibration-"
                        "JSON order (sync first if sync_bit is on, then "
                        "frame bits). E.g. '2,3,4' for the 3rd-5th of 5 "
                        "loaded board ADC channels.")
    p.add_argument("--calibration", type=Path, required=True,
                   help="path to the calibration JSON for this rig")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="if given, write per-playback decode CSVs and a "
                        "summary CSV here")
    p.add_argument("--margin-s", type=float, default=5.0,
                   help="extra wall-clock margin on each side of the "
                        "expected playback window (default 5.0 s)")
    p.add_argument("--clock-offset-s", type=float, default=0.0,
                   help="clock offset between the playback machine's "
                        "host clock (which wrote the playback log's "
                        "timestamps) and the recording machine's host "
                        "clock (which named the recording files), "
                        "defined as recording_clock - playback_clock. "
                        "Default 0 (same machine). For separate-host "
                        "setups, run once with offset=0 and use the "
                        "median segment_drift_s from the summary CSV "
                        "as the offset for subsequent runs.")
    p.add_argument("--scale", default=1.0,
                   help="passed through to decode_sync_tags as scale "
                        "(default 1.0; 'intan_aux' / 'volts' etc. work)")
    args = p.parse_args(argv)

    # Try to parse scale as a float; fall back to string (preset name).
    try:
        scale = float(args.scale)
    except ValueError:
        scale = args.scale

    results = decode_stream(
        playback_log=args.playback_log,
        recording_dir=args.recording_dir,
        loader=args.loader,
        pd_channels=args.pd_channels,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        margin_s=args.margin_s,
        clock_offset_s=args.clock_offset_s,
        sample_rate_for_units=scale,
    )

    n_decoded = sum(1 for r in results if r.status == "decoded")
    n_missing = sum(1 for r in results if r.status == "missing")
    n_failed = sum(1 for r in results if r.status == "failed")
    print(
        f"\n[decode_stream] done: {n_decoded} decoded / {n_missing} "
        f"missing / {n_failed} failed (of {len(results)} playbacks)",
        file=sys.stderr,
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
