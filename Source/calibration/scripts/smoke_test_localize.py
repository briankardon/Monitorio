"""Manual smoke test: baselines + structured-light coarse localization (piece 4).

Full chain up to this point: detect live PDs, then run Gray-coded stripe
patterns along X and Y to localize each to within 2**(k_min-1) pixels.
Visual verification at the end shows white circles on screen at each
detected position -- the circles should overlap the physical PDs.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\smoke_test_localize.py \\
        [--display N] [--device NAME] [--channels N] [--sample-rate HZ] \\
        [--k-min K] [--settle SEC]

Defaults: display 0, first DAQ, all channels fit at DC sample rate,
          5 kHz DC sample rate, k_min=5 (32-px stripes), 200 ms settle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the calibration package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from calibration.daq import DAQ, list_ai_channels, list_devices
from calibration.display import Display, list_displays
from calibration.procedure import (
    DEFAULT_DC_SAMPLE_RATE,
    DEFAULT_K_MIN,
    DEFAULT_SETTLE_TIME_S,
    characterize_baselines,
    localize_coarse,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--display", type=int, default=0, help="display index")
    p.add_argument("--device", type=str, default=None, help="DAQ device name")
    p.add_argument("--channels", type=int, default=None, help="channels to scan")
    p.add_argument(
        "--sample-rate", type=float, default=None, dest="sample_rate",
        help="per-channel DC sample rate (Hz)",
    )
    p.add_argument(
        "--k-min", type=int, default=DEFAULT_K_MIN, dest="k_min",
        help="finest stripe bit (2**k_min px); must exceed PD diameter",
    )
    p.add_argument(
        "--settle", type=float, default=DEFAULT_SETTLE_TIME_S,
        help="seconds between each screen flip and sampling",
    )
    return p.parse_args(argv)


def _both(display, msg):
    print(msg)
    display.message(msg + "\n\nPress any key to continue. ESC to quit.")
    display.flip()
    return display.wait_for_key()  # True on ESC


def main() -> int:
    args = _parse_args(sys.argv[1:])

    displays = list_displays()
    devices = list_devices()
    if not devices or not displays:
        print("No DAQ or display found.")
        return 1

    device_name = args.device if args.device is not None else devices[0]
    all_chans = list_ai_channels(device_name)
    effective_rate = (
        args.sample_rate if args.sample_rate is not None
        else DEFAULT_DC_SAMPLE_RATE
    )

    with DAQ(device_name) as daq:
        max_at_rate = int(daq.max_multi_channel_rate // effective_rate)
        default_n = min(len(all_chans), max_at_rate)
        n_channels = args.channels if args.channels is not None else default_n
        n_channels = min(n_channels, len(all_chans))
        channels = tuple(all_chans[:n_channels])

        print(
            f"Display {args.display}, DAQ {device_name} ({daq.product_type}), "
            f"{n_channels} channel(s) @ {effective_rate:.0f} Hz, "
            f"k_min={args.k_min}, settle={args.settle*1000:.0f} ms"
        )

        with Display(args.display) as display:
            if _both(
                display,
                "Structured-light localization\n\n"
                "Step 1: baselines (noise floor + full-white response).\n"
                "Step 2: Gray-coded stripe patterns along X and Y to\n"
                "        localize each live photodiode to within\n"
                f"        +/- {(1 << args.k_min) // 2} pixels.\n\n"
                "Do not cover the photodiodes during this test.",
            ):
                return 0

            print("\nRunning baselines...")
            baseline = characterize_baselines(
                display, daq, channels=channels,
                settle_time=args.settle, sample_rate=effective_rate,
            )
            live_idx = np.flatnonzero(baseline.liveness())
            live_channels = [baseline.channels[i] for i in live_idx]
            print(
                f"  Found {len(live_channels)} live channel(s): {live_channels}"
            )
            if not live_channels:
                display.message(
                    "No live channels found.\n\n"
                    "Check photodiode wiring, monitor brightness,\n"
                    "and the baseline liveness threshold."
                )
                display.flip()
                display.wait_for_key()
                return 1

            if _both(
                display,
                f"{len(live_channels)} live channel(s) found:\n"
                + "\n".join("  " + c for c in live_channels)
                + "\n\nNow displaying Gray-coded stripe patterns along X and Y.",
            ):
                return 0

            print("\nRunning coarse localization...")
            locs = localize_coarse(
                display, daq, baseline,
                k_min=args.k_min, channels=live_channels,
                settle_time=args.settle, sample_rate=effective_rate,
            )

            # Terminal report.
            print(
                f"\nCoarse localization results "
                f"(+/- {locs.uncertainty_px} px):"
            )
            header = f"  {'channel':<12}{'x (px)':>10}{'y (px)':>10}{'min_conf':>12}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for i, ch in enumerate(locs.channels):
                print(
                    f"  {ch:<12}{locs.x_pixels[i]:>10}"
                    f"{locs.y_pixels[i]:>10}{locs.min_confidence[i]:>12.2f}"
                )

            # Visual verification: draw a circle at each detected position.
            # If the detected positions are correct, each circle lights up
            # exactly over its physical photodiode.
            short = [c.split("/")[-1] for c in locs.channels]
            radius_px = max(locs.uncertainty_px, 20)
            points = list(zip(
                locs.x_pixels.tolist(),
                locs.y_pixels.tolist(),
                short,
            ))
            display.annotated_points(points, radius=radius_px)
            display.flip()
            print(
                "\nVisual verification: circles should overlap the physical "
                "photodiodes.\n(Press any key to continue.)"
            )
            display.wait_for_key()

            # On-screen results summary.
            lines = [f"Localization complete  (+/- {locs.uncertainty_px} px)", ""]
            lines.append(f"{'channel':<10}{'x':>8}{'y':>8}{'conf':>8}")
            for i, ch in enumerate(locs.channels):
                lines.append(
                    f"{short[i]:<10}"
                    f"{locs.x_pixels[i]:>8d}"
                    f"{locs.y_pixels[i]:>8d}"
                    f"{locs.min_confidence[i]:>8.2f}"
                )
            lines.append("")
            lines.append("Press any key to exit.")
            display.message("\n".join(lines), size=max(18, display.height // 38))
            display.flip()
            display.wait_for_key()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
