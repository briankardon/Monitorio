"""High-level orchestration of the Monitorio calibration procedure.

Combines `calibration.display` and `calibration.daq` to run the real
measurements. This module grows piece-by-piece; at the moment it exposes
only the baseline characterization step (piece 3):

  - Measure noise floor on a fully black screen.
  - Measure response on a fully white screen.
  - Derive per-channel dynamic range, SNR, and a liveness flag.

Later pieces will add structured-light localization, centroid refinement,
rise-time measurement, and diameter sweep.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from calibration import gray
from calibration.daq import DAQ, Acquisition
from calibration.display import Display


# Default time between `Display.flip()` and the start of DAQ sampling. Gives
# the monitor's pixel response a chance to settle; LCDs want this on the
# order of 100-200 ms (slow pixel ramp + PWM backlight), OLEDs need only
# a few ms. Caller can override per-call or via the orchestration functions.
DEFAULT_SETTLE_TIME_S = 0.2

# Default DAQ sampling window for each per-pattern measurement.
DEFAULT_WINDOW_S = 0.1

# Default per-channel sample rate for DC-level measurements (baselines,
# localization, crosstalk). At 50 kHz per channel the PCIe-6343's mux has
# only ~2 us per channel to settle between neighbors, which leaves a few
# mV of residue from the previous channel on even a low-impedance input.
# 5 kHz gives 10x the settle budget and drops the residue below the
# noise floor, making liveness/localization results insensitive to how
# many channels are simultaneously scanned. Rise-time measurements don't
# use this default -- they need high rate and are run single-channel to
# sidestep the mux entirely.
DEFAULT_DC_SAMPLE_RATE = 5000.0

# Signal-to-noise ratio above which a channel is considered "live" --
# i.e. actually reading light off the screen, not a floating input.
# A channel with dynamic range < LIVENESS_NOISE_STDS * dark_noise_std is
# treated as dead. Very loose because at DEFAULT_DC_SAMPLE_RATE we expect
# a >1000x gap between connected channels and both unconnected channels
# and any residual mux ghosting.
DEFAULT_LIVENESS_THRESHOLD_STDS = 10.0


def measure_after_render(
    display: Display,
    daq: DAQ,
    draw: Callable[[Display], None],
    *,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    channels: tuple[str, ...] | list[str] | None = None,
    sample_rate: float | None = None,
) -> Acquisition:
    """Render a pattern, wait for pixel response to settle, then sample the DAQ.

    draw:        callable that issues Display draw commands. Must not call
                 flip() itself -- this function does the flip after draw.
    settle_time: seconds to wait between flip() and starting sampling.
    duration:    seconds of DAQ sampling.

    The monitor contents on return are whatever `draw` produced.
    """
    draw(display)
    display.flip()
    if settle_time > 0:
        time.sleep(settle_time)
    return daq.acquire(duration=duration, channels=channels, sample_rate=sample_rate)


@dataclass(frozen=True)
class BaselineResult:
    """Dark/bright baseline measurement, plus derived per-channel stats."""

    dark: Acquisition
    bright: Acquisition

    @property
    def channels(self) -> tuple[str, ...]:
        return self.dark.channels

    def dark_mean(self) -> np.ndarray:
        return self.dark.mean()

    def dark_std(self) -> np.ndarray:
        return self.dark.std()

    def bright_mean(self) -> np.ndarray:
        return self.bright.mean()

    def dynamic_range(self) -> np.ndarray:
        """Per-channel (bright_mean - dark_mean) in volts."""
        return self.bright.mean() - self.dark.mean()

    def snr(self) -> np.ndarray:
        """Per-channel dynamic_range / dark_std.

        dark_std is clamped at a small positive floor so a pathologically
        quiet floating input doesn't produce an Inf.
        """
        noise = np.maximum(self.dark_std(), 1e-9)
        return self.dynamic_range() / noise

    def liveness(
        self, threshold_stds: float = DEFAULT_LIVENESS_THRESHOLD_STDS,
    ) -> np.ndarray:
        """Bool mask: which channels respond to the screen state.

        A channel is "live" iff its dynamic range exceeds threshold_stds
        multiples of its own dark-noise std. With threshold_stds=10 and
        the >100x signal/noise gap we see between connected and floating
        channels, this is effectively binary.
        """
        return self.dynamic_range() > threshold_stds * self.dark_std()


# Finest Gray-code stripe bit used by localize_coarse(). Bit k produces
# 2**k -pixel stripes; if this drops below the photodiode's effective
# diameter on the screen, each PD averages across multiple stripes and
# the bit decode becomes random. A 1206 PD on a typical HD monitor spans
# roughly 10-20 pixels, so k_min=5 (32-px stripes) gives comfortable
# headroom. Localization uncertainty is +/- 2**(k_min-1) pixels, refined
# later by a bar sweep (piece 5).
DEFAULT_K_MIN = 5


def characterize_baselines(
    display: Display,
    daq: DAQ,
    *,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    channels: tuple[str, ...] | list[str] | None = None,
    sample_rate: float | None = None,
) -> BaselineResult:
    """Measure per-channel noise floor (black screen) and full-scale response (white screen).

    Returns a BaselineResult from which the caller can derive dynamic
    range, SNR, and a liveness mask. Leaves the screen black when done so
    subsequent measurements start from a defined state.

    sample_rate defaults to DEFAULT_DC_SAMPLE_RATE (slower than the DAQ's
    default) because DC measurements don't benefit from fast scanning and
    the slower rate eliminates multiplexer settling crosstalk that would
    otherwise show up as small ghost responses on unconnected channels.
    """
    rate = sample_rate if sample_rate is not None else DEFAULT_DC_SAMPLE_RATE
    dark = measure_after_render(
        display, daq, lambda d: d.black(),
        settle_time=settle_time, duration=duration,
        channels=channels, sample_rate=rate,
    )
    bright = measure_after_render(
        display, daq, lambda d: d.white(),
        settle_time=settle_time, duration=duration,
        channels=channels, sample_rate=rate,
    )
    # Restore black so the next step doesn't leak white light into whatever
    # measurement follows.
    display.black()
    display.flip()
    return BaselineResult(dark=dark, bright=bright)


@dataclass(frozen=True)
class CoarseLocations:
    """Per-channel (x, y) pixel localization from structured-light patterns.

    x_pixels, y_pixels are integer pixel coordinates -- block centers from
    the Gray-code decode. True PD center lies within +/- uncertainty_px of
    the reported value (uniform distribution, assuming the PD is a point).

    min_confidence: per-channel lowest bit-decode confidence across all
    patterns, normalized to [0, 1] where 1 is cleanly above or below
    threshold and 0 is exactly at threshold. Channels with values near 0
    had at least one pattern where the PD straddled a stripe boundary;
    the bit could have gone either way, and the reported coordinate may
    be off by a block on that axis.
    """

    channels: tuple[str, ...]
    x_pixels: np.ndarray
    y_pixels: np.ndarray
    uncertainty_px: int
    min_confidence: np.ndarray


def localize_coarse(
    display: Display,
    daq: DAQ,
    baseline: BaselineResult,
    *,
    k_min: int = DEFAULT_K_MIN,
    channels: tuple[str, ...] | list[str] | None = None,
    settle_time: float = DEFAULT_SETTLE_TIME_S,
    duration: float = DEFAULT_WINDOW_S,
    sample_rate: float | None = None,
) -> CoarseLocations:
    """Locate each live photodiode to within 2**k_min pixels via structured light.

    For each axis, displays ceil(log2(axis_length)) - k_min + 1 Gray-coded
    stripe patterns at successively finer bit widths. Each PD's mean
    response to each pattern is thresholded at the midpoint of its
    baseline (dark_mean + bright_mean) / 2 to recover one bit of its
    Gray-coded position. The stack of bits is decoded into a block index,
    and the PD is reported at the center of that block.

    channels: which channels to localize. Defaults to baseline.liveness() --
        i.e. only channels that showed a real dark-to-bright response.
    k_min: finest pattern bit. See DEFAULT_K_MIN above for tuning.
    """
    rate = sample_rate if sample_rate is not None else DEFAULT_DC_SAMPLE_RATE

    # Pick channels to localize.
    if channels is None:
        live_mask = baseline.liveness()
        target_channels = tuple(
            c for c, live in zip(baseline.channels, live_mask) if live
        )
    else:
        target_channels = tuple(channels)
    if not target_channels:
        raise ValueError("No channels to localize (baseline found no live channels)")

    # Per-target dark/bright/threshold.
    baseline_idx = [baseline.channels.index(c) for c in target_channels]
    dark_m = baseline.dark_mean()[baseline_idx]
    bright_m = baseline.bright_mean()[baseline_idx]
    thresholds = (dark_m + bright_m) / 2.0
    half_range = np.maximum((bright_m - dark_m) / 2.0, 1e-9)

    n_live = len(target_channels)

    def scan_axis(axis: str, length: int):
        """Return (pixel_positions, min_confidence) for one axis."""
        K = math.ceil(math.log2(length)) - 1
        if K < k_min:
            raise ValueError(
                f"axis length {length} is too short for k_min={k_min} "
                f"(requires at least 2**{k_min + 1}={2 ** (k_min + 1)} pixels)"
            )
        n_bits = K - k_min + 1
        bits = np.zeros((n_live, n_bits), dtype=np.int64)
        confidences = np.zeros((n_live, n_bits))

        # Iterate coarsest bit (K) to finest (k_min). idx 0 is MSB of G_top.
        for idx, bit in enumerate(range(K, k_min - 1, -1)):
            acq = measure_after_render(
                display, daq,
                lambda d, b=bit: d.gray_stripes(axis, b),
                settle_time=settle_time, duration=duration,
                channels=target_channels, sample_rate=rate,
            )
            resp = acq.mean()  # (n_live,)
            bits[:, idx] = (resp > thresholds).astype(np.int64)
            confidences[:, idx] = np.abs(resp - thresholds) / half_range

        # Pack bits into G_top (idx 0 is MSB, idx n_bits-1 is LSB).
        g_top = np.zeros(n_live, dtype=np.int64)
        for idx in range(n_bits):
            g_top |= bits[:, idx] << (n_bits - 1 - idx)

        # G_top == grayEncode(block_index) -- see the derivation in
        # calibration.gray's module docstring.
        block_indices = gray.decode(g_top, n_bits=n_bits)

        # Report the center of each 2**k_min-wide block.
        positions = block_indices * (1 << k_min) + (1 << (k_min - 1))
        positions = np.clip(positions, 0, length - 1)

        return positions.astype(np.int64), confidences.min(axis=1)

    x_pixels, x_conf = scan_axis("x", display.width)
    y_pixels, y_conf = scan_axis("y", display.height)

    # Leave the screen black so subsequent steps start from a defined state.
    display.black()
    display.flip()

    return CoarseLocations(
        channels=target_channels,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        uncertainty_px=(1 << k_min) // 2,
        min_confidence=np.minimum(x_conf, y_conf),
    )
