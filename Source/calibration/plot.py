"""Matplotlib visualizations for calibration results.

Pure presentation code -- takes in result dataclasses from
calibration.procedure and produces Figures. All functions accept
`show=True` to call plt.show() themselves (blocking) and/or
`save_path=...` to write a PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from calibration.procedure import CrosstalkResult, FineLocations, RiseTimeResult


def _short(channel_name: str) -> str:
    return channel_name.split("/")[-1]


def plot_refine(
    fine: FineLocations,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """One row per channel, two columns: X-axis sweep and Y-axis sweep.

    Each subplot shows response (V above dark) vs. bar position (px),
    with the detected centroid marked as a vertical line.
    """
    n = len(fine.channels)
    fig, axes = plt.subplots(
        n, 2, figsize=(11, 2.0 + 1.6 * n), sharex="col", squeeze=False,
    )
    fig.suptitle("Spatial response (bar-sweep refinement)", fontsize=12)

    for i, ch in enumerate(fine.channels):
        for col, (axis_name, sweeps, center, fwhm) in enumerate([
            ("x", fine.x_sweeps, fine.x_pixels[i], fine.x_fwhm_px[i]),
            ("y", fine.y_sweeps, fine.y_pixels[i], fine.y_fwhm_px[i]),
        ]):
            ax = axes[i, col]
            pos, resp = sweeps[i]
            ax.plot(pos, resp, "-", linewidth=1.2)
            ax.axhline(0, color="0.7", linewidth=0.7)
            if np.isfinite(center):
                ax.axvline(
                    center, color="tab:red", linestyle="--", linewidth=1,
                    label=f"centroid = {center:.2f}",
                )
                ax.legend(loc="upper right", fontsize=8)
            ax.set_title(
                f"{_short(ch)}  {axis_name}-sweep   FWHM = {fwhm} px",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            if i == n - 1:
                ax.set_xlabel(f"{axis_name} pixel")
            if col == 0:
                ax.set_ylabel("V above dark")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path is not None:
        fig.savefig(str(save_path), dpi=120)
    if show:
        plt.show()
    return fig


def plot_crosstalk(
    result: CrosstalkResult,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """Heatmap of the normalized crosstalk matrix.

    Row i = "PD i illuminated"; column j = "channel j's response as
    fraction of its own dynamic range". Diagonal entries should be
    ~1.0; off-diagonal should be near 0 for an acceptable setup. Cells
    above `warn_threshold` are outlined.
    """
    labels = [c.split("/")[-1] for c in result.channels]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(result.matrix, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("channel read (normalized to own dynamic range)")
    ax.set_ylabel("PD circle illuminated")
    ax.set_title(
        f"Crosstalk matrix  (threshold = {result.warn_threshold:.1%}, "
        f"{'OK' if result.acceptable else 'FAIL'})"
    )

    # Annotate each cell with its value and outline cells above threshold.
    n = len(labels)
    for i in range(n):
        for j in range(n):
            v = result.matrix[i, j]
            color = "white" if v < 0.5 else "black"
            ax.text(
                j, i, f"{v:.2f}" if np.isfinite(v) else "nan",
                ha="center", va="center", color=color, fontsize=9,
            )
            if i != j and np.isfinite(v) and abs(v) >= result.warn_threshold:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="tab:red", linewidth=2,
                ))

    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of channel's full bright")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=120)
    if show:
        plt.show()
    return fig


def plot_rise_time(
    result: RiseTimeResult,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """One row per channel, two columns: rise trace and fall trace.

    Each trace is overlaid with:
      - 10% and 90% horizontal reference lines (detector thresholds)
      - vertical line at the sustained 10% crossing (latency start)
      - vertical line at the sustained 90% crossing (latency + duration)
      - vertical line at the pre-flip time (when flip() was called)
    """
    n = len(result.channels)
    rate = float(result.sample_rate)
    pre_flip_ms = result.pre_flip_s * 1000.0
    t_ms = np.arange(result.rise_trace.shape[1]) / rate * 1000.0

    fig, axes = plt.subplots(
        n, 2, figsize=(11, 2.0 + 1.6 * n), sharex="col", squeeze=False,
    )
    fig.suptitle(
        f"Temporal response ({rate / 1000:.0f} kHz, "
        f"sustain = {result.sustain_s * 1000:.1f} ms)",
        fontsize=12,
    )

    for i, ch in enumerate(result.channels):
        for col, (kind, trace, duration, latency) in enumerate([
            ("rise", result.rise_trace[i], result.rise_duration_s[i], result.rise_latency_s[i]),
            ("fall", result.fall_trace[i], result.fall_duration_s[i], result.fall_latency_s[i]),
        ]):
            ax = axes[i, col]
            ax.plot(t_ms, trace, "-", linewidth=0.8)

            # Detector reference levels: baseline (first 5%) + plateau (last 10%).
            n_samples = trace.size
            n_pre = max(20, n_samples // 20)
            n_post = max(20, n_samples // 10)
            baseline = float(trace[:n_pre].mean())
            plateau = float(trace[-n_post:].mean())
            y10 = baseline + 0.1 * (plateau - baseline)
            y90 = baseline + 0.9 * (plateau - baseline)
            ax.axhline(y10, color="0.7", linestyle=":", linewidth=0.8)
            ax.axhline(y90, color="0.7", linestyle=":", linewidth=0.8)

            # pre_flip marker: when pygame.flip() was actually issued.
            ax.axvline(
                pre_flip_ms, color="tab:gray", linestyle="--", linewidth=0.8,
                alpha=0.7,
            )

            # Detected 10% and 90% crossings.
            if np.isfinite(latency):
                ax.axvline(
                    latency * 1000, color="tab:red", linestyle="-", linewidth=1,
                )
                if np.isfinite(duration):
                    ax.axvline(
                        (latency + duration) * 1000, color="tab:red",
                        linestyle="-", linewidth=1,
                    )

            dur_str = f"{duration*1000:.2f} ms" if np.isfinite(duration) else "n/a"
            lat_str = f"{latency*1000:.1f} ms" if np.isfinite(latency) else "n/a"
            ax.set_title(
                f"{_short(ch)}  {kind}:  dur={dur_str}, lat={lat_str}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
            if i == n - 1:
                ax.set_xlabel("time (ms)")
            if col == 0:
                ax.set_ylabel("V")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path is not None:
        fig.savefig(str(save_path), dpi=120)
    if show:
        plt.show()
    return fig
