"""Fullscreen drawing primitives for photodiode calibration.

Thin wrapper around pygame-ce. Exposes only the patterns the calibration
procedure needs: uniform fills, a thin bar (for 1D sweeps), a single filled
circle (for diameter sweeps), and Gray-coded stripe patterns (for
structured-light localization).

Resolution is queried at runtime from the OS; the caller only picks which
display to open on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame


@dataclass(frozen=True)
class DisplayInfo:
    index: int
    width: int
    height: int


def list_displays() -> list[DisplayInfo]:
    """Return (index, width, height) for every attached display.

    Opens and closes pygame's display subsystem transiently; does not
    create a window.
    """
    pygame.display.init()
    try:
        sizes = pygame.display.get_desktop_sizes()
        return [DisplayInfo(i, w, h) for i, (w, h) in enumerate(sizes)]
    finally:
        pygame.display.quit()


class Display:
    """Fullscreen pygame window on a chosen monitor, with calibration primitives.

    Typical use:

        with Display(display_index=1) as d:
            d.black(); d.flip()
            d.vertical_bar(x_start=500, width=4); d.flip()

    All draw methods leave the back buffer dirty; call `flip()` to present.
    Screen is filled black at the start of each shape-drawing method (bar,
    circle, gray_stripes), so adjacent frames don't bleed into each other.
    """

    def __init__(self, display_index: int = 0, fullscreen: bool = True):
        pygame.display.init()
        sizes = pygame.display.get_desktop_sizes()
        if display_index < 0 or display_index >= len(sizes):
            pygame.display.quit()
            raise ValueError(
                f"display_index={display_index} but {len(sizes)} display(s) attached"
            )
        self._display_index = display_index
        self._width, self._height = sizes[display_index]

        flags = pygame.FULLSCREEN if fullscreen else 0
        self._screen = pygame.display.set_mode(
            (self._width, self._height),
            flags=flags,
            display=display_index,
        )
        pygame.mouse.set_visible(False)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def display_index(self) -> int:
        return self._display_index

    def close(self):
        pygame.display.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # -------- draw primitives --------

    def fill(self, value: int):
        """Fill the entire screen with a uniform gray value in [0, 255]."""
        self._screen.fill((value, value, value))

    def black(self):
        self.fill(0)

    def white(self):
        self.fill(255)

    def vertical_bar(self, x_start: int, width: int = 1):
        """Black screen with a white vertical bar of `width` px starting at x_start.

        Used for 1D X-axis sweeps: the bar marches across the screen and each
        photodiode's response vs. bar position is its 1D sensitivity profile.
        """
        self.fill(0)
        pygame.draw.rect(
            self._screen, (255, 255, 255),
            (x_start, 0, width, self._height),
        )

    def horizontal_bar(self, y_start: int, height: int = 1):
        """Black screen with a white horizontal bar of `height` px starting at y_start."""
        self.fill(0)
        pygame.draw.rect(
            self._screen, (255, 255, 255),
            (0, y_start, self._width, height),
        )

    def circle(self, cx: int, cy: int, radius: int):
        """Black screen with a single filled white circle at (cx, cy) of given radius."""
        self.fill(0)
        pygame.draw.circle(self._screen, (255, 255, 255), (cx, cy), radius)

    def gray_stripes(self, axis: str, bit: int):
        """Render a Gray-coded stripe pattern along `axis` ('x' or 'y') for bit `bit`.

        Theory
        ------
        Structured-light localization with Gray code. For each pixel position p
        along the axis, compute g = grayEncode(p) = p XOR (p >> 1), and color
        the column (or row) white if bit `bit` of g is 1, else black.

        A photodiode sitting over position p will read bit `bit` of grayEncode(p)
        from this pattern. Displaying patterns for bits K down to k_min (where
        2**K >= screen dimension) gives each photodiode a (K - k_min + 1)-bit
        Gray-coded reading of its position, localizing it to within 2**k_min
        pixels. After that, a centroid refinement on a 1D bar sweep pinpoints
        the center to sub-pixel precision.

        Why Gray code (not plain binary): adjacent positions differ in exactly
        one bit of their Gray code, so a photodiode straddling a stripe boundary
        returns a reading that decodes to one of the two adjacent positions --
        never a wildly wrong position from multiple bits being mid-transition.

        Stripe widths
        -------------
        Bit `bit` produces stripes 2**bit pixels wide. For a 1920-px screen,
        bit 10 splits the screen roughly in half; bit 0 alternates every pixel.
        The finest practical bit is set by photodiode size -- once stripe width
        drops below the PD's sensitive area, the photodiode averages over both
        colors and the response collapses to ~0.5.
        """
        if axis == "x":
            positions = np.arange(self._width, dtype=np.int64)
        elif axis == "y":
            positions = np.arange(self._height, dtype=np.int64)
        else:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

        gray = positions ^ (positions >> 1)
        stripe = (((gray >> bit) & 1) * 255).astype(np.uint8)  # (N,)

        # Broadcast the 1D stripe pattern to a full (H, W, 3) RGB array.
        if axis == "x":
            # Each column takes one value from `stripe`.
            pattern = np.broadcast_to(
                stripe[None, :, None], (self._height, self._width, 3),
            )
        else:
            # Each row takes one value from `stripe`.
            pattern = np.broadcast_to(
                stripe[:, None, None], (self._height, self._width, 3),
            )

        # pygame's surfarray uses (width, height, 3) axis order; transpose.
        # .copy() because blit_array requires a C-contiguous buffer.
        pygame.surfarray.blit_array(
            self._screen, pattern.transpose(1, 0, 2).copy(),
        )

    # -------- presentation and input --------

    def flip(self):
        """Present the back buffer and pump events so the OS keeps the window live."""
        pygame.display.flip()
        pygame.event.pump()

    def wait_for_key(self) -> bool:
        """Block until a key press or window close. Returns True on ESC/quit."""
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                return event.key == pygame.K_ESCAPE
