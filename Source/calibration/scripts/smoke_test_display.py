"""Manual smoke test for calibration.display.

Walks through every drawing primitive, fullscreen. Press any key to advance
to the next frame, ESC (or close the window) to quit early.

Usage:
    venv\\Scripts\\python Source\\calibration\\scripts\\smoke_test_display.py [display_index]

If display_index is omitted, display 0 is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the calibration package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from calibration.display import Display, list_displays


def main() -> int:
    displays = list_displays()
    print("Attached displays:")
    for d in displays:
        print(f"  [{d.index}] {d.width} x {d.height}")

    display_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(
        f"\nUsing display {display_index}. "
        "Press ANY key to advance, ESC to quit.\n"
    )

    with Display(display_index) as d:
        W, H = d.width, d.height

        steps: list[tuple[str, callable]] = [
            ("Black fill", lambda: d.black()),
            ("White fill", lambda: d.white()),
            (f"Vertical bar  @ x=W/4={W // 4}, w=4", lambda: d.vertical_bar(W // 4, 4)),
            (f"Vertical bar  @ x=3W/4={3 * W // 4}, w=4", lambda: d.vertical_bar(3 * W // 4, 4)),
            (f"Horizontal bar @ y=H/4={H // 4}, h=4", lambda: d.horizontal_bar(H // 4, 4)),
            (f"Horizontal bar @ y=3H/4={3 * H // 4}, h=4", lambda: d.horizontal_bar(3 * H // 4, 4)),
            ("Circle at center, r=50", lambda: d.circle(W // 2, H // 2, 50)),
            ("Circle at center, r=200", lambda: d.circle(W // 2, H // 2, 200)),
            # Structured-light patterns: highest bit first (coarsest stripes)
            # down to a fairly fine one. For a 1920-px-wide display the coarsest
            # useful bit is ceil(log2(W)) - 1 = 10.
            ("Gray stripes X, bit=10 (~half-screen)", lambda: d.gray_stripes("x", 10)),
            ("Gray stripes X, bit=9",                lambda: d.gray_stripes("x", 9)),
            ("Gray stripes X, bit=8",                lambda: d.gray_stripes("x", 8)),
            ("Gray stripes X, bit=6",                lambda: d.gray_stripes("x", 6)),
            ("Gray stripes X, bit=4 (16-px stripes)", lambda: d.gray_stripes("x", 4)),
            ("Gray stripes Y, bit=9",                lambda: d.gray_stripes("y", 9)),
            ("Gray stripes Y, bit=5",                lambda: d.gray_stripes("y", 5)),
        ]

        for label, draw in steps:
            print(f"  -> {label}")
            draw()
            d.flip()
            if d.wait_for_key():
                print("\nQuit requested.")
                return 0

        print("\nAll primitives displayed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
