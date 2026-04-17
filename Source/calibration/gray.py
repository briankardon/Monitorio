"""Reflected binary Gray code -- pure numpy.

Mirror of the MATLAB-utils grayEncode/grayDecode pair. Used on the Python
side to decode photodiode responses collected from structured-light
localization (calibration.procedure.localize_coarse) and will be used by
the future decoder that reads frame numbers out of a DAQ recording.

Gray code property we rely on: for any p, grayEncode(p) and grayEncode(p+1)
differ in exactly one bit. Structured-light localization exploits the
corollary that the top K bits of grayEncode(p) identify which 2**(N-K)
contiguous block of positions p lies in -- i.e. if we observe those top
K bits we get block_index = grayDecode(top_bits).
"""

from __future__ import annotations

import numpy as np


def encode(n):
    """Convert nonneg integer(s) to reflected binary Gray code. Shape-preserving."""
    arr = np.asarray(n)
    if np.any(arr < 0):
        raise ValueError("encode requires nonnegative values")
    return arr ^ (arr >> 1)


def decode(g, n_bits: int = 32):
    """Convert Gray code back to plain integer(s).

    n_bits bounds the bit width of the inputs; the decode does
    ceil(log2(n_bits)) XOR passes. Default 32 covers anything we'll see.
    """
    arr = np.asarray(g, dtype=np.int64)
    if np.any(arr < 0):
        raise ValueError("decode requires nonnegative values")
    n = arr.copy()
    shift = 1
    while shift < n_bits:
        n = n ^ (n >> shift)
        shift *= 2
    return n
