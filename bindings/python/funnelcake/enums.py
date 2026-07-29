# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""Flag and enum constants mirroring the FUSED_* macros in funnelcake.h."""

from enum import IntEnum, IntFlag


class Scale(IntFlag):
    """Downscale step flags. A single scaler must use flags from ONE family."""

    X1_5 = 1 << 0   # thirds family
    X2 = 1 << 1     # pow2 family
    X3 = 1 << 2     # thirds
    X4 = 1 << 3     # pow2
    X6 = 1 << 4     # thirds
    X8 = 1 << 5     # pow2
    X12 = 1 << 6    # thirds
    X16 = 1 << 7    # pow2

    THIRDS_MASK = X1_5 | X3 | X6 | X12
    POW2_MASK = X2 | X4 | X8 | X16


class Upscale(IntFlag):
    """Upscale cascade levels; request a contiguous prefix only."""

    X2 = 1 << 0
    X4 = 1 << 1
    X8 = 1 << 2
    X16 = 1 << 3
    X32 = 1 << 4


class Option(IntFlag):
    NONE = 0
    NO_CROP = 1 << 0       # reject steps needing a crop (default: crop + warn)
    NO_FALLBACK = 1 << 1   # reject steps that can't use SIMD (default: scalar + warn)


class PixelFormat(IntEnum):
    I010 = 0  # 4:2:0 planar
    P010 = 1  # 4:2:0 semi-planar (interleaved UV)
    I210 = 2  # 4:2:2 planar (decimated to 4:2:0)
    P210 = 3  # 4:2:2 semi-planar (decimated to 4:2:0)


class Transfer(IntEnum):
    PQ = 0   # SMPTE ST 2084 (HDR10)
    HLG = 1  # Hybrid Log-Gamma


class Range(IntEnum):
    LIMITED = 0  # video range
    FULL = 1     # full / PC range


class TonemapCurve(IntEnum):
    HABLE = 0     # filmic (default)
    REINHARD = 1
    BT2390 = 2
    CUSTOM = 3    # use TonemapConfig.custom_lut
