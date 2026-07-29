# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""Idiomatic Python bindings for the funnelcake SIMD YUV scaler and HDR
tone-mapper (ctypes; no third-party dependencies).

Thin and safe: :class:`Frame`/:class:`HdrFrame` allocate input planes with the
32-byte alignment the SIMD kernels require, the scaler classes wrap the
create/run/free lifecycle (context-manager friendly), and outputs are exposed
as zero-copy ``memoryview``s over library-owned buffers.
"""

from ._native import simd_available
from .enums import Option, PixelFormat, Range, Scale, TonemapCurve, Transfer, Upscale
from .errors import FunnelcakeError, Warnings
from .frame import Frame, HdrFrame
from .hdr import HdrConfig, HdrOutput, HdrScaler, TonemapConfig
from .scaler import Output, Scaler, ScalerConfig

__all__ = [
    "simd_available",
    "Scale",
    "Upscale",
    "Option",
    "PixelFormat",
    "Transfer",
    "Range",
    "TonemapCurve",
    "FunnelcakeError",
    "Warnings",
    "Frame",
    "HdrFrame",
    "Scaler",
    "ScalerConfig",
    "Output",
    "HdrScaler",
    "HdrConfig",
    "TonemapConfig",
    "HdrOutput",
]
