# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""ctypes glue: shared-library loading, struct definitions mirroring
``funnelcake.h``, function prototypes, and small view/alloc helpers.

``ctypes.Structure`` lays fields out per the platform C ABI, so the structs
match the header automatically (no manual padding). The total sizes are
cross-checked at import against the real C ``sizeof`` (the helper functions now
live in the core libfunnelcake), so any drift fails loudly here.
"""

import os
import sys
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_int,
    c_char_p,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_void_p,
    cast,
    sizeof,
)


# ---- shared library discovery ----

def _lib_ext() -> str:
    if sys.platform == "darwin":
        return "dylib"
    if os.name == "nt":
        return "dll"
    return "so"


def _lib_path(name: str) -> str:
    # Default: the directory containing this package (bindings/python), which is
    # where `make bindings-python` places the libraries. Override with the
    # FUNNELCAKE_LIBDIR environment variable.
    default_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libdir = os.environ.get("FUNNELCAKE_LIBDIR") or default_dir
    return os.path.join(libdir, f"lib{name}.{_lib_ext()}")


_core = CDLL(_lib_path("funnelcake"))


# ---- struct layouts (mirror include/funnelcake.h) ----

class LogConfig(Structure):
    _fields_ = [
        ("target", c_int),
        ("file", c_void_p),
        ("callback", c_void_p),
        ("callback_ctx", c_void_p),
    ]


class ScaleOutput(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
        ("y_stride", c_int),
        ("uv_stride", c_int),
        ("plane_y", POINTER(c_uint8)),
        ("plane_u", POINTER(c_uint8)),
        ("plane_v", POINTER(c_uint8)),
        ("fallback", c_int),
    ]


class HdrOutput(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int),
        ("y_stride", c_int),
        ("uv_stride", c_int),
        ("plane_y", POINTER(c_uint16)),
        ("plane_u", POINTER(c_uint16)),
        ("plane_v", POINTER(c_uint16)),
        ("fallback", c_int),
    ]


class TonemapConfigC(Structure):
    _fields_ = [
        ("curve", c_int),
        ("peak_nits", c_int),
        ("target_nits", c_int),
        ("custom_lut", POINTER(c_uint8)),
        ("src_range", c_int),
        ("dst_range", c_int),
    ]


class ScalerCtx(Structure):
    _fields_ = [
        ("src_width", c_int),
        ("src_height", c_int),
        ("src_y_stride", c_int),
        ("src_uv_stride", c_int),
        ("requested_flags", c_uint32),
        ("options", c_uint32),
        ("log_errors", LogConfig),
        ("log_warnings", LogConfig),
        ("upscale_flags", c_uint32),
        ("upscale_tail_1_5x", c_int),
        ("achieved_flags", c_uint32),
        ("rejected_flags", c_uint32),
        ("effective_width", c_int),
        ("effective_height", c_int),
        ("outputs", ScaleOutput * 8),
        ("achieved_upscale_flags", c_uint32),
        ("achieved_upscale_tail", c_int),
        ("upscale_outputs", ScaleOutput * 6),
        ("_internal", c_void_p),
    ]


class HdrCtx(Structure):
    _fields_ = [
        ("src_width", c_int),
        ("src_height", c_int),
        ("src_y_stride", c_int),
        ("src_uv_stride", c_int),
        ("src_format", c_int),
        ("src_transfer", c_int),
        ("requested_flags", c_uint32),
        ("hdr_flags", c_uint32),
        ("sdr_flags", c_uint32),
        ("options", c_uint32),
        ("tonemap_1x", c_int),
        ("tonemap", TonemapConfigC),
        ("log_errors", LogConfig),
        ("log_warnings", LogConfig),
        ("achieved_hdr_flags", c_uint32),
        ("achieved_sdr_flags", c_uint32),
        ("rejected_flags", c_uint32),
        ("effective_width", c_int),
        ("effective_height", c_int),
        ("hdr_outputs", HdrOutput * 8),
        ("sdr_outputs", ScaleOutput * 8),
        ("output_1x", ScaleOutput),
        ("upscale_flags", c_uint32),
        ("upscale_tail_1_5x", c_int),
        ("upscale_sdr_flags", c_uint32),
        ("upscale_sdr_tail_1_5x", c_int),
        ("achieved_upscale_flags", c_uint32),
        ("achieved_upscale_tail", c_int),
        ("achieved_upscale_sdr_flags", c_uint32),
        ("achieved_upscale_sdr_tail", c_int),
        ("upscale_hdr_outputs", HdrOutput * 6),
        ("upscale_sdr_outputs", ScaleOutput * 6),
        ("_internal", c_void_p),
    ]


# ---- function prototypes ----

_core.fused_scaler_init.argtypes = [POINTER(ScalerCtx)]
_core.fused_scaler_init.restype = c_int
_core.fused_scaler_run.argtypes = [POINTER(ScalerCtx), c_void_p, c_void_p, c_void_p]
_core.fused_scaler_run.restype = None
_core.fused_scaler_free.argtypes = [POINTER(ScalerCtx)]
_core.fused_scaler_free.restype = None

_core.fused_hdr_init.argtypes = [POINTER(HdrCtx)]
_core.fused_hdr_init.restype = c_int
_core.fused_hdr_run.argtypes = [POINTER(HdrCtx), c_void_p, c_void_p, c_void_p]
_core.fused_hdr_run.restype = None
_core.fused_hdr_free.argtypes = [POINTER(HdrCtx)]
_core.fused_hdr_free.restype = None

_core.fused_simd_available.argtypes = []
_core.fused_simd_available.restype = c_int

# The binding helpers (aligned alloc, stride math, ctx-sizeof guards) live in
# the core libfunnelcake, so they are bound on _core as well.
_core.fused_aligned_alloc.argtypes = [c_size_t, c_size_t]
_core.fused_aligned_alloc.restype = c_void_p
_core.fused_free.argtypes = [c_void_p]
_core.fused_free.restype = None
_core.fused_plane_strides.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_core.fused_plane_strides.restype = None
_core.fused_plane_strides_16.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
_core.fused_plane_strides_16.restype = None
_core.fused_scaler_ctx_sizeof.argtypes = []
_core.fused_scaler_ctx_sizeof.restype = c_size_t
_core.fused_hdr_ctx_sizeof.argtypes = []
_core.fused_hdr_ctx_sizeof.restype = c_size_t


# ---- import-time guards ----

if sizeof(ScalerCtx) != _core.fused_scaler_ctx_sizeof():
    raise RuntimeError(
        f"ScalerCtx layout mismatch: Python {sizeof(ScalerCtx)} "
        f"vs C {_core.fused_scaler_ctx_sizeof()}"
    )
if sizeof(HdrCtx) != _core.fused_hdr_ctx_sizeof():
    raise RuntimeError(
        f"HdrCtx layout mismatch: Python {sizeof(HdrCtx)} "
        f"vs C {_core.fused_hdr_ctx_sizeof()}"
    )

# Win the library's one-time CPU-detection race before any user thread runs.
_core.fused_simd_available()


# ---- thin call wrappers + helpers ----

def simd_available() -> bool:
    return _core.fused_simd_available() == 1


def plane_strides(width: int):
    ys, uvs = c_int(), c_int()
    _core.fused_plane_strides(width, byref(ys), byref(uvs))
    return ys.value, uvs.value


def plane_strides_16(width: int):
    ys, uvs = c_int(), c_int()
    _core.fused_plane_strides_16(width, byref(ys), byref(uvs))
    return ys.value, uvs.value


def aligned_alloc(size: int) -> int:
    """Allocate `size` bytes 32-byte aligned; returns an integer address."""
    addr = _core.fused_aligned_alloc(32, size)
    if not addr:
        raise MemoryError("funnelcake: aligned allocation failed")
    return addr


def aligned_free(addr: int) -> None:
    if addr:
        _core.fused_free(c_void_p(addr))


def view_u8(addr: int, n: int, owner=None, readonly=False):
    """Zero-copy memoryview of `n` bytes at `addr` (format 'B'), or None."""
    if not addr or n <= 0:
        return None
    # .cast('B') strips the byte-order marker ctypes arrays carry, which
    # otherwise blocks slice assignment.
    exporter = (c_uint8 * n).from_address(addr)
    exporter._owner = owner
    view = memoryview(exporter).cast("B")
    return view.toreadonly() if readonly else view


def view_u16(addr: int, n: int, owner=None, readonly=False):
    """Zero-copy memoryview of `n` uint16 samples at `addr` (format 'H'), or None."""
    if not addr or n <= 0:
        return None
    exporter = (c_uint8 * (n * 2)).from_address(addr)
    exporter._owner = owner
    view = memoryview(exporter).cast("B").cast("H")
    return view.toreadonly() if readonly else view


def ptr_addr(ptr) -> int:
    """Integer address of a ctypes POINTER (0 if NULL)."""
    return cast(ptr, c_void_p).value or 0


class Storage:
    """Own native storage independently of frame/scaler wrapper lifetime."""
    def __init__(self, ctx=None, hdr=False):
        self.ctx, self.hdr, self.planes = ctx, hdr, []

    def alloc(self, size):
        addr = aligned_alloc(size)
        self.planes.append(addr)
        return addr

    def __del__(self):
        if self.ctx is not None:
            fn = _core.fused_hdr_free if self.hdr else _core.fused_scaler_free
            fn(byref(self.ctx))
        for addr in self.planes:
            aligned_free(addr)


def validate_dimensions(width, height):
    if (not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0 or width % 2 or height % 2
            or width > 33554431 or height > 33554431
            or ((width * 2 + 31) & ~31) * height > 2147483647):
        raise ValueError("frame dimensions must be positive, even and fit native indexing")


def single_flag(flag):
    return isinstance(flag, int) and 0 < flag <= 0xffffffff and flag & (flag - 1) == 0


_core.fused_version.argtypes = []
_core.fused_version.restype = c_char_p
_core.fused_backend.argtypes = []
_core.fused_backend.restype = c_char_p


def version():
    """Version of the loaded native library."""
    return _core.fused_version().decode("ascii")


def backend():
    """Preferred native backend; individual outputs may fall back to scalar."""
    return _core.fused_backend().decode("ascii")


def check_integer_fields(config, signed=(), unsigned=()):
    for name in signed + unsigned:
        value = getattr(config, name)
        low, high = (-2147483648, 2147483647) if name in signed else (0, 4294967295)
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} is outside the native integer range")
