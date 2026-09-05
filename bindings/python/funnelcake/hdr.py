# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""10-bit scaler with tone mapping."""

from ctypes import POINTER, byref, c_uint8, cast
from dataclasses import dataclass, field
from typing import Optional

from . import _native
from .enums import PixelFormat, Range, Transfer, TonemapCurve
from .errors import FunnelcakeError, Warnings
from .frame import HdrFrame
from .scaler import Output, _output_from, _trailing_zeros


@dataclass
class TonemapConfig:
    """Tone-mapping configuration. Defaults: Hable curve, library-default nits
    (1000 peak / 100 target), limited range in and out.
    """

    curve: int = TonemapCurve.HABLE
    peak_nits: int = 0
    target_nits: int = 0
    src_range: int = Range.LIMITED
    dst_range: int = Range.LIMITED
    custom_lut: Optional[bytes] = None  # 1024 entries, only for TonemapCurve.CUSTOM


@dataclass
class HdrConfig:
    """Configuration for an :class:`HdrScaler`. ``hdr_flags`` and ``sdr_flags``
    must each be a subset of ``flags``.
    """

    src_width: int
    src_height: int
    format: int = PixelFormat.I010
    transfer: int = Transfer.PQ
    flags: int = 0
    hdr_flags: int = 0
    sdr_flags: int = 0
    options: int = 0
    tonemap_1x: bool = False
    tonemap: TonemapConfig = field(default_factory=TonemapConfig)
    upscale_flags: int = 0
    upscale_tail_1_5x: bool = False
    upscale_sdr_flags: int = 0
    upscale_sdr_tail_1_5x: bool = False


@dataclass
class HdrOutput:
    """A read-only view of one 10-bit output plane set. Samples are uint16
    (10 significant bits). Strides are in bytes. The memoryviews alias
    scaler-owned memory, valid until the next ``run`` or ``close``.
    """

    width: int
    height: int
    y_stride: int
    uv_stride: int
    fallback: bool
    y: Optional[memoryview]
    u: Optional[memoryview]
    v: Optional[memoryview]
    _owner: object = None  # pins the producing scaler; see scaler.Output._owner


    def _row(self, plane, row):
        height = self.height if plane == "y" else (self.height + 1) // 2
        width = self.width if plane == "y" else self.width // 2
        stride = (self.y_stride if plane == "y" else self.uv_stride) // 2
        if not 0 <= row < height:
            raise IndexError("row out of range")
        return getattr(self, plane)[row * stride:row * stride + width]

    def y_row(self, row):
        return self._row("y", row)

    def u_row(self, row):
        return self._row("u", row)

    def v_row(self, row):
        return self._row("v", row)

    def copy(self):
        """Own tightly packed planes that survive future runs and close."""
        planes = []
        for plane in ("y", "u", "v"):
            rows = self.height if plane == "y" else (self.height + 1) // 2
            data = b"".join(self._row(plane, r).tobytes() for r in range(rows))
            planes.append(memoryview(data).cast("H"))
        return HdrOutput(self.width, self.height, self.width * 2,
                       self.width // 2 * 2, self.fallback, *planes)


def _hdr_output_from(o, owner=None) -> HdrOutput:
    h = o.height
    chroma_h = (h + 1) // 2
    return HdrOutput(
        width=o.width,
        height=h,
        y_stride=o.y_stride,
        uv_stride=o.uv_stride,
        fallback=bool(o.fallback),
        y=_native.view_u16(_native.ptr_addr(o.plane_y), (o.y_stride // 2) * h, owner._storage, True),
        u=_native.view_u16(_native.ptr_addr(o.plane_u), (o.uv_stride // 2) * chroma_h, owner._storage, True),
        v=_native.view_u16(_native.ptr_addr(o.plane_v), (o.uv_stride // 2) * chroma_h, owner._storage, True),
        _owner=owner,
    )


class HdrScaler:
    """An initialized 10-bit scaling / tone-mapping context. Owns its output
    buffers; views are valid until the next ``run`` or ``close``.
    """

    def __init__(self, config: HdrConfig):
        lut = config.tonemap.custom_lut
        if lut is not None and len(lut) != 1024:
            raise ValueError(
                f"custom tone-map LUT must be exactly 1024 bytes, got {len(lut)}"
            )
        _native.validate_dimensions(config.src_width, config.src_height)
        ctx = _native.HdrCtx()
        fmt = PixelFormat(config.format)
        semi_planar = fmt in (PixelFormat.P010, PixelFormat.P210)
        ys, uv = _native.plane_strides_16(config.src_width)

        ctx.src_width = config.src_width
        ctx.src_height = config.src_height
        ctx.src_y_stride = ys
        ctx.src_uv_stride = ys if semi_planar else uv
        ctx.src_format = int(config.format)
        ctx.src_transfer = int(config.transfer)
        ctx.requested_flags = int(config.flags)
        ctx.hdr_flags = int(config.hdr_flags)
        ctx.sdr_flags = int(config.sdr_flags)
        ctx.options = int(config.options)
        ctx.tonemap_1x = 1 if config.tonemap_1x else 0

        tm = config.tonemap
        ctx.tonemap.curve = int(tm.curve)
        ctx.tonemap.peak_nits = tm.peak_nits
        ctx.tonemap.target_nits = tm.target_nits
        ctx.tonemap.src_range = int(tm.src_range)
        ctx.tonemap.dst_range = int(tm.dst_range)
        # Keep the LUT buffer alive across init (the library reads it there).
        self._lut_buf = None
        if tm.custom_lut:
            self._lut_buf = (c_uint8 * len(tm.custom_lut)).from_buffer_copy(tm.custom_lut)
            ctx.tonemap.custom_lut = cast(self._lut_buf, POINTER(c_uint8))

        ctx.upscale_flags = int(config.upscale_flags)
        ctx.upscale_tail_1_5x = 1 if config.upscale_tail_1_5x else 0
        ctx.upscale_sdr_flags = int(config.upscale_sdr_flags)
        ctx.upscale_sdr_tail_1_5x = 1 if config.upscale_sdr_tail_1_5x else 0

        rc = _native._core.fused_hdr_init(byref(ctx))
        if rc < 0:
            raise FunnelcakeError(rc)
        self._ctx = ctx
        self._storage = _native.Storage(ctx, hdr=True)
        self.warnings = Warnings(rc)
        self._src_width = config.src_width
        self._src_height = config.src_height
        self._format = int(config.format)
        self._closed = False

    def run(self, frame: HdrFrame) -> None:
        """Scale and tone-map one 10-bit frame.

        The frame must have the same dimensions and format as the scaler's
        config; a mismatch raises ValueError rather than reading out of bounds.
        """
        self._check_open()
        if not frame._y:
            raise RuntimeError("frame is closed")
        if (frame.width != self._src_width or frame.height != self._src_height
                or int(frame.format) != self._format):
            raise ValueError(
                f"frame {frame.width}x{frame.height} fmt={int(frame.format)} does not match "
                f"scaler {self._src_width}x{self._src_height} fmt={self._format}"
            )
        _native._core.fused_hdr_run(byref(self._ctx), frame._y, frame._u, frame._v)

    @property
    def effective_width(self) -> int:
        return self._ctx.effective_width

    @property
    def effective_height(self) -> int:
        return self._ctx.effective_height

    def hdr_output(self, flag: int) -> Optional[HdrOutput]:
        """The 10-bit HDR output for a downscale flag, or None."""
        self._check_open()
        if not _native.single_flag(flag) or not (self._ctx.achieved_hdr_flags & int(flag)):
            return None
        return _hdr_output_from(self._ctx.hdr_outputs[_trailing_zeros(int(flag))], self)

    def sdr_output(self, flag: int) -> Optional[Output]:
        """The tone-mapped 8-bit output for a downscale flag, or None."""
        self._check_open()
        if not _native.single_flag(flag) or not (self._ctx.achieved_sdr_flags & int(flag)):
            return None
        return _output_from(self._ctx.sdr_outputs[_trailing_zeros(int(flag))], self)

    def tonemap_1x_output(self) -> Optional[Output]:
        """The 1:1 tone-mapped SDR copy, if produced."""
        self._check_open()
        if not _native.ptr_addr(self._ctx.output_1x.plane_y):
            return None
        return _output_from(self._ctx.output_1x, self)

    def upscale_hdr_output(self, flag: int) -> Optional[HdrOutput]:
        self._check_open()
        if not _native.single_flag(flag) or not (self._ctx.achieved_upscale_flags & int(flag)):
            return None
        return _hdr_output_from(self._ctx.upscale_hdr_outputs[_trailing_zeros(int(flag))], self)

    def upscale_sdr_output(self, flag: int) -> Optional[Output]:
        self._check_open()
        if not _native.single_flag(flag) or not (self._ctx.achieved_upscale_sdr_flags & int(flag)):
            return None
        return _output_from(self._ctx.upscale_sdr_outputs[_trailing_zeros(int(flag))], self)

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self._storage = None
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("scaler is closed")

    def __enter__(self) -> "HdrScaler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()
