# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""8-bit scaler."""

from ctypes import byref
from dataclasses import dataclass
from typing import Optional

from . import _native
from .errors import FunnelcakeError, Warnings
from .frame import Frame


def _trailing_zeros(flag: int) -> int:
    return (flag & -flag).bit_length() - 1


@dataclass
class ScalerConfig:
    """Configuration for a :class:`Scaler`. Source strides are derived from
    ``src_width``, so a :class:`Frame` of the same size matches automatically.
    """

    src_width: int
    src_height: int
    flags: int = 0          # Scale.* (one family)
    upscale_flags: int = 0  # Upscale.* (contiguous prefix)
    upscale_tail_1_5x: bool = False
    options: int = 0        # Option.*


@dataclass
class Output:
    """A read-only view of one 8-bit output plane set. The ``y``/``u``/``v``
    memoryviews alias scaler-owned memory and are valid only until the
    producing scaler's next ``run`` or ``close``.
    """

    width: int
    height: int
    y_stride: int
    uv_stride: int
    fallback: bool
    y: Optional[memoryview]
    u: Optional[memoryview]
    v: Optional[memoryview]
    # Pins the producing scaler so that, while this Output is referenced, the
    # scaler is not garbage-collected (which would free these buffers). Copy
    # data out if you need it past the scaler's lifetime.
    _owner: object = None


def _output_from(o, owner=None) -> Output:
    h = o.height
    chroma_h = (h + 1) // 2
    return Output(
        width=o.width,
        height=h,
        y_stride=o.y_stride,
        uv_stride=o.uv_stride,
        fallback=bool(o.fallback),
        y=_native.view_u8(_native.ptr_addr(o.plane_y), o.y_stride * h),
        u=_native.view_u8(_native.ptr_addr(o.plane_u), o.uv_stride * chroma_h),
        v=_native.view_u8(_native.ptr_addr(o.plane_v), o.uv_stride * chroma_h),
        _owner=owner,
    )


class Scaler:
    """An initialized 8-bit scaling context. Owns its output buffers; the
    :class:`Output` views are valid until the next ``run`` or ``close``.
    """

    def __init__(self, config: ScalerConfig):
        _native.validate_dimensions(config.src_width, config.src_height)
        ctx = _native.ScalerCtx()
        ys, uvs = _native.plane_strides(config.src_width)
        ctx.src_width = config.src_width
        ctx.src_height = config.src_height
        ctx.src_y_stride = ys
        ctx.src_uv_stride = uvs
        ctx.requested_flags = int(config.flags)
        ctx.options = int(config.options)
        ctx.upscale_flags = int(config.upscale_flags)
        ctx.upscale_tail_1_5x = 1 if config.upscale_tail_1_5x else 0

        rc = _native._core.fused_scaler_init(byref(ctx))
        if rc < 0:
            raise FunnelcakeError(rc)
        self._ctx = ctx
        self.warnings = Warnings(rc)
        self._src_width = config.src_width
        self._src_height = config.src_height
        self._closed = False

    def run(self, frame: Frame) -> None:
        """Scale one frame, filling all achieved outputs.

        The frame must have the same dimensions as the scaler's config; a
        mismatch raises ValueError rather than reading out of bounds.
        """
        self._check_open()
        if not frame._y:
            raise RuntimeError("frame is closed")
        if frame.width != self._src_width or frame.height != self._src_height:
            raise ValueError(
                f"frame {frame.width}x{frame.height} does not match scaler source "
                f"{self._src_width}x{self._src_height}"
            )
        _native._core.fused_scaler_run(byref(self._ctx), frame._y, frame._u, frame._v)

    @property
    def effective_width(self) -> int:
        return self._ctx.effective_width

    @property
    def effective_height(self) -> int:
        return self._ctx.effective_height

    @property
    def achieved_flags(self) -> int:
        return self._ctx.achieved_flags

    def output(self, flag: int) -> Optional[Output]:
        """The downscale output for a single ``Scale.*`` flag, or None."""
        if not (self._ctx.achieved_flags & int(flag)):
            return None
        return _output_from(self._ctx.outputs[_trailing_zeros(int(flag))], self)

    def upscale_output(self, flag: int) -> Optional[Output]:
        """An upscale-cascade output for a single ``Upscale.*`` flag, or None."""
        if not (self._ctx.achieved_upscale_flags & int(flag)):
            return None
        return _output_from(self._ctx.upscale_outputs[_trailing_zeros(int(flag))], self)

    def upscale_tail(self) -> Optional[Output]:
        """The 1.5x upscale tail output, if produced."""
        if not self._ctx.achieved_upscale_tail:
            return None
        return _output_from(self._ctx.upscale_outputs[5], self)  # FUSED_UP_IDX_TAIL

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            _native._core.fused_scaler_free(byref(self._ctx))
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("scaler is closed")

    def __enter__(self) -> "Scaler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()
