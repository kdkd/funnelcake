# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""Aligned input frames."""

from . import _native
from .enums import PixelFormat


class Frame:
    """An 8-bit I420 (YUV 4:2:0 planar) input frame.

    The three planes are allocated 32-byte aligned with 32-byte-aligned strides
    — exactly what the SIMD kernels need — so a Frame is always safe to pass to
    ``Scaler.run``. Fill the planes through the ``y``/``u``/``v`` memoryviews.
    Use as a context manager, or call ``close()``.
    """

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        self.width = width
        self.height = height
        self.y_stride, self.uv_stride = _native.plane_strides(width)
        self._chroma_h = (height + 1) // 2
        self._y = _native.aligned_alloc(self.y_stride * height)
        self._u = _native.aligned_alloc(self.uv_stride * self._chroma_h)
        self._v = _native.aligned_alloc(self.uv_stride * self._chroma_h)

    def y(self) -> memoryview:
        """Writable luma plane (`y_stride * height` bytes)."""
        return _native.view_u8(self._y, self.y_stride * self.height)

    def u(self) -> memoryview:
        """Writable Cb plane."""
        return _native.view_u8(self._u, self.uv_stride * self._chroma_h)

    def v(self) -> memoryview:
        """Writable Cr plane."""
        return _native.view_u8(self._v, self.uv_stride * self._chroma_h)

    def close(self) -> None:
        for attr in ("_y", "_u", "_v"):
            addr = getattr(self, attr, 0)
            if addr:
                _native.aligned_free(addr)
                setattr(self, attr, 0)

    def __enter__(self) -> "Frame":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()


class HdrFrame:
    """A 10-bit input frame (samples in the low 10 bits of each uint16).

    Layout follows ``format``: planar formats (I010/I210) use three planes;
    semi-planar (P010/P210) use a luma plane plus one interleaved-UV plane and
    ``v()`` returns ``None``. All planes are 32-byte aligned.
    """

    def __init__(self, width: int, height: int, format: PixelFormat):
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        self.width = width
        self.height = height
        self.format = PixelFormat(format)
        semi_planar = self.format in (PixelFormat.P010, PixelFormat.P210)
        is_422 = self.format in (PixelFormat.I210, PixelFormat.P210)

        self.y_stride, uv = _native.plane_strides_16(width)
        self._chroma_h = height if is_422 else (height + 1) // 2
        self._semi_planar = semi_planar

        self._y = _native.aligned_alloc(self.y_stride * height)
        if semi_planar:
            self.uv_stride = self.y_stride  # interleaved UV row == luma byte width
            self._u = _native.aligned_alloc(self.uv_stride * self._chroma_h)
            self._v = 0
        else:
            self.uv_stride = uv
            self._u = _native.aligned_alloc(self.uv_stride * self._chroma_h)
            self._v = _native.aligned_alloc(self.uv_stride * self._chroma_h)

    def y(self) -> memoryview:
        """Writable luma plane (uint16 samples)."""
        return _native.view_u16(self._y, (self.y_stride // 2) * self.height)

    def u(self) -> memoryview:
        """Cb plane (I010/I210) or interleaved UV plane (P010/P210)."""
        return _native.view_u16(self._u, (self.uv_stride // 2) * self._chroma_h)

    def v(self):
        """Cr plane for planar formats; ``None`` for P010/P210."""
        if not self._v:
            return None
        return _native.view_u16(self._v, (self.uv_stride // 2) * self._chroma_h)

    def close(self) -> None:
        for attr in ("_y", "_u", "_v"):
            addr = getattr(self, attr, 0)
            if addr:
                _native.aligned_free(addr)
                setattr(self, attr, 0)

    def __enter__(self) -> "HdrFrame":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()
