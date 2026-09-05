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
        _native.validate_dimensions(width, height)
        self._storage = _native.Storage()
        self._width = width
        self._height = height
        self._y_stride, self._uv_stride = _native.plane_strides(width)
        self._chroma_h = (height + 1) // 2
        self._y = self._storage.alloc(self._y_stride * height)
        self._u = self._storage.alloc(self._uv_stride * self._chroma_h)
        self._v = self._storage.alloc(self._uv_stride * self._chroma_h)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def y_stride(self):
        return self._y_stride

    @property
    def uv_stride(self):
        return self._uv_stride

    def y(self) -> memoryview:
        """Writable luma plane (`y_stride * height` bytes)."""
        return _native.view_u8(self._y, self._y_stride * self._height, self._storage)

    def u(self) -> memoryview:
        """Writable Cb plane."""
        return _native.view_u8(self._u, self._uv_stride * self._chroma_h, self._storage)

    def v(self) -> memoryview:
        """Writable Cr plane."""
        return _native.view_u8(self._v, self._uv_stride * self._chroma_h, self._storage)

    def close(self) -> None:
        self._y = self._u = self._v = 0
        self._storage = None

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
        _native.validate_dimensions(width, height)
        self._storage = _native.Storage()
        self._width = width
        self._height = height
        self._format = PixelFormat(format)
        semi_planar = self._format in (PixelFormat.P010, PixelFormat.P210)
        is_422 = self._format in (PixelFormat.I210, PixelFormat.P210)

        self._y_stride, uv = _native.plane_strides_16(width)
        self._chroma_h = height if is_422 else (height + 1) // 2
        self._semi_planar = semi_planar

        self._y = self._storage.alloc(self._y_stride * height)
        if semi_planar:
            self._uv_stride = self._y_stride  # interleaved UV row == luma byte width
            self._u = self._storage.alloc(self._uv_stride * self._chroma_h)
            self._v = 0
        else:
            self._uv_stride = uv
            self._u = self._storage.alloc(self._uv_stride * self._chroma_h)
            self._v = self._storage.alloc(self._uv_stride * self._chroma_h)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def y_stride(self):
        return self._y_stride

    @property
    def uv_stride(self):
        return self._uv_stride

    @property
    def format(self):
        return self._format

    def y(self) -> memoryview:
        """Writable luma plane (uint16 samples)."""
        return _native.view_u16(self._y, (self._y_stride // 2) * self._height, self._storage)

    def u(self) -> memoryview:
        """Cb plane (I010/I210) or interleaved UV plane (P010/P210)."""
        return _native.view_u16(self._u, (self._uv_stride // 2) * self._chroma_h, self._storage)

    def v(self):
        """Cr plane for planar formats; ``None`` for P010/P210."""
        if not self._v:
            return None
        return _native.view_u16(self._v, (self._uv_stride // 2) * self._chroma_h, self._storage)

    def close(self) -> None:
        self._y = self._u = self._v = 0
        self._storage = None

    def __enter__(self) -> "HdrFrame":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        self.close()
