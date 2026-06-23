// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};

use crate::{align32, PixelFormat};

/// 32-byte alignment for plane allocations.
const ALIGN: usize = 32;

/// Plane size in bytes. Computed in `usize` to avoid i32 overflow on large
/// frames (e.g. `y_stride * height` exceeds i32 well before 16K resolutions).
fn nbytes(stride: i32, rows: i32) -> usize {
    stride as usize * rows as usize
}

/// Number of u16 samples for a 16-bit plane of the given byte stride and rows.
fn nsamples(byte_stride: i32, rows: i32) -> usize {
    (byte_stride as usize / 2) * rows as usize
}

fn alloc_plane(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size, ALIGN).expect("valid layout");
    // SAFETY: layout has non-zero size (callers pass positive plane sizes).
    let p = unsafe { alloc_zeroed(layout) };
    if p.is_null() {
        handle_alloc_error(layout);
    }
    p
}

fn free_plane(p: *mut u8, size: usize) {
    if !p.is_null() {
        let layout = Layout::from_size_align(size, ALIGN).expect("valid layout");
        // SAFETY: p came from alloc_plane with the same size/alignment.
        unsafe { dealloc(p, layout) };
    }
}

/// An 8-bit I420 (YUV 4:2:0 planar) input frame. The three planes are
/// allocated with 32-byte alignment and 32-byte-aligned strides — exactly what
/// the SIMD kernels need — so a `Frame` is always safe to pass to
/// [`crate::Scaler::run`]. Fill the planes through the `*_mut` accessors.
pub struct Frame {
    width: i32,
    height: i32,
    y_stride: i32,
    uv_stride: i32,
    chroma_h: i32,
    y: *mut u8,
    u: *mut u8,
    v: *mut u8,
}

impl Frame {
    /// Allocates an aligned I420 frame. `width`/`height` should be even.
    ///
    /// # Panics
    /// If a dimension is not positive.
    pub fn new(width: i32, height: i32) -> Frame {
        assert!(width > 0 && height > 0, "frame dimensions must be positive");
        let y_stride = align32(width);
        let uv_stride = align32(width / 2);
        let chroma_h = (height + 1) / 2;
        Frame {
            width,
            height,
            y_stride,
            uv_stride,
            chroma_h,
            y: alloc_plane(nbytes(y_stride, height)),
            u: alloc_plane(nbytes(uv_stride, chroma_h)),
            v: alloc_plane(nbytes(uv_stride, chroma_h)),
        }
    }

    pub fn width(&self) -> i32 {
        self.width
    }
    pub fn height(&self) -> i32 {
        self.height
    }
    pub fn y_stride(&self) -> i32 {
        self.y_stride
    }
    pub fn uv_stride(&self) -> i32 {
        self.uv_stride
    }

    /// Writable luma plane (`y_stride * height` bytes).
    pub fn y_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.y, nbytes(self.y_stride, self.height)) }
    }
    /// Writable Cb plane.
    pub fn u_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.u, nbytes(self.uv_stride, self.chroma_h)) }
    }
    /// Writable Cr plane.
    pub fn v_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.v, nbytes(self.uv_stride, self.chroma_h)) }
    }

    // Raw pointers for the scaler run call (crate-internal).
    pub(crate) fn planes(&self) -> (*const u8, *const u8, *const u8) {
        (self.y, self.u, self.v)
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        free_plane(self.y, nbytes(self.y_stride, self.height));
        free_plane(self.u, nbytes(self.uv_stride, self.chroma_h));
        free_plane(self.v, nbytes(self.uv_stride, self.chroma_h));
    }
}

/// A 10-bit input frame (samples in the low 10 bits of each `u16`). Layout
/// follows `format`: planar formats (I010/I210) use three planes; semi-planar
/// (P010/P210) use a luma plane plus one interleaved-UV plane and `v_mut`
/// returns `None`. All planes are 32-byte aligned.
pub struct HdrFrame {
    width: i32,
    height: i32,
    format: PixelFormat,
    y_stride: i32,
    uv_stride: i32,
    chroma_h: i32,
    semi_planar: bool,
    y: *mut u8,
    u: *mut u8,
    v: *mut u8,
}

impl HdrFrame {
    /// Allocates an aligned 10-bit frame for the given format.
    ///
    /// # Panics
    /// If a dimension is not positive.
    pub fn new(width: i32, height: i32, format: PixelFormat) -> HdrFrame {
        assert!(width > 0 && height > 0, "frame dimensions must be positive");
        let semi_planar = matches!(format, PixelFormat::P010 | PixelFormat::P210);
        let is_422 = matches!(format, PixelFormat::I210 | PixelFormat::P210);
        let y_stride = align32(width * 2);
        let chroma_h = if is_422 { height } else { (height + 1) / 2 };

        let y = alloc_plane(nbytes(y_stride, height));
        let (uv_stride, u, v) = if semi_planar {
            let uv = y_stride; // interleaved UV row == luma byte width
            (uv, alloc_plane(nbytes(uv, chroma_h)), std::ptr::null_mut())
        } else {
            let uv = align32((width / 2) * 2);
            (uv, alloc_plane(nbytes(uv, chroma_h)), alloc_plane(nbytes(uv, chroma_h)))
        };

        HdrFrame {
            width,
            height,
            format,
            y_stride,
            uv_stride,
            chroma_h,
            semi_planar,
            y,
            u,
            v,
        }
    }

    pub fn width(&self) -> i32 {
        self.width
    }
    pub fn height(&self) -> i32 {
        self.height
    }
    pub fn format(&self) -> PixelFormat {
        self.format
    }
    pub fn y_stride(&self) -> i32 {
        self.y_stride
    }
    pub fn uv_stride(&self) -> i32 {
        self.uv_stride
    }

    /// Writable luma plane (16-bit samples).
    pub fn y_mut(&mut self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.y as *mut u16, nsamples(self.y_stride, self.height)) }
    }
    /// Cb plane (I010/I210) or interleaved UV plane (P010/P210).
    pub fn u_mut(&mut self) -> &mut [u16] {
        unsafe {
            std::slice::from_raw_parts_mut(self.u as *mut u16, nsamples(self.uv_stride, self.chroma_h))
        }
    }
    /// Cr plane for planar formats; `None` for P010/P210.
    pub fn v_mut(&mut self) -> Option<&mut [u16]> {
        if self.v.is_null() {
            return None;
        }
        Some(unsafe {
            std::slice::from_raw_parts_mut(self.v as *mut u16, nsamples(self.uv_stride, self.chroma_h))
        })
    }

    pub(crate) fn planes(&self) -> (*const u16, *const u16, *const u16) {
        (self.y as *const u16, self.u as *const u16, self.v as *const u16)
    }
}

impl Drop for HdrFrame {
    fn drop(&mut self) {
        free_plane(self.y, nbytes(self.y_stride, self.height));
        free_plane(self.u, nbytes(self.uv_stride, self.chroma_h));
        if !self.semi_planar {
            free_plane(self.v, nbytes(self.uv_stride, self.chroma_h));
        }
    }
}
