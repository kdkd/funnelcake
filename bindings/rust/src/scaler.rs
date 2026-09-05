// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

use std::marker::PhantomData;

use crate::{align32, ensure_detect, ffi, Error, Frame, Warnings};

/// Configuration for a [`Scaler`]. Source strides are derived from `src_width`,
/// so a [`Frame`] built with the same width/height matches automatically.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalerConfig {
    pub src_width: i32,
    pub src_height: i32,
    /// Downscale steps (`SCALE_*`), one family only.
    pub flags: u32,
    /// Upscale cascade (`UPSCALE_*`), a contiguous prefix.
    pub upscale_flags: u32,
    pub upscale_tail_1_5x: bool,
    /// `OPT_*` bitmask.
    pub options: u32,
}

impl ScalerConfig {
    /// A plain downscale with no upscaling or options.
    pub fn new(src_width: i32, src_height: i32, flags: u32) -> ScalerConfig {
        ScalerConfig {
            src_width,
            src_height,
            flags,
            ..Default::default()
        }
    }
}

/// An initialized 8-bit scaling context. Owns its output buffers; the
/// [`Output`] views borrow `&self`, so the borrow checker prevents using them
/// across a `run` (which needs `&mut self`) or after `drop`.
pub struct Scaler {
    ctx: Box<ffi::ScalerCtx>,
    warnings: Warnings,
    src_width: i32,
    src_height: i32,
}

impl Scaler {
    /// Validates the configuration and allocates output buffers.
    pub fn new(cfg: &ScalerConfig) -> Result<Scaler, Error> {
        if !crate::valid_dimensions(cfg.src_width, cfg.src_height, 2) { return Err(Error::BadDimensions); }
        ensure_detect();
        // SAFETY: ScalerCtx is a repr(C) POD; an all-zero value is valid
        // (null pointers, zeroed log configs => stderr defaults).
        let mut ctx: Box<ffi::ScalerCtx> = Box::new(unsafe { std::mem::zeroed() });

        ctx.src_width = cfg.src_width;
        ctx.src_height = cfg.src_height;
        ctx.src_y_stride = align32(cfg.src_width);
        ctx.src_uv_stride = align32(cfg.src_width / 2);
        ctx.requested_flags = cfg.flags;
        ctx.options = cfg.options;
        ctx.upscale_flags = cfg.upscale_flags;
        ctx.upscale_tail_1_5x = cfg.upscale_tail_1_5x as i32;

        let rc = unsafe { ffi::fused_scaler_init(&mut *ctx) };
        if rc < 0 {
            return Err(Error::from_code(rc));
        }
        Ok(Scaler {
            ctx,
            warnings: Warnings(rc as u32),
            src_width: cfg.src_width,
            src_height: cfg.src_height,
        })
    }

    /// Non-fatal conditions reported by init.
    pub fn warnings(&self) -> Warnings {
        self.warnings
    }

    /// Scales one frame, filling all achieved outputs.
    ///
    /// # Panics
    /// If the frame's dimensions differ from the scaler's configured source
    /// size (the planes would be the wrong size, leading to out-of-bounds reads
    /// in the native kernels).
    pub fn run(&mut self, frame: &Frame) {
        assert!(
            frame.width() == self.src_width && frame.height() == self.src_height,
            "frame {}x{} does not match scaler source {}x{}",
            frame.width(),
            frame.height(),
            self.src_width,
            self.src_height
        );
        let (y, u, v) = frame.planes();
        unsafe { ffi::fused_scaler_run(&mut *self.ctx, y, u, v) }
    }

    /// Source luma width actually used (after any crop).
    pub fn effective_width(&self) -> i32 {
        self.ctx.effective_width
    }
    /// Source luma height actually used (after any crop).
    pub fn effective_height(&self) -> i32 {
        self.ctx.effective_height
    }
    /// Bitmask of downscale steps that were produced.
    pub fn achieved_flags(&self) -> u32 {
        self.ctx.achieved_flags
    }

    /// The downscale output for a single `SCALE_*` flag.
    pub fn output(&self, flag: u32) -> Option<Output<'_>> {
        if self.ctx.achieved_flags & flag == 0 {
            return None;
        }
        let idx = flag.trailing_zeros() as usize;
        Some(Output::from_c(&self.ctx.outputs[idx]))
    }

    /// An upscale-cascade output for a single `UPSCALE_*` flag.
    pub fn upscale_output(&self, flag: u32) -> Option<Output<'_>> {
        if self.ctx.achieved_upscale_flags & flag == 0 {
            return None;
        }
        let idx = flag.trailing_zeros() as usize;
        Some(Output::from_c(&self.ctx.upscale_outputs[idx]))
    }

    /// The 1.5x upscale tail output, if produced.
    pub fn upscale_tail(&self) -> Option<Output<'_>> {
        if self.ctx.achieved_upscale_tail == 0 {
            return None;
        }
        // Slot 5 == FUSED_UP_IDX_TAIL.
        Some(Output::from_c(&self.ctx.upscale_outputs[5]))
    }
}

impl Drop for Scaler {
    fn drop(&mut self) {
        unsafe { ffi::fused_scaler_free(&mut *self.ctx) }
    }
}

/// A read-only view of one 8-bit output plane set. The plane slices borrow the
/// producing [`Scaler`], so they cannot outlive it or survive its next `run`.
pub struct Output<'a> {
    pub width: i32,
    pub height: i32,
    pub y_stride: i32,
    pub uv_stride: i32,
    pub fallback: bool,
    y: *const u8,
    u: *const u8,
    v: *const u8,
    chroma_h: i32,
    _marker: PhantomData<&'a Scaler>,
}

impl<'a> Output<'a> {
    pub(crate) fn from_c(o: &'a ffi::ScaleOutput) -> Output<'a> {
        Output {
            width: o.width,
            height: o.height,
            y_stride: o.y_stride,
            uv_stride: o.uv_stride,
            fallback: o.fallback != 0,
            y: o.plane_y,
            u: o.plane_u,
            v: o.plane_v,
            chroma_h: (o.height + 1) / 2,
            _marker: PhantomData,
        }
    }

    /// Luma plane (`y_stride * height` bytes).
    pub fn y(&self) -> &'a [u8] {
        let n = self.y_stride as usize * self.height as usize;
        unsafe { std::slice::from_raw_parts(self.y, n) }
    }
    /// Cb plane.
    pub fn u(&self) -> &'a [u8] {
        let n = self.uv_stride as usize * self.chroma_h as usize;
        unsafe { std::slice::from_raw_parts(self.u, n) }
    }
    /// Cr plane.
    pub fn v(&self) -> &'a [u8] {
        let n = self.uv_stride as usize * self.chroma_h as usize;
        unsafe { std::slice::from_raw_parts(self.v, n) }
    }
}
