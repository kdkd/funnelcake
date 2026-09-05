// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

use std::marker::PhantomData;

use crate::{
    align32, ensure_detect, ffi, Error, HdrFrame, PixelFormat, Range, Transfer, TonemapCurve,
    Warnings,
};

/// Tone-mapping configuration. The default is the Hable curve, library-default
/// nits (1000 peak / 100 target), and limited range in and out.
#[derive(Debug, Clone, Default)]
pub struct TonemapConfig {
    pub curve: TonemapCurve,
    /// Source peak (0 = library default 1000).
    pub peak_nits: i32,
    /// SDR target (0 = library default 100).
    pub target_nits: i32,
    pub src_range: Range,
    pub dst_range: Range,
    /// 1024-entry Y LUT, used only with [`TonemapCurve::Custom`]. Copied into
    /// the library at init.
    pub custom_lut: Option<Vec<u8>>,
}

/// Configuration for an [`HdrScaler`]. `hdr_flags` and `sdr_flags` must each be
/// a subset of `flags`.
#[derive(Debug, Clone, Default)]
pub struct HdrConfig {
    pub src_width: i32,
    pub src_height: i32,
    pub format: PixelFormat,
    pub transfer: Transfer,
    /// Requested downscale steps, one family.
    pub flags: u32,
    /// Subset: produce 10-bit HDR outputs.
    pub hdr_flags: u32,
    /// Subset: produce tone-mapped 8-bit outputs.
    pub sdr_flags: u32,
    pub options: u32,
    /// Also produce a 1:1 tone-mapped SDR copy.
    pub tonemap_1x: bool,
    pub tonemap: TonemapConfig,
    pub upscale_flags: u32,
    pub upscale_tail_1_5x: bool,
    /// Subset of `upscale_flags`: also produce tone-mapped SDR copies.
    pub upscale_sdr_flags: u32,
    pub upscale_sdr_tail_1_5x: bool,
}

/// An initialized 10-bit scaling / tone-mapping context. Owns its output
/// buffers; views borrow `&self` (valid until the next `run` or drop).
pub struct HdrScaler {
    ctx: Box<ffi::HdrCtx>,
    warnings: Warnings,
    src_width: i32,
    src_height: i32,
    format: PixelFormat,
}

impl HdrScaler {
    /// Builds tone-mapping LUTs and allocates outputs.
    pub fn new(cfg: &HdrConfig) -> Result<HdrScaler, Error> {
        if !crate::valid_dimensions(cfg.src_width, cfg.src_height, 2) { return Err(Error::BadDimensions); }
        ensure_detect();
        let mut ctx: Box<ffi::HdrCtx> = Box::new(unsafe { std::mem::zeroed() });

        let semi_planar = matches!(cfg.format, PixelFormat::P010 | PixelFormat::P210);
        let y_stride = align32(cfg.src_width * 2);
        let uv_stride = if semi_planar {
            y_stride
        } else {
            align32((cfg.src_width / 2) * 2)
        };

        ctx.src_width = cfg.src_width;
        ctx.src_height = cfg.src_height;
        ctx.src_y_stride = y_stride;
        ctx.src_uv_stride = uv_stride;
        ctx.src_format = cfg.format as i32;
        ctx.src_transfer = cfg.transfer as i32;
        ctx.requested_flags = cfg.flags;
        ctx.hdr_flags = cfg.hdr_flags;
        ctx.sdr_flags = cfg.sdr_flags;
        ctx.options = cfg.options;
        ctx.tonemap_1x = cfg.tonemap_1x as i32;

        ctx.tonemap.curve = cfg.tonemap.curve as i32;
        ctx.tonemap.peak_nits = cfg.tonemap.peak_nits;
        ctx.tonemap.target_nits = cfg.tonemap.target_nits;
        ctx.tonemap.src_range = cfg.tonemap.src_range as i32;
        ctx.tonemap.dst_range = cfg.tonemap.dst_range as i32;
        // `cfg` (and thus the LUT) lives for this whole call; the library reads
        // the LUT during init only, so the borrow is sufficient.
        if let Some(lut) = &cfg.tonemap.custom_lut {
            if lut.len() != 1024 {
                return Err(Error::CustomLutLength);
            }
            ctx.tonemap.custom_lut = lut.as_ptr();
        }

        ctx.upscale_flags = cfg.upscale_flags;
        ctx.upscale_tail_1_5x = cfg.upscale_tail_1_5x as i32;
        ctx.upscale_sdr_flags = cfg.upscale_sdr_flags;
        ctx.upscale_sdr_tail_1_5x = cfg.upscale_sdr_tail_1_5x as i32;

        let rc = unsafe { ffi::fused_hdr_init(&mut *ctx) };
        if rc < 0 {
            return Err(Error::from_code(rc));
        }
        Ok(HdrScaler {
            ctx,
            warnings: Warnings(rc as u32),
            src_width: cfg.src_width,
            src_height: cfg.src_height,
            format: cfg.format,
        })
    }

    pub fn warnings(&self) -> Warnings {
        self.warnings
    }

    /// Scales and tone-maps one 10-bit frame.
    ///
    /// # Panics
    /// If the frame's dimensions or format differ from the scaler's
    /// configuration (the planes would be the wrong size/layout).
    pub fn run(&mut self, frame: &HdrFrame) {
        assert!(
            frame.width() == self.src_width
                && frame.height() == self.src_height
                && frame.format() == self.format,
            "frame {}x{} {:?} does not match scaler {}x{} {:?}",
            frame.width(),
            frame.height(),
            frame.format(),
            self.src_width,
            self.src_height,
            self.format
        );
        let (y, u, v) = frame.planes();
        unsafe { ffi::fused_hdr_run(&mut *self.ctx, y, u, v) }
    }

    pub fn effective_width(&self) -> i32 {
        self.ctx.effective_width
    }
    pub fn effective_height(&self) -> i32 {
        self.ctx.effective_height
    }

    /// The 10-bit HDR output for a downscale flag.
    pub fn hdr_output(&self, flag: u32) -> Option<HdrOutput<'_>> {
        if !flag.is_power_of_two() || self.ctx.achieved_hdr_flags & flag == 0 {
            return None;
        }
        let idx = flag.trailing_zeros() as usize;
        Some(HdrOutput::from_c(&self.ctx.hdr_outputs[idx]))
    }

    /// The tone-mapped 8-bit output for a downscale flag.
    pub fn sdr_output(&self, flag: u32) -> Option<crate::Output<'_>> {
        if !flag.is_power_of_two() || self.ctx.achieved_sdr_flags & flag == 0 {
            return None;
        }
        let idx = flag.trailing_zeros() as usize;
        Some(crate::scaler::Output::from_c(&self.ctx.sdr_outputs[idx]))
    }

    /// The 1:1 tone-mapped SDR copy, if produced.
    pub fn tonemap_1x_output(&self) -> Option<crate::Output<'_>> {
        if self.ctx.output_1x.plane_y.is_null() {
            return None;
        }
        Some(crate::scaler::Output::from_c(&self.ctx.output_1x))
    }

    /// A 10-bit upscale-cascade output for a flag.
    pub fn upscale_hdr_output(&self, flag: u32) -> Option<HdrOutput<'_>> {
        if !flag.is_power_of_two() || self.ctx.achieved_upscale_flags & flag == 0 {
            return None;
        }
        let idx = flag.trailing_zeros() as usize;
        Some(HdrOutput::from_c(&self.ctx.upscale_hdr_outputs[idx]))
    }

    /// A tone-mapped 8-bit upscale-cascade output for a flag.
    pub fn upscale_sdr_output(&self, flag: u32) -> Option<crate::Output<'_>> {
        if !flag.is_power_of_two() || self.ctx.achieved_upscale_sdr_flags & flag == 0 {
            return None;
        }
        let idx = flag.trailing_zeros() as usize;
        Some(crate::scaler::Output::from_c(&self.ctx.upscale_sdr_outputs[idx]))
    }
}

impl Drop for HdrScaler {
    fn drop(&mut self) {
        unsafe { ffi::fused_hdr_free(&mut *self.ctx) }
    }
}

/// Allocation metadata cannot be modified by callers.
/// ```compile_fail
/// fn change(mut out: funnelcake::HdrOutput<'_>) {
///     out.height = 4096;
/// }
/// ```
/// A read-only view of one 10-bit output plane set. Samples are unsigned 16-bit
/// (10 significant bits). Strides are in bytes. The slices borrow the producing
/// [`HdrScaler`].
pub struct HdrOutput<'a> {
    width: i32,
    height: i32,
    y_stride: i32,
    uv_stride: i32,
    pub fallback: bool,
    y: *const u16,
    u: *const u16,
    v: *const u16,
    chroma_h: i32,
    _marker: PhantomData<&'a HdrScaler>,
}

impl<'a> HdrOutput<'a> {
    fn from_c(o: &'a ffi::HdrOutput) -> HdrOutput<'a> {
        HdrOutput {
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

    pub fn width(&self) -> i32 { self.width }
    pub fn height(&self) -> i32 { self.height }
    pub fn y_stride(&self) -> i32 { self.y_stride }
    pub fn uv_stride(&self) -> i32 { self.uv_stride }

    /// Luma plane (16-bit samples), `(y_stride/2) * height` elements.
    pub fn y(&self) -> &'a [u16] {
        let n = (self.y_stride as usize / 2) * self.height as usize;
        unsafe { std::slice::from_raw_parts(self.y, n) }
    }
    /// Cb plane (16-bit samples).
    pub fn u(&self) -> &'a [u16] {
        let n = (self.uv_stride as usize / 2) * self.chroma_h as usize;
        unsafe { std::slice::from_raw_parts(self.u, n) }
    }
    /// Cr plane (16-bit samples).
    pub fn v(&self) -> &'a [u16] {
        let n = (self.uv_stride as usize / 2) * self.chroma_h as usize;
        unsafe { std::slice::from_raw_parts(self.v, n) }
    }
}
