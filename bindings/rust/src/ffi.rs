// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

//! Raw FFI declarations mirroring `include/funnelcake.h`.
//!
//! `#[repr(C)]` makes the Rust structs follow the platform C ABI, so the
//! layouts match the header automatically (no manual padding, no bindgen).
//! The `layout_sizes_match_c` test cross-checks total sizes against the real C
//! `sizeof` via the binding-helper functions, catching any field-list mistake.

use std::ffi::{c_int, c_void};

#[repr(C)]
pub struct LogConfig {
    pub target: c_int,
    pub file: *mut c_void,
    pub callback: *mut c_void,
    pub callback_ctx: *mut c_void,
}

#[repr(C)]
pub struct ScaleOutput {
    pub width: c_int,
    pub height: c_int,
    pub y_stride: c_int,
    pub uv_stride: c_int,
    pub plane_y: *mut u8,
    pub plane_u: *mut u8,
    pub plane_v: *mut u8,
    pub fallback: c_int,
}

#[repr(C)]
pub struct HdrOutput {
    pub width: c_int,
    pub height: c_int,
    pub y_stride: c_int,
    pub uv_stride: c_int,
    pub plane_y: *mut u16,
    pub plane_u: *mut u16,
    pub plane_v: *mut u16,
    pub fallback: c_int,
}

#[repr(C)]
pub struct TonemapConfig {
    pub curve: c_int,
    pub peak_nits: c_int,
    pub target_nits: c_int,
    pub custom_lut: *const u8,
    pub src_range: c_int,
    pub dst_range: c_int,
}

#[repr(C)]
pub struct ScalerCtx {
    pub src_width: c_int,
    pub src_height: c_int,
    pub src_y_stride: c_int,
    pub src_uv_stride: c_int,
    pub requested_flags: u32,
    pub options: u32,
    pub log_errors: LogConfig,
    pub log_warnings: LogConfig,
    pub upscale_flags: u32,
    pub upscale_tail_1_5x: c_int,
    pub achieved_flags: u32,
    pub rejected_flags: u32,
    pub effective_width: c_int,
    pub effective_height: c_int,
    pub outputs: [ScaleOutput; 8],
    pub achieved_upscale_flags: u32,
    pub achieved_upscale_tail: c_int,
    pub upscale_outputs: [ScaleOutput; 6],
    pub _internal: *mut c_void,
}

#[repr(C)]
pub struct HdrCtx {
    pub src_width: c_int,
    pub src_height: c_int,
    pub src_y_stride: c_int,
    pub src_uv_stride: c_int,
    pub src_format: c_int,
    pub src_transfer: c_int,
    pub requested_flags: u32,
    pub hdr_flags: u32,
    pub sdr_flags: u32,
    pub options: u32,
    pub tonemap_1x: c_int,
    pub tonemap: TonemapConfig,
    pub log_errors: LogConfig,
    pub log_warnings: LogConfig,
    pub achieved_hdr_flags: u32,
    pub achieved_sdr_flags: u32,
    pub rejected_flags: u32,
    pub effective_width: c_int,
    pub effective_height: c_int,
    pub hdr_outputs: [HdrOutput; 8],
    pub sdr_outputs: [ScaleOutput; 8],
    pub output_1x: ScaleOutput,
    pub upscale_flags: u32,
    pub upscale_tail_1_5x: c_int,
    pub upscale_sdr_flags: u32,
    pub upscale_sdr_tail_1_5x: c_int,
    pub achieved_upscale_flags: u32,
    pub achieved_upscale_tail: c_int,
    pub achieved_upscale_sdr_flags: u32,
    pub achieved_upscale_sdr_tail: c_int,
    pub upscale_hdr_outputs: [HdrOutput; 6],
    pub upscale_sdr_outputs: [ScaleOutput; 6],
    pub _internal: *mut c_void,
}

extern "C" {
    pub fn fused_scaler_init(ctx: *mut ScalerCtx) -> c_int;
    pub fn fused_scaler_run(ctx: *mut ScalerCtx, y: *const u8, u: *const u8, v: *const u8);
    pub fn fused_scaler_free(ctx: *mut ScalerCtx);

    pub fn fused_hdr_init(ctx: *mut HdrCtx) -> c_int;
    pub fn fused_hdr_run(ctx: *mut HdrCtx, y: *const u16, u: *const u16, v: *const u16);
    pub fn fused_hdr_free(ctx: *mut HdrCtx);

    pub fn fused_simd_available() -> c_int;

    // From libfunnelcake_helpers (built by build.rs). Used by the layout test;
    // unused in a normal (non-test) build of the crate.
    #[allow(dead_code)]
    pub fn fused_scaler_ctx_sizeof() -> usize;
    #[allow(dead_code)]
    pub fn fused_hdr_ctx_sizeof() -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn layout_sizes_match_c() {
        // Guards the hand-written repr(C) structs against drift in funnelcake.h.
        unsafe {
            assert_eq!(size_of::<ScalerCtx>(), fused_scaler_ctx_sizeof(), "ScalerCtx size");
            assert_eq!(size_of::<HdrCtx>(), fused_hdr_ctx_sizeof(), "HdrCtx size");
        }
    }
}
