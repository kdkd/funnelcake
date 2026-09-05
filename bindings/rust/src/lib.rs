// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

//! Idiomatic Rust bindings for the funnelcake SIMD YUV scaler and HDR
//! tone-mapper.
//!
//! Thin and safe: [`Frame`]/[`HdrFrame`] allocate input planes with the
//! 32-byte alignment the SIMD kernels require (via [`std::alloc`]), the scaler
//! types wrap the create/run/free lifecycle with [`Drop`], and outputs borrow
//! the scaler so the "valid until next run" rule is enforced at compile time.

mod ffi;
mod frame;
mod hdr;
mod scaler;

pub use frame::{Frame, HdrFrame};
pub use hdr::{HdrOutput, HdrScaler, HdrConfig, TonemapConfig};
pub use scaler::{Output, Scaler, ScalerConfig};

use std::sync::Once;

// ---- scale / upscale / option flags (bitmasks; combine with `|`) ----

/// Downscale step flags. A single scaler must use flags from ONE family.
pub const SCALE_1_5X: u32 = 1 << 0; // thirds family
pub const SCALE_2X: u32 = 1 << 1; // pow2 family
pub const SCALE_3X: u32 = 1 << 2; // thirds family
pub const SCALE_4X: u32 = 1 << 3; // pow2 family
pub const SCALE_6X: u32 = 1 << 4; // thirds family
pub const SCALE_8X: u32 = 1 << 5; // pow2 family
pub const SCALE_12X: u32 = 1 << 6; // thirds family
pub const SCALE_16X: u32 = 1 << 7; // pow2 family

pub const SCALE_THIRDS_MASK: u32 = SCALE_1_5X | SCALE_3X | SCALE_6X | SCALE_12X;
pub const SCALE_POW2_MASK: u32 = SCALE_2X | SCALE_4X | SCALE_8X | SCALE_16X;

/// Upscale cascade levels. The requested set must be a contiguous prefix.
pub const UPSCALE_2X: u32 = 1 << 0;
pub const UPSCALE_4X: u32 = 1 << 1;
pub const UPSCALE_8X: u32 = 1 << 2;
pub const UPSCALE_16X: u32 = 1 << 3;
pub const UPSCALE_32X: u32 = 1 << 4;

/// Reject steps that would require cropping (default: crop + warn).
pub const OPT_NO_CROP: u32 = 1 << 0;
/// Reject steps that cannot use SIMD (default: scalar + warn).
pub const OPT_NO_FALLBACK: u32 = 1 << 1;

// ---- enumerated parameters ----

/// 10-bit input pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum PixelFormat {
    #[default]
    I010 = 0, // 4:2:0 planar
    P010 = 1, // 4:2:0 semi-planar (interleaved UV)
    I210 = 2, // 4:2:2 planar (decimated to 4:2:0)
    P210 = 3, // 4:2:2 semi-planar (decimated to 4:2:0)
}

/// HDR source transfer function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum Transfer {
    #[default]
    Pq = 0, // SMPTE ST 2084 (HDR10)
    Hlg = 1, // Hybrid Log-Gamma
}

/// Quantization range for tone mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum Range {
    #[default]
    Limited = 0, // video range
    Full = 1,    // full / PC range
}

/// Tone-mapping curve preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum TonemapCurve {
    #[default]
    Hable = 0, // filmic
    Reinhard = 1,
    Bt2390 = 2,
    Custom = 3, // use TonemapConfig::custom_lut
}

// ---- errors and warnings ----

/// A hard error returned by a scaler constructor (negative `FUSED_ERR_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidFlags,
    NoSteps,
    BadDimensions,
    BadAlignment,
    /// A custom tone-map LUT was supplied with a length other than 1024.
    CustomLutLength,
    Unknown(i32),
}

impl Error {
    pub(crate) fn from_code(code: i32) -> Error {
        match code {
            -1 => Error::InvalidFlags,
            -2 => Error::NoSteps,
            -3 => Error::BadDimensions,
            -4 => Error::BadAlignment,
            other => Error::Unknown(other),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Error::InvalidFlags => "invalid flags (mixed families or unknown bits)",
            Error::NoSteps => "no valid scale steps requested",
            Error::BadDimensions => "bad source dimensions",
            Error::BadAlignment => "source strides are not 32-byte aligned",
            Error::CustomLutLength => "custom tone-map LUT must be exactly 1024 bytes",
            Error::Unknown(_) => "unknown error",
        };
        write!(f, "funnelcake: {msg}")
    }
}

impl std::error::Error for Error {}

/// Non-fatal conditions reported by a successful init.
#[derive(Debug, Clone, Copy)]
pub struct Warnings(pub u32);

impl Warnings {
    /// A step fell back to the scalar kernel.
    pub fn scalar(&self) -> bool {
        self.0 & (1 << 0) != 0
    }
    /// At least one requested step was rejected.
    pub fn partial(&self) -> bool {
        self.0 & (1 << 1) != 0
    }
    /// The source was cropped to fit.
    pub fn cropped(&self) -> bool {
        self.0 & (1 << 2) != 0
    }
    /// Every requested output was produced with SIMD and no cropping.
    pub fn perfect(&self) -> bool {
        self.0 == 0
    }
}

// ---- shared helpers ----

static DETECT: Once = Once::new();

/// Force the library's one-time CPU probe before any thread can race on it.
pub(crate) fn ensure_detect() {
    DETECT.call_once(|| unsafe {
        let _ = ffi::fused_simd_available();
    });
}

/// Reports whether the scalers will use vectorized kernels on this machine.
pub fn simd_available() -> bool {
    ensure_detect();
    unsafe { ffi::fused_simd_available() == 1 }
}

/// Round up to the next multiple of 32 (the SIMD stride/alignment unit).
pub(crate) fn align32(n: i32) -> i32 {
    n.checked_add(31).filter(|_| n > 0).map(|v| v & !31).unwrap_or(0)
}

pub(crate) fn valid_dimensions(width: i32, height: i32, bytes: i32) -> bool {
    width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0
        && width <= i32::MAX / 64 && height <= i32::MAX / 64
        && i64::from(align32(width * bytes)) * i64::from(height) <= i64::from(i32::MAX)
}

/// Tightly packed, independent output planes. Samples are u8 or u16.
#[derive(Debug, Clone)]
pub struct OwnedOutput<T> {
    pub width: i32,
    pub height: i32,
    pub y: Vec<T>,
    pub u: Vec<T>,
    pub v: Vec<T>,
}
