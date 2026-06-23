// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

use funnelcake::*;

/// 256x256 -> 2x downscale. With SIMD the step must not fall back, which proves
/// the frame's planes were correctly aligned.
#[test]
fn sdr_round_trip() {
    let (w, h) = (256, 256);
    let mut f = Frame::new(w, h);
    f.y_mut().fill(128);
    f.u_mut().fill(128);
    f.v_mut().fill(128);

    let mut s = Scaler::new(&ScalerConfig::new(w, h, SCALE_2X)).expect("init");
    s.run(&f);

    let out = s.output(SCALE_2X).expect("2x produced");
    assert_eq!(out.width, w / 2);
    assert_eq!(out.height, h / 2);
    assert_eq!(out.y_stride % 32, 0);
    assert_eq!(out.uv_stride % 32, 0);
    assert_eq!(out.y().len(), (out.y_stride * out.height) as usize);

    if simd_available() {
        assert!(!out.fallback, "SIMD available but step fell back (alignment broken?)");
        assert!(s.warnings().perfect(), "expected a perfect result");
    }
}

/// 10-bit I010 -> 2x producing HDR + tone-mapped SDR + a 1:1 copy.
#[test]
fn hdr_round_trip() {
    let (w, h) = (256, 256);
    let mut f = HdrFrame::new(w, h, PixelFormat::I010);
    f.y_mut().fill(512);
    f.u_mut().fill(512);
    if let Some(v) = f.v_mut() {
        v.fill(512);
    }

    let cfg = HdrConfig {
        src_width: w,
        src_height: h,
        format: PixelFormat::I010,
        transfer: Transfer::Pq,
        flags: SCALE_2X,
        hdr_flags: SCALE_2X,
        sdr_flags: SCALE_2X,
        tonemap_1x: true,
        tonemap: TonemapConfig {
            curve: TonemapCurve::Bt2390,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut s = HdrScaler::new(&cfg).expect("hdr init");
    s.run(&f);

    let hdr = s.hdr_output(SCALE_2X).expect("hdr 2x");
    assert_eq!((hdr.width, hdr.height), (w / 2, h / 2));
    assert_eq!(hdr.y().len(), ((hdr.y_stride / 2) * hdr.height) as usize);

    let sdr = s.sdr_output(SCALE_2X).expect("sdr 2x");
    assert_eq!((sdr.width, sdr.height), (w / 2, h / 2));

    let one = s.tonemap_1x_output().expect("tonemap 1x");
    assert_eq!((one.width, one.height), (w, h));
}

/// Mixing thirds and pow2 families is a hard error.
#[test]
fn invalid_flags() {
    match Scaler::new(&ScalerConfig::new(256, 256, SCALE_2X | SCALE_3X)) {
        Err(e) => assert_eq!(e, Error::InvalidFlags),
        Ok(_) => panic!("expected an error for mixed-family flags"),
    }
}

/// Constant input must yield constant output, with each plane carrying its own
/// value — this exercises the actual data path (plane mapping, strides,
/// pointers), not just output dimensions.
#[test]
fn flat_field_values() {
    let (w, h) = (256, 256);
    let mut f = Frame::new(w, h);
    f.y_mut().fill(128);
    f.u_mut().fill(64);
    f.v_mut().fill(192);

    let mut s = Scaler::new(&ScalerConfig::new(w, h, SCALE_2X)).expect("init");
    s.run(&f);
    let out = s.output(SCALE_2X).expect("2x");

    assert_row_const(out.y(), out.y_stride, out.width, out.height, 128, "Y");
    assert_row_const(out.u(), out.uv_stride, out.width / 2, out.height / 2, 64, "U");
    assert_row_const(out.v(), out.uv_stride, out.width / 2, out.height / 2, 192, "V");
}

fn assert_row_const(plane: &[u8], stride: i32, width: i32, height: i32, want: u8, name: &str) {
    for row in [0, height - 1] {
        let base = (row * stride) as usize;
        for col in 0..width as usize {
            assert_eq!(plane[base + col], want, "{name} plane at row {row} col {col}");
        }
    }
}

/// A scaler can be reused across frames; the second result must reflect the
/// second frame's content.
#[test]
fn reuse_scaler() {
    let (w, h) = (256, 256);
    let mut f = Frame::new(w, h);
    let mut s = Scaler::new(&ScalerConfig::new(w, h, SCALE_2X)).expect("init");

    f.y_mut().fill(10);
    s.run(&f);
    assert_eq!(s.output(SCALE_2X).unwrap().y()[0], 10);

    f.y_mut().fill(200);
    s.run(&f);
    assert_eq!(s.output(SCALE_2X).unwrap().y()[0], 200);
}

/// The semi-planar P010 path produces correctly-sized outputs.
#[test]
fn hdr_p010() {
    let (w, h) = (256, 256);
    let mut f = HdrFrame::new(w, h, PixelFormat::P010);
    f.y_mut().fill(600);
    f.u_mut().fill(512); // interleaved UV
    assert!(f.v_mut().is_none(), "P010 has no separate V plane");

    let cfg = HdrConfig {
        src_width: w,
        src_height: h,
        format: PixelFormat::P010,
        transfer: Transfer::Pq,
        flags: SCALE_2X,
        hdr_flags: SCALE_2X,
        ..Default::default()
    };
    let mut s = HdrScaler::new(&cfg).expect("init");
    s.run(&f);
    let hdr = s.hdr_output(SCALE_2X).expect("hdr 2x");
    assert_eq!((hdr.width, hdr.height), (w / 2, h / 2));
}

/// A custom LUT of the wrong length is rejected (would otherwise be an
/// out-of-bounds read in the native LUT builder).
#[test]
fn custom_lut_wrong_length() {
    let cfg = HdrConfig {
        src_width: 256,
        src_height: 256,
        flags: SCALE_2X,
        sdr_flags: SCALE_2X,
        tonemap: TonemapConfig {
            curve: TonemapCurve::Custom,
            custom_lut: Some(vec![0u8; 100]),
            ..Default::default()
        },
        ..Default::default()
    };
    match HdrScaler::new(&cfg) {
        Err(e) => assert_eq!(e, Error::CustomLutLength),
        Ok(_) => panic!("expected CustomLutLength error"),
    }
}

/// Running with a mismatched frame size panics rather than reading OOB.
#[test]
#[should_panic(expected = "does not match scaler source")]
fn frame_size_mismatch_panics() {
    let mut s = Scaler::new(&ScalerConfig::new(256, 256, SCALE_2X)).expect("init");
    let f = Frame::new(128, 128);
    s.run(&f);
}
