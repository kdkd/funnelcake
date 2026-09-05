use funnelcake::*;
#[test]
fn reject_invalid_frame_sizes() {
    for (w, h) in [(0, 2), (1, 2), (128, 3), (i32::MAX - 1, 2), (32768, 32768)] {
        assert!(std::panic::catch_unwind(|| Frame::new(w, h)).is_err());
        assert!(std::panic::catch_unwind(|| HdrFrame::new(w, h, PixelFormat::P010)).is_err());
        assert!(Scaler::new(&ScalerConfig::new(w, h, SCALE_2X)).is_err());
    }
}
