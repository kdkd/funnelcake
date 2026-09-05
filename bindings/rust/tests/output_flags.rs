use funnelcake::*;
#[test]
fn rejects_output_masks() {
    let s = Scaler::new(&ScalerConfig::new(256, 64, SCALE_4X)).unwrap();
    for flag in [0, SCALE_2X | SCALE_4X, SCALE_4X | (1 << 31)] {
        assert!(s.output(flag).is_none());
    }
    assert_eq!(s.output(SCALE_4X).unwrap().width(), 64);
}
