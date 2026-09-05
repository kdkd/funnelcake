use funnelcake::*;
#[test]
fn packed_output_survives_scaler() {
    let mut frame=Frame::new(136,64);
    frame.y_mut().fill(91);
    let owned;
    {
        let mut scaler=Scaler::new(&ScalerConfig::new(136,64,SCALE_2X)).unwrap();
        scaler.run(&frame);
        let out=scaler.output(SCALE_2X).unwrap();
        assert_eq!(out.y_row(0).len(),out.width() as usize);
        owned=out.to_owned();
    }
    assert_eq!(owned.y.len(),(owned.width*owned.height) as usize);
    assert!(owned.y.iter().all(|&x|x==91));
}
