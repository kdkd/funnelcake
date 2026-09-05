use funnelcake::*;
#[test]
fn native_diagnostics() {
    assert!(!version().is_empty());
    assert!(matches!(backend(),"scalar"|"avx2"|"avx512"|"neon"|"rvv"));
}
