package funnelcake
import "testing"
func TestRejectInvalidDimensions(t *testing.T) {
    for _, wh := range [][2]int{{0,2},{1,2},{128,3},{2147483646,2},{32768,32768}} {
        if f, err := NewFrame(wh[0],wh[1]); err == nil { f.Close(); t.Fatalf("accepted %v",wh) }
        if f, err := NewHDRFrame(wh[0],wh[1],PixP010); err == nil { f.Close(); t.Fatalf("accepted HDR %v",wh) }
        if s, err := NewScaler(ScalerConfig{SrcWidth: wh[0],SrcHeight: wh[1],Flags:Scale2X}); err == nil { s.Close();t.Fatalf("accepted scaler %v",wh) }
    }
}
