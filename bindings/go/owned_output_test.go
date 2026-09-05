package funnelcake

import "testing"

func TestPackedOutput(t *testing.T) {
	f, _ := NewFrame(136, 64)
	defer f.Close()
	for i := range f.Y() {
		f.Y()[i] = 91
	}
	s, err := NewScaler(ScalerConfig{SrcWidth: 136, SrcHeight: 64, Flags: Scale2X})
	if err != nil {
		t.Fatal(err)
	}
	s.Run(f)
	o, _ := s.Output(Scale2X)
	if len(o.YRow(0)) != o.Width {
		t.Fatal("row includes padding")
	}
	y, u, v := o.Packed()
	s.Close()
	if len(y) != o.Width*o.Height || len(u) != o.Width/2*(o.Height/2) || len(v) != len(u) {
		t.Fatal("bad packed size")
	}
	for _, value := range y {
		if value != 91 {
			t.Fatal("bad copy")
		}
	}
}
