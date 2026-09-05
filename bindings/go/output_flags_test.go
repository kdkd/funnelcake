package funnelcake

import "testing"

func TestOutputFlags(t *testing.T) {
	s, err := NewScaler(ScalerConfig{SrcWidth: 256, SrcHeight: 64, Flags: Scale4X})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, flag := range []ScaleFlag{0, Scale2X | Scale4X, Scale4X | (1 << 31)} {
		if _, ok := s.Output(flag); ok {
			t.Fatalf("accepted %x", flag)
		}
	}
}
