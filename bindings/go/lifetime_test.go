package funnelcake

import (
	"runtime"
	"testing"
)

func TestPlaneOwnership(t *testing.T) {
	f, err := NewFrame(128, 64)
	if err != nil {
		t.Fatal(err)
	}
	y := f.Y()
	y[0] = 93
	f.Height = 1 << 20
	f.YStride = 1 << 20
	if len(f.Y()) != 8192 {
		t.Fatal("mutable metadata changed allocation view")
	}
	f.Close()
	f = nil
	runtime.GC()
	if y[0] != 93 {
		t.Fatal("frame slice did not survive close")
	}
	h, err := NewHDRFrame(128, 64, PixP010)
	if err != nil {
		t.Fatal(err)
	}
	if len(h.U()) != 128*32 {
		t.Fatal("wrong interleaved plane size")
	}
	uv := h.U()
	uv[0] = 511
	h.Close()
	h = nil
	runtime.GC()
	if uv[0] != 511 {
		t.Fatal("HDR slice did not survive close")
	}
}

func TestOutputCopies(t *testing.T) {
	f, _ := NewFrame(128, 64)
	defer f.Close()
	for i := range f.Y() {
		f.Y()[i] = 88
	}
	s, err := NewScaler(ScalerConfig{SrcWidth: 128, SrcHeight: 64, Flags: Scale2X})
	if err != nil {
		t.Fatal(err)
	}
	s.Run(f)
	out, ok := s.Output(Scale2X)
	if !ok {
		t.Fatal("missing output")
	}
	out.Height = 1 << 20
	out.YStride = 1 << 20
	y := out.Y()
	if len(y) != 2048 {
		t.Fatal("mutable metadata changed output length")
	}
	y[0] = 11
	if out.Y()[0] != 88 {
		t.Fatal("returned output aliases native storage")
	}
	s.Close()
	runtime.GC()
	if y[0] != 11 {
		t.Fatal("owned copy changed")
	}
}
