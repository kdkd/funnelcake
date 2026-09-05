package funnelcake

import (
	"strconv"
	"testing"
)

func TestRejectWideConfigIntegers(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("native int already has C width")
	}
	value := int64(1) << 40
	cfg := HDRConfig{SrcWidth: 128, SrcHeight: 64, Flags: Scale2X, HDRFlags: Scale2X}
	cfg.Tonemap.PeakNits = int(value)
	if s, err := NewHDRScaler(cfg); err == nil {
		s.Close()
		t.Fatal("accepted oversized nits")
	}
}
