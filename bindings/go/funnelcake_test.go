// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

package funnelcake

import (
	"errors"
	"testing"
)

// fillByte sets every element of b to v.
func fillByte(b []byte, v byte) {
	for i := range b {
		b[i] = v
	}
}

// fill16 sets every element of b to v.
func fill16(b []uint16, v uint16) {
	for i := range b {
		b[i] = v
	}
}

// TestSDRRoundTrip builds an aligned 256x256 frame, runs a 2x downscale, and
// checks the output geometry. With SIMD available the step must NOT fall back
// to scalar — which proves the frame's planes were correctly aligned.
func TestSDRRoundTrip(t *testing.T) {
	const w, h = 256, 256

	f, err := NewFrame(w, h)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	defer f.Close()
	fillByte(f.Y(), 128)
	fillByte(f.U(), 128)
	fillByte(f.V(), 128)

	s, err := NewScaler(ScalerConfig{SrcWidth: w, SrcHeight: h, Flags: Scale2X})
	if err != nil {
		t.Fatalf("NewScaler: %v", err)
	}
	defer s.Close()

	s.Run(f)

	out, ok := s.Output(Scale2X)
	if !ok {
		t.Fatal("2x output was not produced")
	}
	if out.Width != w/2 || out.Height != h/2 {
		t.Errorf("2x output dims = %dx%d, want %dx%d", out.Width, out.Height, w/2, h/2)
	}
	if out.YStride%32 != 0 || out.UVStride%32 != 0 {
		t.Errorf("output strides not 32-aligned: y=%d uv=%d", out.YStride, out.UVStride)
	}
	if len(out.Y()) != out.YStride*out.Height {
		t.Errorf("Y view len = %d, want %d", len(out.Y()), out.YStride*out.Height)
	}
	if SIMDAvailable() {
		if out.Fallback {
			t.Error("SIMD is available but the 2x step fell back to scalar (alignment broken?)")
		}
		if !s.Warnings.Perfect() {
			t.Errorf("expected a perfect result, got warnings scalar=%v partial=%v cropped=%v",
				s.Warnings.Scalar(), s.Warnings.Partial(), s.Warnings.Cropped())
		}
	} else {
		t.Log("SIMD not available on this build/CPU; scalar path exercised")
	}
}

// TestHDRRoundTrip runs a 10-bit I010 source through a 2x step producing both a
// 10-bit HDR output and a tone-mapped SDR output, plus a 1:1 tone-mapped copy.
func TestHDRRoundTrip(t *testing.T) {
	const w, h = 256, 256

	f, err := NewHDRFrame(w, h, PixI010)
	if err != nil {
		t.Fatalf("NewHDRFrame: %v", err)
	}
	defer f.Close()
	fill16(f.Y(), 512)
	fill16(f.U(), 512)
	fill16(f.V(), 512)

	s, err := NewHDRScaler(HDRConfig{
		SrcWidth:  w,
		SrcHeight: h,
		Format:    PixI010,
		Transfer:  TRCPQ,
		Flags:     Scale2X,
		HDRFlags:  Scale2X,
		SDRFlags:  Scale2X,
		Tonemap1x: true,
		Tonemap:   TonemapConfig{Curve: TonemapBT2390},
	})
	if err != nil {
		t.Fatalf("NewHDRScaler: %v", err)
	}
	defer s.Close()

	s.Run(f)

	hdr, ok := s.HDROutput(Scale2X)
	if !ok {
		t.Fatal("2x HDR output not produced")
	}
	if hdr.Width != w/2 || hdr.Height != h/2 {
		t.Errorf("HDR 2x dims = %dx%d, want %dx%d", hdr.Width, hdr.Height, w/2, h/2)
	}
	if len(hdr.Y()) != (hdr.YStride/2)*hdr.Height {
		t.Errorf("HDR Y view len = %d, want %d", len(hdr.Y()), (hdr.YStride/2)*hdr.Height)
	}

	sdr, ok := s.SDROutput(Scale2X)
	if !ok {
		t.Fatal("2x SDR output not produced")
	}
	if sdr.Width != w/2 || sdr.Height != h/2 {
		t.Errorf("SDR 2x dims = %dx%d, want %dx%d", sdr.Width, sdr.Height, w/2, h/2)
	}

	one, ok := s.Tonemap1xOutput()
	if !ok {
		t.Fatal("tonemap_1x output not produced")
	}
	if one.Width != w || one.Height != h {
		t.Errorf("tonemap_1x dims = %dx%d, want %dx%d", one.Width, one.Height, w, h)
	}
}

// assertRowConst checks the first and last output rows are a constant value
// across the active width — catching plane swaps, stride errors, bad pointers.
func assertRowConst(t *testing.T, name string, plane []byte, stride, width, height int, want byte) {
	t.Helper()
	for _, row := range []int{0, height - 1} {
		base := row * stride
		for col := 0; col < width; col++ {
			if plane[base+col] != want {
				t.Fatalf("%s plane row %d col %d = %d, want %d", name, row, col, plane[base+col], want)
				return
			}
		}
	}
}

// TestFlatFieldValues feeds a constant frame and checks each output plane comes
// back as its own constant — exercising the data path, not just dimensions.
func TestFlatFieldValues(t *testing.T) {
	const w, h = 256, 256
	f, err := NewFrame(w, h)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	fillByte(f.Y(), 128)
	fillByte(f.U(), 64)
	fillByte(f.V(), 192)

	s, err := NewScaler(ScalerConfig{SrcWidth: w, SrcHeight: h, Flags: Scale2X})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Run(f)

	out, ok := s.Output(Scale2X)
	if !ok {
		t.Fatal("no 2x output")
	}
	assertRowConst(t, "Y", out.Y(), out.YStride, out.Width, out.Height, 128)
	assertRowConst(t, "U", out.U(), out.UVStride, out.Width/2, out.Height/2, 64)
	assertRowConst(t, "V", out.V(), out.UVStride, out.Width/2, out.Height/2, 192)
}

// TestReuseScaler runs the same scaler on two frames.
func TestReuseScaler(t *testing.T) {
	const w, h = 256, 256
	f, _ := NewFrame(w, h)
	defer f.Close()
	s, _ := NewScaler(ScalerConfig{SrcWidth: w, SrcHeight: h, Flags: Scale2X})
	defer s.Close()

	fillByte(f.Y(), 10)
	s.Run(f)
	if out, _ := s.Output(Scale2X); out.Y()[0] != 10 {
		t.Fatalf("first run Y=%d, want 10", out.Y()[0])
	}
	fillByte(f.Y(), 200)
	s.Run(f)
	if out, _ := s.Output(Scale2X); out.Y()[0] != 200 {
		t.Fatalf("second run Y=%d, want 200", out.Y()[0])
	}
}

// TestHDRP010 exercises the semi-planar input path.
func TestHDRP010(t *testing.T) {
	const w, h = 256, 256
	f, err := NewHDRFrame(w, h, PixP010)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.V() != nil {
		t.Error("P010 should have no separate V plane")
	}

	s, err := NewHDRScaler(HDRConfig{
		SrcWidth: w, SrcHeight: h, Format: PixP010, Transfer: TRCPQ,
		Flags: Scale2X, HDRFlags: Scale2X,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Run(f)
	if hdr, ok := s.HDROutput(Scale2X); !ok || hdr.Width != w/2 {
		t.Fatalf("P010 2x HDR output wrong: ok=%v", ok)
	}
}

// TestCustomLUTLength rejects a wrong-sized LUT before it reaches C.
func TestCustomLUTLength(t *testing.T) {
	_, err := NewHDRScaler(HDRConfig{
		SrcWidth: 256, SrcHeight: 256, Format: PixI010, Transfer: TRCPQ,
		Flags: Scale2X, SDRFlags: Scale2X,
		Tonemap: TonemapConfig{Curve: TonemapCustom, CustomLUT: make([]byte, 100)},
	})
	if !errors.Is(err, ErrCustomLUTLength) {
		t.Fatalf("err = %v, want ErrCustomLUTLength", err)
	}
}

// TestFrameMismatchPanics confirms a wrong-sized frame is rejected, not read OOB.
func TestFrameMismatchPanics(t *testing.T) {
	s, _ := NewScaler(ScalerConfig{SrcWidth: 256, SrcHeight: 256, Flags: Scale2X})
	defer s.Close()
	f, _ := NewFrame(128, 128)
	defer f.Close()
	defer func() {
		if recover() == nil {
			t.Error("expected a panic on frame/scaler size mismatch")
		}
	}()
	s.Run(f)
}

// TestUseAfterClose confirms a closed scaler panics cleanly instead of passing
// a nil context into C and segfaulting the process.
func TestUseAfterClose(t *testing.T) {
	s, _ := NewScaler(ScalerConfig{SrcWidth: 256, SrcHeight: 256, Flags: Scale2X})
	f, _ := NewFrame(256, 256)
	defer f.Close()
	s.Close()
	defer func() {
		if recover() == nil {
			t.Error("expected a panic when running a closed Scaler")
		}
	}()
	s.Run(f)
}

// TestRunClosedFrame confirms a closed frame is rejected, not passed as null.
func TestRunClosedFrame(t *testing.T) {
	s, _ := NewScaler(ScalerConfig{SrcWidth: 256, SrcHeight: 256, Flags: Scale2X})
	defer s.Close()
	f, _ := NewFrame(256, 256)
	f.Close()
	defer func() {
		if recover() == nil {
			t.Error("expected a panic when running with a closed Frame")
		}
	}()
	s.Run(f)
}

// TestInvalidFlags confirms a hard error surfaces as a typed Error.
func TestInvalidFlags(t *testing.T) {
	// Mixing the thirds (3x) and pow2 (2x) families is invalid.
	_, err := NewScaler(ScalerConfig{SrcWidth: 256, SrcHeight: 256, Flags: Scale2X | Scale3X})
	if err == nil {
		t.Fatal("expected an error for mixed-family flags, got nil")
	}
	if !errors.Is(err, ErrInvalidFlags) {
		t.Errorf("error = %v, want ErrInvalidFlags", err)
	}
}
