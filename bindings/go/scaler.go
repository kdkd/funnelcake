// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

package funnelcake

/*
#include <stdlib.h>
#include "funnelcake.h"
#include "funnelcake_helpers.h"
*/
import "C"

import (
	"fmt"
	"math/bits"
	"runtime"
	"unsafe"
)

// ScalerConfig describes an 8-bit downscale/upscale job. Strides for the source
// are derived from SrcWidth, so a Frame built with NewFrame(SrcWidth, SrcHeight)
// matches automatically.
type ScalerConfig struct {
	SrcWidth      int
	SrcHeight     int
	Flags         ScaleFlag   // downscale outputs (one family only)
	UpscaleFlags  UpscaleFlag // contiguous-prefix upscale cascade (optional)
	UpscaleTail15 bool        // append a 1.5x tail to the upscale cascade
	Options       Option
}

// Scaler is an initialized 8-bit scaling context. It owns its output buffers;
// they are valid until the next Run or until Close.
type Scaler struct {
	ctx       *C.fused_scaler_ctx_t // C memory
	Warnings  Warnings              // non-fatal conditions reported by init
	srcWidth  int
	srcHeight int
}

// NewScaler validates the configuration, allocates output buffers, and returns
// a ready Scaler. A negative library return becomes an Error; non-fatal
// conditions are reported via the Warnings field (also check SIMDAvailable).
func NewScaler(cfg ScalerConfig) (*Scaler, error) {
	if !validDimensions(cfg.SrcWidth, cfg.SrcHeight) {
		return nil, ErrBadDimensions
	}
	ensureDetect()

	ctx := (*C.fused_scaler_ctx_t)(C.calloc(1, C.size_t(unsafe.Sizeof(C.fused_scaler_ctx_t{}))))
	if ctx == nil {
		return nil, errAlloc
	}

	var ys, uvs C.int
	C.fused_plane_strides(C.int(cfg.SrcWidth), &ys, &uvs)

	ctx.src_width = C.int(cfg.SrcWidth)
	ctx.src_height = C.int(cfg.SrcHeight)
	ctx.src_y_stride = ys
	ctx.src_uv_stride = uvs
	ctx.requested_flags = C.uint32_t(cfg.Flags)
	ctx.options = C.uint32_t(cfg.Options)
	ctx.upscale_flags = C.uint32_t(cfg.UpscaleFlags)
	ctx.upscale_tail_1_5x = b2i(cfg.UpscaleTail15)

	rc := C.fused_scaler_init(ctx)
	if rc < 0 {
		C.free(unsafe.Pointer(ctx))
		return nil, Error(int(rc))
	}

	s := &Scaler{
		ctx:       ctx,
		Warnings:  Warnings(uint32(rc)),
		srcWidth:  cfg.SrcWidth,
		srcHeight: cfg.SrcHeight,
	}
	runtime.SetFinalizer(s, (*Scaler).Close)
	return s, nil
}

// Run scales one frame, filling all achieved outputs. The frame must come from
// NewFrame with the same width/height passed to NewScaler; a mismatch panics
// rather than reading out of bounds in the native kernels.
func (s *Scaler) Run(f *Frame) {
	if s.ctx == nil {
		panic("funnelcake: Run on a closed Scaler")
	}
	if f.y == nil {
		panic("funnelcake: Run with a closed Frame")
	}
	if f.width != s.srcWidth || f.height != s.srcHeight {
		panic(fmt.Sprintf("funnelcake: frame %dx%d does not match scaler source %dx%d",
			f.width, f.height, s.srcWidth, s.srcHeight))
	}
	C.fused_scaler_run(s.ctx,
		(*C.uint8_t)(f.y), (*C.uint8_t)(f.u), (*C.uint8_t)(f.v))
	runtime.KeepAlive(f)
	runtime.KeepAlive(s)
}

// checkOpen panics with a clear message if the scaler has been closed, instead
// of letting a nil context reach C and segfault the process.
func (s *Scaler) checkOpen() {
	if s.ctx == nil {
		panic("funnelcake: use of a closed Scaler")
	}
}

// EffectiveWidth is the source luma width actually used (after any crop).
func (s *Scaler) EffectiveWidth() int { s.checkOpen(); return int(s.ctx.effective_width) }

// EffectiveHeight is the source luma height actually used (after any crop).
func (s *Scaler) EffectiveHeight() int { s.checkOpen(); return int(s.ctx.effective_height) }

// AchievedFlags reports which downscale steps were produced.
func (s *Scaler) AchievedFlags() ScaleFlag { s.checkOpen(); return ScaleFlag(s.ctx.achieved_flags) }

// Output returns the downscale output for a single flag, and whether it was
// produced.
func (s *Scaler) Output(flag ScaleFlag) (Output, bool) {
	s.checkOpen()
	if flag == 0 || flag&(flag-1) != 0 || uint32(s.ctx.achieved_flags)&uint32(flag) == 0 {
		return Output{}, false
	}
	idx := bits.TrailingZeros32(uint32(flag))
	o := cToOutput(&s.ctx.outputs[idx])
	o.keepalive = s
	return o, true
}

// UpscaleOutput returns an upscale-cascade output for a single flag.
func (s *Scaler) UpscaleOutput(flag UpscaleFlag) (Output, bool) {
	s.checkOpen()
	if flag == 0 || flag&(flag-1) != 0 || uint32(s.ctx.achieved_upscale_flags)&uint32(flag) == 0 {
		return Output{}, false
	}
	idx := bits.TrailingZeros32(uint32(flag))
	o := cToOutput(&s.ctx.upscale_outputs[idx])
	o.keepalive = s
	return o, true
}

// UpscaleTail returns the 1.5x upscale tail output, if it was produced.
func (s *Scaler) UpscaleTail() (Output, bool) {
	s.checkOpen()
	if s.ctx.achieved_upscale_tail == 0 {
		return Output{}, false
	}
	o := cToOutput(&s.ctx.upscale_outputs[C.FUSED_UP_IDX_TAIL])
	o.keepalive = s
	return o, true
}

// Close releases all buffers allocated by init. Safe to call more than once.
func (s *Scaler) Close() {
	if s.ctx != nil {
		C.fused_scaler_free(s.ctx)
		C.free(unsafe.Pointer(s.ctx))
		s.ctx = nil
	}
	runtime.SetFinalizer(s, nil)
}

// Output describes a native output. Y, U and V return owned copies; obtain
// those copies before closing the scaler. Public geometry is a snapshot and
// never controls native access bounds.
type Output struct {
	Width    int
	Height   int
	YStride  int
	UVStride int
	Fallback bool // true if the scalar kernel was used for this step

	y, u, v       unsafe.Pointer
	chromaH       int
	ySize, uvSize int
	// keepalive pins the producing scaler so that, as long as this Output is
	// reachable, the scaler's finalizer cannot run and free the buffers these
	// views alias. Copy data out if you need it past the scaler's lifetime.
	keepalive interface{}
}

// Y copies the luma plane, including row padding.
func (o Output) Y() []byte { return o.copyPlane(o.y, o.ySize) }

// U copies the Cb plane, including row padding.
func (o Output) U() []byte { return o.copyPlane(o.u, o.uvSize) }

// V copies the Cr plane, including row padding.
func (o Output) V() []byte { return o.copyPlane(o.v, o.uvSize) }

func cToOutput(co *C.fused_scale_output_t) Output {
	h := int(co.height)
	return Output{
		Width:    int(co.width),
		Height:   h,
		YStride:  int(co.y_stride),
		UVStride: int(co.uv_stride),
		Fallback: co.fallback != 0,
		y:        unsafe.Pointer(co.plane_y),
		u:        unsafe.Pointer(co.plane_u),
		v:        unsafe.Pointer(co.plane_v),
		chromaH:  chroma420Height(h),
		ySize:    int(co.y_stride) * h,
		uvSize:   int(co.uv_stride) * chroma420Height(h),
	}
}

func b2i(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// Plane accessors return owned copies because Go slices cannot retain a
// foreign allocation's owner or detect explicit Close after returning.
func (o Output) copyPlane(p unsafe.Pointer, n int) []byte {
	if o.keepalive == nil {
		return nil
	}
	switch owner := o.keepalive.(type) {
	case *Scaler:
		owner.checkOpen()
	case *HDRScaler:
		owner.checkOpen()
	}
	result := append([]byte(nil), byteView(p, n)...)
	runtime.KeepAlive(o.keepalive)
	return result
}
