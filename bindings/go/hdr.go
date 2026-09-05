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

// TonemapConfig controls how 10-bit HDR is mapped to 8-bit SDR outputs. The
// zero value selects the Hable curve, a 1000-nit peak, a 100-nit target, and
// limited (video) range in and out.
type TonemapConfig struct {
	Curve      TonemapCurve
	PeakNits   int
	TargetNits int
	SrcRange   Range
	DstRange   Range
	// CustomLUT, when Curve is TonemapCustom, is a 1024-entry uint8 Y LUT. It
	// is copied into the scaler at init, so the caller need not retain it.
	CustomLUT []byte
}

// HDRConfig describes a 10-bit scaling job with optional tone mapping. HDRFlags
// and SDRFlags must be subsets of Flags. Source strides are derived from
// SrcWidth and Format, matching a frame built with NewHDRFrame.
type HDRConfig struct {
	SrcWidth  int
	SrcHeight int
	Format    PixelFormat
	Transfer  TransferFunc

	Flags    ScaleFlag // requested downscale steps (one family only)
	HDRFlags ScaleFlag // subset: produce 10-bit HDR outputs
	SDRFlags ScaleFlag // subset: produce 8-bit tone-mapped outputs
	Options  Option

	Tonemap1x bool // also produce a 1:1 tone-mapped SDR copy
	Tonemap   TonemapConfig

	UpscaleFlags     UpscaleFlag
	UpscaleTail15    bool
	UpscaleSDRFlags  UpscaleFlag // subset of UpscaleFlags: tone-mapped SDR copies
	UpscaleSDRTail15 bool
}

// HDRScaler is an initialized 10-bit scaling/tone-mapping context. It owns its
// output buffers; they are valid until the next Run or until Close.
type HDRScaler struct {
	ctx       *C.fused_hdr_ctx_t
	customLUT unsafe.Pointer // C copy of TonemapConfig.CustomLUT, if any
	Warnings  Warnings
	srcWidth  int
	srcHeight int
	format    PixelFormat
}

// NewHDRScaler validates the configuration, builds tone-mapping LUTs, allocates
// outputs, and returns a ready HDRScaler.
func NewHDRScaler(cfg HDRConfig) (*HDRScaler, error) {
	if !validDimensions(cfg.SrcWidth, cfg.SrcHeight) {
		return nil, ErrBadDimensions
	}
	for _, value := range []int{int(cfg.Format), int(cfg.Transfer), int(cfg.Tonemap.Curve), cfg.Tonemap.PeakNits, cfg.Tonemap.TargetNits, int(cfg.Tonemap.SrcRange), int(cfg.Tonemap.DstRange)} {
		if int64(value) < -2147483648 || int64(value) > 2147483647 {
			return nil, ErrInvalidFlags
		}
	}
	ensureDetect()

	if n := len(cfg.Tonemap.CustomLUT); n != 0 && n != 1024 {
		return nil, ErrCustomLUTLength
	}

	ctx := (*C.fused_hdr_ctx_t)(C.calloc(1, C.size_t(unsafe.Sizeof(C.fused_hdr_ctx_t{}))))
	if ctx == nil {
		return nil, errAlloc
	}

	var ys, uvs C.int
	C.fused_plane_strides_16(C.int(cfg.SrcWidth), &ys, &uvs)

	ctx.src_width = C.int(cfg.SrcWidth)
	ctx.src_height = C.int(cfg.SrcHeight)
	ctx.src_format = C.int(cfg.Format)
	ctx.src_transfer = C.int(cfg.Transfer)
	ctx.src_y_stride = ys
	if cfg.Format.isSemiPlanar() {
		ctx.src_uv_stride = ys // interleaved UV row matches luma byte width
	} else {
		ctx.src_uv_stride = uvs
	}

	ctx.requested_flags = C.uint32_t(cfg.Flags)
	ctx.hdr_flags = C.uint32_t(cfg.HDRFlags)
	ctx.sdr_flags = C.uint32_t(cfg.SDRFlags)
	ctx.options = C.uint32_t(cfg.Options)
	ctx.tonemap_1x = b2i(cfg.Tonemap1x)

	ctx.tonemap.curve = C.int(cfg.Tonemap.Curve)
	ctx.tonemap.peak_nits = C.int(cfg.Tonemap.PeakNits)
	ctx.tonemap.target_nits = C.int(cfg.Tonemap.TargetNits)
	ctx.tonemap.src_range = C.int(cfg.Tonemap.SrcRange)
	ctx.tonemap.dst_range = C.int(cfg.Tonemap.DstRange)

	var customLUT unsafe.Pointer
	if len(cfg.Tonemap.CustomLUT) > 0 {
		// Copy into C memory; the field must not hold a Go pointer.
		customLUT = C.CBytes(cfg.Tonemap.CustomLUT)
		ctx.tonemap.custom_lut = (*C.uint8_t)(customLUT)
	}

	ctx.upscale_flags = C.uint32_t(cfg.UpscaleFlags)
	ctx.upscale_tail_1_5x = b2i(cfg.UpscaleTail15)
	ctx.upscale_sdr_flags = C.uint32_t(cfg.UpscaleSDRFlags)
	ctx.upscale_sdr_tail_1_5x = b2i(cfg.UpscaleSDRTail15)

	rc := C.fused_hdr_init(ctx)
	if rc < 0 {
		if customLUT != nil {
			C.free(customLUT)
		}
		C.free(unsafe.Pointer(ctx))
		return nil, Error(int(rc))
	}

	h := &HDRScaler{
		ctx:       ctx,
		customLUT: customLUT,
		Warnings:  Warnings(uint32(rc)),
		srcWidth:  cfg.SrcWidth,
		srcHeight: cfg.SrcHeight,
		format:    cfg.Format,
	}
	runtime.SetFinalizer(h, (*HDRScaler).Close)
	return h, nil
}

// Run scales and tone-maps one 10-bit frame. The frame must come from
// NewHDRFrame with the same width/height/format passed to NewHDRScaler; a
// mismatch panics rather than reading out of bounds in the native kernels.
func (h *HDRScaler) Run(f *HDRFrame) {
	if h.ctx == nil {
		panic("funnelcake: Run on a closed HDRScaler")
	}
	if f.y == nil {
		panic("funnelcake: Run with a closed HDRFrame")
	}
	if f.width != h.srcWidth || f.height != h.srcHeight || f.format != h.format {
		panic(fmt.Sprintf("funnelcake: frame %dx%d fmt=%d does not match scaler %dx%d fmt=%d",
			f.width, f.height, f.format, h.srcWidth, h.srcHeight, h.format))
	}
	C.fused_hdr_run(h.ctx,
		(*C.uint16_t)(f.y), (*C.uint16_t)(f.u), (*C.uint16_t)(f.v))
	runtime.KeepAlive(f)
	runtime.KeepAlive(h)
}

// checkOpen panics with a clear message if the scaler has been closed.
func (h *HDRScaler) checkOpen() {
	if h.ctx == nil {
		panic("funnelcake: use of a closed HDRScaler")
	}
}

// EffectiveWidth is the source luma width actually used (after any crop).
func (h *HDRScaler) EffectiveWidth() int { h.checkOpen(); return int(h.ctx.effective_width) }

// EffectiveHeight is the source luma height actually used (after any crop).
func (h *HDRScaler) EffectiveHeight() int { h.checkOpen(); return int(h.ctx.effective_height) }

// HDROutput returns the 10-bit HDR output for a downscale flag.
func (h *HDRScaler) HDROutput(flag ScaleFlag) (HDROutput, bool) {
	h.checkOpen()
	if flag == 0 || flag&(flag-1) != 0 || uint32(h.ctx.achieved_hdr_flags)&uint32(flag) == 0 {
		return HDROutput{}, false
	}
	idx := bits.TrailingZeros32(uint32(flag))
	o := cToHDROutput(&h.ctx.hdr_outputs[idx])
	o.keepalive = h
	return o, true
}

// SDROutput returns the 8-bit tone-mapped output for a downscale flag.
func (h *HDRScaler) SDROutput(flag ScaleFlag) (Output, bool) {
	h.checkOpen()
	if flag == 0 || flag&(flag-1) != 0 || uint32(h.ctx.achieved_sdr_flags)&uint32(flag) == 0 {
		return Output{}, false
	}
	idx := bits.TrailingZeros32(uint32(flag))
	o := cToOutput(&h.ctx.sdr_outputs[idx])
	o.keepalive = h
	return o, true
}

// Tonemap1xOutput returns the 1:1 tone-mapped SDR copy, if it was produced.
func (h *HDRScaler) Tonemap1xOutput() (Output, bool) {
	h.checkOpen()
	if h.ctx.output_1x.plane_y == nil {
		return Output{}, false
	}
	o := cToOutput(&h.ctx.output_1x)
	o.keepalive = h
	return o, true
}

// UpscaleHDROutput returns a 10-bit upscale-cascade output for a flag.
func (h *HDRScaler) UpscaleHDROutput(flag UpscaleFlag) (HDROutput, bool) {
	h.checkOpen()
	if flag == 0 || flag&(flag-1) != 0 || uint32(h.ctx.achieved_upscale_flags)&uint32(flag) == 0 {
		return HDROutput{}, false
	}
	idx := bits.TrailingZeros32(uint32(flag))
	o := cToHDROutput(&h.ctx.upscale_hdr_outputs[idx])
	o.keepalive = h
	return o, true
}

// UpscaleSDROutput returns a tone-mapped 8-bit upscale-cascade output for a flag.
func (h *HDRScaler) UpscaleSDROutput(flag UpscaleFlag) (Output, bool) {
	h.checkOpen()
	if flag == 0 || flag&(flag-1) != 0 || uint32(h.ctx.achieved_upscale_sdr_flags)&uint32(flag) == 0 {
		return Output{}, false
	}
	idx := bits.TrailingZeros32(uint32(flag))
	o := cToOutput(&h.ctx.upscale_sdr_outputs[idx])
	o.keepalive = h
	return o, true
}

// Close releases all buffers allocated by init. Safe to call more than once.
func (h *HDRScaler) Close() {
	if h.ctx != nil {
		C.fused_hdr_free(h.ctx)
		C.free(unsafe.Pointer(h.ctx))
		h.ctx = nil
	}
	if h.customLUT != nil {
		C.free(h.customLUT)
		h.customLUT = nil
	}
	runtime.SetFinalizer(h, nil)
}

// HDROutput describes a native 10-bit output. Plane accessors return owned
// copies, obtained before closing the scaler. Strides are in bytes; public
// geometry is a snapshot and never controls native access bounds.
type HDROutput struct {
	Width    int
	Height   int
	YStride  int
	UVStride int
	Fallback bool

	y, u, v                                      unsafe.Pointer
	chromaH                                      int
	ySize, uvSize                                int
	rowWidth, rowHeight, rowYStride, rowUVStride int
	keepalive                                    interface{} // pins the producing HDRScaler; see Output.keepalive
}

// Y returns the luma plane copy as uint16 samples.
func (o HDROutput) Y() []uint16 { return o.copyPlane(o.y, o.ySize/2) }

// U returns the Cb plane copy as uint16 samples.
func (o HDROutput) U() []uint16 { return o.copyPlane(o.u, o.uvSize/2) }

// V returns the Cr plane copy as uint16 samples.
func (o HDROutput) V() []uint16 { return o.copyPlane(o.v, o.uvSize/2) }

func cToHDROutput(co *C.fused_hdr_output_t) HDROutput {
	h := int(co.height)
	return HDROutput{
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
		rowWidth: int(co.width), rowHeight: h, rowYStride: int(co.y_stride), rowUVStride: int(co.uv_stride),
		uvSize: int(co.uv_stride) * chroma420Height(h),
	}
}

// Plane accessors return owned copies because Go slices cannot retain a
// foreign allocation's owner or detect explicit Close after returning.
func (o HDROutput) copyPlane(p unsafe.Pointer, n int) []uint16 {
	if o.keepalive == nil {
		return nil
	}
	switch owner := o.keepalive.(type) {
	case *Scaler:
		owner.checkOpen()
	case *HDRScaler:
		owner.checkOpen()
	}
	result := append([]uint16(nil), u16View(p, n)...)
	runtime.KeepAlive(o.keepalive)
	return result
}

// YRow copies only the active samples of a row.
func (o HDROutput) YRow(row int) []uint16 {
	if row < 0 || row >= o.rowHeight {
		panic("funnelcake: row out of range")
	}
	return o.copyPlane(unsafe.Add(o.y, row*o.rowYStride), o.rowWidth)
}

// URow copies only the active samples of a row.
func (o HDROutput) URow(row int) []uint16 {
	if row < 0 || row >= (o.rowHeight+1)/2 {
		panic("funnelcake: row out of range")
	}
	return o.copyPlane(unsafe.Add(o.u, row*o.rowUVStride), o.rowWidth/2)
}

// VRow copies only the active samples of a row.
func (o HDROutput) VRow(row int) []uint16 {
	if row < 0 || row >= (o.rowHeight+1)/2 {
		panic("funnelcake: row out of range")
	}
	return o.copyPlane(unsafe.Add(o.v, row*o.rowUVStride), o.rowWidth/2)
}

// Packed copies all active rows, omitting stride padding.
func (o HDROutput) Packed() (y, u, v []uint16) {
	for r := 0; r < o.rowHeight; r++ {
		y = append(y, o.YRow(r)...)
	}
	for r := 0; r < (o.rowHeight+1)/2; r++ {
		u = append(u, o.URow(r)...)
		v = append(v, o.VRow(r)...)
	}
	return
}
