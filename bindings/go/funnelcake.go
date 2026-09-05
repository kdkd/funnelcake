// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

// Package funnelcake provides idiomatic Go bindings for the funnelcake SIMD
// YUV scaler and HDR tone-mapper.
//
// The bindings are thin and safe: they wrap the C library's create/run/free
// lifecycle, translate its return codes into Go errors and warnings, and give
// callers Frame / HDRFrame types whose buffers carry the 32-byte alignment the
// SIMD kernels require — correct by construction, so you never compute a stride
// or call aligned_alloc yourself.
//
// Output plane accessors return owned Go slices. Obtain them before closing
// the scaler; the returned data survives later Run and Close calls.
package funnelcake

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: ${SRCDIR}/../../libfunnelcake.a -lm

#include <stdlib.h>
#include "funnelcake.h"
#include "funnelcake_helpers.h"
*/
import "C"

import "sync"

// ScaleFlag selects a downscale output. A single Scaler must use flags from
// only one family (thirds or power-of-two).
type ScaleFlag uint32

const (
	Scale1_5X ScaleFlag = C.FUSED_SCALE_1_5X // 3:2 reduction  (thirds family)
	Scale2X   ScaleFlag = C.FUSED_SCALE_2X   // 2:1 reduction  (pow2 family)
	Scale3X   ScaleFlag = C.FUSED_SCALE_3X   // 3:1 reduction  (thirds family)
	Scale4X   ScaleFlag = C.FUSED_SCALE_4X   // 4:1 reduction  (pow2 family)
	Scale6X   ScaleFlag = C.FUSED_SCALE_6X   // 6:1 reduction  (thirds family)
	Scale8X   ScaleFlag = C.FUSED_SCALE_8X   // 8:1 reduction  (pow2 family)
	Scale12X  ScaleFlag = C.FUSED_SCALE_12X  // 12:1 reduction (thirds family)
	Scale16X  ScaleFlag = C.FUSED_SCALE_16X  // 16:1 reduction (pow2 family)

	ScaleThirdsMask ScaleFlag = C.FUSED_SCALE_THIRDS_MASK
	ScalePow2Mask   ScaleFlag = C.FUSED_SCALE_POW2_MASK
)

// UpscaleFlag selects an upscale level. The set requested must be a contiguous
// prefix of the 2x cascade (e.g. {2x}, {2x,4x}, {2x,4x,8x}).
type UpscaleFlag uint32

const (
	Upscale2X  UpscaleFlag = C.FUSED_UPSCALE_2X
	Upscale4X  UpscaleFlag = C.FUSED_UPSCALE_4X
	Upscale8X  UpscaleFlag = C.FUSED_UPSCALE_8X
	Upscale16X UpscaleFlag = C.FUSED_UPSCALE_16X
	Upscale32X UpscaleFlag = C.FUSED_UPSCALE_32X
)

// Option flags tune init behavior.
type Option uint32

const (
	// OptNoCrop rejects steps that would require cropping the source, rather
	// than silently cropping and setting the cropped warning bit.
	OptNoCrop Option = C.FUSED_OPT_NO_CROP
	// OptNoFallback rejects steps that cannot use the SIMD kernel, rather than
	// falling back to scalar and setting the scalar warning bit.
	OptNoFallback Option = C.FUSED_OPT_NO_FALLBACK
)

// PixelFormat is the layout of a 10-bit HDR source.
type PixelFormat int

const (
	PixI010 PixelFormat = C.FUSED_PIX_I010 // 4:2:0 planar: separate Y, U, V
	PixP010 PixelFormat = C.FUSED_PIX_P010 // 4:2:0 semi-planar: Y + interleaved UV
	PixI210 PixelFormat = C.FUSED_PIX_I210 // 4:2:2 planar (decimated to 4:2:0)
	PixP210 PixelFormat = C.FUSED_PIX_P210 // 4:2:2 semi-planar (decimated to 4:2:0)
)

// TransferFunc is the HDR source transfer function.
type TransferFunc int

const (
	TRCPQ  TransferFunc = C.FUSED_TRC_PQ  // SMPTE ST 2084 (HDR10)
	TRCHLG TransferFunc = C.FUSED_TRC_HLG // Hybrid Log-Gamma
)

// Range is a quantization range for tone mapping.
type Range int

const (
	RangeLimited Range = C.FUSED_RANGE_LIMITED // video range (default)
	RangeFull    Range = C.FUSED_RANGE_FULL    // full/PC range
)

// TonemapCurve selects the tone-mapping curve preset.
type TonemapCurve int

const (
	TonemapHable    TonemapCurve = C.FUSED_TONEMAP_HABLE    // filmic (good default)
	TonemapReinhard TonemapCurve = C.FUSED_TONEMAP_REINHARD // simple, lower contrast
	TonemapBT2390   TonemapCurve = C.FUSED_TONEMAP_BT2390   // broadcast reference
	TonemapCustom   TonemapCurve = C.FUSED_TONEMAP_CUSTOM   // caller-supplied 1024-entry LUT
)

// detectOnce forces the library's one-time CPU-feature probe before any user
// goroutine can race on it. The probe in detect.c uses an unsynchronized
// check-then-set; running it once here, serialized, removes the race.
var detectOnce sync.Once

func ensureDetect() {
	detectOnce.Do(func() { C.fused_simd_available() })
}

// SIMDAvailable reports whether the scalers will use vectorized kernels on this
// machine. When false, an otherwise-perfect init reports a scalar warning and
// every output's Fallback is true.
func SIMDAvailable() bool {
	ensureDetect()
	return C.fused_simd_available() == 1
}

// Version reports the linked native library version.
func Version() string { return C.GoString(C.fused_version()) }

// Backend reports the preferred native backend. Outputs may use scalar fallback.
func Backend() string { return C.GoString(C.fused_backend()) }
