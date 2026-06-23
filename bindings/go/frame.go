// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

package funnelcake

/*
#include "funnelcake.h"
#include "funnelcake_helpers.h"
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// errAlloc is returned when an aligned plane allocation fails (out of memory).
var errAlloc = errors.New("funnelcake: aligned allocation failed")

// chroma420Height returns the chroma plane height for a 4:2:0 source/output of
// the given luma height.
func chroma420Height(h int) int { return (h + 1) / 2 }

// Frame is an 8-bit I420 (YUV 4:2:0 planar) input frame. Its three planes are
// allocated with 32-byte alignment and 32-byte-aligned strides, exactly what
// the SIMD kernels expect, so a Frame is always safe to hand to Scaler.Run.
//
// Fill the planes via the Y, U and V byte-slice views. Call Close when done;
// a finalizer frees the planes as a backstop if you forget.
type Frame struct {
	Width    int
	Height   int
	YStride  int // bytes per luma row
	UVStride int // bytes per chroma row

	y, u, v unsafe.Pointer // 32-byte-aligned C memory
	chromaH int
}

// NewFrame allocates an aligned I420 frame. width and height should be even
// (4:2:0 chroma is half-resolution in each axis).
func NewFrame(width, height int) (*Frame, error) {
	if width <= 0 || height <= 0 {
		return nil, ErrBadDimensions
	}
	var ys, uvs C.int
	C.fused_plane_strides(C.int(width), &ys, &uvs)

	f := &Frame{
		Width:    width,
		Height:   height,
		YStride:  int(ys),
		UVStride: int(uvs),
		chromaH:  chroma420Height(height),
	}
	f.y = C.fused_aligned_alloc(32, C.size_t(f.YStride*height))
	f.u = C.fused_aligned_alloc(32, C.size_t(f.UVStride*f.chromaH))
	f.v = C.fused_aligned_alloc(32, C.size_t(f.UVStride*f.chromaH))
	if f.y == nil || f.u == nil || f.v == nil {
		f.Close()
		return nil, errAlloc
	}
	runtime.SetFinalizer(f, (*Frame).Close)
	return f, nil
}

// Y returns a writable view of the luma plane (len = YStride*Height).
func (f *Frame) Y() []byte { return byteView(f.y, f.YStride*f.Height) }

// U returns a writable view of the Cb plane (len = UVStride*chromaHeight).
func (f *Frame) U() []byte { return byteView(f.u, f.UVStride*f.chromaH) }

// V returns a writable view of the Cr plane (len = UVStride*chromaHeight).
func (f *Frame) V() []byte { return byteView(f.v, f.UVStride*f.chromaH) }

// Close frees the frame's planes. Safe to call more than once.
func (f *Frame) Close() {
	if f.y != nil {
		C.fused_free(f.y)
		f.y = nil
	}
	if f.u != nil {
		C.fused_free(f.u)
		f.u = nil
	}
	if f.v != nil {
		C.fused_free(f.v)
		f.v = nil
	}
	runtime.SetFinalizer(f, nil)
}

// HDRFrame is a 10-bit input frame. Layout depends on Format: I010/I210 use
// three planar uint16 planes; P010/P210 use a luma plane plus a single
// interleaved-UV plane (V is unused). All planes are 32-byte aligned.
type HDRFrame struct {
	Width    int
	Height   int
	Format   PixelFormat
	YStride  int // bytes per luma row
	UVStride int // bytes per chroma row (interleaved row for P010/P210)

	y, u, v unsafe.Pointer
	chromaH int
}

// is422 reports whether the format is a 4:2:2 layout (chroma at full height).
func (pf PixelFormat) is422() bool { return pf == PixI210 || pf == PixP210 }

// isSemiPlanar reports whether chroma is interleaved into a single UV plane.
func (pf PixelFormat) isSemiPlanar() bool { return pf == PixP010 || pf == PixP210 }

// NewHDRFrame allocates an aligned 10-bit frame for the given format.
func NewHDRFrame(width, height int, format PixelFormat) (*HDRFrame, error) {
	if width <= 0 || height <= 0 {
		return nil, ErrBadDimensions
	}
	if format < PixI010 || format > PixP210 {
		return nil, ErrInvalidFlags
	}
	var ys, uvs C.int
	C.fused_plane_strides_16(C.int(width), &ys, &uvs)

	chromaH := chroma420Height(height)
	if format.is422() {
		chromaH = height
	}

	f := &HDRFrame{
		Width:   width,
		Height:  height,
		Format:  format,
		YStride: int(ys),
		chromaH: chromaH,
	}
	f.y = C.fused_aligned_alloc(32, C.size_t(f.YStride*height))
	if format.isSemiPlanar() {
		// Interleaved UV row is the same byte width as a luma row.
		f.UVStride = int(ys)
		f.u = C.fused_aligned_alloc(32, C.size_t(f.UVStride*chromaH))
	} else {
		f.UVStride = int(uvs)
		f.u = C.fused_aligned_alloc(32, C.size_t(f.UVStride*chromaH))
		f.v = C.fused_aligned_alloc(32, C.size_t(f.UVStride*chromaH))
	}
	if f.y == nil || f.u == nil || (!format.isSemiPlanar() && f.v == nil) {
		f.Close()
		return nil, errAlloc
	}
	runtime.SetFinalizer(f, (*HDRFrame).Close)
	return f, nil
}

// Y returns a writable view of the luma plane as uint16 samples.
func (f *HDRFrame) Y() []uint16 { return u16View(f.y, (f.YStride/2)*f.Height) }

// U returns the Cb plane (I010/I210) or the interleaved UV plane (P010/P210).
func (f *HDRFrame) U() []uint16 { return u16View(f.u, (f.UVStride/2)*f.chromaH) }

// V returns the Cr plane for planar formats, or nil for P010/P210.
func (f *HDRFrame) V() []uint16 { return u16View(f.v, (f.UVStride/2)*f.chromaH) }

// Close frees the frame's planes. Safe to call more than once.
func (f *HDRFrame) Close() {
	if f.y != nil {
		C.fused_free(f.y)
		f.y = nil
	}
	if f.u != nil {
		C.fused_free(f.u)
		f.u = nil
	}
	if f.v != nil {
		C.fused_free(f.v)
		f.v = nil
	}
	runtime.SetFinalizer(f, nil)
}

// byteView builds a []byte aliasing C memory, or nil if the pointer is nil.
func byteView(p unsafe.Pointer, n int) []byte {
	if p == nil || n <= 0 {
		return nil
	}
	return unsafe.Slice((*byte)(p), n)
}

// u16View builds a []uint16 aliasing C memory, or nil if the pointer is nil.
func u16View(p unsafe.Pointer, n int) []uint16 {
	if p == nil || n <= 0 {
		return nil
	}
	return unsafe.Slice((*uint16)(p), n)
}
