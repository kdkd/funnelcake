// Copyright (c) 2020-2026 Kevin Day
//
// SPDX-License-Identifier: BSD-2-Clause-Patent
// See LICENSE.md in the project root for full license text.

package funnelcake

import "errors"

// ErrCustomLUTLength is returned by NewHDRScaler when a custom tone-map LUT is
// supplied with a length other than 1024.
var ErrCustomLUTLength = errors.New("funnelcake: custom tone-map LUT must be exactly 1024 bytes")

// Error is a hard error returned by init. It wraps one of the negative
// FUSED_ERR_* return codes. Compare with errors.Is against the sentinels below.
type Error int

const (
	ErrInvalidFlags  Error = -1 // flags from both families, or unknown bits
	ErrNoSteps       Error = -2 // no valid step flags after filtering
	ErrBadDimensions Error = -3 // src_width/height <= 0 or too small
	ErrBadAlignment  Error = -4 // strides not 32-byte aligned
)

func (e Error) Error() string {
	switch e {
	case ErrInvalidFlags:
		return "funnelcake: invalid flags (mixed families or unknown bits)"
	case ErrNoSteps:
		return "funnelcake: no valid scale steps requested"
	case ErrBadDimensions:
		return "funnelcake: bad source dimensions"
	case ErrBadAlignment:
		return "funnelcake: source strides are not 32-byte aligned"
	default:
		return "funnelcake: unknown error"
	}
}

// Warnings is the composable bitmask of non-fatal conditions reported by a
// successful init. A zero value means a perfect (FUSED_OK) result.
type Warnings uint32

const (
	warnScalar  Warnings = 1 << 0
	warnPartial Warnings = 1 << 1
	warnCropped Warnings = 1 << 2
)

// Scalar reports whether at least one step fell back to the scalar kernel.
func (w Warnings) Scalar() bool { return w&warnScalar != 0 }

// Partial reports whether at least one requested step was rejected.
func (w Warnings) Partial() bool { return w&warnPartial != 0 }

// Cropped reports whether the source was cropped to satisfy dimension constraints.
func (w Warnings) Cropped() bool { return w&warnCropped != 0 }

// Perfect reports whether init produced every requested output with SIMD and
// no cropping.
func (w Warnings) Perfect() bool { return w == 0 }
