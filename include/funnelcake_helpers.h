/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * funnelcake_helpers.h - small, additive helper surface for language bindings.
 *
 * These functions exist purely to make foreign-language bindings (Go, Rust,
 * Python, Java) correct and easy to write. They live in their own translation
 * unit (src/bindings_support.c) and ship inside libfunnelcake, but the core
 * scaling/HDR paths never call them, so they do not affect any library output.
 * Raw C users who only need the scaler/HDR API can ignore this header; when
 * static-linking they pay nothing, as the unused objects are dropped at link.
 *
 * The two things every binding needs and would otherwise reimplement:
 *   1. allocation that matches the 32-byte alignment the SIMD kernels expect,
 *   2. the stride round-up used for source planes.
 */

#ifndef FUNNELCAKE_HELPERS_H
#define FUNNELCAKE_HELPERS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * fused_aligned_alloc - allocate `size` bytes aligned to `alignment` bytes.
 *
 * Returns NULL on failure (bad alignment or out of memory). The returned
 * pointer must be released with fused_free, never the host language's own
 * free. `alignment` must be a power of two and a multiple of sizeof(void*);
 * pass 32 for funnelcake source/output planes.
 */
void *fused_aligned_alloc(size_t alignment, size_t size);

/*
 * fused_free - release a pointer returned by fused_aligned_alloc.
 * NULL is a safe no-op.
 */
void fused_free(void *p);

/*
 * fused_plane_strides - compute 32-byte-aligned strides for an 8-bit I420
 * source of the given luma width. Writes the Y-plane stride and the U/V-plane
 * stride (bytes per row). Either out pointer may be NULL to skip it.
 */
void fused_plane_strides(int width, int *y_stride, int *uv_stride);

/*
 * fused_plane_strides_16 - compute 32-byte-aligned byte strides for a 10-bit
 * planar (I010/I210) source of the given luma width. `y_stride` holds
 * width*2 rounded up; `uv_stride` holds the per-plane chroma stride
 * (width/2 samples * 2 bytes) rounded up.
 *
 * For semi-planar interleaved chroma (P010/P210), the interleaved UV row is
 * the same width in bytes as the luma row, so use `y_stride` for it.
 */
void fused_plane_strides_16(int width, int *y_stride, int *uv_stride);

/*
 * Context struct sizes. Bindings that mirror the context layout by hand (e.g.
 * the Java FFM binding, which has no struct generator) call these at startup
 * and assert their mirrored layout's size matches, turning any ABI/layout
 * drift into a loud, immediate failure instead of silent memory corruption.
 */
size_t fused_scaler_ctx_sizeof(void);
size_t fused_hdr_ctx_sizeof(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FUNNELCAKE_HELPERS_H */
