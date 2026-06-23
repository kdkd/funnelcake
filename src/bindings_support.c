/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * bindings_support.c - implementation of the language-binding helper surface
 * declared in include/funnelcake_helpers.h.
 *
 * This file is part of LIB_SRCS, so the helpers ship inside libfunnelcake and a
 * single installed library + headers is enough to build any language binding.
 * Nothing here is on the scaling hot path, and the scaler/HDR code never calls
 * these functions, so including them does not change any library output or
 * behavior; raw C users who link the static archive get these objects dropped
 * as unused.
 *
 * _POSIX_C_SOURCE is set here so the file compiles correctly no matter which
 * toolchain compiles it (e.g. a binding that builds it directly without the
 * Makefile's CFLAGS).
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L  /* posix_memalign */
#endif

#include "funnelcake_helpers.h"
#include "funnelcake.h"

#include <stdlib.h>

/* Round `n` up to the next multiple of 32 (the SIMD stride/alignment unit). */
static int round_up_32(int n)
{
    return (n + 31) & ~31;
}

void *fused_aligned_alloc(size_t alignment, size_t size)
{
    void *p = NULL;
    /* posix_memalign requires size > 0 on some platforms to return a usable
     * pointer; a zero-byte request is treated as a one-byte allocation so the
     * caller always gets a non-NULL, freeable pointer on success. */
    if (size == 0) {
        size = 1;
    }
    if (posix_memalign(&p, alignment, size) != 0) {
        return NULL;
    }
    return p;
}

void fused_free(void *p)
{
    free(p);
}

void fused_plane_strides(int width, int *y_stride, int *uv_stride)
{
    if (y_stride) {
        *y_stride = round_up_32(width);
    }
    if (uv_stride) {
        *uv_stride = round_up_32(width / 2);
    }
}

void fused_plane_strides_16(int width, int *y_stride, int *uv_stride)
{
    if (y_stride) {
        *y_stride = round_up_32(width * 2);
    }
    if (uv_stride) {
        /* chroma_w = width/2 samples, 2 bytes each */
        *uv_stride = round_up_32((width / 2) * 2);
    }
}

size_t fused_scaler_ctx_sizeof(void)
{
    return sizeof(fused_scaler_ctx_t);
}

size_t fused_hdr_ctx_sizeof(void)
{
    return sizeof(fused_hdr_ctx_t);
}
