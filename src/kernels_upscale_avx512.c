/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_upscale_avx512.c - AVX-512 (x86_64) upscale kernels, SDR + HDR.
 *
 * Entry points mirror kernels_upscale_avx2.c:
 *   fused_kernel_upscale_avx512        - upscale-only (SDR)
 *   fused_kernel_thirds_up_avx512      - thirds downscale + upscale (SDR)
 *   fused_kernel_pow2_up_avx512        - pow2 downscale + upscale (SDR)
 *   fused_kernel_upscale_hdr_avx512    - upscale-only (HDR)
 *   fused_kernel_thirds_up_hdr_avx512  - thirds downscale + upscale (HDR)
 *   fused_kernel_pow2_up_hdr_avx512    - pow2 downscale + upscale (HDR)
 *
 * No upscale path has a 512-bit implementation yet, so the upscale-only
 * entries delegate to their AVX2 counterparts.  The combined entries are
 * two-stage sequencers (downscale pass, then upscale pass) and each stage
 * routes independently to the best implementation available: the
 * downscale stage to the AVX-512 downscale entries, the upscale stage to
 * AVX2.  Dispatch (funnelcake.c / funnelcake_hdr.c) already routes
 * AVX-512-capable hosts here, so improving any stage only ever means
 * editing kernel files - no plumbing changes.  Note that the large
 * terminal 2x upscales are store-bandwidth bound and stream past the
 * cache (see up_2x_stream_min_bytes in the AVX2 upscaler); wider vectors
 * cannot lift a DRAM floor, so the upscale delegations may well be
 * permanent.
 *
 * When real 512-bit code lands it must follow the same compile-time
 * self-stubbing pattern as kernels_avx512.c: real kernels inside a
 * __AVX512F__/__AVX512BW__/__AVX512VL__/__AVX512VBMI__ guard, delegation
 * in the #else, so compilers without AVX-512 support keep building this
 * file with no flags and no loss.  Until then the file is flag-agnostic -
 * plain delegation compiles identically either way.
 *
 * Guarded by __x86_64__ so this file is a no-op on other platforms.
 */

#if defined(__x86_64__)

#include "internal.h"

/* --- SDR ---------------------------------------------------------------- */

void fused_kernel_upscale_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
}

void fused_kernel_thirds_up_avx512(const fused_kernel_params_t *p,
                                   const uint8_t *src_y,
                                   const uint8_t *src_u,
                                   const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
    }
}

/* --- HDR ---------------------------------------------------------------- */

void fused_kernel_upscale_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
}

void fused_kernel_thirds_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                       const uint16_t *src_y,
                                       const uint16_t *src_u,
                                       const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_hdr_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_hdr_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
    }
}

#endif /* __x86_64__ */
