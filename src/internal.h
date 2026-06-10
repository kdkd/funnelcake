/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#ifndef FUNNELCAKE_INTERNAL_H
#define FUNNELCAKE_INTERNAL_H

#include <stdint.h>
#include "funnelcake.h"

#include <stddef.h>
#include <stdlib.h>     /* posix_memalign */
#if defined(__linux__)
#  include <sys/mman.h> /* madvise, MADV_HUGEPAGE */
#endif

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */

/* Maximum number of simultaneous output steps */
#define FUSED_MAX_STEPS     8

/* Allocations at or above this size get a transparent-huge-page hint
 * (madvise(MADV_HUGEPAGE)) on Linux.  Output planes large enough to span
 * the L1 dTLB and benefit from uninterrupted hardware-prefetch streams
 * across page boundaries cross this threshold; smaller buffers (chroma
 * planes of deep cascades, scratch pools) sit below it and would only
 * waste memory if rounded up to a 2 MB huge page. */
#define FUSED_THP_HINT_THRESHOLD ((size_t)(2 * 1024 * 1024))

/* posix_memalign + size-gated MADV_HUGEPAGE hint.  The madvise call is
 * a hint: kernels with transparent_hugepage=never silently ignore it,
 * and on non-Linux platforms the call is compiled out entirely.  The
 * underlying allocation is a normal heap pointer that free()s normally. */
static inline int fused_alloc_aligned(void **out, size_t alignment, size_t size)
{
    int rc = posix_memalign(out, alignment, size);
    if (rc != 0) return rc;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (size >= FUSED_THP_HINT_THRESHOLD) {
        (void)madvise(*out, size, MADV_HUGEPAGE);
    }
#endif
    return rc;
}

/* Scratch pool bump allocator.  Kernels that need several internal
 * scratch buffers carve them out of a single persistent pool allocated
 * in init, avoiding per-frame malloc/free which causes first-touch
 * page faults and inflates max per-frame latency.  The bump cursor is
 * rounded up to a 64-byte boundary before each allocation so every
 * carved sub-buffer is cache-line aligned. */
typedef struct {
    uint8_t *base;
    size_t   used;
    size_t   size;
} fused_scratch_t;

static inline void fused_scratch_init(fused_scratch_t *s,
                                      uint8_t *base, size_t size)
{
    s->base = base;
    s->used = 0;
    s->size = size;
}

static inline void *fused_scratch_alloc(fused_scratch_t *s, size_t n)
{
    /* Align the cursor up to 64 bytes. */
    size_t aligned = (s->used + 63) & ~(size_t)63;
    if (aligned + n > s->size) return NULL;
    void *p = s->base + aligned;
    s->used = aligned + n;
    return p;
}

/* Emit a one-time diagnostic when a scratch pool is exhausted. Defined in
 * log.c. Called by the scalar kernels when fused_scratch_alloc fails - an
 * init-sizing invariant violation that must never happen in correct code. */
void fused_scratch_exhausted_warn(void);

/* Kernel family identifiers */
#define FUSED_FAMILY_THIRDS 0   /* 1.5x/3x/6x/12x - divide-by-3 cascade */
#define FUSED_FAMILY_POW2   1   /* 2x/4x/8x/16x   - power-of-two cascade */


/* --------------------------------------------------------------------------
 * fused_kernel_params_t
 *
 * Precomputed parameters passed by pointer to each kernel invocation.
 * Filled once by fused_scaler_init; read-only during fused_scaler_run.
 * -------------------------------------------------------------------------- */

typedef struct {
    /* Effective source dimensions (after crop-to-fit, <= original) */
    int src_width;          /* luma pixels across the effective region      */
    int src_height;         /* luma rows in the effective region            */
    int src_y_stride;       /* bytes per luma row in the source buffer      */
    int src_uv_stride;      /* bytes per chroma row in the source buffer    */

    /* Kernel family and cascade shape */
    int family;             /* FUSED_FAMILY_THIRDS or FUSED_FAMILY_POW2     */
    int cascade_depth;      /* number of cascade levels (1..FUSED_MAX_STEPS) */
    int vert_period;        /* source rows consumed per output row group     */
                            /* 3 for thirds family, 2 for pow2 family        */

    /* Which outputs in out[] are live (bitmask, bit i set => out[i] valid) */
    uint32_t active_outputs;

    /* Per-output geometry and destination pointers (indexed 0..FUSED_MAX_STEPS-1)
     * Only entries where (active_outputs >> i) & 1 are valid. */
    struct {
        int      width;         /* luma output width in pixels              */
        int      height;        /* luma output height in pixels             */
        int      y_stride;      /* bytes per luma row in destination        */
        int      uv_stride;     /* bytes per chroma row in destination      */
        uint8_t *plane_y;       /* destination luma plane base pointer      */
        uint8_t *plane_u;       /* destination Cb plane base pointer        */
        uint8_t *plane_v;       /* destination Cr plane base pointer        */
    } out[FUSED_MAX_STEPS];

    /* Precomputed inner-loop counts (derived from effective src dimensions) */
    int chunks_per_row;     /* number of full SIMD-width chunks per row     */
    int tail_bytes;         /* leftover luma bytes after full chunks        */
    int row_groups;         /* number of complete vert_period-row groups    */

    /* --------------------------------------------------------------------
     * Upscale cascade - optional. Zero values mean "no upscaling".
     *
     * The upscale chain is a contiguous prefix of pow2 2x levels (0..5)
     * followed optionally by a 1.5x tail. up_out[0..4] hold the 2x..32x
     * outputs; up_out[5] holds the 1.5x tail (indexed as FUSED_UP_IDX_TAIL).
     * Slots not in upscale_active have NULL plane pointers.
     * -------------------------------------------------------------------- */
    int      upscale_cascade_depth;   /* 0..5 - number of 2x pow2 levels     */
    int      upscale_tail_1_5x;       /* 0 or 1                              */
    uint32_t upscale_active;          /* bit i set => up_out[i] is live      */

    struct {
        int      width;
        int      height;
        int      y_stride;
        int      uv_stride;
        uint8_t *plane_y;
        uint8_t *plane_u;
        uint8_t *plane_v;
    } up_out[FUSED_MAX_UPSCALE_STEPS];

    /* Persistent scratch row buffer for upscale helpers that need a
     * vertical-blend intermediate.  Sized at init to the widest input
     * row any upscale helper will see (deepest 2x level or the 1.5x tail
     * input).  Shared across Y/U/V passes since they are processed
     * sequentially.  NULL if no upscale is active. */
    uint8_t *upscale_scratch;

    /* Second persistent scratch row, the same width as upscale_scratch and
     * carved from the same allocation (points one aligned row past it).  The
     * scalar 2x upscale expands each row in two auto-vectorizable passes -
     * one writes horizontal midpoints here, the other interleaves them with
     * the source row - which needs a temp distinct from upscale_scratch (the
     * vertical-blend row).  NULL if no upscale is active. */
    uint8_t *upscale_scratch2;

    /* Scratch pool for downscale kernels.  Allocated once at init and
     * sized to the max bytes the selected kernel family needs for its
     * internal vertical-cascade / horizontal-cascade buffers.  Kernels
     * carve sub-buffers from this pool instead of per-frame malloc,
     * which was causing first-touch page faults and inflating the max
     * per-frame latency.  NULL if no downscale is active. */
    uint8_t *scratch_pool;
    size_t   scratch_pool_size;
} fused_kernel_params_t;


/* --------------------------------------------------------------------------
 * fused_kernel_fn
 *
 * All kernel entry points share this signature. The kernel reads src_y/u/v
 * through the strides and dimensions in p, writing to p->out[i].plane_*
 * for each active output.
 * -------------------------------------------------------------------------- */

typedef void (*fused_kernel_fn)(const fused_kernel_params_t *p,
                                const uint8_t *src_y,
                                const uint8_t *src_u,
                                const uint8_t *src_v);


/* --------------------------------------------------------------------------
 * fused_internal_t
 *
 * The opaque blob stored in fused_scaler_ctx_t._internal.
 * Allocated and populated by fused_scaler_init; freed by fused_scaler_free.
 * -------------------------------------------------------------------------- */

typedef struct {
    fused_kernel_params_t params;
    fused_kernel_fn       kernel_fn;
    int                   has_simd;  /* 1 if a SIMD kernel was selected, 0 = scalar only */
} fused_internal_t;


/* --------------------------------------------------------------------------
 * Kernel entry point declarations
 *
 * Scalar kernels are always available.
 * AVX2 kernels are declared only on x86_64.
 * NEON kernels are declared only on aarch64.
 * -------------------------------------------------------------------------- */

/* Scalar (always compiled) */
void fused_kernel_thirds_scalar(const fused_kernel_params_t *p,
                                const uint8_t *src_y,
                                const uint8_t *src_u,
                                const uint8_t *src_v);

void fused_kernel_pow2_scalar(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v);

#if defined(__x86_64__)
/* AVX2 (x86_64 only) */
void fused_kernel_thirds_avx2(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);

void fused_kernel_pow2_avx2(const fused_kernel_params_t *p,
                             const uint8_t *src_y,
                             const uint8_t *src_u,
                             const uint8_t *src_v);

/* AVX-512 (x86_64 only).  These symbols always exist on x86_64: when the
 * compiler can't build real AVX-512 code the kernels_*_avx512.c files
 * compile self-stubbed delegations to the AVX2 kernels instead.  Dispatch
 * must gate on fused_avx512_compiled() && caps->has_avx512 so the stubs
 * are never selected.  Entry points not yet ported to 512-bit also
 * delegate to AVX2 internally - the swap to the avx512 table is always
 * safe. */
int fused_avx512_compiled(void);

void fused_kernel_thirds_avx512(const fused_kernel_params_t *p,
                                const uint8_t *src_y,
                                const uint8_t *src_u,
                                const uint8_t *src_v);

void fused_kernel_pow2_avx512(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v);
#endif /* __x86_64__ */

#if defined(__aarch64__)
/* NEON (aarch64 only) */
void fused_kernel_thirds_neon(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);

void fused_kernel_pow2_neon(const fused_kernel_params_t *p,
                             const uint8_t *src_y,
                             const uint8_t *src_u,
                             const uint8_t *src_v);
#endif /* __aarch64__ */

#if defined(__riscv) && (__riscv_xlen == 64)
/* RVV (riscv64 only) */
void fused_kernel_thirds_rvv(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v);

void fused_kernel_pow2_rvv(const fused_kernel_params_t *p,
                            const uint8_t *src_y,
                            const uint8_t *src_u,
                            const uint8_t *src_v);
#endif /* __riscv */


/* --------------------------------------------------------------------------
 * Upscale kernel entry points (SDR)
 *
 * Three variants:
 *   - upscale-only: downscale flags are zero, only upscale outputs produced
 *   - thirds_up:    downscale = thirds family AND upscale both active
 *   - pow2_up:      downscale = pow2 family AND upscale both active
 *
 * Scalar always available. SIMD guarded by arch.
 * -------------------------------------------------------------------------- */

void fused_kernel_upscale_scalar(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v);

void fused_kernel_thirds_up_scalar(const fused_kernel_params_t *p,
                                   const uint8_t *src_y,
                                   const uint8_t *src_u,
                                   const uint8_t *src_v);

void fused_kernel_pow2_up_scalar(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v);

#if defined(__x86_64__)
void fused_kernel_upscale_avx2(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);

void fused_kernel_thirds_up_avx2(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v);

void fused_kernel_pow2_up_avx2(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);

void fused_kernel_upscale_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v);

void fused_kernel_thirds_up_avx512(const fused_kernel_params_t *p,
                                   const uint8_t *src_y,
                                   const uint8_t *src_u,
                                   const uint8_t *src_v);

void fused_kernel_pow2_up_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v);
#endif /* __x86_64__ */

#if defined(__aarch64__)
void fused_kernel_upscale_neon(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);

void fused_kernel_thirds_up_neon(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v);

void fused_kernel_pow2_up_neon(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);
#endif /* __aarch64__ */

#if defined(__riscv) && (__riscv_xlen == 64)
void fused_kernel_upscale_rvv(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);

void fused_kernel_thirds_up_rvv(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v);

void fused_kernel_pow2_up_rvv(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v);
#endif /* __riscv */


/* ==========================================================================
 * HDR10 internal types
 *
 * Parallel to the 8-bit types above, but with uint16_t planes for 10-bit
 * samples. Used by the HDR kernel files and funnelcake_hdr.c.
 * ========================================================================== */


/* --------------------------------------------------------------------------
 * fused_hdr_kernel_params_t
 *
 * 10-bit analog of fused_kernel_params_t. Same structure, wider plane
 * pointers. Filled by fused_hdr_init; read-only during fused_hdr_run.
 * -------------------------------------------------------------------------- */

typedef struct {
    int src_width;
    int src_height;
    int src_y_stride;       /* bytes per luma row in source                */
    int src_uv_stride;      /* bytes per chroma row (or interleaved UV)    */

    int family;
    int cascade_depth;
    int vert_period;

    uint32_t active_outputs;

    struct {
        int       width;
        int       height;
        int       y_stride;     /* bytes per row */
        int       uv_stride;    /* bytes per row */
        uint16_t *plane_y;
        uint16_t *plane_u;
        uint16_t *plane_v;
    } out[FUSED_MAX_STEPS];

    int chunks_per_row;     /* full SIMD-width chunks per row (16-bit elements) */
    int tail_elements;      /* leftover luma elements after full chunks         */
    int row_groups;

    /* Pre-computed element strides (stride / sizeof(uint16_t)).  Avoids a
     * division in the kernel entry points on every frame. */
    int src_y_el_stride;
    int src_uv_el_stride;

    /* P010 format: kernel deinterleaves chroma at load time */
    int is_p010;            /* 1 = interleaved UV source, 0 = planar */

    /* Pre-allocated P010 deinterleave buffers (avoids per-frame malloc).
     * Only allocated when is_p010 == 1.  Each holds one chroma plane at
     * source chroma dimensions with 32-byte aligned stride. */
    uint16_t *p010_tmp_u;   /* NULL if !is_p010 */
    uint16_t *p010_tmp_v;   /* NULL if !is_p010 */
    int       p010_tmp_stride;  /* bytes per row (32-byte aligned) */

    /* --------------------------------------------------------------------
     * Upscale cascade - HDR-only for this iteration (no tonemapping).
     * Same layout as the SDR fused_kernel_params_t but with uint16_t planes.
     * -------------------------------------------------------------------- */
    int      upscale_cascade_depth;
    int      upscale_tail_1_5x;
    uint32_t upscale_hdr_active;

    struct {
        int       width;
        int       height;
        int       y_stride;
        int       uv_stride;
        uint16_t *plane_y;
        uint16_t *plane_u;
        uint16_t *plane_v;
    } hdr_up_out[FUSED_MAX_UPSCALE_STEPS];

    /* Persistent scratch row buffer for HDR upscale helpers (uint16_t).
     * Same semantics as the SDR upscale_scratch above. */
    uint16_t *upscale_scratch_hdr;

    /* Second HDR scratch row carved from the same allocation as
     * upscale_scratch_hdr.  See upscale_scratch2 in fused_kernel_params_t
     * for rationale (two-pass 2x interleave). NULL if no upscale active. */
    uint16_t *upscale_scratch_hdr2;

    /* Scratch pool for HDR downscale kernels.  See fused_kernel_params_t
     * for rationale.  Sized in bytes; kernels cast to uint16_t* and use
     * offset arithmetic for their sub-buffers. */
    uint8_t *scratch_pool;
    size_t   scratch_pool_size;
} fused_hdr_kernel_params_t;

/* Deinterleave the effective P010/P210 chroma region into the persistent
 * planar temp buffers allocated by fused_hdr_init.  For P210, src_uv_el_stride
 * has already been doubled by init so this naturally skips every other chroma
 * row while producing a 4:2:0 planar temp. */
static inline int fused_hdr_deinterleave_p010(const fused_hdr_kernel_params_t *p,
                                              const uint16_t *src_uv)
{
    if (!p->is_p010) return 0;
    if (!src_uv || !p->p010_tmp_u || !p->p010_tmp_v) return -1;

    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;
    int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

    for (int y = 0; y < chroma_h; y++) {
        const uint16_t *row = src_uv + (size_t)y * (size_t)p->src_uv_el_stride;
        uint16_t *u_row = p->p010_tmp_u + (size_t)y * (size_t)planar_el_stride;
        uint16_t *v_row = p->p010_tmp_v + (size_t)y * (size_t)planar_el_stride;
        for (int x = 0; x < chroma_w; x++) {
            u_row[x] = row[2 * x];
            v_row[x] = row[2 * x + 1];
        }
    }

    return 0;
}


/* --------------------------------------------------------------------------
 * fused_hdr_kernel_fn
 *
 * 10-bit kernel entry point signature. Same pattern as fused_kernel_fn
 * but with uint16_t source planes.
 * -------------------------------------------------------------------------- */

typedef void (*fused_hdr_kernel_fn)(const fused_hdr_kernel_params_t *p,
                                    const uint16_t *src_y,
                                    const uint16_t *src_u,
                                    const uint16_t *src_v);


/* --------------------------------------------------------------------------
 * fused_hdr_internal_t
 *
 * Opaque blob stored in fused_hdr_ctx_t._internal.
 * -------------------------------------------------------------------------- */

typedef struct {
    fused_hdr_kernel_params_t params;
    fused_hdr_kernel_fn       kernel_fn;
    int                       has_simd;

    /* Tone mapping LUTs - generated at init from transfer + curve config.
     *
     * All color science (EOTF, tone curve, SDR OETF, range handling) is
     * compiled into pq_to_sdr at init.  Per-pixel work is then three
     * saturating adds and three byte lookups - no floats anywhere.
     *
     * lut_y: 10-bit code -> 8-bit SDR output (1024 entries).  For built-in
     *   curves this is a copy of pq_to_sdr; for FUSED_TONEMAP_CUSTOM it is
     *   the caller's LUT.  Used by the custom-LUT luma pass.
     *
     * pq_to_sdr: 10-bit PQ/HLG code -> 8-bit tone-mapped SDR gamma output.
     *   Applied per channel: gamma-domain R', G', B' (each Y' + a chroma
     *   delta, below) index this one shared table.  Per-channel tone
     *   mapping through one monotonic LUT in PQ space is equivalent to
     *   per-channel in linear space, and PQ indexing is perceptually
     *   uniform so shadow resolution is preserved by construction.
     *
     * pq_*_delta: per-chroma-sample offsets in Y'-code units, from the
     *   BT.2020 NCL inverse in the gamma domain:
     *     R' = Y' + 1.4746*Cr            -> pq_r_delta[Cr]
     *     B' = Y' + 1.8814*Cb            -> pq_b_delta[Cb]
     *     G' = (Y' - Kr*R' - Kb*B')/Kg
     *        = Y' - (Kr*1.4746*Cr + Kb*1.8814*Cb)/Kg
     *                                    -> pq_g_delta_cr[Cr] + pq_g_delta_cb[Cb]
     *   (using 1 - Kr - Kb = Kg).  Limited/full range scaling between the
     *   chroma and luma code domains is folded in at init.
     */
    uint8_t  lut_y[1024];
    uint8_t  pq_to_sdr[1024];       /* PQ/HLG code -> tone-mapped SDR */
    int16_t  pq_r_delta[1024];      /* Cr code -> R' offset from Y' */
    int16_t  pq_b_delta[1024];      /* Cb code -> B' offset from Y' */
    int16_t  pq_g_delta_cr[1024];   /* Cr code -> G' offset contribution */
    int16_t  pq_g_delta_cb[1024];   /* Cb code -> G' offset contribution */

    /* Q10 coefficients defining the delta tables:
     *   pq_*_delta[i] = (int16_t)(((i - 512) * coef + (1 << 9)) >> 10)
     * SIMD kernels evaluate this exact integer form arithmetically
     * instead of gathering from the tables, so the coefficients are the
     * single source of truth for scalar/SIMD bit-exactness.  Q10 keeps
     * every coefficient inside int16, so the kernels evaluate the form
     * as one 16-bit rounding multiply-high of (x << 5):
     *   pmulhrsw / vqrdmulh / vsmul@RNU == ((x*32)*coef + 16384) >> 15. */
    int32_t  delta_coef_r;
    int32_t  delta_coef_b;
    int32_t  delta_coef_g_cr;
    int32_t  delta_coef_g_cb;

    /* SDR YCbCr re-encode constants, derived from dst_range at init.
     * cb/cr_out_scale are 8.8 fixed point. */
    int      cb_out_scale;
    int      cr_out_scale;
    int      chroma_out_min;
    int      chroma_out_max;

    /* Temp 10-bit buffers for SDR-only outputs (no HDR output requested
     * at that step, but we need a 10-bit intermediate to tone map from).
     * Only allocated for steps in sdr_flags but not in hdr_flags. */
    struct { uint16_t *y, *u, *v; } sdr_temp[FUSED_MAX_STEPS];

    /* Which outputs need tone mapping after scaling */
    uint32_t sdr_flags;
    int      tonemap_1x;
    int      is_custom_lut;  /* 1 if using FUSED_TONEMAP_CUSTOM (skip RGB chroma path) */
} fused_hdr_internal_t;


/* --------------------------------------------------------------------------
 * HDR kernel entry point declarations
 * -------------------------------------------------------------------------- */

/* Scalar (always compiled) */
void fused_kernel_thirds_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                    const uint16_t *src_y,
                                    const uint16_t *src_u,
                                    const uint16_t *src_v);

void fused_kernel_pow2_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src_y,
                                  const uint16_t *src_u,
                                  const uint16_t *src_v);

#if defined(__x86_64__)
/* AVX2 (x86_64 only) */
void fused_kernel_thirds_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);

void fused_kernel_pow2_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                 const uint16_t *src_y,
                                 const uint16_t *src_u,
                                 const uint16_t *src_v);

void fused_kernel_thirds_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                    const uint16_t *src_y,
                                    const uint16_t *src_u,
                                    const uint16_t *src_v);

void fused_kernel_pow2_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src_y,
                                  const uint16_t *src_u,
                                  const uint16_t *src_v);
#endif /* __x86_64__ */

#if defined(__aarch64__)
/* NEON (aarch64 only) */
void fused_kernel_thirds_hdr_neon(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);

void fused_kernel_pow2_hdr_neon(const fused_hdr_kernel_params_t *p,
                                 const uint16_t *src_y,
                                 const uint16_t *src_u,
                                 const uint16_t *src_v);
#endif /* __aarch64__ */

#if defined(__riscv) && (__riscv_xlen == 64)
/* RVV (riscv64 only) */
void fused_kernel_thirds_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src_y,
                                  const uint16_t *src_u,
                                  const uint16_t *src_v);

void fused_kernel_pow2_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                const uint16_t *src_y,
                                const uint16_t *src_u,
                                const uint16_t *src_v);
#endif /* __riscv */


/* --------------------------------------------------------------------------
 * HDR upscale kernel entry points
 * -------------------------------------------------------------------------- */

void fused_kernel_upscale_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v);

void fused_kernel_thirds_up_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                       const uint16_t *src_y,
                                       const uint16_t *src_u,
                                       const uint16_t *src_v);

void fused_kernel_pow2_up_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v);

#if defined(__x86_64__)
void fused_kernel_upscale_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);

void fused_kernel_thirds_up_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v);

void fused_kernel_pow2_up_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);

void fused_kernel_upscale_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v);

void fused_kernel_thirds_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                       const uint16_t *src_y,
                                       const uint16_t *src_u,
                                       const uint16_t *src_v);

void fused_kernel_pow2_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v);
#endif /* __x86_64__ */

#if defined(__aarch64__)
void fused_kernel_upscale_hdr_neon(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);

void fused_kernel_thirds_up_hdr_neon(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v);

void fused_kernel_pow2_up_hdr_neon(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);
#endif /* __aarch64__ */

#if defined(__riscv) && (__riscv_xlen == 64)
void fused_kernel_upscale_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);

void fused_kernel_thirds_up_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v);

void fused_kernel_pow2_up_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v);
#endif /* __riscv */


#endif /* FUNNELCAKE_INTERNAL_H */
