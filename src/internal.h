#ifndef FUNNELCAKE_INTERNAL_H
#define FUNNELCAKE_INTERNAL_H

#include <stdint.h>
#include "funnelcake.h"

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */

/* Maximum number of simultaneous output steps */
#define FUSED_MAX_STEPS     8

/* Kernel family identifiers */
#define FUSED_FAMILY_THIRDS 0   /* 1.5x/3x/6x/12x — divide-by-3 cascade */
#define FUSED_FAMILY_POW2   1   /* 2x/4x/8x/16x   — power-of-two cascade */


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
} fused_hdr_kernel_params_t;


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

    /* Tone mapping LUTs — generated at init from transfer + curve config.
     *
     * lut_y: 10-bit PQ/HLG input → 8-bit BT.709 gamma output (1024 entries).
     *   Used by the luma pass for fast per-pixel tone mapping.
     *
     * pq_to_linear: 10-bit PQ code → linear luminance as float [0, 1]
     *   where 1.0 = 10000 nits.  Used by the chroma pass to reconstruct
     *   linear-light R, G, B from YCbCr for correct gamut mapping.
     *
     * linear_to_sdr: linear luminance [0, 1] quantized to 12 bits → 8-bit
     *   SDR gamma output.  Incorporates tone curve + BT.709 OETF.
     *   Indexed by (linear_value * 4095).  Used by the chroma pass to
     *   tone-map the reconstructed R, G, B channels individually.
     */
    uint8_t  lut_y[1024];
    float    pq_to_linear[1024];    /* PQ code → linear [0,1] */
    uint8_t  linear_to_sdr[4096];   /* linear 12-bit → 8-bit SDR gamma */

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


#endif /* FUNNELCAKE_INTERNAL_H */
