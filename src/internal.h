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
    int src_uv_height;      /* chroma rows in the effective source region   */
    int chroma_format;      /* FUSED_CHROMA_*                               */

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


#endif /* FUNNELCAKE_INTERNAL_H */
