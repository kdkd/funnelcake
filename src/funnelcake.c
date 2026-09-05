#include <stdatomic.h>
#include <limits.h>
/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "funnelcake.h"
#include "internal.h"
#include "detect.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * Step descriptor table
 *
 * Ordered from shallowest (bit 0) to deepest (bit 7).
 * ratio_n/ratio_d encode: out = src * ratio_n / ratio_d
 * -------------------------------------------------------------------------- */

typedef struct {
    uint32_t    flag;       /* FUSED_SCALE_* bit */
    int         bit;        /* bit index 0..7 */
    int         ratio_n;    /* numerator   (out = src * n / d) */
    int         ratio_d;    /* denominator */
    const char *name;       /* human-readable for diagnostics */
} step_desc_t;

static const step_desc_t k_steps[8] = {
    { FUSED_SCALE_1_5X,  0, 2, 3,  "FUSED_SCALE_1_5X"  },
    { FUSED_SCALE_2X,    1, 1, 2,  "FUSED_SCALE_2X"    },
    { FUSED_SCALE_3X,    2, 1, 3,  "FUSED_SCALE_3X"    },
    { FUSED_SCALE_4X,    3, 1, 4,  "FUSED_SCALE_4X"    },
    { FUSED_SCALE_6X,    4, 1, 6,  "FUSED_SCALE_6X"    },
    { FUSED_SCALE_8X,    5, 1, 8,  "FUSED_SCALE_8X"    },
    { FUSED_SCALE_12X,   6, 1, 12, "FUSED_SCALE_12X"   },
    { FUSED_SCALE_16X,   7, 1, 16, "FUSED_SCALE_16X"   },
};

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

/* Round down x to the nearest multiple of m (m must be a power-of-two or
 * any positive integer; this version works for any positive m). */
static int round_down(int x, int m) {
    return (x / m) * m;
}

/* Round up x to the next multiple of 32. */
static int stride_for(int width) {
    return (width + 31) & ~31;
}

/* --------------------------------------------------------------------------
 * fused_scaler_init
 * -------------------------------------------------------------------------- */

static int fused_scaler_init_impl(fused_scaler_ctx_t *ctx, size_t *planned,
                                fused_internal_t *query_state)
{
    if (!ctx) return FUSED_ERR_BAD_DIMENSIONS;

    /* ------------------------------------------------------------------ */
    /* 1. Validate source dimensions                                        */
    /* ------------------------------------------------------------------ */

    if (ctx->src_width <= 0 || ctx->src_height <= 0 ||
        ctx->src_width > INT_MAX / 64 || ctx->src_height > INT_MAX / 64 ||
        (ctx->src_width & 1) || (ctx->src_height & 1)) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: src_width=%d src_height=%d must be positive and even\n",
            ctx->src_width, ctx->src_height);
        return FUSED_ERR_BAD_DIMENSIONS;
    }

    if (ctx->src_y_stride < ctx->src_width ||
        (ctx->src_y_stride & 31)) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: src_y_stride=%d must be >= src_width=%d and a multiple of 32\n",
            ctx->src_y_stride, ctx->src_width);
        return FUSED_ERR_BAD_ALIGNMENT;
    }

    {
        int min_uv_stride = ctx->src_width / 2;
        if (ctx->src_uv_stride < min_uv_stride ||
            (ctx->src_uv_stride & 31)) {
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake: src_uv_stride=%d must be >= src_width/2=%d and a multiple of 32\n",
                ctx->src_uv_stride, min_uv_stride);
            return FUSED_ERR_BAD_ALIGNMENT;
        }
    }

    /* Kernels use int row offsets and HDR 4:2:2 doubles the UV stride. */
    if (ctx->src_y_stride <= 0 || ctx->src_uv_stride <= 0 ||
        ctx->src_y_stride > INT_MAX / ctx->src_height ||
        ctx->src_uv_stride > INT_MAX / ctx->src_height) {
        return FUSED_ERR_BAD_DIMENSIONS;
    }

    /* ------------------------------------------------------------------ */
    /* 2. Validate requested_flags and upscale_flags                        */
    /* ------------------------------------------------------------------ */

    uint32_t req       = ctx->requested_flags;
    uint32_t up_req    = ctx->upscale_flags;
    int      up_tail   = ctx->upscale_tail_1_5x ? 1 : 0;

    /* Relaxed zero-flags rule: allow req == 0 iff upscale is requested. */
    if (req == 0 && up_req == 0 && !up_tail) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: requested_flags and upscale_flags are both zero\n");
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* Check for unknown downscale bits (outside 0..7) */
    if (req & ~0xFFu) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: requested_flags 0x%08X contains unknown bits\n", req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* Check for unknown upscale bits */
    if (up_req & ~FUSED_UPSCALE_POW2_MASK) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: upscale_flags 0x%08X contains unknown bits\n", up_req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* Upscale cascade must be a contiguous prefix: {}, {2x}, {2x,4x}, etc.
     * Equivalent to: the set bits form (2^N - 1) for some N in [0..5]. */
    {
        uint32_t tmp = up_req;
        while (tmp & 1) tmp >>= 1;
        if (tmp != 0) {
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake: upscale_flags 0x%02X is not a contiguous prefix "
                "(must be {}, {2x}, {2x,4x}, {2x,4x,8x}, ...)\n", up_req);
            return FUSED_ERR_INVALID_FLAGS;
        }
    }

    int has_thirds = (req & FUSED_SCALE_THIRDS_MASK) != 0;
    int has_pow2   = (req & FUSED_SCALE_POW2_MASK)   != 0;

    if (has_thirds && has_pow2) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: requested_flags 0x%02X mixes thirds and pow2 families\n", req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* Family: if no downscale requested, default to THIRDS (unused for upscale
     * dispatch, but the params struct needs something).  The upscale-only
     * dispatch path will use fused_kernel_upscale_* which ignores family. */
    int family;
    if (has_thirds) family = FUSED_FAMILY_THIRDS;
    else if (has_pow2) family = FUSED_FAMILY_POW2;
    else family = FUSED_FAMILY_POW2;  /* arbitrary - upscale-only */

    /* ------------------------------------------------------------------ */
    /* 3. Compute effective dimensions (crop-to-fit)                        */
    /* ------------------------------------------------------------------ */

    int eff_w = ctx->src_width;
    int eff_h = ctx->src_height;

    if (!(ctx->options & FUSED_OPT_NO_CROP) && req != 0) {
        /*
         * The crop divisor must ensure ALL output luma dimensions are:
         *   (a) exact integers (source divisible by ratio denominator)
         *   (b) even (YUV420 requires even luma dimensions)
         *
         * For pow2: output = source / ratio. Even output requires source
         * divisible by 2 * ratio.
         *
         * For thirds: 1.5x = src*2/3 (even when src%3==0 since 2*3k/3=2k).
         * 3x = src/3 even requires src%6==0. 6x = src/6 even requires
         * src%12==0. 12x = src/12 even requires src%24==0.
         * Height also needs divisibility by the vertical period (6 or 12).
         *
         * Upscale paths never constrain source dimensions beyond the basic
         * even-luma requirement enforced in step 1, so they contribute no
         * additional crop divisor.
         */
        if (family == FUSED_FAMILY_THIRDS) {
            int w_div = 3;   /* 1.5x: src%3==0 sufficient */
            if (req & FUSED_SCALE_3X)  w_div = 6;
            if (req & FUSED_SCALE_6X)  w_div = 12;
            if (req & FUSED_SCALE_12X) w_div = 24;
            eff_w = round_down(eff_w, w_div);

            /* Height: vertical period is 6 (or 12 if 12x active), plus
             * the even-output constraint for the deepest step */
            int h_div = 6;
            if (req & FUSED_SCALE_3X)  { if (h_div < 6)  h_div = 6;  }
            if (req & FUSED_SCALE_6X)  { if (h_div < 12) h_div = 12; }
            if (req & FUSED_SCALE_12X) { if (h_div < 24) h_div = 24; }
            eff_h = round_down(eff_h, h_div);
        } else {
            /* pow2: deepest ratio * 2 ensures even output */
            int max_ratio = 2;
            if (req & FUSED_SCALE_4X)  max_ratio = 4;
            if (req & FUSED_SCALE_8X)  max_ratio = 8;
            if (req & FUSED_SCALE_16X) max_ratio = 16;
            int div = max_ratio * 2;
            eff_w = round_down(eff_w, div);
            eff_h = round_down(eff_h, div);
        }
    }

    if (eff_w <= 0 || eff_h <= 0) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: after crop-to-fit, effective dimensions %dx%d are non-positive\n",
            eff_w, eff_h);
        return FUSED_ERR_BAD_DIMENSIONS;
    }

    ctx->effective_width  = eff_w;
    ctx->effective_height = eff_h;

    int warn_bits = 0;
    if (eff_w != ctx->src_width || eff_h != ctx->src_height) {
        warn_bits |= FUSED_WARN_BIT_CROPPED;
    }

    /* ------------------------------------------------------------------ */
    /* 4. CPU detection & kernel selection                                  */
    /* ------------------------------------------------------------------ */

    const fused_cpu_caps_t *caps = fused_detect_cpu();

    int has_simd = 0;
    fused_kernel_fn simd_thirds_fn    = NULL;
    fused_kernel_fn simd_pow2_fn      = NULL;
    fused_kernel_fn simd_upscale_fn   = NULL;
    fused_kernel_fn simd_thirds_up_fn = NULL;
    fused_kernel_fn simd_pow2_up_fn   = NULL;
    fused_kernel_fn simd_pow2_chroma64_fn = NULL;

#if defined(__aarch64__)
    if (caps->has_neon) {
        has_simd = 1;
        simd_thirds_fn    = fused_kernel_thirds_neon;
        simd_pow2_fn      = fused_kernel_pow2_neon;
        simd_upscale_fn   = fused_kernel_upscale_neon;
        simd_thirds_up_fn = fused_kernel_thirds_up_neon;
        simd_pow2_up_fn   = fused_kernel_pow2_up_neon;
    }
#elif defined(__x86_64__)
    if (caps->has_avx2) {
        has_simd = 1;
        simd_thirds_fn    = fused_kernel_thirds_avx2;
        simd_pow2_fn      = fused_kernel_pow2_avx2;
        simd_upscale_fn   = fused_kernel_upscale_avx2;
        simd_thirds_up_fn = fused_kernel_thirds_up_avx2;
        simd_pow2_up_fn   = fused_kernel_pow2_up_avx2;

        /* AVX-512 needs both the hardware (has_avx512: F+BW+VL+VBMI with
         * OS ZMM state) and a build whose compiler accepted the AVX-512
         * flags (fused_avx512_compiled).  Entry points not yet ported to
         * 512-bit delegate to their AVX2 counterparts internally, so the
         * whole table swaps at once. */
        if (caps->has_avx512 && fused_avx512_compiled()) {
            simd_thirds_fn    = fused_kernel_thirds_avx512;
            simd_pow2_fn      = fused_kernel_pow2_avx512;
            simd_pow2_chroma64_fn = fused_kernel_pow2_chroma64_avx512;
            simd_upscale_fn   = fused_kernel_upscale_avx512;
            simd_thirds_up_fn = fused_kernel_thirds_up_avx512;
            simd_pow2_up_fn   = fused_kernel_pow2_up_avx512;
        }
    }
#elif defined(__riscv) && (__riscv_xlen == 64)
    if (caps->has_rvv) {
        has_simd = 1;
        simd_thirds_fn    = fused_kernel_thirds_rvv;
        simd_pow2_fn      = fused_kernel_pow2_rvv;
        simd_upscale_fn   = fused_kernel_upscale_rvv;
        simd_thirds_up_fn = fused_kernel_thirds_up_rvv;
        simd_pow2_up_fn   = fused_kernel_pow2_up_rvv;
    }
#else
    (void)caps;
#endif

    if (!has_simd) {
        /* One-time stderr notice.  Suppressed when scalar was explicitly
         * requested via FUNNELCAKE_FORCE_SCALAR (the parity test toggles
         * this on and off; printing the warning on every flip would flood
         * the test output and confuse readers into thinking SIMD is broken). */
        static atomic_int g_no_simd_warned = 0;
        const char *force_scalar_env = getenv("FUNNELCAKE_FORCE_SCALAR");
        int forced_scalar = (force_scalar_env != NULL && force_scalar_env[0] != '\0');
        if (!planned && !forced_scalar && !atomic_exchange(&g_no_simd_warned, 1)) {
            fprintf(stderr,
                "funnelcake: no SIMD support detected; using scalar kernel\n");
        }
        warn_bits |= FUSED_WARN_BIT_SCALAR;
    }

    /* ------------------------------------------------------------------ */
    /* 5. Validate each step and allocate output buffers                    */
    /* ------------------------------------------------------------------ */

    uint32_t achieved = 0;
    uint32_t rejected = 0;

    /* Zero all output slots first */
    memset(ctx->outputs, 0, sizeof(ctx->outputs));

    for (int i = 0; i < 8; i++) {
        const step_desc_t *sd = &k_steps[i];

        if (!(req & sd->flag)) continue;  /* not requested */

        /* Compute output dimensions */
        int out_w = eff_w * sd->ratio_n / sd->ratio_d;
        int out_h = eff_h * sd->ratio_n / sd->ratio_d;

        /* (a) Exact integer division check */
        if (out_w * sd->ratio_d != eff_w * sd->ratio_n ||
            out_h * sd->ratio_d != eff_h * sd->ratio_n) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: %s rejected: output dimensions %dx%d not exact"
                " (source %dx%d ratio %d/%d)\n",
                sd->name, out_w, out_h, eff_w, eff_h, sd->ratio_n, sd->ratio_d);
            rejected |= sd->flag;
            continue;
        }

        /* (b) Output luma dimensions must be even */
        if ((out_w & 1) || (out_h & 1)) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: %s rejected: output luma %dx%d has odd dimension\n",
                sd->name, out_w, out_h);
            rejected |= sd->flag;
            continue;
        }

        /* (d) Minimum size */
        if (out_w < 32 || out_h < 2) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: %s rejected: output %dx%d is too small (min 32x2)\n",
                sd->name, out_w, out_h);
            rejected |= sd->flag;
            continue;
        }

        /* (c) SIMD chroma width constraint */
        int chroma_w = out_w / 2;
        /* Scalar is used for this step if there is no SIMD at all, or if SIMD
         * exists but the chroma width is misaligned (handled below). Either
         * way the per-output contract requires fallback to report scalar. */
        int step_fallback = !has_simd;

        if (has_simd && (chroma_w & 31)) {
            if (ctx->options & FUSED_OPT_NO_FALLBACK) {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake: %s rejected: chroma output width %d is not a"
                    " multiple of 32 (source %dx%d -> %dx scale luma %d -> chroma %d)\n",
                    sd->name, chroma_w,
                    eff_w, eff_h, sd->ratio_d, out_w, chroma_w);
                rejected |= sd->flag;
                continue;
            } else {
                /* Scalar fallback for this step */
                step_fallback = 1;
                warn_bits |= FUSED_WARN_BIT_SCALAR;
            }
        }

        /* Allocate output buffers */
        int y_stride  = stride_for(out_w);
        int uv_stride = stride_for(chroma_w);
        int chroma_h  = out_h / 2;

        void *py = NULL, *pu = NULL, *pv = NULL;
        if (fused_init_alloc(&py, 32, (size_t)y_stride  * (size_t)out_h, planned)  != 0 ||
            fused_init_alloc(&pu, 32, (size_t)uv_stride * (size_t)chroma_h, planned) != 0 ||
            fused_init_alloc(&pv, 32, (size_t)uv_stride * (size_t)chroma_h, planned) != 0) {
            /* Allocation failure - free what we got and reject this step */
            free(py); free(pu); free(pv);
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: %s rejected: out-of-memory allocating output planes\n",
                sd->name);
            rejected |= sd->flag;
            continue;
        }

        ctx->outputs[i].width    = out_w;
        ctx->outputs[i].height   = out_h;
        ctx->outputs[i].y_stride  = y_stride;
        ctx->outputs[i].uv_stride = uv_stride;
        ctx->outputs[i].plane_y  = (uint8_t *)py;
        ctx->outputs[i].plane_u  = (uint8_t *)pu;
        ctx->outputs[i].plane_v  = (uint8_t *)pv;
        ctx->outputs[i].fallback = step_fallback;

        achieved |= sd->flag;
    }

    /* ------------------------------------------------------------------ */
    /* 5b. Validate upscale levels and allocate upscale output buffers      */
    /* ------------------------------------------------------------------ */

    /* Upscale dimensions: level k (0..4) = eff * 2^(k+1).
     * Tail when N==0 reads source directly -> tail = eff*3/2.
     * Tail when N>=1 reads level N-1 output -> tail = eff*2^N * 3/2.
     * Soft-reject individual levels whose luma dimension exceeds 16384. */
    #define FUSED_UPSCALE_SIZE_CAP 16384

    uint32_t up_achieved = 0;
    int      up_achieved_tail = 0;

    memset(ctx->upscale_outputs, 0, sizeof(ctx->upscale_outputs));

    /* Recompute cascade depth from the validated contiguous-prefix mask */
    int up_N = 0;
    {
        uint32_t tmp = up_req;
        while (tmp & 1) { up_N++; tmp >>= 1; }
    }

    for (int k = 0; k < up_N; k++) {
        int up_w = eff_w << (k + 1);
        int up_h = eff_h << (k + 1);

        if (up_w > FUSED_UPSCALE_SIZE_CAP || up_h > FUSED_UPSCALE_SIZE_CAP) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: upscale level %dx rejected: output %dx%d exceeds "
                "size cap %d\n", (1 << (k + 1)), up_w, up_h,
                FUSED_UPSCALE_SIZE_CAP);
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            continue;
        }

        int uv_w = up_w / 2;
        int y_stride  = stride_for(up_w);
        int uv_stride = stride_for(uv_w);
        int chroma_h  = up_h / 2;

        void *py = NULL, *pu = NULL, *pv = NULL;
        if (fused_init_alloc(&py, 32, (size_t)y_stride  * (size_t)up_h, planned)   != 0 ||
            fused_init_alloc(&pu, 32, (size_t)uv_stride * (size_t)chroma_h, planned) != 0 ||
            fused_init_alloc(&pv, 32, (size_t)uv_stride * (size_t)chroma_h, planned) != 0) {
            free(py); free(pu); free(pv);
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: upscale level %dx rejected: out-of-memory\n",
                (1 << (k + 1)));
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            continue;
        }

        ctx->upscale_outputs[k].width     = up_w;
        ctx->upscale_outputs[k].height    = up_h;
        ctx->upscale_outputs[k].y_stride  = y_stride;
        ctx->upscale_outputs[k].uv_stride = uv_stride;
        ctx->upscale_outputs[k].plane_y   = (uint8_t *)py;
        ctx->upscale_outputs[k].plane_u   = (uint8_t *)pu;
        ctx->upscale_outputs[k].plane_v   = (uint8_t *)pv;
        ctx->upscale_outputs[k].fallback  = !has_simd;

        up_achieved |= (1u << k);
    }

    /* 1.5x tail */
    if (up_tail) {
        int tail_src_w, tail_src_h;
        if (up_N == 0) {
            tail_src_w = eff_w;
            tail_src_h = eff_h;
        } else {
            /* Tail reads from the deepest pow2 level - but only if that level
             * was actually achieved. If the deepest level was rejected, the
             * tail cannot be produced either. */
            if (!(up_achieved & (1u << (up_N - 1)))) {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake: upscale 1.5x tail rejected: deepest pow2 "
                    "level %dx was not achieved\n", (1 << up_N));
                warn_bits |= FUSED_WARN_BIT_PARTIAL;
                goto tail_done;
            }
            tail_src_w = eff_w << up_N;
            tail_src_h = eff_h << up_N;
        }

        int tail_w = tail_src_w * 3 / 2;
        int tail_h = tail_src_h * 3 / 2;

        /* Source dimensions must be even for the 1.5x output to be exact and
         * even-aligned (YUV420). eff_w/eff_h are guaranteed even by step 1. */
        if (tail_w > FUSED_UPSCALE_SIZE_CAP || tail_h > FUSED_UPSCALE_SIZE_CAP) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: upscale 1.5x tail rejected: output %dx%d "
                "exceeds size cap %d\n", tail_w, tail_h, FUSED_UPSCALE_SIZE_CAP);
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            goto tail_done;
        }

        /* Tail output dimensions must be even (YUV420 constraint). */
        if ((tail_w & 1) || (tail_h & 1)) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: upscale 1.5x tail rejected: output %dx%d has "
                "odd dimension\n", tail_w, tail_h);
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            goto tail_done;
        }

        int tail_uv_w   = tail_w / 2;
        int y_stride    = stride_for(tail_w);
        int uv_stride   = stride_for(tail_uv_w);
        int chroma_h    = tail_h / 2;

        void *py = NULL, *pu = NULL, *pv = NULL;
        if (fused_init_alloc(&py, 32, (size_t)y_stride  * (size_t)tail_h, planned)   != 0 ||
            fused_init_alloc(&pu, 32, (size_t)uv_stride * (size_t)chroma_h, planned) != 0 ||
            fused_init_alloc(&pv, 32, (size_t)uv_stride * (size_t)chroma_h, planned) != 0) {
            free(py); free(pu); free(pv);
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake: upscale 1.5x tail rejected: out-of-memory\n");
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            goto tail_done;
        }

        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].width     = tail_w;
        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].height    = tail_h;
        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].y_stride  = y_stride;
        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].uv_stride = uv_stride;
        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].plane_y   = (uint8_t *)py;
        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].plane_u   = (uint8_t *)pu;
        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].plane_v   = (uint8_t *)pv;
        ctx->upscale_outputs[FUSED_UP_IDX_TAIL].fallback  = !has_simd;

        up_achieved_tail = 1;
    }
tail_done:

    ctx->achieved_upscale_flags = up_achieved;
    ctx->achieved_upscale_tail  = up_achieved_tail;

    /* ------------------------------------------------------------------ */
    /* 6. Check that at least one step was achieved                         */
    /* ------------------------------------------------------------------ */

    ctx->achieved_flags = achieved;
    ctx->rejected_flags = rejected;

    if (rejected) warn_bits |= FUSED_WARN_BIT_PARTIAL;

    int want_down = (achieved != 0);
    int want_up   = (up_achieved != 0) || up_achieved_tail;

    if (!want_down && !want_up) {
        /* Free any planes allocated before the first success */
        fused_scaler_free(ctx);
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: no valid output steps after validation\n");
        return FUSED_ERR_NO_STEPS;
    }

    /* ------------------------------------------------------------------ */
    /* 7. Build kernel params and internal state                            */
    /* ------------------------------------------------------------------ */

    fused_internal_t *state = planned ? query_state : calloc(1, sizeof(fused_internal_t));
    if (planned) *planned += sizeof(fused_internal_t);
    if (!state) {
        fused_scaler_free(ctx);
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: out-of-memory allocating internal state\n");
        return FUSED_ERR_NO_STEPS;
    }

    fused_kernel_params_t *p = &state->params;

    p->src_width   = eff_w;
    p->src_height  = eff_h;
    p->src_y_stride  = ctx->src_y_stride;
    p->src_uv_stride = ctx->src_uv_stride;

    p->family = family;

    /* Cascade depth: number of bit positions with achieved flags */
    int cascade_depth = 0;
    for (int i = 0; i < 8; i++) {
        if (achieved & (1u << i)) cascade_depth++;
    }
    p->cascade_depth = cascade_depth;

    p->vert_period = (family == FUSED_FAMILY_THIRDS) ? 3 : 2;
    p->active_outputs = achieved;

    /* Fill per-output geometry */
    for (int i = 0; i < 8; i++) {
        if (!(achieved & (1u << i))) continue;
        p->out[i].width    = ctx->outputs[i].width;
        p->out[i].height   = ctx->outputs[i].height;
        p->out[i].y_stride  = ctx->outputs[i].y_stride;
        p->out[i].uv_stride = ctx->outputs[i].uv_stride;
        p->out[i].plane_y  = ctx->outputs[i].plane_y;
        p->out[i].plane_u  = ctx->outputs[i].plane_u;
        p->out[i].plane_v  = ctx->outputs[i].plane_v;
    }

    /* Precomputed loop counts (based on SIMD width = 32 luma bytes) */
    p->chunks_per_row = eff_w / 32;
    p->tail_bytes     = eff_w % 32;
    p->row_groups     = eff_h / p->vert_period;

    /* Upscale parameters */
    p->upscale_cascade_depth = up_N;
    p->upscale_tail_1_5x     = up_achieved_tail;
    p->upscale_active        = up_achieved | (up_achieved_tail
                                               ? (1u << FUSED_UP_IDX_TAIL)
                                               : 0u);

    for (int k = 0; k < FUSED_MAX_UPSCALE_STEPS; k++) {
        if (!(p->upscale_active & (1u << k))) continue;
        p->up_out[k].width     = ctx->upscale_outputs[k].width;
        p->up_out[k].height    = ctx->upscale_outputs[k].height;
        p->up_out[k].y_stride  = ctx->upscale_outputs[k].y_stride;
        p->up_out[k].uv_stride = ctx->upscale_outputs[k].uv_stride;
        p->up_out[k].plane_y   = ctx->upscale_outputs[k].plane_y;
        p->up_out[k].plane_u   = ctx->upscale_outputs[k].plane_u;
        p->up_out[k].plane_v   = ctx->upscale_outputs[k].plane_v;
    }

    /* Upscale scratch buffer - one persistent row sized to the widest
     * source row any upscale helper will process.  Allocated once here
     * so the per-frame hot path does not malloc/free, which was causing
     * first-touch page faults and high max latency. */
    p->upscale_scratch  = NULL;
    p->upscale_scratch2 = NULL;
    if (want_up) {
        int max_scratch_w = 0;
        if (up_N >= 1) {
            /* Deepest 2x helper at level N-1 has input width eff_w << (N-1). */
            int dv = eff_w << (up_N - 1);
            if (dv > max_scratch_w) max_scratch_w = dv;
        }
        if (up_achieved_tail) {
            /* 1.5x tail reads either source (N==0) or the deepest pow2
             * output (width eff_w << N). */
            int tv = (up_N == 0) ? eff_w : (eff_w << up_N);
            if (tv > max_scratch_w) max_scratch_w = tv;
        }
        if (max_scratch_w > 0) {
            /* Two aligned rows from one allocation: row 0 is the
             * vertical-blend buffer, row 1 the two-pass interleave temp.
             * Each row is 64-byte aligned so both stay SIMD-friendly. */
            size_t row = (size_t)((max_scratch_w + 63) & ~63);
            void *sp = NULL;
            if (fused_init_alloc(&sp, 64, row * 2, planned) == 0) {
                p->upscale_scratch  = (uint8_t *)sp;
                p->upscale_scratch2 = sp ? (uint8_t *)sp + row : NULL;
            } else {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake: failed to allocate upscale scratch buffer\n");
            }
        }
    }

    /* Downscale scratch pool - pre-allocated at init to avoid per-frame
     * malloc in the kernel hot path.  Sized to the max bytes the selected
     * kernel family will need for its vertical and horizontal cascade
     * scratch buffers (plus generous 64-byte alignment padding between
     * sub-buffers). */
    p->scratch_pool      = NULL;
    p->scratch_pool_size = 0;
    if (want_down) {
        size_t pool_bytes = 0;
        if (family == FUSED_FAMILY_POW2) {
            /* Per kernels_scalar.c / kernels_neon.c / kernels_avx2.c:
             * vert_buf[k] = vert_rows[k] * src_w for k=0..deepest,
             * plus h_buf = src_w.
             * vert_rows[k] = group_rows >> (k+1) where group_rows = 2<<deepest.
             * Sum over k=0..deepest = group_rows - 1.
             * Deepest bit set in achieved determines deepest level. */
            int down_deepest = -1;
            for (int i = 3; i >= 0; i--) {
                if (achieved & (1u << (1 + 2 * i))) { down_deepest = i; break; }
            }
            if (down_deepest < 0) down_deepest = 0;
            int group_rows = 2 << down_deepest;
            size_t vert_total = (size_t)(group_rows - 1) * (size_t)eff_w;
            pool_bytes  = vert_total + (size_t)eff_w;
            pool_bytes += (size_t)(down_deepest + 2) * 64;  /* alignment slack */
        } else {
            /* Thirds: up to 8 full-width rows (v01,v23,v45, v3x_0,v3x_1, v6x,
             * v6x_prev, blend_tmp) + h_3x_buf (~eff_w/3) + h_6x_buf (~eff_w/6). */
            pool_bytes  = (size_t)eff_w * 8;
            pool_bytes += (size_t)((eff_w / 3) + (eff_w / 6 + 1));
            pool_bytes += 12 * 64;  /* alignment slack */
        }
        if (pool_bytes > 0) {
            void *sp = NULL;
            size_t aligned_bytes = (pool_bytes + 63) & ~(size_t)63;
            if (fused_init_alloc(&sp, 64, aligned_bytes, planned) == 0) {
                p->scratch_pool      = (uint8_t *)sp;
                p->scratch_pool_size = aligned_bytes;
            } else {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake: failed to allocate downscale scratch pool "
                    "(%zu bytes)\n", aligned_bytes);
            }
        }
    }

    /* Select kernel function based on (want_down, want_up) */
    if (has_simd) {
        if (want_down && want_up) {
            state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                   ? simd_thirds_up_fn : simd_pow2_up_fn;
        } else if (want_up) {
            state->kernel_fn = simd_upscale_fn;
        } else {
            const uint32_t deep_pow2 = (1u << 3) | (1u << 5) | (1u << 7);
            if (family == FUSED_FAMILY_POW2
                    && simd_pow2_chroma64_fn != NULL
                    && (achieved & deep_pow2) != 0
                    && ((eff_w / 2) & 127) == 64) {
                state->kernel_fn = simd_pow2_chroma64_fn;
            } else {
                state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                       ? simd_thirds_fn : simd_pow2_fn;
            }
        }
    } else {
        if (want_down && want_up) {
            state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                   ? fused_kernel_thirds_up_scalar
                                   : fused_kernel_pow2_up_scalar;
        } else if (want_up) {
            state->kernel_fn = fused_kernel_upscale_scalar;
        } else {
            state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                   ? fused_kernel_thirds_scalar
                                   : fused_kernel_pow2_scalar;
        }
    }
    state->has_simd = has_simd;

    ctx->_internal = planned ? NULL : state;

    return warn_bits;  /* 0 == FUSED_OK if nothing was warned */
}

/* --------------------------------------------------------------------------
 * fused_scaler_run
 * -------------------------------------------------------------------------- */

void fused_scaler_run(fused_scaler_ctx_t *ctx,
                      const uint8_t *src_y,
                      const uint8_t *src_u,
                      const uint8_t *src_v)
{
    if (!ctx) return;
    fused_internal_t *state = (fused_internal_t *)ctx->_internal;
    if (!state || !state->kernel_fn) return;

    /* Check source plane alignment */
    if (((uintptr_t)src_y & 31) || ((uintptr_t)src_u & 31) || ((uintptr_t)src_v & 31)) {
        /* Warn once about misaligned source planes */
        static atomic_int warned = 0;
        if (!atomic_exchange(&warned, 1)) {
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake: source planes are not 32-byte aligned "
                "(Y=%p U=%p V=%p). Falling back to scalar kernel. "
                "Performance will be significantly reduced.",
                (const void*)src_y, (const void*)src_u, (const void*)src_v);
        }
        /* Fall back to scalar - pick the variant matching the configured
         * (want_down, want_up) combination. */
        const fused_kernel_params_t *p = &state->params;
        int want_down = (p->active_outputs != 0);
        int want_up   = (p->upscale_active != 0);
        if (want_down && want_up) {
            if (p->family == FUSED_FAMILY_THIRDS)
                fused_kernel_thirds_up_scalar(p, src_y, src_u, src_v);
            else
                fused_kernel_pow2_up_scalar(p, src_y, src_u, src_v);
        } else if (want_up) {
            fused_kernel_upscale_scalar(p, src_y, src_u, src_v);
        } else if (p->family == FUSED_FAMILY_THIRDS) {
            fused_kernel_thirds_scalar(p, src_y, src_u, src_v);
        } else {
            fused_kernel_pow2_scalar(p, src_y, src_u, src_v);
        }
        return;
    }

    state->kernel_fn(&state->params, src_y, src_u, src_v);
}

/* --------------------------------------------------------------------------
 * fused_scaler_free
 * -------------------------------------------------------------------------- */

void fused_scaler_free(fused_scaler_ctx_t *ctx)
{
    if (!ctx) return;
    for (int i = 0; i < 8; i++) {
        free(ctx->outputs[i].plane_y);
        free(ctx->outputs[i].plane_u);
        free(ctx->outputs[i].plane_v);
        memset(&ctx->outputs[i], 0, sizeof(fused_scale_output_t));
    }
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) {
        free(ctx->upscale_outputs[i].plane_y);
        free(ctx->upscale_outputs[i].plane_u);
        free(ctx->upscale_outputs[i].plane_v);
        memset(&ctx->upscale_outputs[i], 0, sizeof(fused_scale_output_t));
    }
    if (ctx->_internal) {
        fused_internal_t *state = (fused_internal_t *)ctx->_internal;
        free(state->params.upscale_scratch);
        state->params.upscale_scratch = NULL;
        /* upscale_scratch2 aliases into the upscale_scratch allocation;
         * it must not be freed separately, only cleared. */
        state->params.upscale_scratch2 = NULL;
        free(state->params.scratch_pool);
        state->params.scratch_pool = NULL;
        state->params.scratch_pool_size = 0;
    }
    free(ctx->_internal);
    ctx->_internal              = NULL;
    ctx->achieved_flags         = 0;
    ctx->rejected_flags         = 0;
    ctx->achieved_upscale_flags = 0;
    ctx->achieved_upscale_tail  = 0;
}

int fused_scaler_init(fused_scaler_ctx_t *ctx)
{
    return fused_scaler_init_impl(ctx, NULL, NULL);
}

int fused_scaler_query(const fused_scaler_ctx_t *config, fused_scaler_ctx_t *layout,
                        size_t *memory_bytes)
{
    if (!config || !layout || !memory_bytes) return FUSED_ERR_BAD_DIMENSIONS;
    fused_scaler_ctx_t copy = *config;
    fused_internal_t state = {0};
    copy._internal = NULL;
    memset(copy.outputs, 0, sizeof(copy.outputs));
    memset(copy.upscale_outputs, 0, sizeof(copy.upscale_outputs));
    copy.log_errors.target = FUSED_LOG_SUPPRESS;
    copy.log_warnings.target = FUSED_LOG_SUPPRESS;
    size_t bytes = 0;
    int rc = fused_scaler_init_impl(&copy, &bytes, &state);
    if (rc >= 0) {
        *layout = copy;
        *memory_bytes = bytes;
    }
    return rc;
}
