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

int fused_scaler_init(fused_scaler_ctx_t *ctx)
{
    if (!ctx) return FUSED_ERR_BAD_DIMENSIONS;

    /* ------------------------------------------------------------------ */
    /* 1. Validate source dimensions                                        */
    /* ------------------------------------------------------------------ */

    if (ctx->src_width <= 0 || ctx->src_height <= 0 ||
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

    /* ------------------------------------------------------------------ */
    /* 2. Validate requested_flags                                          */
    /* ------------------------------------------------------------------ */

    uint32_t req = ctx->requested_flags;

    if (req == 0) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: requested_flags is zero\n");
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* Check for unknown bits (outside 0..7) */
    if (req & ~0xFFu) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: requested_flags 0x%08X contains unknown bits\n", req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    int has_thirds = (req & FUSED_SCALE_THIRDS_MASK) != 0;
    int has_pow2   = (req & FUSED_SCALE_POW2_MASK)   != 0;

    if (has_thirds && has_pow2) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: requested_flags 0x%02X mixes thirds and pow2 families\n", req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    int family = has_thirds ? FUSED_FAMILY_THIRDS : FUSED_FAMILY_POW2;

    /* ------------------------------------------------------------------ */
    /* 3. Compute effective dimensions (crop-to-fit)                        */
    /* ------------------------------------------------------------------ */

    int eff_w = ctx->src_width;
    int eff_h = ctx->src_height;

    if (!(ctx->options & FUSED_OPT_NO_CROP)) {
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
    fused_kernel_fn simd_thirds_fn = NULL;
    fused_kernel_fn simd_pow2_fn   = NULL;

#if defined(__aarch64__)
    if (caps->has_neon) {
        has_simd = 1;
        simd_thirds_fn = fused_kernel_thirds_neon;
        simd_pow2_fn   = fused_kernel_pow2_neon;
    }
#elif defined(__x86_64__)
    if (caps->has_avx2) {
        has_simd = 1;
        simd_thirds_fn = fused_kernel_thirds_avx2;
        simd_pow2_fn   = fused_kernel_pow2_avx2;
    }
#else
    (void)caps;
#endif

    if (!has_simd) {
        /* One-time stderr notice */
        static int g_no_simd_warned = 0;
        if (!g_no_simd_warned) {
            g_no_simd_warned = 1;
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
        int step_fallback = 0;

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
        if (posix_memalign(&py, 32, (size_t)y_stride  * (size_t)out_h)  != 0 ||
            posix_memalign(&pu, 32, (size_t)uv_stride * (size_t)chroma_h) != 0 ||
            posix_memalign(&pv, 32, (size_t)uv_stride * (size_t)chroma_h) != 0) {
            /* Allocation failure — free what we got and reject this step */
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
    /* 6. Check that at least one step was achieved                         */
    /* ------------------------------------------------------------------ */

    ctx->achieved_flags = achieved;
    ctx->rejected_flags = rejected;

    if (rejected) warn_bits |= FUSED_WARN_BIT_PARTIAL;

    if (achieved == 0) {
        /* Free any planes allocated before the first success (none, but be safe) */
        fused_scaler_free(ctx);
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake: no valid output steps after validation\n");
        return FUSED_ERR_NO_STEPS;
    }

    /* ------------------------------------------------------------------ */
    /* 7. Build kernel params and internal state                            */
    /* ------------------------------------------------------------------ */

    fused_internal_t *state = calloc(1, sizeof(fused_internal_t));
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

    /* Select kernel function */
    if (has_simd) {
        state->kernel_fn = (family == FUSED_FAMILY_THIRDS) ? simd_thirds_fn
                                                            : simd_pow2_fn;
    } else {
        state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                           ? fused_kernel_thirds_scalar
                           : fused_kernel_pow2_scalar;
    }
    state->has_simd = has_simd;

    ctx->_internal = state;

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
        static int warned = 0;
        if (!warned) {
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake: source planes are not 32-byte aligned "
                "(Y=%p U=%p V=%p). Falling back to scalar kernel. "
                "Performance will be significantly reduced.",
                (const void*)src_y, (const void*)src_u, (const void*)src_v);
            warned = 1;
        }
        /* Fall back to scalar */
        if (state->params.family == FUSED_FAMILY_THIRDS)
            fused_kernel_thirds_scalar(&state->params, src_y, src_u, src_v);
        else
            fused_kernel_pow2_scalar(&state->params, src_y, src_u, src_v);
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
    free(ctx->_internal);
    ctx->_internal      = NULL;
    ctx->achieved_flags = 0;
    ctx->rejected_flags = 0;
}
