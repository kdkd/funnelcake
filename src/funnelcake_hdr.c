/* --------------------------------------------------------------------------
 * funnelcake_hdr.c - HDR10 dispatch and initialization logic
 *
 * 10-bit analog of funnelcake.c. Handles validation, crop-to-fit, CPU
 * detection, kernel selection, output buffer allocation, tone mapping LUT
 * generation, and per-frame dispatch for HDR10/HLG inputs.
 *
 * The scaling kernel produces 10-bit planar outputs. An optional tone
 * mapping pass converts those to 8-bit SDR using precomputed LUTs.
 * -------------------------------------------------------------------------- */

#include "funnelcake.h"
#include "internal.h"
#include "detect.h"
#include "log.h"
#include "tonemap.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * Step descriptor table (duplicated locally from funnelcake.c)
 * -------------------------------------------------------------------------- */

typedef struct {
    uint32_t    flag;
    int         bit;
    int         ratio_n;
    int         ratio_d;
    const char *name;
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

static int round_down(int x, int m) {
    return (x / m) * m;
}

/* 8-bit stride: round up width (in bytes) to 32-byte boundary */
static int stride_for(int width) {
    return (width + 31) & ~31;
}

/* 10-bit stride: round up width*2 (bytes per row of uint16_t) to 32-byte boundary */
static int stride_for_hdr(int width) {
    return (width * 2 + 31) & ~31;
}

/* --------------------------------------------------------------------------
 * fused_hdr_init
 * -------------------------------------------------------------------------- */

int fused_hdr_init(fused_hdr_ctx_t *ctx)
{
    if (!ctx) return FUSED_ERR_BAD_DIMENSIONS;

    /* ------------------------------------------------------------------ */
    /* 1. Validate source dimensions                                        */
    /* ------------------------------------------------------------------ */

    if (ctx->src_width <= 0 || ctx->src_height <= 0 ||
        (ctx->src_width & 1) || (ctx->src_height & 1)) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: src_width=%d src_height=%d must be positive and even\n",
            ctx->src_width, ctx->src_height);
        return FUSED_ERR_BAD_DIMENSIONS;
    }

    /* Y plane stride: must hold width * sizeof(uint16_t) and be 32-byte aligned */
    int min_y_stride = ctx->src_width * 2;
    if (ctx->src_y_stride < min_y_stride ||
        (ctx->src_y_stride & 31)) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: src_y_stride=%d must be >= %d and a multiple of 32\n",
            ctx->src_y_stride, min_y_stride);
        return FUSED_ERR_BAD_ALIGNMENT;
    }

    /* UV stride validation depends on format:
     *   P010/P210: interleaved UV, stride must hold chroma_w * 2 * sizeof(uint16_t)
     *   I010/I210: separate U and V planes, stride must hold chroma_w * sizeof(uint16_t) */
    {
        int chroma_w = ctx->src_width / 2;
        int min_uv_stride;
        if (ctx->src_format == FUSED_PIX_P010 || ctx->src_format == FUSED_PIX_P210) {
            min_uv_stride = chroma_w * 2 * 2;  /* interleaved UV pairs, 2 bytes each */
        } else {
            min_uv_stride = chroma_w * 2;       /* single plane, 2 bytes per sample */
        }
        if (ctx->src_uv_stride < min_uv_stride ||
            (ctx->src_uv_stride & 31)) {
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake-hdr: src_uv_stride=%d must be >= %d and a multiple of 32\n",
                ctx->src_uv_stride, min_uv_stride);
            return FUSED_ERR_BAD_ALIGNMENT;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 2. Validate format and transfer                                      */
    /* ------------------------------------------------------------------ */

    if (ctx->src_format < FUSED_PIX_I010 || ctx->src_format > FUSED_PIX_P210) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: src_format=%d is invalid (must be 0..3)\n",
            ctx->src_format);
        return FUSED_ERR_INVALID_FLAGS;
    }

    if (ctx->src_transfer != FUSED_TRC_PQ && ctx->src_transfer != FUSED_TRC_HLG) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: src_transfer=%d is invalid (must be 0 or 1)\n",
            ctx->src_transfer);
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* hdr_flags and sdr_flags must be subsets of requested_flags */
    if (ctx->hdr_flags & ~ctx->requested_flags) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: hdr_flags 0x%02X contains bits not in requested_flags 0x%02X\n",
            ctx->hdr_flags, ctx->requested_flags);
        return FUSED_ERR_INVALID_FLAGS;
    }
    if (ctx->sdr_flags & ~ctx->requested_flags) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: sdr_flags 0x%02X contains bits not in requested_flags 0x%02X\n",
            ctx->sdr_flags, ctx->requested_flags);
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* At least one output type must be requested.  Upscale-only is valid:
     * relax this check if upscale_flags or upscale_tail_1_5x are set. */
    if ((ctx->hdr_flags | ctx->sdr_flags) == 0 && !ctx->tonemap_1x &&
        ctx->upscale_flags == 0 && !ctx->upscale_tail_1_5x) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: hdr_flags | sdr_flags is zero and tonemap_1x is not set;"
            " need at least one output\n");
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* Custom tonemap requires a LUT */
    if (ctx->tonemap.curve == FUSED_TONEMAP_CUSTOM && ctx->tonemap.custom_lut == NULL) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: FUSED_TONEMAP_CUSTOM requires non-NULL custom_lut\n");
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* ------------------------------------------------------------------ */
    /* 3. Validate requested_flags and upscale_flags                        */
    /* ------------------------------------------------------------------ */

    uint32_t req     = ctx->requested_flags;
    uint32_t up_req  = ctx->upscale_flags;
    int      up_tail = ctx->upscale_tail_1_5x ? 1 : 0;

    if (req == 0 && !ctx->tonemap_1x && up_req == 0 && !up_tail) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: requested_flags, upscale_flags, and tonemap_1x "
            "are all zero\n");
        return FUSED_ERR_INVALID_FLAGS;
    }

    if (req & ~0xFFu) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: requested_flags 0x%08X contains unknown bits\n", req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    if (up_req & ~FUSED_UPSCALE_POW2_MASK) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: upscale_flags 0x%08X contains unknown bits\n", up_req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    /* Upscale cascade must be a contiguous prefix. */
    {
        uint32_t tmp = up_req;
        while (tmp & 1) tmp >>= 1;
        if (tmp != 0) {
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake-hdr: upscale_flags 0x%02X is not a contiguous "
                "prefix\n", up_req);
            return FUSED_ERR_INVALID_FLAGS;
        }
    }

    int has_thirds = (req & FUSED_SCALE_THIRDS_MASK) != 0;
    int has_pow2   = (req & FUSED_SCALE_POW2_MASK)   != 0;

    if (has_thirds && has_pow2) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: requested_flags 0x%02X mixes thirds and pow2 families\n", req);
        return FUSED_ERR_INVALID_FLAGS;
    }

    int family;
    if (has_thirds) family = FUSED_FAMILY_THIRDS;
    else if (has_pow2) family = FUSED_FAMILY_POW2;
    else family = FUSED_FAMILY_POW2;  /* arbitrary - upscale-only or tonemap_1x only */

    /* ------------------------------------------------------------------ */
    /* 4. Crop-to-fit                                                       */
    /* ------------------------------------------------------------------ */

    int eff_w = ctx->src_width;
    int eff_h = ctx->src_height;

    if (!(ctx->options & FUSED_OPT_NO_CROP) && req != 0) {
        if (family == FUSED_FAMILY_THIRDS) {
            int w_div = 3;
            if (req & FUSED_SCALE_3X)  w_div = 6;
            if (req & FUSED_SCALE_6X)  w_div = 12;
            if (req & FUSED_SCALE_12X) w_div = 24;
            eff_w = round_down(eff_w, w_div);

            int h_div = 6;
            if (req & FUSED_SCALE_3X)  { if (h_div < 6)  h_div = 6;  }
            if (req & FUSED_SCALE_6X)  { if (h_div < 12) h_div = 12; }
            if (req & FUSED_SCALE_12X) { if (h_div < 24) h_div = 24; }
            eff_h = round_down(eff_h, h_div);
        } else {
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
            "funnelcake-hdr: after crop-to-fit, effective dimensions %dx%d are non-positive\n",
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
    /* 5. CPU detection & kernel selection                                  */
    /* ------------------------------------------------------------------ */

    const fused_cpu_caps_t *caps = fused_detect_cpu();

    int has_simd = 0;
    fused_hdr_kernel_fn simd_thirds_fn    = NULL;
    fused_hdr_kernel_fn simd_pow2_fn      = NULL;
    fused_hdr_kernel_fn simd_upscale_fn   = NULL;
    fused_hdr_kernel_fn simd_thirds_up_fn = NULL;
    fused_hdr_kernel_fn simd_pow2_up_fn   = NULL;

#if defined(__aarch64__)
    if (caps->has_neon) {
        has_simd = 1;
        simd_thirds_fn    = fused_kernel_thirds_hdr_neon;
        simd_pow2_fn      = fused_kernel_pow2_hdr_neon;
        simd_upscale_fn   = fused_kernel_upscale_hdr_neon;
        simd_thirds_up_fn = fused_kernel_thirds_up_hdr_neon;
        simd_pow2_up_fn   = fused_kernel_pow2_up_hdr_neon;
    }
#elif defined(__x86_64__)
    if (caps->has_avx2) {
        has_simd = 1;
        simd_thirds_fn    = fused_kernel_thirds_hdr_avx2;
        simd_pow2_fn      = fused_kernel_pow2_hdr_avx2;
        simd_upscale_fn   = fused_kernel_upscale_hdr_avx2;
        simd_thirds_up_fn = fused_kernel_thirds_up_hdr_avx2;
        simd_pow2_up_fn   = fused_kernel_pow2_up_hdr_avx2;
    }
#else
    (void)caps;
#endif

    if (!has_simd) {
        static int g_no_simd_warned = 0;
        if (!g_no_simd_warned) {
            g_no_simd_warned = 1;
            fprintf(stderr,
                "funnelcake-hdr: no SIMD support detected; using scalar kernel\n");
        }
        warn_bits |= FUSED_WARN_BIT_SCALAR;
    }

    /* ------------------------------------------------------------------ */
    /* 6. Validate each step and allocate output buffers                    */
    /* ------------------------------------------------------------------ */

    uint32_t achieved_hdr = 0;
    uint32_t achieved_sdr = 0;
    uint32_t rejected     = 0;

    /* Zero all output slots */
    memset(ctx->hdr_outputs, 0, sizeof(ctx->hdr_outputs));
    memset(ctx->sdr_outputs, 0, sizeof(ctx->sdr_outputs));

    /* Allocate internal state early so we can store sdr_temp pointers */
    fused_hdr_internal_t *state = calloc(1, sizeof(fused_hdr_internal_t));
    if (!state) {
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: out-of-memory allocating internal state\n");
        return FUSED_ERR_NO_STEPS;
    }

    for (int i = 0; i < 8; i++) {
        const step_desc_t *sd = &k_steps[i];

        if (!(req & sd->flag)) continue;

        /* Compute output dimensions */
        int out_w = eff_w * sd->ratio_n / sd->ratio_d;
        int out_h = eff_h * sd->ratio_n / sd->ratio_d;

        /* (a) Exact integer division check */
        if (out_w * sd->ratio_d != eff_w * sd->ratio_n ||
            out_h * sd->ratio_d != eff_h * sd->ratio_n) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: %s rejected: output dimensions %dx%d not exact"
                " (source %dx%d ratio %d/%d)\n",
                sd->name, out_w, out_h, eff_w, eff_h, sd->ratio_n, sd->ratio_d);
            rejected |= sd->flag;
            continue;
        }

        /* (b) Output luma dimensions must be even */
        if ((out_w & 1) || (out_h & 1)) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: %s rejected: output luma %dx%d has odd dimension\n",
                sd->name, out_w, out_h);
            rejected |= sd->flag;
            continue;
        }

        /* (c) Minimum size */
        if (out_w < 32 || out_h < 2) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: %s rejected: output %dx%d is too small (min 32x2)\n",
                sd->name, out_w, out_h);
            rejected |= sd->flag;
            continue;
        }

        /* (d) SIMD chroma width constraint */
        int chroma_w = out_w / 2;
        int step_fallback = 0;

        if (has_simd && (chroma_w & 31)) {
            if (ctx->options & FUSED_OPT_NO_FALLBACK) {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake-hdr: %s rejected: chroma output width %d is not a"
                    " multiple of 32 (source %dx%d -> %dx scale luma %d -> chroma %d)\n",
                    sd->name, chroma_w,
                    eff_w, eff_h, sd->ratio_d, out_w, chroma_w);
                rejected |= sd->flag;
                continue;
            } else {
                step_fallback = 1;
                warn_bits |= FUSED_WARN_BIT_SCALAR;
            }
        }

        int step_wants_hdr = (ctx->hdr_flags & sd->flag) != 0;
        int step_wants_sdr = (ctx->sdr_flags & sd->flag) != 0;

        int chroma_h = out_h / 2;

        /* Allocate HDR (10-bit) output if requested */
        if (step_wants_hdr) {
            int y_stride  = stride_for_hdr(out_w);
            int uv_stride = stride_for_hdr(chroma_w);

            void *py = NULL, *pu = NULL, *pv = NULL;
            if (posix_memalign(&py, 32, (size_t)y_stride  * (size_t)out_h)     != 0 ||
                posix_memalign(&pu, 32, (size_t)uv_stride * (size_t)chroma_h)  != 0 ||
                posix_memalign(&pv, 32, (size_t)uv_stride * (size_t)chroma_h)  != 0) {
                free(py); free(pu); free(pv);
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake-hdr: %s rejected: out-of-memory allocating HDR output planes\n",
                    sd->name);
                rejected |= sd->flag;
                continue;
            }

            ctx->hdr_outputs[i].width     = out_w;
            ctx->hdr_outputs[i].height    = out_h;
            ctx->hdr_outputs[i].y_stride  = y_stride;
            ctx->hdr_outputs[i].uv_stride = uv_stride;
            ctx->hdr_outputs[i].plane_y   = (uint16_t *)py;
            ctx->hdr_outputs[i].plane_u   = (uint16_t *)pu;
            ctx->hdr_outputs[i].plane_v   = (uint16_t *)pv;
            ctx->hdr_outputs[i].fallback  = step_fallback;

            achieved_hdr |= sd->flag;
        }

        /* Allocate SDR (8-bit) output if requested */
        if (step_wants_sdr) {
            int y_stride  = stride_for(out_w);
            int uv_stride = stride_for(chroma_w);

            void *py = NULL, *pu = NULL, *pv = NULL;
            if (posix_memalign(&py, 32, (size_t)y_stride  * (size_t)out_h)     != 0 ||
                posix_memalign(&pu, 32, (size_t)uv_stride * (size_t)chroma_h)  != 0 ||
                posix_memalign(&pv, 32, (size_t)uv_stride * (size_t)chroma_h)  != 0) {
                free(py); free(pu); free(pv);
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake-hdr: %s rejected: out-of-memory allocating SDR output planes\n",
                    sd->name);
                /* If we already allocated HDR planes for this step, free them */
                if (step_wants_hdr && (achieved_hdr & sd->flag)) {
                    free(ctx->hdr_outputs[i].plane_y);
                    free(ctx->hdr_outputs[i].plane_u);
                    free(ctx->hdr_outputs[i].plane_v);
                    memset(&ctx->hdr_outputs[i], 0, sizeof(fused_hdr_output_t));
                    achieved_hdr &= ~sd->flag;
                }
                rejected |= sd->flag;
                continue;
            }

            ctx->sdr_outputs[i].width     = out_w;
            ctx->sdr_outputs[i].height    = out_h;
            ctx->sdr_outputs[i].y_stride  = y_stride;
            ctx->sdr_outputs[i].uv_stride = uv_stride;
            ctx->sdr_outputs[i].plane_y   = (uint8_t *)py;
            ctx->sdr_outputs[i].plane_u   = (uint8_t *)pu;
            ctx->sdr_outputs[i].plane_v   = (uint8_t *)pv;
            ctx->sdr_outputs[i].fallback  = step_fallback;

            achieved_sdr |= sd->flag;

            /* If SDR is requested but HDR is not, allocate temp 10-bit buffers
             * for the intermediate scaled result before tone mapping */
            if (!step_wants_hdr) {
                int hdr_y_stride  = stride_for_hdr(out_w);
                int hdr_uv_stride = stride_for_hdr(chroma_w);

                void *ty = NULL, *tu = NULL, *tv = NULL;
                if (posix_memalign(&ty, 32, (size_t)hdr_y_stride  * (size_t)out_h)    != 0 ||
                    posix_memalign(&tu, 32, (size_t)hdr_uv_stride * (size_t)chroma_h) != 0 ||
                    posix_memalign(&tv, 32, (size_t)hdr_uv_stride * (size_t)chroma_h) != 0) {
                    free(ty); free(tu); free(tv);
                    /* Roll back SDR allocation for this step */
                    free(ctx->sdr_outputs[i].plane_y);
                    free(ctx->sdr_outputs[i].plane_u);
                    free(ctx->sdr_outputs[i].plane_v);
                    memset(&ctx->sdr_outputs[i], 0, sizeof(fused_scale_output_t));
                    achieved_sdr &= ~sd->flag;
                    fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                        "funnelcake-hdr: %s rejected: out-of-memory allocating temp 10-bit buffers\n",
                        sd->name);
                    rejected |= sd->flag;
                    continue;
                }

                state->sdr_temp[i].y = (uint16_t *)ty;
                state->sdr_temp[i].u = (uint16_t *)tu;
                state->sdr_temp[i].v = (uint16_t *)tv;
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6b. Validate upscale levels and allocate HDR upscale buffers         */
    /* ------------------------------------------------------------------ */

    #define FUSED_HDR_UPSCALE_SIZE_CAP 16384

    uint32_t up_achieved_hdr = 0;
    int      up_achieved_tail = 0;

    memset(ctx->upscale_hdr_outputs, 0, sizeof(ctx->upscale_hdr_outputs));

    int up_N = 0;
    {
        uint32_t tmp = up_req;
        while (tmp & 1) { up_N++; tmp >>= 1; }
    }

    for (int k = 0; k < up_N; k++) {
        int up_w = eff_w << (k + 1);
        int up_h = eff_h << (k + 1);

        if (up_w > FUSED_HDR_UPSCALE_SIZE_CAP || up_h > FUSED_HDR_UPSCALE_SIZE_CAP) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: upscale level %dx rejected: output %dx%d "
                "exceeds size cap %d\n", (1 << (k + 1)), up_w, up_h,
                FUSED_HDR_UPSCALE_SIZE_CAP);
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            continue;
        }

        int uv_w = up_w / 2;
        int y_stride  = stride_for_hdr(up_w);
        int uv_stride = stride_for_hdr(uv_w);
        int chroma_h  = up_h / 2;

        void *py = NULL, *pu = NULL, *pv = NULL;
        if (posix_memalign(&py, 32, (size_t)y_stride  * (size_t)up_h)   != 0 ||
            posix_memalign(&pu, 32, (size_t)uv_stride * (size_t)chroma_h) != 0 ||
            posix_memalign(&pv, 32, (size_t)uv_stride * (size_t)chroma_h) != 0) {
            free(py); free(pu); free(pv);
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: upscale level %dx rejected: out-of-memory\n",
                (1 << (k + 1)));
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            continue;
        }

        ctx->upscale_hdr_outputs[k].width     = up_w;
        ctx->upscale_hdr_outputs[k].height    = up_h;
        ctx->upscale_hdr_outputs[k].y_stride  = y_stride;
        ctx->upscale_hdr_outputs[k].uv_stride = uv_stride;
        ctx->upscale_hdr_outputs[k].plane_y   = (uint16_t *)py;
        ctx->upscale_hdr_outputs[k].plane_u   = (uint16_t *)pu;
        ctx->upscale_hdr_outputs[k].plane_v   = (uint16_t *)pv;
        ctx->upscale_hdr_outputs[k].fallback  = !has_simd;

        up_achieved_hdr |= (1u << k);
    }

    if (up_tail) {
        int tail_src_w, tail_src_h;
        if (up_N == 0) {
            tail_src_w = eff_w;
            tail_src_h = eff_h;
        } else {
            if (!(up_achieved_hdr & (1u << (up_N - 1)))) {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake-hdr: upscale 1.5x tail rejected: deepest pow2 "
                    "level %dx was not achieved\n", (1 << up_N));
                warn_bits |= FUSED_WARN_BIT_PARTIAL;
                goto hdr_tail_done;
            }
            tail_src_w = eff_w << up_N;
            tail_src_h = eff_h << up_N;
        }

        int tail_w = tail_src_w * 3 / 2;
        int tail_h = tail_src_h * 3 / 2;

        if (tail_w > FUSED_HDR_UPSCALE_SIZE_CAP ||
            tail_h > FUSED_HDR_UPSCALE_SIZE_CAP) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: upscale 1.5x tail rejected: output %dx%d "
                "exceeds size cap %d\n", tail_w, tail_h,
                FUSED_HDR_UPSCALE_SIZE_CAP);
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            goto hdr_tail_done;
        }

        if ((tail_w & 1) || (tail_h & 1)) {
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: upscale 1.5x tail rejected: output %dx%d has "
                "odd dimension\n", tail_w, tail_h);
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            goto hdr_tail_done;
        }

        int tail_uv_w   = tail_w / 2;
        int y_stride    = stride_for_hdr(tail_w);
        int uv_stride   = stride_for_hdr(tail_uv_w);
        int chroma_h    = tail_h / 2;

        void *py = NULL, *pu = NULL, *pv = NULL;
        if (posix_memalign(&py, 32, (size_t)y_stride  * (size_t)tail_h)   != 0 ||
            posix_memalign(&pu, 32, (size_t)uv_stride * (size_t)chroma_h) != 0 ||
            posix_memalign(&pv, 32, (size_t)uv_stride * (size_t)chroma_h) != 0) {
            free(py); free(pu); free(pv);
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: upscale 1.5x tail rejected: out-of-memory\n");
            warn_bits |= FUSED_WARN_BIT_PARTIAL;
            goto hdr_tail_done;
        }

        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].width     = tail_w;
        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].height    = tail_h;
        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].y_stride  = y_stride;
        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].uv_stride = uv_stride;
        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].plane_y   = (uint16_t *)py;
        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].plane_u   = (uint16_t *)pu;
        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].plane_v   = (uint16_t *)pv;
        ctx->upscale_hdr_outputs[FUSED_UP_IDX_TAIL].fallback  = !has_simd;

        up_achieved_tail = 1;
    }
hdr_tail_done:

    ctx->achieved_upscale_flags = up_achieved_hdr;
    ctx->achieved_upscale_tail  = up_achieved_tail;

    /* ------------------------------------------------------------------ */
    /* Check that at least one step was achieved (or tonemap_1x is set)     */
    /* ------------------------------------------------------------------ */

    uint32_t achieved_any = achieved_hdr | achieved_sdr;

    ctx->achieved_hdr_flags = achieved_hdr;
    ctx->achieved_sdr_flags = achieved_sdr;
    ctx->rejected_flags     = rejected;

    if (rejected) warn_bits |= FUSED_WARN_BIT_PARTIAL;

    int hdr_want_up = (up_achieved_hdr != 0) || up_achieved_tail;

    if (achieved_any == 0 && !ctx->tonemap_1x && !hdr_want_up) {
        fused_hdr_free(ctx);
        free(state);
        ctx->_internal = NULL;
        fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
            "funnelcake-hdr: no valid output steps after validation\n");
        return FUSED_ERR_NO_STEPS;
    }

    /* ------------------------------------------------------------------ */
    /* 7. Allocate 1:1 tonemap output (if tonemap_1x)                       */
    /* ------------------------------------------------------------------ */

    memset(&ctx->output_1x, 0, sizeof(fused_scale_output_t));

    if (ctx->tonemap_1x) {
        int y_stride  = stride_for(eff_w);
        int uv_stride = stride_for(eff_w / 2);
        int chroma_h  = eff_h / 2;

        void *py = NULL, *pu = NULL, *pv = NULL;
        if (posix_memalign(&py, 32, (size_t)y_stride  * (size_t)eff_h)    != 0 ||
            posix_memalign(&pu, 32, (size_t)uv_stride * (size_t)chroma_h) != 0 ||
            posix_memalign(&pv, 32, (size_t)uv_stride * (size_t)chroma_h) != 0) {
            free(py); free(pu); free(pv);
            fused_hdr_free(ctx);
            free(state);
            ctx->_internal = NULL;
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake-hdr: out-of-memory allocating 1:1 tonemap output\n");
            return FUSED_ERR_NO_STEPS;
        }

        ctx->output_1x.width     = eff_w;
        ctx->output_1x.height    = eff_h;
        ctx->output_1x.y_stride  = y_stride;
        ctx->output_1x.uv_stride = uv_stride;
        ctx->output_1x.plane_y   = (uint8_t *)py;
        ctx->output_1x.plane_u   = (uint8_t *)pu;
        ctx->output_1x.plane_v   = (uint8_t *)pv;
        ctx->output_1x.fallback  = 0;
    }

    /* ------------------------------------------------------------------ */
    /* 8. Generate tone mapping LUTs                                        */
    /* ------------------------------------------------------------------ */

    if (achieved_sdr || ctx->tonemap_1x) {
        fused_tonemap_generate_luts(state, ctx->src_transfer,
                                    &ctx->tonemap, &ctx->log_warnings);
    }

    /* ------------------------------------------------------------------ */
    /* 9. Build kernel params                                               */
    /* ------------------------------------------------------------------ */

    fused_hdr_kernel_params_t *p = &state->params;

    p->src_width    = eff_w;
    p->src_height   = eff_h;
    p->src_y_stride = ctx->src_y_stride;

    /* For 4:2:2 (I210/P210): chroma has a row for every luma row, but
     * the kernel processes in row pairs (4:2:0 output), so double the
     * stride to skip every other chroma row.
     * For 4:2:0 (I010/P010): use the stride as-is. */
    if (ctx->src_format == FUSED_PIX_I210 || ctx->src_format == FUSED_PIX_P210) {
        p->src_uv_stride = ctx->src_uv_stride * 2;
    } else {
        p->src_uv_stride = ctx->src_uv_stride;
    }

    p->family = family;

    int cascade_depth = 0;
    for (int i = 0; i < 8; i++) {
        if (achieved_any & (1u << i)) cascade_depth++;
    }
    p->cascade_depth = cascade_depth;

    p->vert_period = (family == FUSED_FAMILY_THIRDS) ? 3 : 2;

    /* The kernel produces 10-bit output for all needed steps */
    p->active_outputs = achieved_any;

    p->is_p010 = (ctx->src_format == FUSED_PIX_P010 ||
                  ctx->src_format == FUSED_PIX_P210) ? 1 : 0;

    /* Pre-allocate P010 deinterleave buffers so the kernel entry points
     * don't have to malloc/free them on every fused_hdr_run() call. */
    p->p010_tmp_u = NULL;
    p->p010_tmp_v = NULL;
    p->p010_tmp_stride = 0;
    if (p->is_p010) {
        int chroma_w = eff_w / 2;
        int chroma_h = eff_h / 2;
        int tmp_stride = stride_for_hdr(chroma_w);  /* 32-byte aligned */
        size_t tmp_bytes = (size_t)tmp_stride * (size_t)chroma_h;

        if (posix_memalign((void **)&p->p010_tmp_u, 32, tmp_bytes) != 0 ||
            posix_memalign((void **)&p->p010_tmp_v, 32, tmp_bytes) != 0) {
            free(p->p010_tmp_u);
            free(p->p010_tmp_v);
            p->p010_tmp_u = NULL;
            p->p010_tmp_v = NULL;
            fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                "funnelcake-hdr: failed to allocate P010 deinterleave buffers\n");
            /* Fall through - kernels will malloc per-frame as fallback */
        } else {
            p->p010_tmp_stride = tmp_stride;
        }
    }

    /* Fill per-output geometry and destination pointers.
     * For steps in hdr_flags: point at hdr_outputs (kernel writes there).
     * For steps in both hdr_flags and sdr_flags: also point at hdr_outputs
     *   (tone mapper reads from hdr_outputs later).
     * For steps in sdr_flags only: point at sdr_temp (kernel writes there,
     *   tone mapper reads from there). */
    for (int i = 0; i < 8; i++) {
        if (!(achieved_any & (1u << i))) continue;

        int out_w = 0, out_h = 0;

        if (achieved_hdr & (1u << i)) {
            out_w = ctx->hdr_outputs[i].width;
            out_h = ctx->hdr_outputs[i].height;
        } else if (achieved_sdr & (1u << i)) {
            out_w = ctx->sdr_outputs[i].width;
            out_h = ctx->sdr_outputs[i].height;
        }

        int chroma_w = out_w / 2;

        p->out[i].width    = out_w;
        p->out[i].height   = out_h;
        p->out[i].y_stride  = stride_for_hdr(out_w);
        p->out[i].uv_stride = stride_for_hdr(chroma_w);

        if (achieved_hdr & (1u << i)) {
            /* HDR output exists - kernel writes to hdr_outputs */
            p->out[i].plane_y = ctx->hdr_outputs[i].plane_y;
            p->out[i].plane_u = ctx->hdr_outputs[i].plane_u;
            p->out[i].plane_v = ctx->hdr_outputs[i].plane_v;
        } else {
            /* SDR only - kernel writes to temp 10-bit buffers */
            p->out[i].plane_y = state->sdr_temp[i].y;
            p->out[i].plane_u = state->sdr_temp[i].u;
            p->out[i].plane_v = state->sdr_temp[i].v;
        }
    }

    /* Precomputed loop counts: 16 uint16_t elements per YMM register */
    p->chunks_per_row = eff_w / 16;
    p->tail_elements  = eff_w % 16;
    p->row_groups     = eff_h / p->vert_period;

    /* Pre-computed element strides avoid a division per kernel call */
    p->src_y_el_stride  = p->src_y_stride / (int)sizeof(uint16_t);
    p->src_uv_el_stride = p->src_uv_stride / (int)sizeof(uint16_t);

    /* Upscale params */
    p->upscale_cascade_depth = up_N;
    p->upscale_tail_1_5x     = up_achieved_tail;
    p->upscale_hdr_active    = up_achieved_hdr | (up_achieved_tail
                                                  ? (1u << FUSED_UP_IDX_TAIL)
                                                  : 0u);

    for (int k = 0; k < FUSED_MAX_UPSCALE_STEPS; k++) {
        if (!(p->upscale_hdr_active & (1u << k))) continue;
        p->hdr_up_out[k].width     = ctx->upscale_hdr_outputs[k].width;
        p->hdr_up_out[k].height    = ctx->upscale_hdr_outputs[k].height;
        p->hdr_up_out[k].y_stride  = ctx->upscale_hdr_outputs[k].y_stride;
        p->hdr_up_out[k].uv_stride = ctx->upscale_hdr_outputs[k].uv_stride;
        p->hdr_up_out[k].plane_y   = ctx->upscale_hdr_outputs[k].plane_y;
        p->hdr_up_out[k].plane_u   = ctx->upscale_hdr_outputs[k].plane_u;
        p->hdr_up_out[k].plane_v   = ctx->upscale_hdr_outputs[k].plane_v;
    }

    /* Select kernel function based on (want_down, want_up). */
    int want_down = (achieved_any != 0);
    int want_up   = (p->upscale_hdr_active != 0);

    /* Upscale scratch row buffer (HDR, u16).  One-time allocation to
     * keep the per-frame hot path free of malloc/free. */
    p->upscale_scratch_hdr = NULL;
    if (want_up) {
        int max_scratch_w = 0;
        if (up_N >= 1) {
            int dv = eff_w << (up_N - 1);
            if (dv > max_scratch_w) max_scratch_w = dv;
        }
        if (up_achieved_tail) {
            int tv = (up_N == 0) ? eff_w : (eff_w << up_N);
            if (tv > max_scratch_w) max_scratch_w = tv;
        }
        if (max_scratch_w > 0) {
            size_t bytes = (size_t)((max_scratch_w + 63) & ~63) * sizeof(uint16_t);
            void *sp = NULL;
            if (posix_memalign(&sp, 64, bytes) == 0) {
                p->upscale_scratch_hdr = (uint16_t *)sp;
            } else {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake-hdr: failed to allocate upscale scratch buffer\n");
            }
        }
    }

    /* HDR downscale scratch pool - parallel of the SDR path.  Buffers
     * are uint16_t so total byte size is doubled. */
    p->scratch_pool      = NULL;
    p->scratch_pool_size = 0;
    if (want_down) {
        size_t pool_bytes = 0;
        if (family == FUSED_FAMILY_POW2) {
            int down_deepest = -1;
            for (int i = 3; i >= 0; i--) {
                if (achieved_any & (1u << (1 + 2 * i))) { down_deepest = i; break; }
            }
            if (down_deepest < 0) down_deepest = 0;
            int group_rows = 2 << down_deepest;
            size_t vert_total = (size_t)(group_rows - 1) * (size_t)eff_w * sizeof(uint16_t);
            pool_bytes  = vert_total + (size_t)eff_w * sizeof(uint16_t);
            pool_bytes += (size_t)(down_deepest + 2) * 64;
        } else {
            pool_bytes  = (size_t)eff_w * 8 * sizeof(uint16_t);
            pool_bytes += (size_t)((eff_w / 3) + (eff_w / 6 + 1)) * sizeof(uint16_t);
            pool_bytes += 12 * 64;
        }
        if (pool_bytes > 0) {
            void *sp = NULL;
            size_t aligned_bytes = (pool_bytes + 63) & ~(size_t)63;
            if (posix_memalign(&sp, 64, aligned_bytes) == 0) {
                p->scratch_pool      = (uint8_t *)sp;
                p->scratch_pool_size = aligned_bytes;
            } else {
                fused_log(&ctx->log_warnings, FUSED_LOG_WARN,
                    "funnelcake-hdr: failed to allocate downscale scratch pool\n");
            }
        }
    }

    if (has_simd) {
        if (want_down && want_up) {
            state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                   ? simd_thirds_up_fn : simd_pow2_up_fn;
        } else if (want_up) {
            state->kernel_fn = simd_upscale_fn;
        } else {
            state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                   ? simd_thirds_fn : simd_pow2_fn;
        }
    } else {
        if (want_down && want_up) {
            state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                   ? fused_kernel_thirds_up_hdr_scalar
                                   : fused_kernel_pow2_up_hdr_scalar;
        } else if (want_up) {
            state->kernel_fn = fused_kernel_upscale_hdr_scalar;
        } else {
            state->kernel_fn = (family == FUSED_FAMILY_THIRDS)
                                   ? fused_kernel_thirds_hdr_scalar
                                   : fused_kernel_pow2_hdr_scalar;
        }
    }
    state->has_simd   = has_simd;
    state->sdr_flags  = achieved_sdr;
    state->tonemap_1x = ctx->tonemap_1x;
    state->is_custom_lut = (ctx->tonemap.curve == FUSED_TONEMAP_CUSTOM) ? 1 : 0;

    ctx->_internal = state;

    return warn_bits;  /* 0 == FUSED_OK if nothing was warned */
}

/* --------------------------------------------------------------------------
 * fused_hdr_run
 * -------------------------------------------------------------------------- */

void fused_hdr_run(fused_hdr_ctx_t *ctx,
                   const uint16_t *src_y,
                   const uint16_t *src_u,
                   const uint16_t *src_v)
{
    if (!ctx) return;
    fused_hdr_internal_t *state = (fused_hdr_internal_t *)ctx->_internal;
    if (!state || !state->kernel_fn) return;

    /* ------------------------------------------------------------------ */
    /* 1. Check alignment - fall back to scalar if misaligned               */
    /* ------------------------------------------------------------------ */

    int is_p010 = state->params.is_p010;

    if (((uintptr_t)src_y & 31) || ((uintptr_t)src_u & 31) ||
        (!is_p010 && src_v && ((uintptr_t)src_v & 31))) {
        static int warned = 0;
        if (!warned) {
            fused_log(&ctx->log_errors, FUSED_LOG_ERROR,
                "funnelcake-hdr: source planes are not 32-byte aligned "
                "(Y=%p U=%p V=%p). Falling back to scalar kernel. "
                "Performance will be significantly reduced.",
                (const void *)src_y, (const void *)src_u, (const void *)src_v);
            warned = 1;
        }
        const fused_hdr_kernel_params_t *p = &state->params;
        int hdr_want_down = (p->active_outputs != 0);
        int hdr_want_up   = (p->upscale_hdr_active != 0);
        if (hdr_want_down && hdr_want_up) {
            if (p->family == FUSED_FAMILY_THIRDS)
                fused_kernel_thirds_up_hdr_scalar(p, src_y, src_u, src_v);
            else
                fused_kernel_pow2_up_hdr_scalar(p, src_y, src_u, src_v);
        } else if (hdr_want_up) {
            fused_kernel_upscale_hdr_scalar(p, src_y, src_u, src_v);
        } else if (p->family == FUSED_FAMILY_THIRDS) {
            fused_kernel_thirds_hdr_scalar(p, src_y, src_u, src_v);
        } else {
            fused_kernel_pow2_hdr_scalar(p, src_y, src_u, src_v);
        }
    } else {
        /* ------------------------------------------------------------------ */
        /* 2. Run the 10-bit scaling kernel                                     */
        /* ------------------------------------------------------------------ */

        state->kernel_fn(&state->params, src_y, src_u, src_v);
    }

    /* ------------------------------------------------------------------ */
    /* 3. Apply tone mapping to each SDR output                             */
    /* ------------------------------------------------------------------ */

    if (state->sdr_flags) {
        for (int i = 0; i < 8; i++) {
            if (!(state->sdr_flags & (1u << i))) continue;

            /* Determine the 10-bit source for tone mapping:
             * If HDR output exists at this step, read from hdr_outputs.
             * Otherwise, read from the temp 10-bit buffers. */
            const uint16_t *tm_src_y, *tm_src_u, *tm_src_v;
            int tm_y_stride, tm_uv_stride;

            if (ctx->achieved_hdr_flags & (1u << i)) {
                tm_src_y    = ctx->hdr_outputs[i].plane_y;
                tm_src_u    = ctx->hdr_outputs[i].plane_u;
                tm_src_v    = ctx->hdr_outputs[i].plane_v;
                tm_y_stride  = ctx->hdr_outputs[i].y_stride;
                tm_uv_stride = ctx->hdr_outputs[i].uv_stride;
            } else {
                tm_src_y    = state->sdr_temp[i].y;
                tm_src_u    = state->sdr_temp[i].u;
                tm_src_v    = state->sdr_temp[i].v;
                tm_y_stride  = stride_for_hdr(ctx->sdr_outputs[i].width);
                tm_uv_stride = stride_for_hdr(ctx->sdr_outputs[i].width / 2);
            }

            fused_tonemap_apply(state,
                tm_src_y, tm_y_stride,
                tm_src_u, tm_uv_stride,
                tm_src_v,
                ctx->sdr_outputs[i].plane_y, ctx->sdr_outputs[i].y_stride,
                ctx->sdr_outputs[i].plane_u, ctx->sdr_outputs[i].uv_stride,
                ctx->sdr_outputs[i].plane_v,
                ctx->sdr_outputs[i].width, ctx->sdr_outputs[i].height);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 4. 1:1 tone map (source resolution, no scaling)                      */
    /* ------------------------------------------------------------------ */

    if (state->tonemap_1x) {
        if (is_p010) {
            fused_tonemap_apply_p010(state,
                src_y, ctx->src_y_stride,
                src_u, ctx->src_uv_stride,
                ctx->output_1x.plane_y, ctx->output_1x.y_stride,
                ctx->output_1x.plane_u, ctx->output_1x.uv_stride,
                ctx->output_1x.plane_v,
                ctx->effective_width, ctx->effective_height);
        } else {
            fused_tonemap_apply(state,
                src_y, ctx->src_y_stride,
                src_u, ctx->src_uv_stride,
                src_v,
                ctx->output_1x.plane_y, ctx->output_1x.y_stride,
                ctx->output_1x.plane_u, ctx->output_1x.uv_stride,
                ctx->output_1x.plane_v,
                ctx->effective_width, ctx->effective_height);
        }
    }
}

/* --------------------------------------------------------------------------
 * fused_hdr_free
 * -------------------------------------------------------------------------- */

void fused_hdr_free(fused_hdr_ctx_t *ctx)
{
    if (!ctx) return;

    /* Free HDR output planes */
    for (int i = 0; i < 8; i++) {
        free(ctx->hdr_outputs[i].plane_y);
        free(ctx->hdr_outputs[i].plane_u);
        free(ctx->hdr_outputs[i].plane_v);
        memset(&ctx->hdr_outputs[i], 0, sizeof(fused_hdr_output_t));
    }

    /* Free SDR output planes */
    for (int i = 0; i < 8; i++) {
        free(ctx->sdr_outputs[i].plane_y);
        free(ctx->sdr_outputs[i].plane_u);
        free(ctx->sdr_outputs[i].plane_v);
        memset(&ctx->sdr_outputs[i], 0, sizeof(fused_scale_output_t));
    }

    /* Free temp 10-bit buffers, P010 buffers, and internal state */
    fused_hdr_internal_t *state = (fused_hdr_internal_t *)ctx->_internal;
    if (state) {
        for (int i = 0; i < 8; i++) {
            free(state->sdr_temp[i].y);
            free(state->sdr_temp[i].u);
            free(state->sdr_temp[i].v);
        }
        free(state->params.p010_tmp_u);
        free(state->params.p010_tmp_v);
        free(state->params.upscale_scratch_hdr);
        free(state->params.scratch_pool);
        free(state);
    }

    /* Free 1:1 tonemap output planes */
    free(ctx->output_1x.plane_y);
    free(ctx->output_1x.plane_u);
    free(ctx->output_1x.plane_v);
    memset(&ctx->output_1x, 0, sizeof(fused_scale_output_t));

    /* Free upscale HDR output planes */
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) {
        free(ctx->upscale_hdr_outputs[i].plane_y);
        free(ctx->upscale_hdr_outputs[i].plane_u);
        free(ctx->upscale_hdr_outputs[i].plane_v);
        memset(&ctx->upscale_hdr_outputs[i], 0, sizeof(fused_hdr_output_t));
    }

    ctx->_internal              = NULL;
    ctx->achieved_hdr_flags     = 0;
    ctx->achieved_sdr_flags     = 0;
    ctx->rejected_flags         = 0;
    ctx->achieved_upscale_flags = 0;
    ctx->achieved_upscale_tail  = 0;
}
