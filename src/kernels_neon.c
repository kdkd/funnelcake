/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_neon.c - NEON (aarch64) fused downscale kernels.
 *
 * Two entry points:
 *   fused_kernel_pow2_neon   - power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_neon - thirds family (1.5x/3x/6x/12x)
 *
 * Both process YUV420 I420 frames plane-by-plane.
 *
 * "Fused" means vertical and horizontal reduction happen in the same pass
 * over source memory, rather than scaling vertically into an intermediate
 * full-width row buffer and then horizontally in a second pass.
 *
 * The thirds kernel (1.5x/3x/6x/12x) reads source rows in groups of 6,
 * which matches the vertical period of the thirds reduction.  All 6 rows
 * are loaded simultaneously so the vertical intermediates (pair averages
 * and bilinear blends) never need to be written to memory.  Horizontal
 * filtering is applied immediately for each 48-byte column chunk.
 *
 * NEON has a significant advantage for deinterleaving: vld3q_u8 is a
 * single hardware instruction that loads 48 consecutive bytes and
 * automatically separates them into three 16-byte vectors (one per
 * component of the ABC triplet).  There is no equivalent in SSE/AVX2,
 * which requires the table-based shuffle approach.  Each source row is
 * deinterleaved as it is loaded; because the vertical filters are lane-wise,
 * all later vertical intermediates remain deinterleaved without extra work.
 *
 * The pow2 kernel (2x/4x/8x/16x) uses a vertical cascade into temporary
 * buffers followed by a horizontal halving cascade, for the same reasons
 * as the AVX2 version.
 */

#if defined(__aarch64__)

#include "internal.h"
#include <arm_neon.h>
#include <string.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * Scalar helpers (shared with horizontal thirds phase)
 * ----------------------------------------------------------------------- */

/* avg_u8: rounded average of two bytes, (a+b+1)>>1.  The +1 causes ties to
 * round up, which matches the rounding behavior of vrhaddq_u8 (NEON) and
 * vpavgb (x86).  Keeping scalar and SIMD paths consistent matters for
 * correctness of the tail handling. */
static inline uint8_t avg_u8(uint8_t a, uint8_t b)
{
    return (uint8_t)(((uint16_t)a + (uint16_t)b + 1) >> 1);
}

/* blend_2_1: bilinear blend for the 3:2 horizontal reduction.
 *
 * In a 3:2 reduction, each source triplet (A, B, C) produces two output
 * pixels.  The first output sits 1/3 of the way through the triplet
 * (weighted toward A), the second sits 2/3 of the way (weighted toward C).
 * B is the center pixel shared between both blends.
 *
 *   output at 1/3: (A*171 + B*85 + 128) >> 8   [call as blend_2_1(A, B)]
 *   output at 2/3: (C*171 + B*85 + 128) >> 8   [call as blend_2_1(C, B)]
 *
 * The weights 171/256 ≈ 2/3 and 85/256 ≈ 1/3 implement bilinear
 * interpolation via integer multiply-and-shift instead of division. */
static inline uint8_t blend_2_1(uint8_t a, uint8_t b)
{
    return (uint8_t)(((uint16_t)a * 171 + (uint16_t)b * 85 + 128) >> 8);
}

/* div3_u16: integer division by 3 for the 3x (box-of-3) horizontal filter.
 *
 * (x * 0x5556) >> 16 is an integer approximation of x/3.  The magic
 * multiplier 0x5556/0x10000 = 21846/65536 ≈ 1/3, and the result is exact
 * for all x in [0, 765] - the maximum sum of three uint8 values. */
static inline uint8_t div3_u16(uint16_t sum)
{
    return (uint8_t)((sum * (uint32_t)0x5556) >> 16);
}

/* -----------------------------------------------------------------------
 * Horizontal NEON filters (thirds family)
 * ----------------------------------------------------------------------- */

/* Horizontal 1.5x filter (NEON): 3:2 bilinear reduction.
 * Every 3 source pixels -> 2 output pixels via weighted blend.
 * Processes 48 input -> 32 output bytes per NEON chunk.
 *
 * vld3q_u8 loads 48 bytes and automatically separates them into three
 * 16-byte vectors (val[0]=all first-of-triplet A, val[1]=all second B,
 * val[2]=all third C), which is the deinterleave that the SSE version
 * has to perform manually with shuffle tables.
 *
 * Geometry: output pixel 0 is at 1/3 position in the source triplet
 * (biased toward A): (A*171 + B*85 + 128) >> 8.  Output pixel 1 is at
 * 2/3 position (biased toward C): (C*171 + B*85 + 128) >> 8.  B is the
 * center pixel shared between both blends.  Inputs are widened to 16-bit
 * before multiplying because the weighted sums can exceed 255 before the
 * final shift.
 *
 * The interleave step reorders from [all out0, all out1] into the correct
 * memory layout [out0[0], out1[0], out0[1], out1[1], ...]. */
static void h_filter_1_5x(const uint8_t *restrict src, int src_w,
                           uint8_t *restrict dst, int dst_w)
{
    int chunks = dst_w / 32;
    uint8x8_t w171 = vdup_n_u8(171);
    uint8x8_t w85  = vdup_n_u8(85);

    for (int c = 0; c < chunks; c++) {
        uint8x16x3_t loaded = vld3q_u8(src + c * 48);
        uint8x16_t A = loaded.val[0];  /* first of each triple */
        uint8x16_t B = loaded.val[1];  /* second of each triple */
        uint8x16_t C = loaded.val[2];  /* third of each triple */

        /* Output 0: pixel at 1/3 position, weighted toward A */
        uint16x8_t t0_lo = vmull_u8(vget_low_u8(A), w171);
        t0_lo = vmlal_u8(t0_lo, vget_low_u8(B), w85);
        uint16x8_t t0_hi = vmull_u8(vget_high_u8(A), w171);
        t0_hi = vmlal_u8(t0_hi, vget_high_u8(B), w85);
        uint8x16_t out0 = vcombine_u8(vrshrn_n_u16(t0_lo, 8),
                                       vrshrn_n_u16(t0_hi, 8));

        /* Output 1: pixel at 2/3 position, weighted toward C */
        uint16x8_t t1_lo = vmull_u8(vget_low_u8(C), w171);
        t1_lo = vmlal_u8(t1_lo, vget_low_u8(B), w85);
        uint16x8_t t1_hi = vmull_u8(vget_high_u8(C), w171);
        t1_hi = vmlal_u8(t1_hi, vget_high_u8(B), w85);
        uint8x16_t out1 = vcombine_u8(vrshrn_n_u16(t1_lo, 8),
                                       vrshrn_n_u16(t1_hi, 8));

        /* Interleave: [out0[0], out1[0], out0[1], out1[1], ...] */
        uint8x16x2_t interleaved = vzipq_u8(out0, out1);
        vst1q_u8(dst + c * 32,      interleaved.val[0]);
        vst1q_u8(dst + c * 32 + 16, interleaved.val[1]);
    }

    /* Scalar tail */
    int x_out = chunks * 32;
    for (int x_in = chunks * 48; x_in < src_w - 2 && x_out < dst_w - 1;
         x_in += 3, x_out += 2) {
        dst[x_out]     = blend_2_1(src[x_in],     src[x_in + 1]);
        dst[x_out + 1] = blend_2_1(src[x_in + 2], src[x_in + 1]);
    }
}

/* Horizontal 3x filter (NEON): box average of 3 source pixels.
 * Processes 48 input -> 16 output bytes per NEON chunk.
 *
 * vld3q_u8 handles the deinterleave in a single instruction.  The three
 * uint8 components are then widened to 16-bit before summing, because the
 * sum can reach 765 (3 * 255) which overflows uint8.  Division by 3 uses
 * the same 0x5556 multiply-high trick as the scalar div3_u16 and the SSE
 * version, giving exact results for sums in [0, 765]. */
static void h_filter_3x(const uint8_t *restrict src, int src_w,
                         uint8_t *restrict dst, int dst_w)
{
    (void)src_w;
    int chunks = dst_w / 16;

    for (int c = 0; c < chunks; c++) {
        uint8x16x3_t loaded = vld3q_u8(src + c * 48);

        /* Widen and sum A + B + C */
        uint16x8_t sum_lo = vaddl_u8(vget_low_u8(loaded.val[0]),
                                      vget_low_u8(loaded.val[1]));
        sum_lo = vaddw_u8(sum_lo, vget_low_u8(loaded.val[2]));

        uint16x8_t sum_hi = vaddl_u8(vget_high_u8(loaded.val[0]),
                                      vget_high_u8(loaded.val[1]));
        sum_hi = vaddw_u8(sum_hi, vget_high_u8(loaded.val[2]));

        /* Divide by 3: (sum * 0x5556) >> 16
         * Use widening multiply: vmull_u16 gives u32, then shrn to u16 */
        uint16x8_t magic = vdupq_n_u16(0x5556);

        /* Process low 4 elements */
        uint32x4_t prod_0 = vmull_u16(vget_low_u16(sum_lo), vget_low_u16(magic));
        uint32x4_t prod_1 = vmull_u16(vget_high_u16(sum_lo), vget_high_u16(magic));
        uint16x4_t d_lo_0 = vshrn_n_u32(prod_0, 16);
        uint16x4_t d_lo_1 = vshrn_n_u32(prod_1, 16);

        uint32x4_t prod_2 = vmull_u16(vget_low_u16(sum_hi), vget_low_u16(magic));
        uint32x4_t prod_3 = vmull_u16(vget_high_u16(sum_hi), vget_high_u16(magic));
        uint16x4_t d_hi_0 = vshrn_n_u32(prod_2, 16);
        uint16x4_t d_hi_1 = vshrn_n_u32(prod_3, 16);

        uint8x8_t out_lo = vmovn_u16(vcombine_u16(d_lo_0, d_lo_1));
        uint8x8_t out_hi = vmovn_u16(vcombine_u16(d_hi_0, d_hi_1));
        vst1q_u8(dst + c * 16, vcombine_u8(out_lo, out_hi));
    }

    /* Scalar tail */
    for (int x = chunks * 16; x < dst_w; x++) {
        uint16_t sum = (uint16_t)src[3*x] + src[3*x+1] + src[3*x+2];
        dst[x] = div3_u16(sum);
    }
}

/* Horizontal halve filter (NEON): pairwise average.
 * Processes 16 input -> 8 output bytes per NEON chunk.
 *
 * vpaddlq_u8 horizontally adds adjacent byte pairs into 16-bit values in
 * a single instruction.  vrshrn_n_u16 narrows back to 8-bit with rounding,
 * equivalent to a rounded right shift by 1: (a+b+1)>>1. */
static void h_filter_halve(const uint8_t *restrict src,
                           uint8_t *restrict dst, int dst_w)
{
    int src_bytes = dst_w * 2;
    int n_chunks = src_bytes / 16;
    int out_x = 0;

    for (int c = 0; c < n_chunks; c++) {
        uint8x16_t v = vld1q_u8(src + c * 16);
        uint16x8_t pair_sum = vpaddlq_u8(v);
        uint8x8_t result = vrshrn_n_u16(pair_sum, 1);
        vst1_u8(dst + out_x, result);
        out_x += 8;
    }

    int tail_in = n_chunks * 16;
    for (int tx = tail_in; tx + 1 < src_bytes; tx += 2) {
        dst[out_x++] = avg_u8(src[tx], src[tx + 1]);
    }
}

/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single plane (NEON)
 *
 * Vertical: NEON vrhaddq_u8 cascade
 * Horizontal: NEON vpaddlq_u8 + vrshrn_n_u16 cascade
 * ----------------------------------------------------------------------- */

static void __attribute__((hot)) scale_plane_pow2_neon_buffered(
    const uint8_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint8_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    (void)dst_heights;

    static const int bit_pos[4] = { 1, 3, 5, 7 };

    /* Determine deepest active level (0=2x .. 3=16x). */
    int deepest = -1;
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) {
            deepest = k;
            break;
        }
    }
    if (deepest < 0) return;

    int group_rows = (2 << deepest);
    int num_groups = src_h / group_rows;

    /* Carve scratch buffers from the persistent pool (init-time alloc). */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint8_t *vert_buf[4] = { NULL, NULL, NULL, NULL };
    int vert_rows[4];

    for (int k = 0; k <= deepest; k++) {
        vert_rows[k] = group_rows >> (k + 1);
        vert_buf[k] = (uint8_t *)fused_scratch_alloc(
            &scratch, (size_t)vert_rows[k] * (size_t)src_w);
        if (!vert_buf[k]) return;
    }

    /* Horizontal cascade buffer */
    uint8_t *h_buf = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)src_w);
    if (!h_buf) return;

    int out_row[4] = { 0, 0, 0, 0 };

    /* NEON chunk count: process 16 bytes at a time */
    int neon_chunks = src_w / 16;

    for (int g = 0; g < num_groups; g++) {
        const uint8_t *grp_base = src + (size_t)g * (size_t)group_rows * (size_t)src_stride;

        /* -- Vertical cascade (NEON) --------------------------------- */

        /* Level 0 (2x vertical): pairwise average source rows */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint8_t *restrict ra = grp_base + (size_t)(2 * r)     * (size_t)src_stride;
            const uint8_t *restrict rb = grp_base + (size_t)(2 * r + 1) * (size_t)src_stride;
            uint8_t *restrict dst_row  = vert_buf[0] + (size_t)r * (size_t)src_w;

            int x = 0;
            for (int c = 0; c < neon_chunks; c++, x += 16) {
                uint8x16_t va = vld1q_u8(ra + x);
                uint8x16_t vb = vld1q_u8(rb + x);
                vst1q_u8(dst_row + x, vrhaddq_u8(va, vb));
            }
            for (; x < src_w; x++) {
                dst_row[x] = avg_u8(ra[x], rb[x]);
            }
        }

        /* Deeper levels: pairwise average previous level */
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint8_t *restrict ra = vert_buf[k - 1] + (size_t)(2 * r)     * (size_t)src_w;
                const uint8_t *restrict rb = vert_buf[k - 1] + (size_t)(2 * r + 1) * (size_t)src_w;
                uint8_t *restrict dst_row  = vert_buf[k] + (size_t)r * (size_t)src_w;

                int x = 0;
                for (int c = 0; c < neon_chunks; c++, x += 16) {
                    uint8x16_t va = vld1q_u8(ra + x);
                    uint8x16_t vb = vld1q_u8(rb + x);
                    vst1q_u8(dst_row + x, vrhaddq_u8(va, vb));
                }
                for (; x < src_w; x++) {
                    dst_row[x] = avg_u8(ra[x], rb[x]);
                }
            }
        }

        /* -- Horizontal cascade (NEON) + output write ---------------- */

        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint8_t *restrict vert_row = vert_buf[k] + (size_t)r * (size_t)src_w;

                /* Horizontal cascade: (k+1) halvings.
                 * Each halving: out[x] = avg(in[2x], in[2x+1]).
                 * Use NEON: vpaddlq_u8 for adjacent pair sums -> u16,
                 * then vrshrn_n_u16 to narrow with rounding. */
                int cur_w = src_w;
                const uint8_t *cur_src = vert_row;

                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    int next_w = cur_w >> 1;
                    int neon_in_chunks = cur_w / 16;
                    int out_x = 0;

                    for (int c = 0; c < neon_in_chunks; c++) {
                        uint8x16_t v = vld1q_u8(cur_src + c * 16);
                        uint16x8_t pair_sum = vpaddlq_u8(v);
                        uint8x8_t result = vrshrn_n_u16(pair_sum, 1);
                        /* Store 8 output bytes */
                        vst1_u8(h_buf + out_x, result);
                        out_x += 8;
                    }
                    /* Scalar tail */
                    int tail_in = neon_in_chunks * 16;
                    for (int tx = tail_in; tx + 1 < cur_w; tx += 2) {
                        h_buf[out_x++] = avg_u8(cur_src[tx], cur_src[tx + 1]);
                    }

                    cur_w = next_w;
                    cur_src = h_buf;
                }

                /* Write to output plane */
                uint8_t *restrict out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_strides[k];
                memcpy(out, h_buf, (size_t)dst_widths[k]);
                out_row[k]++;
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}

/* Register-tiled power-of-two reductions.  The common aligned-width path
 * keeps the complete vertical tree for one 16-column tile in registers and
 * immediately applies the required number of horizontal halvings. */
static inline uint8x16_t pow2_hhalve_reg(uint8x16_t v)
{
    uint16x8_t sum = vpaddlq_u8(v);
    uint8x8_t out = vrshrn_n_u16(sum, 1);
    return vcombine_u8(out, vdup_n_u8(0));
}

static inline void pow2_store_h1(uint8x16_t v, uint8_t *dst)
{
    v = pow2_hhalve_reg(v);
    vst1_u8(dst, vget_low_u8(v));
}

static inline void pow2_store_h2(uint8x16_t v, uint8_t *dst)
{
    v = pow2_hhalve_reg(pow2_hhalve_reg(v));
    vst1_lane_u32((uint32_t *)dst,
                  vreinterpret_u32_u8(vget_low_u8(v)), 0);
}

static inline void pow2_store_h3(uint8x16_t v, uint8_t *dst)
{
    v = pow2_hhalve_reg(pow2_hhalve_reg(pow2_hhalve_reg(v)));
    vst1_lane_u16((uint16_t *)dst,
                  vreinterpret_u16_u8(vget_low_u8(v)), 0);
}

static inline void pow2_store_h4(uint8x16_t v, uint8_t *dst)
{
    v = pow2_hhalve_reg(
        pow2_hhalve_reg(pow2_hhalve_reg(pow2_hhalve_reg(v))));
    vst1_lane_u8(dst, vget_low_u8(v), 0);
}

static void pow2_tiled_depth0(
    const uint8_t *restrict src, int src_w, int src_h, int src_stride,
    uint32_t active_outputs, uint8_t *restrict dst[4], int strides[4])
{
    int groups = src_h / 2;
    int emit0 = (active_outputs & (1u << 1)) != 0;
    for (int g = 0; g < groups; g++) {
        const uint8_t *r0 = src + (size_t)(2 * g) * src_stride;
        const uint8_t *r1 = r0 + src_stride;
        for (int x = 0; x < src_w; x += 16) {
            uint8x16_t v0 = vrhaddq_u8(vld1q_u8(r0 + x),
                                        vld1q_u8(r1 + x));
            if (emit0)
                pow2_store_h1(v0, dst[0] + (size_t)g * strides[0] + x / 2);
        }
    }
}

static void pow2_tiled_depth1(
    const uint8_t *restrict src, int src_w, int src_h, int src_stride,
    uint32_t active_outputs, uint8_t *restrict dst[4], int strides[4])
{
    int groups = src_h / 4;
    int emit0 = (active_outputs & (1u << 1)) != 0;
    int emit1 = (active_outputs & (1u << 3)) != 0;
    for (int g = 0; g < groups; g++) {
        const uint8_t *base = src + (size_t)(4 * g) * src_stride;
        for (int x = 0; x < src_w; x += 16) {
            uint8x16_t v0 = vrhaddq_u8(vld1q_u8(base + x),
                                        vld1q_u8(base + src_stride + x));
            uint8x16_t v1 = vrhaddq_u8(vld1q_u8(base + 2 * (size_t)src_stride + x),
                                        vld1q_u8(base + 3 * (size_t)src_stride + x));
            if (emit0) {
                pow2_store_h1(v0, dst[0] + (size_t)(2 * g) * strides[0] + x / 2);
                pow2_store_h1(v1, dst[0] + (size_t)(2 * g + 1) * strides[0] + x / 2);
            }
            uint8x16_t v2 = vrhaddq_u8(v0, v1);
            if (emit1)
                pow2_store_h2(v2, dst[1] + (size_t)g * strides[1] + x / 4);
        }
    }
}

static void pow2_tiled_depth2(
    const uint8_t *restrict src, int src_w, int src_h, int src_stride,
    uint32_t active_outputs, uint8_t *restrict dst[4], int strides[4])
{
    int groups = src_h / 8;
    int emit0 = (active_outputs & (1u << 1)) != 0;
    int emit1 = (active_outputs & (1u << 3)) != 0;
    int emit2 = (active_outputs & (1u << 5)) != 0;
    for (int g = 0; g < groups; g++) {
        const uint8_t *base = src + (size_t)(8 * g) * src_stride;
        for (int x = 0; x < src_w; x += 16) {
            uint8x16_t v0 = vrhaddq_u8(vld1q_u8(base + x),
                                        vld1q_u8(base + src_stride + x));
            uint8x16_t v1 = vrhaddq_u8(vld1q_u8(base + 2 * (size_t)src_stride + x),
                                        vld1q_u8(base + 3 * (size_t)src_stride + x));
            uint8x16_t v2 = vrhaddq_u8(vld1q_u8(base + 4 * (size_t)src_stride + x),
                                        vld1q_u8(base + 5 * (size_t)src_stride + x));
            uint8x16_t v3 = vrhaddq_u8(vld1q_u8(base + 6 * (size_t)src_stride + x),
                                        vld1q_u8(base + 7 * (size_t)src_stride + x));
            if (emit0) {
                uint8_t *out = dst[0] + (size_t)(4 * g) * strides[0] + x / 2;
                pow2_store_h1(v0, out);
                pow2_store_h1(v1, out + strides[0]);
                pow2_store_h1(v2, out + 2 * (size_t)strides[0]);
                pow2_store_h1(v3, out + 3 * (size_t)strides[0]);
            }
            uint8x16_t v4 = vrhaddq_u8(v0, v1);
            uint8x16_t v5 = vrhaddq_u8(v2, v3);
            if (emit1) {
                uint8_t *out = dst[1] + (size_t)(2 * g) * strides[1] + x / 4;
                pow2_store_h2(v4, out);
                pow2_store_h2(v5, out + strides[1]);
            }
            uint8x16_t v6 = vrhaddq_u8(v4, v5);
            if (emit2)
                pow2_store_h3(v6, dst[2] + (size_t)g * strides[2] + x / 8);
        }
    }
}

static void pow2_tiled_depth3(
    const uint8_t *restrict src, int src_w, int src_h, int src_stride,
    uint32_t active_outputs, uint8_t *restrict dst[4], int strides[4])
{
    int groups = src_h / 16;
    int emit0 = (active_outputs & (1u << 1)) != 0;
    int emit1 = (active_outputs & (1u << 3)) != 0;
    int emit2 = (active_outputs & (1u << 5)) != 0;
    int emit3 = (active_outputs & (1u << 7)) != 0;
    for (int g = 0; g < groups; g++) {
        const uint8_t *base = src + (size_t)(16 * g) * src_stride;
        for (int x = 0; x < src_w; x += 16) {
            uint8x16_t v[8];
#if defined(__clang__)
            #pragma clang loop unroll(full)
#elif defined(__GNUC__)
            #pragma GCC unroll 8
#endif
            for (int r = 0; r < 8; r++) {
                const uint8_t *ra = base + (size_t)(2 * r) * src_stride + x;
                v[r] = vrhaddq_u8(vld1q_u8(ra), vld1q_u8(ra + src_stride));
            }
            if (emit0) {
                uint8_t *out = dst[0] + (size_t)(8 * g) * strides[0] + x / 2;
#if defined(__clang__)
                #pragma clang loop unroll(full)
#elif defined(__GNUC__)
                #pragma GCC unroll 8
#endif
                for (int r = 0; r < 8; r++)
                    pow2_store_h1(v[r], out + (size_t)r * strides[0]);
            }
            uint8x16_t w0 = vrhaddq_u8(v[0], v[1]);
            uint8x16_t w1 = vrhaddq_u8(v[2], v[3]);
            uint8x16_t w2 = vrhaddq_u8(v[4], v[5]);
            uint8x16_t w3 = vrhaddq_u8(v[6], v[7]);
            if (emit1) {
                uint8_t *out = dst[1] + (size_t)(4 * g) * strides[1] + x / 4;
                pow2_store_h2(w0, out);
                pow2_store_h2(w1, out + strides[1]);
                pow2_store_h2(w2, out + 2 * (size_t)strides[1]);
                pow2_store_h2(w3, out + 3 * (size_t)strides[1]);
            }
            uint8x16_t z0 = vrhaddq_u8(w0, w1);
            uint8x16_t z1 = vrhaddq_u8(w2, w3);
            if (emit2) {
                uint8_t *out = dst[2] + (size_t)(2 * g) * strides[2] + x / 8;
                pow2_store_h3(z0, out);
                pow2_store_h3(z1, out + strides[2]);
            }
            uint8x16_t q0 = vrhaddq_u8(z0, z1);
            if (emit3)
                pow2_store_h4(q0, dst[3] + (size_t)g * strides[3] + x / 16);
        }
    }
}

static void __attribute__((hot)) scale_plane_pow2_neon(
    const uint8_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint8_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    if ((src_w & 15) != 0) {
        scale_plane_pow2_neon_buffered(src, src_w, src_h, src_stride,
            active_outputs, dst_planes, dst_widths, dst_strides, dst_heights,
            scratch_pool_base, scratch_pool_size);
        return;
    }

    int deepest = -1;
    static const int bit_pos[4] = { 1, 3, 5, 7 };
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) {
            deepest = k;
            break;
        }
    }
    switch (deepest) {
    case 0:
        pow2_tiled_depth0(src, src_w, src_h, src_stride,
                          active_outputs, dst_planes, dst_strides);
        break;
    case 1:
        pow2_tiled_depth1(src, src_w, src_h, src_stride,
                          active_outputs, dst_planes, dst_strides);
        break;
    case 2:
        pow2_tiled_depth2(src, src_w, src_h, src_stride,
                          active_outputs, dst_planes, dst_strides);
        break;
    case 3:
        pow2_tiled_depth3(src, src_w, src_h, src_stride,
                          active_outputs, dst_planes, dst_strides);
        break;
    default:
        break;
    }
}


/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single plane (NEON fused vertical+horizontal)
 *
 * Per-chunk fused architecture: source rows are processed in groups of 6
 * (matching the vertical period of the thirds reduction).  For each 48-byte
 * column chunk, all 6 rows are loaded, vertical pair averages and bilinear
 * blends are computed entirely in NEON registers, and horizontal filtering
 * is applied immediately without writing any intermediate row buffer.
 *
 * Vertical: NEON vrhaddq_u8 for pairwise avgs, vmull_u8/vmlal_u8 for
 *           bilinear blends (1.5x)
 * Horizontal: source-time vld3q_u8 + blend/box-of-3/halve helpers
 *
 * A 6-source-row group produces 4 output rows at 1.5x, 2 at 3x, and 1
 * at 6x.  12x requires pairing two consecutive 6x rows via a ping-pong
 * buffer scheme.
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Fused vertical+horizontal chunk helpers
 *
 * These inline functions perform horizontal filtering on a 48-byte chunk
 * (3 x uint8x16_t) that was deinterleaved at the source load and then
 * vertically reduced in registers.
 * ----------------------------------------------------------------------- */

/* NEON divide-by-3 for a uint16x8 of sums (each max 765).
 * Returns uint8x8_t: (sum * 0x5556) >> 16, narrowed to u8. */
static inline uint8x8_t neon_div3_u16x8(uint16x8_t sum)
{
    uint16x8_t magic = vdupq_n_u16(0x5556);
    uint32x4_t p0 = vmull_u16(vget_low_u16(sum),  vget_low_u16(magic));
    uint32x4_t p1 = vmull_u16(vget_high_u16(sum), vget_high_u16(magic));
    uint16x4_t d0 = vshrn_n_u32(p0, 16);
    uint16x4_t d1 = vshrn_n_u32(p1, 16);
    return vmovn_u16(vcombine_u16(d0, d1));
}

/* Horizontal 1.5x on one deinterleaved chunk (A, B, C each 16 bytes).
 * Produces 32 output bytes stored at dst. */
static inline void h_chunk_1_5x(uint8x16_t A, uint8x16_t B, uint8x16_t C,
                                 uint8_t *restrict dst,
                                 uint8x8_t w171, uint8x8_t w85)
{
    /* out0 = (A*171 + B*85 + 128) >> 8 */
    uint16x8_t t0_lo = vmull_u8(vget_low_u8(A), w171);
    t0_lo = vmlal_u8(t0_lo, vget_low_u8(B), w85);
    uint16x8_t t0_hi = vmull_u8(vget_high_u8(A), w171);
    t0_hi = vmlal_u8(t0_hi, vget_high_u8(B), w85);
    uint8x16_t out0 = vcombine_u8(vrshrn_n_u16(t0_lo, 8),
                                   vrshrn_n_u16(t0_hi, 8));

    /* out1 = (C*171 + B*85 + 128) >> 8 */
    uint16x8_t t1_lo = vmull_u8(vget_low_u8(C), w171);
    t1_lo = vmlal_u8(t1_lo, vget_low_u8(B), w85);
    uint16x8_t t1_hi = vmull_u8(vget_high_u8(C), w171);
    t1_hi = vmlal_u8(t1_hi, vget_high_u8(B), w85);
    uint8x16_t out1 = vcombine_u8(vrshrn_n_u16(t1_lo, 8),
                                   vrshrn_n_u16(t1_hi, 8));

    /* Interleave: [out0[0], out1[0], out0[1], out1[1], ...] */
    uint8x16x2_t interleaved = vzipq_u8(out0, out1);
    vst1q_u8(dst,      interleaved.val[0]);
    vst1q_u8(dst + 16, interleaved.val[1]);
}

/* Horizontal 3x on one deinterleaved chunk (A, B, C each 16 bytes).
 * Produces 16 output bytes stored at dst. */
static inline uint8x16_t h_chunk_3x(uint8x16_t A, uint8x16_t B, uint8x16_t C,
                                     uint8_t *restrict dst)
{
    uint16x8_t sum_lo = vaddl_u8(vget_low_u8(A), vget_low_u8(B));
    sum_lo = vaddw_u8(sum_lo, vget_low_u8(C));
    uint16x8_t sum_hi = vaddl_u8(vget_high_u8(A), vget_high_u8(B));
    sum_hi = vaddw_u8(sum_hi, vget_high_u8(C));

    uint8x8_t out_lo = neon_div3_u16x8(sum_lo);
    uint8x8_t out_hi = neon_div3_u16x8(sum_hi);
    uint8x16_t result = vcombine_u8(out_lo, out_hi);
    vst1q_u8(dst, result);
    return result;
}

/* Horizontal 3x computation without a store.  The 6x path consumes the
 * register immediately, so materializing the 3x intermediate in memory only
 * creates a store-forwarding dependency. */
static inline uint8x16_t h_chunk_3x_reg(uint8x16_t A, uint8x16_t B,
                                        uint8x16_t C)
{
    uint16x8_t sum_lo = vaddl_u8(vget_low_u8(A), vget_low_u8(B));
    sum_lo = vaddw_u8(sum_lo, vget_low_u8(C));
    uint16x8_t sum_hi = vaddl_u8(vget_high_u8(A), vget_high_u8(B));
    sum_hi = vaddw_u8(sum_hi, vget_high_u8(C));

    return vcombine_u8(neon_div3_u16x8(sum_lo),
                       neon_div3_u16x8(sum_hi));
}

/* Horizontal 6x cascaded from a 3x result (16 bytes -> 8 bytes).
 * Stores 8 output bytes at dst. Returns the 8-byte result. */
static inline uint8x8_t h_chunk_6x(uint8x16_t result_3x, uint8_t *restrict dst)
{
    uint16x8_t pair_sum = vpaddlq_u8(result_3x);
    uint8x8_t result = vrshrn_n_u16(pair_sum, 1);
    vst1_u8(dst, result);
    return result;
}

/* Bilinear blend of two 16-byte registers: (a*171 + b*85 + 128) >> 8.
 * SIMD equivalent of blend_2_1 for a full 16-byte register - used during
 * the 1.5x vertical blending phase to blend between two row-pair averages. */
static inline uint8x16_t neon_blend_reg(uint8x16_t a, uint8x16_t b,
                                         uint8x8_t w171, uint8x8_t w85)
{
    uint16x8_t lo = vmull_u8(vget_low_u8(a), w171);
    lo = vmlal_u8(lo, vget_low_u8(b), w85);
    uint16x8_t hi = vmull_u8(vget_high_u8(a), w171);
    hi = vmlal_u8(hi, vget_high_u8(b), w85);
    return vcombine_u8(vrshrn_n_u16(lo, 8), vrshrn_n_u16(hi, 8));
}


static void __attribute__((hot)) scale_plane_thirds_neon(
    const uint8_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint8_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    (void)dst_heights;

    static const int bit_pos[4] = { 0, 2, 4, 6 };

    int deepest = -1;
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) {
            deepest = k;
            break;
        }
    }
    if (deepest < 0) return;

    int need_1_5x = (active_outputs & (1u << 0)) != 0;
    int need_3x   = (deepest >= 1);
    int need_6x   = (deepest >= 2);
    int need_12x  = (deepest >= 3);

    int base6_groups = src_h / 6;
    size_t row_bytes = (size_t)src_w;

    /* Carve scratch buffers from the persistent pool (init-time alloc). */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    /* For 12x: two buffers to hold 6x vertical intermediates across
     * consecutive 6-row groups. On even groups we write to v6x_cur,
     * then swap pointers so the odd group can read the previous. */
    uint8_t *v6x_buf_a = NULL, *v6x_buf_b = NULL;
    uint8_t *v6x_cur = NULL, *v6x_prev = NULL;
    if (need_12x) {
        v6x_buf_a = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
        v6x_buf_b = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
        if (!v6x_buf_a || !v6x_buf_b) return;
        v6x_cur  = v6x_buf_a;
        v6x_prev = v6x_buf_b;
    }

    /* Horizontal scratch for 12x (3x -> halve -> halve) */
    int w_3x = src_w / 3;
    int w_6x = w_3x / 2;
    uint8_t *h_3x_buf = NULL, *h_6x_buf = NULL;
    if (need_12x && (active_outputs & (1u << 6))) {
        h_3x_buf = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)w_3x);
        h_6x_buf = (uint8_t *)fused_scratch_alloc(
            &scratch, (size_t)(w_6x > 0 ? w_6x : 1));
        if (!h_3x_buf || !h_6x_buf) return;
    }

    /* NEON weight constants */
    uint8x8_t w171 = vdup_n_u8(171);
    uint8x8_t w85  = vdup_n_u8(85);

    /* Chunk geometry */
    int full_chunks = src_w / 48;
    int tail_start  = full_chunks * 48;
    int tail_cols   = src_w - tail_start;

    /* Output row cursors */
    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint8_t *grp = src + (size_t)g6 * 6 * (size_t)src_stride;

        const uint8_t *restrict row0 = grp;
        const uint8_t *restrict row1 = grp + (size_t)src_stride;
        const uint8_t *restrict row2 = grp + (size_t)2 * (size_t)src_stride;
        const uint8_t *restrict row3 = grp + (size_t)3 * (size_t)src_stride;
        const uint8_t *restrict row4 = grp + (size_t)4 * (size_t)src_stride;
        const uint8_t *restrict row5 = grp + (size_t)5 * (size_t)src_stride;

        /* Compute output row base pointers */
        uint8_t *dst_1_5x_r0 = NULL, *dst_1_5x_r1 = NULL;
        uint8_t *dst_1_5x_r2 = NULL, *dst_1_5x_r3 = NULL;
        if (need_1_5x) {
            int ds = dst_strides[0];
            dst_1_5x_r0 = dst_planes[0] + (size_t)out_row[0]       * (size_t)ds;
            dst_1_5x_r1 = dst_planes[0] + (size_t)(out_row[0] + 1) * (size_t)ds;
            dst_1_5x_r2 = dst_planes[0] + (size_t)(out_row[0] + 2) * (size_t)ds;
            dst_1_5x_r3 = dst_planes[0] + (size_t)(out_row[0] + 3) * (size_t)ds;
        }

        uint8_t *dst_3x_r0 = NULL, *dst_3x_r1 = NULL;
        if (active_outputs & (1u << 2)) {
            int ds = dst_strides[1];
            dst_3x_r0 = dst_planes[1] + (size_t)out_row[1]       * (size_t)ds;
            dst_3x_r1 = dst_planes[1] + (size_t)(out_row[1] + 1) * (size_t)ds;
        }

        uint8_t *dst_6x_r0 = NULL;
        if (active_outputs & (1u << 4)) {
            dst_6x_r0 = dst_planes[2] + (size_t)out_row[2] * (size_t)dst_strides[2];
        }

        /* ============================================================
         * MAIN CHUNK LOOP: process 48 source columns at a time.
         * Vertical intermediates stay in NEON registers; horizontal
         * filtering is applied immediately per chunk, so no intermediate
         * row buffers are needed.  A 6-source-row group produces 4 output
         * rows at 1.5x, 2 rows at 3x, and 1 row at 6x.
         * ============================================================ */
#if defined(__clang__)
        #pragma clang loop unroll_count(2)
#elif defined(__GNUC__)
        #pragma GCC unroll 2
#endif
        for (int ci = 0; ci < full_chunks; ci++) {
            int cx = ci * 48;
            int out_off_1_5x = ci * 32;  /* 48 -> 32 output bytes */
            int out_off_3x   = ci * 16;  /* 48 -> 16 output bytes */
            int out_off_6x   = ci * 8;   /* 48 ->  8 output bytes */

            /* Deinterleave at the source load.  Every vertical operation is
             * lane-wise, so the permutation commutes exactly with the
             * vertical cascade and avoids register -> stack -> ld3 round
             * trips for each output row. */
            uint8x16x3_t r0 = vld3q_u8(row0 + cx);
            uint8x16x3_t r1 = vld3q_u8(row1 + cx);
            uint8x16x3_t r2 = vld3q_u8(row2 + cx);
            uint8x16x3_t r3 = vld3q_u8(row3 + cx);
            uint8x16x3_t r4 = vld3q_u8(row4 + cx);
            uint8x16x3_t r5 = vld3q_u8(row5 + cx);

            /* --- VERTICAL PAIRWISE AVERAGES (in registers) ---
             * Average adjacent row pairs: rows 0+1 -> v01, rows 2+3 -> v23,
             * rows 4+5 -> v45.  These three intermediates represent the
             * vertical center of each pair and are reused across all output
             * levels without writing to memory. */
            uint8x16_t v01a = vrhaddq_u8(r0.val[0], r1.val[0]);
            uint8x16_t v01b = vrhaddq_u8(r0.val[1], r1.val[1]);
            uint8x16_t v01c = vrhaddq_u8(r0.val[2], r1.val[2]);
            uint8x16_t v23a = vrhaddq_u8(r2.val[0], r3.val[0]);
            uint8x16_t v23b = vrhaddq_u8(r2.val[1], r3.val[1]);
            uint8x16_t v23c = vrhaddq_u8(r2.val[2], r3.val[2]);
            uint8x16_t v45a = vrhaddq_u8(r4.val[0], r5.val[0]);
            uint8x16_t v45b = vrhaddq_u8(r4.val[1], r5.val[1]);
            uint8x16_t v45c = vrhaddq_u8(r4.val[2], r5.val[2]);

            /* --- 1.5x OUTPUT (4 rows) ---
             * A 6-source-row group produces 4 output rows at 1.5x.  Rows 0
             * and 3 come directly from v01 and v45 (the pair averages
             * themselves).  Rows 1 and 2 are bilinear blends between
             * adjacent pair averages - blend(v01,v23) and blend(v23,v45) -
             * representing the vertical positions 1/3 and 2/3 of the way
             * through the group. */
            if (need_1_5x) {
                /* Row 0: v01 */
                h_chunk_1_5x(v01a, v01b, v01c,
                             dst_1_5x_r0 + out_off_1_5x, w171, w85);

                /* Row 1: blend(v01, v23) */
                {
                    uint8x16_t ba = neon_blend_reg(v01a, v23a, w171, w85);
                    uint8x16_t bb = neon_blend_reg(v01b, v23b, w171, w85);
                    uint8x16_t bc = neon_blend_reg(v01c, v23c, w171, w85);
                    h_chunk_1_5x(ba, bb, bc,
                                 dst_1_5x_r1 + out_off_1_5x, w171, w85);
                }

                /* Row 2: blend(v23, v45) */
                {
                    uint8x16_t ba = neon_blend_reg(v23a, v45a, w171, w85);
                    uint8x16_t bb = neon_blend_reg(v23b, v45b, w171, w85);
                    uint8x16_t bc = neon_blend_reg(v23c, v45c, w171, w85);
                    h_chunk_1_5x(ba, bb, bc,
                                 dst_1_5x_r2 + out_off_1_5x, w171, w85);
                }

                /* Row 3: v45 */
                h_chunk_1_5x(v45a, v45b, v45c,
                             dst_1_5x_r3 + out_off_1_5x, w171, w85);
            }

            /* --- 3x VERTICAL + HORIZONTAL (2 rows) ---
             * A 6-source-row group produces 2 output rows at 3x.  The 3x
             * vertical reduction averages each pair-average with the next:
             * avg(v01,v23) and avg(v23,v45), which together represent a
             * 3:1 reduction of the 6 source rows. */
            uint8x16_t v3x0a = vdupq_n_u8(0), v3x0b = vdupq_n_u8(0), v3x0c = vdupq_n_u8(0);
            uint8x16_t v3x1a = vdupq_n_u8(0), v3x1b = vdupq_n_u8(0), v3x1c = vdupq_n_u8(0);
            if (need_3x) {
                v3x0a = vrhaddq_u8(v01a, v23a);
                v3x0b = vrhaddq_u8(v01b, v23b);
                v3x0c = vrhaddq_u8(v01c, v23c);
                v3x1a = vrhaddq_u8(v23a, v45a);
                v3x1b = vrhaddq_u8(v23b, v45b);
                v3x1c = vrhaddq_u8(v23c, v45c);

                if (active_outputs & (1u << 2)) {
                    h_chunk_3x(v3x0a, v3x0b, v3x0c,
                               dst_3x_r0 + out_off_3x);

                    h_chunk_3x(v3x1a, v3x1b, v3x1c,
                               dst_3x_r1 + out_off_3x);
                }
            }

            /* --- 6x VERTICAL (1 row) + save for 12x + 6x horizontal ---
             * A 6-source-row group produces 1 output row at 6x, by averaging
             * the two 3x intermediates: avg(v3x0, v3x1) =
             * avg(avg(v01,v23), avg(v23,v45)). */
            if (need_6x) {
                uint8x16_t v6xa = vrhaddq_u8(v3x0a, v3x1a);
                uint8x16_t v6xb = vrhaddq_u8(v3x0b, v3x1b);
                uint8x16_t v6xc = vrhaddq_u8(v3x0c, v3x1c);

                /* Save 6x vertical intermediate for 12x pairing */
                if (need_12x) {
                    vst1q_u8(v6x_cur + cx,      v6xa);
                    vst1q_u8(v6x_cur + cx + 16, v6xb);
                    vst1q_u8(v6x_cur + cx + 32, v6xc);
                }

                if (active_outputs & (1u << 4)) {
                    uint8x16_t r3x = h_chunk_3x_reg(v6xa, v6xb, v6xc);
                    h_chunk_6x(r3x, dst_6x_r0 + out_off_6x);
                }
            }
        } /* end chunk loop */

        /* ============================================================
         * TAIL: handle remaining columns with scalar h_filter functions
         * ============================================================ */
        if (tail_cols > 0) {
            uint8_t tail_v01[48], tail_v23[48], tail_v45[48];

            /* Compute vertical intermediates for tail */
            for (int x = 0; x < tail_cols; x++) {
                int sx = tail_start + x;
                tail_v01[x] = avg_u8(row0[sx], row1[sx]);
                tail_v23[x] = avg_u8(row2[sx], row3[sx]);
                tail_v45[x] = avg_u8(row4[sx], row5[sx]);
            }

            uint8_t tail_v3x0[48], tail_v3x1[48];
            if (need_3x) {
                for (int x = 0; x < tail_cols; x++) {
                    tail_v3x0[x] = avg_u8(tail_v01[x], tail_v23[x]);
                    tail_v3x1[x] = avg_u8(tail_v23[x], tail_v45[x]);
                }
            }

            uint8_t tail_v6x[48];
            if (need_6x) {
                for (int x = 0; x < tail_cols; x++) {
                    tail_v6x[x] = avg_u8(tail_v3x0[x], tail_v3x1[x]);
                }
                if (need_12x) {
                    memcpy(v6x_cur + tail_start, tail_v6x, (size_t)tail_cols);
                }
            }

            /* How many output pixels the NEON chunks already produced */
            int tail_out_1_5x = full_chunks * 32;
            int tail_out_3x   = full_chunks * 16;
            int tail_out_6x   = full_chunks * 8;

            /* 1.5x tail */
            if (need_1_5x) {
                int dw_rem = dst_widths[0] - tail_out_1_5x;
                h_filter_1_5x(tail_v01, tail_cols,
                              dst_1_5x_r0 + tail_out_1_5x, dw_rem);

                uint8_t tail_blend[48];
                for (int x = 0; x < tail_cols; x++)
                    tail_blend[x] = blend_2_1(tail_v01[x], tail_v23[x]);
                h_filter_1_5x(tail_blend, tail_cols,
                              dst_1_5x_r1 + tail_out_1_5x, dw_rem);

                for (int x = 0; x < tail_cols; x++)
                    tail_blend[x] = blend_2_1(tail_v23[x], tail_v45[x]);
                h_filter_1_5x(tail_blend, tail_cols,
                              dst_1_5x_r2 + tail_out_1_5x, dw_rem);

                h_filter_1_5x(tail_v45, tail_cols,
                              dst_1_5x_r3 + tail_out_1_5x, dw_rem);
            }

            /* 3x tail */
            if (active_outputs & (1u << 2)) {
                int dw_rem = dst_widths[1] - tail_out_3x;
                h_filter_3x(tail_v3x0, tail_cols,
                            dst_3x_r0 + tail_out_3x, dw_rem);
                h_filter_3x(tail_v3x1, tail_cols,
                            dst_3x_r1 + tail_out_3x, dw_rem);
            }

            /* 6x tail */
            if (active_outputs & (1u << 4)) {
                int dw_rem = dst_widths[2] - tail_out_6x;
                int w3_tail = tail_cols / 3;
                uint8_t tail_h3x[16];
                h_filter_3x(tail_v6x, tail_cols, tail_h3x, w3_tail);
                h_filter_halve(tail_h3x, dst_6x_r0 + tail_out_6x, dw_rem);
            }
        }

        /* Update output row cursors */
        if (need_1_5x) out_row[0] += 4;
        if (active_outputs & (1u << 2)) out_row[1] += 2;
        if (active_outputs & (1u << 4)) out_row[2] += 1;

        /* ============================================================
         * 12x handling: pair two consecutive 6-row groups.
         * v6x_cur holds this group's 6x vertical intermediate (src_w bytes).
         * On even groups: swap pointers so current becomes previous.
         * On odd groups: average prev with current, apply horizontal, output.
         *
         * 12x requires averaging two consecutive 6x rows, each derived from
         * a different 6-row source group.  Since the two 6x rows come from
         * different iterations of the outer loop, we save each 6x
         * intermediate in v6x_cur and pair them every two iterations.
         * The ping-pong swap avoids copying.
         * ============================================================ */
        if (need_12x) {
            if ((g6 & 1) == 0) {
                /* Swap so v6x_cur data becomes v6x_prev for the next group */
                uint8_t *tmp = v6x_prev;
                v6x_prev = v6x_cur;
                v6x_cur  = tmp;
            } else {
                /* Average v6x_prev (even group) with v6x_cur (odd group) */
                /* Compute 12x vertical into v6x_prev (reuse as scratch) */
                int neon16 = src_w / 16;
                int x = 0;
                for (int c = 0; c < neon16; c++, x += 16) {
                    uint8x16_t a = vld1q_u8(v6x_prev + x);
                    uint8x16_t b = vld1q_u8(v6x_cur + x);
                    vst1q_u8(v6x_prev + x, vrhaddq_u8(a, b));
                }
                for (; x < src_w; x++) {
                    v6x_prev[x] = avg_u8(v6x_prev[x], v6x_cur[x]);
                }

                if (active_outputs & (1u << 6)) {
                    int dw = dst_widths[3];
                    int ds = dst_strides[3];

                    /* Full chunks are already deinterleaved in v6x_prev.
                     * Convert each directly to its 3x horizontal result;
                     * the scalar tail remains in ordinary row order. */
                    for (int ci = 0; ci < full_chunks; ci++) {
                        int cx = ci * 48;
                        uint8x16_t A = vld1q_u8(v6x_prev + cx);
                        uint8x16_t B = vld1q_u8(v6x_prev + cx + 16);
                        uint8x16_t C = vld1q_u8(v6x_prev + cx + 32);
                        h_chunk_3x(A, B, C, h_3x_buf + ci * 16);
                    }
                    if (tail_cols > 0) {
                        int tail_w3 = w_3x - full_chunks * 16;
                        h_filter_3x(v6x_prev + tail_start, tail_cols,
                                    h_3x_buf + full_chunks * 16, tail_w3);
                    }
                    h_filter_halve(h_3x_buf, h_6x_buf, w_6x);
                    h_filter_halve(h_6x_buf,
                                   dst_planes[3] + (size_t)out_row[3] * (size_t)ds, dw);
                    out_row[3]++;
                }
            }
        }
    } /* end g6 loop */

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}


/* -----------------------------------------------------------------------
 * Public entry points
 *
 * YUV420 I420 has the chroma planes at half the luma dimensions in both
 * axes.  We process the Y plane at full size, then U and V at half width
 * and half height with the same kernel.
 * ----------------------------------------------------------------------- */

void __attribute__((hot)) fused_kernel_pow2_neon(const fused_kernel_params_t *p,
                            const uint8_t *src_y,
                            const uint8_t *src_u,
                            const uint8_t *src_v)
{
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    uint8_t *y_planes[4], *u_planes[4], *v_planes[4];
    int y_widths[4], y_heights[4], y_strides[4];
    int uv_widths[4], uv_heights[4], uv_strides[4];

    for (int k = 0; k < 4; k++) {
        int b = bit_pos[k];
        if (p->active_outputs & (1u << b)) {
            y_planes[k]  = p->out[b].plane_y;
            y_widths[k]  = p->out[b].width;
            y_heights[k] = p->out[b].height;
            y_strides[k] = p->out[b].y_stride;

            u_planes[k]  = p->out[b].plane_u;
            v_planes[k]  = p->out[b].plane_v;
            uv_widths[k]  = p->out[b].width / 2;
            uv_heights[k] = p->out[b].height / 2;
            uv_strides[k] = p->out[b].uv_stride;
        } else {
            y_planes[k] = u_planes[k] = v_planes[k] = NULL;
            y_widths[k] = y_heights[k] = y_strides[k] = 0;
            uv_widths[k] = uv_heights[k] = uv_strides[k] = 0;
        }
    }

    /* Y plane */
    scale_plane_pow2_neon(src_y,
                          p->src_width, p->src_height, p->src_y_stride,
                          p->active_outputs,
                          y_planes, y_widths, y_strides, y_heights,
                          p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_pow2_neon(src_u,
                          p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                          p->active_outputs,
                          u_planes, uv_widths, uv_strides, uv_heights,
                          p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_pow2_neon(src_v,
                          p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                          p->active_outputs,
                          v_planes, uv_widths, uv_strides, uv_heights,
                          p->scratch_pool, p->scratch_pool_size);
}


void __attribute__((hot)) fused_kernel_thirds_neon(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v)
{
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    uint8_t *y_planes[4], *u_planes[4], *v_planes[4];
    int y_widths[4], y_heights[4], y_strides[4];
    int uv_widths[4], uv_heights[4], uv_strides[4];

    for (int k = 0; k < 4; k++) {
        int b = bit_pos[k];
        if (p->active_outputs & (1u << b)) {
            y_planes[k]  = p->out[b].plane_y;
            y_widths[k]  = p->out[b].width;
            y_heights[k] = p->out[b].height;
            y_strides[k] = p->out[b].y_stride;

            u_planes[k]  = p->out[b].plane_u;
            v_planes[k]  = p->out[b].plane_v;
            uv_widths[k]  = p->out[b].width / 2;
            uv_heights[k] = p->out[b].height / 2;
            uv_strides[k] = p->out[b].uv_stride;
        } else {
            y_planes[k] = u_planes[k] = v_planes[k] = NULL;
            y_widths[k] = y_heights[k] = y_strides[k] = 0;
            uv_widths[k] = uv_heights[k] = uv_strides[k] = 0;
        }
    }

    /* Y plane */
    scale_plane_thirds_neon(src_y,
                            p->src_width, p->src_height, p->src_y_stride,
                            p->active_outputs,
                            y_planes, y_widths, y_strides, y_heights,
                            p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_thirds_neon(src_u,
                            p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                            p->active_outputs,
                            u_planes, uv_widths, uv_strides, uv_heights,
                            p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_thirds_neon(src_v,
                            p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                            p->active_outputs,
                            v_planes, uv_widths, uv_strides, uv_heights,
                            p->scratch_pool, p->scratch_pool_size);
}

#endif /* __aarch64__ */
