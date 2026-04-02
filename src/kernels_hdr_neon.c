/*
 * kernels_hdr_neon.c -- NEON (aarch64) fused downscale kernels for
 *                       10-bit HDR content (uint16_t samples).
 *
 * Two entry points:
 *   fused_kernel_pow2_hdr_neon   -- power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_hdr_neon -- thirds family (1.5x/3x/6x/12x)
 *
 * Both process YUV420 frames plane-by-plane (I010) or with interleaved
 * chroma deinterleaving (P010).
 *
 * This is the 10-bit (uint16_t) analog of kernels_neon.c.  Key arithmetic
 * differences from the 8-bit NEON kernels:
 *
 *   - Elements are uint16_t, so a Q register holds 8 elements (not 16).
 *   - Pairwise average: vrhaddq_u16 (not vrhaddq_u8).
 *   - Blend (2/3, 1/3): widen to uint32_t via vmull_u16/vmlal_u16 because
 *     10-bit * 171 = max 174,933 which overflows uint16_t.
 *   - Divide by 3: vmull_u16 with magic 21846, shift right 16, giving exact
 *     results for sums up to 3069 (3 * 1023).
 *   - Horizontal halving: vpaddlq_u16 sums pairs into uint32_t, then
 *     vrshrn_n_u32 narrows back to uint16_t with rounding.
 *   - Deinterleave: vld3q_u16 loads 24 uint16_t (48 bytes) and separates
 *     into three 8-element vectors (vs vld3q_u8 for 48 bytes / 16 elements).
 *   - Chunk sizes: 48 bytes = 24 uint16_t = 8 triplets (thirds family),
 *     16 bytes = 8 uint16_t elements (pow2 family).
 *
 * Guarded by __aarch64__ so this file is a no-op on other platforms.
 */

#if defined(__aarch64__)

#include "internal.h"
#include <arm_neon.h>
#include <string.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * Scalar helpers (shared with horizontal tail handling)
 * ----------------------------------------------------------------------- */

/* avg_u16: rounded average of two uint16 values, (a+b+1)>>1.
 * Matches the rounding behavior of vrhaddq_u16 (NEON) to keep scalar
 * tail and SIMD paths consistent. */
static inline uint16_t avg_u16(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a + (uint32_t)b + 1) >> 1);
}

/* blend_2_1_u16: bilinear blend for the 3:2 horizontal reduction.
 * (a * 171 + b * 85 + 128) >> 8  ~  a*2/3 + b*1/3.
 * Must use uint32_t: 10-bit a * 171 = max 174,933 which overflows uint16_t. */
static inline uint16_t blend_2_1_u16(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a * 171 + (uint32_t)b * 85 + 128) >> 8);
}

/* div3_u32: integer divide-by-3 for a sum of three uint16 values.
 * (sum * 21846) >> 16 is exact for sum <= 3069 (max 3 * 1023).
 * 3069 * 21846 = 67,027,674 which fits in uint32_t. */
static inline uint16_t div3_u32(uint32_t sum)
{
    return (uint16_t)(((uint32_t)sum * 21846u) >> 16);
}

/* -----------------------------------------------------------------------
 * NEON helpers for 10-bit arithmetic
 * ----------------------------------------------------------------------- */

/* NEON divide-by-3 for a uint16x8_t of sums (each max 3069).
 * Returns uint16x8_t: (sum * 21846) >> 16.
 * Uses widening multiply: vmull_u16 gives uint32x4_t, then vshrn_n_u32
 * narrows back to uint16x4_t. */
static inline uint16x8_t neon_div3_u16x8_hdr(uint16x8_t sum)
{
    uint16x4_t magic = vdup_n_u16(21846);   /* 0x5556 */
    uint32x4_t p_lo = vmull_u16(vget_low_u16(sum),  magic);
    uint32x4_t p_hi = vmull_u16(vget_high_u16(sum), magic);
    uint16x4_t d_lo = vshrn_n_u32(p_lo, 16);
    uint16x4_t d_hi = vshrn_n_u32(p_hi, 16);
    return vcombine_u16(d_lo, d_hi);
}

/* Bilinear blend of two 8-element uint16 registers:
 * (a * 171 + b * 85 + 128) >> 8.
 * Widens to uint32_t because 1023 * 171 = 174,933 overflows uint16_t.
 * Max intermediate: 1023*171 + 1023*85 + 128 = 261,960 + 86,955 + 128 = 349,043
 * which fits in uint32_t. */
static inline uint16x8_t neon_blend_reg_hdr(uint16x8_t a, uint16x8_t b)
{
    uint16x4_t w171 = vdup_n_u16(171);
    uint16x4_t w85  = vdup_n_u16(85);

    /* Low half */
    uint32x4_t lo = vmull_u16(vget_low_u16(a), w171);
    lo = vmlal_u16(lo, vget_low_u16(b), w85);
    lo = vaddq_u32(lo, vdupq_n_u32(128));
    uint16x4_t res_lo = vshrn_n_u32(lo, 8);

    /* High half */
    uint32x4_t hi = vmull_u16(vget_high_u16(a), w171);
    hi = vmlal_u16(hi, vget_high_u16(b), w85);
    hi = vaddq_u32(hi, vdupq_n_u32(128));
    uint16x4_t res_hi = vshrn_n_u32(hi, 8);

    return vcombine_u16(res_lo, res_hi);
}

/* -----------------------------------------------------------------------
 * Horizontal filter functions for 10-bit
 *
 * These are standalone row filters used by the pow2 horizontal cascade
 * and by the thirds tail handling.  The fused thirds chunk loop uses
 * the inline h_chunk_* helpers below instead.
 * ----------------------------------------------------------------------- */

/* Horizontal 1.5x filter (NEON): 3:2 bilinear reduction for 10-bit.
 * Every 3 source uint16_t -> 2 output uint16_t via weighted blend.
 * Processes 24 input elements (8 triplets) -> 16 output elements per NEON chunk.
 *
 * vld3q_u16 loads 24 uint16_t (48 bytes) and deinterleaves into three
 * 8-element vectors: A (first of triplet), B (middle), C (third). */
static void h_filter_1_5x_hdr(const uint16_t *restrict src, int src_w,
                               uint16_t *restrict dst, int dst_w)
{
    int chunks = dst_w / 16;
    uint16x4_t w171 = vdup_n_u16(171);
    uint16x4_t w85  = vdup_n_u16(85);

    for (int c = 0; c < chunks; c++) {
        uint16x8x3_t loaded = vld3q_u16(src + c * 24);
        uint16x8_t A = loaded.val[0];
        uint16x8_t B = loaded.val[1];
        uint16x8_t C = loaded.val[2];

        /* Output 0: (A*171 + B*85 + 128) >> 8 */
        uint32x4_t t0_lo = vmull_u16(vget_low_u16(A), w171);
        t0_lo = vmlal_u16(t0_lo, vget_low_u16(B), w85);
        t0_lo = vaddq_u32(t0_lo, vdupq_n_u32(128));
        uint32x4_t t0_hi = vmull_u16(vget_high_u16(A), w171);
        t0_hi = vmlal_u16(t0_hi, vget_high_u16(B), w85);
        t0_hi = vaddq_u32(t0_hi, vdupq_n_u32(128));
        uint16x8_t out0 = vcombine_u16(vshrn_n_u32(t0_lo, 8),
                                        vshrn_n_u32(t0_hi, 8));

        /* Output 1: (C*171 + B*85 + 128) >> 8 */
        uint32x4_t t1_lo = vmull_u16(vget_low_u16(C), w171);
        t1_lo = vmlal_u16(t1_lo, vget_low_u16(B), w85);
        t1_lo = vaddq_u32(t1_lo, vdupq_n_u32(128));
        uint32x4_t t1_hi = vmull_u16(vget_high_u16(C), w171);
        t1_hi = vmlal_u16(t1_hi, vget_high_u16(B), w85);
        t1_hi = vaddq_u32(t1_hi, vdupq_n_u32(128));
        uint16x8_t out1 = vcombine_u16(vshrn_n_u32(t1_lo, 8),
                                        vshrn_n_u32(t1_hi, 8));

        /* Interleave: [out0[0], out1[0], out0[1], out1[1], ...] */
        uint16x8x2_t interleaved = vzipq_u16(out0, out1);
        vst1q_u16(dst + c * 16,     interleaved.val[0]);
        vst1q_u16(dst + c * 16 + 8, interleaved.val[1]);
    }

    /* Scalar tail */
    int x_out = chunks * 16;
    for (int x_in = chunks * 24; x_in < src_w - 2 && x_out < dst_w - 1;
         x_in += 3, x_out += 2) {
        dst[x_out]     = blend_2_1_u16(src[x_in],     src[x_in + 1]);
        dst[x_out + 1] = blend_2_1_u16(src[x_in + 2], src[x_in + 1]);
    }
}

/* Horizontal 3x filter (NEON): box average of 3 source uint16_t -> 1 output.
 * Processes 24 input elements -> 8 output elements per NEON chunk.
 *
 * vld3q_u16 handles deinterleave.  Sum of three 10-bit values (max 3069)
 * fits in uint16_t, so we can add in 16-bit then use the div3 helper. */
static void h_filter_3x_hdr(const uint16_t *restrict src, int src_w,
                             uint16_t *restrict dst, int dst_w)
{
    (void)src_w;
    int chunks = dst_w / 8;

    for (int c = 0; c < chunks; c++) {
        uint16x8x3_t loaded = vld3q_u16(src + c * 24);

        /* Sum A + B + C (all uint16, sum max 3069 fits in uint16) */
        uint16x8_t sum = vaddq_u16(loaded.val[0], loaded.val[1]);
        sum = vaddq_u16(sum, loaded.val[2]);

        /* Divide by 3: (sum * 21846) >> 16 */
        uint16x8_t result = neon_div3_u16x8_hdr(sum);
        vst1q_u16(dst + c * 8, result);
    }

    /* Scalar tail */
    for (int x = chunks * 8; x < dst_w; x++) {
        uint32_t sum = (uint32_t)src[3 * x]
                     + (uint32_t)src[3 * x + 1]
                     + (uint32_t)src[3 * x + 2];
        dst[x] = div3_u32(sum);
    }
}

/* Horizontal halve filter (NEON): pairwise average for 10-bit.
 * Processes 8 input uint16_t (16 bytes) -> 4 output uint16_t per NEON chunk.
 *
 * vpaddlq_u16 sums adjacent uint16 pairs into uint32 (8 -> 4 elements).
 * vrshrn_n_u32(..., 1) narrows with rounding: (sum+1)>>1. */
static void h_filter_halve_hdr(const uint16_t *restrict src,
                                uint16_t *restrict dst, int dst_w)
{
    int src_elems = dst_w * 2;
    int n_chunks = src_elems / 8;   /* 8 uint16_t per Q register load */
    int out_x = 0;

    for (int c = 0; c < n_chunks; c++) {
        uint16x8_t v = vld1q_u16(src + c * 8);
        uint32x4_t pair_sum = vpaddlq_u16(v);
        uint16x4_t result = vrshrn_n_u32(pair_sum, 1);
        vst1_u16(dst + out_x, result);
        out_x += 4;
    }

    /* Scalar tail */
    int tail_in = n_chunks * 8;
    for (int tx = tail_in; tx + 1 < src_elems; tx += 2) {
        dst[out_x++] = avg_u16(src[tx], src[tx + 1]);
    }
}

/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single 10-bit plane (NEON)
 *
 * Vertical: NEON vrhaddq_u16 cascade
 * Horizontal: NEON vpaddlq_u16 + vrshrn_n_u32 cascade
 *
 * Same algorithm as the 8-bit scale_plane_pow2_neon but operating on
 * uint16_t elements with 8 elements per Q register instead of 16.
 * ----------------------------------------------------------------------- */

static void __attribute__((hot)) scale_plane_pow2_hdr_neon(
    const uint16_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint16_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4])
{
    (void)dst_heights;

    /* Convert byte stride to element stride */
    int src_el_stride = src_stride / (int)sizeof(uint16_t);

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

    /* Allocate vertical intermediate row buffers (uint16_t elements). */
    uint16_t *vert_buf[4] = { NULL, NULL, NULL, NULL };
    int vert_rows[4];

    for (int k = 0; k <= deepest; k++) {
        vert_rows[k] = group_rows >> (k + 1);
        vert_buf[k] = (uint16_t *)malloc((size_t)vert_rows[k] * (size_t)src_w * sizeof(uint16_t));
        if (!vert_buf[k]) {
            for (int j = 0; j < k; j++) free(vert_buf[j]);
            return;
        }
    }

    /* Horizontal cascade buffer: max input width = src_w elements. */
    uint16_t *h_buf = (uint16_t *)malloc((size_t)src_w * sizeof(uint16_t));
    if (!h_buf) {
        for (int k = 0; k <= deepest; k++) free(vert_buf[k]);
        return;
    }

    int out_row[4] = { 0, 0, 0, 0 };

    /* NEON chunk count: 8 uint16_t elements per Q register */
    int neon_chunks = src_w / 8;

    for (int g = 0; g < num_groups; g++) {
        const uint16_t *grp_base = src + (size_t)g * (size_t)group_rows * (size_t)src_el_stride;

        /* -- Vertical cascade (NEON) --------------------------------- */

        /* Level 0 (2x vertical): pairwise average source rows */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint16_t *restrict ra = grp_base + (size_t)(2 * r)     * (size_t)src_el_stride;
            const uint16_t *restrict rb = grp_base + (size_t)(2 * r + 1) * (size_t)src_el_stride;
            uint16_t *restrict dst_row  = vert_buf[0] + (size_t)r * (size_t)src_w;

            int x = 0;
            for (int c = 0; c < neon_chunks; c++, x += 8) {
                uint16x8_t va = vld1q_u16(ra + x);
                uint16x8_t vb = vld1q_u16(rb + x);
                vst1q_u16(dst_row + x, vrhaddq_u16(va, vb));
            }
            for (; x < src_w; x++) {
                dst_row[x] = avg_u16(ra[x], rb[x]);
            }
        }

        /* Deeper levels: pairwise average previous level */
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict ra = vert_buf[k - 1] + (size_t)(2 * r)     * (size_t)src_w;
                const uint16_t *restrict rb = vert_buf[k - 1] + (size_t)(2 * r + 1) * (size_t)src_w;
                uint16_t *restrict dst_row  = vert_buf[k] + (size_t)r * (size_t)src_w;

                int x = 0;
                for (int c = 0; c < neon_chunks; c++, x += 8) {
                    uint16x8_t va = vld1q_u16(ra + x);
                    uint16x8_t vb = vld1q_u16(rb + x);
                    vst1q_u16(dst_row + x, vrhaddq_u16(va, vb));
                }
                for (; x < src_w; x++) {
                    dst_row[x] = avg_u16(ra[x], rb[x]);
                }
            }
        }

        /* -- Horizontal cascade (NEON) + output write ---------------- */

        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;

            /* Convert destination byte stride to element stride */
            int dst_el_stride = dst_strides[k] / (int)sizeof(uint16_t);

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict vert_row = vert_buf[k] + (size_t)r * (size_t)src_w;

                /* Horizontal cascade: (k+1) halvings.
                 * Each halving: out[x] = avg(in[2x], in[2x+1]).
                 * Use NEON: vpaddlq_u16 for adjacent pair sums -> u32,
                 * then vrshrn_n_u32 to narrow with rounding. */
                int cur_w = src_w;
                const uint16_t *cur_src = vert_row;

                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    int next_w = cur_w >> 1;
                    int neon_in_chunks = cur_w / 8;
                    int out_x = 0;

                    for (int c = 0; c < neon_in_chunks; c++) {
                        uint16x8_t v = vld1q_u16(cur_src + c * 8);
                        uint32x4_t pair_sum = vpaddlq_u16(v);
                        uint16x4_t result = vrshrn_n_u32(pair_sum, 1);
                        /* Store 4 output elements */
                        vst1_u16(h_buf + out_x, result);
                        out_x += 4;
                    }
                    /* Scalar tail */
                    int tail_in = neon_in_chunks * 8;
                    for (int tx = tail_in; tx + 1 < cur_w; tx += 2) {
                        h_buf[out_x++] = avg_u16(cur_src[tx], cur_src[tx + 1]);
                    }

                    cur_w = next_w;
                    cur_src = h_buf;
                }

                /* Write to output plane */
                uint16_t *restrict out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_el_stride;
                memcpy(out, h_buf, (size_t)dst_widths[k] * sizeof(uint16_t));
                out_row[k]++;
            }
        }
    }

    free(h_buf);
    for (int k = 0; k <= deepest; k++) free(vert_buf[k]);
}


/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single 10-bit plane (NEON fused vertical+horizontal)
 *
 * Per-chunk fused architecture: source rows are processed in groups of 6
 * (matching the vertical period of the thirds reduction).  For each 48-byte
 * column chunk (= 24 uint16_t = 8 triplets), all 6 rows are loaded,
 * vertical pair averages and bilinear blends are computed entirely in
 * NEON registers, and horizontal filtering is applied immediately.
 *
 * Key differences from the 8-bit version:
 *   - Q registers hold 8 uint16_t elements (not 16 uint8_t).
 *   - A 48-byte chunk covers 8 triplets (not 16).
 *   - vld3q_u16 deinterleaves 24 uint16_t into three 8-element vectors.
 *   - Output counts per chunk:
 *       1.5x: 16 elements (vs 32 bytes in 8-bit)
 *       3x:    8 elements (vs 16 bytes in 8-bit)
 *       6x:    4 elements (vs 8 bytes in 8-bit)
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Fused vertical+horizontal chunk helpers for 10-bit
 *
 * These inline functions perform horizontal filtering on a 48-byte chunk
 * (3 x uint16x8_t = 24 elements) that has already been vertically
 * reduced in registers.
 *
 * The store+reload through a 48-byte stack buffer enables vld3q_u16
 * hardware deinterleave (essentially free on Apple Silicon L1).
 * ----------------------------------------------------------------------- */

/* Horizontal 1.5x on one deinterleaved 10-bit chunk (A, B, C each 8 elements).
 * Produces 16 output uint16_t stored at dst.
 * Blend weights: out0 = (A*171 + B*85 + 128) >> 8
 *                out1 = (C*171 + B*85 + 128) >> 8 */
static inline void h_chunk_1_5x_hdr(uint16x8_t A, uint16x8_t B, uint16x8_t C,
                                     uint16_t *restrict dst)
{
    uint16x4_t w171 = vdup_n_u16(171);
    uint16x4_t w85  = vdup_n_u16(85);

    /* Output 0: pixel at 1/3 position, weighted toward A */
    uint32x4_t t0_lo = vmull_u16(vget_low_u16(A), w171);
    t0_lo = vmlal_u16(t0_lo, vget_low_u16(B), w85);
    t0_lo = vaddq_u32(t0_lo, vdupq_n_u32(128));
    uint32x4_t t0_hi = vmull_u16(vget_high_u16(A), w171);
    t0_hi = vmlal_u16(t0_hi, vget_high_u16(B), w85);
    t0_hi = vaddq_u32(t0_hi, vdupq_n_u32(128));
    uint16x8_t out0 = vcombine_u16(vshrn_n_u32(t0_lo, 8),
                                    vshrn_n_u32(t0_hi, 8));

    /* Output 1: pixel at 2/3 position, weighted toward C */
    uint32x4_t t1_lo = vmull_u16(vget_low_u16(C), w171);
    t1_lo = vmlal_u16(t1_lo, vget_low_u16(B), w85);
    t1_lo = vaddq_u32(t1_lo, vdupq_n_u32(128));
    uint32x4_t t1_hi = vmull_u16(vget_high_u16(C), w171);
    t1_hi = vmlal_u16(t1_hi, vget_high_u16(B), w85);
    t1_hi = vaddq_u32(t1_hi, vdupq_n_u32(128));
    uint16x8_t out1 = vcombine_u16(vshrn_n_u32(t1_lo, 8),
                                    vshrn_n_u32(t1_hi, 8));

    /* Interleave: [out0[0], out1[0], out0[1], out1[1], ...] */
    uint16x8x2_t interleaved = vzipq_u16(out0, out1);
    vst1q_u16(dst,     interleaved.val[0]);
    vst1q_u16(dst + 8, interleaved.val[1]);
}

/* Horizontal 3x on one deinterleaved 10-bit chunk (A, B, C each 8 elements).
 * Produces 8 output uint16_t stored at dst.  Returns the result register.
 * Sum of three 10-bit values (max 3069) fits in uint16_t. */
static inline uint16x8_t h_chunk_3x_hdr(uint16x8_t A, uint16x8_t B, uint16x8_t C,
                                          uint16_t *restrict dst)
{
    uint16x8_t sum = vaddq_u16(A, B);
    sum = vaddq_u16(sum, C);
    uint16x8_t result = neon_div3_u16x8_hdr(sum);
    vst1q_u16(dst, result);
    return result;
}

/* Horizontal 6x cascaded from a 3x result (8 elements -> 4 elements).
 * Pairwise average: vpaddlq_u16 -> uint32x4, then vrshrn_n_u32 to narrow.
 * Stores 4 output uint16_t at dst.  Returns the 4-element result. */
static inline uint16x4_t h_chunk_6x_hdr(uint16x8_t result_3x,
                                          uint16_t *restrict dst)
{
    uint32x4_t pair_sum = vpaddlq_u16(result_3x);
    uint16x4_t result = vrshrn_n_u32(pair_sum, 1);
    vst1_u16(dst, result);
    return result;
}

/* Deinterleave a 48-byte chunk (3 x uint16x8_t = 24 elements) using vld3q_u16.
 *
 * vld3q_u16 loads 24 consecutive uint16_t and deinterleaves them into three
 * 8-element vectors.  Since vld3q_u16 takes a memory address, we store the
 * three vertically-blended registers to a stack buffer and reload. */
static inline uint16x8x3_t deinterleave_chunk_hdr(
    uint16x8_t a, uint16x8_t b, uint16x8_t c,
    uint16_t chunk_buf[24])
{
    vst1q_u16(chunk_buf,      a);
    vst1q_u16(chunk_buf + 8,  b);
    vst1q_u16(chunk_buf + 16, c);
    return vld3q_u16(chunk_buf);
}


static void __attribute__((hot)) scale_plane_thirds_hdr_neon(
    const uint16_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint16_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4])
{
    (void)dst_heights;

    /* Convert byte stride to element stride */
    int src_el_stride = src_stride / (int)sizeof(uint16_t);

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
    size_t row_bytes = (size_t)src_w * sizeof(uint16_t);

    /* For 12x: two buffers to hold 6x vertical intermediates across
     * consecutive 6-row groups.  Ping-pong swap avoids copying. */
    uint16_t *v6x_buf_a = NULL, *v6x_buf_b = NULL;
    uint16_t *v6x_cur = NULL, *v6x_prev = NULL;
    if (need_12x) {
        if (posix_memalign((void **)&v6x_buf_a, 32, row_bytes) != 0 ||
            posix_memalign((void **)&v6x_buf_b, 32, row_bytes) != 0) {
            free(v6x_buf_a);
            free(v6x_buf_b);
            return;
        }
        v6x_cur  = v6x_buf_a;
        v6x_prev = v6x_buf_b;
    }

    /* Horizontal scratch for 12x (3x -> halve -> halve) */
    int w_3x = src_w / 3;
    int w_6x = w_3x / 2;
    uint16_t *h_3x_buf = NULL, *h_6x_buf = NULL;
    if (need_12x && (active_outputs & (1u << 6))) {
        if (posix_memalign((void **)&h_3x_buf, 32, (size_t)w_3x * sizeof(uint16_t)) != 0 ||
            posix_memalign((void **)&h_6x_buf, 32, (size_t)(w_6x > 0 ? w_6x : 1) * sizeof(uint16_t)) != 0) {
            free(v6x_buf_a); free(v6x_buf_b);
            free(h_3x_buf); free(h_6x_buf);
            return;
        }
    }

    /* Chunk geometry: 48 bytes = 24 uint16_t = 8 triplets per chunk.
     * Each chunk of 24 elements covers 3 Q registers (3 x 8 elements). */
    int full_chunks = src_w / 24;
    int tail_start  = full_chunks * 24;
    int tail_cols   = src_w - tail_start;

    /* Deinterleave buffer (stack-allocated, 24 uint16_t = 48 bytes) */
    uint16_t __attribute__((aligned(16))) chunk_buf[24];

    /* Output row cursors */
    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint16_t *grp = src + (size_t)g6 * 6 * (size_t)src_el_stride;

        const uint16_t *restrict row0 = grp;
        const uint16_t *restrict row1 = grp + (size_t)src_el_stride;
        const uint16_t *restrict row2 = grp + (size_t)2 * (size_t)src_el_stride;
        const uint16_t *restrict row3 = grp + (size_t)3 * (size_t)src_el_stride;
        const uint16_t *restrict row4 = grp + (size_t)4 * (size_t)src_el_stride;
        const uint16_t *restrict row5 = grp + (size_t)5 * (size_t)src_el_stride;

        /* Compute output row base pointers (element pointers, not byte) */
        uint16_t *dst_1_5x_r0 = NULL, *dst_1_5x_r1 = NULL;
        uint16_t *dst_1_5x_r2 = NULL, *dst_1_5x_r3 = NULL;
        if (need_1_5x) {
            int ds_el = dst_strides[0] / (int)sizeof(uint16_t);
            dst_1_5x_r0 = dst_planes[0] + (size_t)out_row[0]       * (size_t)ds_el;
            dst_1_5x_r1 = dst_planes[0] + (size_t)(out_row[0] + 1) * (size_t)ds_el;
            dst_1_5x_r2 = dst_planes[0] + (size_t)(out_row[0] + 2) * (size_t)ds_el;
            dst_1_5x_r3 = dst_planes[0] + (size_t)(out_row[0] + 3) * (size_t)ds_el;
        }

        uint16_t *dst_3x_r0 = NULL, *dst_3x_r1 = NULL;
        if (active_outputs & (1u << 2)) {
            int ds_el = dst_strides[1] / (int)sizeof(uint16_t);
            dst_3x_r0 = dst_planes[1] + (size_t)out_row[1]       * (size_t)ds_el;
            dst_3x_r1 = dst_planes[1] + (size_t)(out_row[1] + 1) * (size_t)ds_el;
        }

        uint16_t *dst_6x_r0 = NULL;
        if (active_outputs & (1u << 4)) {
            int ds_el = dst_strides[2] / (int)sizeof(uint16_t);
            dst_6x_r0 = dst_planes[2] + (size_t)out_row[2] * (size_t)ds_el;
        }

        /* ============================================================
         * MAIN CHUNK LOOP: process 24 source elements (8 triplets) at
         * a time.  Each chunk is 3 Q registers of uint16x8_t.
         *
         * Vertical intermediates stay in NEON registers; horizontal
         * filtering is applied immediately per chunk.  A 6-source-row
         * group produces 4 output rows at 1.5x, 2 at 3x, and 1 at 6x.
         * ============================================================ */
#if defined(__clang__)
        #pragma clang loop unroll_count(2)
#elif defined(__GNUC__)
        #pragma GCC unroll 2
#endif
        for (int ci = 0; ci < full_chunks; ci++) {
            int cx = ci * 24;           /* element offset into source row */
            int out_off_1_5x = ci * 16; /* 24 -> 16 output elements */
            int out_off_3x   = ci * 8;  /* 24 ->  8 output elements */
            int out_off_6x   = ci * 4;  /* 24 ->  4 output elements */

            /* --- LOAD 6 rows x 3 registers = 18 loads ---
             * Each row needs 3 vld1q_u16 for 24 uint16_t elements.
             * Loading all before arithmetic allows OoO overlap. */
            uint16x8_t r0a = vld1q_u16(row0 + cx);
            uint16x8_t r0b = vld1q_u16(row0 + cx + 8);
            uint16x8_t r0c = vld1q_u16(row0 + cx + 16);
            uint16x8_t r1a = vld1q_u16(row1 + cx);
            uint16x8_t r1b = vld1q_u16(row1 + cx + 8);
            uint16x8_t r1c = vld1q_u16(row1 + cx + 16);
            uint16x8_t r2a = vld1q_u16(row2 + cx);
            uint16x8_t r2b = vld1q_u16(row2 + cx + 8);
            uint16x8_t r2c = vld1q_u16(row2 + cx + 16);
            uint16x8_t r3a = vld1q_u16(row3 + cx);
            uint16x8_t r3b = vld1q_u16(row3 + cx + 8);
            uint16x8_t r3c = vld1q_u16(row3 + cx + 16);
            uint16x8_t r4a = vld1q_u16(row4 + cx);
            uint16x8_t r4b = vld1q_u16(row4 + cx + 8);
            uint16x8_t r4c = vld1q_u16(row4 + cx + 16);
            uint16x8_t r5a = vld1q_u16(row5 + cx);
            uint16x8_t r5b = vld1q_u16(row5 + cx + 8);
            uint16x8_t r5c = vld1q_u16(row5 + cx + 16);

            /* --- VERTICAL PAIRWISE AVERAGES (in registers) ---
             * Average adjacent row pairs using vrhaddq_u16 (rounded). */
            uint16x8_t v01a = vrhaddq_u16(r0a, r1a);
            uint16x8_t v01b = vrhaddq_u16(r0b, r1b);
            uint16x8_t v01c = vrhaddq_u16(r0c, r1c);
            uint16x8_t v23a = vrhaddq_u16(r2a, r3a);
            uint16x8_t v23b = vrhaddq_u16(r2b, r3b);
            uint16x8_t v23c = vrhaddq_u16(r2c, r3c);
            uint16x8_t v45a = vrhaddq_u16(r4a, r5a);
            uint16x8_t v45b = vrhaddq_u16(r4b, r5b);
            uint16x8_t v45c = vrhaddq_u16(r4c, r5c);

            /* --- 1.5x OUTPUT (4 rows) ---
             * Rows 0 and 3 come from v01 and v45 directly.
             * Rows 1 and 2 are bilinear blends between adjacent pair
             * averages: blend(v01,v23) and blend(v23,v45).
             * Blend uses neon_blend_reg_hdr which widens to uint32_t. */
            if (need_1_5x) {
                uint16x8x3_t d;

                /* Row 0: v01 */
                d = deinterleave_chunk_hdr(v01a, v01b, v01c, chunk_buf);
                h_chunk_1_5x_hdr(d.val[0], d.val[1], d.val[2],
                                  dst_1_5x_r0 + out_off_1_5x);

                /* Row 1: blend(v01, v23) */
                {
                    uint16x8_t ba = neon_blend_reg_hdr(v01a, v23a);
                    uint16x8_t bb = neon_blend_reg_hdr(v01b, v23b);
                    uint16x8_t bc = neon_blend_reg_hdr(v01c, v23c);
                    d = deinterleave_chunk_hdr(ba, bb, bc, chunk_buf);
                    h_chunk_1_5x_hdr(d.val[0], d.val[1], d.val[2],
                                      dst_1_5x_r1 + out_off_1_5x);
                }

                /* Row 2: blend(v23, v45) */
                {
                    uint16x8_t ba = neon_blend_reg_hdr(v23a, v45a);
                    uint16x8_t bb = neon_blend_reg_hdr(v23b, v45b);
                    uint16x8_t bc = neon_blend_reg_hdr(v23c, v45c);
                    d = deinterleave_chunk_hdr(ba, bb, bc, chunk_buf);
                    h_chunk_1_5x_hdr(d.val[0], d.val[1], d.val[2],
                                      dst_1_5x_r2 + out_off_1_5x);
                }

                /* Row 3: v45 */
                d = deinterleave_chunk_hdr(v45a, v45b, v45c, chunk_buf);
                h_chunk_1_5x_hdr(d.val[0], d.val[1], d.val[2],
                                  dst_1_5x_r3 + out_off_1_5x);
            }

            /* --- 3x VERTICAL + HORIZONTAL (2 rows) ---
             * avg(v01,v23) and avg(v23,v45), then 3:1 box average. */
            uint16x8_t v3x0a = vdupq_n_u16(0), v3x0b = vdupq_n_u16(0), v3x0c = vdupq_n_u16(0);
            uint16x8_t v3x1a = vdupq_n_u16(0), v3x1b = vdupq_n_u16(0), v3x1c = vdupq_n_u16(0);
            if (need_3x) {
                v3x0a = vrhaddq_u16(v01a, v23a);
                v3x0b = vrhaddq_u16(v01b, v23b);
                v3x0c = vrhaddq_u16(v01c, v23c);
                v3x1a = vrhaddq_u16(v23a, v45a);
                v3x1b = vrhaddq_u16(v23b, v45b);
                v3x1c = vrhaddq_u16(v23c, v45c);

                if (active_outputs & (1u << 2)) {
                    uint16x8x3_t d;

                    d = deinterleave_chunk_hdr(v3x0a, v3x0b, v3x0c, chunk_buf);
                    h_chunk_3x_hdr(d.val[0], d.val[1], d.val[2],
                                    dst_3x_r0 + out_off_3x);

                    d = deinterleave_chunk_hdr(v3x1a, v3x1b, v3x1c, chunk_buf);
                    h_chunk_3x_hdr(d.val[0], d.val[1], d.val[2],
                                    dst_3x_r1 + out_off_3x);
                }
            }

            /* --- 6x VERTICAL (1 row) + save for 12x + 6x horizontal ---
             * Average the two 3x intermediates to get one 6x row. */
            if (need_6x) {
                uint16x8_t v6xa = vrhaddq_u16(v3x0a, v3x1a);
                uint16x8_t v6xb = vrhaddq_u16(v3x0b, v3x1b);
                uint16x8_t v6xc = vrhaddq_u16(v3x0c, v3x1c);

                /* Save 6x vertical intermediate for 12x pairing */
                if (need_12x) {
                    vst1q_u16(v6x_cur + cx,      v6xa);
                    vst1q_u16(v6x_cur + cx + 8,  v6xb);
                    vst1q_u16(v6x_cur + cx + 16, v6xc);
                }

                if (active_outputs & (1u << 4)) {
                    uint16x8x3_t d = deinterleave_chunk_hdr(v6xa, v6xb, v6xc, chunk_buf);
                    uint16x8_t r3x = h_chunk_3x_hdr(d.val[0], d.val[1], d.val[2],
                                                      (uint16_t *)chunk_buf);
                    h_chunk_6x_hdr(r3x, dst_6x_r0 + out_off_6x);
                }
            }
        } /* end chunk loop */

        /* ============================================================
         * TAIL: handle remaining columns with scalar h_filter functions
         * ============================================================ */
        if (tail_cols > 0) {
            /* Tail buffers: max tail_cols < 24 elements */
            uint16_t tail_v01[24], tail_v23[24], tail_v45[24];

            /* Compute vertical intermediates for tail */
            for (int x = 0; x < tail_cols; x++) {
                int sx = tail_start + x;
                tail_v01[x] = avg_u16(row0[sx], row1[sx]);
                tail_v23[x] = avg_u16(row2[sx], row3[sx]);
                tail_v45[x] = avg_u16(row4[sx], row5[sx]);
            }

            uint16_t tail_v3x0[24], tail_v3x1[24];
            if (need_3x) {
                for (int x = 0; x < tail_cols; x++) {
                    tail_v3x0[x] = avg_u16(tail_v01[x], tail_v23[x]);
                    tail_v3x1[x] = avg_u16(tail_v23[x], tail_v45[x]);
                }
            }

            uint16_t tail_v6x[24];
            if (need_6x) {
                for (int x = 0; x < tail_cols; x++) {
                    tail_v6x[x] = avg_u16(tail_v3x0[x], tail_v3x1[x]);
                }
                if (need_12x) {
                    memcpy(v6x_cur + tail_start, tail_v6x,
                           (size_t)tail_cols * sizeof(uint16_t));
                }
            }

            /* How many output elements the NEON chunks already produced */
            int tail_out_1_5x = full_chunks * 16;
            int tail_out_3x   = full_chunks * 8;
            int tail_out_6x   = full_chunks * 4;

            /* 1.5x tail */
            if (need_1_5x) {
                int dw_rem = dst_widths[0] - tail_out_1_5x;
                h_filter_1_5x_hdr(tail_v01, tail_cols,
                                   dst_1_5x_r0 + tail_out_1_5x, dw_rem);

                uint16_t tail_blend[24];
                for (int x = 0; x < tail_cols; x++)
                    tail_blend[x] = blend_2_1_u16(tail_v01[x], tail_v23[x]);
                h_filter_1_5x_hdr(tail_blend, tail_cols,
                                   dst_1_5x_r1 + tail_out_1_5x, dw_rem);

                for (int x = 0; x < tail_cols; x++)
                    tail_blend[x] = blend_2_1_u16(tail_v23[x], tail_v45[x]);
                h_filter_1_5x_hdr(tail_blend, tail_cols,
                                   dst_1_5x_r2 + tail_out_1_5x, dw_rem);

                h_filter_1_5x_hdr(tail_v45, tail_cols,
                                   dst_1_5x_r3 + tail_out_1_5x, dw_rem);
            }

            /* 3x tail */
            if (active_outputs & (1u << 2)) {
                int dw_rem = dst_widths[1] - tail_out_3x;
                h_filter_3x_hdr(tail_v3x0, tail_cols,
                                 dst_3x_r0 + tail_out_3x, dw_rem);
                h_filter_3x_hdr(tail_v3x1, tail_cols,
                                 dst_3x_r1 + tail_out_3x, dw_rem);
            }

            /* 6x tail */
            if (active_outputs & (1u << 4)) {
                int dw_rem = dst_widths[2] - tail_out_6x;
                int w3_tail = tail_cols / 3;
                uint16_t tail_h3x[8];
                h_filter_3x_hdr(tail_v6x, tail_cols, tail_h3x, w3_tail);
                h_filter_halve_hdr(tail_h3x, dst_6x_r0 + tail_out_6x, dw_rem);
            }
        }

        /* Update output row cursors */
        if (need_1_5x) out_row[0] += 4;
        if (active_outputs & (1u << 2)) out_row[1] += 2;
        if (active_outputs & (1u << 4)) out_row[2] += 1;

        /* ============================================================
         * 12x handling: pair two consecutive 6-row groups.
         * v6x_cur holds this group's 6x vertical intermediate (src_w
         * uint16_t elements).
         * On even groups: swap pointers so current becomes previous.
         * On odd groups: average prev with current, apply horizontal, output.
         *
         * 12x requires averaging two consecutive 6x rows, each derived
         * from a different 6-row source group.  The ping-pong swap
         * avoids copying.
         * ============================================================ */
        if (need_12x) {
            if ((g6 & 1) == 0) {
                /* Swap so v6x_cur data becomes v6x_prev for the next group */
                uint16_t *tmp = v6x_prev;
                v6x_prev = v6x_cur;
                v6x_cur  = tmp;
            } else {
                /* Average v6x_prev (even group) with v6x_cur (odd group) */
                int neon8 = src_w / 8;
                int x = 0;
                for (int c = 0; c < neon8; c++, x += 8) {
                    uint16x8_t a = vld1q_u16(v6x_prev + x);
                    uint16x8_t b = vld1q_u16(v6x_cur + x);
                    vst1q_u16(v6x_prev + x, vrhaddq_u16(a, b));
                }
                for (; x < src_w; x++) {
                    v6x_prev[x] = avg_u16(v6x_prev[x], v6x_cur[x]);
                }

                if (active_outputs & (1u << 6)) {
                    int dw = dst_widths[3];
                    int ds_el = dst_strides[3] / (int)sizeof(uint16_t);

                    /* Horizontal: 3x box avg -> halve -> halve = 12:1 */
                    h_filter_3x_hdr(v6x_prev, src_w, h_3x_buf, w_3x);
                    h_filter_halve_hdr(h_3x_buf, h_6x_buf, w_6x);
                    h_filter_halve_hdr(h_6x_buf,
                                        dst_planes[3] + (size_t)out_row[3] * (size_t)ds_el, dw);
                    out_row[3]++;
                }
            }
        }
    } /* end g6 loop */

    free(v6x_buf_a); free(v6x_buf_b);
    free(h_3x_buf);  free(h_6x_buf);
}


/* -----------------------------------------------------------------------
 * Public entry points
 *
 * For I010 (planar): call scale_plane_*_hdr_neon() three times (Y, U, V).
 * For P010 (semi-planar): Y is separate; U and V are interleaved in src_u
 * as UVUV... pairs.  We deinterleave the entire UV plane into temporary
 * planar buffers using NEON vld2q_u16 (loads 16 uint16_t and deinterleaves
 * into two 8-element vectors), then process U and V identically to I010.
 * ----------------------------------------------------------------------- */

void __attribute__((hot)) fused_kernel_pow2_hdr_neon(
    const fused_hdr_kernel_params_t *p,
    const uint16_t *src_y,
    const uint16_t *src_u,
    const uint16_t *src_v)
{
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    uint16_t *y_planes[4], *u_planes[4], *v_planes[4];
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
    scale_plane_pow2_hdr_neon(src_y,
                               p->src_width, p->src_height, p->src_y_stride,
                               p->active_outputs,
                               y_planes, y_widths, y_strides, y_heights);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        /* P010: src_u points to interleaved UV plane (UVUV...).
         * Deinterleave using NEON vld2q_u16 into pre-allocated buffers,
         * then process each plane normally. */
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        /* NEON deinterleave: vld2q_u16 loads 16 uint16_t and separates
         * into even (U) and odd (V) elements. */
        for (int y = 0; y < chroma_h; y++) {
            const uint16_t *row = src_u + (size_t)y * (size_t)uv_el_stride;
            uint16_t *u_row = p->p010_tmp_u + (size_t)y * (size_t)planar_el_stride;
            uint16_t *v_row = p->p010_tmp_v + (size_t)y * (size_t)planar_el_stride;

            int neon_pairs = chroma_w / 8;
            int x = 0;
            for (int c = 0; c < neon_pairs; c++, x += 8) {
                uint16x8x2_t uv = vld2q_u16(row + 2 * x);
                vst1q_u16(u_row + x, uv.val[0]);
                vst1q_u16(v_row + x, uv.val[1]);
            }
            /* Scalar tail */
            for (; x < chroma_w; x++) {
                u_row[x] = row[2 * x];
                v_row[x] = row[2 * x + 1];
            }
        }

        scale_plane_pow2_hdr_neon(p->p010_tmp_u,
                                   chroma_w, chroma_h, p->p010_tmp_stride,
                                   p->active_outputs,
                                   u_planes, uv_widths, uv_strides, uv_heights);

        scale_plane_pow2_hdr_neon(p->p010_tmp_v,
                                   chroma_w, chroma_h, p->p010_tmp_stride,
                                   p->active_outputs,
                                   v_planes, uv_widths, uv_strides, uv_heights);
        /* No free — buffers are owned by the context */
    } else {
        /* I010: separate U and V planes */
        scale_plane_pow2_hdr_neon(src_u,
                                   chroma_w, chroma_h, p->src_uv_stride,
                                   p->active_outputs,
                                   u_planes, uv_widths, uv_strides, uv_heights);

        scale_plane_pow2_hdr_neon(src_v,
                                   chroma_w, chroma_h, p->src_uv_stride,
                                   p->active_outputs,
                                   v_planes, uv_widths, uv_strides, uv_heights);
    }
}


void __attribute__((hot)) fused_kernel_thirds_hdr_neon(
    const fused_hdr_kernel_params_t *p,
    const uint16_t *src_y,
    const uint16_t *src_u,
    const uint16_t *src_v)
{
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    uint16_t *y_planes[4], *u_planes[4], *v_planes[4];
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
    scale_plane_thirds_hdr_neon(src_y,
                                 p->src_width, p->src_height, p->src_y_stride,
                                 p->active_outputs,
                                 y_planes, y_widths, y_strides, y_heights);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        /* P010: deinterleave UV plane into pre-allocated planar buffers */
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        /* NEON deinterleave: vld2q_u16 */
        for (int y = 0; y < chroma_h; y++) {
            const uint16_t *row = src_u + (size_t)y * (size_t)uv_el_stride;
            uint16_t *u_row = p->p010_tmp_u + (size_t)y * (size_t)planar_el_stride;
            uint16_t *v_row = p->p010_tmp_v + (size_t)y * (size_t)planar_el_stride;

            int neon_pairs = chroma_w / 8;
            int x = 0;
            for (int c = 0; c < neon_pairs; c++, x += 8) {
                uint16x8x2_t uv = vld2q_u16(row + 2 * x);
                vst1q_u16(u_row + x, uv.val[0]);
                vst1q_u16(v_row + x, uv.val[1]);
            }
            for (; x < chroma_w; x++) {
                u_row[x] = row[2 * x];
                v_row[x] = row[2 * x + 1];
            }
        }

        scale_plane_thirds_hdr_neon(p->p010_tmp_u,
                                     chroma_w, chroma_h, p->p010_tmp_stride,
                                     p->active_outputs,
                                     u_planes, uv_widths, uv_strides, uv_heights);

        scale_plane_thirds_hdr_neon(p->p010_tmp_v,
                                     chroma_w, chroma_h, p->p010_tmp_stride,
                                     p->active_outputs,
                                     v_planes, uv_widths, uv_strides, uv_heights);
        /* No free — buffers are owned by the context */
    } else {
        /* I010: separate U and V planes */
        scale_plane_thirds_hdr_neon(src_u,
                                     chroma_w, chroma_h, p->src_uv_stride,
                                     p->active_outputs,
                                     u_planes, uv_widths, uv_strides, uv_heights);

        scale_plane_thirds_hdr_neon(src_v,
                                     chroma_w, chroma_h, p->src_uv_stride,
                                     p->active_outputs,
                                     v_planes, uv_widths, uv_strides, uv_heights);
    }
}

#endif /* __aarch64__ */
