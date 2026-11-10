/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_avx2.c - AVX2 (x86_64) fused downscale kernels.
 *
 * Two entry points:
 *   fused_kernel_pow2_avx2   - power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_avx2 - thirds family (1.5x/3x/6x/12x)
 *
 * Both process YUV420 I420 frames plane-by-plane.
 *
 * "Fused" means vertical and horizontal reduction happen in the same pass
 * over source memory, rather than scaling vertically into an intermediate
 * full-width row buffer and then horizontally in a second pass.
 *
 * The thirds kernel (1.5x/3x/6x/12x) reads source rows in groups of 6,
 * which matches the vertical period of the thirds reduction.  All 6 rows
 * are loaded simultaneously into YMM registers so the vertical intermediates
 * (pair averages and bilinear blends) never need to be written to memory.
 * For each 96-byte column chunk, horizontal filtering is applied immediately
 * before moving to the next chunk, so no intermediate row buffer is needed.
 *
 * The pow2 kernel (2x/4x/8x/16x) keeps its vertical levels as separate
 * pairwise-average passes, avoiding the register pressure of a deep reduction
 * tree.  As each vertical vector is produced, however, its horizontal cascade
 * is completed immediately in registers; only vertical data needed by a
 * deeper level is written to scratch.
 *
 * Guarded by __x86_64__ so this file is a no-op on other platforms.
 */

#if defined(__x86_64__)

#include "internal.h"
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * Scalar helpers (used for tail bytes and horizontal thirds phase)
 * ----------------------------------------------------------------------- */

/* avg_u8: rounded average of two bytes, (a+b+1)>>1.  The +1 causes ties to
 * round up, which matches the rounding behavior of vpavgb (x86) and
 * vrhaddq_u8 (NEON).  Keeping scalar and SIMD paths consistent matters for
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
 * SSE shuffle tables for 3-way deinterleave (used by horizontal thirds)
 *
 * The thirds horizontal filter needs to operate separately on the first,
 * second, and third pixel of each source triplet (A, B, C respectively)
 * as SIMD vectors.  In memory the bytes are laid out ABCABCABCABC...  To
 * process 16 output pixels we read 48 source bytes (16 triplets) and must
 * separate them into three 16-element vectors.
 *
 * _mm_shuffle_epi8 can only move bytes within a single 128-bit register.
 * The 48 bytes span three registers (r0, r1, r2), and each component's
 * bytes are spread across all three depending on where in the ABC pattern
 * they fall.  The solution is to apply a separate shuffle to each register
 * with 0x80 zeroing the positions that belong to the other two registers,
 * then OR the three partial results into the final vector.
 *
 * Given 48 contiguous bytes in three 128-bit registers (r0, r1, r2),
 * extract three 16-byte component vectors:
 *   A[g] = src[3g], B[g] = src[3g+1], C[g] = src[3g+2]  for g=0..15
 *
 * The 0x80 value in _mm_shuffle_epi8 zeroes that output position, so OR
 * cleanly merges the three partial results.
 * ----------------------------------------------------------------------- */

#define ALIGN16 __attribute__((aligned(16)))

/* Component A: every 3rd byte starting at offset 0 */
static const uint8_t ALIGN16 shuf_A_r0[16] = { 0, 3, 6, 9, 12, 15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 };
static const uint8_t ALIGN16 shuf_A_r1[16] = { 0x80,0x80,0x80,0x80,0x80,0x80, 2, 5, 8, 11, 14, 0x80,0x80,0x80,0x80,0x80 };
static const uint8_t ALIGN16 shuf_A_r2[16] = { 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 1, 4, 7, 10, 13 };

/* Component B: every 3rd byte starting at offset 1 */
static const uint8_t ALIGN16 shuf_B_r0[16] = { 1, 4, 7, 10, 13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 };
static const uint8_t ALIGN16 shuf_B_r1[16] = { 0x80,0x80,0x80,0x80,0x80, 0, 3, 6, 9, 12, 15, 0x80,0x80,0x80,0x80,0x80 };
static const uint8_t ALIGN16 shuf_B_r2[16] = { 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 2, 5, 8, 11, 14 };

/* Component C: every 3rd byte starting at offset 2 */
static const uint8_t ALIGN16 shuf_C_r0[16] = { 2, 5, 8, 11, 14, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 };
static const uint8_t ALIGN16 shuf_C_r1[16] = { 0x80,0x80,0x80,0x80,0x80, 1, 4, 7, 10, 13, 0x80,0x80,0x80,0x80,0x80,0x80 };
static const uint8_t ALIGN16 shuf_C_r2[16] = { 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80, 0, 3, 6, 9, 12, 15 };

/* Separates 48 interleaved bytes (stored as ABC...ABC across r0, r1, r2)
 * into three 16-element component vectors using the shuffle tables above.
 * Each shuffle selects the bytes belonging to one component from one
 * register and zeroes the rest via the 0x80 mask; OR combines the three
 * partial results into the final vector. */
static inline void deinterleave_3x16(__m128i r0, __m128i r1, __m128i r2,
                                      __m128i *out_A, __m128i *out_B, __m128i *out_C)
{
    __m128i mA0 = _mm_load_si128((const __m128i *)shuf_A_r0);
    __m128i mA1 = _mm_load_si128((const __m128i *)shuf_A_r1);
    __m128i mA2 = _mm_load_si128((const __m128i *)shuf_A_r2);
    *out_A = _mm_or_si128(_mm_or_si128(
        _mm_shuffle_epi8(r0, mA0), _mm_shuffle_epi8(r1, mA1)),
        _mm_shuffle_epi8(r2, mA2));

    __m128i mB0 = _mm_load_si128((const __m128i *)shuf_B_r0);
    __m128i mB1 = _mm_load_si128((const __m128i *)shuf_B_r1);
    __m128i mB2 = _mm_load_si128((const __m128i *)shuf_B_r2);
    *out_B = _mm_or_si128(_mm_or_si128(
        _mm_shuffle_epi8(r0, mB0), _mm_shuffle_epi8(r1, mB1)),
        _mm_shuffle_epi8(r2, mB2));

    __m128i mC0 = _mm_load_si128((const __m128i *)shuf_C_r0);
    __m128i mC1 = _mm_load_si128((const __m128i *)shuf_C_r1);
    __m128i mC2 = _mm_load_si128((const __m128i *)shuf_C_r2);
    *out_C = _mm_or_si128(_mm_or_si128(
        _mm_shuffle_epi8(r0, mC0), _mm_shuffle_epi8(r1, mC1)),
        _mm_shuffle_epi8(r2, mC2));
}

/* -----------------------------------------------------------------------
 * Horizontal SSE filters (thirds family)
 * ----------------------------------------------------------------------- */

/* Horizontal 3x filter (SSE): box average of 3 source pixels.
 * Processes 48 input -> 16 output bytes per SSE chunk.
 *
 * The three uint8 components are widened to 16-bit before summing because
 * their sum can reach 765 (3 * 255), which overflows uint8.  Division by 3
 * is performed with _mm_mulhi_epu16 using the magic multiplier 0x5556
 * (the same approximation as div3_u16), which avoids an actual division
 * instruction and gives exact results for sums in [0, 765]. */
static void h_filter_3x(const uint8_t *restrict src, int src_w,
                         uint8_t *restrict dst, int dst_w)
{
    (void)src_w;
    int chunks = dst_w / 16;
    __m128i zero = _mm_setzero_si128();
    __m128i magic = _mm_set1_epi16((short)0x5556);

    for (int c = 0; c < chunks; c++) {
        __m128i r0 = _mm_loadu_si128((const __m128i *)(src + c * 48));
        __m128i r1 = _mm_loadu_si128((const __m128i *)(src + c * 48 + 16));
        __m128i r2 = _mm_loadu_si128((const __m128i *)(src + c * 48 + 32));

        __m128i A, B, C;
        deinterleave_3x16(r0, r1, r2, &A, &B, &C);

        __m128i sum_lo = _mm_add_epi16(
            _mm_add_epi16(_mm_unpacklo_epi8(A, zero), _mm_unpacklo_epi8(B, zero)),
            _mm_unpacklo_epi8(C, zero));
        __m128i sum_hi = _mm_add_epi16(
            _mm_add_epi16(_mm_unpackhi_epi8(A, zero), _mm_unpackhi_epi8(B, zero)),
            _mm_unpackhi_epi8(C, zero));

        __m128i div_lo = _mm_mulhi_epu16(sum_lo, magic);
        __m128i div_hi = _mm_mulhi_epu16(sum_hi, magic);

        __m128i result = _mm_packus_epi16(div_lo, div_hi);
        _mm_storeu_si128((__m128i *)(dst + c * 16), result);
    }

    for (int x = chunks * 16; x < dst_w; x++) {
        uint16_t sum = (uint16_t)src[3*x] + src[3*x+1] + src[3*x+2];
        dst[x] = div3_u16(sum);
    }
}

/* Horizontal 1.5x filter (SSE): 3:2 bilinear reduction.
 * Every 3 source pixels -> 2 output pixels via weighted blend.
 * Processes 48 input -> 32 output bytes per SSE chunk.
 *
 * Geometry: each source triplet (A, B, C) produces two output pixels.
 * Output pixel 0 sits 1/3 of the way through the triplet (closer to A)
 * and is computed as (A*171 + B*85 + 128) >> 8.  Output pixel 1 sits
 * 2/3 of the way (closer to C) and is (C*171 + B*85 + 128) >> 8.  Both
 * share B as the center influence.  The inputs are widened to 16-bit
 * before multiplying because the weighted sum can slightly exceed 255
 * before the shift.
 *
 * The interleave step at the end reorders the data from [all out0 pixels,
 * all out1 pixels] into the correct memory layout [out0[0], out1[0],
 * out0[1], out1[1], ...] using _mm_unpacklo/hi_epi8. */
static void h_filter_1_5x(const uint8_t *restrict src, int src_w,
                           uint8_t *restrict dst, int dst_w)
{
    int chunks = dst_w / 32;
    __m128i zero = _mm_setzero_si128();
    __m128i w171 = _mm_set1_epi16(171);
    __m128i w85  = _mm_set1_epi16(85);
    __m128i rnd  = _mm_set1_epi16(128);

    for (int c = 0; c < chunks; c++) {
        __m128i r0 = _mm_loadu_si128((const __m128i *)(src + c * 48));
        __m128i r1 = _mm_loadu_si128((const __m128i *)(src + c * 48 + 16));
        __m128i r2 = _mm_loadu_si128((const __m128i *)(src + c * 48 + 32));

        __m128i A, B, C;
        deinterleave_3x16(r0, r1, r2, &A, &B, &C);

        __m128i a_lo = _mm_unpacklo_epi8(A, zero);
        __m128i a_hi = _mm_unpackhi_epi8(A, zero);
        __m128i b_lo = _mm_unpacklo_epi8(B, zero);
        __m128i b_hi = _mm_unpackhi_epi8(B, zero);
        __m128i c_lo = _mm_unpacklo_epi8(C, zero);
        __m128i c_hi = _mm_unpackhi_epi8(C, zero);

        /* out0 = (A*171 + B*85 + 128) >> 8 */
        __m128i bl0_lo = _mm_srli_epi16(_mm_add_epi16(rnd,
            _mm_add_epi16(_mm_mullo_epi16(a_lo, w171), _mm_mullo_epi16(b_lo, w85))), 8);
        __m128i bl0_hi = _mm_srli_epi16(_mm_add_epi16(rnd,
            _mm_add_epi16(_mm_mullo_epi16(a_hi, w171), _mm_mullo_epi16(b_hi, w85))), 8);
        __m128i out0 = _mm_packus_epi16(bl0_lo, bl0_hi);

        /* out1 = (C*171 + B*85 + 128) >> 8 */
        __m128i bl1_lo = _mm_srli_epi16(_mm_add_epi16(rnd,
            _mm_add_epi16(_mm_mullo_epi16(c_lo, w171), _mm_mullo_epi16(b_lo, w85))), 8);
        __m128i bl1_hi = _mm_srli_epi16(_mm_add_epi16(rnd,
            _mm_add_epi16(_mm_mullo_epi16(c_hi, w171), _mm_mullo_epi16(b_hi, w85))), 8);
        __m128i out1 = _mm_packus_epi16(bl1_lo, bl1_hi);

        /* Interleave out0/out1 and store 32 bytes */
        _mm_storeu_si128((__m128i *)(dst + c * 32),      _mm_unpacklo_epi8(out0, out1));
        _mm_storeu_si128((__m128i *)(dst + c * 32 + 16), _mm_unpackhi_epi8(out0, out1));
    }

    /* Scalar tail */
    int x_out = chunks * 32;
    for (int x_in = chunks * 48; x_in < src_w - 2 && x_out < dst_w - 1;
         x_in += 3, x_out += 2) {
        dst[x_out]     = blend_2_1(src[x_in],     src[x_in + 1]);
        dst[x_out + 1] = blend_2_1(src[x_in + 2], src[x_in + 1]);
    }
}

/* Forward declaration - defined after avx2_blend_2_1_u16 below */
static inline __m128i avx2_halve_32_to_16(__m256i v);
static inline __m256i avx2_halve_64_to_32(__m256i v0, __m256i v1);

/* Horizontal halve filter (SSE/AVX2): pairwise average.
 * Processes 32 input -> 16 output bytes per AVX2 chunk via the
 * avx2_halve_32_to_16 helper (forward-declared above). */
static void h_filter_halve(const uint8_t *restrict src,
                           uint8_t *restrict dst, int dst_w)
{
    int src_bytes = dst_w * 2;
    int h_chunks = src_bytes / 32;
    int out_x = 0;

    int c = 0;
    #pragma GCC unroll 4
    for (; c + 1 < h_chunks; c += 2) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(src + c * 32));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(src + c * 32 + 32));
        _mm256_storeu_si256((__m256i *)(dst + out_x),
                            avx2_halve_64_to_32(v0, v1));
        out_x += 32;
    }
    for (; c < h_chunks; c++) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src + c * 32));
        _mm_storeu_si128((__m128i *)(dst + out_x), avx2_halve_32_to_16(v));
        out_x += 16;
    }

    int tail_in = h_chunks * 32;
    for (int tx = tail_in; tx + 1 < src_bytes; tx += 2) {
        dst[out_x++] = avg_u8(src[tx], src[tx + 1]);
    }
}

/* -----------------------------------------------------------------------
 * AVX2 vertical blend helper (85/171 bilinear)
 *
 * Computes (a * 171 + b * 85 + 128) >> 8 across 32 bytes, the SIMD form of
 * the scalar blend_2_1.  171/256 and 85/256 are the 2/3 and 1/3 bilinear
 * weights.  The weighted sum is built from interleaved byte pairs with a
 * single maddubs per half (see the in-function comment for the
 * decomposition).
 *
 * The result is returned as its two 16-bit halves (out_lo / out_hi, the
 * low/high lanes of each 128-bit half) so a horizontal stage that works on
 * 16-bit components can take them directly.  Every lane is in [0,255], so the
 * halves are also the plain zero-extended 16-bit form of the blended bytes.
 * Used by the thirds 1.5x blended rows, where the blend feeds the horizontal
 * bilinear with no deinterleave in between.
 * ----------------------------------------------------------------------- */
static inline void avx2_blend_2_1_u16(__m256i a, __m256i b,
                                      __m256i *out_lo, __m256i *out_hi)
{
    /* 171*a + 85*b decomposes as 256*a + 85*(b - a): maddubs over
     * interleaved (b, a) byte pairs with weights {85, -85} computes the
     * 85*(b - a) term in one multiply, and interleaving the rounding
     * byte 0x80 under a builds 256*a + 128 with no arithmetic at all
     * (a lands in the high byte of each u16, 0x80 in the low).  The
     * intermediate 85*(b - a) is at most +/-21675, inside i16, and the
     * final sum 171*a + 85*b + 128 <= 65408 fits u16, so the logical
     * shift returns the exact blend - the same decomposition the
     * vertical blends in the upscaler use.  Two maddubs and two adds
     * replace four mullo and four adds; the shuffle count is unchanged
     * (four unpacks either way). */
    const __m256i wpair = _mm256_set1_epi16((short)(((-85 & 0xFF) << 8) | 85));
    const __m256i half  = _mm256_set1_epi8((char)0x80);

    __m256i ba_lo = _mm256_unpacklo_epi8(b, a);
    __m256i ba_hi = _mm256_unpackhi_epi8(b, a);

    *out_lo = _mm256_srli_epi16(_mm256_add_epi16(
        _mm256_maddubs_epi16(ba_lo, wpair),
        _mm256_unpacklo_epi8(half, a)), 8);
    *out_hi = _mm256_srli_epi16(_mm256_add_epi16(
        _mm256_maddubs_epi16(ba_hi, wpair),
        _mm256_unpackhi_epi8(half, a)), 8);
}

/* -----------------------------------------------------------------------
 * AVX2 horizontal halving helper
 *
 * Takes 32 input bytes, produces 16 output bytes by averaging adjacent
 * pairs: out[i] = avg(in[2i], in[2i+1]).  Returns a __m128i.
 *
 * We can't simply vpavgb two registers directly, because the pairs we
 * want to average (bytes 0&1, 2&3, etc.) are interleaved within a single
 * 32-byte register, not split across two registers.  The strategy is to
 * first gather all even-indexed bytes into one register and all
 * odd-indexed bytes into another, then average them.
 *
 * Step 1: _mm256_shuffle_epi8 with the shuf_even_odd mask reorders bytes
 * within each 128-bit lane so even-indexed bytes occupy the low 8 bytes
 * and odd-indexed bytes occupy the high 8 bytes.
 *
 * Step 2: _mm256_permute4x64_epi64 with control 0xD8 = 0b_11_01_10_00
 * rearranges the four 64-bit quadwords across the 256-bit register:
 *   output qword 0 <- src qword 0 (even bytes from lane 0)
 *   output qword 1 <- src qword 2 (even bytes from lane 1)
 *   output qword 2 <- src qword 1 (odd bytes from lane 0)
 *   output qword 3 <- src qword 3 (odd bytes from lane 1)
 * After this, the low 128 bits hold all 16 even bytes and the high 128
 * bits hold all 16 odd bytes.
 *
 * Step 3: extract the two halves and _mm_avg_epu8 them.
 * ----------------------------------------------------------------------- */

static inline __m128i avx2_halve_32_to_16(__m256i v)
{
    /* maddubs against {1,1} sums each adjacent byte pair exactly
     * (max 510, no saturation), doing the even/odd pair gather on the
     * multiply port instead of the shuffle port.  pavgw against zero
     * rounds and halves: (sum + 0 + 1) >> 1, the same rounding as
     * vpavgb / avg_u8, so the result is bit-identical to averaging
     * separated even/odd bytes.  packus_epi16 of the result against
     * itself narrows to bytes ([r0-7 r0-7] in lane 0, [r8-15 r8-15]
     * in lane 1) and one vpermq picks qwords 0 and 2 into the low
     * 128 bits - two shuffle-port ops where the even/odd separation
     * needed three (vpshufb + vpermq + vextracti128). */
    const __m256i ones8 = _mm256_set1_epi8(1);
    __m256i sums   = _mm256_maddubs_epi16(v, ones8);   /* 16 x (a + b) */
    __m256i halves = _mm256_avg_epu16(sums, _mm256_setzero_si256());
    __m256i packed = _mm256_packus_epi16(halves, halves);
    return _mm256_castsi256_si128(
        _mm256_permute4x64_epi64(packed, 0x08));       /* qwords 0, 2 */
}

/* -----------------------------------------------------------------------
 * AVX2 horizontal halving helper (paired, 64 -> 32 bytes)
 *
 * Pairwise average of 64 input bytes -> 32 output bytes:
 *   out[i] = avg(in[2i], in[2i+1]) = (in[2i] + in[2i+1] + 1) >> 1.
 *
 * Bit-identical to running avx2_halve_32_to_16 on each 32-byte half, but
 * trades the shuffle-port heavy vpshufb + vpermq + vextracti128 path
 * (6 shuffle-port ops per 32 output bytes) for a vpmaddubsw reduction
 * (2 shuffle-port ops per 32 output bytes) plus a single 256-bit store.
 *
 * maddubs(v, set1_epi8(1)) sums each adjacent byte pair exactly: the
 * products are a*1 + b*1 = a + b in [0, 510], well inside int16 range so
 * no saturation occurs, matching the even/odd pair separation the
 * shuffle path performs.  (a + b + 1) >> 1 is exactly vpavgb's rounded
 * average (see avg_u8).
 *
 * maddubs is per-128-bit-lane, so m0 holds out[0..7] in lane 0 and
 * out[8..15] in lane 1 (linear within itself); m1 holds out[16..23] /
 * out[24..31].  packus(m0, m1) is also per-lane and yields
 * [out0-7 | out16-23 | out8-15 | out24-31]; permute4x64(0xD8) restores
 * the linear [out0-7 | out8-15 | out16-23 | out24-31] order.
 * ----------------------------------------------------------------------- */

static inline __m256i avx2_halve_64_to_32(__m256i v0, __m256i v1)
{
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i zero  = _mm256_setzero_si256();

    __m256i m0 = _mm256_maddubs_epi16(v0, ones8);  /* 16x (a + b), <= 510 */
    __m256i m1 = _mm256_maddubs_epi16(v1, ones8);

    /* pavgw(sum, 0) is exactly (sum + 1) >> 1.  Besides expressing the
     * rounding directly, this keeps GCC from emitting separate add and
     * shift instructions for each half (Clang already folds that pattern). */
    m0 = _mm256_avg_epu16(m0, zero);
    m1 = _mm256_avg_epu16(m1, zero);

    __m256i packed = _mm256_packus_epi16(m0, m1);
    return _mm256_permute4x64_epi64(packed, 0xD8);
}


/* -----------------------------------------------------------------------
 * AVX2 horizontal 16->8 halving helper
 *
 * Takes 16 input bytes (low lane of a __m128i), produces 8 output bytes in
 * the low 64 bits: out[i] = avg(in[2i], in[2i+1]) for i=0..7.  Used by the
 * fused thirds 12x horizontal pipeline (halve_16_to_8 follows the 32->16
 * halve to complete a 6x->12x reduction without a scratch round-trip).
 *
 * maddubs against {1,1} sums each adjacent byte pair (max 510, no
 * saturation) on the multiply port; pavgw against zero computes the
 * rounded halving (sum + 1) >> 1, matching scalar avg_u8 and the vpavgb
 * the other halve helpers use, so the result is bit-identical to running
 * h_filter_halve over the same 16 bytes.  One packus narrows the eight
 * u16 averages into the low 8 bytes - a single shuffle-port op. */
static inline __m128i halve_16_to_8(__m128i v)
{
    __m128i sums   = _mm_maddubs_epi16(v, _mm_set1_epi8(1));
    __m128i halves = _mm_avg_epu16(sums, _mm_setzero_si128());
    return _mm_packus_epi16(halves, halves);    /* low 8 = the averages */
}

/* Finish a horizontal pow2 cascade while the freshly computed vertical
 * averages are still in registers.  The 64-byte form matches one paired
 * iteration of h_filter_halve; the 32-byte form handles the residual chunk.
 * Each successive helper consumes only the valid low bytes produced by the
 * preceding step, so the first 32/16/8/4 (or 16/8/4/2) stored bytes are
 * exactly the old row-at-a-time cascade. */
static inline void store_pow2_64(__m256i v0, __m256i v1, int level,
                                 uint8_t *restrict dst)
{
    __m256i h1 = avx2_halve_64_to_32(v0, v1);
    if (level == 0) {
        _mm256_storeu_si256((__m256i *)dst, h1);
        return;
    }

    __m128i h2 = avx2_halve_32_to_16(h1);
    if (level == 1) {
        _mm_storeu_si128((__m128i *)dst, h2);
        return;
    }

    __m128i h3 = halve_16_to_8(h2);
    if (level == 2) {
        _mm_storel_epi64((__m128i *)dst, h3);
        return;
    }

    __m128i h4 = halve_16_to_8(h3);
    uint32_t packed = (uint32_t)_mm_cvtsi128_si32(h4);
    memcpy(dst, &packed, sizeof(packed));
}

static inline void store_pow2_32(__m256i v, int level,
                                 uint8_t *restrict dst)
{
    __m128i h1 = avx2_halve_32_to_16(v);
    if (level == 0) {
        _mm_storeu_si128((__m128i *)dst, h1);
        return;
    }

    __m128i h2 = halve_16_to_8(h1);
    if (level == 1) {
        _mm_storel_epi64((__m128i *)dst, h2);
        return;
    }

    __m128i h3 = halve_16_to_8(h2);
    if (level == 2) {
        uint32_t packed = (uint32_t)_mm_cvtsi128_si32(h3);
        memcpy(dst, &packed, sizeof(packed));
        return;
    }

    __m128i h4 = halve_16_to_8(h3);
    uint16_t packed = (uint16_t)_mm_extract_epi16(h4, 0);
    memcpy(dst, &packed, sizeof(packed));
}

static inline uint8_t reduce_pow2_scalar(const uint8_t *src, int level)
{
    uint8_t values[16];
    int count = 2 << level;
    memcpy(values, src, (size_t)count);
    while (count > 1) {
        for (int i = 0; i < count / 2; i++)
            values[i] = avg_u8(values[2 * i], values[2 * i + 1]);
        count >>= 1;
    }
    return values[0];
}

static inline __attribute__((always_inline)) void avg_rows_64_avx2(
    const uint8_t *restrict a, const uint8_t *restrict b, int x,
    __m256i *lo, __m256i *hi)
{
    *lo = _mm256_avg_epu8(
        _mm256_loadu_si256((const __m256i *)(a + x)),
        _mm256_loadu_si256((const __m256i *)(b + x)));
    *hi = _mm256_avg_epu8(
        _mm256_loadu_si256((const __m256i *)(a + x + 32)),
        _mm256_loadu_si256((const __m256i *)(b + x + 32)));
}

static inline __attribute__((always_inline)) void avg_nodes_64_avx2(
    __m256i a_lo, __m256i a_hi, __m256i b_lo, __m256i b_hi,
    __m256i *lo, __m256i *hi)
{
    *lo = _mm256_avg_epu8(a_lo, b_lo);
    *hi = _mm256_avg_epu8(a_hi, b_hi);
}

static inline __attribute__((always_inline)) __m256i avg_rows_32_avx2(
    const uint8_t *restrict a, const uint8_t *restrict b, int x)
{
    return _mm256_avg_epu8(
        _mm256_loadu_si256((const __m256i *)(a + x)),
        _mm256_loadu_si256((const __m256i *)(b + x)));
}

/* Build regular 64-byte column tiles as low-pressure binary subtrees.  This
 * preserves the exact rounded-average order while eliminating the vertical
 * scratch ladder for the common aligned video widths. */
static void __attribute__((hot)) scale_plane_pow2_tree_64_avx2(
    const uint8_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs, int deepest,
    uint8_t *restrict dst_planes[4], int dst_strides[4])
{
    const int emit0 = (active_outputs & (1u << 1)) != 0;
    const int emit1 = (active_outputs & (1u << 3)) != 0;
    const int emit2 = (active_outputs & (1u << 5)) != 0;
    const int dense = (deepest == 0)
                   || (deepest == 1 && emit0)
                   || (deepest == 2 && emit0 && emit1)
                   || (deepest == 3 && emit0 && emit1 && emit2);

#define STEP_PTR_AVX2(p, step) do { (p) += (step); __asm__("" : "+r"(p)); } while (0)

#define POW2_TREE_64_AVX2_LOOP(D, DENSE) do {                                  \
    const int group_rows_ = 2 << (D);                                          \
    const int num_groups_ = src_h / group_rows_;                               \
    const ptrdiff_t ss_    = (ptrdiff_t)src_stride;                            \
    const ptrdiff_t ss2_   = ss_ * 2;                                          \
    const ptrdiff_t ds0_   = (ptrdiff_t)dst_strides[0];                        \
    const ptrdiff_t ds1_   = (ptrdiff_t)dst_strides[1];                        \
    const ptrdiff_t ds2_   = (ptrdiff_t)dst_strides[2];                        \
    const ptrdiff_t ds3_   = (ptrdiff_t)dst_strides[3];                        \
    const int e0_ = ((D) == 0 || (DENSE)) ? 1 : emit0;                         \
    const int e1_ = ((D) == 1 || (DENSE)) ? 1 : emit1;                         \
    const int e2_ = ((D) == 2 || (DENSE)) ? 1 : emit2;                         \
    const uint8_t *grp_ = src;                                                 \
    uint8_t *out0_ = e0_ ? dst_planes[0] : NULL;                               \
    uint8_t *out1_ = ((D) >= 1 && e1_) ? dst_planes[1] : NULL;                 \
    uint8_t *out2_ = ((D) >= 2 && e2_) ? dst_planes[2] : NULL;                 \
    uint8_t *out3_ = ((D) >= 3) ? dst_planes[3] : NULL;                        \
    (void)ss2_; (void)ds1_; (void)ds2_; (void)ds3_;                            \
    (void)out1_; (void)out2_; (void)out3_;                                     \
                                                                               \
    for (int g_ = 0; g_ < num_groups_; g_++) {                                 \
        for (int x_ = 0; x_ < src_w; x_ += 64) {                               \
            const uint8_t *sr_ = grp_ + x_;                                    \
            uint8_t *o0_ = e0_ ? (out0_ + ((size_t)x_ >> 1)) : NULL;           \
            if ((D) >= 2) {                                                    \
                __asm__("" : "+r"(sr_));                                       \
                if (e0_) __asm__("" : "+r"(o0_));                              \
            }                                                                  \
            __m256i n0_lo_, n0_hi_;                                            \
                                                                               \
            avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n0_lo_, &n0_hi_);             \
            if (e0_)                                                           \
                store_pow2_64(n0_lo_, n0_hi_, 0, o0_);                         \
                                                                               \
            if ((D) >= 1) {                                                    \
                __m256i n1_lo_, n1_hi_, p0_lo_, p0_hi_;                        \
                uint8_t *o1_ = e1_ ? (out1_ + ((size_t)x_ >> 2)) : NULL;       \
                if ((D) >= 3 && e1_) __asm__("" : "+r"(o1_));                  \
                if ((D) >= 2) {                                                \
                    STEP_PTR_AVX2(sr_, ss2_);                                  \
                    if (e0_) STEP_PTR_AVX2(o0_, ds0_);                         \
                    avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n1_lo_, &n1_hi_);     \
                    if (e0_) store_pow2_64(n1_lo_, n1_hi_, 0, o0_);            \
                } else {                                                       \
                    avg_rows_64_avx2(sr_ + ss2_, sr_ + ss_ * 3, 0,             \
                                     &n1_lo_, &n1_hi_);                        \
                    if (e0_) store_pow2_64(n1_lo_, n1_hi_, 0, o0_ + ds0_);     \
                }                                                              \
                avg_nodes_64_avx2(n0_lo_, n0_hi_, n1_lo_, n1_hi_,              \
                                  &p0_lo_, &p0_hi_);                           \
                if (e1_)                                                       \
                    store_pow2_64(p0_lo_, p0_hi_, 1, o1_);                     \
                                                                               \
                if ((D) >= 2) {                                                \
                    __m256i n2_lo_, n2_hi_, n3_lo_, n3_hi_;                    \
                    __m256i p1_lo_, p1_hi_, q0_lo_, q0_hi_;                    \
                    STEP_PTR_AVX2(sr_, ss2_);                                  \
                    if (e0_) STEP_PTR_AVX2(o0_, ds0_);                         \
                    avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n2_lo_, &n2_hi_);     \
                    if (e0_) store_pow2_64(n2_lo_, n2_hi_, 0, o0_);            \
                                                                               \
                    STEP_PTR_AVX2(sr_, ss2_);                                  \
                    if (e0_) STEP_PTR_AVX2(o0_, ds0_);                         \
                    avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n3_lo_, &n3_hi_);     \
                    if (e0_) store_pow2_64(n3_lo_, n3_hi_, 0, o0_);            \
                                                                               \
                    avg_nodes_64_avx2(n2_lo_, n2_hi_, n3_lo_, n3_hi_,          \
                                      &p1_lo_, &p1_hi_);                       \
                    if (e1_) {                                                 \
                        if ((D) >= 3) STEP_PTR_AVX2(o1_, ds1_);                \
                        else o1_ += ds1_;                                      \
                        store_pow2_64(p1_lo_, p1_hi_, 1, o1_);                 \
                    }                                                          \
                    avg_nodes_64_avx2(p0_lo_, p0_hi_, p1_lo_, p1_hi_,          \
                                      &q0_lo_, &q0_hi_);                       \
                    if (e2_)                                                   \
                        store_pow2_64(q0_lo_, q0_hi_, 2,                       \
                                      out2_ + ((size_t)x_ >> 3));              \
                                                                               \
                    if ((D) >= 3) {                                            \
                        __m256i n4_lo_, n4_hi_, n5_lo_, n5_hi_;                \
                        __m256i n6_lo_, n6_hi_, n7_lo_, n7_hi_;                \
                        __m256i p2_lo_, p2_hi_, p3_lo_, p3_hi_;                \
                        __m256i q1_lo_, q1_hi_, root_lo_, root_hi_;            \
                        STEP_PTR_AVX2(sr_, ss2_);                              \
                        if (e0_) STEP_PTR_AVX2(o0_, ds0_);                     \
                        avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n4_lo_, &n4_hi_); \
                        if (e0_) store_pow2_64(n4_lo_, n4_hi_, 0, o0_);        \
                                                                               \
                        STEP_PTR_AVX2(sr_, ss2_);                              \
                        if (e0_) STEP_PTR_AVX2(o0_, ds0_);                     \
                        avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n5_lo_, &n5_hi_); \
                        if (e0_) store_pow2_64(n5_lo_, n5_hi_, 0, o0_);        \
                                                                               \
                        avg_nodes_64_avx2(n4_lo_, n4_hi_, n5_lo_, n5_hi_,      \
                                          &p2_lo_, &p2_hi_);                   \
                        if (e1_) {                                             \
                            STEP_PTR_AVX2(o1_, ds1_);                          \
                            store_pow2_64(p2_lo_, p2_hi_, 1, o1_);             \
                        }                                                      \
                                                                               \
                        STEP_PTR_AVX2(sr_, ss2_);                              \
                        if (e0_) STEP_PTR_AVX2(o0_, ds0_);                     \
                        avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n6_lo_, &n6_hi_); \
                        if (e0_) store_pow2_64(n6_lo_, n6_hi_, 0, o0_);        \
                                                                               \
                        STEP_PTR_AVX2(sr_, ss2_);                              \
                        if (e0_) STEP_PTR_AVX2(o0_, ds0_);                     \
                        avg_rows_64_avx2(sr_, sr_ + ss_, 0, &n7_lo_, &n7_hi_); \
                        if (e0_) store_pow2_64(n7_lo_, n7_hi_, 0, o0_);        \
                                                                               \
                        avg_nodes_64_avx2(n6_lo_, n6_hi_, n7_lo_, n7_hi_,      \
                                          &p3_lo_, &p3_hi_);                   \
                        if (e1_) {                                             \
                            STEP_PTR_AVX2(o1_, ds1_);                          \
                            store_pow2_64(p3_lo_, p3_hi_, 1, o1_);             \
                        }                                                      \
                        avg_nodes_64_avx2(p2_lo_, p2_hi_, p3_lo_, p3_hi_,      \
                                          &q1_lo_, &q1_hi_);                   \
                        if (e2_)                                               \
                            store_pow2_64(q1_lo_, q1_hi_, 2,                   \
                                          out2_ + ds2_ + ((size_t)x_ >> 3));   \
                        avg_nodes_64_avx2(q0_lo_, q0_hi_, q1_lo_, q1_hi_,      \
                                          &root_lo_, &root_hi_);               \
                        store_pow2_64(root_lo_, root_hi_, 3,                   \
                                      out3_ + ((size_t)x_ >> 4));              \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        grp_ += ss_ * group_rows_;                                             \
        if (e0_) out0_ += ds0_ * (group_rows_ >> 1);                           \
        if ((D) >= 1 && e1_) out1_ += ds1_ * (group_rows_ >> 2);               \
        if ((D) >= 2 && e2_) out2_ += ds2_ * (group_rows_ >> 3);               \
        if ((D) >= 3) out3_ += ds3_;                                           \
    }                                                                          \
} while (0)

    switch (deepest) {
    case 0:
        POW2_TREE_64_AVX2_LOOP(0, 1);
        break;
    case 1:
        if (dense) POW2_TREE_64_AVX2_LOOP(1, 1);
        else       POW2_TREE_64_AVX2_LOOP(1, 0);
        break;
    case 2:
        if (dense) POW2_TREE_64_AVX2_LOOP(2, 1);
        else       POW2_TREE_64_AVX2_LOOP(2, 0);
        break;
    default:
        if (dense) POW2_TREE_64_AVX2_LOOP(3, 1);
        else       POW2_TREE_64_AVX2_LOOP(3, 0);
        break;
    }

#undef POW2_TREE_64_AVX2_LOOP
#undef STEP_PTR_AVX2
}

static void __attribute__((noinline, hot)) scale_plane_pow2_tree_tail_32_avx2(
    const uint8_t *restrict src,
    int x, int src_h, int src_stride,
    uint32_t active_outputs, int deepest,
    uint8_t *restrict dst_planes[4], int dst_strides[4])
{
    const int emit0 = (active_outputs & (1u << 1)) != 0;
    const int emit1 = (active_outputs & (1u << 3)) != 0;
    const int emit2 = (active_outputs & (1u << 5)) != 0;
    const int emit3 = (active_outputs & (1u << 7)) != 0;
    const int group_rows = 2 << deepest;
    const int num_groups = src_h / group_rows;
    const ptrdiff_t ss    = (ptrdiff_t)src_stride;
    const ptrdiff_t ss3   = ss * 3;
    const ptrdiff_t ds0   = (ptrdiff_t)dst_strides[0];
    const ptrdiff_t ds0_3 = ds0 * 3;
    const ptrdiff_t ds1   = (ptrdiff_t)dst_strides[1];
    const ptrdiff_t ds1_3 = ds1 * 3;
    const ptrdiff_t ds2   = (ptrdiff_t)dst_strides[2];
    const ptrdiff_t ds3   = (ptrdiff_t)dst_strides[3];
    const uint8_t *grp = src + x;
    uint8_t *out0 = emit0 ? (dst_planes[0] + ((size_t)x >> 1)) : NULL;
    uint8_t *out1 = emit1 ? (dst_planes[1] + ((size_t)x >> 2)) : NULL;
    uint8_t *out2 = emit2 ? (dst_planes[2] + ((size_t)x >> 3)) : NULL;
    uint8_t *out3 = emit3 ? (dst_planes[3] + ((size_t)x >> 4)) : NULL;

    for (int g = 0; g < num_groups; g++) {
        const uint8_t *sr = grp;
        uint8_t *o0 = out0;
        __m256i n0 = avg_rows_32_avx2(sr, sr + ss, 0);
        if (emit0) store_pow2_32(n0, 0, o0);
        if (deepest >= 1) {
            __m256i n1 = avg_rows_32_avx2(sr + ss * 2, sr + ss3, 0);
            if (emit0) store_pow2_32(n1, 0, o0 + ds0);
            __m256i p0 = _mm256_avg_epu8(n0, n1);
            if (emit1) store_pow2_32(p0, 1, out1);
            if (deepest >= 2) {
                sr += ss * 4;
                __asm__("" : "+r"(sr));
                __m256i n2 = avg_rows_32_avx2(sr, sr + ss, 0);
                if (emit0) store_pow2_32(n2, 0, o0 + ds0 * 2);
                __m256i n3 = avg_rows_32_avx2(sr + ss * 2, sr + ss3, 0);
                if (emit0) store_pow2_32(n3, 0, o0 + ds0_3);
                __m256i p1 = _mm256_avg_epu8(n2, n3);
                if (emit1) store_pow2_32(p1, 1, out1 + ds1);
                __m256i q0 = _mm256_avg_epu8(p0, p1);
                if (emit2) store_pow2_32(q0, 2, out2);
                if (deepest >= 3) {
                    sr += ss * 4;
                    __asm__("" : "+r"(sr));
                    if (emit0) {
                        o0 += ds0 * 4;
                        __asm__("" : "+r"(o0));
                    }
                    __m256i n4 = avg_rows_32_avx2(sr, sr + ss, 0);
                    if (emit0) store_pow2_32(n4, 0, o0);
                    __m256i n5 = avg_rows_32_avx2(sr + ss * 2, sr + ss3, 0);
                    if (emit0) store_pow2_32(n5, 0, o0 + ds0);
                    __m256i p2 = _mm256_avg_epu8(n4, n5);
                    if (emit1) store_pow2_32(p2, 1, out1 + ds1 * 2);

                    sr += ss * 4;
                    __asm__("" : "+r"(sr));
                    __m256i n6 = avg_rows_32_avx2(sr, sr + ss, 0);
                    if (emit0) store_pow2_32(n6, 0, o0 + ds0 * 2);
                    __m256i n7 = avg_rows_32_avx2(sr + ss * 2, sr + ss3, 0);
                    if (emit0) store_pow2_32(n7, 0, o0 + ds0_3);
                    __m256i p3 = _mm256_avg_epu8(n6, n7);
                    if (emit1) store_pow2_32(p3, 1, out1 + ds1_3);
                    __m256i q1 = _mm256_avg_epu8(p2, p3);
                    if (emit2) store_pow2_32(q1, 2, out2 + ds2);
                    if (emit3) store_pow2_32(_mm256_avg_epu8(q0, q1), 3, out3);
                }
            }
        }
        grp += ss * group_rows;
        if (emit0) out0 += ds0 * (group_rows >> 1);
        if (emit1) out1 += ds1 * (group_rows >> 2);
        if (emit2) out2 += ds2 * (group_rows >> 3);
        if (emit3) out3 += ds3;
    }
}


/* -----------------------------------------------------------------------
 * AVX2 box-of-3 horizontal average core (256-bit).
 *
 * From a deinterleaved (A,B,C) triplet (lane 0 = group 1, lane 1 = group 2)
 * computes div3(A+B+C) per element and returns the 32 packed output bytes.
 * The sum reaches 765 = 3*255, so the divide by 3 uses _mm256_mulhi_epu16
 * with magic 0x5556 (exact for sums in [0,765], matching div3_u16) before
 * packing back to 8-bit.  Shared by h_chunk_3x_avx2 (which adds the store)
 * and the fused 12x pipeline (which keeps the result in register). */
static inline __m256i box3_div_avx2(__m256i A, __m256i B, __m256i C)
{
    __m256i zero = _mm256_setzero_si256();
    __m256i magic = _mm256_set1_epi16((short)0x5556);

    /* maddubs(interleave(A,B), 1) forms A_i*1 + B_i*1 = A_i + B_i per pair,
     * in [0,510] (within signed 16-bit range, so no saturation; A,B are the
     * unsigned operand and the constant 1 is the signed operand).  This keeps
     * the pair sum on the multiply port and off the shuffle port.  The
     * interleave puts those sums in the same lane positions as the
     * zero-extended C (lane 0 holds indices 0..7 and 16..23, the high half
     * holds 8..15 and 24..31), so adding the widened C gives the exact sum. */
    __m256i one = _mm256_set1_epi8(1);
    __m256i sum_lo = _mm256_add_epi16(
        _mm256_maddubs_epi16(_mm256_unpacklo_epi8(A, B), one),
        _mm256_unpacklo_epi8(C, zero));
    __m256i sum_hi = _mm256_add_epi16(
        _mm256_maddubs_epi16(_mm256_unpackhi_epi8(A, B), one),
        _mm256_unpackhi_epi8(C, zero));

    __m256i div_lo = _mm256_mulhi_epu16(sum_lo, magic);
    __m256i div_hi = _mm256_mulhi_epu16(sum_hi, magic);

    return _mm256_packus_epi16(div_lo, div_hi);
}


/* -----------------------------------------------------------------------
 * AVX2 256-bit deinterleave: 96 contiguous bytes (3 x __m256i) into three
 * 32-byte component vectors.
 *
 * Input: reg_a (bytes 0-31), reg_b (bytes 32-63), reg_c (bytes 64-95).
 * Output: A[g]=src[3g], B[g]=src[3g+1], C[g]=src[3g+2] for g=0..31.
 *
 * The 96 bytes are interleaved ABCABC... and we need to separate them into
 * three 32-byte component vectors.
 *
 * AVX2's vpshufb works independently within each 128-bit lane - a byte
 * can only be moved to another position within the same 16-byte lane, not
 * across the lane boundary.  This means we can't apply 32-byte shuffles
 * directly to reg_a/reg_b/reg_c as loaded.
 *
 * The solution is to split the 96 bytes into two independent 48-byte groups
 * and place each group's three 16-byte sub-blocks into the corresponding
 * lanes of three new registers.  Then vpshufb with a broadcast mask
 * processes both groups in parallel, one per lane.
 *
 *   Group 1 (bytes  0-47): a.lo = bytes  0-15 ("r0")
 *                           a.hi = bytes 16-31 ("r1")
 *                           b.lo = bytes 32-47 ("r2")
 *   Group 2 (bytes 48-95): b.hi = bytes 48-63 ("r0")
 *                           c.lo = bytes 64-79 ("r1")
 *                           c.hi = bytes 80-95 ("r2")
 *
 * _mm256_permute2x128_si256(a, b, imm8): bits [1:0] select which
 * 128-bit source from a/b goes to output lane 0; bits [5:4] select
 * output lane 1.
 *   0x30 = 0b00_11_00_00: lane0 <- a.lane0, lane1 <- b.lane1
 *   0x21 = 0b00_10_00_01: lane0 <- a.lane1, lane1 <- b.lane0
 *
 * This produces:
 *   nr0 = permute(reg_a, reg_b, 0x30): [a.lo | b.hi]  ("r0" for each group)
 *   nr1 = permute(reg_a, reg_c, 0x21): [a.hi | c.lo]  ("r1" for each group)
 *   nr2 = permute(reg_b, reg_c, 0x30): [b.lo | c.hi]  ("r2" for each group)
 *
 * Broadcasting the same 128-bit shuffle masks to both lanes and applying
 * vpshufb then deinterleaves both groups simultaneously.
 * ----------------------------------------------------------------------- */

static inline void deinterleave_3x32(__m256i reg_a, __m256i reg_b, __m256i reg_c,
                                      __m256i *out_A, __m256i *out_B, __m256i *out_C)
{
    /* Rearrange so each lane contains one 48-byte group's register:
     *   nr0: [a.lo | b.hi]  -- "r0" for each group
     *   nr1: [a.hi | c.lo]  -- "r1" for each group
     *   nr2: [b.lo | c.hi]  -- "r2" for each group  */
    __m256i nr0 = _mm256_permute2x128_si256(reg_a, reg_b, 0x30);
    __m256i nr1 = _mm256_permute2x128_si256(reg_a, reg_c, 0x21);
    __m256i nr2 = _mm256_permute2x128_si256(reg_b, reg_c, 0x30);

    /* Broadcast each 128-bit shuffle mask to both lanes */
    __m256i mA0 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_A_r0));
    __m256i mA1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_A_r1));
    __m256i mA2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_A_r2));

    *out_A = _mm256_or_si256(_mm256_or_si256(
        _mm256_shuffle_epi8(nr0, mA0),
        _mm256_shuffle_epi8(nr1, mA1)),
        _mm256_shuffle_epi8(nr2, mA2));

    __m256i mB0 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_B_r0));
    __m256i mB1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_B_r1));
    __m256i mB2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_B_r2));

    *out_B = _mm256_or_si256(_mm256_or_si256(
        _mm256_shuffle_epi8(nr0, mB0),
        _mm256_shuffle_epi8(nr1, mB1)),
        _mm256_shuffle_epi8(nr2, mB2));

    __m256i mC0 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_C_r0));
    __m256i mC1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_C_r1));
    __m256i mC2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_C_r2));

    *out_C = _mm256_or_si256(_mm256_or_si256(
        _mm256_shuffle_epi8(nr0, mC0),
        _mm256_shuffle_epi8(nr1, mC1)),
        _mm256_shuffle_epi8(nr2, mC2));
}


/* -----------------------------------------------------------------------
 * Horizontal 1.5x on one deinterleaved 256-bit chunk (A, B, C each 32
 * bytes, with lane 0 = group 1, lane 1 = group 2).
 *
 * Produces 64 output bytes (32 per group) stored at dst.
 *
 * We process two 48-byte source groups in parallel - one per YMM lane.
 * After blending, each lane of out0 holds 16 blend-result bytes for its
 * group, and similarly for out1.
 *
 * unpacklo/hi interleaves out0 and out1 within each lane, producing the
 * correct [out0[i], out1[i], ...] pixel order for each group.  However,
 * unpacklo puts group1's first half in lane 0 and group2's first half in
 * lane 1, while unpackhi puts group1's second half in lane 0 and group2's
 * second half in lane 1.  The two permute2x128 calls reorganize this into
 * contiguous 32-byte output for each group:
 *   store0 = [group1_lo | group1_hi]
 *   store1 = [group2_lo | group2_hi]
 * ----------------------------------------------------------------------- */

/* Core horizontal 1.5x bilinear on ALREADY-WIDENED u16 component halves.
 * a_lo/a_hi/b_lo/b_hi/c_lo/c_hi are the per-128-bit-lane low/high 16-bit
 * widenings of components A,B,C (exactly _mm256_unpacklo/hi_epi8(X,zero)),
 * every lane in [0,255].  Produces 64 interleaved output bytes at dst.
 * The thirds 1.5x blended rows hold their blend result as 16-bit values and
 * pass it straight in. */
static inline void h_chunk_1_5x_u16_avx2(__m256i a_lo, __m256i a_hi,
                                         __m256i b_lo, __m256i b_hi,
                                         __m256i c_lo, __m256i c_hi,
                                         uint8_t *restrict dst)
{
    __m256i w171 = _mm256_set1_epi16(171);
    __m256i w85  = _mm256_set1_epi16(85);
    __m256i rnd  = _mm256_set1_epi16(128);

    /* b*85 is shared by both output pixels of the triplet. */
    __m256i b85_lo = _mm256_mullo_epi16(b_lo, w85);
    __m256i b85_hi = _mm256_mullo_epi16(b_hi, w85);

    /* out0 = (A*171 + B*85 + 128) >> 8 */
    __m256i bl0_lo = _mm256_srli_epi16(_mm256_add_epi16(rnd,
        _mm256_add_epi16(_mm256_mullo_epi16(a_lo, w171), b85_lo)), 8);
    __m256i bl0_hi = _mm256_srli_epi16(_mm256_add_epi16(rnd,
        _mm256_add_epi16(_mm256_mullo_epi16(a_hi, w171), b85_hi)), 8);
    __m256i out0 = _mm256_packus_epi16(bl0_lo, bl0_hi);

    /* out1 = (C*171 + B*85 + 128) >> 8 */
    __m256i bl1_lo = _mm256_srli_epi16(_mm256_add_epi16(rnd,
        _mm256_add_epi16(_mm256_mullo_epi16(c_lo, w171), b85_lo)), 8);
    __m256i bl1_hi = _mm256_srli_epi16(_mm256_add_epi16(rnd,
        _mm256_add_epi16(_mm256_mullo_epi16(c_hi, w171), b85_hi)), 8);
    __m256i out1 = _mm256_packus_epi16(bl1_lo, bl1_hi);

    /* Interleave out0/out1 pairs */
    __m256i interl_lo = _mm256_unpacklo_epi8(out0, out1);
    __m256i interl_hi = _mm256_unpackhi_epi8(out0, out1);

    /* Fix lane crossing: group outputs contiguous */
    __m256i store0 = _mm256_permute2x128_si256(interl_lo, interl_hi, 0x20);
    __m256i store1 = _mm256_permute2x128_si256(interl_lo, interl_hi, 0x31);

    /* dst is 32-byte aligned: output buffers are posix_memalign(32) and
     * out_off_1_5x = ci*64, which is always a multiple of 32. */
    _mm256_store_si256((__m256i *)(dst),      store0);
    _mm256_store_si256((__m256i *)(dst + 32), store1);
}

/* u8 entry point: widen A,B,C to u16 then run the core.  Used for the 1.5x
 * rows that come straight from a pair average (rows 0 and 3), where the input
 * is a packed u8 component vector. */
static inline void h_chunk_1_5x_avx2(__m256i A, __m256i B, __m256i C,
                                      uint8_t *restrict dst)
{
    /* The packed rows do not need the general six-mullo u16 path.  Apply the
     * byte-domain signed-difference blend directly to (A,B) and (C,B): four
     * maddubs replace six mullo operations and shorten the multiply chain,
     * at the cost of two additional byte unpacks. */
    __m256i out0_lo, out0_hi, out1_lo, out1_hi;
    avx2_blend_2_1_u16(A, B, &out0_lo, &out0_hi);
    avx2_blend_2_1_u16(C, B, &out1_lo, &out1_hi);

    __m256i out0 = _mm256_packus_epi16(out0_lo, out0_hi);
    __m256i out1 = _mm256_packus_epi16(out1_lo, out1_hi);
    __m256i interl_lo = _mm256_unpacklo_epi8(out0, out1);
    __m256i interl_hi = _mm256_unpackhi_epi8(out0, out1);

    _mm256_store_si256((__m256i *)dst,
        _mm256_permute2x128_si256(interl_lo, interl_hi, 0x20));
    _mm256_store_si256((__m256i *)(dst + 32),
        _mm256_permute2x128_si256(interl_lo, interl_hi, 0x31));
}


/* -----------------------------------------------------------------------
 * Horizontal 3x on one deinterleaved 256-bit chunk (A, B, C each 32
 * bytes, with lane 0 = group 1, lane 1 = group 2).
 *
 * Produces 32 output bytes stored at dst.  Returns the __m256i result
 * for cascading into 6x.
 *
 * Lane crossing note: packus(lo, hi) produces:
 *   lane 0 = lo.lane0 | hi.lane0  (group1: 16 bytes)
 *   lane 1 = lo.lane1 | hi.lane1  (group2: 16 bytes)
 * This is the correct contiguous layout.
 * ----------------------------------------------------------------------- */

static inline __m256i h_chunk_3x_avx2(__m256i A, __m256i B, __m256i C,
                                       uint8_t *restrict dst)
{
    __m256i result = box3_div_avx2(A, B, C);
    /* out_off_3x = ci*32; 32 is a multiple of 32, so dst is 32-byte aligned. */
    _mm256_store_si256((__m256i *)dst, result);
    return result;
}


/* -----------------------------------------------------------------------
 * Horizontal 6x cascaded from a 3x result (32 bytes -> 16 bytes).
 * Uses avx2_halve_32_to_16.  Stores 16 output bytes at dst.
 * ----------------------------------------------------------------------- */

static inline void h_chunk_6x_avx2(__m256i result_3x, uint8_t *restrict dst)
{
    __m128i result = avx2_halve_32_to_16(result_3x);
    _mm_storeu_si128((__m128i *)dst, result);
}


/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single plane (AVX2)
 *
 * Vertical: AVX2 _mm256_avg_epu8 cascade (32 bytes per chunk)
 * Horizontal: AVX2 pairwise average cascade + scalar tail
 *
 * Vertical levels remain separate passes because a single deep reduction tree
 * would exceed the useful YMM working set.  Each pass nevertheless finishes
 * the requested horizontal cascade from the freshly averaged vectors before
 * moving on.  Intermediate vertical rows are stored only when the next level
 * consumes them, and no full-width horizontal scratch row is needed.
 * ----------------------------------------------------------------------- */

static void __attribute__((noinline)) scale_plane_pow2_scratch_avx2(
    const uint8_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs, int deepest,
    uint8_t *restrict dst_planes[4],
    int dst_strides[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    static const int bit_pos[4] = { 1, 3, 5, 7 };
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

    int out_row[4] = { 0, 0, 0, 0 };

    for (int g = 0; g < num_groups; g++) {
        const uint8_t *grp_base = src + (size_t)g * (size_t)group_rows * (size_t)src_stride;

        /* -- Vertical cascade (AVX2) --------------------------------- */

        /* Level 0 (2x vertical): pairwise average source rows.  If the 2x
         * output is requested, emit its horizontal half here while each
         * vertical result is still in registers. */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint8_t *restrict ra = grp_base + (size_t)(2 * r)     * (size_t)src_stride;
            const uint8_t *restrict rb = grp_base + (size_t)(2 * r + 1) * (size_t)src_stride;
            uint8_t *restrict dst_row  = vert_buf[0] + (size_t)r * (size_t)src_w;
            int emit = (active_outputs & (1u << bit_pos[0])) != 0;
            int keep = deepest > 0;
            uint8_t *restrict out = emit ? dst_planes[0]
                + (size_t)out_row[0] * (size_t)dst_strides[0] : NULL;

            int x = 0;
            int out_x = 0;
            for (; x + 64 <= src_w; x += 64, out_x += 32) {
                __m256i v0 = _mm256_avg_epu8(
                    _mm256_loadu_si256((const __m256i *)(ra + x)),
                    _mm256_loadu_si256((const __m256i *)(rb + x)));
                __m256i v1 = _mm256_avg_epu8(
                    _mm256_loadu_si256((const __m256i *)(ra + x + 32)),
                    _mm256_loadu_si256((const __m256i *)(rb + x + 32)));
                if (keep) {
                    _mm256_storeu_si256((__m256i *)(dst_row + x), v0);
                    _mm256_storeu_si256((__m256i *)(dst_row + x + 32), v1);
                }
                if (emit) store_pow2_64(v0, v1, 0, out + out_x);
            }
            if (x + 32 <= src_w) {
                __m256i v = _mm256_avg_epu8(
                    _mm256_loadu_si256((const __m256i *)(ra + x)),
                    _mm256_loadu_si256((const __m256i *)(rb + x)));
                if (keep) _mm256_storeu_si256((__m256i *)(dst_row + x), v);
                if (emit) store_pow2_32(v, 0, out + out_x);
                x += 32;
                out_x += 16;
            }
            int tail_start = x;
            for (; x < src_w; x++) {
                dst_row[x] = avg_u8(ra[x], rb[x]);
            }
            if (emit) {
                for (int sx = tail_start; sx < src_w; sx += 2)
                    out[out_x++] = reduce_pow2_scalar(dst_row + sx, 0);
                out_row[0]++;
            }
        }

        /* Deeper levels use the same shape.  Keeping vertical levels as
         * separate passes avoids a high-pressure register tree, but folding
         * the horizontal result into the producing pass removes every
         * completed-row reload and every in-place h_buf pass. */
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint8_t *restrict ra = vert_buf[k - 1] + (size_t)(2 * r)     * (size_t)src_w;
                const uint8_t *restrict rb = vert_buf[k - 1] + (size_t)(2 * r + 1) * (size_t)src_w;
                uint8_t *restrict dst_row  = vert_buf[k] + (size_t)r * (size_t)src_w;
                int emit = (active_outputs & (1u << bit_pos[k])) != 0;
                int keep = k < deepest;
                uint8_t *restrict out = emit ? dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_strides[k] : NULL;

                int x = 0;
                int out_x = 0;
                int out_per_64 = 32 >> k;
                for (; x + 64 <= src_w; x += 64, out_x += out_per_64) {
                    __m256i v0 = _mm256_avg_epu8(
                        _mm256_loadu_si256((const __m256i *)(ra + x)),
                        _mm256_loadu_si256((const __m256i *)(rb + x)));
                    __m256i v1 = _mm256_avg_epu8(
                        _mm256_loadu_si256((const __m256i *)(ra + x + 32)),
                        _mm256_loadu_si256((const __m256i *)(rb + x + 32)));
                    if (keep) {
                        _mm256_storeu_si256((__m256i *)(dst_row + x), v0);
                        _mm256_storeu_si256((__m256i *)(dst_row + x + 32), v1);
                    }
                    if (emit) store_pow2_64(v0, v1, k, out + out_x);
                }
                if (x + 32 <= src_w) {
                    __m256i v = _mm256_avg_epu8(
                        _mm256_loadu_si256((const __m256i *)(ra + x)),
                        _mm256_loadu_si256((const __m256i *)(rb + x)));
                    if (keep) _mm256_storeu_si256((__m256i *)(dst_row + x), v);
                    if (emit) store_pow2_32(v, k, out + out_x);
                    x += 32;
                    out_x += 16 >> k;
                }
                int tail_start = x;
                for (; x < src_w; x++) {
                    dst_row[x] = avg_u8(ra[x], rb[x]);
                }
                if (emit) {
                    int step = 2 << k;
                    for (int sx = tail_start; sx < src_w; sx += step)
                        out[out_x++] = reduce_pow2_scalar(dst_row + sx, k);
                    out_row[k]++;
                }
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}

static void __attribute__((hot)) scale_plane_pow2_avx2(
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
    (void)dst_widths;

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

    if ((src_w & 63) == 0) {
        scale_plane_pow2_tree_64_avx2(src, src_w, src_h, src_stride,
                                      active_outputs, deepest,
                                      dst_planes, dst_strides);
        return;
    }
    if ((src_w & 31) == 0) {
        int paired_w = src_w & ~63;
        if (paired_w > 0) {
            scale_plane_pow2_tree_64_avx2(src, paired_w, src_h, src_stride,
                                          active_outputs, deepest,
                                          dst_planes, dst_strides);
        }
        scale_plane_pow2_tree_tail_32_avx2(src, paired_w, src_h, src_stride,
                                           active_outputs, deepest,
                                           dst_planes, dst_strides);
        return;
    }

    scale_plane_pow2_scratch_avx2(src, src_w, src_h, src_stride,
                                  active_outputs, deepest,
                                  dst_planes, dst_strides,
                                  scratch_pool_base, scratch_pool_size);
}


/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single plane (AVX2 fused vertical+horizontal)
 *
 * Per-chunk fused architecture: source rows are processed in groups of 6
 * (matching the vertical period of the thirds reduction).  For each 96-byte
 * column chunk, all 6 rows are loaded, vertical pair averages and bilinear
 * blends are computed entirely in YMM registers, and horizontal filtering
 * is applied immediately via 256-bit deinterleave and arithmetic - no
 * intermediate row buffer needed.
 *
 * Vertical: AVX2 _mm256_avg_epu8 for pairwise avgs, avx2_blend_2_1_u16 for
 *           bilinear blends (1.5x)
 * Horizontal: AVX2 deinterleave_3x32 + h_chunk_{1_5x,3x,6x}_avx2
 *
 * A 6-source-row group produces 4 output rows at 1.5x, 2 rows at 3x, and
 * 1 row at 6x.  12x requires pairing two consecutive 6x rows via a
 * ping-pong buffer scheme.
 * ----------------------------------------------------------------------- */

static void __attribute__((noinline)) scale_plane_thirds_tail_strip_avx2(
    const uint8_t *restrict src,
    int full_chunks, int tail_cols, int base6_groups, int src_stride,
    uint32_t active_outputs,
    int need_1_5x, int need_3x, int need_6x, int need_12x,
    uint8_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4])
{
    const int tail_start    = full_chunks * 96;
    const int tail_out_1_5x = full_chunks * 64;
    const int tail_out_3x   = full_chunks * 32;
    const int tail_out_6x   = full_chunks * 16;
    const int tail_out_12x  = full_chunks * 8;
    const ptrdiff_t ss  = (ptrdiff_t)src_stride;
    const ptrdiff_t ds0 = (ptrdiff_t)dst_strides[0];
    const ptrdiff_t ds1 = (ptrdiff_t)dst_strides[1];
    const ptrdiff_t ds2 = (ptrdiff_t)dst_strides[2];
    const ptrdiff_t ds3 = (ptrdiff_t)dst_strides[3];
    uint8_t *p0 = dst_planes[0];
    uint8_t *p1 = dst_planes[1];
    uint8_t *p2 = dst_planes[2];
    uint8_t *p3 = dst_planes[3];
    const int dw0_rem = need_1_5x ? (dst_widths[0] - tail_out_1_5x) : 0;
    const int dw1_rem = (active_outputs & (1u << 2)) ? (dst_widths[1] - tail_out_3x) : 0;
    const int dw2_rem = (active_outputs & (1u << 4)) ? (dst_widths[2] - tail_out_6x) : 0;
    const int dw3_rem = (active_outputs & (1u << 6)) ? (dst_widths[3] - tail_out_12x) : 0;

    uint8_t tail_v6x_prev[96];

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint8_t *grp = src + (size_t)g6 * 6 * (size_t)ss + tail_start;
        const uint8_t *row0 = grp;
        const uint8_t *row1 = grp + ss;
        const uint8_t *row2 = grp + ss * 2;
        const uint8_t *row3 = grp + ss * 3;
        const uint8_t *row4 = grp + ss * 4;
        const uint8_t *row5 = grp + ss * 5;

        uint8_t tail_v01[96], tail_v23[96], tail_v45[96];
        for (int x = 0; x < tail_cols; x++) {
            tail_v01[x] = avg_u8(row0[x], row1[x]);
            tail_v23[x] = avg_u8(row2[x], row3[x]);
            tail_v45[x] = avg_u8(row4[x], row5[x]);
        }

        uint8_t tail_v3x0[96], tail_v3x1[96];
        if (need_3x) {
            for (int x = 0; x < tail_cols; x++) {
                tail_v3x0[x] = avg_u8(tail_v01[x], tail_v23[x]);
                tail_v3x1[x] = avg_u8(tail_v23[x], tail_v45[x]);
            }
        }

        uint8_t tail_v6x[96];
        if (need_6x) {
            for (int x = 0; x < tail_cols; x++) {
                tail_v6x[x] = avg_u8(tail_v3x0[x], tail_v3x1[x]);
            }
        }

        if (need_1_5x) {
            uint8_t *r0 = p0 + (size_t)(g6 * 4) * (size_t)ds0 + tail_out_1_5x;
            h_filter_1_5x(tail_v01, tail_cols, r0, dw0_rem);

            uint8_t tail_blend[96];
            for (int x = 0; x < tail_cols; x++)
                tail_blend[x] = blend_2_1(tail_v01[x], tail_v23[x]);
            h_filter_1_5x(tail_blend, tail_cols, r0 + ds0, dw0_rem);

            for (int x = 0; x < tail_cols; x++)
                tail_blend[x] = blend_2_1(tail_v23[x], tail_v45[x]);
            h_filter_1_5x(tail_blend, tail_cols, r0 + ds0 * 2, dw0_rem);

            h_filter_1_5x(tail_v45, tail_cols, r0 + ds0 * 3, dw0_rem);
        }

        if (active_outputs & (1u << 2)) {
            uint8_t *r0 = p1 + (size_t)(g6 * 2) * (size_t)ds1 + tail_out_3x;
            h_filter_3x(tail_v3x0, tail_cols, r0, dw1_rem);
            h_filter_3x(tail_v3x1, tail_cols, r0 + ds1, dw1_rem);
        }

        if (active_outputs & (1u << 4)) {
            uint8_t *r0 = p2 + (size_t)g6 * (size_t)ds2 + tail_out_6x;
            int w3_tail = tail_cols / 3;
            uint8_t tail_h3x[32];
            h_filter_3x(tail_v6x, tail_cols, tail_h3x, w3_tail);
            h_filter_halve(tail_h3x, r0, dw2_rem);
        }

        if (need_12x) {
            if ((g6 & 1) == 0) {
                memcpy(tail_v6x_prev, tail_v6x, (size_t)tail_cols);
            } else if ((active_outputs & (1u << 6)) && dw3_rem > 0) {
                uint8_t tail_v12x[96], tail_h3x[32], tail_h6x[16];
                for (int x = 0; x < tail_cols; x++)
                    tail_v12x[x] = avg_u8(tail_v6x_prev[x], tail_v6x[x]);
                int w3_tail = tail_cols / 3;
                int w6_tail = w3_tail / 2;
                h_filter_3x(tail_v12x, tail_cols, tail_h3x, w3_tail);
                h_filter_halve(tail_h3x, tail_h6x, w6_tail);
                h_filter_halve(tail_h6x,
                               p3 + (size_t)(g6 >> 1) * (size_t)ds3 + tail_out_12x,
                               dw3_rem);
            }
        }
    }
}

static void __attribute__((hot, aligned(64))) scale_plane_thirds_avx2(
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

    const int need_1_5x = (active_outputs & (1u << 0)) != 0;
    const int emit_3x   = (active_outputs & (1u << 2)) != 0;
    const int emit_6x   = (active_outputs & (1u << 4)) != 0;
    const int emit_12x  = (active_outputs & (1u << 6)) != 0;
    const int need_3x   = (deepest >= 1);
    const int need_6x   = (deepest >= 2);
    const int need_12x  = (deepest >= 3);

    const int base6_groups = src_h / 6;
    const int full_chunks  = src_w / 96;
    const int tail_cols    = src_w - full_chunks * 96;

    /* Single planar 12x carry buffer ([A 32B | B 32B | C 32B] per 96B chunk)
     * written on even groups and consumed directly inside the odd-group chunk
     * loop.  96 = 8 * 12 never crosses a 12x boundary, so full chunks fuse
     * regardless of whether tail_cols > 0. */
    uint8_t *v6x_prev = NULL;
    if (need_12x && full_chunks > 0) {
        fused_scratch_t scratch;
        fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);
        v6x_prev = (uint8_t *)fused_scratch_alloc(&scratch,
                                                  (size_t)full_chunks * 96);
        if (!v6x_prev) return;
    }

    if (full_chunks > 0) {
        const ptrdiff_t ss  = (ptrdiff_t)src_stride;
        const ptrdiff_t ds0 = (ptrdiff_t)dst_strides[0];
        const ptrdiff_t ds1 = (ptrdiff_t)dst_strides[1];
        const ptrdiff_t ds2 = (ptrdiff_t)dst_strides[2];
        const ptrdiff_t ds3 = (ptrdiff_t)dst_strides[3];

        const uint8_t *grp = src;
        uint8_t *p0 = need_1_5x ? dst_planes[0] : NULL;
        uint8_t *p1 = emit_3x   ? dst_planes[1] : NULL;
        uint8_t *p2 = emit_6x   ? dst_planes[2] : NULL;
        uint8_t *p3 = emit_12x  ? dst_planes[3] : NULL;

        for (int g6 = 0; g6 < base6_groups; g6++) {
            const uint8_t *restrict row0 = grp;
            const uint8_t *restrict row1 = grp + ss;
            const uint8_t *restrict row2 = grp + ss * 2;
            const uint8_t *restrict row3 = grp + ss * 3;
            const uint8_t *restrict row4 = grp + ss * 4;
            const uint8_t *restrict row5 = grp + ss * 5;

            uint8_t *dst_1_5x_r0 = p0;
            uint8_t *dst_1_5x_r1 = need_1_5x ? (p0 + ds0)     : NULL;
            uint8_t *dst_1_5x_r2 = need_1_5x ? (p0 + ds0 * 2) : NULL;
            uint8_t *dst_1_5x_r3 = need_1_5x ? (p0 + ds0 * 3) : NULL;
            uint8_t *dst_3x_r0   = p1;
            uint8_t *dst_3x_r1   = emit_3x ? (p1 + ds1) : NULL;
            uint8_t *dst_6x_r0   = p2;
            uint8_t *dst_12x_r0  = p3;

            /* ============================================================
             * MAIN CHUNK LOOP: process 96 source columns at a time.
             * Vertical intermediates stay in YMM registers; horizontal
             * filtering is applied immediately per chunk.
             * ============================================================ */
            for (int ci = 0; ci < full_chunks; ci++) {
                int cx = ci * 96;
                int out_off_1_5x = ci * 64;  /* 96 -> 64 output bytes */
                int out_off_3x   = ci * 32;  /* 96 -> 32 output bytes */
                int out_off_6x   = ci * 16;  /* 96 -> 16 output bytes */

                /* --- LOAD 6 rows x 3 YMM = 18 loads --- */
                __m256i r0a = _mm256_load_si256((const __m256i *)(row0 + cx));
                __m256i r0b = _mm256_load_si256((const __m256i *)(row0 + cx + 32));
                __m256i r0c = _mm256_load_si256((const __m256i *)(row0 + cx + 64));
                __m256i r1a = _mm256_load_si256((const __m256i *)(row1 + cx));
                __m256i r1b = _mm256_load_si256((const __m256i *)(row1 + cx + 32));
                __m256i r1c = _mm256_load_si256((const __m256i *)(row1 + cx + 64));
                __m256i r2a = _mm256_load_si256((const __m256i *)(row2 + cx));
                __m256i r2b = _mm256_load_si256((const __m256i *)(row2 + cx + 32));
                __m256i r2c = _mm256_load_si256((const __m256i *)(row2 + cx + 64));
                __m256i r3a = _mm256_load_si256((const __m256i *)(row3 + cx));
                __m256i r3b = _mm256_load_si256((const __m256i *)(row3 + cx + 32));
                __m256i r3c = _mm256_load_si256((const __m256i *)(row3 + cx + 64));
                __m256i r4a = _mm256_load_si256((const __m256i *)(row4 + cx));
                __m256i r4b = _mm256_load_si256((const __m256i *)(row4 + cx + 32));
                __m256i r4c = _mm256_load_si256((const __m256i *)(row4 + cx + 64));
                __m256i r5a = _mm256_load_si256((const __m256i *)(row5 + cx));
                __m256i r5b = _mm256_load_si256((const __m256i *)(row5 + cx + 32));
                __m256i r5c = _mm256_load_si256((const __m256i *)(row5 + cx + 64));

                /* --- VERTICAL PAIRWISE AVERAGES --- */
                __m256i v01a = _mm256_avg_epu8(r0a, r1a);
                __m256i v01b = _mm256_avg_epu8(r0b, r1b);
                __m256i v01c = _mm256_avg_epu8(r0c, r1c);
                __m256i v23a = _mm256_avg_epu8(r2a, r3a);
                __m256i v23b = _mm256_avg_epu8(r2b, r3b);
                __m256i v23c = _mm256_avg_epu8(r2c, r3c);
                __m256i v45a = _mm256_avg_epu8(r4a, r5a);
                __m256i v45b = _mm256_avg_epu8(r4b, r5b);
                __m256i v45c = _mm256_avg_epu8(r4c, r5c);

                if (need_1_5x) {
                    __m256i Av01, Bv01, Cv01, Av23, Bv23, Cv23, Av45, Bv45, Cv45;
                    deinterleave_3x32(v01a, v01b, v01c, &Av01, &Bv01, &Cv01);
                    deinterleave_3x32(v23a, v23b, v23c, &Av23, &Bv23, &Cv23);
                    deinterleave_3x32(v45a, v45b, v45c, &Av45, &Bv45, &Cv45);

                    h_chunk_1_5x_avx2(Av01, Bv01, Cv01, dst_1_5x_r0 + out_off_1_5x);
                    {
                        __m256i bA_lo, bA_hi, bB_lo, bB_hi, bC_lo, bC_hi;
                        avx2_blend_2_1_u16(Av01, Av23, &bA_lo, &bA_hi);
                        avx2_blend_2_1_u16(Bv01, Bv23, &bB_lo, &bB_hi);
                        avx2_blend_2_1_u16(Cv01, Cv23, &bC_lo, &bC_hi);
                        h_chunk_1_5x_u16_avx2(bA_lo, bA_hi, bB_lo, bB_hi,
                                              bC_lo, bC_hi,
                                              dst_1_5x_r1 + out_off_1_5x);
                    }
                    {
                        __m256i bA_lo, bA_hi, bB_lo, bB_hi, bC_lo, bC_hi;
                        avx2_blend_2_1_u16(Av23, Av45, &bA_lo, &bA_hi);
                        avx2_blend_2_1_u16(Bv23, Bv45, &bB_lo, &bB_hi);
                        avx2_blend_2_1_u16(Cv23, Cv45, &bC_lo, &bC_hi);
                        h_chunk_1_5x_u16_avx2(bA_lo, bA_hi, bB_lo, bB_hi,
                                              bC_lo, bC_hi,
                                              dst_1_5x_r2 + out_off_1_5x);
                    }
                    h_chunk_1_5x_avx2(Av45, Bv45, Cv45, dst_1_5x_r3 + out_off_1_5x);

                    /* 3x vertical reduction in component space (reused by 6x/12x). */
                    __m256i A3x0 = _mm256_setzero_si256();
                    __m256i B3x0 = _mm256_setzero_si256();
                    __m256i C3x0 = _mm256_setzero_si256();
                    __m256i A3x1 = _mm256_setzero_si256();
                    __m256i B3x1 = _mm256_setzero_si256();
                    __m256i C3x1 = _mm256_setzero_si256();
                    if (need_3x) {
                        A3x0 = _mm256_avg_epu8(Av01, Av23);
                        B3x0 = _mm256_avg_epu8(Bv01, Bv23);
                        C3x0 = _mm256_avg_epu8(Cv01, Cv23);
                        A3x1 = _mm256_avg_epu8(Av23, Av45);
                        B3x1 = _mm256_avg_epu8(Bv23, Bv45);
                        C3x1 = _mm256_avg_epu8(Cv23, Cv45);

                        if (emit_3x) {
                            h_chunk_3x_avx2(A3x0, B3x0, C3x0, dst_3x_r0 + out_off_3x);
                            h_chunk_3x_avx2(A3x1, B3x1, C3x1, dst_3x_r1 + out_off_3x);
                        }
                    }

                    if (emit_6x || need_12x) {
                        __m256i A6x = _mm256_avg_epu8(A3x0, A3x1);
                        __m256i B6x = _mm256_avg_epu8(B3x0, B3x1);
                        __m256i C6x = _mm256_avg_epu8(C3x0, C3x1);
                        if (emit_6x) {
                            h_chunk_6x_avx2(box3_div_avx2(A6x, B6x, C6x),
                                            dst_6x_r0 + out_off_6x);
                        }
                        if (need_12x) {
                            if ((g6 & 1) == 0) {
                                _mm256_storeu_si256((__m256i *)(v6x_prev + cx),      A6x);
                                _mm256_storeu_si256((__m256i *)(v6x_prev + cx + 32), B6x);
                                _mm256_storeu_si256((__m256i *)(v6x_prev + cx + 64), C6x);
                            } else if (emit_12x) {
                                __m256i A12 = _mm256_avg_epu8(
                                    _mm256_loadu_si256((const __m256i *)(v6x_prev + cx)), A6x);
                                __m256i B12 = _mm256_avg_epu8(
                                    _mm256_loadu_si256((const __m256i *)(v6x_prev + cx + 32)), B6x);
                                __m256i C12 = _mm256_avg_epu8(
                                    _mm256_loadu_si256((const __m256i *)(v6x_prev + cx + 64)), C6x);
                                __m256i box = box3_div_avx2(A12, B12, C12);
                                __m128i six = avx2_halve_32_to_16(box);
                                __m128i twl = halve_16_to_8(six);
                                _mm_storel_epi64((__m128i *)(dst_12x_r0 + ci * 8), twl);
                            }
                        }
                    }
                } else {
                    /* No 1.5x: deinterleave each reduced row directly. */
                    __m256i v3x0a = _mm256_setzero_si256();
                    __m256i v3x0b = _mm256_setzero_si256();
                    __m256i v3x0c = _mm256_setzero_si256();
                    __m256i v3x1a = _mm256_setzero_si256();
                    __m256i v3x1b = _mm256_setzero_si256();
                    __m256i v3x1c = _mm256_setzero_si256();
                    if (need_3x) {
                        v3x0a = _mm256_avg_epu8(v01a, v23a);
                        v3x0b = _mm256_avg_epu8(v01b, v23b);
                        v3x0c = _mm256_avg_epu8(v01c, v23c);
                        v3x1a = _mm256_avg_epu8(v23a, v45a);
                        v3x1b = _mm256_avg_epu8(v23b, v45b);
                        v3x1c = _mm256_avg_epu8(v23c, v45c);

                        if (emit_3x) {
                            __m256i A, B, C;
                            deinterleave_3x32(v3x0a, v3x0b, v3x0c, &A, &B, &C);
                            h_chunk_3x_avx2(A, B, C, dst_3x_r0 + out_off_3x);
                            deinterleave_3x32(v3x1a, v3x1b, v3x1c, &A, &B, &C);
                            h_chunk_3x_avx2(A, B, C, dst_3x_r1 + out_off_3x);
                        }
                    }
                    if (emit_6x || need_12x) {
                        __m256i v6xa = _mm256_avg_epu8(v3x0a, v3x1a);
                        __m256i v6xb = _mm256_avg_epu8(v3x0b, v3x1b);
                        __m256i v6xc = _mm256_avg_epu8(v3x0c, v3x1c);
                        __m256i A, B, C;
                        deinterleave_3x32(v6xa, v6xb, v6xc, &A, &B, &C);
                        if (emit_6x) {
                            h_chunk_6x_avx2(box3_div_avx2(A, B, C),
                                            dst_6x_r0 + out_off_6x);
                        }
                        if (need_12x) {
                            if ((g6 & 1) == 0) {
                                _mm256_storeu_si256((__m256i *)(v6x_prev + cx),      A);
                                _mm256_storeu_si256((__m256i *)(v6x_prev + cx + 32), B);
                                _mm256_storeu_si256((__m256i *)(v6x_prev + cx + 64), C);
                            } else if (emit_12x) {
                                __m256i A12 = _mm256_avg_epu8(
                                    _mm256_loadu_si256((const __m256i *)(v6x_prev + cx)), A);
                                __m256i B12 = _mm256_avg_epu8(
                                    _mm256_loadu_si256((const __m256i *)(v6x_prev + cx + 32)), B);
                                __m256i C12 = _mm256_avg_epu8(
                                    _mm256_loadu_si256((const __m256i *)(v6x_prev + cx + 64)), C);
                                __m256i box = box3_div_avx2(A12, B12, C12);
                                __m128i six = avx2_halve_32_to_16(box);
                                __m128i twl = halve_16_to_8(six);
                                _mm_storel_epi64((__m128i *)(dst_12x_r0 + ci * 8), twl);
                            }
                        }
                    }
                }
            } /* end chunk loop */

            grp += ss * 6;
            if (need_1_5x) p0 += ds0 * 4;
            if (emit_3x)   p1 += ds1 * 2;
            if (emit_6x)   p2 += ds2;
            if (emit_12x && (g6 & 1)) p3 += ds3;
        } /* end g6 loop */
    }

    if (tail_cols > 0) {
        scale_plane_thirds_tail_strip_avx2(
            src, full_chunks, tail_cols, base6_groups, src_stride,
            active_outputs, need_1_5x, need_3x, need_6x, need_12x,
            dst_planes, dst_widths, dst_strides);
    }
}


/* -----------------------------------------------------------------------
 * Public entry points
 *
 * YUV420 I420 has the chroma planes at half the luma dimensions in both
 * axes.  We process the Y plane at full size, then U and V at half width
 * and half height with the same kernel.
 * ----------------------------------------------------------------------- */

void __attribute__((hot)) fused_kernel_pow2_avx2(const fused_kernel_params_t *p,
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
    scale_plane_pow2_avx2(src_y,
                          p->src_width, p->src_height, p->src_y_stride,
                          p->active_outputs,
                          y_planes, y_widths, y_strides, y_heights,
                          p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_pow2_avx2(src_u,
                          p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                          p->active_outputs,
                          u_planes, uv_widths, uv_strides, uv_heights,
                          p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_pow2_avx2(src_v,
                          p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                          p->active_outputs,
                          v_planes, uv_widths, uv_strides, uv_heights,
                          p->scratch_pool, p->scratch_pool_size);

    _mm256_zeroupper();
}


void __attribute__((hot)) fused_kernel_thirds_avx2(const fused_kernel_params_t *p,
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
    scale_plane_thirds_avx2(src_y,
                            p->src_width, p->src_height, p->src_y_stride,
                            p->active_outputs,
                            y_planes, y_widths, y_strides, y_heights,
                            p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_thirds_avx2(src_u,
                            p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                            p->active_outputs,
                            u_planes, uv_widths, uv_strides, uv_heights,
                            p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_thirds_avx2(src_v,
                            p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                            p->active_outputs,
                            v_planes, uv_widths, uv_strides, uv_heights,
                            p->scratch_pool, p->scratch_pool_size);

    _mm256_zeroupper();
}

#endif /* __x86_64__ */
