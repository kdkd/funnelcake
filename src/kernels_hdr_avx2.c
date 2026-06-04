/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_hdr_avx2.c - AVX2 (x86_64) 10-bit HDR fused downscale kernels.
 *
 * Two entry points:
 *   fused_kernel_pow2_hdr_avx2   - power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_hdr_avx2 - thirds family (1.5x/3x/6x/12x)
 *
 * Both process YUV420 I010 (planar) or P010 (semi-planar) frames
 * plane-by-plane, with uint16_t samples (10-bit values, max 1023).
 *
 * Adapted from the 8-bit kernels_avx2.c.  Key differences:
 *   - Each YMM register holds 16 uint16_t elements (not 32 uint8_t).
 *   - Pairwise averaging uses _mm256_avg_epu16 (AVX2): (a + b + 1) >> 1.
 *   - Bilinear blends require widening to 32-bit because 1023 * 171
 *     = 174,933 overflows uint16_t.
 *   - Deinterleave shuffle tables operate on 2-byte elements instead
 *     of single bytes.
 *   - P010 chroma deinterleaving is handled at the entry point by
 *     separating even/odd uint16_t elements into planar U and V buffers.
 *
 * Guarded by __x86_64__ so this file is a no-op on other platforms.
 */

#if defined(__x86_64__)

#include "internal.h"
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * Scalar helpers (used for tail elements and horizontal filters)
 * ----------------------------------------------------------------------- */

/* avg_u16: rounded average of two uint16 values, (a+b+1)>>1.
 * Matches _mm256_avg_epu16 used in the vector path.  For 10-bit values
 * (max 1023), a+b+1 max = 2047 which fits in uint16_t. */
static inline uint16_t avg_u16(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a + (uint32_t)b + 1) >> 1);
}

/* blend_2_1_u16: bilinear blend for the 3:2 horizontal reduction.
 * (a * 171 + b * 85 + 128) >> 8  ~  a*2/3 + b*1/3.
 * Must use uint32_t: 10-bit a * 171 = max 174,933 overflows uint16_t. */
static inline uint16_t blend_2_1_u16(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a * 171 + (uint32_t)b * 85 + 128) >> 8);
}

/* div3_u32: integer divide-by-3 for a sum of three uint16 values.
 * Max sum = 3069 (3 * 1023).  21846 / 65536 ~ 1/3.
 * Exact for sum <= 3069.  Uses uint32_t multiply. */
static inline uint16_t div3_u32(uint32_t sum)
{
    return (uint16_t)(((uint32_t)sum * 21846u) >> 16);
}


/* -----------------------------------------------------------------------
 * Horizontal scalar filters for 10-bit (used for tail handling)
 * ----------------------------------------------------------------------- */

/* Horizontal 1.5x filter: 3:2 bilinear reduction for uint16_t.
 * Every 3 source elements -> 2 output elements via weighted blend. */
static void h_filter_1_5x_hdr(
    const uint16_t *restrict src, int src_w,
    uint16_t *restrict dst, int dst_w)
{
    int x_out = 0;
    for (int x_in = 0; x_in < src_w - 2 && x_out < dst_w - 1;
         x_in += 3, x_out += 2) {
        dst[x_out]     = blend_2_1_u16(src[x_in],     src[x_in + 1]);
        dst[x_out + 1] = blend_2_1_u16(src[x_in + 2], src[x_in + 1]);
    }
}

/* Horizontal 3x filter: box average of 3 source elements for uint16_t. */
static void h_filter_3x_hdr(
    const uint16_t *restrict src, int src_w,
    uint16_t *restrict dst, int dst_w)
{
    (void)src_w;
    for (int x = 0; x < dst_w; x++) {
        uint32_t sum = (uint32_t)src[3 * x]
                     + (uint32_t)src[3 * x + 1]
                     + (uint32_t)src[3 * x + 2];
        dst[x] = div3_u32(sum);
    }
}

/* Horizontal pairwise halving for uint16_t. */
static void h_filter_halve_hdr(
    const uint16_t *restrict src,
    uint16_t *restrict dst, int dst_w)
{
    for (int x = 0; x < dst_w; x++) {
        dst[x] = avg_u16(src[2 * x], src[2 * x + 1]);
    }
}


/* -----------------------------------------------------------------------
 * AVX2 pairwise average for uint16_t
 *
 * AVX2 provides _mm256_avg_epu16 directly (Haswell+); it computes
 * (a + b + 1) >> 1 per element, which is exactly what we need for the
 * vertical pairwise averages.  This wrapper exists only as a named
 * shorthand to keep call sites symmetric with the 8-bit kernel.
 * ----------------------------------------------------------------------- */

static inline __m256i avx2_avg_u16(__m256i a, __m256i b)
{
    return _mm256_avg_epu16(a, b);
}


/* -----------------------------------------------------------------------
 * AVX2 vertical blend helper for 10-bit
 *
 * Computes (a * 171 + b * 85 + 128) >> 8 across 16 uint16_t elements.
 * 1023 * 171 = 174,933 overflows 16 bits, so the weighted sum is built in
 * 32-bit.  Interleaving a and b into (a_i, b_i) pairs and running one
 * vpmaddwd against the weight pair {171, 85} gives 171*a_i + 85*b_i for each
 * pair directly, a single multiply-add in place of two multiplies and an add.
 * 10-bit inputs keep every product and the sum well inside a signed 32-bit
 * lane.  unpacklo/hi feed the low and high element groups and packus_epi32
 * narrows back to 16-bit; both work within 128-bit lanes, so the packed
 * result stays in element order.
 * ----------------------------------------------------------------------- */

static inline __m256i avx2_blend_2_1_hdr(__m256i a, __m256i b)
{
    /* Weight pair: the low 16 bits multiply a, the high 16 bits multiply b. */
    const __m256i w   = _mm256_set1_epi32((85 << 16) | 171);
    const __m256i rnd = _mm256_set1_epi32(128);

    __m256i ab_lo = _mm256_unpacklo_epi16(a, b);
    __m256i ab_hi = _mm256_unpackhi_epi16(a, b);

    __m256i lo = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_madd_epi16(ab_lo, w), rnd), 8);
    __m256i hi = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_madd_epi16(ab_hi, w), rnd), 8);

    return _mm256_packus_epi32(lo, hi);
}


/* -----------------------------------------------------------------------
 * AVX2 horizontal halving helper for 10-bit
 *
 * Input: __m256i with 16 uint16_t elements (32 bytes).
 * Output: __m128i with 8 uint16_t elements (16 bytes).
 *
 * Separates even-indexed (0,2,4,6) and odd-indexed (1,3,5,7) 16-bit
 * words within each 128-bit lane using vpshufb, then gathers evens
 * and odds across lanes with vpermq, and averages them.
 *
 * The shuffle mask moves 2-byte word pairs:
 *   Within each lane (8 words = 16 bytes):
 *     Even words (0,2,4,6) -> bytes 0-7 (low 64 bits)
 *     Odd words  (1,3,5,7) -> bytes 8-15 (high 64 bits)
 *
 * vpermq(0xD8) = 11_01_10_00 rearranges qwords:
 *   qword 0 = even from lane 0, qword 1 = even from lane 1
 *   qword 2 = odd from lane 0,  qword 3 = odd from lane 1
 * Low 128 = all 8 even elements, high 128 = all 8 odd elements.
 * ----------------------------------------------------------------------- */

static inline __m128i avx2_halve_16_to_8_hdr(__m256i v)
{
    /* Within each 128-bit lane, separate even-indexed 16-bit words
     * (positions 0,2,4,6) from odd-indexed words (positions 1,3,5,7).
     *
     * Word 0 = bytes 0,1; Word 1 = bytes 2,3; Word 2 = bytes 4,5;
     * Word 3 = bytes 6,7; Word 4 = bytes 8,9; Word 5 = bytes 10,11;
     * Word 6 = bytes 12,13; Word 7 = bytes 14,15.
     *
     * Even words (0,2,4,6) byte offsets: 0,1, 4,5, 8,9, 12,13
     * Odd  words (1,3,5,7) byte offsets: 2,3, 6,7, 10,11, 14,15  */
    static const uint8_t shuf_even_odd[32] __attribute__((aligned(32))) = {
        /* lane 0: even words to low 64 bits, odd words to high 64 bits */
        0, 1, 4, 5, 8, 9, 12, 13,    2, 3, 6, 7, 10, 11, 14, 15,
        /* lane 1: same pattern */
        0, 1, 4, 5, 8, 9, 12, 13,    2, 3, 6, 7, 10, 11, 14, 15
    };
    __m256i mask = _mm256_load_si256((const __m256i *)shuf_even_odd);
    __m256i shuffled = _mm256_shuffle_epi8(v, mask);

    /* After shuffle, within each 128-bit lane:
     *   bytes 0-7  (qword lo): 4 even-indexed uint16_t elements
     *   bytes 8-15 (qword hi): 4 odd-indexed uint16_t elements
     *
     * vpermq with control 0xD8 = 11_01_10_00 rearranges qwords:
     *   dst qword 0 = src qword 0 (even lane0)
     *   dst qword 1 = src qword 2 (even lane1)
     *   dst qword 2 = src qword 1 (odd lane0)
     *   dst qword 3 = src qword 3 (odd lane1)
     *
     * Result: low 128 = all 8 even elements, high 128 = all 8 odd elements */
    __m256i permuted = _mm256_permute4x64_epi64(shuffled, 0xD8);

    __m128i even = _mm256_castsi256_si128(permuted);
    __m128i odd  = _mm256_extracti128_si256(permuted, 1);

    /* Average: (even + odd + 1) >> 1.  No _mm_avg_epu16 exists either,
     * so we compute manually.  10-bit values: max sum = 2047 fits in uint16. */
    __m128i sum = _mm_add_epi16(even, odd);
    __m128i one = _mm_set1_epi16(1);
    return _mm_srli_epi16(_mm_add_epi16(sum, one), 1);
}


/* -----------------------------------------------------------------------
 * AVX2 horizontal halving helper for 10-bit, paired: 32 -> 16 samples.
 *
 * Averages adjacent pairs from two input YMM (32 uint16_t) into one output
 * YMM (16 uint16_t):  out[i] = (in[2i] + in[2i+1] + 1) >> 1.
 *
 * Separating the even- and odd-indexed samples before averaging would lean
 * on the shuffle ports (vpshufb / vpermq / vextracti128).  vpmaddwd does the
 * pairing for free instead: multiplying each 16-bit sample by a constant 1
 * and summing adjacent products yields in[2i] + in[2i+1] directly, as 32-bit
 * values.  10-bit inputs (max 1023) stay positive and the pair sum is at most
 * 2046, so it fits a 32-bit lane with no overflow.  The rounded average
 * (sum + 1) >> 1 is then a 32-bit add and shift.  This keeps the work off the
 * shuffle ports and emits a single wide store per 16 outputs.
 *
 * vpmaddwd works within each 128-bit lane, so after packus_epi32 the 16
 * results are grouped by lane rather than in order; permute4x64(0xD8) puts
 * them back into linear sample order.
 * ----------------------------------------------------------------------- */

static inline __m256i avx2_halve_32_to_16_hdr(__m256i v0, __m256i v1)
{
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i one32  = _mm256_set1_epi32(1);

    __m256i m0 = _mm256_madd_epi16(v0, ones16);  /* 8x (a + b), <= 2046 */
    __m256i m1 = _mm256_madd_epi16(v1, ones16);

    m0 = _mm256_srli_epi32(_mm256_add_epi32(m0, one32), 1);  /* (a+b+1)>>1 */
    m1 = _mm256_srli_epi32(_mm256_add_epi32(m1, one32), 1);

    __m256i packed = _mm256_packus_epi32(m0, m1);
    return _mm256_permute4x64_epi64(packed, 0xD8);
}


/* -----------------------------------------------------------------------
 * AVX2 horizontal halving helper for 10-bit: 8 -> 4 elements (XMM -> 64-bit)
 *
 * out[k] = avg(in[2k], in[2k+1]) = (in[2k] + in[2k+1] + 1) >> 1, for k in 0..3.
 * Same even/odd-word separation as avx2_halve_16_to_8_hdr but on a single
 * 128-bit lane: vpshufb gathers the 4 even-indexed words into the low 64 bits
 * and the 4 odd-indexed words into the high 64 bits, then (sum+1)>>1.  The 4
 * results sit in the low 64 bits, ready for _mm_storel_epi64.
 * ----------------------------------------------------------------------- */

static inline __m128i halve_8_to_4_hdr(__m128i v)
{
    static const uint8_t shuf_even_odd8[16] __attribute__((aligned(16))) = {
        /* even words 0,2,4,6 -> low 64; odd words 1,3,5,7 -> high 64 */
        0, 1, 4, 5, 8, 9, 12, 13,    2, 3, 6, 7, 10, 11, 14, 15
    };
    __m128i mask = _mm_load_si128((const __m128i *)shuf_even_odd8);
    __m128i s = _mm_shuffle_epi8(v, mask);
    __m128i even = s;
    __m128i odd  = _mm_srli_si128(s, 8);
    /* 10-bit values: max sum = 2046 fits in uint16. */
    __m128i sum = _mm_add_epi16(even, odd);
    __m128i one = _mm_set1_epi16(1);
    return _mm_srli_epi16(_mm_add_epi16(sum, one), 1);
}


/* -----------------------------------------------------------------------
 * SSE shuffle tables for 10-bit 3-way deinterleave
 *
 * For uint16_t elements, each element is 2 bytes.  Given 48 bytes
 * (24 uint16_t elements = 8 triplets) across three 128-bit registers
 * (r0, r1, r2 each holding 8 elements), the triplet layout is:
 *
 *   r0 = [A0 B0 C0 A1 B1 C1 A2 B2]   (elements 0-7)
 *   r1 = [C2 A3 B3 C3 A4 B4 C4 A5]   (elements 8-15)
 *   r2 = [B5 C5 A6 B6 C6 A7 B7 C7]   (elements 16-23)
 *
 * Component A positions (element indices within each register):
 *   from r0: elements 0, 3, 6 => byte offsets 0,1,  6,7,  12,13
 *   from r1: elements 1, 4, 7 => byte offsets 2,3,  8,9,  14,15
 *   from r2: elements 2, 5    => byte offsets 4,5,  10,11
 *
 * The shuffle tables place each component's bytes at the correct
 * output position and set 0x80 for positions that come from other
 * registers.  OR-combining three shuffled results produces the
 * complete 8-element component vector.
 * ----------------------------------------------------------------------- */

#define ALIGN16 __attribute__((aligned(16)))

/* Component A: elements at triplet positions 0, 3, 6, ... */
/* From r0: elements 0, 3, 6 -> output positions 0, 1, 2 */
static const uint8_t ALIGN16 shuf_A16_r0[16] = {
    0, 1, 6, 7, 12, 13,                           /* 3 elements from r0 */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* padding for r1 slots */
    0x80, 0x80, 0x80, 0x80                          /* padding for r2 slots */
};
/* From r1: elements 1, 4, 7 -> output positions 3, 4, 5 */
static const uint8_t ALIGN16 shuf_A16_r1[16] = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* r0 slots */
    2, 3, 8, 9, 14, 15,                             /* 3 elements from r1 */
    0x80, 0x80, 0x80, 0x80                          /* r2 slots */
};
/* From r2: elements 2, 5 -> output positions 6, 7 */
static const uint8_t ALIGN16 shuf_A16_r2[16] = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* r0 slots */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* r1 slots */
    4, 5, 10, 11                                    /* 2 elements from r2 */
};

/* Component B: elements at triplet positions 1, 4, 7, ...
 *   from r0: elements 1, 4, 7 => byte offsets 2,3,  8,9,  14,15
 *   from r1: elements 2, 5    => byte offsets 4,5,  10,11
 *   from r2: elements 0, 3, 6 => byte offsets 0,1,  6,7,  12,13 */
static const uint8_t ALIGN16 shuf_B16_r0[16] = {
    2, 3, 8, 9, 14, 15,                             /* 3 elements from r0 */
    0x80, 0x80, 0x80, 0x80,                         /* r1 slots */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80              /* r2 slots */
};
static const uint8_t ALIGN16 shuf_B16_r1[16] = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* r0 slots */
    4, 5, 10, 11,                                   /* 2 elements from r1 */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80              /* r2 slots */
};
static const uint8_t ALIGN16 shuf_B16_r2[16] = {
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* r0 slots */
    0x80, 0x80, 0x80, 0x80,                         /* r1 slots */
    0, 1, 6, 7, 12, 13                              /* 3 elements from r2 */
};

/* Component C: elements at triplet positions 2, 5, 8, ...
 *   from r0: elements 2, 5    => byte offsets 4,5,  10,11
 *   from r1: elements 0, 3, 6 => byte offsets 0,1,  6,7,  12,13
 *   from r2: elements 1, 4, 7 => byte offsets 2,3,  8,9,  14,15 */
static const uint8_t ALIGN16 shuf_C16_r0[16] = {
    4, 5, 10, 11,                                   /* 2 elements from r0 */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* r1 slots */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80              /* r2 slots */
};
static const uint8_t ALIGN16 shuf_C16_r1[16] = {
    0x80, 0x80, 0x80, 0x80,                         /* r0 slots */
    0, 1, 6, 7, 12, 13,                             /* 3 elements from r1 */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80              /* r2 slots */
};
static const uint8_t ALIGN16 shuf_C16_r2[16] = {
    0x80, 0x80, 0x80, 0x80,                         /* r0 slots */
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,            /* r1 slots */
    2, 3, 8, 9, 14, 15                              /* 3 elements from r2 */
};


/* -----------------------------------------------------------------------
 * AVX2 256-bit deinterleave for 10-bit: 96 bytes (3 x __m256i,
 * 48 uint16_t elements = 16 triplets) into three 16-element vectors.
 *
 * Same strategy as the 8-bit deinterleave_3x32: split the 96 bytes
 * into two independent 48-byte groups and place each in a separate
 * 128-bit lane for parallel processing.
 *
 * Input: reg_a (bytes 0-31, 16 elements), reg_b (bytes 32-63),
 *        reg_c (bytes 64-95).
 *
 * Group 1 (bytes 0-47,  24 elements = 8 triplets):
 *   a.lo = elements 0-7   ("r0")
 *   a.hi = elements 8-15  ("r1")
 *   b.lo = elements 16-23 ("r2")
 *
 * Group 2 (bytes 48-95, 24 elements = 8 triplets):
 *   b.hi = elements 24-31 ("r0")
 *   c.lo = elements 32-39 ("r1")
 *   c.hi = elements 40-47 ("r2")
 *
 * vperm2i128 aligns so each lane has the matching sub-block:
 *   nr0 = [a.lo | b.hi]   "r0" for each group
 *   nr1 = [a.hi | c.lo]   "r1" for each group
 *   nr2 = [b.lo | c.hi]   "r2" for each group
 *
 * Broadcasting the 128-bit shuffle masks to both lanes and applying
 * vpshufb then deinterleaves both groups simultaneously.
 * ----------------------------------------------------------------------- */

static inline void deinterleave_3x16_hdr(__m256i reg_a, __m256i reg_b, __m256i reg_c,
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
    __m256i mA0 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_A16_r0));
    __m256i mA1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_A16_r1));
    __m256i mA2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_A16_r2));

    *out_A = _mm256_or_si256(_mm256_or_si256(
        _mm256_shuffle_epi8(nr0, mA0),
        _mm256_shuffle_epi8(nr1, mA1)),
        _mm256_shuffle_epi8(nr2, mA2));

    __m256i mB0 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_B16_r0));
    __m256i mB1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_B16_r1));
    __m256i mB2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_B16_r2));

    *out_B = _mm256_or_si256(_mm256_or_si256(
        _mm256_shuffle_epi8(nr0, mB0),
        _mm256_shuffle_epi8(nr1, mB1)),
        _mm256_shuffle_epi8(nr2, mB2));

    __m256i mC0 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_C16_r0));
    __m256i mC1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_C16_r1));
    __m256i mC2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)shuf_C16_r2));

    *out_C = _mm256_or_si256(_mm256_or_si256(
        _mm256_shuffle_epi8(nr0, mC0),
        _mm256_shuffle_epi8(nr1, mC1)),
        _mm256_shuffle_epi8(nr2, mC2));
}


/* -----------------------------------------------------------------------
 * Horizontal 1.5x chunk helper for 10-bit (AVX2)
 *
 * Takes deinterleaved A, B, C (each 16 uint16_t in a YMM, lane 0 =
 * group 1, lane 1 = group 2).  Produces 32 output elements (64 bytes)
 * stored at dst.
 *
 * The blend for 10-bit requires widening to 32-bit because
 * 1023 * 171 = 174,933 overflows uint16_t.  We process each YMM's
 * low and high halves separately, widening 16->32, multiplying,
 * shifting, and packing back to 16-bit.
 *
 * After blending, the interleave step reorders [out0, out1] into
 * memory layout [out0[0], out1[0], out0[1], out1[1], ...] using
 * _mm256_unpacklo/hi_epi16, then vperm2i128 to fix lane crossing.
 * ----------------------------------------------------------------------- */

static inline void h_chunk_1_5x_hdr_avx2(__m256i A, __m256i B, __m256i C,
                                           uint16_t *restrict dst)
{
    /* out0 = (A * 171 + B * 85 + 128) >> 8 */
    __m256i out0 = avx2_blend_2_1_hdr(A, B);

    /* out1 = (C * 171 + B * 85 + 128) >> 8 */
    __m256i out1 = avx2_blend_2_1_hdr(C, B);

    /* Interleave out0/out1 pairs (16-bit elements).
     * unpacklo_epi16 interleaves the low 4 elements per lane,
     * unpackhi_epi16 interleaves the high 4 elements per lane. */
    __m256i interl_lo = _mm256_unpacklo_epi16(out0, out1);
    __m256i interl_hi = _mm256_unpackhi_epi16(out0, out1);

    /* Fix lane crossing: group outputs contiguous.
     * After unpacklo/hi, each lane has interleaved data from one group.
     * vperm2i128 gathers both halves of each group:
     *   store0 = [group1_lo | group1_hi] = lane0 of interl_lo | lane0 of interl_hi
     *   store1 = [group2_lo | group2_hi] = lane1 of interl_lo | lane1 of interl_hi */
    __m256i store0 = _mm256_permute2x128_si256(interl_lo, interl_hi, 0x20);
    __m256i store1 = _mm256_permute2x128_si256(interl_lo, interl_hi, 0x31);

    _mm256_store_si256((__m256i *)(dst),      store0);
    _mm256_store_si256((__m256i *)(dst + 16), store1);
}


/* -----------------------------------------------------------------------
 * Horizontal 3x chunk helper for 10-bit (AVX2)
 *
 * Takes deinterleaved A, B, C (each 16 uint16_t, lane 0 = group 1,
 * lane 1 = group 2).  Produces 16 output elements (32 bytes) stored
 * at dst.  Returns the __m256i result for cascading into 6x.
 *
 * Divide-by-3: sum of three uint16_t values (max 3069) fits in uint16_t.
 * _mm256_mulhi_epu16(sum, 0x5556) computes (sum * 21846) >> 16 ~ sum/3.
 * Same magic constant as the 8-bit version because the sum still fits
 * in 16 bits.
 * ----------------------------------------------------------------------- */

/* Box-of-3 divide: average a deinterleaved triplet (A, B, C) by adding the
 * three and dividing by 3 via (sum * 0x5556) >> 16.  For 10-bit values the
 * sum is at most 3069, which fits in 16 bits, so this magic-constant divide
 * is exact.  Returning the result in a register lets the 12x pipeline chain
 * straight into the next step; h_chunk_3x_hdr_avx2 wraps this and also writes
 * the 3x row out. */
static inline __m256i box3_div_hdr(__m256i A, __m256i B, __m256i C)
{
    __m256i magic = _mm256_set1_epi16((short)0x5556);
    __m256i sum = _mm256_add_epi16(_mm256_add_epi16(A, B), C);
    return _mm256_mulhi_epu16(sum, magic);
}

static inline __m256i h_chunk_3x_hdr_avx2(__m256i A, __m256i B, __m256i C,
                                            uint16_t *restrict dst)
{
    __m256i result = box3_div_hdr(A, B, C);
    _mm256_store_si256((__m256i *)dst, result);
    return result;
}


/* -----------------------------------------------------------------------
 * Horizontal 6x chunk helper for 10-bit (AVX2)
 *
 * Cascaded from a 3x result: 16 uint16_t elements -> 8 uint16_t
 * elements via pairwise halving.  Stores 16 bytes (8 elements) at dst.
 * ----------------------------------------------------------------------- */

static inline void h_chunk_6x_hdr_avx2(__m256i result_3x, uint16_t *restrict dst)
{
    __m128i result = avx2_halve_16_to_8_hdr(result_3x);
    _mm_storeu_si128((__m128i *)dst, result);
}


/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single 10-bit plane (AVX2)
 *
 * Vertical: AVX2 avx2_avg_u16 cascade (16 elements per YMM chunk)
 * Horizontal: AVX2 pairwise halving cascade + scalar tail
 *
 * Same architecture as the 8-bit scale_plane_pow2_avx2 but operating
 * on uint16_t elements.  Each YMM register holds 16 elements (32 bytes)
 * instead of 32 bytes of 8-bit data.
 * ----------------------------------------------------------------------- */

static void __attribute__((hot)) scale_plane_pow2_hdr_avx2(
    const uint16_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint16_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    (void)dst_heights;
    (void)dst_widths;

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

    /* Carve scratch buffers from the persistent pool (init-time alloc). */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint16_t *vert_buf[4] = { NULL, NULL, NULL, NULL };
    int vert_rows[4];

    for (int k = 0; k <= deepest; k++) {
        vert_rows[k] = group_rows >> (k + 1);
        vert_buf[k] = (uint16_t *)fused_scratch_alloc(
            &scratch, (size_t)vert_rows[k] * (size_t)src_w * sizeof(uint16_t));
        if (!vert_buf[k]) return;
    }

    /* Horizontal cascade buffer */
    uint16_t *h_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)src_w * sizeof(uint16_t));
    if (!h_buf) return;

    int out_row[4] = { 0, 0, 0, 0 };
    int emit_2x_inline = (active_outputs & (1u << bit_pos[0])) != 0;
    int dst_2x_el_stride = emit_2x_inline
        ? dst_strides[0] / (int)sizeof(uint16_t)
        : 0;

    /* AVX2 chunk count: 16 uint16_t elements = 32 bytes per YMM */
    int avx2_chunks = src_w / 16;

    for (int g = 0; g < num_groups; g++) {
        const uint16_t *grp_base = src
            + (size_t)g * (size_t)group_rows * (size_t)src_el_stride;

        /* -- Vertical cascade (AVX2) --------------------------------- */

        /* Level 0 (2x vertical): pairwise average source rows */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint16_t *restrict ra = grp_base
                + (size_t)(2 * r) * (size_t)src_el_stride;
            const uint16_t *restrict rb = grp_base
                + (size_t)(2 * r + 1) * (size_t)src_el_stride;
            uint16_t *restrict dst_row = vert_buf[0]
                + (size_t)r * (size_t)src_w;
            uint16_t *restrict out_2x = emit_2x_inline
                ? dst_planes[0] + (size_t)out_row[0] * (size_t)dst_2x_el_stride
                : NULL;

            int x = 0;
            int out_x = 0;
            int c = 0;
            for (; emit_2x_inline && c + 1 < avx2_chunks; c += 2, x += 32) {
                __m256i va = _mm256_loadu_si256((const __m256i *)(ra + x));
                __m256i vb = _mm256_loadu_si256((const __m256i *)(rb + x));
                __m256i avg0 = avx2_avg_u16(va, vb);

                va = _mm256_loadu_si256((const __m256i *)(ra + x + 16));
                vb = _mm256_loadu_si256((const __m256i *)(rb + x + 16));
                __m256i avg1 = avx2_avg_u16(va, vb);

                _mm256_storeu_si256((__m256i *)(dst_row + x), avg0);
                _mm256_storeu_si256((__m256i *)(dst_row + x + 16), avg1);
                _mm256_storeu_si256((__m256i *)(out_2x + out_x),
                                    avx2_halve_32_to_16_hdr(avg0, avg1));
                out_x += 16;
            }
            for (; c < avx2_chunks; c++, x += 16) {
                __m256i va = _mm256_loadu_si256((const __m256i *)(ra + x));
                __m256i vb = _mm256_loadu_si256((const __m256i *)(rb + x));
                __m256i avg = avx2_avg_u16(va, vb);
                _mm256_storeu_si256((__m256i *)(dst_row + x), avg);
                if (emit_2x_inline) {
                    __m128i half = avx2_halve_16_to_8_hdr(avg);
                    _mm_storeu_si128((__m128i *)(out_2x + out_x), half);
                    out_x += 8;
                }
            }
            for (; x < src_w; x++) {
                dst_row[x] = avg_u16(ra[x], rb[x]);
            }
            if (emit_2x_inline) {
                int tail_in = avx2_chunks * 16;
                for (int tx = tail_in; tx + 1 < src_w; tx += 2) {
                    out_2x[out_x++] = avg_u16(dst_row[tx], dst_row[tx + 1]);
                }
                out_row[0]++;
            }
        }

        /* Deeper levels: pairwise average previous level */
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict ra = vert_buf[k - 1]
                    + (size_t)(2 * r) * (size_t)src_w;
                const uint16_t *restrict rb = vert_buf[k - 1]
                    + (size_t)(2 * r + 1) * (size_t)src_w;
                uint16_t *restrict dst_row = vert_buf[k]
                    + (size_t)r * (size_t)src_w;

                int x = 0;
                for (int c = 0; c < avx2_chunks; c++, x += 16) {
                    __m256i va = _mm256_loadu_si256((const __m256i *)(ra + x));
                    __m256i vb = _mm256_loadu_si256((const __m256i *)(rb + x));
                    _mm256_storeu_si256((__m256i *)(dst_row + x),
                                        avx2_avg_u16(va, vb));
                }
                for (; x < src_w; x++) {
                    dst_row[x] = avg_u16(ra[x], rb[x]);
                }
            }
        }

        /* -- Horizontal cascade (AVX2) + output write ---------------- */

        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;
            if (k == 0 && emit_2x_inline) continue;

            /* Convert destination byte stride to element stride */
            int dst_el_stride = dst_strides[k] / (int)sizeof(uint16_t);

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict vert_row = vert_buf[k]
                    + (size_t)r * (size_t)src_w;

                /* Horizontal cascade: (k+1) halvings */
                int cur_w = src_w;
                const uint16_t *cur_src = vert_row;

                /* Output plane row this reduction level writes to. */
                uint16_t *out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_el_stride;

                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    int next_w = cur_w >> 1;
                    /* Number of whole 16-sample chunks in this row. */
                    int h_chunks = cur_w / 16;
                    int out_x = 0;

                    /* The last halving of the cascade writes straight into the
                     * output plane; earlier halvings write the scratch row
                     * h_buf that feeds the next step.  The vector stores stop
                     * at h_chunks*8 elements, which is at most the destination
                     * width (src_w is divisible by 2^(k+1), so every step's
                     * width stays even), so the row padding past the image edge
                     * is never written.  The unroll keeps enough independent
                     * pairs in flight to hide the multiply latency of the
                     * paired halve.  step_dst is deliberately not marked
                     * restrict: on intermediate steps it is h_buf and overlaps
                     * cur_src, but each load happens before the store that
                     * trails it, so writing in place is safe. */
                    uint16_t *step_dst = (hstep == k) ? out : h_buf;

                    int c = 0;
                    #pragma GCC unroll 4
                    for (; c + 1 < h_chunks; c += 2) {
                        __m256i v0 = _mm256_loadu_si256(
                            (const __m256i *)(cur_src + c * 16));
                        __m256i v1 = _mm256_loadu_si256(
                            (const __m256i *)(cur_src + c * 16 + 16));
                        _mm256_storeu_si256((__m256i *)(step_dst + out_x),
                                            avx2_halve_32_to_16_hdr(v0, v1));
                        out_x += 16;
                    }
                    /* Trailing single chunk for an odd chunk count. */
                    for (; c < h_chunks; c++) {
                        __m256i v = _mm256_loadu_si256(
                            (const __m256i *)(cur_src + c * 16));
                        __m128i result = avx2_halve_16_to_8_hdr(v);
                        _mm_storeu_si128((__m128i *)(step_dst + out_x), result);
                        out_x += 8;
                    }
                    /* Scalar tail for remaining elements */
                    int tail_in = h_chunks * 16;
                    for (int tx = tail_in; tx + 1 < cur_w; tx += 2) {
                        step_dst[out_x++] = avg_u16(cur_src[tx], cur_src[tx + 1]);
                    }

                    cur_w = next_w;
                    cur_src = step_dst;
                }

                out_row[k]++;
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}


/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single 10-bit plane (AVX2 fused vertical+horizontal)
 *
 * Per-chunk fused architecture: source rows are processed in groups of 6
 * (matching the vertical period of the thirds reduction).  For each
 * 96-byte (48-element) column chunk, all 6 rows are loaded, vertical
 * pair averages and bilinear blends are computed entirely in YMM
 * registers, and horizontal filtering is applied immediately - no
 * intermediate row buffer needed.
 *
 * With 16-bit elements, each YMM holds 16 elements, and three YMMs hold
 * 48 elements = 16 triplets.  So 96 bytes = 3 x 32 bytes = 48 elements
 * = 16 triplets per chunk.  This produces:
 *   1.5x: 16 triplets x 2 outputs = 32 output elements (64 bytes)
 *   3x:   16 output elements (32 bytes)
 *   6x:   8 output elements (16 bytes)
 *
 * Vertical: avx2_avg_u16 for pairwise avgs, avx2_blend_2_1_hdr for
 *           bilinear blends (1.5x)
 * Horizontal: deinterleave_3x16_hdr + h_chunk_{1_5x,3x,6x}_hdr_avx2
 * ----------------------------------------------------------------------- */

static void __attribute__((hot)) scale_plane_thirds_hdr_avx2(
    const uint16_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint16_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
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

    /* Carve scratch buffers from the persistent pool (init-time alloc). */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    /* For 12x: two buffers to hold 6x vertical intermediates across
     * consecutive 6-row groups.  Ping-pong swap avoids copying. */
    uint16_t *v6x_buf_a = NULL, *v6x_buf_b = NULL;
    uint16_t *v6x_cur = NULL, *v6x_prev = NULL;
    if (need_12x) {
        v6x_buf_a = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
        v6x_buf_b = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
        if (!v6x_buf_a || !v6x_buf_b) return;
        v6x_cur  = v6x_buf_a;
        v6x_prev = v6x_buf_b;
    }

    /* Horizontal scratch for 12x (3x -> halve -> halve) */
    int w_3x = src_w / 3;
    int w_6x = w_3x / 2;
    uint16_t *h_3x_buf = NULL, *h_6x_buf = NULL;
    if (need_12x && (active_outputs & (1u << 6))) {
        size_t buf_3x_bytes = (size_t)w_3x * sizeof(uint16_t);
        size_t buf_6x_bytes = (size_t)(w_6x > 0 ? w_6x : 1) * sizeof(uint16_t);
        h_3x_buf = (uint16_t *)fused_scratch_alloc(&scratch, buf_3x_bytes);
        h_6x_buf = (uint16_t *)fused_scratch_alloc(&scratch, buf_6x_bytes);
        if (!h_3x_buf || !h_6x_buf) return;
    }

    /* Chunk geometry: 48 source elements per chunk (16 triplets).
     * 48 elements x 2 bytes = 96 bytes = 3 x YMM.
     * LCM(16 elements per YMM, 3 elements per triplet) = 48 elements. */
    int full_chunks = src_w / 48;
    int tail_start  = full_chunks * 48;
    int tail_cols   = src_w - tail_start;

    /* Output row cursors */
    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint16_t *grp = src
            + (size_t)g6 * 6 * (size_t)src_el_stride;

        const uint16_t *restrict row0 = grp;
        const uint16_t *restrict row1 = grp + (size_t)src_el_stride;
        const uint16_t *restrict row2 = grp + (size_t)2 * (size_t)src_el_stride;
        const uint16_t *restrict row3 = grp + (size_t)3 * (size_t)src_el_stride;
        const uint16_t *restrict row4 = grp + (size_t)4 * (size_t)src_el_stride;
        const uint16_t *restrict row5 = grp + (size_t)5 * (size_t)src_el_stride;

        /* Compute output row base pointers */
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
         * MAIN CHUNK LOOP: process 48 source elements (16 triplets)
         * at a time.  Vertical intermediates stay in YMM registers;
         * horizontal filtering is applied immediately per chunk.
         * ============================================================ */
        for (int ci = 0; ci < full_chunks; ci++) {
            int cx = ci * 48;  /* element offset */

            /* Output element offsets per scale factor:
             * 48 elements -> 32 output elements at 1.5x (16 triplets x 2)
             * 48 elements -> 16 output elements at 3x   (16 triplets x 1)
             * 48 elements ->  8 output elements at 6x   (16 triplets / 2) */
            int out_off_1_5x = ci * 32;
            int out_off_3x   = ci * 16;
            int out_off_6x   = ci * 8;

            /* --- LOAD 6 rows x 3 YMM = 18 loads ---
             * Each YMM holds 16 uint16_t elements (32 bytes).
             * Three YMMs per row = 48 elements of that row. */
            __m256i r0a = _mm256_load_si256((const __m256i *)(row0 + cx));
            __m256i r0b = _mm256_load_si256((const __m256i *)(row0 + cx + 16));
            __m256i r0c = _mm256_load_si256((const __m256i *)(row0 + cx + 32));
            __m256i r1a = _mm256_load_si256((const __m256i *)(row1 + cx));
            __m256i r1b = _mm256_load_si256((const __m256i *)(row1 + cx + 16));
            __m256i r1c = _mm256_load_si256((const __m256i *)(row1 + cx + 32));
            __m256i r2a = _mm256_load_si256((const __m256i *)(row2 + cx));
            __m256i r2b = _mm256_load_si256((const __m256i *)(row2 + cx + 16));
            __m256i r2c = _mm256_load_si256((const __m256i *)(row2 + cx + 32));
            __m256i r3a = _mm256_load_si256((const __m256i *)(row3 + cx));
            __m256i r3b = _mm256_load_si256((const __m256i *)(row3 + cx + 16));
            __m256i r3c = _mm256_load_si256((const __m256i *)(row3 + cx + 32));
            __m256i r4a = _mm256_load_si256((const __m256i *)(row4 + cx));
            __m256i r4b = _mm256_load_si256((const __m256i *)(row4 + cx + 16));
            __m256i r4c = _mm256_load_si256((const __m256i *)(row4 + cx + 32));
            __m256i r5a = _mm256_load_si256((const __m256i *)(row5 + cx));
            __m256i r5b = _mm256_load_si256((const __m256i *)(row5 + cx + 16));
            __m256i r5c = _mm256_load_si256((const __m256i *)(row5 + cx + 32));

            /* --- VERTICAL PAIRWISE AVERAGES ---
             * Average adjacent row pairs: rows 0+1 -> v01, rows 2+3 -> v23,
             * rows 4+5 -> v45. */
            __m256i v01a = avx2_avg_u16(r0a, r1a);
            __m256i v01b = avx2_avg_u16(r0b, r1b);
            __m256i v01c = avx2_avg_u16(r0c, r1c);
            __m256i v23a = avx2_avg_u16(r2a, r3a);
            __m256i v23b = avx2_avg_u16(r2b, r3b);
            __m256i v23c = avx2_avg_u16(r2c, r3c);
            __m256i v45a = avx2_avg_u16(r4a, r5a);
            __m256i v45b = avx2_avg_u16(r4b, r5b);
            __m256i v45c = avx2_avg_u16(r4c, r5c);

            /* --- Deinterleave once, reduce in component space ---
             * deinterleave_3x16_hdr is a pure element permutation P, and every
             * vertical reduction here (avx2_avg_u16, avx2_blend_2_1_hdr) is
             * pointwise, call it f; the two commute, so P(f(x,y)) ==
             * f(P(x),P(y)).  When 1.5x is active the group emits four or more
             * output rows that all come from the same three pair averages
             * v01/v23/v45, so this deinterleaves those three once and computes
             * every output level in A/B/C component space - three deinterleaves
             * per chunk.  Deinterleaving is the most expensive (shuffle-port)
             * step here.  When only 3x/6x is active there are at most three such
             * rows, so the else branch deinterleaves each reduced row directly,
             * which issues fewer shuffles.
             *
             * 12x needs the raw interleaved ABCABC 6x row (its h_filter_3x_hdr
             * reads v6x_cur linearly), so the raw 6x row is produced and stored
             * here, independent of the branch below, while the raw pair averages
             * are still live.  avg(avg(v01,v23),avg(v23,v45)) is the 6x reduction
             * of the three pair averages. */
            if (need_12x) {
                __m256i v6xa = avx2_avg_u16(avx2_avg_u16(v01a, v23a),
                                            avx2_avg_u16(v23a, v45a));
                __m256i v6xb = avx2_avg_u16(avx2_avg_u16(v01b, v23b),
                                            avx2_avg_u16(v23b, v45b));
                __m256i v6xc = avx2_avg_u16(avx2_avg_u16(v01c, v23c),
                                            avx2_avg_u16(v23c, v45c));
                _mm256_storeu_si256((__m256i *)(v6x_cur + cx),      v6xa);
                _mm256_storeu_si256((__m256i *)(v6x_cur + cx + 16), v6xb);
                _mm256_storeu_si256((__m256i *)(v6x_cur + cx + 32), v6xc);
            }

            if (need_1_5x) {
                __m256i Av01, Bv01, Cv01, Av23, Bv23, Cv23, Av45, Bv45, Cv45;
                deinterleave_3x16_hdr(v01a, v01b, v01c, &Av01, &Bv01, &Cv01);
                deinterleave_3x16_hdr(v23a, v23b, v23c, &Av23, &Bv23, &Cv23);
                deinterleave_3x16_hdr(v45a, v45b, v45c, &Av45, &Bv45, &Cv45);

                /* 1.5x rows.  Rows 0/3 are v01/v45 directly; rows 1/2 are the
                 * vertical bilinear blends computed component-wise (blend
                 * commutes with the deinterleave). */
                h_chunk_1_5x_hdr_avx2(Av01, Bv01, Cv01, dst_1_5x_r0 + out_off_1_5x);
                {
                    __m256i bA = avx2_blend_2_1_hdr(Av01, Av23);
                    __m256i bB = avx2_blend_2_1_hdr(Bv01, Bv23);
                    __m256i bC = avx2_blend_2_1_hdr(Cv01, Cv23);
                    h_chunk_1_5x_hdr_avx2(bA, bB, bC,
                                          dst_1_5x_r1 + out_off_1_5x);
                }
                {
                    __m256i bA = avx2_blend_2_1_hdr(Av23, Av45);
                    __m256i bB = avx2_blend_2_1_hdr(Bv23, Bv45);
                    __m256i bC = avx2_blend_2_1_hdr(Cv23, Cv45);
                    h_chunk_1_5x_hdr_avx2(bA, bB, bC,
                                          dst_1_5x_r2 + out_off_1_5x);
                }
                h_chunk_1_5x_hdr_avx2(Av45, Bv45, Cv45, dst_1_5x_r3 + out_off_1_5x);

                /* 3x vertical reduction in component space (reused by 6x). */
                __m256i A3x0 = _mm256_setzero_si256();
                __m256i B3x0 = _mm256_setzero_si256();
                __m256i C3x0 = _mm256_setzero_si256();
                __m256i A3x1 = _mm256_setzero_si256();
                __m256i B3x1 = _mm256_setzero_si256();
                __m256i C3x1 = _mm256_setzero_si256();
                if (need_3x) {
                    A3x0 = avx2_avg_u16(Av01, Av23);
                    B3x0 = avx2_avg_u16(Bv01, Bv23);
                    C3x0 = avx2_avg_u16(Cv01, Cv23);
                    A3x1 = avx2_avg_u16(Av23, Av45);
                    B3x1 = avx2_avg_u16(Bv23, Bv45);
                    C3x1 = avx2_avg_u16(Cv23, Cv45);

                    if (active_outputs & (1u << 2)) {
                        h_chunk_3x_hdr_avx2(A3x0, B3x0, C3x0, dst_3x_r0 + out_off_3x);
                        h_chunk_3x_hdr_avx2(A3x1, B3x1, C3x1, dst_3x_r1 + out_off_3x);
                    }
                }

                /* 6x = halve(box3(avg(v3x0,v3x1))) in component space. */
                if (need_6x && (active_outputs & (1u << 4))) {
                    __m256i A6x = avx2_avg_u16(A3x0, A3x1);
                    __m256i B6x = avx2_avg_u16(B3x0, B3x1);
                    __m256i C6x = avx2_avg_u16(C3x0, C3x1);
                    uint16_t __attribute__((aligned(32))) scratch_3x[16];
                    __m256i r3x = h_chunk_3x_hdr_avx2(A6x, B6x, C6x, scratch_3x);
                    h_chunk_6x_hdr_avx2(r3x, dst_6x_r0 + out_off_6x);
                }
            } else {
                /* No 1.5x active: deinterleave each reduced row directly.  With
                 * only 3x/6x there are at most three such rows, so deinterleaving
                 * all three pair averages up front would issue more shuffles
                 * than this does. */
                __m256i v3x0a = _mm256_setzero_si256();
                __m256i v3x0b = _mm256_setzero_si256();
                __m256i v3x0c = _mm256_setzero_si256();
                __m256i v3x1a = _mm256_setzero_si256();
                __m256i v3x1b = _mm256_setzero_si256();
                __m256i v3x1c = _mm256_setzero_si256();
                if (need_3x) {
                    v3x0a = avx2_avg_u16(v01a, v23a);
                    v3x0b = avx2_avg_u16(v01b, v23b);
                    v3x0c = avx2_avg_u16(v01c, v23c);
                    v3x1a = avx2_avg_u16(v23a, v45a);
                    v3x1b = avx2_avg_u16(v23b, v45b);
                    v3x1c = avx2_avg_u16(v23c, v45c);

                    if (active_outputs & (1u << 2)) {
                        __m256i A, B, C;
                        deinterleave_3x16_hdr(v3x0a, v3x0b, v3x0c, &A, &B, &C);
                        h_chunk_3x_hdr_avx2(A, B, C, dst_3x_r0 + out_off_3x);
                        deinterleave_3x16_hdr(v3x1a, v3x1b, v3x1c, &A, &B, &C);
                        h_chunk_3x_hdr_avx2(A, B, C, dst_3x_r1 + out_off_3x);
                    }
                }
                if (need_6x && (active_outputs & (1u << 4))) {
                    __m256i v6xa = avx2_avg_u16(v3x0a, v3x1a);
                    __m256i v6xb = avx2_avg_u16(v3x0b, v3x1b);
                    __m256i v6xc = avx2_avg_u16(v3x0c, v3x1c);
                    __m256i A, B, C;
                    deinterleave_3x16_hdr(v6xa, v6xb, v6xc, &A, &B, &C);
                    uint16_t __attribute__((aligned(32))) scratch_3x[16];
                    __m256i r3x = h_chunk_3x_hdr_avx2(A, B, C, scratch_3x);
                    h_chunk_6x_hdr_avx2(r3x, dst_6x_r0 + out_off_6x);
                }
            }
        } /* end chunk loop */

        /* ============================================================
         * TAIL: handle remaining columns (< 48 elements) with scalar
         * h_filter.  Compute vertical intermediates into stack buffers,
         * then apply scalar horizontal filters.
         *
         * Max tail = 47 elements, so stack buffers of 48 elements suffice.
         * ============================================================ */
        if (tail_cols > 0) {
            uint16_t tail_v01[48], tail_v23[48], tail_v45[48];

            for (int x = 0; x < tail_cols; x++) {
                int sx = tail_start + x;
                tail_v01[x] = avg_u16(row0[sx], row1[sx]);
                tail_v23[x] = avg_u16(row2[sx], row3[sx]);
                tail_v45[x] = avg_u16(row4[sx], row5[sx]);
            }

            uint16_t tail_v3x0[48], tail_v3x1[48];
            if (need_3x) {
                for (int x = 0; x < tail_cols; x++) {
                    tail_v3x0[x] = avg_u16(tail_v01[x], tail_v23[x]);
                    tail_v3x1[x] = avg_u16(tail_v23[x], tail_v45[x]);
                }
            }

            uint16_t tail_v6x[48];
            if (need_6x) {
                for (int x = 0; x < tail_cols; x++) {
                    tail_v6x[x] = avg_u16(tail_v3x0[x], tail_v3x1[x]);
                }
                if (need_12x) {
                    memcpy(v6x_cur + tail_start, tail_v6x,
                           (size_t)tail_cols * sizeof(uint16_t));
                }
            }

            /* How many output elements the AVX2 chunks already produced */
            int tail_out_1_5x = full_chunks * 32;
            int tail_out_3x   = full_chunks * 16;
            int tail_out_6x   = full_chunks * 8;

            /* 1.5x tail */
            if (need_1_5x) {
                int dw_rem = dst_widths[0] - tail_out_1_5x;

                h_filter_1_5x_hdr(tail_v01, tail_cols,
                                  dst_1_5x_r0 + tail_out_1_5x, dw_rem);

                uint16_t tail_blend[48];
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
                uint16_t tail_h3x[16];
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
         * v6x_cur holds this group's 6x vertical intermediate.
         * On even groups: swap pointers (current becomes previous).
         * On odd groups: average prev with current, apply horizontal, output.
         * ============================================================ */
        if (need_12x) {
            if ((g6 & 1) == 0) {
                /* Swap so v6x_cur data becomes v6x_prev for the next group */
                uint16_t *tmp = v6x_prev;
                v6x_prev = v6x_cur;
                v6x_cur  = tmp;
            } else {
                /* Average v6x_prev (even group) with v6x_cur (odd group) */
                int avx2_16 = src_w / 16;
                int x = 0;
                for (int c = 0; c < avx2_16; c++, x += 16) {
                    __m256i a = _mm256_loadu_si256((const __m256i *)(v6x_prev + x));
                    __m256i b = _mm256_loadu_si256((const __m256i *)(v6x_cur + x));
                    _mm256_storeu_si256((__m256i *)(v6x_prev + x),
                                        avx2_avg_u16(a, b));
                }
                for (; x < src_w; x++) {
                    v6x_prev[x] = avg_u16(v6x_prev[x], v6x_cur[x]);
                }

                if (active_outputs & (1u << 6)) {
                    int dw = dst_widths[3];
                    int ds_el = dst_strides[3] / (int)sizeof(uint16_t);

                    uint16_t *restrict out12 = dst_planes[3]
                        + (size_t)out_row[3] * (size_t)ds_el;

                    /* Fused register-resident 12:1 horizontal reduction when
                     * src_w is a multiple of 48 samples (96 bytes = the
                     * deinterleave chunk).  Per chunk: 48 source samples ->
                     * deinterleave_3x16_hdr -> box3_div_hdr (16 3x samples) ->
                     * avx2_halve_16_to_8_hdr (8 6x) -> halve_8_to_4_hdr (4 12x)
                     * -> 64-bit store, with the 3x and 6x stages staying in
                     * registers.  This is chunk-local: 48 = 3*16, 16 = 2*8,
                     * 8 = 2*4, so every triplet and halving-pair boundary sits
                     * inside the chunk and each 12x output is the exact 12:1
                     * average, matching the buffered path in the else branch.
                     * It writes exactly src_w/12 == dst_widths[3] samples
                     * (src_w = 48m gives 4m), so nothing past the row edge is
                     * touched.  Widths that are not a multiple of 48 take the
                     * buffered path. */
                    if (src_w % 48 == 0) {
                        int fc12 = src_w / 48;
                        for (int c = 0; c < fc12; c++) {
                            const uint16_t *cs = v6x_prev + (size_t)c * 48;
                            __m256i ra = _mm256_loadu_si256((const __m256i *)cs);
                            __m256i rb = _mm256_loadu_si256(
                                (const __m256i *)(cs + 16));
                            __m256i rc = _mm256_loadu_si256(
                                (const __m256i *)(cs + 32));
                            __m256i A, B, C;
                            deinterleave_3x16_hdr(ra, rb, rc, &A, &B, &C);
                            __m256i box = box3_div_hdr(A, B, C);
                            __m128i six = avx2_halve_16_to_8_hdr(box);
                            __m128i twl = halve_8_to_4_hdr(six);
                            _mm_storel_epi64(
                                (__m128i *)(out12 + (size_t)c * 4), twl);
                        }
                    } else {
                        h_filter_3x_hdr(v6x_prev, src_w, h_3x_buf, w_3x);
                        h_filter_halve_hdr(h_3x_buf, h_6x_buf, w_6x);
                        h_filter_halve_hdr(h_6x_buf, out12, dw);
                    }
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
 * For I010 (planar): call scale_plane_*_hdr_avx2() three times (Y, U, V).
 * For P010 (semi-planar): Y is still a separate plane; U and V are
 * interleaved in src_u as UVUV... pairs of uint16_t.  We deinterleave
 * the entire chroma plane up front into temporary planar U and V buffers
 * using AVX2, then process U and V identically to I010.
 * ----------------------------------------------------------------------- */

void __attribute__((hot)) fused_kernel_pow2_hdr_avx2(
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
    scale_plane_pow2_hdr_avx2(src_y,
                               p->src_width, p->src_height, p->src_y_stride,
                               p->active_outputs,
                               y_planes, y_widths, y_strides, y_heights,
                               p->scratch_pool, p->scratch_pool_size);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        /* P010: deinterleave UV into pre-allocated aligned planar buffers */
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        static const uint8_t deint_shuf[32] __attribute__((aligned(32))) = {
            0, 1, 4, 5, 8, 9, 12, 13,    2, 3, 6, 7, 10, 11, 14, 15,
            0, 1, 4, 5, 8, 9, 12, 13,    2, 3, 6, 7, 10, 11, 14, 15
        };
        __m256i shuf_mask = _mm256_load_si256((const __m256i *)deint_shuf);

        for (int y = 0; y < chroma_h; y++) {
            const uint16_t *row = src_u + (size_t)y * (size_t)uv_el_stride;
            uint16_t *u_row = p->p010_tmp_u + (size_t)y * (size_t)planar_el_stride;
            uint16_t *v_row = p->p010_tmp_v + (size_t)y * (size_t)planar_el_stride;

            int x = 0;
            int avx2_pairs = chroma_w / 8;
            for (int c = 0; c < avx2_pairs; c++, x += 8) {
                __m256i interleaved = _mm256_loadu_si256(
                    (const __m256i *)(row + c * 16));
                __m256i shuffled = _mm256_shuffle_epi8(interleaved, shuf_mask);
                __m256i permuted = _mm256_permute4x64_epi64(shuffled, 0xD8);
                _mm_storeu_si128((__m128i *)(u_row + x),
                                 _mm256_castsi256_si128(permuted));
                _mm_storeu_si128((__m128i *)(v_row + x),
                                 _mm256_extracti128_si256(permuted, 1));
            }
            for (; x < chroma_w; x++) {
                u_row[x] = row[2 * x];
                v_row[x] = row[2 * x + 1];
            }
        }

        scale_plane_pow2_hdr_avx2(p->p010_tmp_u,
                                   chroma_w, chroma_h, p->p010_tmp_stride,
                                   p->active_outputs,
                                   u_planes, uv_widths, uv_strides, uv_heights,
                                   p->scratch_pool, p->scratch_pool_size);
        scale_plane_pow2_hdr_avx2(p->p010_tmp_v,
                                   chroma_w, chroma_h, p->p010_tmp_stride,
                                   p->active_outputs,
                                   v_planes, uv_widths, uv_strides, uv_heights,
                                   p->scratch_pool, p->scratch_pool_size);
    } else {
        /* I010: separate U and V planes */
        scale_plane_pow2_hdr_avx2(src_u,
                                   chroma_w, chroma_h, p->src_uv_stride,
                                   p->active_outputs,
                                   u_planes, uv_widths, uv_strides, uv_heights,
                                   p->scratch_pool, p->scratch_pool_size);

        scale_plane_pow2_hdr_avx2(src_v,
                                   chroma_w, chroma_h, p->src_uv_stride,
                                   p->active_outputs,
                                   v_planes, uv_widths, uv_strides, uv_heights,
                                   p->scratch_pool, p->scratch_pool_size);
    }

    _mm256_zeroupper();
}


void __attribute__((hot)) fused_kernel_thirds_hdr_avx2(
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
    scale_plane_thirds_hdr_avx2(src_y,
                                 p->src_width, p->src_height, p->src_y_stride,
                                 p->active_outputs,
                                 y_planes, y_widths, y_strides, y_heights,
                                 p->scratch_pool, p->scratch_pool_size);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        /* P010: deinterleave UV into pre-allocated aligned planar buffers */
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        static const uint8_t deint_shuf[32] __attribute__((aligned(32))) = {
            0, 1, 4, 5, 8, 9, 12, 13,    2, 3, 6, 7, 10, 11, 14, 15,
            0, 1, 4, 5, 8, 9, 12, 13,    2, 3, 6, 7, 10, 11, 14, 15
        };
        __m256i shuf_mask = _mm256_load_si256((const __m256i *)deint_shuf);

        for (int y = 0; y < chroma_h; y++) {
            const uint16_t *row = src_u + (size_t)y * (size_t)uv_el_stride;
            uint16_t *u_row = p->p010_tmp_u + (size_t)y * (size_t)planar_el_stride;
            uint16_t *v_row = p->p010_tmp_v + (size_t)y * (size_t)planar_el_stride;

            int x = 0;
            int avx2_pairs = chroma_w / 8;
            for (int c = 0; c < avx2_pairs; c++, x += 8) {
                __m256i interleaved = _mm256_loadu_si256(
                    (const __m256i *)(row + c * 16));
                __m256i shuffled = _mm256_shuffle_epi8(interleaved, shuf_mask);
                __m256i permuted = _mm256_permute4x64_epi64(shuffled, 0xD8);
                _mm_storeu_si128((__m128i *)(u_row + x),
                                 _mm256_castsi256_si128(permuted));
                _mm_storeu_si128((__m128i *)(v_row + x),
                                 _mm256_extracti128_si256(permuted, 1));
            }
            for (; x < chroma_w; x++) {
                u_row[x] = row[2 * x];
                v_row[x] = row[2 * x + 1];
            }
        }

        scale_plane_thirds_hdr_avx2(p->p010_tmp_u,
                                     chroma_w, chroma_h, p->p010_tmp_stride,
                                     p->active_outputs,
                                     u_planes, uv_widths, uv_strides, uv_heights,
                                     p->scratch_pool, p->scratch_pool_size);
        scale_plane_thirds_hdr_avx2(p->p010_tmp_v,
                                     chroma_w, chroma_h, p->p010_tmp_stride,
                                     p->active_outputs,
                                     v_planes, uv_widths, uv_strides, uv_heights,
                                     p->scratch_pool, p->scratch_pool_size);
    } else {
        /* I010: separate U and V planes */
        scale_plane_thirds_hdr_avx2(src_u,
                                     chroma_w, chroma_h, p->src_uv_stride,
                                     p->active_outputs,
                                     u_planes, uv_widths, uv_strides, uv_heights,
                                     p->scratch_pool, p->scratch_pool_size);

        scale_plane_thirds_hdr_avx2(src_v,
                                     chroma_w, chroma_h, p->src_uv_stride,
                                     p->active_outputs,
                                     v_planes, uv_widths, uv_strides, uv_heights,
                                     p->scratch_pool, p->scratch_pool_size);
    }

    _mm256_zeroupper();
}

#endif /* __x86_64__ */
