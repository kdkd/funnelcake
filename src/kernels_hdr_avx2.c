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
 *   - There is NO _mm256_avg_epu16 intrinsic in AVX2, so pairwise
 *     averaging is implemented manually: (a + b + 1) >> 1.
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
 * Matches the manual AVX2 implementation below.  For 10-bit values
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
 * AVX2 provides _mm256_avg_epu8 for 8-bit pairwise averaging but has
 * NO corresponding _mm256_avg_epu16 intrinsic.  We implement the
 * equivalent manually: (a + b + 1) >> 1.
 *
 * For 10-bit values (max 1023), a + b + 1 = max 2047, which fits in
 * uint16_t without overflow, so we can use simple 16-bit add + shift
 * without widening to 32-bit.
 * ----------------------------------------------------------------------- */

static inline __m256i avx2_avg_u16(__m256i a, __m256i b)
{
    __m256i sum = _mm256_add_epi16(a, b);
    __m256i one = _mm256_set1_epi16(1);
    return _mm256_srli_epi16(_mm256_add_epi16(sum, one), 1);
}


/* -----------------------------------------------------------------------
 * AVX2 vertical blend helper for 10-bit
 *
 * Computes (a * 171 + b * 85 + 128) >> 8 across 16 uint16_t elements.
 * Must widen to 32-bit because 1023 * 171 = 174,933 overflows uint16_t.
 *
 * _mm256_unpacklo/hi_epi16 with zero widens 16->32 per lane.
 * _mm256_packus_epi32 narrows 32->16 per lane.  Since both unpack and
 * pack operate within 128-bit lanes, lane affiliation is preserved and
 * the packed result is in correct order.
 * ----------------------------------------------------------------------- */

static inline __m256i avx2_blend_2_1_hdr(__m256i a, __m256i b)
{
    __m256i zero = _mm256_setzero_si256();
    __m256i w171 = _mm256_set1_epi32(171);
    __m256i w85  = _mm256_set1_epi32(85);
    __m256i rnd  = _mm256_set1_epi32(128);

    /* Widen low 4 uint16_t per lane to 4 uint32_t per lane */
    __m256i a_lo = _mm256_unpacklo_epi16(a, zero);
    __m256i a_hi = _mm256_unpackhi_epi16(a, zero);
    __m256i b_lo = _mm256_unpacklo_epi16(b, zero);
    __m256i b_hi = _mm256_unpackhi_epi16(b, zero);

    /* blend = (a * 171 + b * 85 + 128) >> 8 */
    __m256i r_lo = _mm256_srli_epi32(
        _mm256_add_epi32(rnd, _mm256_add_epi32(
            _mm256_mullo_epi32(a_lo, w171), _mm256_mullo_epi32(b_lo, w85))), 8);
    __m256i r_hi = _mm256_srli_epi32(
        _mm256_add_epi32(rnd, _mm256_add_epi32(
            _mm256_mullo_epi32(a_hi, w171), _mm256_mullo_epi32(b_hi, w85))), 8);

    /* Pack 32-bit back to 16-bit unsigned.  _mm256_packus_epi32 packs
     * per-lane: lane 0 gets lo.lane0 | hi.lane0, lane 1 gets lo.lane1 |
     * hi.lane1.  This preserves correct element order because unpacklo/hi
     * also operates per-lane. */
    return _mm256_packus_epi32(r_lo, r_hi);
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

static inline __m256i h_chunk_3x_hdr_avx2(__m256i A, __m256i B, __m256i C,
                                            uint16_t *restrict dst)
{
    __m256i magic = _mm256_set1_epi16((short)0x5556);

    /* Sum: for 10-bit values, A+B+C max = 3069 which fits in uint16_t */
    __m256i sum = _mm256_add_epi16(_mm256_add_epi16(A, B), C);

    /* Divide by 3: (sum * 0x5556) >> 16 via mulhi_epu16 */
    __m256i result = _mm256_mulhi_epu16(sum, magic);

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

            /* Convert destination byte stride to element stride */
            int dst_el_stride = dst_strides[k] / (int)sizeof(uint16_t);

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict vert_row = vert_buf[k]
                    + (size_t)r * (size_t)src_w;

                /* Horizontal cascade: (k+1) halvings */
                int cur_w = src_w;
                const uint16_t *cur_src = vert_row;

                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    int next_w = cur_w >> 1;
                    /* AVX2: 16 elements per YMM, halve to 8 output elements */
                    int h_chunks = cur_w / 16;
                    int out_x = 0;

                    for (int c = 0; c < h_chunks; c++) {
                        __m256i v = _mm256_loadu_si256(
                            (const __m256i *)(cur_src + c * 16));
                        __m128i result = avx2_halve_16_to_8_hdr(v);
                        _mm_storeu_si128((__m128i *)(h_buf + out_x), result);
                        out_x += 8;
                    }
                    /* Scalar tail for remaining elements */
                    int tail_in = h_chunks * 16;
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
             * rows 4+5 -> v45.  Uses manual avg since _mm256_avg_epu16
             * does not exist in AVX2. */
            __m256i v01a = avx2_avg_u16(r0a, r1a);
            __m256i v01b = avx2_avg_u16(r0b, r1b);
            __m256i v01c = avx2_avg_u16(r0c, r1c);
            __m256i v23a = avx2_avg_u16(r2a, r3a);
            __m256i v23b = avx2_avg_u16(r2b, r3b);
            __m256i v23c = avx2_avg_u16(r2c, r3c);
            __m256i v45a = avx2_avg_u16(r4a, r5a);
            __m256i v45b = avx2_avg_u16(r4b, r5b);
            __m256i v45c = avx2_avg_u16(r4c, r5c);

            /* --- 1.5x OUTPUT (4 rows x 32 output elements) ---
             * 6 source rows -> 4 output rows at 1.5x vertical.
             *   Row 0: v01
             *   Row 1: blend_2_1(v01, v23)
             *   Row 2: blend_2_1(v23, v45)  -- note: blend(v23,v45) means
             *          v23 has weight 2/3, v45 has weight 1/3
             *   Row 3: v45
             * Each is then horizontally 1.5x-filtered. */
            if (need_1_5x) {
                __m256i A, B, C;

                /* Row 0: v01 */
                deinterleave_3x16_hdr(v01a, v01b, v01c, &A, &B, &C);
                h_chunk_1_5x_hdr_avx2(A, B, C, dst_1_5x_r0 + out_off_1_5x);

                /* Row 1: blend(v01, v23) */
                {
                    __m256i ba = avx2_blend_2_1_hdr(v01a, v23a);
                    __m256i bb = avx2_blend_2_1_hdr(v01b, v23b);
                    __m256i bc = avx2_blend_2_1_hdr(v01c, v23c);
                    deinterleave_3x16_hdr(ba, bb, bc, &A, &B, &C);
                    h_chunk_1_5x_hdr_avx2(A, B, C, dst_1_5x_r1 + out_off_1_5x);
                }

                /* Row 2: blend(v23, v45) */
                {
                    __m256i ba = avx2_blend_2_1_hdr(v23a, v45a);
                    __m256i bb = avx2_blend_2_1_hdr(v23b, v45b);
                    __m256i bc = avx2_blend_2_1_hdr(v23c, v45c);
                    deinterleave_3x16_hdr(ba, bb, bc, &A, &B, &C);
                    h_chunk_1_5x_hdr_avx2(A, B, C, dst_1_5x_r2 + out_off_1_5x);
                }

                /* Row 3: v45 */
                deinterleave_3x16_hdr(v45a, v45b, v45c, &A, &B, &C);
                h_chunk_1_5x_hdr_avx2(A, B, C, dst_1_5x_r3 + out_off_1_5x);
            }

            /* --- 3x VERTICAL + HORIZONTAL (2 rows x 16 output elements) ---
             * avg(v01,v23) and avg(v23,v45) produce 3:1 vertical reduction. */
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

            /* --- 6x VERTICAL (1 row) + save for 12x + horizontal ---
             * avg(v3x0, v3x1) = avg(avg(v01,v23), avg(v23,v45)). */
            if (need_6x) {
                __m256i v6xa = avx2_avg_u16(v3x0a, v3x1a);
                __m256i v6xb = avx2_avg_u16(v3x0b, v3x1b);
                __m256i v6xc = avx2_avg_u16(v3x0c, v3x1c);

                /* Save 6x vertical intermediate for 12x pairing */
                if (need_12x) {
                    _mm256_storeu_si256((__m256i *)(v6x_cur + cx),      v6xa);
                    _mm256_storeu_si256((__m256i *)(v6x_cur + cx + 16), v6xb);
                    _mm256_storeu_si256((__m256i *)(v6x_cur + cx + 32), v6xc);
                }

                if (active_outputs & (1u << 4)) {
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

                    /* Horizontal: 3x box avg -> halve -> halve = 12:1 */
                    h_filter_3x_hdr(v6x_prev, src_w, h_3x_buf, w_3x);
                    h_filter_halve_hdr(h_3x_buf, h_6x_buf, w_6x);
                    h_filter_halve_hdr(h_6x_buf,
                                       dst_planes[3]
                                           + (size_t)out_row[3] * (size_t)ds_el,
                                       dw);
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
