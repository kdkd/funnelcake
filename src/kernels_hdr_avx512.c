/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_hdr_avx512.c - AVX-512 (x86_64) fused HDR (10-bit) downscale
 * kernels.
 *
 * Entry points mirror kernels_hdr_avx2.c:
 *   fused_kernel_pow2_hdr_avx512   - power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_hdr_avx512 - thirds family (1.5x/3x/6x/12x)
 *
 * HDR is where 512-bit registers earn their keep: with uint16_t samples a
 * YMM holds only 16 elements, which is why the HDR kernels historically
 * cost 2-4x per output byte compared to SDR.  A ZMM holds 32, and the
 * element-shuffling that 16-bit data needs maps onto vpermt2w - a plain
 * AVX512BW instruction with full 64-lane reach - so the deinterleave and
 * interleave steps that cost the AVX2 kernel its shuffle budget collapse
 * the same way the SDR kernel's did under vpermt2b:
 *
 *   - 3-way component deinterleave: two chained vpermt2w per component
 *     (6 ops per 96 elements, vs 18 shuffle-class ops per 48 in AVX2)
 *   - 1.5x output interleave: one vpermt2w per 32 output elements
 *   - pairwise halving: vpermt2w even/odd gather + one vpavgw
 *   - P010 chroma deinterleave: two vpermt2w per 32 chroma columns
 *
 * The arithmetic transplants from AVX2 unchanged: the 171/85 blend stays
 * in the 32-bit madd domain (1023*171 overflows 16 bits), box-of-3 stays
 * in 16 bits (max sum 3069), and the unpack/packus pairs around the blend
 * are per-128-bit-lane symmetric, so results come out linear at any
 * register width.  Tails run the same vector bodies under element masks -
 * no scalar tail exists in this file.
 *
 * COMPILE-TIME SELF-STUBBING: same scheme as kernels_avx512.c.  The
 * Makefile only passes -mavx512* flags when the compiler accepts them;
 * without the flags the __AVX512*__ macros are absent and this file
 * builds the stub branch at the bottom (delegations to AVX2 that dispatch
 * never selects, because fused_avx512_compiled() returns 0).
 *
 * Guarded by __x86_64__ so this file is a no-op on other platforms.
 */

#if defined(__x86_64__)

#include "internal.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512VL__) && defined(__AVX512VBMI__)

#include <immintrin.h>

#define ALIGN64 __attribute__((aligned(64)))

/* -----------------------------------------------------------------------
 * vpermt2w index tables.
 *
 * Deinterleave: component X's lane i must receive source element 3i+X of
 * the 96-element chunk.  Pass 1 covers lanes whose source element lies in
 * the first two vectors (indices 0-63); pass 2 keeps those lanes
 * (identity indices 0-31 select pass 1's result) and fills the rest from
 * the third vector (indices 32-63 select its elements).  Generated and
 * verified against the 3i+X mapping.
 * ----------------------------------------------------------------------- */
static const uint16_t ALIGN64 deintw_A1[32] = {
      0,   3,   6,   9,  12,  15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,
     48,  51,  54,  57,  60,  63,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};
static const uint16_t ALIGN64 deintw_A2[32] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  34,  37,  40,  43,  46,  49,  52,  55,  58,  61
};
static const uint16_t ALIGN64 deintw_B1[32] = {
      1,   4,   7,  10,  13,  16,  19,  22,  25,  28,  31,  34,  37,  40,  43,  46,
     49,  52,  55,  58,  61,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};
static const uint16_t ALIGN64 deintw_B2[32] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  32,  35,  38,  41,  44,  47,  50,  53,  56,  59,  62
};
static const uint16_t ALIGN64 deintw_C1[32] = {
      2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
     50,  53,  56,  59,  62,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};
static const uint16_t ALIGN64 deintw_C2[32] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  33,  36,  39,  42,  45,  48,  51,  54,  57,  60,  63
};

/* Even/odd element gather over a 64-element pair (halving, and P010
 * UV separation - U samples are the even elements, V the odd). */
static const uint16_t ALIGN64 halvew_even[32] = {
      0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30,
     32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62
};
static const uint16_t ALIGN64 halvew_odd[32] = {
      1,   3,   5,   7,   9,  11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
     33,  35,  37,  39,  41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63
};

/* 1.5x output interleave: element 2k from out0[k], 2k+1 from out1[k]. */
static const uint16_t ALIGN64 il15w_lo[32] = {
      0,  32,   1,  33,   2,  34,   3,  35,   4,  36,   5,  37,   6,  38,   7,  39,
      8,  40,   9,  41,  10,  42,  11,  43,  12,  44,  13,  45,  14,  46,  15,  47
};
static const uint16_t ALIGN64 il15w_hi[32] = {
     16,  48,  17,  49,  18,  50,  19,  51,  20,  52,  21,  53,  22,  54,  23,  55,
     24,  56,  25,  57,  26,  58,  27,  59,  28,  60,  29,  61,  30,  62,  31,  63
};

/* n low bits set; n in [0, 32]. */
static inline __mmask32 hdr_mask32(int n)
{
    return (n >= 32) ? (__mmask32)~(uint32_t)0
                     : (__mmask32)(((uint32_t)1 << n) - 1);
}

/* -----------------------------------------------------------------------
 * Halving helpers - rounded pairwise average of adjacent elements,
 * (a+b+1)>>1, identical to vpavgw and the scalar avg_u16.
 * ----------------------------------------------------------------------- */

/* 64 input elements (two vectors) -> 32 output elements.  Two vpermt2w
 * gathers split the pair stream into evens and odds in linear order, one
 * vpavgw averages them - the same three-op shape the SDR halve uses. */
static inline __m512i avx512_halve_64w_to_32w(__m512i v0, __m512i v1)
{
    const __m512i even = _mm512_load_si512((const void *)halvew_even);
    const __m512i odd  = _mm512_load_si512((const void *)halvew_odd);

    return _mm512_avg_epu16(_mm512_permutex2var_epi16(v0, even, v1),
                            _mm512_permutex2var_epi16(v0, odd,  v1));
}

/* 32 input elements -> 16.  vpmaddwd against {1,1} sums adjacent pairs
 * into 32-bit lanes (max 2046, no overflow), the rounded halving is an
 * add+shift, and vpmovdw truncates straight back to 16-bit - exact, since
 * every value fits 10 bits. */
static inline __m256i avx512_halve_32w_to_16w(__m512i v)
{
    __m512i sums = _mm512_madd_epi16(v, _mm512_set1_epi16(1));
    sums = _mm512_srli_epi32(_mm512_add_epi32(sums, _mm512_set1_epi32(1)), 1);
    return _mm512_cvtepi32_epi16(sums);
}

/* 16 input elements -> 8 (256-bit form, for the fused 12x cascade). */
static inline __m128i avx512_halve_16w_to_8w(__m256i v)
{
    __m256i sums = _mm256_madd_epi16(v, _mm256_set1_epi16(1));
    sums = _mm256_srli_epi32(_mm256_add_epi32(sums, _mm256_set1_epi32(1)), 1);
    return _mm256_cvtepi32_epi16(sums);
}

/* -----------------------------------------------------------------------
 * Vertical pairwise row average across one full row of uint16_t.
 * chunks counts whole 32-element vectors; tail_mask covers the remaining
 * 0-31 elements with one masked op.
 * ----------------------------------------------------------------------- */
static inline void avx512_avg_rows_hdr(const uint16_t *restrict ra,
                                       const uint16_t *restrict rb,
                                       uint16_t *restrict dst,
                                       int chunks, __mmask32 tail_mask)
{
    int x = 0;
    for (int c = 0; c < chunks; c++, x += 32) {
        __m512i va = _mm512_loadu_si512((const void *)(ra + x));
        __m512i vb = _mm512_loadu_si512((const void *)(rb + x));
        _mm512_storeu_si512((void *)(dst + x), _mm512_avg_epu16(va, vb));
    }
    if (tail_mask) {
        __m512i va = _mm512_maskz_loadu_epi16(tail_mask, ra + x);
        __m512i vb = _mm512_maskz_loadu_epi16(tail_mask, rb + x);
        _mm512_mask_storeu_epi16(dst + x, tail_mask,
                                 _mm512_avg_epu16(va, vb));
    }
}

/* -----------------------------------------------------------------------
 * Thirds building blocks
 * ----------------------------------------------------------------------- */

/* Separate 96 interleaved elements (za=0-31, zb=32-63, zc=64-95) into the
 * three 32-element component vectors A[i]=src[3i], B[i]=src[3i+1],
 * C[i]=src[3i+2].  Two vpermt2w per component. */
static inline void deinterleave_3x32w(__m512i za, __m512i zb, __m512i zc,
                                      __m512i *out_A, __m512i *out_B, __m512i *out_C)
{
    *out_A = _mm512_permutex2var_epi16(
        _mm512_permutex2var_epi16(za, _mm512_load_si512((const void *)deintw_A1), zb),
        _mm512_load_si512((const void *)deintw_A2), zc);
    *out_B = _mm512_permutex2var_epi16(
        _mm512_permutex2var_epi16(za, _mm512_load_si512((const void *)deintw_B1), zb),
        _mm512_load_si512((const void *)deintw_B2), zc);
    *out_C = _mm512_permutex2var_epi16(
        _mm512_permutex2var_epi16(za, _mm512_load_si512((const void *)deintw_C1), zb),
        _mm512_load_si512((const void *)deintw_C2), zc);
}

/* Box-of-3 average with exact div3.  The sum of three 10-bit values is at
 * most 3069, comfortably inside 16 bits, so this is a plain add-add-mulhi
 * with the 0x5556 magic - exact for sums in [0, 3069]. */
static inline __m512i box3_div_hdr_avx512(__m512i A, __m512i B, __m512i C)
{
    const __m512i magic = _mm512_set1_epi16((short)0x5556);
    return _mm512_mulhi_epu16(_mm512_add_epi16(_mm512_add_epi16(A, B), C),
                              magic);
}

/* (a*171 + b*85 + 128) >> 8 on 32 elements.  10-bit a times 171 overflows
 * 16 bits, so the weighted sum runs in the 32-bit madd domain: interleave
 * (a_i, b_i) pairs, one vpmaddwd against {171, 85} per half, round, shift,
 * pack back.  unpack/packus are per-lane symmetric: linear in, linear out. */
static inline __m512i avx512_blend_2_1_hdr(__m512i a, __m512i b)
{
    const __m512i w   = _mm512_set1_epi32((85 << 16) | 171);
    const __m512i rnd = _mm512_set1_epi32(128);

    __m512i lo = _mm512_srli_epi32(_mm512_add_epi32(
        _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), w), rnd), 8);
    __m512i hi = _mm512_srli_epi32(_mm512_add_epi32(
        _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), w), rnd), 8);

    return _mm512_packus_epi32(lo, hi);
}

/* Horizontal 1.5x on one deinterleaved component triplet: 32 triplets in,
 * 64 interleaved output elements stored at dst.  `full` is literal at
 * every call site (always inlined), so the masked stores only exist in
 * the tail instantiation. */
static inline __attribute__((always_inline)) void h_chunk_1_5x_hdr_avx512(
    __m512i A, __m512i B, __m512i C,
    uint16_t *restrict dst, int full, __mmask32 m_lo, __mmask32 m_hi)
{
    __m512i out0 = avx512_blend_2_1_hdr(A, B);
    __m512i out1 = avx512_blend_2_1_hdr(C, B);

    __m512i ilo = _mm512_permutex2var_epi16(
        out0, _mm512_load_si512((const void *)il15w_lo), out1);
    __m512i ihi = _mm512_permutex2var_epi16(
        out0, _mm512_load_si512((const void *)il15w_hi), out1);

    if (full) {
        _mm512_storeu_si512((void *)dst,        ilo);
        _mm512_storeu_si512((void *)(dst + 32), ihi);
    } else {
        _mm512_mask_storeu_epi16(dst,      m_lo, ilo);
        _mm512_mask_storeu_epi16(dst + 32, m_hi, ihi);
    }
}

/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single 10-bit plane (AVX-512)
 *
 * Same two-phase structure as scale_plane_pow2_hdr_avx2 at twice the
 * element width, with every partial chunk handled by one masked op.
 * ----------------------------------------------------------------------- */
static void __attribute__((hot)) scale_plane_pow2_hdr_avx512(
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

    int src_el_stride = src_stride / (int)sizeof(uint16_t);

    static const int bit_pos[4] = { 1, 3, 5, 7 };

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

    uint16_t *h_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)src_w * sizeof(uint16_t));
    if (!h_buf) return;

    int out_row[4] = { 0, 0, 0, 0 };

    /* Vertical chunk geometry: 32 elements per vector plus a masked tail. */
    int v_chunks = src_w / 32;
    int v_rem    = src_w - v_chunks * 32;
    __mmask32 v_mask = v_rem ? hdr_mask32(v_rem) : 0;

    for (int g = 0; g < num_groups; g++) {
        const uint16_t *grp_base = src
            + (size_t)g * (size_t)group_rows * (size_t)src_el_stride;

        /* -- Vertical cascade --------------------------------------- */

        for (int r = 0; r < vert_rows[0]; r++) {
            avx512_avg_rows_hdr(
                grp_base + (size_t)(2 * r)     * (size_t)src_el_stride,
                grp_base + (size_t)(2 * r + 1) * (size_t)src_el_stride,
                vert_buf[0] + (size_t)r * (size_t)src_w,
                v_chunks, v_mask);
        }

        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                avx512_avg_rows_hdr(
                    vert_buf[k - 1] + (size_t)(2 * r)     * (size_t)src_w,
                    vert_buf[k - 1] + (size_t)(2 * r + 1) * (size_t)src_w,
                    vert_buf[k] + (size_t)r * (size_t)src_w,
                    v_chunks, v_mask);
            }
        }

        /* -- Horizontal cascade + output write ----------------------- */

        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;

            int dst_el_stride = dst_strides[k] / (int)sizeof(uint16_t);

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict vert_row = vert_buf[k]
                    + (size_t)r * (size_t)src_w;

                uint16_t *restrict out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_el_stride;

                /* (k+1) halvings; the final one writes the output plane,
                 * earlier ones write h_buf.  Masked stores never touch
                 * row padding past the image edge. */
                int cur_w = src_w;
                const uint16_t *cur_src = vert_row;

                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    uint16_t *step_dst = (hstep == k) ? out : h_buf;

                    int h_chunks = cur_w / 32;
                    int out_x = 0;

                    int c = 0;
                    #pragma GCC unroll 4
                    for (; c + 1 < h_chunks; c += 2) {
                        __m512i v0 = _mm512_loadu_si512(
                            (const void *)(cur_src + c * 32));
                        __m512i v1 = _mm512_loadu_si512(
                            (const void *)(cur_src + c * 32 + 32));
                        _mm512_storeu_si512((void *)(step_dst + out_x),
                                            avx512_halve_64w_to_32w(v0, v1));
                        out_x += 32;
                    }
                    for (; c < h_chunks; c++) {
                        __m512i v = _mm512_loadu_si512(
                            (const void *)(cur_src + c * 32));
                        _mm256_storeu_si256((__m256i *)(step_dst + out_x),
                                            avx512_halve_32w_to_16w(v));
                        out_x += 16;
                    }

                    /* Masked tail: same pattern as the SDR kernel - the
                     * output mask clips the garbage halves above the valid
                     * region, and an odd final element (no pair partner)
                     * is dropped to match the scalar tails elsewhere. */
                    int rem = cur_w - h_chunks * 32;
                    if (rem > 1) {
                        __mmask32 in_m = hdr_mask32(rem);
                        __mmask16 out_m = (__mmask16)((1u << (rem >> 1)) - 1);
                        __m512i v = _mm512_maskz_loadu_epi16(in_m,
                                        cur_src + (size_t)h_chunks * 32);
                        _mm256_mask_storeu_epi16(step_dst + out_x, out_m,
                                                 avx512_halve_32w_to_16w(v));
                    }

                    cur_w >>= 1;
                    cur_src = step_dst;
                }

                out_row[k]++;
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}

/* -----------------------------------------------------------------------
 * One 96-element column chunk of a 6-row group, fused vertical+horizontal.
 * Element-domain twin of the SDR thirds_chunk_avx512: `full` is literal
 * at both call sites, the tail instantiation masks loads and stores and
 * reuses the identical pipeline (maskz zeros stay above the valid region
 * because every op is positionally elementwise in component space).
 * ----------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void thirds_chunk_hdr_avx512(
    const uint16_t *restrict row0, const uint16_t *restrict row1,
    const uint16_t *restrict row2, const uint16_t *restrict row3,
    const uint16_t *restrict row4, const uint16_t *restrict row5,
    int cx, int cols, int full,
    uint32_t active_outputs,
    int need_1_5x, int need_3x, int need_6x, int need_12x,
    uint16_t *dst_1_5x_r0, uint16_t *dst_1_5x_r1,
    uint16_t *dst_1_5x_r2, uint16_t *dst_1_5x_r3, int out15,
    uint16_t *dst_3x_r0, uint16_t *dst_3x_r1, int out3,
    uint16_t *dst_6x_r0, int out6,
    uint16_t *v6x_dst, int wcomp)
{
    /* Load + store masks (tail instantiation only; folded away when full). */
    __mmask32 mka = 0, mkb = 0, mkc = 0;
    __mmask32 m15lo = 0, m15hi = 0, m3 = 0;
    __mmask16 m6 = 0;
    if (!full) {
        mka = hdr_mask32(cols < 32 ? cols : 32);
        mkb = hdr_mask32(cols < 32 ? 0 : (cols < 64 ? cols - 32 : 32));
        mkc = hdr_mask32(cols < 64 ? 0 : cols - 64);
        int o15 = 2 * wcomp;
        m15lo = hdr_mask32(o15 < 32 ? o15 : 32);
        m15hi = hdr_mask32(o15 < 32 ? 0 : o15 - 32);
        m3 = hdr_mask32(wcomp);
        m6 = (__mmask16)((1u << (wcomp / 2)) - 1);
    }

    __m512i r0a, r0b, r0c, r1a, r1b, r1c, r2a, r2b, r2c;
    __m512i r3a, r3b, r3c, r4a, r4b, r4c, r5a, r5b, r5c;
    if (full) {
        r0a = _mm512_loadu_si512((const void *)(row0 + cx));
        r0b = _mm512_loadu_si512((const void *)(row0 + cx + 32));
        r0c = _mm512_loadu_si512((const void *)(row0 + cx + 64));
        r1a = _mm512_loadu_si512((const void *)(row1 + cx));
        r1b = _mm512_loadu_si512((const void *)(row1 + cx + 32));
        r1c = _mm512_loadu_si512((const void *)(row1 + cx + 64));
        r2a = _mm512_loadu_si512((const void *)(row2 + cx));
        r2b = _mm512_loadu_si512((const void *)(row2 + cx + 32));
        r2c = _mm512_loadu_si512((const void *)(row2 + cx + 64));
        r3a = _mm512_loadu_si512((const void *)(row3 + cx));
        r3b = _mm512_loadu_si512((const void *)(row3 + cx + 32));
        r3c = _mm512_loadu_si512((const void *)(row3 + cx + 64));
        r4a = _mm512_loadu_si512((const void *)(row4 + cx));
        r4b = _mm512_loadu_si512((const void *)(row4 + cx + 32));
        r4c = _mm512_loadu_si512((const void *)(row4 + cx + 64));
        r5a = _mm512_loadu_si512((const void *)(row5 + cx));
        r5b = _mm512_loadu_si512((const void *)(row5 + cx + 32));
        r5c = _mm512_loadu_si512((const void *)(row5 + cx + 64));
    } else {
        r0a = _mm512_maskz_loadu_epi16(mka, row0 + cx);
        r0b = _mm512_maskz_loadu_epi16(mkb, row0 + cx + 32);
        r0c = _mm512_maskz_loadu_epi16(mkc, row0 + cx + 64);
        r1a = _mm512_maskz_loadu_epi16(mka, row1 + cx);
        r1b = _mm512_maskz_loadu_epi16(mkb, row1 + cx + 32);
        r1c = _mm512_maskz_loadu_epi16(mkc, row1 + cx + 64);
        r2a = _mm512_maskz_loadu_epi16(mka, row2 + cx);
        r2b = _mm512_maskz_loadu_epi16(mkb, row2 + cx + 32);
        r2c = _mm512_maskz_loadu_epi16(mkc, row2 + cx + 64);
        r3a = _mm512_maskz_loadu_epi16(mka, row3 + cx);
        r3b = _mm512_maskz_loadu_epi16(mkb, row3 + cx + 32);
        r3c = _mm512_maskz_loadu_epi16(mkc, row3 + cx + 64);
        r4a = _mm512_maskz_loadu_epi16(mka, row4 + cx);
        r4b = _mm512_maskz_loadu_epi16(mkb, row4 + cx + 32);
        r4c = _mm512_maskz_loadu_epi16(mkc, row4 + cx + 64);
        r5a = _mm512_maskz_loadu_epi16(mka, row5 + cx);
        r5b = _mm512_maskz_loadu_epi16(mkb, row5 + cx + 32);
        r5c = _mm512_maskz_loadu_epi16(mkc, row5 + cx + 64);
    }

    /* Vertical pairwise averages. */
    __m512i v01a = _mm512_avg_epu16(r0a, r1a);
    __m512i v01b = _mm512_avg_epu16(r0b, r1b);
    __m512i v01c = _mm512_avg_epu16(r0c, r1c);
    __m512i v23a = _mm512_avg_epu16(r2a, r3a);
    __m512i v23b = _mm512_avg_epu16(r2b, r3b);
    __m512i v23c = _mm512_avg_epu16(r2c, r3c);
    __m512i v45a = _mm512_avg_epu16(r4a, r5a);
    __m512i v45b = _mm512_avg_epu16(r4b, r5b);
    __m512i v45c = _mm512_avg_epu16(r4c, r5c);

    /* Deinterleave the pair averages once; everything below runs in
     * component space (the reductions are pointwise and commute with the
     * permutation).  At 6 vpermt2w per deinterleave a separate no-1.5x
     * code path is not worth maintaining. */
    __m512i Av01, Bv01, Cv01, Av23, Bv23, Cv23, Av45, Bv45, Cv45;
    deinterleave_3x32w(v01a, v01b, v01c, &Av01, &Bv01, &Cv01);
    deinterleave_3x32w(v23a, v23b, v23c, &Av23, &Bv23, &Cv23);
    deinterleave_3x32w(v45a, v45b, v45c, &Av45, &Bv45, &Cv45);

    if (need_1_5x) {
        h_chunk_1_5x_hdr_avx512(Av01, Bv01, Cv01,
                                dst_1_5x_r0 + out15, full, m15lo, m15hi);
        {
            __m512i bA = avx512_blend_2_1_hdr(Av01, Av23);
            __m512i bB = avx512_blend_2_1_hdr(Bv01, Bv23);
            __m512i bC = avx512_blend_2_1_hdr(Cv01, Cv23);
            h_chunk_1_5x_hdr_avx512(bA, bB, bC,
                                    dst_1_5x_r1 + out15, full, m15lo, m15hi);
        }
        {
            __m512i bA = avx512_blend_2_1_hdr(Av23, Av45);
            __m512i bB = avx512_blend_2_1_hdr(Bv23, Bv45);
            __m512i bC = avx512_blend_2_1_hdr(Cv23, Cv45);
            h_chunk_1_5x_hdr_avx512(bA, bB, bC,
                                    dst_1_5x_r2 + out15, full, m15lo, m15hi);
        }
        h_chunk_1_5x_hdr_avx512(Av45, Bv45, Cv45,
                                dst_1_5x_r3 + out15, full, m15lo, m15hi);
    }

    /* 3x vertical reduction in component space (reused by 6x/12x). */
    __m512i A3x0 = _mm512_setzero_si512();
    __m512i B3x0 = _mm512_setzero_si512();
    __m512i C3x0 = _mm512_setzero_si512();
    __m512i A3x1 = _mm512_setzero_si512();
    __m512i B3x1 = _mm512_setzero_si512();
    __m512i C3x1 = _mm512_setzero_si512();
    if (need_3x) {
        A3x0 = _mm512_avg_epu16(Av01, Av23);
        B3x0 = _mm512_avg_epu16(Bv01, Bv23);
        C3x0 = _mm512_avg_epu16(Cv01, Cv23);
        A3x1 = _mm512_avg_epu16(Av23, Av45);
        B3x1 = _mm512_avg_epu16(Bv23, Bv45);
        C3x1 = _mm512_avg_epu16(Cv23, Cv45);

        if (active_outputs & (1u << 2)) {
            __m512i r0 = box3_div_hdr_avx512(A3x0, B3x0, C3x0);
            __m512i r1 = box3_div_hdr_avx512(A3x1, B3x1, C3x1);
            if (full) {
                _mm512_storeu_si512((void *)(dst_3x_r0 + out3), r0);
                _mm512_storeu_si512((void *)(dst_3x_r1 + out3), r1);
            } else {
                _mm512_mask_storeu_epi16(dst_3x_r0 + out3, m3, r0);
                _mm512_mask_storeu_epi16(dst_3x_r1 + out3, m3, r1);
            }
        }
    }

    /* 6x components feed the 6x output row and/or the planar 12x
     * intermediate (need_12x implies need_3x). */
    if ((need_6x && (active_outputs & (1u << 4))) || need_12x) {
        __m512i A6x = _mm512_avg_epu16(A3x0, A3x1);
        __m512i B6x = _mm512_avg_epu16(B3x0, B3x1);
        __m512i C6x = _mm512_avg_epu16(C3x0, C3x1);
        if (need_6x && (active_outputs & (1u << 4))) {
            __m256i six = avx512_halve_32w_to_16w(
                              box3_div_hdr_avx512(A6x, B6x, C6x));
            if (full) {
                _mm256_storeu_si256((__m256i *)(dst_6x_r0 + out6), six);
            } else {
                _mm256_mask_storeu_epi16(dst_6x_r0 + out6, m6, six);
            }
        }
        if (need_12x) {
            if (full) {
                _mm512_storeu_si512((void *)(v6x_dst),      A6x);
                _mm512_storeu_si512((void *)(v6x_dst + 32), B6x);
                _mm512_storeu_si512((void *)(v6x_dst + 64), C6x);
            } else {
                _mm512_mask_storeu_epi16(v6x_dst,             m3, A6x);
                _mm512_mask_storeu_epi16(v6x_dst + wcomp,     m3, B6x);
                _mm512_mask_storeu_epi16(v6x_dst + 2 * wcomp, m3, C6x);
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single 10-bit plane (AVX-512 fused)
 *
 * Same fused 6-row-group architecture as the SDR thirds kernel; the 12x
 * intermediate is always stored component-planar ([A|B|C] per 96-element
 * chunk) so the fused 12x consumer needs no deinterleave of its own.
 * ----------------------------------------------------------------------- */
static void __attribute__((hot)) scale_plane_thirds_hdr_avx512(
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

    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint16_t *v6x_cur = NULL, *v6x_prev = NULL;
    if (need_12x) {
        size_t row_bytes = (size_t)src_w * sizeof(uint16_t);
        v6x_cur  = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
        v6x_prev = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
        if (!v6x_cur || !v6x_prev) return;
    }

    /* Chunk geometry: 96 elements (32 triplets) per chunk; the remainder
     * is a multiple of 6 and runs through the same body under masks. */
    int full_chunks = src_w / 96;
    int tail_start  = full_chunks * 96;
    int tail_cols   = src_w - tail_start;
    int tail_wcomp  = tail_cols / 3;

    /* Pointwise vector geometry for the 12x between-group average. */
    int v_chunks = src_w / 32;
    __mmask32 v_mask = (src_w % 32) ? hdr_mask32(src_w % 32) : 0;

    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint16_t *grp = src + (size_t)g6 * 6 * (size_t)src_el_stride;

        const uint16_t *restrict row0 = grp;
        const uint16_t *restrict row1 = grp + (size_t)src_el_stride;
        const uint16_t *restrict row2 = grp + (size_t)2 * (size_t)src_el_stride;
        const uint16_t *restrict row3 = grp + (size_t)3 * (size_t)src_el_stride;
        const uint16_t *restrict row4 = grp + (size_t)4 * (size_t)src_el_stride;
        const uint16_t *restrict row5 = grp + (size_t)5 * (size_t)src_el_stride;

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

        for (int ci = 0; ci < full_chunks; ci++) {
            thirds_chunk_hdr_avx512(row0, row1, row2, row3, row4, row5,
                ci * 96, 96, /*full=*/1,
                active_outputs, need_1_5x, need_3x, need_6x, need_12x,
                dst_1_5x_r0, dst_1_5x_r1, dst_1_5x_r2, dst_1_5x_r3, ci * 64,
                dst_3x_r0, dst_3x_r1, ci * 32,
                dst_6x_r0, ci * 16,
                v6x_cur ? v6x_cur + (size_t)ci * 96 : NULL, 32);
        }
        if (tail_cols > 0) {
            thirds_chunk_hdr_avx512(row0, row1, row2, row3, row4, row5,
                tail_start, tail_cols, /*full=*/0,
                active_outputs, need_1_5x, need_3x, need_6x, need_12x,
                dst_1_5x_r0, dst_1_5x_r1, dst_1_5x_r2, dst_1_5x_r3,
                full_chunks * 64,
                dst_3x_r0, dst_3x_r1, full_chunks * 32,
                dst_6x_r0, full_chunks * 16,
                v6x_cur ? v6x_cur + (size_t)full_chunks * 96 : NULL,
                tail_wcomp);
        }

        if (need_1_5x) out_row[0] += 4;
        if (active_outputs & (1u << 2)) out_row[1] += 2;
        if (active_outputs & (1u << 4)) out_row[2] += 1;

        if (need_12x) {
            if ((g6 & 1) == 0) {
                uint16_t *tmp = v6x_prev;
                v6x_prev = v6x_cur;
                v6x_cur  = tmp;
            } else {
                /* In-place pointwise average (aliasing is deliberate; each
                 * vector is fully loaded before its store). */
                {
                    int x = 0;
                    for (int c = 0; c < v_chunks; c++, x += 32) {
                        __m512i a = _mm512_loadu_si512((const void *)(v6x_prev + x));
                        __m512i b = _mm512_loadu_si512((const void *)(v6x_cur + x));
                        _mm512_storeu_si512((void *)(v6x_prev + x),
                                            _mm512_avg_epu16(a, b));
                    }
                    if (v_mask) {
                        __m512i a = _mm512_maskz_loadu_epi16(v_mask, v6x_prev + x);
                        __m512i b = _mm512_maskz_loadu_epi16(v_mask, v6x_cur + x);
                        _mm512_mask_storeu_epi16(v6x_prev + x, v_mask,
                                                 _mm512_avg_epu16(a, b));
                    }
                }

                if (active_outputs & (1u << 6)) {
                    int ds_el = dst_strides[3] / (int)sizeof(uint16_t);
                    uint16_t *restrict out12 = dst_planes[3]
                        + (size_t)out_row[3] * (size_t)ds_el;

                    /* 96 planar component elements -> 8 output elements:
                     * box-of-3, halve, halve, all in registers. */
                    for (int c = 0; c < full_chunks; c++) {
                        const uint16_t *cs = v6x_prev + (size_t)c * 96;
                        __m512i A = _mm512_loadu_si512((const void *)cs);
                        __m512i B = _mm512_loadu_si512((const void *)(cs + 32));
                        __m512i C = _mm512_loadu_si512((const void *)(cs + 64));
                        __m256i six = avx512_halve_32w_to_16w(
                                          box3_div_hdr_avx512(A, B, C));
                        _mm_storeu_si128((__m128i *)(out12 + (size_t)c * 8),
                                         avx512_halve_16w_to_8w(six));
                    }
                    if (tail_cols > 0) {
                        const uint16_t *cs = v6x_prev + (size_t)tail_start;
                        __mmask32 mw = hdr_mask32(tail_wcomp);
                        __m512i A = _mm512_maskz_loadu_epi16(mw, cs);
                        __m512i B = _mm512_maskz_loadu_epi16(mw, cs + tail_wcomp);
                        __m512i C = _mm512_maskz_loadu_epi16(mw, cs + 2 * tail_wcomp);
                        __m256i six = avx512_halve_32w_to_16w(
                                          box3_div_hdr_avx512(A, B, C));
                        __mmask8 m12 = (__mmask8)((1u << (tail_wcomp / 4)) - 1);
                        _mm_mask_storeu_epi16(out12 + (size_t)full_chunks * 8,
                                              m12, avx512_halve_16w_to_8w(six));
                    }
                    out_row[3]++;
                }
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}

/* -----------------------------------------------------------------------
 * P010 chroma deinterleave
 *
 * Interleaved UVUV... pairs are just an even/odd element split, which is
 * exactly what the halving gather tables describe: two vpermt2w produce
 * 32 U and 32 V samples from 64 interleaved elements.  Tails are masked.
 * ----------------------------------------------------------------------- */
static void deinterleave_p010_chroma_avx512(
    const uint16_t *src_uv,
    int chroma_w, int chroma_h, int uv_el_stride,
    uint16_t *tmp_u, uint16_t *tmp_v, int planar_el_stride)
{
    const __m512i even = _mm512_load_si512((const void *)halvew_even);
    const __m512i odd  = _mm512_load_si512((const void *)halvew_odd);

    int blocks = chroma_w / 32;
    int rem    = chroma_w - blocks * 32;     /* leftover UV pairs */
    __mmask32 in_m0 = 0, in_m1 = 0, out_m = 0;
    if (rem) {
        int rem_el = 2 * rem;
        in_m0 = hdr_mask32(rem_el < 32 ? rem_el : 32);
        in_m1 = hdr_mask32(rem_el < 32 ? 0 : rem_el - 32);
        out_m = hdr_mask32(rem);
    }

    for (int y = 0; y < chroma_h; y++) {
        const uint16_t *row = src_uv + (size_t)y * (size_t)uv_el_stride;
        uint16_t *u_row = tmp_u + (size_t)y * (size_t)planar_el_stride;
        uint16_t *v_row = tmp_v + (size_t)y * (size_t)planar_el_stride;

        int x = 0;
        for (int b = 0; b < blocks; b++, x += 32) {
            __m512i uv0 = _mm512_loadu_si512((const void *)(row + 2 * x));
            __m512i uv1 = _mm512_loadu_si512((const void *)(row + 2 * x + 32));
            _mm512_storeu_si512((void *)(u_row + x),
                                _mm512_permutex2var_epi16(uv0, even, uv1));
            _mm512_storeu_si512((void *)(v_row + x),
                                _mm512_permutex2var_epi16(uv0, odd, uv1));
        }
        if (rem) {
            __m512i uv0 = _mm512_maskz_loadu_epi16(in_m0, row + 2 * x);
            __m512i uv1 = _mm512_maskz_loadu_epi16(in_m1, row + 2 * x + 32);
            _mm512_mask_storeu_epi16(u_row + x, out_m,
                                     _mm512_permutex2var_epi16(uv0, even, uv1));
            _mm512_mask_storeu_epi16(v_row + x, out_m,
                                     _mm512_permutex2var_epi16(uv0, odd, uv1));
        }
    }
}

/* -----------------------------------------------------------------------
 * Public entry points
 *
 * For I010 (planar): scale Y, U, V separately.  For P010 (semi-planar):
 * deinterleave the chroma plane into the pre-allocated planar temps,
 * then proceed as I010.
 * ----------------------------------------------------------------------- */

void __attribute__((hot)) fused_kernel_pow2_hdr_avx512(
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
    scale_plane_pow2_hdr_avx512(src_y,
                                p->src_width, p->src_height, p->src_y_stride,
                                p->active_outputs,
                                y_planes, y_widths, y_strides, y_heights,
                                p->scratch_pool, p->scratch_pool_size);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        deinterleave_p010_chroma_avx512(src_u, chroma_w, chroma_h,
                                        uv_el_stride, p->p010_tmp_u,
                                        p->p010_tmp_v, planar_el_stride);

        scale_plane_pow2_hdr_avx512(p->p010_tmp_u,
                                    chroma_w, chroma_h, p->p010_tmp_stride,
                                    p->active_outputs,
                                    u_planes, uv_widths, uv_strides, uv_heights,
                                    p->scratch_pool, p->scratch_pool_size);
        scale_plane_pow2_hdr_avx512(p->p010_tmp_v,
                                    chroma_w, chroma_h, p->p010_tmp_stride,
                                    p->active_outputs,
                                    v_planes, uv_widths, uv_strides, uv_heights,
                                    p->scratch_pool, p->scratch_pool_size);
    } else {
        scale_plane_pow2_hdr_avx512(src_u,
                                    chroma_w, chroma_h, p->src_uv_stride,
                                    p->active_outputs,
                                    u_planes, uv_widths, uv_strides, uv_heights,
                                    p->scratch_pool, p->scratch_pool_size);

        scale_plane_pow2_hdr_avx512(src_v,
                                    chroma_w, chroma_h, p->src_uv_stride,
                                    p->active_outputs,
                                    v_planes, uv_widths, uv_strides, uv_heights,
                                    p->scratch_pool, p->scratch_pool_size);
    }

    _mm256_zeroupper();
}

void __attribute__((hot)) fused_kernel_thirds_hdr_avx512(
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
    scale_plane_thirds_hdr_avx512(src_y,
                                  p->src_width, p->src_height, p->src_y_stride,
                                  p->active_outputs,
                                  y_planes, y_widths, y_strides, y_heights,
                                  p->scratch_pool, p->scratch_pool_size);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        deinterleave_p010_chroma_avx512(src_u, chroma_w, chroma_h,
                                        uv_el_stride, p->p010_tmp_u,
                                        p->p010_tmp_v, planar_el_stride);

        scale_plane_thirds_hdr_avx512(p->p010_tmp_u,
                                      chroma_w, chroma_h, p->p010_tmp_stride,
                                      p->active_outputs,
                                      u_planes, uv_widths, uv_strides, uv_heights,
                                      p->scratch_pool, p->scratch_pool_size);
        scale_plane_thirds_hdr_avx512(p->p010_tmp_v,
                                      chroma_w, chroma_h, p->p010_tmp_stride,
                                      p->active_outputs,
                                      v_planes, uv_widths, uv_strides, uv_heights,
                                      p->scratch_pool, p->scratch_pool_size);
    } else {
        scale_plane_thirds_hdr_avx512(src_u,
                                      chroma_w, chroma_h, p->src_uv_stride,
                                      p->active_outputs,
                                      u_planes, uv_widths, uv_strides, uv_heights,
                                      p->scratch_pool, p->scratch_pool_size);

        scale_plane_thirds_hdr_avx512(src_v,
                                      chroma_w, chroma_h, p->src_uv_stride,
                                      p->active_outputs,
                                      v_planes, uv_widths, uv_strides, uv_heights,
                                      p->scratch_pool, p->scratch_pool_size);
    }

    _mm256_zeroupper();
}

#else /* compiler lacks AVX-512 support - self-stubbed build */

/* Stub build: no -mavx512* flags, no AVX-512 instructions in this object.
 * Dispatch never selects these (fused_avx512_compiled() returns 0); the
 * delegations exist to satisfy the linker and stay correct regardless. */

void fused_kernel_pow2_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src_y,
                                  const uint16_t *src_u,
                                  const uint16_t *src_v)
{
    fused_kernel_pow2_hdr_avx2(p, src_y, src_u, src_v);
}

void fused_kernel_thirds_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                    const uint16_t *src_y,
                                    const uint16_t *src_u,
                                    const uint16_t *src_v)
{
    fused_kernel_thirds_hdr_avx2(p, src_y, src_u, src_v);
}

#endif /* AVX-512 feature macros */

#endif /* __x86_64__ */
