/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_avx512.c - AVX-512 (x86_64) fused downscale kernels.
 *
 * Two entry points, mirroring kernels_avx2.c:
 *   fused_kernel_pow2_avx512   - power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_avx512 - thirds family (1.5x/3x/6x/12x)
 *
 * These kernels target the F+BW+VL+VBMI feature set (Ice Lake and newer
 * Intel, Zen 4 and newer AMD).  Runtime dispatch (funnelcake.c) selects
 * them only when fused_detect_cpu() reports that full set, so Skylake-SP
 * class machines - AVX-512 without VBMI, plus heavy 512-bit downclocking -
 * stay on the AVX2 kernels where they are faster anyway.
 *
 * Entry points that do not yet have a 512-bit implementation delegate to
 * their AVX2 counterparts.  That delegation is the intended architecture,
 * not a placeholder hack: where 512-bit measurably does not pay (or has
 * not yet been measured to pay), the AVX2 kernel IS the best
 * implementation, and the delegating call costs one tail-call.
 *
 * COMPILE-TIME SELF-STUBBING: this file is always part of the x86_64
 * build, but the Makefile only passes the -mavx512* flags when a
 * try-compile probe shows the compiler accepts them.  The __AVX512*__
 * macros below are defined by the compiler if and only if those flags
 * were passed, so on an older compiler this file quietly builds the
 * stub branch at the bottom: fused_avx512_compiled() returns 0 (which
 * keeps dispatch off this path entirely) and the entry points delegate
 * to AVX2 just in case.  No compiler version checks anywhere - the
 * macros answer the only question that matters: "was this translation
 * unit actually built with AVX-512 enabled?"
 *
 * Guarded by __x86_64__ so this file is a no-op on other platforms.
 */

#if defined(__x86_64__)

#include "internal.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512VL__) && defined(__AVX512VBMI__)

#include <immintrin.h>
#include <string.h>

/* Real AVX-512 build: dispatch may select this kernel set. */
int fused_avx512_compiled(void)
{
    return 1;
}

/* -----------------------------------------------------------------------
 * Horizontal halving helpers
 *
 * Pairwise average: out[i] = (in[2i] + in[2i+1] + 1) >> 1, the same
 * rounded average as vpavgb and the scalar avg_u8 in kernels_avx2.c, so
 * the output is bit-identical to the AVX2 and scalar paths.
 *
 * The trick carried over from AVX2: maddubs against {1,1} sums each
 * adjacent byte pair exactly (max 510, no saturation) on the multiply
 * port, and vpavgw against zero performs the rounded halving.  What
 * AVX-512 adds is the narrowing: vpmovwb truncates 16-bit lanes straight
 * to bytes across the full register - no per-lane packus dance - and
 * since every halved value fits in [0,255] the truncation is exact.
 * ----------------------------------------------------------------------- */

/* 64 input bytes -> 32 output bytes.  Three instructions of real work
 * (maddubs, avgw, vpmovwb), and the result comes out already in linear
 * order - the lane-order repair that costs AVX2 a vpermq simply never
 * has to happen. */
static inline __m256i avx512_halve_64_to_32(__m512i v)
{
    const __m512i ones8 = _mm512_set1_epi8(1);
    __m512i halves = _mm512_avg_epu16(_mm512_maddubs_epi16(v, ones8),
                                      _mm512_setzero_si512());
    return _mm512_cvtepi16_epi8(halves);
}

/* VBMI byte-permute index tables for the paired halve below.  vpermt2b
 * treats index values 0-63 as bytes of its first source and 64-127 as
 * bytes of its second, so these two tables pull all even-indexed and all
 * odd-indexed bytes of a 128-byte pair into linear order. */
#define ALIGN64 __attribute__((aligned(64)))
static const uint8_t ALIGN64 halve_even_idx[64] = {
      0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,  28,  30,
     32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62,
     64,  66,  68,  70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,
     96,  98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126
};
static const uint8_t ALIGN64 halve_odd_idx[64] = {
      1,   3,   5,   7,   9,  11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
     33,  35,  37,  39,  41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63,
     65,  67,  69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89,  91,  93,  95,
     97,  99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127
};

/* 128 input bytes -> 64 output bytes (paired form, one full-width store).
 *
 * This is the pairwise average phrased the way VBMI wants it: two
 * vpermt2b gathers split the 128-byte pair into its even and odd bytes -
 * already in linear order, no per-lane pack repair needed - and a single
 * vpavgb does the rounded average.  Three ops where the maddubs+avgw+
 * packus+vpermq formulation needs six; benchmarked identical on Zen 5
 * (the loop is memory-bound either way) but this form leaves the
 * multiply port untouched and keeps the dependency chain at two links,
 * headroom that future fused loops can spend. */
static inline __m512i avx512_halve_128_to_64(__m512i v0, __m512i v1)
{
    const __m512i even = _mm512_load_si512((const void *)halve_even_idx);
    const __m512i odd  = _mm512_load_si512((const void *)halve_odd_idx);

    return _mm512_avg_epu8(_mm512_permutex2var_epi8(v0, even, v1),
                           _mm512_permutex2var_epi8(v0, odd,  v1));
}

/* 32 -> 16 and 16 -> 8 byte forms complete the in-register pow2 cascade. */
static inline __m128i avx512_halve_32_to_16(__m256i v)
{
    __m256i halves = _mm256_avg_epu16(
        _mm256_maddubs_epi16(v, _mm256_set1_epi8(1)),
        _mm256_setzero_si256());
    return _mm256_cvtepi16_epi8(halves);
}

static inline __m128i avx512_halve_16_to_8(__m128i v)
{
    __m128i halves = _mm_avg_epu16(
        _mm_maddubs_epi16(v, _mm_set1_epi8(1)),
        _mm_setzero_si128());
    return _mm_packus_epi16(halves, halves);
}

/* Finish a horizontal pow2 cascade while the freshly computed vertical
 * averages are still in registers.  The helpers match the 128-byte paired
 * loop and its 64-byte residual respectively. */
static inline void store_pow2_128_avx512(__m512i v0, __m512i v1, int level,
                                         uint8_t *restrict dst)
{
    __m512i h1 = avx512_halve_128_to_64(v0, v1);
    if (level == 0) {
        _mm512_storeu_si512((void *)dst, h1);
        return;
    }

    __m256i h2 = avx512_halve_64_to_32(h1);
    if (level == 1) {
        _mm256_storeu_si256((__m256i *)dst, h2);
        return;
    }

    __m128i h3 = avx512_halve_32_to_16(h2);
    if (level == 2) {
        _mm_storeu_si128((__m128i *)dst, h3);
        return;
    }

    _mm_storel_epi64((__m128i *)dst, avx512_halve_16_to_8(h3));
}

static inline void store_pow2_64_avx512(__m512i v, int level,
                                        uint8_t *restrict dst)
{
    __m256i h1 = avx512_halve_64_to_32(v);
    if (level == 0) {
        _mm256_storeu_si256((__m256i *)dst, h1);
        return;
    }

    __m128i h2 = avx512_halve_32_to_16(h1);
    if (level == 1) {
        _mm_storeu_si128((__m128i *)dst, h2);
        return;
    }

    __m128i h3 = avx512_halve_16_to_8(h2);
    if (level == 2) {
        _mm_storel_epi64((__m128i *)dst, h3);
        return;
    }

    uint32_t packed = (uint32_t)_mm_cvtsi128_si32(
        avx512_halve_16_to_8(h3));
    memcpy(dst, &packed, sizeof(packed));
}

static inline uint8_t reduce_pow2_scalar_avx512(const uint8_t *src, int level)
{
    uint8_t values[16];
    int count = 2 << level;
    memcpy(values, src, (size_t)count);
    while (count > 1) {
        for (int i = 0; i < count / 2; i++)
            values[i] = (uint8_t)(((unsigned)values[2 * i]
                                 + (unsigned)values[2 * i + 1] + 1) >> 1);
        count >>= 1;
    }
    return values[0];
}

/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single plane (AVX-512)
 *
 * Each vertical pass emits its requested horizontal reduction while the
 * averaged vectors are still live.  Only rows consumed by a deeper vertical
 * level reach scratch, eliminating completed-row reloads and the former
 * in-place horizontal cascade.  Full chunks consume 128 source bytes at a
 * time; a 64-byte residual stays vectorized and uncommon shorter tails use
 * the bit-identical scalar reduction.
 * ----------------------------------------------------------------------- */
static void __attribute__((hot)) scale_plane_pow2_avx512(
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

        /* Level 0: emit the 2x horizontal result from the live vertical
         * averages, retaining the full-width rows only when a deeper level
         * consumes them. */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint8_t *restrict ra = grp_base
                + (size_t)(2 * r) * (size_t)src_stride;
            const uint8_t *restrict rb = grp_base
                + (size_t)(2 * r + 1) * (size_t)src_stride;
            uint8_t *restrict dst_row = vert_buf[0] + (size_t)r * (size_t)src_w;
            int emit = (active_outputs & (1u << bit_pos[0])) != 0;
            int keep = deepest > 0;
            uint8_t *restrict out = emit ? dst_planes[0]
                + (size_t)out_row[0] * (size_t)dst_strides[0] : NULL;

            int x = 0;
            int out_x = 0;
            for (; x + 128 <= src_w; x += 128, out_x += 64) {
                __m512i v0 = _mm512_avg_epu8(
                    _mm512_loadu_si512((const void *)(ra + x)),
                    _mm512_loadu_si512((const void *)(rb + x)));
                __m512i v1 = _mm512_avg_epu8(
                    _mm512_loadu_si512((const void *)(ra + x + 64)),
                    _mm512_loadu_si512((const void *)(rb + x + 64)));
                if (keep) {
                    _mm512_storeu_si512((void *)(dst_row + x), v0);
                    _mm512_storeu_si512((void *)(dst_row + x + 64), v1);
                }
                if (emit) store_pow2_128_avx512(v0, v1, 0, out + out_x);
            }
            if (x + 64 <= src_w) {
                __m512i v = _mm512_avg_epu8(
                    _mm512_loadu_si512((const void *)(ra + x)),
                    _mm512_loadu_si512((const void *)(rb + x)));
                if (keep) _mm512_storeu_si512((void *)(dst_row + x), v);
                if (emit) store_pow2_64_avx512(v, 0, out + out_x);
                x += 64;
                out_x += 32;
            }
            int tail_start = x;
            for (; x < src_w; x++)
                dst_row[x] = (uint8_t)(((unsigned)ra[x] + rb[x] + 1) >> 1);
            if (emit) {
                for (int sx = tail_start; sx < src_w; sx += 2)
                    out[out_x++] = reduce_pow2_scalar_avx512(dst_row + sx, 0);
                out_row[0]++;
            }
        }

        /* Deeper vertical passes keep the low-pressure row-by-row shape,
         * but each pass now finishes its horizontal output before spilling
         * the next row.  This removes every completed-row reload and the
         * entire in-place horizontal scratch cascade. */
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint8_t *restrict ra = vert_buf[k - 1]
                    + (size_t)(2 * r) * (size_t)src_w;
                const uint8_t *restrict rb = vert_buf[k - 1]
                    + (size_t)(2 * r + 1) * (size_t)src_w;
                uint8_t *restrict dst_row = vert_buf[k] + (size_t)r * (size_t)src_w;
                int emit = (active_outputs & (1u << bit_pos[k])) != 0;
                int keep = k < deepest;
                uint8_t *restrict out = emit ? dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_strides[k] : NULL;

                int x = 0;
                int out_x = 0;
                int out_per_128 = 64 >> k;
                for (; x + 128 <= src_w; x += 128, out_x += out_per_128) {
                    __m512i v0 = _mm512_avg_epu8(
                        _mm512_loadu_si512((const void *)(ra + x)),
                        _mm512_loadu_si512((const void *)(rb + x)));
                    __m512i v1 = _mm512_avg_epu8(
                        _mm512_loadu_si512((const void *)(ra + x + 64)),
                        _mm512_loadu_si512((const void *)(rb + x + 64)));
                    if (keep) {
                        _mm512_storeu_si512((void *)(dst_row + x), v0);
                        _mm512_storeu_si512((void *)(dst_row + x + 64), v1);
                    }
                    if (emit) store_pow2_128_avx512(v0, v1, k, out + out_x);
                }
                if (x + 64 <= src_w) {
                    __m512i v = _mm512_avg_epu8(
                        _mm512_loadu_si512((const void *)(ra + x)),
                        _mm512_loadu_si512((const void *)(rb + x)));
                    if (keep) _mm512_storeu_si512((void *)(dst_row + x), v);
                    if (emit) store_pow2_64_avx512(v, k, out + out_x);
                    x += 64;
                    out_x += 32 >> k;
                }
                int tail_start = x;
                for (; x < src_w; x++)
                    dst_row[x] = (uint8_t)(((unsigned)ra[x] + rb[x] + 1) >> 1);
                if (emit) {
                    int step = 2 << k;
                    for (int sx = tail_start; sx < src_w; sx += step)
                        out[out_x++] = reduce_pow2_scalar_avx512(dst_row + sx, k);
                    out_row[k]++;
                }
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}


/* =========================================================================
 * Thirds family (1.5x/3x/6x/12x) - the VBMI showcase.
 *
 * Same fused architecture as scale_plane_thirds_avx2: source rows in
 * groups of 6, vertical intermediates held in registers, horizontal
 * filtering applied per column chunk.  The chunk is 192 source bytes
 * (LCM of 64 and 3), twice the AVX2 chunk.
 *
 * What changes at 512-bit is the price of moving bytes.  The AVX2 kernel's
 * binding cost is its 3-way deinterleave: vpshufb cannot cross 128-bit
 * lanes, so separating ABCABC... into A/B/C costs 3 vperm2i128 + 9 vpshufb
 * + 6 vpor per 96 bytes - 36 shuffle-class ops per 192 bytes.  vpermt2b
 * crosses all 64 lanes and reads from a 128-byte source pair, so here the
 * same separation is two chained vpermt2b per component: 6 ops per 192
 * bytes, a sixth of the traffic on the port that used to be the
 * bottleneck.  The 1.5x output interleave collapses the same way (two
 * vpermt2b instead of two unpacks plus two vperm2i128).
 *
 * The arithmetic transplants unchanged from AVX2: unpacklo/hi and packus
 * are still per-128-bit-lane and those lanes quadruple, but the pairs are
 * symmetric (unpack low 8 bytes of each block -> pack them back to the
 * same 8 bytes), so values never leave their block and the results come
 * out linear with no lane repair.
 *
 * Tails run the SAME vector body under masks: the effective width is
 * always a multiple of 6, the remainder after whole 192-byte chunks is
 * loaded with maskz (zeros above), every op is positionally elementwise
 * in component space, and output masks clip the garbage lanes.  No scalar
 * tail exists in this kernel.
 * ========================================================================= */

/* vpermt2b index tables for the 3-source deinterleave.  Component X's
 * lane i must receive source byte 3i+X of the 192-byte chunk.  Pass 1
 * (deint_X1) covers lanes whose source byte falls in the first two
 * vectors (indices 0-127); pass 2 (deint_X2) keeps those lanes (identity
 * indices 0-63 select pass 1's result) and fills the rest from the third
 * vector (indices 64-127 select its bytes).  Generated and verified
 * against the 3i+X mapping. */
static const uint8_t ALIGN64 deint_A1[64] = {
      0,   3,   6,   9,  12,  15,  18,  21,  24,  27,  30,  33,  36,  39,  42,  45,
     48,  51,  54,  57,  60,  63,  66,  69,  72,  75,  78,  81,  84,  87,  90,  93,
     96,  99, 102, 105, 108, 111, 114, 117, 120, 123, 126,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};
static const uint8_t ALIGN64 deint_A2[64] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  65,  68,  71,  74,  77,
     80,  83,  86,  89,  92,  95,  98, 101, 104, 107, 110, 113, 116, 119, 122, 125
};
static const uint8_t ALIGN64 deint_B1[64] = {
      1,   4,   7,  10,  13,  16,  19,  22,  25,  28,  31,  34,  37,  40,  43,  46,
     49,  52,  55,  58,  61,  64,  67,  70,  73,  76,  79,  82,  85,  88,  91,  94,
     97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};
static const uint8_t ALIGN64 deint_B2[64] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  66,  69,  72,  75,  78,
     81,  84,  87,  90,  93,  96,  99, 102, 105, 108, 111, 114, 117, 120, 123, 126
};
static const uint8_t ALIGN64 deint_C1[64] = {
      2,   5,   8,  11,  14,  17,  20,  23,  26,  29,  32,  35,  38,  41,  44,  47,
     50,  53,  56,  59,  62,  65,  68,  71,  74,  77,  80,  83,  86,  89,  92,  95,
     98, 101, 104, 107, 110, 113, 116, 119, 122, 125,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};
static const uint8_t ALIGN64 deint_C2[64] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  64,  67,  70,  73,  76,  79,
     82,  85,  88,  91,  94,  97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127
};

/* 1.5x output interleave: byte 2k from out0[k], byte 2k+1 from out1[k]. */
static const uint8_t ALIGN64 il15_lo[64] = {
      0,  64,   1,  65,   2,  66,   3,  67,   4,  68,   5,  69,   6,  70,   7,  71,
      8,  72,   9,  73,  10,  74,  11,  75,  12,  76,  13,  77,  14,  78,  15,  79,
     16,  80,  17,  81,  18,  82,  19,  83,  20,  84,  21,  85,  22,  86,  23,  87,
     24,  88,  25,  89,  26,  90,  27,  91,  28,  92,  29,  93,  30,  94,  31,  95
};
static const uint8_t ALIGN64 il15_hi[64] = {
     32,  96,  33,  97,  34,  98,  35,  99,  36, 100,  37, 101,  38, 102,  39, 103,
     40, 104,  41, 105,  42, 106,  43, 107,  44, 108,  45, 109,  46, 110,  47, 111,
     48, 112,  49, 113,  50, 114,  51, 115,  52, 116,  53, 117,  54, 118,  55, 119,
     56, 120,  57, 121,  58, 122,  59, 123,  60, 124,  61, 125,  62, 126,  63, 127
};

/* Separate 192 interleaved bytes (za=0-63, zb=64-127, zc=128-191) into the
 * three 64-byte component vectors A[i]=src[3i], B[i]=src[3i+1],
 * C[i]=src[3i+2].  Two vpermt2b per component; that is the whole trick. */
static inline void deinterleave_3x64(__m512i za, __m512i zb, __m512i zc,
                                     __m512i *out_A, __m512i *out_B, __m512i *out_C)
{
    *out_A = _mm512_permutex2var_epi8(
        _mm512_permutex2var_epi8(za, _mm512_load_si512((const void *)deint_A1), zb),
        _mm512_load_si512((const void *)deint_A2), zc);
    *out_B = _mm512_permutex2var_epi8(
        _mm512_permutex2var_epi8(za, _mm512_load_si512((const void *)deint_B1), zb),
        _mm512_load_si512((const void *)deint_B2), zc);
    *out_C = _mm512_permutex2var_epi8(
        _mm512_permutex2var_epi8(za, _mm512_load_si512((const void *)deint_C1), zb),
        _mm512_load_si512((const void *)deint_C2), zc);
}

/* Box-of-3 average with exact div3, transplanted from box3_div_avx2: the
 * maddubs pair-sum keeps A+B off the shuffle port, the widened C joins it,
 * and mulhi by 0x5556 divides exactly for sums in [0,765].  unpack/packus
 * are per-lane symmetric so the output is linear. */
static inline __m512i box3_div_avx512(__m512i A, __m512i B, __m512i C)
{
    const __m512i zero  = _mm512_setzero_si512();
    const __m512i one   = _mm512_set1_epi8(1);
    const __m512i magic = _mm512_set1_epi16((short)0x5556);

    __m512i sum_lo = _mm512_add_epi16(
        _mm512_maddubs_epi16(_mm512_unpacklo_epi8(A, B), one),
        _mm512_unpacklo_epi8(C, zero));
    __m512i sum_hi = _mm512_add_epi16(
        _mm512_maddubs_epi16(_mm512_unpackhi_epi8(A, B), one),
        _mm512_unpackhi_epi8(C, zero));

    return _mm512_packus_epi16(_mm512_mulhi_epu16(sum_lo, magic),
                               _mm512_mulhi_epu16(sum_hi, magic));
}

/* Vertical 2:1 bilinear blend, (a*171 + b*85 + 128) >> 8, returned as the
 * two per-lane 16-bit halves so the horizontal 1.5x can consume them
 * directly.  Widening the bytes makes the exact signed-difference
 * VPMULHRSW identity available without a later pack/unpack round trip. */
static inline void avx512_blend_2_1_u16(__m512i a, __m512i b,
                                        __m512i *out_lo, __m512i *out_hi)
{
    const __m512i zero  = _mm512_setzero_si512();
    const __m512i scale = _mm512_set1_epi16(85 * 128);
    __m512i a_lo = _mm512_unpacklo_epi8(a, zero);
    __m512i a_hi = _mm512_unpackhi_epi8(a, zero);
    __m512i b_lo = _mm512_unpacklo_epi8(b, zero);
    __m512i b_hi = _mm512_unpackhi_epi8(b, zero);

    *out_lo = _mm512_add_epi16(a_lo, _mm512_mulhrs_epi16(
        _mm512_sub_epi16(b_lo, a_lo), scale));
    *out_hi = _mm512_add_epi16(a_hi, _mm512_mulhrs_epi16(
        _mm512_sub_epi16(b_hi, a_hi), scale));
}

/* Horizontal 1.5x bilinear on already-widened u16 component halves.
 * Produces 128 interleaved output bytes.  `full` is a literal at every
 * call site (the function is always inlined), so the masked branch only
 * exists in the tail instantiation. */
static inline __attribute__((always_inline)) void h_chunk_1_5x_u16_avx512(
    __m512i a_lo, __m512i a_hi, __m512i b_lo, __m512i b_hi,
    __m512i c_lo, __m512i c_hi,
    uint8_t *restrict dst, int full, __mmask64 m_lo, __mmask64 m_hi)
{
    /*
     * (171*a + 85*b + 128) >> 8
     *     == a + round((b-a) * 85/256)
     *
     * Every widened component is in [0,255], so the signed difference is
     * safely in [-255,255].  VPMULHRSW with 85*128 performs the exact rounded
     * division by 256, replacing each mullo/mullo/add/add/shift chain with
     * sub/mulhrs/add while preserving bit-for-bit scalar parity.
     */
    const __m512i scale = _mm512_set1_epi16(85 * 128);

    /* out0 blends A toward B; out1 blends C toward B. */
    __m512i out0 = _mm512_packus_epi16(
        _mm512_add_epi16(a_lo, _mm512_mulhrs_epi16(
            _mm512_sub_epi16(b_lo, a_lo), scale)),
        _mm512_add_epi16(a_hi, _mm512_mulhrs_epi16(
            _mm512_sub_epi16(b_hi, a_hi), scale)));
    __m512i out1 = _mm512_packus_epi16(
        _mm512_add_epi16(c_lo, _mm512_mulhrs_epi16(
            _mm512_sub_epi16(b_lo, c_lo), scale)),
        _mm512_add_epi16(c_hi, _mm512_mulhrs_epi16(
            _mm512_sub_epi16(b_hi, c_hi), scale)));

    /* Pixel interleave [out0[0], out1[0], out0[1], ...]: one vpermt2b per
     * 64 output bytes, where AVX2 needs an unpack plus a vperm2i128. */
    __m512i ilo = _mm512_permutex2var_epi8(
        out0, _mm512_load_si512((const void *)il15_lo), out1);
    __m512i ihi = _mm512_permutex2var_epi8(
        out0, _mm512_load_si512((const void *)il15_hi), out1);

    if (full) {
        _mm512_storeu_si512((void *)dst,        ilo);
        _mm512_storeu_si512((void *)(dst + 64), ihi);
    } else {
        _mm512_mask_storeu_epi8(dst,      m_lo, ilo);
        _mm512_mask_storeu_epi8(dst + 64, m_hi, ihi);
    }
}

/* Packed u8 entry used for 1.5x rows that come straight from a pair average
 * (rows 0 and 3 of each group). */
static inline __attribute__((always_inline)) void h_chunk_1_5x_avx512(
    __m512i A, __m512i B, __m512i C,
    uint8_t *restrict dst, int full, __mmask64 m_lo, __mmask64 m_hi)
{
    /* Form the two exact blends directly from packed components instead of
     * routing through the general six-operand widened core.  Keeping B packed
     * until both blends are visible also lets the compiler share its widens. */
    __m512i out0_lo, out0_hi, out1_lo, out1_hi;
    avx512_blend_2_1_u16(A, B, &out0_lo, &out0_hi);
    avx512_blend_2_1_u16(C, B, &out1_lo, &out1_hi);

    __m512i out0 = _mm512_packus_epi16(out0_lo, out0_hi);
    __m512i out1 = _mm512_packus_epi16(out1_lo, out1_hi);
    __m512i ilo = _mm512_permutex2var_epi8(
        out0, _mm512_load_si512((const void *)il15_lo), out1);
    __m512i ihi = _mm512_permutex2var_epi8(
        out0, _mm512_load_si512((const void *)il15_hi), out1);

    if (full) {
        _mm512_storeu_si512((void *)dst, ilo);
        _mm512_storeu_si512((void *)(dst + 64), ihi);
    } else {
        _mm512_mask_storeu_epi8(dst, m_lo, ilo);
        _mm512_mask_storeu_epi8(dst + 64, m_hi, ihi);
    }
}

/* n low bits set; n in [0, 64]. */
static inline __mmask64 fused_mask64(int n)
{
    return (n >= 64) ? (__mmask64)~(uint64_t)0
                     : (__mmask64)(((uint64_t)1 << n) - 1);
}

/* -----------------------------------------------------------------------
 * One 192-byte column chunk of a 6-row group, fused vertical+horizontal.
 *
 * Always inlined with `full` literal at both call sites: the full-chunk
 * instantiation has no masking at all, the tail instantiation (cols < 192,
 * cols % 6 == 0, at most once per row group) masks its loads and stores
 * and reuses the identical vector pipeline.  maskz loads zero the lanes
 * past cols; every operation from there on is positionally elementwise in
 * component space (component lane i only ever meets other lane-i values),
 * so the garbage stays above the valid region and the output masks clip
 * it.  v6x_dst, when non-NULL, receives this chunk's 6x components in
 * planar [A|B|C] layout (width wcomp each) for the fused 12x consumer.
 * ----------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void thirds_chunk_avx512(
    const uint8_t *restrict row0, const uint8_t *restrict row1,
    const uint8_t *restrict row2, const uint8_t *restrict row3,
    const uint8_t *restrict row4, const uint8_t *restrict row5,
    int cx, int cols, int full,
    uint32_t active_outputs,
    int need_1_5x, int need_3x, int need_6x, int need_12x,
    uint8_t *dst_1_5x_r0, uint8_t *dst_1_5x_r1,
    uint8_t *dst_1_5x_r2, uint8_t *dst_1_5x_r3, int out15,
    uint8_t *dst_3x_r0, uint8_t *dst_3x_r1, int out3,
    uint8_t *dst_6x_r0, int out6,
    uint8_t *v6x_dst, int wcomp)
{
    /* Load + store masks (tail instantiation only; folded away when full). */
    __mmask64 mka = 0, mkb = 0, mkc = 0;
    __mmask64 m15lo = 0, m15hi = 0, m3 = 0;
    __mmask32 m6 = 0;
    if (!full) {
        mka = fused_mask64(cols < 64 ? cols : 64);
        mkb = fused_mask64(cols < 64 ? 0 : (cols < 128 ? cols - 64 : 64));
        mkc = fused_mask64(cols < 128 ? 0 : cols - 128);
        int o15 = 2 * wcomp;
        m15lo = fused_mask64(o15 < 64 ? o15 : 64);
        m15hi = fused_mask64(o15 < 64 ? 0 : o15 - 64);
        m3 = fused_mask64(wcomp);
        m6 = (__mmask32)((~(uint32_t)0) >> (32 - wcomp / 2));
    }

    /* Load all 6 rows (3 vectors each) up front so the out-of-order engine
     * can overlap them with the arithmetic below. */
    __m512i r0a, r0b, r0c, r1a, r1b, r1c, r2a, r2b, r2c;
    __m512i r3a, r3b, r3c, r4a, r4b, r4c, r5a, r5b, r5c;
    if (full) {
        r0a = _mm512_loadu_si512((const void *)(row0 + cx));
        r0b = _mm512_loadu_si512((const void *)(row0 + cx + 64));
        r0c = _mm512_loadu_si512((const void *)(row0 + cx + 128));
        r1a = _mm512_loadu_si512((const void *)(row1 + cx));
        r1b = _mm512_loadu_si512((const void *)(row1 + cx + 64));
        r1c = _mm512_loadu_si512((const void *)(row1 + cx + 128));
        r2a = _mm512_loadu_si512((const void *)(row2 + cx));
        r2b = _mm512_loadu_si512((const void *)(row2 + cx + 64));
        r2c = _mm512_loadu_si512((const void *)(row2 + cx + 128));
        r3a = _mm512_loadu_si512((const void *)(row3 + cx));
        r3b = _mm512_loadu_si512((const void *)(row3 + cx + 64));
        r3c = _mm512_loadu_si512((const void *)(row3 + cx + 128));
        r4a = _mm512_loadu_si512((const void *)(row4 + cx));
        r4b = _mm512_loadu_si512((const void *)(row4 + cx + 64));
        r4c = _mm512_loadu_si512((const void *)(row4 + cx + 128));
        r5a = _mm512_loadu_si512((const void *)(row5 + cx));
        r5b = _mm512_loadu_si512((const void *)(row5 + cx + 64));
        r5c = _mm512_loadu_si512((const void *)(row5 + cx + 128));
    } else {
        r0a = _mm512_maskz_loadu_epi8(mka, row0 + cx);
        r0b = _mm512_maskz_loadu_epi8(mkb, row0 + cx + 64);
        r0c = _mm512_maskz_loadu_epi8(mkc, row0 + cx + 128);
        r1a = _mm512_maskz_loadu_epi8(mka, row1 + cx);
        r1b = _mm512_maskz_loadu_epi8(mkb, row1 + cx + 64);
        r1c = _mm512_maskz_loadu_epi8(mkc, row1 + cx + 128);
        r2a = _mm512_maskz_loadu_epi8(mka, row2 + cx);
        r2b = _mm512_maskz_loadu_epi8(mkb, row2 + cx + 64);
        r2c = _mm512_maskz_loadu_epi8(mkc, row2 + cx + 128);
        r3a = _mm512_maskz_loadu_epi8(mka, row3 + cx);
        r3b = _mm512_maskz_loadu_epi8(mkb, row3 + cx + 64);
        r3c = _mm512_maskz_loadu_epi8(mkc, row3 + cx + 128);
        r4a = _mm512_maskz_loadu_epi8(mka, row4 + cx);
        r4b = _mm512_maskz_loadu_epi8(mkb, row4 + cx + 64);
        r4c = _mm512_maskz_loadu_epi8(mkc, row4 + cx + 128);
        r5a = _mm512_maskz_loadu_epi8(mka, row5 + cx);
        r5b = _mm512_maskz_loadu_epi8(mkb, row5 + cx + 64);
        r5c = _mm512_maskz_loadu_epi8(mkc, row5 + cx + 128);
    }

    /* Vertical pairwise averages - the three intermediates every output
     * level derives from. */
    __m512i v01a = _mm512_avg_epu8(r0a, r1a);
    __m512i v01b = _mm512_avg_epu8(r0b, r1b);
    __m512i v01c = _mm512_avg_epu8(r0c, r1c);
    __m512i v23a = _mm512_avg_epu8(r2a, r3a);
    __m512i v23b = _mm512_avg_epu8(r2b, r3b);
    __m512i v23c = _mm512_avg_epu8(r2c, r3c);
    __m512i v45a = _mm512_avg_epu8(r4a, r5a);
    __m512i v45b = _mm512_avg_epu8(r4b, r5b);
    __m512i v45c = _mm512_avg_epu8(r4c, r5c);

    /* Deinterleave the three pair averages ONCE and compute every level in
     * A/B/C component space (deinterleave commutes with the pointwise
     * avg/blend reductions).  At 6 permutes per deinterleave this is cheap
     * enough that the AVX2 kernel's separate no-1.5x path - which dodged
     * one deinterleave in some configs - is not worth a second code path. */
    __m512i Av01, Bv01, Cv01, Av23, Bv23, Cv23, Av45, Bv45, Cv45;
    deinterleave_3x64(v01a, v01b, v01c, &Av01, &Bv01, &Cv01);
    deinterleave_3x64(v23a, v23b, v23c, &Av23, &Bv23, &Cv23);
    deinterleave_3x64(v45a, v45b, v45c, &Av45, &Bv45, &Cv45);

    if (need_1_5x) {
        /* Rows 0/3 use the pair averages directly; rows 1/2 are vertical
         * bilinear blends kept in 16-bit form and fed straight into the
         * horizontal stage. */
        h_chunk_1_5x_avx512(Av01, Bv01, Cv01,
                            dst_1_5x_r0 + out15, full, m15lo, m15hi);
        {
            __m512i bA_lo, bA_hi, bB_lo, bB_hi, bC_lo, bC_hi;
            avx512_blend_2_1_u16(Av01, Av23, &bA_lo, &bA_hi);
            avx512_blend_2_1_u16(Bv01, Bv23, &bB_lo, &bB_hi);
            avx512_blend_2_1_u16(Cv01, Cv23, &bC_lo, &bC_hi);
            h_chunk_1_5x_u16_avx512(bA_lo, bA_hi, bB_lo, bB_hi, bC_lo, bC_hi,
                                    dst_1_5x_r1 + out15, full, m15lo, m15hi);
        }
        {
            __m512i bA_lo, bA_hi, bB_lo, bB_hi, bC_lo, bC_hi;
            avx512_blend_2_1_u16(Av23, Av45, &bA_lo, &bA_hi);
            avx512_blend_2_1_u16(Bv23, Bv45, &bB_lo, &bB_hi);
            avx512_blend_2_1_u16(Cv23, Cv45, &bC_lo, &bC_hi);
            h_chunk_1_5x_u16_avx512(bA_lo, bA_hi, bB_lo, bB_hi, bC_lo, bC_hi,
                                    dst_1_5x_r2 + out15, full, m15lo, m15hi);
        }
        h_chunk_1_5x_avx512(Av45, Bv45, Cv45,
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
        A3x0 = _mm512_avg_epu8(Av01, Av23);
        B3x0 = _mm512_avg_epu8(Bv01, Bv23);
        C3x0 = _mm512_avg_epu8(Cv01, Cv23);
        A3x1 = _mm512_avg_epu8(Av23, Av45);
        B3x1 = _mm512_avg_epu8(Bv23, Bv45);
        C3x1 = _mm512_avg_epu8(Cv23, Cv45);

        if (active_outputs & (1u << 2)) {
            __m512i r0 = box3_div_avx512(A3x0, B3x0, C3x0);
            __m512i r1 = box3_div_avx512(A3x1, B3x1, C3x1);
            if (full) {
                _mm512_storeu_si512((void *)(dst_3x_r0 + out3), r0);
                _mm512_storeu_si512((void *)(dst_3x_r1 + out3), r1);
            } else {
                _mm512_mask_storeu_epi8(dst_3x_r0 + out3, m3, r0);
                _mm512_mask_storeu_epi8(dst_3x_r1 + out3, m3, r1);
            }
        }
    }

    /* 6x components: feed the 6x output row and/or the planar 12x
     * intermediate (need_12x implies need_3x, so the 3x components above
     * are real values here). */
    if ((need_6x && (active_outputs & (1u << 4))) || need_12x) {
        __m512i A6x = _mm512_avg_epu8(A3x0, A3x1);
        __m512i B6x = _mm512_avg_epu8(B3x0, B3x1);
        __m512i C6x = _mm512_avg_epu8(C3x0, C3x1);
        if (need_6x && (active_outputs & (1u << 4))) {
            __m256i six = avx512_halve_64_to_32(box3_div_avx512(A6x, B6x, C6x));
            if (full) {
                _mm256_storeu_si256((__m256i *)(dst_6x_r0 + out6), six);
            } else {
                _mm256_mask_storeu_epi8(dst_6x_r0 + out6, m6, six);
            }
        }
        if (need_12x) {
            if (full) {
                _mm512_storeu_si512((void *)(v6x_dst),       A6x);
                _mm512_storeu_si512((void *)(v6x_dst + 64),  B6x);
                _mm512_storeu_si512((void *)(v6x_dst + 128), C6x);
            } else {
                _mm512_mask_storeu_epi8(v6x_dst,             m3, A6x);
                _mm512_mask_storeu_epi8(v6x_dst + wcomp,     m3, B6x);
                _mm512_mask_storeu_epi8(v6x_dst + 2 * wcomp, m3, C6x);
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single plane (AVX-512 fused vertical+horizontal)
 *
 * A 6-source-row group produces 4 output rows at 1.5x, 2 at 3x, 1 at 6x;
 * 12x pairs two consecutive groups' 6x intermediates via the same
 * ping-pong scheme as the AVX2 kernel, but the intermediate is ALWAYS
 * stored in per-chunk component-planar layout ([A|B|C] per 192-byte
 * chunk; the tail chunk packs its three wcomp-wide components the same
 * way).  The pointwise vertical average between the two groups preserves
 * that layout, so the fused 12x consumer reads components directly:
 * box3 -> halve -> halve, 192 planar bytes to 16 output bytes with zero
 * shuffle work beyond the cascade itself.
 * ----------------------------------------------------------------------- */
static void __attribute__((hot)) scale_plane_thirds_avx512(
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

    /* 12x ping-pong buffers for the planar 6x intermediates. */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint8_t *v6x_cur = NULL, *v6x_prev = NULL;
    if (need_12x) {
        v6x_cur  = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)src_w);
        v6x_prev = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)src_w);
        if (!v6x_cur || !v6x_prev) return;
    }

    /* Chunk geometry: 192 source bytes per chunk (LCM of 64 and 3); the
     * remainder is a multiple of 6 (guaranteed by the effective-width
     * checks at init) and runs through the same body under masks. */
    int full_chunks = src_w / 192;
    int tail_start  = full_chunks * 192;
    int tail_cols   = src_w - tail_start;
    int tail_wcomp  = tail_cols / 3;

    /* Pointwise vector geometry for the 12x between-group average. */
    int v_chunks = src_w / 64;
    __mmask64 v_mask = (src_w % 64)
        ? fused_mask64(src_w % 64) : 0;

    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint8_t *grp = src + (size_t)g6 * 6 * (size_t)src_stride;

        const uint8_t *restrict row0 = grp;
        const uint8_t *restrict row1 = grp + (size_t)src_stride;
        const uint8_t *restrict row2 = grp + (size_t)2 * (size_t)src_stride;
        const uint8_t *restrict row3 = grp + (size_t)3 * (size_t)src_stride;
        const uint8_t *restrict row4 = grp + (size_t)4 * (size_t)src_stride;
        const uint8_t *restrict row5 = grp + (size_t)5 * (size_t)src_stride;

        /* Output row base pointers for this group. */
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

        /* Full chunks, then at most one masked tail chunk. */
        for (int ci = 0; ci < full_chunks; ci++) {
            thirds_chunk_avx512(row0, row1, row2, row3, row4, row5,
                ci * 192, 192, /*full=*/1,
                active_outputs, need_1_5x, need_3x, need_6x, need_12x,
                dst_1_5x_r0, dst_1_5x_r1, dst_1_5x_r2, dst_1_5x_r3, ci * 128,
                dst_3x_r0, dst_3x_r1, ci * 64,
                dst_6x_r0, ci * 32,
                v6x_cur ? v6x_cur + (size_t)ci * 192 : NULL, 64);
        }
        if (tail_cols > 0) {
            thirds_chunk_avx512(row0, row1, row2, row3, row4, row5,
                tail_start, tail_cols, /*full=*/0,
                active_outputs, need_1_5x, need_3x, need_6x, need_12x,
                dst_1_5x_r0, dst_1_5x_r1, dst_1_5x_r2, dst_1_5x_r3,
                full_chunks * 128,
                dst_3x_r0, dst_3x_r1, full_chunks * 64,
                dst_6x_r0, full_chunks * 32,
                v6x_cur ? v6x_cur + (size_t)full_chunks * 192 : NULL,
                tail_wcomp);
        }

        if (need_1_5x) out_row[0] += 4;
        if (active_outputs & (1u << 2)) out_row[1] += 2;
        if (active_outputs & (1u << 4)) out_row[2] += 1;

        /* 12x: pair two consecutive groups' 6x intermediates.  Even groups
         * just become "previous"; odd groups average against it (pointwise,
         * so the planar chunk layout survives) and run the fused horizontal
         * cascade. */
        if (need_12x) {
            if ((g6 & 1) == 0) {
                uint8_t *tmp = v6x_prev;
                v6x_prev = v6x_cur;
                v6x_cur  = tmp;
            } else {
                /* In-place pointwise average (v6x_prev aliases the
                 * destination; each vector is fully loaded before its store). */
                {
                    int x = 0;
                    for (int c = 0; c < v_chunks; c++, x += 64) {
                        __m512i a = _mm512_loadu_si512((const void *)(v6x_prev + x));
                        __m512i b = _mm512_loadu_si512((const void *)(v6x_cur + x));
                        _mm512_storeu_si512((void *)(v6x_prev + x),
                                            _mm512_avg_epu8(a, b));
                    }
                    if (v_mask) {
                        __m512i a = _mm512_maskz_loadu_epi8(v_mask, v6x_prev + x);
                        __m512i b = _mm512_maskz_loadu_epi8(v_mask, v6x_cur + x);
                        _mm512_mask_storeu_epi8(v6x_prev + x, v_mask,
                                                _mm512_avg_epu8(a, b));
                    }
                }

                if (active_outputs & (1u << 6)) {
                    uint8_t *restrict out12 = dst_planes[3]
                        + (size_t)out_row[3] * (size_t)dst_strides[3];

                    /* 192 planar component bytes -> 16 output bytes:
                     * box-of-3 (3x), halve (6x), halve (12x), all in
                     * registers - the components were deinterleaved back
                     * when the chunk loop had them for free, so this
                     * cascade does no shuffle work it didn't need anyway. */
                    for (int c = 0; c < full_chunks; c++) {
                        const uint8_t *cs = v6x_prev + (size_t)c * 192;
                        __m512i A = _mm512_loadu_si512((const void *)cs);
                        __m512i B = _mm512_loadu_si512((const void *)(cs + 64));
                        __m512i C = _mm512_loadu_si512((const void *)(cs + 128));
                        __m256i six = avx512_halve_64_to_32(
                                          box3_div_avx512(A, B, C));
                        _mm_storeu_si128((__m128i *)(out12 + (size_t)c * 16),
                                         avx512_halve_32_to_16(six));
                    }
                    if (tail_cols > 0) {
                        const uint8_t *cs = v6x_prev + (size_t)tail_start;
                        __mmask64 mw = fused_mask64(tail_wcomp);
                        __m512i A = _mm512_maskz_loadu_epi8(mw, cs);
                        __m512i B = _mm512_maskz_loadu_epi8(mw, cs + tail_wcomp);
                        __m512i C = _mm512_maskz_loadu_epi8(mw, cs + 2 * tail_wcomp);
                        __m256i six = avx512_halve_64_to_32(
                                          box3_div_avx512(A, B, C));
                        __mmask16 m12 = (__mmask16)((1u << (tail_wcomp / 4)) - 1);
                        _mm_mask_storeu_epi8(out12 + (size_t)full_chunks * 16,
                                             m12, avx512_halve_32_to_16(six));
                    }
                    out_row[3]++;
                }
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}

/* -----------------------------------------------------------------------
 * Public entry points
 *
 * YUV420 I420 has the chroma planes at half the luma dimensions in both
 * axes.  We process the Y plane at full size, then U and V at half width
 * and half height with the same kernel.
 * ----------------------------------------------------------------------- */

void __attribute__((hot)) fused_kernel_pow2_avx512(const fused_kernel_params_t *p,
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
    scale_plane_pow2_avx512(src_y,
                            p->src_width, p->src_height, p->src_y_stride,
                            p->active_outputs,
                            y_planes, y_widths, y_strides, y_heights,
                            p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_pow2_avx512(src_u,
                            p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                            p->active_outputs,
                            u_planes, uv_widths, uv_strides, uv_heights,
                            p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_pow2_avx512(src_v,
                            p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                            p->active_outputs,
                            v_planes, uv_widths, uv_strides, uv_heights,
                            p->scratch_pool, p->scratch_pool_size);

    _mm256_zeroupper();
}

void __attribute__((hot)) fused_kernel_thirds_avx512(const fused_kernel_params_t *p,
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
    scale_plane_thirds_avx512(src_y,
                              p->src_width, p->src_height, p->src_y_stride,
                              p->active_outputs,
                              y_planes, y_widths, y_strides, y_heights,
                              p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_thirds_avx512(src_u,
                              p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                              p->active_outputs,
                              u_planes, uv_widths, uv_strides, uv_heights,
                              p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_thirds_avx512(src_v,
                              p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                              p->active_outputs,
                              v_planes, uv_widths, uv_strides, uv_heights,
                              p->scratch_pool, p->scratch_pool_size);

    _mm256_zeroupper();
}

#else /* compiler lacks AVX-512 support - self-stubbed build */

/* Stub build: the Makefile probe found the compiler can't build AVX-512,
 * so no -mavx512* flags were passed and no AVX-512 instruction exists
 * anywhere in this object.  Returning 0 here keeps runtime dispatch on
 * the AVX2/scalar paths; the entry points below exist only to satisfy
 * the linker and delegate to AVX2 so even a miswired call stays correct. */
int fused_avx512_compiled(void)
{
    return 0;
}

void fused_kernel_pow2_avx512(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v)
{
    fused_kernel_pow2_avx2(p, src_y, src_u, src_v);
}

void fused_kernel_thirds_avx512(const fused_kernel_params_t *p,
                                const uint8_t *src_y,
                                const uint8_t *src_u,
                                const uint8_t *src_v)
{
    fused_kernel_thirds_avx2(p, src_y, src_u, src_v);
}

#endif /* AVX-512 feature macros */

#endif /* __x86_64__ */
