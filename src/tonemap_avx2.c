/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * tonemap_avx2.c - AVX2 built-in-curve tone mapping.
 *
 * Entry points mirror the scalar reference in tonemap.c:
 *   fused_tonemap_apply_avx2      - planar I010 chroma
 *   fused_tonemap_apply_p010_avx2 - interleaved P010 chroma
 *
 * Same pipeline as the AVX-512 kernel, with one structural difference:
 * AVX2 has no cross-lane byte permute, so the 1024-entry pq_to_sdr lookup
 * uses vpgatherdd (8 dword loads per op, low byte kept).  Gathers are the
 * cost center here - 12 per 32-pixel row - which is why this kernel is a
 * "several-x", not "order-of-magnitude", win over scalar; it exists for
 * the AVX2-only fleet where that still halves the dominant HDR cost.
 *
 * Everything else follows the scalar reference exactly: Q10 chroma
 * deltas evaluated arithmetically from state->delta_coef_* (the same
 * integer form the scalar tables are generated from), index adds and
 * the luma dot in 16-bit (modular; max value 65408), chroma-rate gamut
 * and encode dots accumulated in 32-bit.  Bit-exactness against
 * fused_tonemap_apply_scalar is enforced by the parity test.
 *
 * AVX2 has no element masks, so ragged edges (chroma_w % 16) fall back
 * to fused_tm_block - the same shared inline the scalar loops use.
 *
 * The gather indexes a 4096-entry padded copy of pq_to_sdr built once
 * per frame (see TM_LUT_BIAS): the indices carry a bias of 1024 that is
 * folded into the chroma-rate deltas, and the padding replicates the end
 * values, so Y' + delta needs no clamp before the lookup.  vpgatherdd
 * loads 4 bytes per index and the highest index is 3011, so the over-read
 * stays inside the padded buffer; the extra bytes are masked off.
 *
 * The chroma-rate gamut and encode dots run on word pairs with vpmaddwd
 * ([r,g].[425,-150] + [b,128].[-19,1] and so on) rather than dword
 * multiplies: every operand is byte-range or an 8.8 scale, so the pair
 * products are exact and vpmulld never appears.
 *
 * This file is compiled with -mavx2 unconditionally (baseline compilers
 * all support it) and selected at runtime behind caps->has_avx2.
 */

#if defined(__x86_64__)

#include "internal.h"
#include "tonemap.h"

#include <immintrin.h>

#define ALIGN32 __attribute__((aligned(32)))

#include <string.h>

/* Packed word pair for vpmaddwd: low word first. */
#define TM_PAIR(lo, hi) \
    ((int)(((uint32_t)(uint16_t)(hi) << 16) | (uint16_t)(lo)))

/* -----------------------------------------------------------------------
 * Clamp-free indexing.
 *
 * The scalar reference clamps Y' + delta to [0, 1023] before indexing the
 * 1024-entry pq_to_sdr table.  |delta| is bounded by 512 * 1.8814 < 1024
 * (the largest Q10 NCL coefficient, full-range Cb - see tonemap.c), so
 * Y' + delta + TM_LUT_BIAS always lands in [60, 3011].  The kernel
 * therefore gathers from a 4096-entry copy of the table whose entries
 * outside the central 1024 replicate the end values, which is exactly
 * what the clamp would have selected.  The bias is folded into the
 * deltas at chroma rate, so each per-channel index costs a single add.
 * ----------------------------------------------------------------------- */

#define TM_LUT_BIAS 1024
#define TM_LUT_PAD  4096

static void tm_build_padded_lut(const uint8_t *lut, uint8_t *pad)
{
    memset(pad, lut[0], TM_LUT_BIAS);
    memcpy(pad + TM_LUT_BIAS, lut, 1024);
    memset(pad + TM_LUT_BIAS + 1024, lut[1023],
           TM_LUT_PAD - TM_LUT_BIAS - 1024);
}

/* -----------------------------------------------------------------------
 * Padded byte LUT lookup for 16 word indices -> 16 words.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) __m256i
tm_lut16_avx2(const uint8_t *lut, __m256i idx_words)
{
    __m256i i0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(idx_words));
    __m256i i1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(idx_words, 1));
    __m256i g0 = _mm256_i32gather_epi32((const int *)lut, i0, 1);
    __m256i g1 = _mm256_i32gather_epi32((const int *)lut, i1, 1);
    __m256i ff = _mm256_set1_epi32(0xFF);
    g0 = _mm256_and_si256(g0, ff);
    g1 = _mm256_and_si256(g1, ff);
    /* packusdw interleaves 128-bit lanes; vpermq 0xD8 restores order */
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(g0, g1), 0xD8);
}

/* -----------------------------------------------------------------------
 * Q10 delta evaluation for 16 chroma codes (words) -> 16 deltas (words).
 *
 * pmulhrsw of (code - 512) << 5 by the Q10 coefficient computes
 *   ((x*32) * coef + 16384) >> 15  ==  (x*coef + 512) >> 10
 * which is exactly the scalar table definition, entirely in 16-bit.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) __m256i
tm_delta16_avx2(__m256i codes, __m256i coef, __m256i bias)
{
    __m256i x = _mm256_slli_epi16(
        _mm256_sub_epi16(codes, _mm256_set1_epi16(512)), 5);
    return _mm256_add_epi16(_mm256_mulhrs_epi16(x, coef), bias);
}

/* dG = g_cr(Cr) + g_cb(Cb): each term rounds exactly like its scalar
 * table entry, and the sum (magnitude < 1024) stays in 16-bit. */
static inline __attribute__((always_inline)) __m256i
tm_delta16_g_avx2(__m256i cb, __m256i cr, __m256i gcb, __m256i gcr,
                  __m256i bias)
{
    __m256i c512 = _mm256_set1_epi16(512);
    __m256i xb = _mm256_slli_epi16(_mm256_sub_epi16(cb, c512), 5);
    __m256i xr = _mm256_slli_epi16(_mm256_sub_epi16(cr, c512), 5);
    return _mm256_add_epi16(
        _mm256_add_epi16(_mm256_mulhrs_epi16(xr, gcr), bias),
        _mm256_mulhrs_epi16(xb, gcb));
}

/* -----------------------------------------------------------------------
 * 16-pixel row segment: indices, lookups, luma dot.
 * Returns the r/g/b/y values as 16 words each.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_seg16_avx2(const uint16_t *sy, __m256i dr, __m256i dg, __m256i db,
              const uint8_t *lut,
              __m256i *r_out, __m256i *g_out, __m256i *b_out, __m256i *y_out)
{
    __m256i mask10 = _mm256_set1_epi16(0x3FF);

    __m256i y = _mm256_and_si256(
        _mm256_loadu_si256((const __m256i *)sy), mask10);

    /* Biased indices into the padded table; no clamp needed. */
    __m256i ir = _mm256_add_epi16(y, dr);
    __m256i ig = _mm256_add_epi16(y, dg);
    __m256i ib = _mm256_add_epi16(y, db);

    __m256i r = tm_lut16_avx2(lut, ir);
    __m256i g = tm_lut16_avx2(lut, ig);
    __m256i b = tm_lut16_avx2(lut, ib);

    /* (67r + 174g + 15b + 128) >> 8, modular 16-bit (max 65408) */
    __m256i ys = _mm256_srli_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_mullo_epi16(r, _mm256_set1_epi16(67)),
                _mm256_mullo_epi16(g, _mm256_set1_epi16(174))),
            _mm256_add_epi16(
                _mm256_mullo_epi16(b, _mm256_set1_epi16(15)),
                _mm256_set1_epi16(128))), 8);

    *r_out = r;
    *g_out = g;
    *b_out = b;
    *y_out = ys;
}

/* -----------------------------------------------------------------------
 * Chroma-rate gamut + BT.709 encode dots for the 8 even-position pixels
 * of a 16-word segment.  Mirrors fused_tm_rgb_to_chroma_sdr exactly.
 *
 * Even-pixel extraction is free in the dword view of a word vector: the
 * low word of dword i is pixel 2i.  Masking r to its low word and
 * shifting g up forms the [r, g] pair vpmaddwd consumes; [b, 128] pairs
 * the third channel with the rounding constant.  The encode step packs
 * [x709, y] the same way so (x709 - y) * scale is one vpmaddwd against
 * [scale, -scale].  Every product is exact in 32-bit.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chroma_dots_avx2(__m256i r, __m256i g, __m256i b, __m256i y,
                    __m256i cbs_pair, __m256i crs_pair,
                    __m256i cmin, __m256i cmax,
                    __m256i *cb_out, __m256i *cr_out)
{
    __m256i lo16 = _mm256_set1_epi32(0xFFFF);
    __m256i c128 = _mm256_set1_epi32(128);
    __m256i zero = _mm256_setzero_si256();
    __m256i c255 = _mm256_set1_epi32(255);

    __m256i rg   = _mm256_or_si256(_mm256_and_si256(r, lo16),
                                   _mm256_slli_epi32(g, 16));
    __m256i b128 = _mm256_or_si256(_mm256_and_si256(b, lo16),
                                   _mm256_set1_epi32(128 << 16));
    __m256i ye   = _mm256_slli_epi32(y, 16);

    /* Gamut rows * 256 (each sums to 256): see tonemap.c */
    __m256i r709 = _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(rg,   _mm256_set1_epi32(TM_PAIR(425, -150))),
        _mm256_madd_epi16(b128, _mm256_set1_epi32(TM_PAIR(-19, 1)))), 8);
    __m256i b709 = _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(rg,   _mm256_set1_epi32(TM_PAIR(-5, -26))),
        _mm256_madd_epi16(b128, _mm256_set1_epi32(TM_PAIR(287, 1)))), 8);
    r709 = _mm256_min_epi32(_mm256_max_epi32(r709, zero), c255);
    b709 = _mm256_min_epi32(_mm256_max_epi32(b709, zero), c255);

    /* cb = clamp(128 + ((b709 - y)*cbs + 128 >> 8), cmin, cmax) */
    __m256i cb = _mm256_add_epi32(c128, _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_or_si256(b709, ye), cbs_pair), c128), 8));
    __m256i cr = _mm256_add_epi32(c128, _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_or_si256(r709, ye), crs_pair), c128), 8));

    *cb_out = _mm256_min_epi32(_mm256_max_epi32(cb, cmin), cmax);
    *cr_out = _mm256_min_epi32(_mm256_max_epi32(cr, cmin), cmax);
}

/* -----------------------------------------------------------------------
 * One chunk: 16 chroma samples = 2 rows x 32 luma pixels (full only;
 * ragged edges are handled by the caller with fused_tm_block).
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chunk_avx2(const fused_hdr_internal_t *state, const uint8_t *lut,
              const uint16_t *su, const uint16_t *sv,
              const uint16_t *suv, int p010,
              const uint16_t *sy0, const uint16_t *sy1,
              uint8_t *dy0, uint8_t *dy1,
              uint8_t *du, uint8_t *dv)
{
    __m256i mask10 = _mm256_set1_epi16(0x3FF);
    __m256i bias   = _mm256_set1_epi16(TM_LUT_BIAS);

    /* ---- chroma load ---- */
    __m256i cb, cr;
    if (p010) {
        __m256i uv0 = _mm256_loadu_si256((const __m256i *)suv);
        __m256i uv1 = _mm256_loadu_si256((const __m256i *)(suv + 16));
        __m256i m10d = _mm256_set1_epi32(0x3FF);
        __m256i cb0 = _mm256_and_si256(uv0, m10d);
        __m256i cb1 = _mm256_and_si256(uv1, m10d);
        __m256i cr0 = _mm256_and_si256(_mm256_srli_epi32(uv0, 16), m10d);
        __m256i cr1 = _mm256_and_si256(_mm256_srli_epi32(uv1, 16), m10d);
        cb = _mm256_permute4x64_epi64(_mm256_packus_epi32(cb0, cb1), 0xD8);
        cr = _mm256_permute4x64_epi64(_mm256_packus_epi32(cr0, cr1), 0xD8);
    } else {
        cb = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)su), mask10);
        cr = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)sv), mask10);
    }

    /* ---- Q10 deltas at chroma rate ---- */
    __m256i dr = tm_delta16_avx2(
        cr, _mm256_set1_epi16((short)state->delta_coef_r), bias);
    __m256i db = tm_delta16_avx2(
        cb, _mm256_set1_epi16((short)state->delta_coef_b), bias);
    __m256i dg = tm_delta16_g_avx2(
        cb, cr,
        _mm256_set1_epi16((short)state->delta_coef_g_cb),
        _mm256_set1_epi16((short)state->delta_coef_g_cr), bias);

    /* ---- expand to luma rate: 16 words -> 2x16 duplicated words ----
     * vpermq 0xD8 makes each 128-bit lane hold the right source words,
     * then in-lane unpacks duplicate them. */
    __m256i drq = _mm256_permute4x64_epi64(dr, 0xD8);
    __m256i dgq = _mm256_permute4x64_epi64(dg, 0xD8);
    __m256i dbq = _mm256_permute4x64_epi64(db, 0xD8);
    __m256i dr0 = _mm256_unpacklo_epi16(drq, drq);  /* px 0..15  */
    __m256i dr1 = _mm256_unpackhi_epi16(drq, drq);  /* px 16..31 */
    __m256i dg0 = _mm256_unpacklo_epi16(dgq, dgq);
    __m256i dg1 = _mm256_unpackhi_epi16(dgq, dgq);
    __m256i db0 = _mm256_unpacklo_epi16(dbq, dbq);
    __m256i db1 = _mm256_unpackhi_epi16(dbq, dbq);

    /* ---- two luma rows, two 16-pixel segments each ---- */
    __m256i r00, g00, b00, y00, r01, g01, b01, y01;
    __m256i rt, gt, bt, yt;

    tm_seg16_avx2(sy0,      dr0, dg0, db0, lut, &r00, &g00, &b00, &y00);
    tm_seg16_avx2(sy0 + 16, dr1, dg1, db1, lut, &r01, &g01, &b01, &y01);
    _mm256_storeu_si256((__m256i *)dy0,
        _mm256_permute4x64_epi64(_mm256_packus_epi16(y00, y01), 0xD8));

    tm_seg16_avx2(sy1,      dr0, dg0, db0, lut, &rt, &gt, &bt, &yt);
    __m256i ylo = yt;
    tm_seg16_avx2(sy1 + 16, dr1, dg1, db1, lut, &rt, &gt, &bt, &yt);
    _mm256_storeu_si256((__m256i *)dy1,
        _mm256_permute4x64_epi64(_mm256_packus_epi16(ylo, yt), 0xD8));

    /* ---- chroma outputs from row 0's top-left pixels ---- */
    int cbs = state->cb_out_scale, crs = state->cr_out_scale;
    __m256i cbs_pair = _mm256_set1_epi32(TM_PAIR(cbs, -cbs));
    __m256i crs_pair = _mm256_set1_epi32(TM_PAIR(crs, -crs));
    __m256i cmin = _mm256_set1_epi32(state->chroma_out_min);
    __m256i cmax = _mm256_set1_epi32(state->chroma_out_max);

    __m256i cb_lo, cr_lo, cb_hi, cr_hi;
    tm_chroma_dots_avx2(r00, g00, b00, y00, cbs_pair, crs_pair, cmin, cmax,
                        &cb_lo, &cr_lo);
    tm_chroma_dots_avx2(r01, g01, b01, y01, cbs_pair, crs_pair, cmin, cmax,
                        &cb_hi, &cr_hi);

    /* 16 dwords -> 16 bytes */
    __m256i cbw = _mm256_permute4x64_epi64(
        _mm256_packs_epi32(cb_lo, cb_hi), 0xD8);
    __m256i crw = _mm256_permute4x64_epi64(
        _mm256_packs_epi32(cr_lo, cr_hi), 0xD8);
    __m128i cbb = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
        _mm256_packus_epi16(cbw, cbw), 0x08));
    __m128i crb = _mm256_castsi256_si128(_mm256_permute4x64_epi64(
        _mm256_packus_epi16(crw, crw), 0x08));

    _mm_storeu_si128((__m128i *)du, cbb);
    _mm_storeu_si128((__m128i *)dv, crb);
}

/* -----------------------------------------------------------------------
 * Frame driver.  p010 is a compile-time literal at both call sites.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_frame_avx2(const fused_hdr_internal_t *state,
              const uint16_t *src_y, int src_y_stride,
              const uint16_t *src_u, const uint16_t *src_v,
              const uint16_t *src_uv, int src_uv_stride, int p010,
              uint8_t *dst_y, int dst_y_stride,
              uint8_t *dst_u, int dst_uv_stride, uint8_t *dst_v,
              int width, int height)
{
    int chroma_w = width  / 2;
    int chroma_h = height / 2;

    int src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    int src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);

    int simd_cw = chroma_w & ~15;

    uint8_t lut_pad[TM_LUT_PAD] __attribute__((aligned(64)));
    tm_build_padded_lut(state->pq_to_sdr, lut_pad);

    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *su  = src_u  ? src_u  + cy * src_uv_pitch : 0;
        const uint16_t *sv  = src_v  ? src_v  + cy * src_uv_pitch : 0;
        const uint16_t *suv = src_uv ? src_uv + cy * src_uv_pitch : 0;
        const uint16_t *sy0 = src_y + (cy * 2) * src_y_pitch;
        const uint16_t *sy1 = sy0 + src_y_pitch;
        uint8_t        *dy0 = dst_y + (cy * 2) * dst_y_stride;
        uint8_t        *dy1 = dy0 + dst_y_stride;
        uint8_t        *du  = dst_u + cy * dst_uv_stride;
        uint8_t        *dv  = dst_v + cy * dst_uv_stride;

        int cx = 0;
        for (; cx < simd_cw; cx += 16) {
            tm_chunk_avx2(state, lut_pad,
                          p010 ? 0 : su + cx, p010 ? 0 : sv + cx,
                          p010 ? suv + cx * 2 : 0, p010,
                          sy0 + cx * 2, sy1 + cx * 2,
                          dy0 + cx * 2, dy1 + cx * 2,
                          du + cx, dv + cx);
        }
        for (; cx < chroma_w; cx++) {
            int cb_10 = p010 ? (suv[cx * 2] & 0x3FF) : (su[cx] & 0x3FF);
            int cr_10 = p010 ? (suv[cx * 2 + 1] & 0x3FF) : (sv[cx] & 0x3FF);
            fused_tm_block(state, cb_10, cr_10,
                           sy0 + cx * 2, sy1 + cx * 2,
                           dy0 + cx * 2, dy1 + cx * 2,
                           &du[cx], &dv[cx]);
        }
    }

    _mm256_zeroupper();
}

__attribute__((hot))
void fused_tonemap_apply_avx2(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_avx2(state, src_y, src_y_stride, src_u, src_v,
                  0, src_uv_stride, 0,
                  dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                  width, height);
}

__attribute__((hot))
void fused_tonemap_apply_p010_avx2(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_avx2(state, src_y, src_y_stride, 0, 0,
                  src_uv, src_uv_stride, 1,
                  dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                  width, height);
}

#endif /* __x86_64__ */
