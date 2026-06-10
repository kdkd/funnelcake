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
 * Everything else follows the scalar reference exactly: Q15 chroma
 * deltas evaluated arithmetically from state->delta_coef_* (the same
 * integer form the scalar tables are generated from), index clamps and
 * the luma dot in 16-bit (modular; max value 65408), chroma-rate gamut
 * and encode dots in 32-bit.  Bit-exactness against
 * fused_tonemap_apply_scalar is enforced by the parity test.
 *
 * AVX2 has no element masks, so ragged edges (chroma_w % 16) fall back
 * to fused_tm_block - the same shared inline the scalar loops use.
 *
 * NOTE on gather safety: vpgatherdd loads 4 bytes at lut+idx with idx up
 * to 1023, reading up to 3 bytes past pq_to_sdr.  pq_to_sdr is followed
 * by the delta arrays inside fused_hdr_internal_t, so the over-read stays
 * inside the same object; the extra bytes are masked off.
 *
 * This file is compiled with -mavx2 unconditionally (baseline compilers
 * all support it) and selected at runtime behind caps->has_avx2.
 */

#if defined(__x86_64__)

#include "internal.h"
#include "tonemap.h"

#include <immintrin.h>

#define ALIGN32 __attribute__((aligned(32)))

/* vpshufb mask: compact the even words of each 128-bit lane into the
 * lane's low 8 bytes (used to pull the 2x2-block top-left values out of
 * a 16-word vector). */
static const ALIGN32 uint8_t shuf_even_words[32] = {
     0, 1, 4, 5, 8, 9,12,13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
     0, 1, 4, 5, 8, 9,12,13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
};

/* -----------------------------------------------------------------------
 * 1024-entry byte LUT lookup for 16 word indices -> 16 words.
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
 * Q15 delta evaluation for 16 chroma codes (words) -> 16 deltas (words).
 *   delta = ((code - 512) * coef + 16384) >> 15
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) __m256i
tm_delta16_avx2(__m256i codes, __m256i coef)
{
    __m256i c512 = _mm256_set1_epi32(512);
    __m256i bias = _mm256_set1_epi32(16384);
    __m256i x0 = _mm256_sub_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(codes)), c512);
    __m256i x1 = _mm256_sub_epi32(
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(codes, 1)), c512);
    __m256i d0 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(x0, coef), bias), 15);
    __m256i d1 = _mm256_srai_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(x1, coef), bias), 15);
    /* values fit int16; packssdw is exact, vpermq fixes lane order */
    return _mm256_permute4x64_epi64(_mm256_packs_epi32(d0, d1), 0xD8);
}

/* dG needs the 32-bit sum of both contributions before narrowing. */
static inline __attribute__((always_inline)) __m256i
tm_delta16_g_avx2(__m256i cb, __m256i cr, __m256i gcb, __m256i gcr)
{
    __m256i c512 = _mm256_set1_epi32(512);
    __m256i bias = _mm256_set1_epi32(16384);
    __m256i xb0 = _mm256_sub_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(cb)), c512);
    __m256i xb1 = _mm256_sub_epi32(
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(cb, 1)), c512);
    __m256i xr0 = _mm256_sub_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(cr)), c512);
    __m256i xr1 = _mm256_sub_epi32(
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(cr, 1)), c512);
    __m256i d0 = _mm256_add_epi32(
        _mm256_srai_epi32(_mm256_add_epi32(
            _mm256_mullo_epi32(xr0, gcr), bias), 15),
        _mm256_srai_epi32(_mm256_add_epi32(
            _mm256_mullo_epi32(xb0, gcb), bias), 15));
    __m256i d1 = _mm256_add_epi32(
        _mm256_srai_epi32(_mm256_add_epi32(
            _mm256_mullo_epi32(xr1, gcr), bias), 15),
        _mm256_srai_epi32(_mm256_add_epi32(
            _mm256_mullo_epi32(xb1, gcb), bias), 15));
    return _mm256_permute4x64_epi64(_mm256_packs_epi32(d0, d1), 0xD8);
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
    __m256i zero   = _mm256_setzero_si256();
    __m256i top    = _mm256_set1_epi16(1023);

    __m256i y = _mm256_and_si256(
        _mm256_loadu_si256((const __m256i *)sy), mask10);

    __m256i ir = _mm256_min_epi16(_mm256_max_epi16(
                     _mm256_add_epi16(y, dr), zero), top);
    __m256i ig = _mm256_min_epi16(_mm256_max_epi16(
                     _mm256_add_epi16(y, dg), zero), top);
    __m256i ib = _mm256_min_epi16(_mm256_max_epi16(
                     _mm256_add_epi16(y, db), zero), top);

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

/* Compact the even words of a 16-word vector into 8 words (xmm). */
static inline __attribute__((always_inline)) __m128i
tm_even_words(__m256i v)
{
    __m256i s = _mm256_shuffle_epi8(
        v, _mm256_load_si256((const __m256i *)shuf_even_words));
    return _mm256_castsi256_si128(_mm256_permute4x64_epi64(s, 0x08));
}

/* -----------------------------------------------------------------------
 * Chroma-rate gamut + BT.709 encode dots for 8 dwords (one xmm of
 * even-position words widened).  Mirrors fused_tm_rgb_to_chroma_sdr.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chroma_dots_avx2(__m256i r, __m256i g, __m256i b, __m256i y,
                    __m256i cbs, __m256i crs, __m256i cmin, __m256i cmax,
                    __m256i *cb_out, __m256i *cr_out)
{
    __m256i c128 = _mm256_set1_epi32(128);
    __m256i zero = _mm256_setzero_si256();
    __m256i c255 = _mm256_set1_epi32(255);

    __m256i r709 = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(r, _mm256_set1_epi32(425)),
                _mm256_mullo_epi32(g, _mm256_set1_epi32(-150))),
            _mm256_add_epi32(
                _mm256_mullo_epi32(b, _mm256_set1_epi32(-19)), c128)), 8);
    __m256i b709 = _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(r, _mm256_set1_epi32(-5)),
                _mm256_mullo_epi32(g, _mm256_set1_epi32(-26))),
            _mm256_add_epi32(
                _mm256_mullo_epi32(b, _mm256_set1_epi32(287)), c128)), 8);
    r709 = _mm256_min_epi32(_mm256_max_epi32(r709, zero), c255);
    b709 = _mm256_min_epi32(_mm256_max_epi32(b709, zero), c255);

    __m256i cb = _mm256_add_epi32(c128, _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(b709, y), cbs), c128), 8));
    __m256i cr = _mm256_add_epi32(c128, _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(r709, y), crs), c128), 8));

    *cb_out = _mm256_min_epi32(_mm256_max_epi32(cb, cmin), cmax);
    *cr_out = _mm256_min_epi32(_mm256_max_epi32(cr, cmin), cmax);
}

/* -----------------------------------------------------------------------
 * One chunk: 16 chroma samples = 2 rows x 32 luma pixels (full only;
 * ragged edges are handled by the caller with fused_tm_block).
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chunk_avx2(const fused_hdr_internal_t *state,
              const uint16_t *su, const uint16_t *sv,
              const uint16_t *suv, int p010,
              const uint16_t *sy0, const uint16_t *sy1,
              uint8_t *dy0, uint8_t *dy1,
              uint8_t *du, uint8_t *dv)
{
    const uint8_t *lut = state->pq_to_sdr;
    __m256i mask10 = _mm256_set1_epi16(0x3FF);

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

    /* ---- Q15 deltas at chroma rate ---- */
    __m256i dr = tm_delta16_avx2(cr, _mm256_set1_epi32(state->delta_coef_r));
    __m256i db = tm_delta16_avx2(cb, _mm256_set1_epi32(state->delta_coef_b));
    __m256i dg = tm_delta16_g_avx2(cb, cr,
                                   _mm256_set1_epi32(state->delta_coef_g_cb),
                                   _mm256_set1_epi32(state->delta_coef_g_cr));

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
    __m128i re = tm_even_words(r00), re1 = tm_even_words(r01);
    __m128i ge = tm_even_words(g00), ge1 = tm_even_words(g01);
    __m128i be = tm_even_words(b00), be1 = tm_even_words(b01);
    __m128i ye = tm_even_words(y00), ye1 = tm_even_words(y01);

    __m256i cbs  = _mm256_set1_epi32(state->cb_out_scale);
    __m256i crs  = _mm256_set1_epi32(state->cr_out_scale);
    __m256i cmin = _mm256_set1_epi32(state->chroma_out_min);
    __m256i cmax = _mm256_set1_epi32(state->chroma_out_max);

    __m256i cb_lo, cr_lo, cb_hi, cr_hi;
    tm_chroma_dots_avx2(_mm256_cvtepu16_epi32(re), _mm256_cvtepu16_epi32(ge),
                        _mm256_cvtepu16_epi32(be), _mm256_cvtepu16_epi32(ye),
                        cbs, crs, cmin, cmax, &cb_lo, &cr_lo);
    tm_chroma_dots_avx2(_mm256_cvtepu16_epi32(re1), _mm256_cvtepu16_epi32(ge1),
                        _mm256_cvtepu16_epi32(be1), _mm256_cvtepu16_epi32(ye1),
                        cbs, crs, cmin, cmax, &cb_hi, &cr_hi);

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
            tm_chunk_avx2(state,
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
