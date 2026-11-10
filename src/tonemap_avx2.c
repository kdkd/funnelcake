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
 * To keep 16-register AVX2 spill-free and relieve shuffle-port pressure:
 *   1. tm_lut16_avx2 omits the cross-lane vpermq after vpackusdw, leaving
 *      each 16-word segment in [0..3, 8..11 | 4..7, 12..15] order.  All
 *      elementwise luma and even-dword chroma operations commute with this
 *      order (every dword still holds pixel 2i in its low word), and a
 *      single vpermd on packed luma / in-lane unpack on packed chroma
 *      restores natural pixel order once at output store time (saving 12
 *      cross-lane permutes per 32-pixel chunk, plus up to 5 more in chroma
 *      load/expand and 3 in chroma byte packing).
 *   2. Chunks run segment-major (pixels 0..15 of both rows, then pixels
 *      16..31 of both rows) so only 3 chroma-rate delta YMMs stay live
 *      across the first half instead of 6 luma-rate delta YMMs, and each
 *      gather pair packs immediately to 1 YMM before the next gather pair.
 *   3. State coefficients are cached in local scalars before the loop so
 *      uint8_t output stores cannot force aliasing reloads, and any ragged
 *      right-edge strip runs in an outlined pass after the SIMD loop.
 *
 * Everything else follows the scalar reference exactly: Q10 chroma
 * deltas evaluated arithmetically from state->delta_coef_* (the same
 * integer form the scalar tables are generated from), index adds and
 * the luma dot in 16-bit (modular; max value 65408), chroma-rate gamut
 * and encode dots accumulated in 32-bit.  Bit-exactness against
 * fused_tonemap_apply_scalar is enforced by the parity test.
 */

#if defined(__x86_64__)

#include "internal.h"
#include "tonemap.h"

#include <immintrin.h>
#include <stddef.h>
#include <string.h>

#define ALIGN32 __attribute__((aligned(32)))

/* Packed word pair for vpmaddwd: low word first. */
#define TM_PAIR(lo, hi) \
    ((int)(((uint32_t)(uint16_t)(hi) << 16) | (uint16_t)(lo)))

/* vpermd table restoring natural byte order after vpackuswb on two
 * [0..3, 8..11 | 4..7, 12..15]-ordered word vectors. */
static const ALIGN32 int32_t idx_interleave_d[8] = {
    0, 4, 1, 5, 2, 6, 3, 7
};

typedef struct {
    short coef_r;
    short coef_b;
    short coef_g_cr;
    short coef_g_cb;
    int   cbs_pair;
    int   crs_pair;
    int   cmin;
    int   cmax;
} tm_params_avx2_t;

static inline tm_params_avx2_t
tm_load_params_avx2(const fused_hdr_internal_t *state)
{
    int cbs = state->cb_out_scale;
    int crs = state->cr_out_scale;
    tm_params_avx2_t p;
    p.coef_r    = (short)state->delta_coef_r;
    p.coef_b    = (short)state->delta_coef_b;
    p.coef_g_cr = (short)state->delta_coef_g_cr;
    p.coef_g_cb = (short)state->delta_coef_g_cb;
    p.cbs_pair  = TM_PAIR(cbs, -cbs);
    p.crs_pair  = TM_PAIR(crs, -crs);
    p.cmin      = state->chroma_out_min;
    p.cmax      = state->chroma_out_max;
    return p;
}

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
 * Padded byte LUT lookup for 16 word indices -> 16 words in packusdw
 * lane-interleaved order [0..3, 8..11 | 4..7, 12..15].
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) __m256i
tm_lut16_avx2(const uint8_t *lut, __m256i idx_words)
{
    __m256i i0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(idx_words));
    __m256i i1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(idx_words, 1));
    __m256i g0 = _mm256_i32gather_epi32((const int *)lut, i0, 1);
    __m256i g1 = _mm256_i32gather_epi32((const int *)lut, i1, 1);
    __m256i ff = _mm256_set1_epi32(0xFF);
    __m256i res = _mm256_packus_epi32(_mm256_and_si256(g0, ff),
                                      _mm256_and_si256(g1, ff));
    /* Keep the compiler from floating 6-12 gathers ahead of their packusdw,
     * which would hold up to 12 unpacked YMMs live on 16-register AVX2. */
    __asm__("" : "+x"(res));
    return res;
}

static inline __attribute__((always_inline)) __m256i
tm_delta16_avx2(__m256i codes, __m256i coef, __m256i bias)
{
    __m256i x = _mm256_slli_epi16(
        _mm256_sub_epi16(codes, _mm256_set1_epi16(512)), 5);
    return _mm256_add_epi16(_mm256_mulhrs_epi16(x, coef), bias);
}

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

static inline __attribute__((always_inline)) void
tm_seg16_avx2(const uint16_t *sy, __m256i dr, __m256i dg, __m256i db,
              const uint8_t *lut, int keep_rgb,
              __m256i *r_out, __m256i *g_out, __m256i *b_out, __m256i *y_out)
{
    __m256i y = _mm256_and_si256(
        _mm256_loadu_si256((const __m256i *)sy), _mm256_set1_epi16(0x3FF));

    __m256i r = tm_lut16_avx2(lut, _mm256_add_epi16(y, dr));
    __m256i b = tm_lut16_avx2(lut, _mm256_add_epi16(y, db));

    __m256i rb_dot;
    if (!keep_rgb) {
        rb_dot = _mm256_maddubs_epi16(
            _mm256_or_si256(r, _mm256_slli_epi16(b, 8)),
            _mm256_set1_epi16((15 << 8) | 67));
    }

    __m256i g = tm_lut16_avx2(lut, _mm256_add_epi16(y, dg));

    if (keep_rgb) {
        rb_dot = _mm256_maddubs_epi16(
            _mm256_or_si256(r, _mm256_slli_epi16(b, 8)),
            _mm256_set1_epi16((15 << 8) | 67));
    }

    __m256i ys = _mm256_srli_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(rb_dot,
                             _mm256_mullo_epi16(g, _mm256_set1_epi16(174))),
            _mm256_set1_epi16(128)), 8);

    if (keep_rgb) {
        *r_out = r;
        *g_out = g;
        *b_out = b;
    }
    *y_out = ys;
}

static inline __attribute__((always_inline)) void
tm_chroma_dots_avx2(__m256i r, __m256i g, __m256i b, __m256i y,
                    const tm_params_avx2_t *p,
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

    __m256i r709 = _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(rg,   _mm256_set1_epi32(TM_PAIR(425, -150))),
        _mm256_madd_epi16(b128, _mm256_set1_epi32(TM_PAIR(-19, 1)))), 8);
    __m256i b709 = _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(rg,   _mm256_set1_epi32(TM_PAIR(-5, -26))),
        _mm256_madd_epi16(b128, _mm256_set1_epi32(TM_PAIR(287, 1)))), 8);
    r709 = _mm256_min_epi32(_mm256_max_epi32(r709, zero), c255);
    b709 = _mm256_min_epi32(_mm256_max_epi32(b709, zero), c255);

    __m256i cbs_pair = _mm256_set1_epi32(p->cbs_pair);
    __m256i crs_pair = _mm256_set1_epi32(p->crs_pair);
    __m256i cmin     = _mm256_set1_epi32(p->cmin);
    __m256i cmax     = _mm256_set1_epi32(p->cmax);

    __m256i cb = _mm256_add_epi32(c128, _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_or_si256(b709, ye), cbs_pair), c128), 8));
    __m256i cr = _mm256_add_epi32(c128, _mm256_srai_epi32(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_or_si256(r709, ye), crs_pair), c128), 8));

    *cb_out = _mm256_min_epi32(_mm256_max_epi32(cb, cmin), cmax);
    *cr_out = _mm256_min_epi32(_mm256_max_epi32(cr, cmin), cmax);
}

/* -----------------------------------------------------------------------
 * One chunk: 16 chroma samples = 2 rows x 32 luma pixels.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chunk_avx2(const tm_params_avx2_t *p, const uint8_t *lut,
              const uint16_t *su, const uint16_t *sv,
              const uint16_t *suv, int p010,
              const uint16_t *sy0, const uint16_t *sy1,
              uint8_t *dy0, uint8_t *dy1,
              uint8_t *du, uint8_t *dv)
{
    __m256i bias = _mm256_set1_epi16(TM_LUT_BIAS);

    /* ---- chroma load in [0..3, 8..11 | 4..7, 12..15] qword order ---- */
    __m256i cb, cr;
    if (p010) {
        __m256i uv0 = _mm256_loadu_si256((const __m256i *)suv);
        __m256i uv1 = _mm256_loadu_si256((const __m256i *)(suv + 16));
        __m256i m10d = _mm256_set1_epi32(0x3FF);
        __m256i cb0 = _mm256_and_si256(uv0, m10d);
        __m256i cb1 = _mm256_and_si256(uv1, m10d);
        __m256i cr0 = _mm256_and_si256(_mm256_srli_epi32(uv0, 16), m10d);
        __m256i cr1 = _mm256_and_si256(_mm256_srli_epi32(uv1, 16), m10d);
        cb = _mm256_packus_epi32(cb0, cb1);
        cr = _mm256_packus_epi32(cr0, cr1);
    } else {
        __m256i mask10 = _mm256_set1_epi16(0x3FF);
        cb = _mm256_permute4x64_epi64(
            _mm256_and_si256(_mm256_loadu_si256((const __m256i *)su), mask10),
            0xD8);
        cr = _mm256_permute4x64_epi64(
            _mm256_and_si256(_mm256_loadu_si256((const __m256i *)sv), mask10),
            0xD8);
    }

    __m256i dr = tm_delta16_avx2(cr, _mm256_set1_epi16(p->coef_r), bias);
    __m256i db = tm_delta16_avx2(cb, _mm256_set1_epi16(p->coef_b), bias);
    __m256i dg = tm_delta16_g_avx2(
        cb, cr,
        _mm256_set1_epi16(p->coef_g_cb),
        _mm256_set1_epi16(p->coef_g_cr), bias);

    /* ---- segment 0 (pixels 0..15) of both rows ---- */
    __m256i dr0 = _mm256_unpacklo_epi16(dr, dr);
    __m256i dg0 = _mm256_unpacklo_epi16(dg, dg);
    __m256i db0 = _mm256_unpacklo_epi16(db, db);

    __m256i y10;
    tm_seg16_avx2(sy1, dr0, dg0, db0, lut, 0, 0, 0, 0, &y10);

    __m256i r0, g0, b0, y00;
    tm_seg16_avx2(sy0, dr0, dg0, db0, lut, 1, &r0, &g0, &b0, &y00);

    __m256i cb_lo, cr_lo;
    tm_chroma_dots_avx2(r0, g0, b0, y00, p, &cb_lo, &cr_lo);

    /* ---- segment 1 (pixels 16..31) of both rows ---- */
    __m256i dr1 = _mm256_unpackhi_epi16(dr, dr);
    __m256i dg1 = _mm256_unpackhi_epi16(dg, dg);
    __m256i db1 = _mm256_unpackhi_epi16(db, db);

    __m256i y11;
    tm_seg16_avx2(sy1 + 16, dr1, dg1, db1, lut, 0, 0, 0, 0, &y11);

    __m256i perm_d = _mm256_load_si256((const __m256i *)idx_interleave_d);
    _mm256_storeu_si256((__m256i *)dy1,
        _mm256_permutevar8x32_epi32(_mm256_packus_epi16(y10, y11), perm_d));

    __m256i r1, g1, b1, y01;
    tm_seg16_avx2(sy0 + 16, dr1, dg1, db1, lut, 1, &r1, &g1, &b1, &y01);

    perm_d = _mm256_load_si256((const __m256i *)idx_interleave_d);
    _mm256_storeu_si256((__m256i *)dy0,
        _mm256_permutevar8x32_epi32(_mm256_packus_epi16(y00, y01), perm_d));

    __m256i cb_hi, cr_hi;
    tm_chroma_dots_avx2(r1, g1, b1, y01, p, &cb_hi, &cr_hi);

    __m256i cbw = _mm256_packs_epi32(cb_lo, cb_hi);
    __m256i crw = _mm256_packs_epi32(cr_lo, cr_hi);
    __m256i ccr = _mm256_packus_epi16(cbw, crw);
    __m128i ccr_lo = _mm256_castsi256_si128(ccr);
    __m128i ccr_hi = _mm256_extracti128_si256(ccr, 1);

    _mm_storeu_si128((__m128i *)du, _mm_unpacklo_epi16(ccr_lo, ccr_hi));
    _mm_storeu_si128((__m128i *)dv, _mm_unpackhi_epi16(ccr_lo, ccr_hi));
}

static __attribute__((noinline)) void
tm_strip_tail_i010_avx2(const fused_hdr_internal_t *state,
                        const uint16_t *src_y, ptrdiff_t src_y_pitch,
                        const uint16_t *src_u, const uint16_t *src_v,
                        ptrdiff_t src_uv_pitch,
                        uint8_t *dst_y, ptrdiff_t dst_y_stride,
                        uint8_t *dst_u, uint8_t *dst_v,
                        ptrdiff_t dst_uv_stride,
                        int cx_start, int chroma_w, int chroma_h)
{
    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *su  = src_u + cy * src_uv_pitch;
        const uint16_t *sv  = src_v + cy * src_uv_pitch;
        const uint16_t *sy0 = src_y + (ptrdiff_t)(cy * 2) * src_y_pitch;
        const uint16_t *sy1 = sy0 + src_y_pitch;
        uint8_t        *dy0 = dst_y + (ptrdiff_t)(cy * 2) * dst_y_stride;
        uint8_t        *dy1 = dy0 + dst_y_stride;
        uint8_t        *du  = dst_u + cy * dst_uv_stride;
        uint8_t        *dv  = dst_v + cy * dst_uv_stride;
        for (int cx = cx_start; cx < chroma_w; cx++) {
            fused_tm_block(state, su[cx] & 0x3FF, sv[cx] & 0x3FF,
                           sy0 + cx * 2, sy1 + cx * 2,
                           dy0 + cx * 2, dy1 + cx * 2,
                           &du[cx], &dv[cx]);
        }
    }
}

static __attribute__((noinline)) void
tm_strip_tail_p010_avx2(const fused_hdr_internal_t *state,
                        const uint16_t *src_y, ptrdiff_t src_y_pitch,
                        const uint16_t *src_uv, ptrdiff_t src_uv_pitch,
                        uint8_t *dst_y, ptrdiff_t dst_y_stride,
                        uint8_t *dst_u, uint8_t *dst_v,
                        ptrdiff_t dst_uv_stride,
                        int cx_start, int chroma_w, int chroma_h)
{
    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *suv = src_uv + cy * src_uv_pitch;
        const uint16_t *sy0 = src_y + (ptrdiff_t)(cy * 2) * src_y_pitch;
        const uint16_t *sy1 = sy0 + src_y_pitch;
        uint8_t        *dy0 = dst_y + (ptrdiff_t)(cy * 2) * dst_y_stride;
        uint8_t        *dy1 = dy0 + dst_y_stride;
        uint8_t        *du  = dst_u + cy * dst_uv_stride;
        uint8_t        *dv  = dst_v + cy * dst_uv_stride;
        for (int cx = cx_start; cx < chroma_w; cx++) {
            fused_tm_block(state, suv[cx * 2] & 0x3FF, suv[cx * 2 + 1] & 0x3FF,
                           sy0 + cx * 2, sy1 + cx * 2,
                           dy0 + cx * 2, dy1 + cx * 2,
                           &du[cx], &dv[cx]);
        }
    }
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
    if (chroma_h <= 0 || chroma_w <= 0)
        return;

    ptrdiff_t src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    ptrdiff_t src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);
    ptrdiff_t dy_stride    = dst_y_stride;
    ptrdiff_t duv_stride   = dst_uv_stride;
    int       simd_cw      = chroma_w & ~15;

    if (simd_cw > 0) {
        const tm_params_avx2_t p = tm_load_params_avx2(state);
        uint8_t lut_pad[TM_LUT_PAD] __attribute__((aligned(64)));
        tm_build_padded_lut(state->pq_to_sdr, lut_pad);

        const uint16_t *su  = src_u;
        const uint16_t *sv  = src_v;
        const uint16_t *suv = src_uv;
        const uint16_t *sy0 = src_y;
        uint8_t        *dy0 = dst_y;
        uint8_t        *du  = dst_u;
        uint8_t        *dv  = dst_v;

        for (int cy = 0; cy < chroma_h; cy++) {
            const uint16_t *sy1 = sy0 + src_y_pitch;
            uint8_t        *dy1 = dy0 + dy_stride;
            for (int cx = 0; cx < simd_cw; cx += 16) {
                tm_chunk_avx2(&p, lut_pad,
                              p010 ? 0 : su + cx, p010 ? 0 : sv + cx,
                              p010 ? suv + cx * 2 : 0, p010,
                              sy0 + cx * 2, sy1 + cx * 2,
                              dy0 + cx * 2, dy1 + cx * 2,
                              du + cx, dv + cx);
            }
            if (p010) {
                suv += src_uv_pitch;
            } else {
                su += src_uv_pitch;
                sv += src_uv_pitch;
            }
            sy0 += src_y_pitch * 2;
            dy0 += dy_stride * 2;
            du  += duv_stride;
            dv  += duv_stride;
        }
        _mm256_zeroupper();
    }

    if (simd_cw < chroma_w) {
        if (p010) {
            tm_strip_tail_p010_avx2(state, src_y, src_y_pitch,
                                    src_uv, src_uv_pitch,
                                    dst_y, dy_stride,
                                    dst_u, dst_v, duv_stride,
                                    simd_cw, chroma_w, chroma_h);
        } else {
            tm_strip_tail_i010_avx2(state, src_y, src_y_pitch,
                                    src_u, src_v, src_uv_pitch,
                                    dst_y, dy_stride,
                                    dst_u, dst_v, duv_stride,
                                    simd_cw, chroma_w, chroma_h);
        }
    }
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
