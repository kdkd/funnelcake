/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * tonemap_avx512.c - AVX-512 (VBMI) built-in-curve tone mapping.
 *
 * Entry points mirror the scalar reference in tonemap.c:
 *   fused_tonemap_apply_avx512      - planar I010 chroma
 *   fused_tonemap_apply_p010_avx512 - interleaved P010 chroma
 *
 * The scalar pipeline was deliberately restructured so this kernel is
 * one big in-register table lookup:
 *
 *   per chroma sample:  dR, dB, dG = Q15 linear functions of (Cb, Cr)
 *   per luma pixel:     R'G'B' = clamp(Y' + d{R,G,B}) -> pq_to_sdr[1024]
 *                       Y_out  = (67r + 174g + 15b + 128) >> 8
 *   per 2x2 block:      gamut rows + BT.709 encode dots on the top-left
 *
 * The 1024-byte pq_to_sdr table lives in 16 ZMM registers as 8 pairs of
 * 128 bytes.  vpermt2b indexes 128 bytes per op (index bits [6:0]; higher
 * bits are ignored, so the 10-bit indices need no masking), and the top 3
 * index bits pick one of the 8 candidate vectors through a 3-level
 * vpblendmb tree keyed by vptestmb masks: 8 permutes + 3 tests + 7 blends
 * per 64 lookups per channel.
 *
 * The chroma deltas are NOT gathered: they are evaluated arithmetically
 * from the same Q15 coefficients the scalar LUT generator uses
 * (state->delta_coef_*), which is both cheaper than a 16-bit gather and
 * the mechanism that keeps scalar and SIMD bit-exact.  All remaining
 * arithmetic uses the same widths and shifts as the scalar reference:
 * the luma dot in (modular) 16-bit - max value 65408 fits - and the
 * chroma-rate gamut/encode dots in 32-bit.  Parity tests enforce
 * bit-exactness against fused_tonemap_apply_scalar.
 *
 * Tails run the same chunk body under element masks; no scalar tail
 * exists in this file.
 *
 * COMPILE-TIME SELF-STUBBING: same scheme as kernels_avx512.c.  Without
 * the -mavx512* flags this file builds delegations to the scalar
 * reference that dispatch never selects (fused_tonemap_avx512_compiled()
 * returns 0).
 */

#if defined(__x86_64__)

#include "internal.h"
#include "tonemap.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512VL__) && defined(__AVX512VBMI__)

#include <immintrin.h>

#define ALIGN64 __attribute__((aligned(64)))

/* -----------------------------------------------------------------------
 * Permute index tables (generated and verified with node).
 * ----------------------------------------------------------------------- */

/* vpermw: expand 32 chroma-rate words to 64 luma-rate words (each delta
 * serves the 2x1 pair of luma columns above its chroma sample). */
static const ALIGN64 uint16_t idx_dup_lo[32] = {
     0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
     8, 8, 9, 9,10,10,11,11,12,12,13,13,14,14,15,15
};
static const ALIGN64 uint16_t idx_dup_hi[32] = {
    16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23,
    24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31
};

/* vpermb: compact the even bytes (top-left pixel of each 2x2 block) of a
 * 64-byte vector into the low 32 bytes. */
static const ALIGN64 uint8_t idx_even_b[64] = {
     0, 2, 4, 6, 8,10,12,14,16,18,20,22,24,26,28,30,
    32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* vpermt2w: split 64 interleaved UVUV... words into 32 U and 32 V. */
static const ALIGN64 uint16_t idx_even_w[32] = {
     0, 2, 4, 6, 8,10,12,14,16,18,20,22,24,26,28,30,
    32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62
};
static const ALIGN64 uint16_t idx_odd_w[32] = {
     1, 3, 5, 7, 9,11,13,15,17,19,21,23,25,27,29,31,
    33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63
};

static inline __mmask32 tm_mask32(int n)
{
    if (n <= 0)  return 0;
    if (n >= 32) return (__mmask32)~(uint32_t)0;
    return (__mmask32)(((uint32_t)1 << n) - 1);
}

/* -----------------------------------------------------------------------
 * 1024-entry byte LUT lookup for 64 word indices in [0, 1023].
 *
 * idx_lo holds pixels 0..31, idx_hi pixels 32..63.  The table is held in
 * t[16] (8 pairs of 64 bytes).  Returns 64 looked-up bytes in pixel order.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) __m512i
tm_lut_1024(const __m512i *t, __m512i idx_lo, __m512i idx_hi)
{
    /* Byte-pack the indices.  vpmovwb keeps the low 8 bits: bits [6:0]
     * are the in-pair index vpermt2b consumes (bit 7 is ignored by the
     * instruction).  The pair selector is bits [9:7]. */
    __m512i low = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(idx_lo)),
        _mm512_cvtepi16_epi8(idx_hi), 1);
    __m512i sel = _mm512_inserti64x4(
        _mm512_castsi256_si512(
            _mm512_cvtepi16_epi8(_mm512_srli_epi16(idx_lo, 7))),
        _mm512_cvtepi16_epi8(_mm512_srli_epi16(idx_hi, 7)), 1);

    __m512i v0 = _mm512_permutex2var_epi8(t[ 0], low, t[ 1]);
    __m512i v1 = _mm512_permutex2var_epi8(t[ 2], low, t[ 3]);
    __m512i v2 = _mm512_permutex2var_epi8(t[ 4], low, t[ 5]);
    __m512i v3 = _mm512_permutex2var_epi8(t[ 6], low, t[ 7]);
    __m512i v4 = _mm512_permutex2var_epi8(t[ 8], low, t[ 9]);
    __m512i v5 = _mm512_permutex2var_epi8(t[10], low, t[11]);
    __m512i v6 = _mm512_permutex2var_epi8(t[12], low, t[13]);
    __m512i v7 = _mm512_permutex2var_epi8(t[14], low, t[15]);

    __mmask64 b0 = _mm512_test_epi8_mask(sel, _mm512_set1_epi8(1));
    __mmask64 b1 = _mm512_test_epi8_mask(sel, _mm512_set1_epi8(2));
    __mmask64 b2 = _mm512_test_epi8_mask(sel, _mm512_set1_epi8(4));

    __m512i m01 = _mm512_mask_blend_epi8(b0, v0, v1);
    __m512i m23 = _mm512_mask_blend_epi8(b0, v2, v3);
    __m512i m45 = _mm512_mask_blend_epi8(b0, v4, v5);
    __m512i m67 = _mm512_mask_blend_epi8(b0, v6, v7);
    __m512i m03 = _mm512_mask_blend_epi8(b1, m01, m23);
    __m512i m47 = _mm512_mask_blend_epi8(b1, m45, m67);
    return _mm512_mask_blend_epi8(b2, m03, m47);
}

/* -----------------------------------------------------------------------
 * Q15 delta evaluation at chroma rate.
 *
 * x32_{lo,hi} hold (code - 512) for 32 chroma samples as dwords.
 * Returns the 32 deltas packed back to words:
 *   delta = (x * coef + 16384) >> 15   (identical to the scalar tables)
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) __m512i
tm_delta_q15(__m512i x32_lo, __m512i x32_hi, __m512i coef)
{
    __m512i bias = _mm512_set1_epi32(16384);
    __m512i d_lo = _mm512_srai_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(x32_lo, coef), bias), 15);
    __m512i d_hi = _mm512_srai_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(x32_hi, coef), bias), 15);
    return _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi32_epi16(d_lo)),
        _mm512_cvtepi32_epi16(d_hi), 1);
}

/* -----------------------------------------------------------------------
 * One luma row of a chunk: up to 64 pixels.
 *
 * Adds the (already luma-rate-expanded) deltas to Y', clamps, looks up
 * all three channels, computes the BT.2020-coefficient luma dot, and
 * stores the row.  Returns the r/g/b/y byte vectors for the chroma pass.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_row_avx512(const uint16_t *sy, uint8_t *dy,
              __m512i dr0, __m512i dr1,
              __m512i dg0, __m512i dg1,
              __m512i db0, __m512i db1,
              const __m512i *t, int nl, int full,
              __m512i *rb_out, __m512i *gb_out, __m512i *bb_out,
              __m512i *yb_out)
{
    __m512i mask10 = _mm512_set1_epi16(0x3FF);
    __m512i zero   = _mm512_setzero_si512();
    __m512i top    = _mm512_set1_epi16(1023);

    __m512i y_lo, y_hi;
    if (full) {
        y_lo = _mm512_loadu_si512((const void *)sy);
        y_hi = _mm512_loadu_si512((const void *)(sy + 32));
    } else {
        y_lo = _mm512_maskz_loadu_epi16(tm_mask32(nl), sy);
        y_hi = _mm512_maskz_loadu_epi16(tm_mask32(nl - 32), sy + 32);
    }
    y_lo = _mm512_and_si512(y_lo, mask10);
    y_hi = _mm512_and_si512(y_hi, mask10);

    /* R'/G'/B' indices: clamp(Y' + delta, 0, 1023).  The sum fits int16
     * (|delta| <= ~964), matching the scalar int arithmetic. */
    __m512i ir_lo = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_add_epi16(y_lo, dr0), zero), top);
    __m512i ir_hi = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_add_epi16(y_hi, dr1), zero), top);
    __m512i ig_lo = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_add_epi16(y_lo, dg0), zero), top);
    __m512i ig_hi = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_add_epi16(y_hi, dg1), zero), top);
    __m512i ib_lo = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_add_epi16(y_lo, db0), zero), top);
    __m512i ib_hi = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_add_epi16(y_hi, db1), zero), top);

    __m512i rb = tm_lut_1024(t, ir_lo, ir_hi);
    __m512i gb = tm_lut_1024(t, ig_lo, ig_hi);
    __m512i bb = tm_lut_1024(t, ib_lo, ib_hi);

    /* Y_out = (67r + 174g + 15b + 128) >> 8 in modular 16-bit (max value
     * 65408 < 65536; every product fits 16 bits unsigned). */
    __m512i r16_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(rb));
    __m512i r16_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(rb, 1));
    __m512i g16_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(gb));
    __m512i g16_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(gb, 1));
    __m512i b16_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(bb));
    __m512i b16_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(bb, 1));

    __m512i c67  = _mm512_set1_epi16(67);
    __m512i c174 = _mm512_set1_epi16(174);
    __m512i c15  = _mm512_set1_epi16(15);
    __m512i c128 = _mm512_set1_epi16(128);

    __m512i ys_lo = _mm512_srli_epi16(
        _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mullo_epi16(r16_lo, c67),
                             _mm512_mullo_epi16(g16_lo, c174)),
            _mm512_add_epi16(_mm512_mullo_epi16(b16_lo, c15), c128)), 8);
    __m512i ys_hi = _mm512_srli_epi16(
        _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mullo_epi16(r16_hi, c67),
                             _mm512_mullo_epi16(g16_hi, c174)),
            _mm512_add_epi16(_mm512_mullo_epi16(b16_hi, c15), c128)), 8);

    __m512i yb = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi16_epi8(ys_lo)),
        _mm512_cvtepi16_epi8(ys_hi), 1);

    if (full) {
        _mm512_storeu_si512((void *)dy, yb);
    } else {
        _mm256_mask_storeu_epi8(dy, tm_mask32(nl),
                                _mm512_castsi512_si256(yb));
        _mm256_mask_storeu_epi8(dy + 32, tm_mask32(nl - 32),
                                _mm512_extracti64x4_epi64(yb, 1));
    }

    *rb_out = rb;
    *gb_out = gb;
    *bb_out = bb;
    *yb_out = yb;
}

/* -----------------------------------------------------------------------
 * Chroma-rate gamut + BT.709 encode dots for 16 dwords.
 *
 * Mirrors tonemap_rgb_to_chroma_sdr exactly (32-bit arithmetic, same
 * rounding and clamps).
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chroma_dots(__m512i r, __m512i g, __m512i b, __m512i y,
               __m512i cbs, __m512i crs, __m512i cmin, __m512i cmax,
               __m512i *cb_out, __m512i *cr_out)
{
    __m512i c128  = _mm512_set1_epi32(128);
    __m512i zero  = _mm512_setzero_si512();
    __m512i c255  = _mm512_set1_epi32(255);

    /* Gamut rows * 256 (each sums to 256): see tonemap.c */
    __m512i r709 = _mm512_srai_epi32(
        _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_mullo_epi32(r, _mm512_set1_epi32(425)),
                _mm512_mullo_epi32(g, _mm512_set1_epi32(-150))),
            _mm512_add_epi32(
                _mm512_mullo_epi32(b, _mm512_set1_epi32(-19)), c128)), 8);
    __m512i b709 = _mm512_srai_epi32(
        _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_mullo_epi32(r, _mm512_set1_epi32(-5)),
                _mm512_mullo_epi32(g, _mm512_set1_epi32(-26))),
            _mm512_add_epi32(
                _mm512_mullo_epi32(b, _mm512_set1_epi32(287)), c128)), 8);
    r709 = _mm512_min_epi32(_mm512_max_epi32(r709, zero), c255);
    b709 = _mm512_min_epi32(_mm512_max_epi32(b709, zero), c255);

    /* cb = clamp(128 + ((b709 - y)*cbs + 128 >> 8), cmin, cmax) */
    __m512i cb = _mm512_add_epi32(c128, _mm512_srai_epi32(
        _mm512_add_epi32(
            _mm512_mullo_epi32(_mm512_sub_epi32(b709, y), cbs), c128), 8));
    __m512i cr = _mm512_add_epi32(c128, _mm512_srai_epi32(
        _mm512_add_epi32(
            _mm512_mullo_epi32(_mm512_sub_epi32(r709, y), crs), c128), 8));

    *cb_out = _mm512_min_epi32(_mm512_max_epi32(cb, cmin), cmax);
    *cr_out = _mm512_min_epi32(_mm512_max_epi32(cr, cmin), cmax);
}

/* -----------------------------------------------------------------------
 * One chunk: up to 32 chroma samples = 2 rows x 64 luma pixels.
 *
 * p010 and full are compile-time literals at every call site; the masked
 * and interleaved branches fold away.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chunk_avx512(const fused_hdr_internal_t *state,
                const uint16_t *su, const uint16_t *sv,
                const uint16_t *suv, int p010,
                const uint16_t *sy0, const uint16_t *sy1,
                uint8_t *dy0, uint8_t *dy1,
                uint8_t *du, uint8_t *dv,
                const __m512i *t, int nc, int full)
{
    __m512i mask10 = _mm512_set1_epi16(0x3FF);

    /* ---- chroma load ---- */
    __m512i cb, cr;
    if (p010) {
        __m512i uv0, uv1;
        if (full) {
            uv0 = _mm512_loadu_si512((const void *)suv);
            uv1 = _mm512_loadu_si512((const void *)(suv + 32));
        } else {
            int nw = nc * 2;
            uv0 = _mm512_maskz_loadu_epi16(tm_mask32(nw), suv);
            uv1 = _mm512_maskz_loadu_epi16(tm_mask32(nw - 32), suv + 32);
        }
        cb = _mm512_permutex2var_epi16(
            uv0, _mm512_load_si512((const void *)idx_even_w), uv1);
        cr = _mm512_permutex2var_epi16(
            uv0, _mm512_load_si512((const void *)idx_odd_w), uv1);
    } else {
        if (full) {
            cb = _mm512_loadu_si512((const void *)su);
            cr = _mm512_loadu_si512((const void *)sv);
        } else {
            cb = _mm512_maskz_loadu_epi16(tm_mask32(nc), su);
            cr = _mm512_maskz_loadu_epi16(tm_mask32(nc), sv);
        }
    }
    cb = _mm512_and_si512(cb, mask10);
    cr = _mm512_and_si512(cr, mask10);

    /* ---- Q15 deltas at chroma rate ---- */
    __m512i c512 = _mm512_set1_epi32(512);
    __m512i xb_lo = _mm512_sub_epi32(
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(cb)), c512);
    __m512i xb_hi = _mm512_sub_epi32(
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(cb, 1)), c512);
    __m512i xr_lo = _mm512_sub_epi32(
        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(cr)), c512);
    __m512i xr_hi = _mm512_sub_epi32(
        _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(cr, 1)), c512);

    __m512i dr = tm_delta_q15(xr_lo, xr_hi,
                              _mm512_set1_epi32(state->delta_coef_r));
    __m512i db = tm_delta_q15(xb_lo, xb_hi,
                              _mm512_set1_epi32(state->delta_coef_b));
    /* dG = g_cr(Cr) + g_cb(Cb), summed in 32-bit before packing exactly
     * like the scalar int addition of the two table values. */
    __m512i bias = _mm512_set1_epi32(16384);
    __m512i gcr  = _mm512_set1_epi32(state->delta_coef_g_cr);
    __m512i gcb  = _mm512_set1_epi32(state->delta_coef_g_cb);
    __m512i dg_lo = _mm512_add_epi32(
        _mm512_srai_epi32(_mm512_add_epi32(
            _mm512_mullo_epi32(xr_lo, gcr), bias), 15),
        _mm512_srai_epi32(_mm512_add_epi32(
            _mm512_mullo_epi32(xb_lo, gcb), bias), 15));
    __m512i dg_hi = _mm512_add_epi32(
        _mm512_srai_epi32(_mm512_add_epi32(
            _mm512_mullo_epi32(xr_hi, gcr), bias), 15),
        _mm512_srai_epi32(_mm512_add_epi32(
            _mm512_mullo_epi32(xb_hi, gcb), bias), 15));
    __m512i dg = _mm512_inserti64x4(
        _mm512_castsi256_si512(_mm512_cvtepi32_epi16(dg_lo)),
        _mm512_cvtepi32_epi16(dg_hi), 1);

    /* ---- expand deltas to luma rate ---- */
    __m512i dup_lo = _mm512_load_si512((const void *)idx_dup_lo);
    __m512i dup_hi = _mm512_load_si512((const void *)idx_dup_hi);
    __m512i dr0 = _mm512_permutexvar_epi16(dup_lo, dr);
    __m512i dr1 = _mm512_permutexvar_epi16(dup_hi, dr);
    __m512i dg0 = _mm512_permutexvar_epi16(dup_lo, dg);
    __m512i dg1 = _mm512_permutexvar_epi16(dup_hi, dg);
    __m512i db0 = _mm512_permutexvar_epi16(dup_lo, db);
    __m512i db1 = _mm512_permutexvar_epi16(dup_hi, db);

    /* ---- two luma rows ---- */
    int nl = nc * 2;
    __m512i rb0, gb0, bb0, yb0, rbx, gbx, bbx, ybx;
    tm_row_avx512(sy0, dy0, dr0, dr1, dg0, dg1, db0, db1, t, nl, full,
                  &rb0, &gb0, &bb0, &yb0);
    tm_row_avx512(sy1, dy1, dr0, dr1, dg0, dg1, db0, db1, t, nl, full,
                  &rbx, &gbx, &bbx, &ybx);
    (void)rbx; (void)gbx; (void)bbx; (void)ybx;

    /* ---- chroma outputs from row 0's top-left pixels ---- */
    __m512i evb = _mm512_load_si512((const void *)idx_even_b);
    __m256i rq = _mm512_castsi512_si256(_mm512_permutexvar_epi8(evb, rb0));
    __m256i gq = _mm512_castsi512_si256(_mm512_permutexvar_epi8(evb, gb0));
    __m256i bq = _mm512_castsi512_si256(_mm512_permutexvar_epi8(evb, bb0));
    __m256i yq = _mm512_castsi512_si256(_mm512_permutexvar_epi8(evb, yb0));

    __m512i r_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(rq));
    __m512i r_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(rq, 1));
    __m512i g_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(gq));
    __m512i g_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(gq, 1));
    __m512i b_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(bq));
    __m512i b_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(bq, 1));
    __m512i y_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(yq));
    __m512i y_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(yq, 1));

    __m512i cbs  = _mm512_set1_epi32(state->cb_out_scale);
    __m512i crs  = _mm512_set1_epi32(state->cr_out_scale);
    __m512i cmin = _mm512_set1_epi32(state->chroma_out_min);
    __m512i cmax = _mm512_set1_epi32(state->chroma_out_max);

    __m512i cb_lo32, cr_lo32, cb_hi32, cr_hi32;
    tm_chroma_dots(r_lo, g_lo, b_lo, y_lo, cbs, crs, cmin, cmax,
                   &cb_lo32, &cr_lo32);
    tm_chroma_dots(r_hi, g_hi, b_hi, y_hi, cbs, crs, cmin, cmax,
                   &cb_hi32, &cr_hi32);

    __m256i cb_b = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm512_cvtepi32_epi8(cb_lo32)),
        _mm512_cvtepi32_epi8(cb_hi32), 1);
    __m256i cr_b = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm512_cvtepi32_epi8(cr_lo32)),
        _mm512_cvtepi32_epi8(cr_hi32), 1);

    if (full) {
        _mm256_storeu_si256((__m256i *)du, cb_b);
        _mm256_storeu_si256((__m256i *)dv, cr_b);
    } else {
        _mm256_mask_storeu_epi8(du, tm_mask32(nc), cb_b);
        _mm256_mask_storeu_epi8(dv, tm_mask32(nc), cr_b);
    }
}

/* -----------------------------------------------------------------------
 * Frame driver.  p010 is a compile-time literal at both call sites.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_frame_avx512(const fused_hdr_internal_t *state,
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

    /* The whole pq_to_sdr table as 8 vpermt2b source pairs. */
    __m512i t[16];
    for (int i = 0; i < 16; i++)
        t[i] = _mm512_loadu_si512((const void *)(state->pq_to_sdr + 64 * i));

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
        for (; cx + 32 <= chroma_w; cx += 32) {
            tm_chunk_avx512(state,
                            p010 ? 0 : su + cx, p010 ? 0 : sv + cx,
                            p010 ? suv + cx * 2 : 0, p010,
                            sy0 + cx * 2, sy1 + cx * 2,
                            dy0 + cx * 2, dy1 + cx * 2,
                            du + cx, dv + cx, t, 32, 1);
        }
        if (cx < chroma_w) {
            tm_chunk_avx512(state,
                            p010 ? 0 : su + cx, p010 ? 0 : sv + cx,
                            p010 ? suv + cx * 2 : 0, p010,
                            sy0 + cx * 2, sy1 + cx * 2,
                            dy0 + cx * 2, dy1 + cx * 2,
                            du + cx, dv + cx, t, chroma_w - cx, 0);
        }
    }

    _mm256_zeroupper();
}

__attribute__((hot))
void fused_tonemap_apply_avx512(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_avx512(state, src_y, src_y_stride, src_u, src_v,
                    0, src_uv_stride, 0,
                    dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                    width, height);
}

__attribute__((hot))
void fused_tonemap_apply_p010_avx512(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_avx512(state, src_y, src_y_stride, 0, 0,
                    src_uv, src_uv_stride, 1,
                    dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                    width, height);
}

int fused_tonemap_avx512_compiled(void)
{
    return 1;
}

#else /* compiler lacks AVX-512 support - self-stubbed build */

/* Stub build: no -mavx512* flags, no AVX-512 instructions in this object.
 * Dispatch never selects these (fused_tonemap_avx512_compiled() returns
 * 0); the delegations to the scalar reference exist to satisfy the
 * linker and stay correct regardless. */

void fused_tonemap_apply_avx512(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    fused_tonemap_apply_scalar(state, src_y, src_y_stride,
                               src_u, src_uv_stride, src_v,
                               dst_y, dst_y_stride,
                               dst_u, dst_uv_stride, dst_v,
                               width, height);
}

void fused_tonemap_apply_p010_avx512(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    fused_tonemap_apply_p010_scalar(state, src_y, src_y_stride,
                                    src_uv, src_uv_stride,
                                    dst_y, dst_y_stride,
                                    dst_u, dst_uv_stride, dst_v,
                                    width, height);
}

int fused_tonemap_avx512_compiled(void)
{
    return 0;
}

#endif /* AVX-512 feature macros */

#endif /* __x86_64__ */
