/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_upscale_avx512.c - AVX-512 (x86_64) upscale kernels, SDR + HDR.
 *
 * Entry points mirror kernels_upscale_avx2.c:
 *   fused_kernel_upscale_avx512        - upscale-only (SDR)
 *   fused_kernel_thirds_up_avx512      - thirds downscale + upscale (SDR)
 *   fused_kernel_pow2_up_avx512        - pow2 downscale + upscale (SDR)
 *   fused_kernel_upscale_hdr_avx512    - upscale-only (HDR)
 *   fused_kernel_thirds_up_hdr_avx512  - thirds downscale + upscale (HDR)
 *   fused_kernel_pow2_up_hdr_avx512    - pow2 downscale + upscale (HDR)
 *
 * Design:
 *   - 1.5x (2->3) tail upscaling uses native AVX-512 VBMI kernels
 *     (up_1_5x_plane_avx512 for SDR, up_1_5x_plane_avx512_u16 for HDR) that
 *     fuse vertical dual 85/171 blending with horizontal 2->3 VBMI
 *     (vpermt2b / vpermt2w) expansion in a single pass, eliminating the
 *     intermediate scratch row and 128-bit lane-by-lane expansion of AVX2.
 *   - Pure 2x cascade levels are store-bandwidth bound (and full-register
 *     vpermt2b / vpermt2w 2x interleaves add cross-lane permute overhead on
 *     dual-pumped 256-bit AVX-512 implementations), so 2x levels delegate to
 *     the AVX2 2x upscaler.
 *   - P010 upscale-only chroma deinterleaving routes through
 *     fused_deinterleave_p010_avx512 instead of the scalar fallback.
 *
 * Guarded by __AVX512F__/__AVX512BW__/__AVX512VL__/__AVX512VBMI__ with a
 * self-stubbed delegation fallback to AVX2 when compiled without AVX-512 flags.
 */

#if defined(__x86_64__)

#include "internal.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512VL__) && defined(__AVX512VBMI__)

#include "upscale_chunk.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define ALIGN64 __attribute__((aligned(64)))

static inline __mmask64 up_mask64(int n)
{
    return (n >= 64) ? (__mmask64)~(uint64_t)0
                     : (n <= 0 ? 0 : (__mmask64)(((uint64_t)1 << n) - 1));
}

static inline __mmask32 up_mask32(int n)
{
    return (n >= 32) ? (__mmask32)~(uint32_t)0
                     : (n <= 0 ? 0 : (__mmask32)(((uint32_t)1 << n) - 1));
}

/* Shift a 64-byte ZMM right by 1 byte, pulling byte 0 of src2 into byte 63. */
static const uint8_t ALIGN64 shift1_b_tbl[64] = {
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
};

/* Shift a 32-word ZMM right by 1 word, pulling word 0 of src2 into word 31. */
static const uint16_t ALIGN64 shift1_w_tbl[32] = {
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
};

/* 1.5x (2->3) byte interleave from (r, mm = packus_epi16(m1_u16, m2_u16)):
 * 32 input pairs (64B) -> 96 output bytes (64B ZMM + 32B YMM). */
static const uint8_t ALIGN64 up15_b_idx0[64] = {
     0,  64,  72,   2,  65,  73,   4,  66,  74,   6,  67,  75,   8,  68,  76,  10,
    69,  77,  12,  70,  78,  14,  71,  79,  16,  80,  88,  18,  81,  89,  20,  82,
    90,  22,  83,  91,  24,  84,  92,  26,  85,  93,  28,  86,  94,  30,  87,  95,
    32,  96, 104,  34,  97, 105,  36,  98, 106,  38,  99, 107,  40, 100, 108,  42
};
static const uint8_t ALIGN64 up15_b_idx1[64] = {
    101, 109,  44, 102, 110,  46, 103, 111,  48, 112, 120,  50, 113, 121,  52, 114,
    122,  54, 115, 123,  56, 116, 124,  58, 117, 125,  60, 118, 126,  62, 119, 127,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
};

/* 1.5x (2->3) u16 interleave from (r, mm = packus_epi32(m1_u32, m2_u32)):
 * 16 input pairs (32w = 64B) -> 48 output words (32w ZMM + 16w YMM). */
static const uint16_t ALIGN64 up15_w_idx0[32] = {
     0, 32, 36,  2, 33, 37,  4, 34, 38,  6, 35, 39,  8, 40, 44, 10,
    41, 45, 12, 42, 46, 14, 43, 47, 16, 48, 52, 18, 49, 53, 20, 50
};
static const uint16_t ALIGN64 up15_w_idx1[32] = {
    54, 22, 51, 55, 24, 56, 60, 26, 57, 61, 28, 58, 62, 30, 59, 63,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

/* =======================================================================
 * SDR 1.5x (2->3) fused vertical + horizontal upscale (AVX-512 VBMI)
 * ======================================================================= */

/* Vertical 85/171 dual blend sharing bv:
 * out1 = (85*av1 + 171*bv + 128) >> 8
 * out2 = (85*av2 + 171*bv + 128) >> 8 */
static inline __attribute__((always_inline)) void
up_vblend2_21_u8_avx512(__m512i av1, __m512i bv, __m512i av2,
                        __m512i wpair, __m512i half80,
                        __m512i *out1, __m512i *out2)
{
    __m512i t_lo = _mm512_unpacklo_epi8(half80, bv);
    __m512i t_hi = _mm512_unpackhi_epi8(half80, bv);

    __m512i m1_lo = _mm512_maddubs_epi16(_mm512_unpacklo_epi8(av1, bv), wpair);
    __m512i m1_hi = _mm512_maddubs_epi16(_mm512_unpackhi_epi8(av1, bv), wpair);
    *out1 = _mm512_packus_epi16(
        _mm512_srli_epi16(_mm512_add_epi16(m1_lo, t_lo), 8),
        _mm512_srli_epi16(_mm512_add_epi16(m1_hi, t_hi), 8));

    __m512i m2_lo = _mm512_maddubs_epi16(_mm512_unpacklo_epi8(av2, bv), wpair);
    __m512i m2_hi = _mm512_maddubs_epi16(_mm512_unpackhi_epi8(av2, bv), wpair);
    *out2 = _mm512_packus_epi16(
        _mm512_srli_epi16(_mm512_add_epi16(m2_lo, t_lo), 8),
        _mm512_srli_epi16(_mm512_add_epi16(m2_hi, t_hi), 8));
}

/* Horizontal 1.5x emit for 32 pairs (64B -> 96B), full or masked. */
static inline __attribute__((always_inline)) void
up_h_1_5x_emit_u8_avx512(__m512i r, __m512i r_next,
                         __m512i sh1, __m512i idx0, __m512i idx1,
                         __m512i w_ab, __m512i w_bc,
                         __m512i bmask, __m512i half_u16,
                         uint8_t *dst, int full, int out_bytes)
{
    __m512i rn1 = _mm512_permutex2var_epi8(r, sh1, r_next);
    __m512i t   = _mm512_or_si512(_mm512_and_si512(r, bmask), half_u16);

    __m512i m1_u16 = _mm512_srli_epi16(
        _mm512_add_epi16(_mm512_maddubs_epi16(r, w_ab), t), 8);
    __m512i m2_u16 = _mm512_srli_epi16(
        _mm512_add_epi16(_mm512_maddubs_epi16(rn1, w_bc), t), 8);
    __m512i mm = _mm512_packus_epi16(m1_u16, m2_u16);

    __m512i o0 = _mm512_permutex2var_epi8(r, idx0, mm);
    __m256i o1 = _mm512_castsi512_si256(_mm512_permutex2var_epi8(r, idx1, mm));

    if (full) {
        _mm512_storeu_si512((void *)dst, o0);
        _mm256_storeu_si256((__m256i *)(dst + 64), o1);
    } else {
        _mm512_mask_storeu_epi8(dst, up_mask64(out_bytes), o0);
        if (out_bytes > 64) {
            _mm256_mask_storeu_epi8(dst + 64, up_mask32(out_bytes - 64), o1);
        }
    }
}

static void __attribute__((aligned(64), hot))
up_1_5x_plane_avx512(const uint8_t *src, int src_w, int src_h,
                     int src_stride, uint8_t *dst, int dst_stride,
                     uint8_t *scratch)
{
    if (!scratch || src_w < 2 || src_h < 2) return;

    const __m512i sh1      = _mm512_load_si512((const void *)shift1_b_tbl);
    const __m512i idx0     = _mm512_load_si512((const void *)up15_b_idx0);
    const __m512i idx1     = _mm512_load_si512((const void *)up15_b_idx1);
    const __m512i w_ab     = _mm512_set1_epi16((short)0xAB55); /* {85, -85} */
    const __m512i w_bc     = _mm512_set1_epi16((short)((85 << 8) | (-85 & 0xFF)));
    const __m512i bmask    = _mm512_set1_epi16((short)0xFF00);
    const __m512i half_u16 = _mm512_set1_epi16(0x0080);
    const __m512i half80   = _mm512_set1_epi8((char)0x80);

    int pairs_v = src_h / 2;
    int full_chunks = src_w / 64;
    int rem_w = src_w - full_chunks * 64;

    for (int j = 0; j < pairs_v; j++) {
        const uint8_t *r2j  = src + (size_t)(2 * j)     * (size_t)src_stride;
        const uint8_t *r2j1 = r2j + (size_t)src_stride;
        const uint8_t *r2j2 = (2 * j + 2 < src_h) ? r2j1 + (size_t)src_stride : r2j1;

        uint8_t *dst0 = dst + (size_t)(3 * j) * (size_t)dst_stride;
        uint8_t *dst1 = dst0 + (size_t)dst_stride;
        uint8_t *dst2 = dst1 + (size_t)dst_stride;

        __m512i e0 = _mm512_set1_epi8((char)r2j[src_w - 1]);
        __m512i e1 = _mm512_set1_epi8((char)up_blend_21_u8(r2j[src_w - 1], r2j1[src_w - 1]));
        __m512i e2 = _mm512_set1_epi8((char)up_blend_21_u8(r2j2[src_w - 1], r2j1[src_w - 1]));

        if (full_chunks == 0) {
            __mmask64 m_in = up_mask64(rem_w);
            __m512i a0 = _mm512_mask_mov_epi8(e0, m_in, _mm512_maskz_loadu_epi8(m_in, r2j));
            __m512i bv = _mm512_maskz_loadu_epi8(m_in, r2j1);
            __m512i a2 = _mm512_maskz_loadu_epi8(m_in, r2j2);
            __m512i v1, v2;
            up_vblend2_21_u8_avx512(a0, bv, a2, w_ab, half80, &v1, &v2);
            v1 = _mm512_mask_mov_epi8(e1, m_in, v1);
            v2 = _mm512_mask_mov_epi8(e2, m_in, v2);

            int out_bytes = (rem_w / 2) * 3;
            up_h_1_5x_emit_u8_avx512(a0, e0, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst0, 0, out_bytes);
            up_h_1_5x_emit_u8_avx512(v1, e1, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst1, 0, out_bytes);
            up_h_1_5x_emit_u8_avx512(v2, e2, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst2, 0, out_bytes);
        } else {
            __m512i r0_cur = _mm512_loadu_si512((const void *)r2j);
            __m512i v1_cur, v2_cur;
            up_vblend2_21_u8_avx512(
                r0_cur,
                _mm512_loadu_si512((const void *)r2j1),
                _mm512_loadu_si512((const void *)r2j2),
                w_ab, half80, &v1_cur, &v2_cur);

            for (int ci = 1; ci < full_chunks; ci++) {
                size_t sx = (size_t)ci * 64;
                size_t dx = (size_t)(ci - 1) * 96;
                __m512i r0_nxt = _mm512_loadu_si512((const void *)(r2j + sx));
                __m512i v1_nxt, v2_nxt;
                up_vblend2_21_u8_avx512(
                    r0_nxt,
                    _mm512_loadu_si512((const void *)(r2j1 + sx)),
                    _mm512_loadu_si512((const void *)(r2j2 + sx)),
                    w_ab, half80, &v1_nxt, &v2_nxt);

                up_h_1_5x_emit_u8_avx512(r0_cur, r0_nxt, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst0 + dx, 1, 96);
                up_h_1_5x_emit_u8_avx512(v1_cur, v1_nxt, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst1 + dx, 1, 96);
                up_h_1_5x_emit_u8_avx512(v2_cur, v2_nxt, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst2 + dx, 1, 96);

                r0_cur = r0_nxt;
                v1_cur = v1_nxt;
                v2_cur = v2_nxt;
            }

            size_t last_dx = (size_t)(full_chunks - 1) * 96;
            if (rem_w == 0) {
                up_h_1_5x_emit_u8_avx512(r0_cur, e0, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst0 + last_dx, 1, 96);
                up_h_1_5x_emit_u8_avx512(v1_cur, e1, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst1 + last_dx, 1, 96);
                up_h_1_5x_emit_u8_avx512(v2_cur, e2, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst2 + last_dx, 1, 96);
            } else {
                size_t sx = (size_t)full_chunks * 64;
                size_t rem_dx = (size_t)full_chunks * 96;
                __mmask64 m_in = up_mask64(rem_w);
                __m512i r0_rem = _mm512_mask_mov_epi8(e0, m_in, _mm512_maskz_loadu_epi8(m_in, r2j + sx));
                __m512i v1_rem, v2_rem;
                up_vblend2_21_u8_avx512(
                    r0_rem,
                    _mm512_maskz_loadu_epi8(m_in, r2j1 + sx),
                    _mm512_maskz_loadu_epi8(m_in, r2j2 + sx),
                    w_ab, half80, &v1_rem, &v2_rem);
                v1_rem = _mm512_mask_mov_epi8(e1, m_in, v1_rem);
                v2_rem = _mm512_mask_mov_epi8(e2, m_in, v2_rem);

                up_h_1_5x_emit_u8_avx512(r0_cur, r0_rem, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst0 + last_dx, 1, 96);
                up_h_1_5x_emit_u8_avx512(v1_cur, v1_rem, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst1 + last_dx, 1, 96);
                up_h_1_5x_emit_u8_avx512(v2_cur, v2_rem, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst2 + last_dx, 1, 96);

                int out_bytes = (rem_w / 2) * 3;
                up_h_1_5x_emit_u8_avx512(r0_rem, e0, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst0 + rem_dx, 0, out_bytes);
                up_h_1_5x_emit_u8_avx512(v1_rem, e1, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst1 + rem_dx, 0, out_bytes);
                up_h_1_5x_emit_u8_avx512(v2_rem, e2, sh1, idx0, idx1, w_ab, w_bc, bmask, half_u16, dst2 + rem_dx, 0, out_bytes);
            }
        }
    }
}

static void upscale_plane_tail_avx512(const fused_kernel_params_t *p,
                                      const uint8_t *src,
                                      int src_w, int src_h, int src_stride,
                                      int is_chroma)
{
    int N = p->upscale_cascade_depth;

    /* Any preceding 2x levels feed the 1.5x tail, so none of them are terminal. */
    if (N >= 1 && (p->upscale_active & (1u << FUSED_UP_IDX_2X))) {
        uint8_t *dst;
        int dst_stride;
        if (!is_chroma) {
            dst        = p->up_out[FUSED_UP_IDX_2X].plane_y;
            dst_stride = p->up_out[FUSED_UP_IDX_2X].y_stride;
        } else {
            dst        = (is_chroma == 1) ? p->up_out[FUSED_UP_IDX_2X].plane_u
                                          : p->up_out[FUSED_UP_IDX_2X].plane_v;
            dst_stride = p->up_out[FUSED_UP_IDX_2X].uv_stride;
        }
        if (dst) fused_up_2x_plane_avx2(src, src_w, src_h, src_stride,
                                        dst, dst_stride, p->upscale_scratch, 0);
    }

    for (int k = 1; k < N; k++) {
        int up_w = src_w << k;
        int up_h = src_h << k;
        const uint8_t *src_up;
        int src_up_stride;
        uint8_t *dst;
        int dst_stride;

        if (!is_chroma) {
            src_up        = p->up_out[k - 1].plane_y;
            src_up_stride = p->up_out[k - 1].y_stride;
            dst           = p->up_out[k].plane_y;
            dst_stride    = p->up_out[k].y_stride;
        } else {
            src_up        = (is_chroma == 1) ? p->up_out[k - 1].plane_u
                                             : p->up_out[k - 1].plane_v;
            src_up_stride = p->up_out[k - 1].uv_stride;
            dst           = (is_chroma == 1) ? p->up_out[k].plane_u
                                             : p->up_out[k].plane_v;
            dst_stride    = p->up_out[k].uv_stride;
        }
        if (src_up && dst) {
            fused_up_2x_plane_avx2(src_up, up_w, up_h, src_up_stride,
                                   dst, dst_stride, p->upscale_scratch, 0);
        }
    }

    if (p->upscale_active & (1u << FUSED_UP_IDX_TAIL)) {
        const uint8_t *tail_src;
        int tail_src_w, tail_src_h, tail_src_stride;
        uint8_t *dst;
        int dst_stride;

        if (N == 0) {
            tail_src        = src;
            tail_src_w      = src_w;
            tail_src_h      = src_h;
            tail_src_stride = src_stride;
        } else {
            if (!is_chroma) {
                tail_src        = p->up_out[N - 1].plane_y;
                tail_src_stride = p->up_out[N - 1].y_stride;
            } else {
                tail_src        = (is_chroma == 1) ? p->up_out[N - 1].plane_u
                                                   : p->up_out[N - 1].plane_v;
                tail_src_stride = p->up_out[N - 1].uv_stride;
            }
            tail_src_w = src_w << N;
            tail_src_h = src_h << N;
        }

        if (!is_chroma) {
            dst        = p->up_out[FUSED_UP_IDX_TAIL].plane_y;
            dst_stride = p->up_out[FUSED_UP_IDX_TAIL].y_stride;
        } else {
            dst        = (is_chroma == 1) ? p->up_out[FUSED_UP_IDX_TAIL].plane_u
                                          : p->up_out[FUSED_UP_IDX_TAIL].plane_v;
            dst_stride = p->up_out[FUSED_UP_IDX_TAIL].uv_stride;
        }

        if (tail_src && dst) {
            up_1_5x_plane_avx512(tail_src, tail_src_w, tail_src_h, tail_src_stride,
                                 dst, dst_stride, p->upscale_scratch);
        }
    }
}

/* =======================================================================
 * HDR 1.5x (2->3) fused vertical + horizontal upscale (AVX-512)
 * ======================================================================= */

static inline __attribute__((always_inline)) void
up_vblend2_21_u16_avx512(__m512i av1, __m512i bv, __m512i av2,
                         __m512i scale, __m512i *out1, __m512i *out2)
{
    *out1 = _mm512_add_epi16(av1, _mm512_mulhrs_epi16(_mm512_sub_epi16(bv, av1), scale));
    *out2 = _mm512_add_epi16(av2, _mm512_mulhrs_epi16(_mm512_sub_epi16(bv, av2), scale));
}

static inline __attribute__((always_inline)) void
up_h_1_5x_emit_u16_avx512(__m512i r, __m512i r_next,
                          __m512i sh1, __m512i idx0, __m512i idx1,
                          __m512i w_ab, __m512i w_bc, __m512i r128,
                          uint16_t *dst, int full, int out_words)
{
    __m512i rn1    = _mm512_permutex2var_epi16(r, sh1, r_next);
    __m512i m1_u32 = _mm512_srli_epi32(_mm512_add_epi32(_mm512_madd_epi16(r,   w_ab), r128), 8);
    __m512i m2_u32 = _mm512_srli_epi32(_mm512_add_epi32(_mm512_madd_epi16(rn1, w_bc), r128), 8);
    __m512i mm     = _mm512_packus_epi32(m1_u32, m2_u32);

    __m512i o0 = _mm512_permutex2var_epi16(r, idx0, mm);
    __m256i o1 = _mm512_castsi512_si256(_mm512_permutex2var_epi16(r, idx1, mm));

    if (full) {
        _mm512_storeu_si512((void *)dst, o0);
        _mm256_storeu_si256((__m256i *)(dst + 32), o1);
    } else {
        _mm512_mask_storeu_epi16(dst, up_mask32(out_words), o0);
        if (out_words > 32) {
            __mmask16 m16 = (__mmask16)((out_words - 32 >= 16)
                                        ? 0xFFFFu
                                        : ((1u << (out_words - 32)) - 1u));
            _mm256_mask_storeu_epi16(dst + 32, m16, o1);
        }
    }
}

static void __attribute__((aligned(64), hot))
up_1_5x_plane_avx512_u16(const uint16_t *src, int src_w, int src_h,
                         int src_el_stride, uint16_t *dst,
                         int dst_el_stride, uint16_t *scratch)
{
    if (!scratch || src_w < 2 || src_h < 2) return;

    const __m512i sh1   = _mm512_load_si512((const void *)shift1_w_tbl);
    const __m512i idx0  = _mm512_load_si512((const void *)up15_w_idx0);
    const __m512i idx1  = _mm512_load_si512((const void *)up15_w_idx1);
    const __m512i w_ab  = _mm512_set1_epi32((171 << 16) | 85);
    const __m512i w_bc  = _mm512_set1_epi32((85 << 16) | 171);
    const __m512i r128  = _mm512_set1_epi32(128);
    const __m512i scale = _mm512_set1_epi16(171 * 128);

    int pairs_v = src_h / 2;
    int full_chunks = src_w / 32;
    int rem_w = src_w - full_chunks * 32;

    for (int j = 0; j < pairs_v; j++) {
        const uint16_t *r2j  = src + (size_t)(2 * j) * (size_t)src_el_stride;
        const uint16_t *r2j1 = r2j + (size_t)src_el_stride;
        const uint16_t *r2j2 = (2 * j + 2 < src_h) ? r2j1 + (size_t)src_el_stride : r2j1;

        uint16_t *dst0 = dst + (size_t)(3 * j) * (size_t)dst_el_stride;
        uint16_t *dst1 = dst0 + (size_t)dst_el_stride;
        uint16_t *dst2 = dst1 + (size_t)dst_el_stride;

        __m512i e0 = _mm512_set1_epi16((short)r2j[src_w - 1]);
        __m512i e1 = _mm512_set1_epi16((short)up_blend_21_u16(r2j[src_w - 1], r2j1[src_w - 1]));
        __m512i e2 = _mm512_set1_epi16((short)up_blend_21_u16(r2j2[src_w - 1], r2j1[src_w - 1]));

        if (full_chunks == 0) {
            __mmask32 m_in = up_mask32(rem_w);
            __m512i a0 = _mm512_mask_mov_epi16(e0, m_in, _mm512_maskz_loadu_epi16(m_in, r2j));
            __m512i bv = _mm512_maskz_loadu_epi16(m_in, r2j1);
            __m512i a2 = _mm512_maskz_loadu_epi16(m_in, r2j2);
            __m512i v1, v2;
            up_vblend2_21_u16_avx512(a0, bv, a2, scale, &v1, &v2);
            v1 = _mm512_mask_mov_epi16(e1, m_in, v1);
            v2 = _mm512_mask_mov_epi16(e2, m_in, v2);

            int out_words = (rem_w / 2) * 3;
            up_h_1_5x_emit_u16_avx512(a0, e0, sh1, idx0, idx1, w_ab, w_bc, r128, dst0, 0, out_words);
            up_h_1_5x_emit_u16_avx512(v1, e1, sh1, idx0, idx1, w_ab, w_bc, r128, dst1, 0, out_words);
            up_h_1_5x_emit_u16_avx512(v2, e2, sh1, idx0, idx1, w_ab, w_bc, r128, dst2, 0, out_words);
        } else {
            __m512i r0_cur = _mm512_loadu_si512((const void *)r2j);
            __m512i v1_cur, v2_cur;
            up_vblend2_21_u16_avx512(
                r0_cur,
                _mm512_loadu_si512((const void *)r2j1),
                _mm512_loadu_si512((const void *)r2j2),
                scale, &v1_cur, &v2_cur);

            for (int ci = 1; ci < full_chunks; ci++) {
                size_t sx = (size_t)ci * 32;
                size_t dx = (size_t)(ci - 1) * 48;
                __m512i r0_nxt = _mm512_loadu_si512((const void *)(r2j + sx));
                __m512i v1_nxt, v2_nxt;
                up_vblend2_21_u16_avx512(
                    r0_nxt,
                    _mm512_loadu_si512((const void *)(r2j1 + sx)),
                    _mm512_loadu_si512((const void *)(r2j2 + sx)),
                    scale, &v1_nxt, &v2_nxt);

                up_h_1_5x_emit_u16_avx512(r0_cur, r0_nxt, sh1, idx0, idx1, w_ab, w_bc, r128, dst0 + dx, 1, 48);
                up_h_1_5x_emit_u16_avx512(v1_cur, v1_nxt, sh1, idx0, idx1, w_ab, w_bc, r128, dst1 + dx, 1, 48);
                up_h_1_5x_emit_u16_avx512(v2_cur, v2_nxt, sh1, idx0, idx1, w_ab, w_bc, r128, dst2 + dx, 1, 48);

                r0_cur = r0_nxt;
                v1_cur = v1_nxt;
                v2_cur = v2_nxt;
            }

            size_t last_dx = (size_t)(full_chunks - 1) * 48;
            if (rem_w == 0) {
                up_h_1_5x_emit_u16_avx512(r0_cur, e0, sh1, idx0, idx1, w_ab, w_bc, r128, dst0 + last_dx, 1, 48);
                up_h_1_5x_emit_u16_avx512(v1_cur, e1, sh1, idx0, idx1, w_ab, w_bc, r128, dst1 + last_dx, 1, 48);
                up_h_1_5x_emit_u16_avx512(v2_cur, e2, sh1, idx0, idx1, w_ab, w_bc, r128, dst2 + last_dx, 1, 48);
            } else {
                size_t sx = (size_t)full_chunks * 32;
                size_t rem_dx = (size_t)full_chunks * 48;
                __mmask32 m_in = up_mask32(rem_w);
                __m512i r0_rem = _mm512_mask_mov_epi16(e0, m_in, _mm512_maskz_loadu_epi16(m_in, r2j + sx));
                __m512i v1_rem, v2_rem;
                up_vblend2_21_u16_avx512(
                    r0_rem,
                    _mm512_maskz_loadu_epi16(m_in, r2j1 + sx),
                    _mm512_maskz_loadu_epi16(m_in, r2j2 + sx),
                    scale, &v1_rem, &v2_rem);
                v1_rem = _mm512_mask_mov_epi16(e1, m_in, v1_rem);
                v2_rem = _mm512_mask_mov_epi16(e2, m_in, v2_rem);

                up_h_1_5x_emit_u16_avx512(r0_cur, r0_rem, sh1, idx0, idx1, w_ab, w_bc, r128, dst0 + last_dx, 1, 48);
                up_h_1_5x_emit_u16_avx512(v1_cur, v1_rem, sh1, idx0, idx1, w_ab, w_bc, r128, dst1 + last_dx, 1, 48);
                up_h_1_5x_emit_u16_avx512(v2_cur, v2_rem, sh1, idx0, idx1, w_ab, w_bc, r128, dst2 + last_dx, 1, 48);

                int out_words = (rem_w / 2) * 3;
                up_h_1_5x_emit_u16_avx512(r0_rem, e0, sh1, idx0, idx1, w_ab, w_bc, r128, dst0 + rem_dx, 0, out_words);
                up_h_1_5x_emit_u16_avx512(v1_rem, e1, sh1, idx0, idx1, w_ab, w_bc, r128, dst1 + rem_dx, 0, out_words);
                up_h_1_5x_emit_u16_avx512(v2_rem, e2, sh1, idx0, idx1, w_ab, w_bc, r128, dst2 + rem_dx, 0, out_words);
            }
        }
    }
}

static void upscale_plane_hdr_tail_avx512(const fused_hdr_kernel_params_t *p,
                                          const uint16_t *src,
                                          int src_w, int src_h, int src_el_stride,
                                          int is_chroma)
{
    int N = p->upscale_cascade_depth;

    if (N >= 1 && (p->upscale_hdr_active & (1u << FUSED_UP_IDX_2X))) {
        uint16_t *dst;
        int dst_el_stride;
        if (!is_chroma) {
            dst           = p->hdr_up_out[FUSED_UP_IDX_2X].plane_y;
            dst_el_stride = p->hdr_up_out[FUSED_UP_IDX_2X].y_stride / (int)sizeof(uint16_t);
        } else {
            dst           = (is_chroma == 1) ? p->hdr_up_out[FUSED_UP_IDX_2X].plane_u
                                             : p->hdr_up_out[FUSED_UP_IDX_2X].plane_v;
            dst_el_stride = p->hdr_up_out[FUSED_UP_IDX_2X].uv_stride / (int)sizeof(uint16_t);
        }
        if (dst) fused_up_2x_plane_avx2_u16(src, src_w, src_h, src_el_stride,
                                            dst, dst_el_stride,
                                            p->upscale_scratch_hdr, 0);
    }

    for (int k = 1; k < N; k++) {
        int up_w = src_w << k;
        int up_h = src_h << k;
        const uint16_t *src_up;
        int src_up_el_stride;
        uint16_t *dst;
        int dst_el_stride;

        if (!is_chroma) {
            src_up           = p->hdr_up_out[k - 1].plane_y;
            src_up_el_stride = p->hdr_up_out[k - 1].y_stride / (int)sizeof(uint16_t);
            dst              = p->hdr_up_out[k].plane_y;
            dst_el_stride    = p->hdr_up_out[k].y_stride / (int)sizeof(uint16_t);
        } else {
            src_up           = (is_chroma == 1) ? p->hdr_up_out[k - 1].plane_u
                                                : p->hdr_up_out[k - 1].plane_v;
            src_up_el_stride = p->hdr_up_out[k - 1].uv_stride / (int)sizeof(uint16_t);
            dst              = (is_chroma == 1) ? p->hdr_up_out[k].plane_u
                                                : p->hdr_up_out[k].plane_v;
            dst_el_stride    = p->hdr_up_out[k].uv_stride / (int)sizeof(uint16_t);
        }
        if (src_up && dst) {
            fused_up_2x_plane_avx2_u16(src_up, up_w, up_h, src_up_el_stride,
                                       dst, dst_el_stride,
                                       p->upscale_scratch_hdr, 0);
        }
    }

    if (p->upscale_hdr_active & (1u << FUSED_UP_IDX_TAIL)) {
        const uint16_t *tail_src;
        int tail_src_w, tail_src_h, tail_src_el_stride;
        uint16_t *dst;
        int dst_el_stride;

        if (N == 0) {
            tail_src           = src;
            tail_src_w         = src_w;
            tail_src_h         = src_h;
            tail_src_el_stride = src_el_stride;
        } else {
            if (!is_chroma) {
                tail_src           = p->hdr_up_out[N - 1].plane_y;
                tail_src_el_stride = p->hdr_up_out[N - 1].y_stride / (int)sizeof(uint16_t);
            } else {
                tail_src           = (is_chroma == 1) ? p->hdr_up_out[N - 1].plane_u
                                                      : p->hdr_up_out[N - 1].plane_v;
                tail_src_el_stride = p->hdr_up_out[N - 1].uv_stride / (int)sizeof(uint16_t);
            }
            tail_src_w = src_w << N;
            tail_src_h = src_h << N;
        }

        if (!is_chroma) {
            dst           = p->hdr_up_out[FUSED_UP_IDX_TAIL].plane_y;
            dst_el_stride = p->hdr_up_out[FUSED_UP_IDX_TAIL].y_stride / (int)sizeof(uint16_t);
        } else {
            dst           = (is_chroma == 1) ? p->hdr_up_out[FUSED_UP_IDX_TAIL].plane_u
                                             : p->hdr_up_out[FUSED_UP_IDX_TAIL].plane_v;
            dst_el_stride = p->hdr_up_out[FUSED_UP_IDX_TAIL].uv_stride / (int)sizeof(uint16_t);
        }

        if (tail_src && dst) {
            up_1_5x_plane_avx512_u16(tail_src, tail_src_w, tail_src_h,
                                     tail_src_el_stride, dst, dst_el_stride,
                                     p->upscale_scratch_hdr);
        }
    }
}

static void fused_kernel_upscale_hdr_planar_avx512(
    const fused_hdr_kernel_params_t *p,
    const uint16_t *src_y,
    const uint16_t *src_u,
    const uint16_t *src_v,
    int src_uv_el_stride)
{
    if (!p->upscale_tail_1_5x) {
        fused_kernel_upscale_hdr_planar_avx2(
            p, src_y, src_u, src_v, src_uv_el_stride);
        return;
    }
    upscale_plane_hdr_tail_avx512(p, src_y, p->src_width, p->src_height,
                                  p->src_y_el_stride, 0);
    upscale_plane_hdr_tail_avx512(p, src_u, p->src_width / 2, p->src_height / 2,
                                  src_uv_el_stride, 1);
    upscale_plane_hdr_tail_avx512(p, src_v, p->src_width / 2, p->src_height / 2,
                                  src_uv_el_stride, 2);
    _mm256_zeroupper();
}

/* --- Public entry points (native AVX-512) ------------------------------- */

void fused_kernel_upscale_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    if (!p->upscale_tail_1_5x) {
        fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
        return;
    }
    upscale_plane_tail_avx512(p, src_y, p->src_width, p->src_height,
                              p->src_y_stride, 0);
    upscale_plane_tail_avx512(p, src_u, p->src_width / 2, p->src_height / 2,
                              p->src_uv_stride, 1);
    upscale_plane_tail_avx512(p, src_v, p->src_width / 2, p->src_height / 2,
                              p->src_uv_stride, 2);
    _mm256_zeroupper();
}

void fused_kernel_thirds_up_avx512(const fused_kernel_params_t *p,
                                   const uint8_t *src_y,
                                   const uint8_t *src_u,
                                   const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx512(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx512(p, src_y, src_u, src_v);
    }
}

void fused_kernel_upscale_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    const uint16_t *up_src_u = src_u;
    const uint16_t *up_src_v = src_v;
    int up_src_uv_el_stride = p->src_uv_el_stride;

    if (p->is_p010) {
        if (fused_deinterleave_p010_avx512(p, src_u) != 0) return;
        up_src_u = p->p010_tmp_u;
        up_src_v = p->p010_tmp_v;
        up_src_uv_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);
    }

    fused_kernel_upscale_hdr_planar_avx512(
        p, src_y, up_src_u, up_src_v, up_src_uv_el_stride);
}

void fused_kernel_thirds_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                       const uint16_t *src_y,
                                       const uint16_t *src_u,
                                       const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_hdr_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        if (p->active_outputs != 0 && p->is_p010 &&
            p->p010_tmp_u && p->p010_tmp_v) {
            fused_kernel_upscale_hdr_planar_avx512(
                p, src_y, p->p010_tmp_u, p->p010_tmp_v,
                p->p010_tmp_stride / (int)sizeof(uint16_t));
        } else {
            fused_kernel_upscale_hdr_avx512(p, src_y, src_u, src_v);
        }
    }
}

void fused_kernel_pow2_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_hdr_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        if (p->active_outputs != 0 && p->is_p010 &&
            p->p010_tmp_u && p->p010_tmp_v) {
            fused_kernel_upscale_hdr_planar_avx512(
                p, src_y, p->p010_tmp_u, p->p010_tmp_v,
                p->p010_tmp_stride / (int)sizeof(uint16_t));
        } else {
            fused_kernel_upscale_hdr_avx512(p, src_y, src_u, src_v);
        }
    }
}

#else /* Stub delegation when compiled without AVX-512 flags */

void fused_kernel_upscale_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
}

void fused_kernel_thirds_up_avx512(const fused_kernel_params_t *p,
                                   const uint8_t *src_y,
                                   const uint8_t *src_u,
                                   const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_avx512(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
    }
}

void fused_kernel_upscale_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
}

void fused_kernel_thirds_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                       const uint16_t *src_y,
                                       const uint16_t *src_u,
                                       const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_hdr_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        if (p->active_outputs != 0 && p->is_p010 &&
            p->p010_tmp_u && p->p010_tmp_v) {
            fused_kernel_upscale_hdr_planar_avx2(
                p, src_y, p->p010_tmp_u, p->p010_tmp_v,
                p->p010_tmp_stride / (int)sizeof(uint16_t));
        } else {
            fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
        }
    }
}

void fused_kernel_pow2_up_hdr_avx512(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_hdr_avx512(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        if (p->active_outputs != 0 && p->is_p010 &&
            p->p010_tmp_u && p->p010_tmp_v) {
            fused_kernel_upscale_hdr_planar_avx2(
                p, src_y, p->p010_tmp_u, p->p010_tmp_v,
                p->p010_tmp_stride / (int)sizeof(uint16_t));
        } else {
            fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
        }
    }
}

#endif /* AVX-512 feature guard */

#endif /* __x86_64__ */
