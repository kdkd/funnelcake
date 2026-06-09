/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_upscale_avx2.c - AVX2 upscale kernels.
 *
 * Entry points:
 *   fused_kernel_upscale_avx2       - upscale only
 *   fused_kernel_thirds_up_avx2     - combined thirds downscale + upscale
 *   fused_kernel_pow2_up_avx2       - combined pow2 downscale + upscale
 *
 * The upscale primitives are vectorized with SSSE3 / AVX2 intrinsics:
 *   - Horizontal 2x: 128-bit SSE _mm_avg_epu8 + _mm_alignr_epi8 +
 *     unpacklo/hi.  16 source bytes -> 32 destination bytes per iter.
 *     128-bit is used here instead of 256-bit to avoid the expensive
 *     _mm256_permute2x128_si256 that would otherwise be needed to
 *     reorder unpacklo/hi results into linear memory order.  On Zen 1
 *     where permute2x128 is ~1/3 throughput, the 128-bit path is a
 *     clear win; on Zen 2+ and Skylake+ the difference is small because
 *     the workload is memory-bandwidth bound at typical sizes.
 *   - Vertical 2x: _mm256_avg_epu8 to compute the midpoint row (256-bit
 *     because this is a straight linear pipeline with no cross-lane
 *     shuffles, so 256-bit is unambiguously better).
 *   - Horizontal 1.5x (2->3): even/odd deinterleave via _mm_shuffle_epi8,
 *     compute m1/m2 with _mm_mullo_epi16 (sharing 171*b), pre-interleave
 *     m1 and m2 via _mm_unpacklo_epi8, then assemble the 3-way output
 *     via 4 _mm_shuffle_epi8 + 2 OR.  The "c" vector (src[2i+2]) is
 *     produced by loading src+2 and applying the even-deinterleave
 *     mask, which avoids an srli_si128 + insert_epi8 dependency chain
 *     on the fast path.  The last chunk of the row falls back to the
 *     slower srli+insert path to avoid walking past the end.
 *     16 source bytes -> 24 destination bytes per iteration.
 *   - Vertical 1.5x blend row: _mm256_mullo_epi16 with 85/171 weights,
 *     unpack/pack to u8.  32 bytes per iteration.
 *
 * HDR (10-bit) path uses the same structural approach with uint16_t data:
 *   - Horizontal/vertical 2x: _mm_avg_epu16 (128-bit) / _mm256_avg_epu16
 *   - Vertical 1.5x blend: _mm256_madd_epi16 with interleaved (a, b)
 *     pairs and weights {85, 171, ...} for single-instruction u16×u16->u32
 *   - Horizontal 1.5x: 128-bit SSE, 4 pairs per iter, using madd_epi16
 *     and pshufb with u16-aware byte masks to produce 12 dst u16 per iter
 *
 * The combined down+up kernels currently call the existing downscale
 * AVX2 kernel followed by the upscale-only kernel.  The downscale phase
 * touches source memory once and the upscale phase touches it again.
 * Genuine inner-loop fusion (loading source rows once into YMM registers
 * and emitting both directions before moving to the next chunk) is left
 * as a follow-up optimization.
 *
 */

#if defined(__x86_64__)

#include "internal.h"
#include "upscale_chunk.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>


/* -----------------------------------------------------------------------
 * Vectorized 2x horizontal doubling for one row
 * -----------------------------------------------------------------------
 *
 * Reads `src_w` source bytes and writes `2*src_w` destination bytes.
 * Each 32-byte source chunk produces 64 destination bytes:
 *   dst[2i+0] = src[i]
 *   dst[2i+1] = (src[i] + src[i+1] + 1) >> 1   <- _mm256_avg_epu8
 * The rightmost edge replicates src[w-1].
 */
/* aligned(64) keeps the hot loop entry off a 64-byte icache-line
 * boundary — critical on Zen and Intel where the first decode
 * straddling a line can silently regress this path by 10-60%. */
static void __attribute__((aligned(64), hot))
up_h_2x_row_avx2(const uint8_t *src, int src_w, uint8_t *dst)
{
    /* 256-bit AVX2 inner loop.  A 2x horizontal doubling writes two
     * output bytes per source byte, so the binding resource is store
     * throughput: full-width stores halve the store count per output
     * byte compared with an SSE loop.  The shifted source (src[i+1])
     * comes from an overlapping unaligned load - effectively free on
     * cores with fast unaligned loads - instead of register chaining,
     * which keeps the shuffle budget at four ops per 32 source bytes:
     * two unpacks plus the two cross-lane permutes that reassemble the
     * per-lane interleaves into linear byte order. */
    int x = 0;
    int full_chunks = src_w / 32;

    if (full_chunks > 0) {
        for (int c = 0; c + 1 < full_chunks; c++) {
            __m256i r = _mm256_loadu_si256((const __m256i *)(src + x));
            /* Bytes x+1 .. x+32; in bounds because another full chunk
             * follows this one. */
            __m256i r_shifted =
                _mm256_loadu_si256((const __m256i *)(src + x + 1));
            __m256i r_mid = _mm256_avg_epu8(r, r_shifted);

            __m256i ilo = _mm256_unpacklo_epi8(r, r_mid);
            __m256i ihi = _mm256_unpackhi_epi8(r, r_mid);
            __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
            __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

            _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
            _mm256_storeu_si256((__m256i *)(dst + 2 * x + 32), out1);
            x += 32;
        }

        /* Last full chunk: no overlapping load past the row end, so the
         * shifted vector is assembled in-register.  The shifted-in element
         * may be the scalar tail's first source byte or the replicated row
         * edge.  permute2x128 forms [r.hi | edge]; alignr then funnels each
         * lane one byte left, pulling the next lane's (or edge's) first
         * byte into the top slot. */
        __m256i r = _mm256_loadu_si256((const __m256i *)(src + x));
        uint8_t edge = (x + 32 < src_w) ? src[x + 32] : src[src_w - 1];
        __m256i edge_v = _mm256_set1_epi8((char)edge);
        __m256i hi_edge = _mm256_permute2x128_si256(r, edge_v, 0x21);
        __m256i r_shifted = _mm256_alignr_epi8(hi_edge, r, 1);
        __m256i r_mid = _mm256_avg_epu8(r, r_shifted);

        __m256i ilo = _mm256_unpacklo_epi8(r, r_mid);
        __m256i ihi = _mm256_unpackhi_epi8(r, r_mid);
        __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

        _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
        _mm256_storeu_si256((__m256i *)(dst + 2 * x + 32), out1);
        x += 32;
    }

    /* One 16-byte SSE chunk if at least 16 bytes remain, so narrow planes
     * (chroma rows of small sources) stay off the scalar tail. */
    if (x + 16 <= src_w) {
        __m128i r = _mm_loadu_si128((const __m128i *)(src + x));
        uint8_t edge = (x + 16 < src_w) ? src[x + 16] : src[src_w - 1];
        __m128i edge_v = _mm_set1_epi8((char)edge);
        __m128i r_shifted = _mm_alignr_epi8(edge_v, r, 1);
        __m128i r_mid = _mm_avg_epu8(r, r_shifted);

        __m128i out0 = _mm_unpacklo_epi8(r, r_mid);
        __m128i out1 = _mm_unpackhi_epi8(r, r_mid);

        _mm_storeu_si128((__m128i *)(dst + 2 * x +  0), out0);
        _mm_storeu_si128((__m128i *)(dst + 2 * x + 16), out1);
        x += 16;
    }

    /* Tail (less than 16 leftover bytes) - scalar. */
    for (; x < src_w; x++) {
        uint8_t a = src[x];
        uint8_t b = (x + 1 < src_w) ? src[x + 1] : a;
        dst[2 * x + 0] = a;
        dst[2 * x + 1] = up_avg_u8(a, b);
    }
}


/* -----------------------------------------------------------------------
 * Fused vertical 2x average + horizontal doubling for one row.
 *
 * Produces the bilinearly-interpolated ("odd") output row of a 2x upscale.
 * For each 32-byte chunk it computes the vertical average of the two source
 * rows ((a + b + 1) >> 1) in registers and feeds it straight into the
 * horizontal doubling, so the averaged row never has to be written to a
 * scratch buffer and read back.  The horizontal doubling matches
 * up_h_2x_row_avx2, including the overlapping-load trick: the shifted
 * stream is the vertical average of the +1-offset loads, which equals
 * shifting the averaged row because the average is pointwise.
 *
 * aligned(64) mirrors up_h_2x_row_avx2: this 2x path is icache-placement
 * sensitive and the alignment prevents the first iteration's decode from
 * straddling a cache-line boundary.
 * ----------------------------------------------------------------------- */
static void __attribute__((aligned(64), hot))
up_h_2x_vblend_row_avx2(const uint8_t *a_row, const uint8_t *b_row,
                                    int src_w, uint8_t *dst)
{
    int x = 0;
    int full_chunks = src_w / 32;

    if (full_chunks > 0) {
        for (int c = 0; c + 1 < full_chunks; c++) {
            __m256i r = _mm256_avg_epu8(
                _mm256_loadu_si256((const __m256i *)(a_row + x)),
                _mm256_loadu_si256((const __m256i *)(b_row + x)));
            __m256i r_shifted = _mm256_avg_epu8(
                _mm256_loadu_si256((const __m256i *)(a_row + x + 1)),
                _mm256_loadu_si256((const __m256i *)(b_row + x + 1)));
            __m256i r_mid = _mm256_avg_epu8(r, r_shifted);

            __m256i ilo = _mm256_unpacklo_epi8(r, r_mid);
            __m256i ihi = _mm256_unpackhi_epi8(r, r_mid);
            __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
            __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

            _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
            _mm256_storeu_si256((__m256i *)(dst + 2 * x + 32), out1);
            x += 32;
        }

        /* Last full chunk: shifted vector assembled in-register, exactly
         * as in up_h_2x_row_avx2. */
        __m256i r = _mm256_avg_epu8(
            _mm256_loadu_si256((const __m256i *)(a_row + x)),
            _mm256_loadu_si256((const __m256i *)(b_row + x)));
        uint8_t edge = (x + 32 < src_w)
                          ? up_avg_u8(a_row[x + 32], b_row[x + 32])
                          : up_avg_u8(a_row[src_w - 1], b_row[src_w - 1]);
        __m256i edge_v = _mm256_set1_epi8((char)edge);
        __m256i hi_edge = _mm256_permute2x128_si256(r, edge_v, 0x21);
        __m256i r_shifted = _mm256_alignr_epi8(hi_edge, r, 1);
        __m256i r_mid = _mm256_avg_epu8(r, r_shifted);

        __m256i ilo = _mm256_unpacklo_epi8(r, r_mid);
        __m256i ihi = _mm256_unpackhi_epi8(r, r_mid);
        __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

        _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
        _mm256_storeu_si256((__m256i *)(dst + 2 * x + 32), out1);
        x += 32;
    }

    /* One 16-byte SSE chunk if at least 16 bytes remain, so narrow planes
     * (chroma rows of small sources) stay off the scalar tail. */
    if (x + 16 <= src_w) {
        __m128i r = _mm_avg_epu8(
            _mm_loadu_si128((const __m128i *)(a_row + x)),
            _mm_loadu_si128((const __m128i *)(b_row + x)));
        uint8_t edge = (x + 16 < src_w)
                          ? up_avg_u8(a_row[x + 16], b_row[x + 16])
                          : up_avg_u8(a_row[src_w - 1], b_row[src_w - 1]);
        __m128i edge_v = _mm_set1_epi8((char)edge);
        __m128i r_shifted = _mm_alignr_epi8(edge_v, r, 1);
        __m128i r_mid = _mm_avg_epu8(r, r_shifted);

        __m128i out0 = _mm_unpacklo_epi8(r, r_mid);
        __m128i out1 = _mm_unpackhi_epi8(r, r_mid);

        _mm_storeu_si128((__m128i *)(dst + 2 * x +  0), out0);
        _mm_storeu_si128((__m128i *)(dst + 2 * x + 16), out1);
        x += 16;
    }

    /* Tail (less than 16 leftover bytes) - scalar. */
    for (; x < src_w; x++) {
        uint8_t a = up_avg_u8(a_row[x], b_row[x]);
        uint8_t b = (x + 1 < src_w) ? up_avg_u8(a_row[x + 1], b_row[x + 1]) : a;
        dst[2 * x + 0] = a;
        dst[2 * x + 1] = up_avg_u8(a, b);
    }
}


/* -----------------------------------------------------------------------
 * Vertical 2x upscale of one plane (with horizontal doubling fused)
 * ----------------------------------------------------------------------- */

static void up_2x_plane_avx2(const uint8_t *src, int src_w, int src_h,
                             int src_stride, uint8_t *dst, int dst_stride,
                             uint8_t *scratch)
{
    if (!scratch) return;

    for (int i = 0; i < src_h; i++) {
        const uint8_t *row_cur = src + (size_t)i * src_stride;
        const uint8_t *row_nxt = (i + 1 < src_h)
                                    ? src + (size_t)(i + 1) * src_stride
                                    : row_cur;

        /* Even output row: horizontal doubling of the source row directly. */
        up_h_2x_row_avx2(row_cur, src_w,
                         dst + (size_t)(2 * i) * dst_stride);

        /* Odd output row: the vertical average of (row_cur, row_nxt) is
         * computed inline and fused with the horizontal doubling, so it does
         * not need a scratch row between the two steps. */
        up_h_2x_vblend_row_avx2(row_cur, row_nxt, src_w,
                                dst + (size_t)(2 * i + 1) * dst_stride);
    }
}


/* -----------------------------------------------------------------------
 * Vectorized vertical 85/171 blend of two rows (AVX2, 32 bytes at a time)
 *
 * aligned(64) keeps this hot function off icache-line boundaries.
 * The blend is called per output-row pair in the 1.5x cascade and its
 * decode alignment can silently affect that path by several percent.
 * ----------------------------------------------------------------------- */
static inline void __attribute__((aligned(64)))
up_vblend_21_row_avx2(const uint8_t *a_row,
                                         const uint8_t *b_row,
                                         int w, uint8_t *out)
{
    int x = 0;
    /* maddubs decomposition of out = (85*a + 171*b + 128) >> 8, proven
     * bit-identical to the scalar up_blend_21_u8 over all 65536 (a,b) pairs.
     *
     * vpmaddubsw multiplies UNSIGNED bytes by SIGNED bytes and pair-sums into
     * i16.  Direct weights {85,171} would overflow i16 (255*85 + 255*171 =
     * 65280 > 32767), so we use {85,-85}: maddubs(interleave(a,b), {85,-85})
     * = 85*a - 85*b, always in [-21675, 21675] (no saturation).  The missing
     * 256*b + 128 term is built for free by interleaving b with a constant
     * 0x80 low byte: unpack(set1_epi8(0x80), b) = 256*b + 128 per u16 lane.
     * The sum 85*a + 171*b + 128 is always in [128, 65408] < 65536, so the
     * mod-2^16 wrap of vpaddw is exact; >>8 then packus completes it. */
    const __m256i wpair = _mm256_set1_epi16((short)0xAB55); /* bytes {85, -85} */
    const __m256i half  = _mm256_set1_epi8((char)0x80);     /* builds 256*b+128 */

    for (; x + 32 <= w; x += 32) {
        __m256i av = _mm256_loadu_si256((const __m256i *)(a_row + x));
        __m256i bv = _mm256_loadu_si256((const __m256i *)(b_row + x));

        /* Interleave (a,b) per 128-bit lane; maddubs pair-sums each (a_i,b_i)
         * with {85,-85} into 85*(a_i - b_i). */
        __m256i ab_lo = _mm256_unpacklo_epi8(av, bv);
        __m256i ab_hi = _mm256_unpackhi_epi8(av, bv);
        __m256i m_lo  = _mm256_maddubs_epi16(ab_lo, wpair);
        __m256i m_hi  = _mm256_maddubs_epi16(ab_hi, wpair);

        /* Add 256*b + 128 in matching lanes, then logical >>8. */
        __m256i lo = _mm256_srli_epi16(
            _mm256_add_epi16(m_lo, _mm256_unpacklo_epi8(half, bv)), 8);
        __m256i hi = _mm256_srli_epi16(
            _mm256_add_epi16(m_hi, _mm256_unpackhi_epi8(half, bv)), 8);

        /* packus is per-lane; lo/hi came from unpacklo/hi of the same av/bv
         * in matching lanes, so the packed result is linear. */
        __m256i packed = _mm256_packus_epi16(lo, hi);
        _mm256_storeu_si256((__m256i *)(out + x), packed);
    }
    for (; x < w; x++) {
        out[x] = up_blend_21_u8(a_row[x], b_row[x]);
    }
}


/* -----------------------------------------------------------------------
 * Vectorized horizontal 1.5x (2->3) upscale (AVX2)
 * -----------------------------------------------------------------------
 *
 * For each source pair (a, b) with following sample c = src[2i+2], the
 * output triplet is:
 *   dst[3i+0] = a
 *   dst[3i+1] = m1 = (85*a + 171*b + 128) >> 8
 *   dst[3i+2] = m2 = (85*c + 171*b + 128) >> 8
 *
 * The blends are computed directly on the raw interleaved row - no
 * even/odd deinterleave and no u16 widening:
 *   1. maddubs(r, {85,-85}) pair-sums each (a,b) byte pair into
 *      85*a - 85*b, and adding (r & 0xFF00) | 0x80 (which is exactly
 *      256*b + 128, since b is the high byte of each pair) yields
 *      85*a + 171*b + 128 - the same decomposition the vertical blend
 *      up_vblend_21_row_avx2 uses, bit-identical to the scalar blend.
 *   2. the (b,c) pairs for m2 are the same row loaded one byte later
 *      (r_n1 = src + 2p + 1), so maddubs(r_n1, {-85,85}) plus the
 *      shared 256*b + 128 term yields m2.  An overlapping load is
 *      cheaper than building the shifted pairs in-register; only the
 *      bounds-safe slow path for the final chunk shifts in-register.
 *
 * The a bytes pass straight through from the raw load to the output
 * assembly.  SSE has no 3-way interleave store, so the output is
 * assembled via pshufb + OR with precomputed masks; m1 and m2 are
 * packed side by side (packus(m1, m2), m1[i] at byte i and m2[i] at
 * byte 8+i of each lane) and the masks index the halves directly.
 * Per 32 source bytes the whole kernel needs just 7 shuffle-port ops
 * (1 pack, 4 output pshufb, 2 lane extracts) - everything else rides
 * the load, ALU, and store ports.  It is a quiet joy that the 3-way
 * interleave, the expensive part of any 1.5x scaler, shares its
 * pshufb budget with nothing.
 *
 * aligned(64) keeps this hot function off icache-line boundaries —
 * decode alignment of the inner loop can dominate the per-byte cost.
 */
static void __attribute__((aligned(64), hot))
up_h_1_5x_row_avx2(const uint8_t *src, int w, uint8_t *dst)
{
    int pairs = w / 2;
    int full_chunks = pairs / 8;   /* 8 pairs = 16 src bytes = 24 dst bytes */
    int p = 0;

    /* 128-bit fast path reads src[2p .. 2p+16] (r from 2p, r_n1 from
     * 2p+1, both 16 bytes).  We need 2p + 16 < w; the (w - 2) / 16
     * bound (clamped to full_chunks) keeps one byte of margin beyond
     * that.  The last remaining chunk (if any) falls through to the
     * slow path. */
    int safe_chunks = (w > 1) ? (w - 2) / 16 : 0;
    if (safe_chunks > full_chunks) safe_chunks = full_chunks;
    if (safe_chunks < 0)           safe_chunks = 0;

    /* 256-bit fast path processes 16 pairs -> 48 dst bytes per chunk.
     * Reads src[2p .. 2p+32] (r from 2p, r_n1 from 2p+1, both 32
     * bytes).  Need 2p + 32 < w; the (w - 34)/32 bound keeps one byte
     * of margin.  Capped at half of safe_chunks (128-bit) so the
     * 128-bit fast-path residual stays <= 1 chunk. */
    int safe_chunks_256 = (w >= 34) ? ((w - 34) / 32 + 1) : 0;
    if (safe_chunks_256 * 2 > safe_chunks)
        safe_chunks_256 = safe_chunks / 2;

    /* Output interleave masks for the first 16 output bytes.
     *
     * Output position layout for 8 pairs (bytes 0..15 of the stream):
     *   0=a[0], 1=m1[0], 2=m2[0], 3=a[1], 4=m1[1], 5=m2[1],
     *   6=a[2], 7=m1[2], 8=m2[2], 9=a[3], 10=m1[3], 11=m2[3],
     *   12=a[4], 13=m1[4], 14=m2[4], 15=a[5]
     *
     * a[i] sits at byte 2i of the raw load r, so the a masks index r
     * directly.  mm = packus(m1_u16, m2_u16) holds m1[i] at byte i and
     * m2[i] at byte 8+i (of each 128-bit lane). */
    static const int8_t a_lo_bytes[16] = {
        0, -1, -1, 2, -1, -1, 4, -1, -1, 6, -1, -1, 8, -1, -1, 10
    };
    static const int8_t mm_lo_bytes[16] = {
        -1, 0, 8, -1, 1, 9, -1, 2, 10, -1, 3, 11, -1, 4, 12, -1
    };

    /* Interleave masks for the next 8 output bytes (positions 16..23):
     *   16=m1[5], 17=m2[5], 18=a[6], 19=m1[6], 20=m2[6], 21=a[7],
     *   22=m1[7], 23=m2[7] */
    static const int8_t a_hi_bytes[16] = {
        -1, -1, 12, -1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int8_t mm_hi_bytes[16] = {
        5, 13, -1, 6, 14, -1, 7, 15, -1, -1, -1, -1, -1, -1, -1, -1
    };

    const __m128i ma_lo  = _mm_loadu_si128((const __m128i *)a_lo_bytes);
    const __m128i mmm_lo = _mm_loadu_si128((const __m128i *)mm_lo_bytes);
    const __m128i ma_hi  = _mm_loadu_si128((const __m128i *)a_hi_bytes);
    const __m128i mmm_hi = _mm_loadu_si128((const __m128i *)mm_hi_bytes);

    /* maddubs weight pairs: {85,-85} matches the (a,b) layout of r,
     * {-85,85} matches the (b,c) layout of the +1-offset load.  bmask
     * and half build the shared 256*b + 128 term from r. */
    const __m128i w_ab  = _mm_set1_epi16((short)(((-85 & 0xFF) << 8) | 85));
    const __m128i w_bc  = _mm_set1_epi16((short)((85 << 8) | (-85 & 0xFF)));
    const __m128i bmask = _mm_set1_epi16((short)0xFF00);
    const __m128i half  = _mm_set1_epi16(0x0080);

    /* 256-bit mask/constant broadcasts.  vpshufb, vpmaddubsw, vpackuswb
     * and the and/or/add/shift ops are all lane-independent, so the same
     * per-lane mask is broadcast to both 128-bit lanes of the __m256i. */
    const __m256i ma_lo_256  = _mm256_broadcastsi128_si256(ma_lo);
    const __m256i mmm_lo_256 = _mm256_broadcastsi128_si256(mmm_lo);
    const __m256i ma_hi_256  = _mm256_broadcastsi128_si256(ma_hi);
    const __m256i mmm_hi_256 = _mm256_broadcastsi128_si256(mmm_hi);
    const __m256i w_ab_256   = _mm256_broadcastsi128_si256(w_ab);
    const __m256i w_bc_256   = _mm256_broadcastsi128_si256(w_bc);
    const __m256i bmask_256  = _mm256_broadcastsi128_si256(bmask);
    const __m256i half_256   = _mm256_broadcastsi128_si256(half);

    /* --------------------------------------------------------------
     * 256-bit fast path: 16 pairs -> 48 dst bytes per chunk.
     *
     * Each 128-bit lane of the __m256i independently computes its own
     * 8 pairs - lane 0 for pairs [p..p+7] and lane 1 for pairs
     * [p+8..p+15].  Because vpshufb / vpunpcklbw / vpackuswb / vpmullw
     * / vpaddw are all lane-independent, every op in the inner loop
     * covers 2× the bytes of its 128-bit counterpart for ~the same
     * cycle cost, halving per-byte shuffle-port throughput pressure on
     * Zen and Intel alike.
     *
     * The critical side-effect of lane independence is how the output
     * comes out of the kernel: after vpunpcklbw(m1, m2), mm lane 0
     * contains the m1/m2 interleave for pairs [p..p+7] and mm lane 1
     * for pairs [p+8..p+15].  The output assembly masks (ma_lo/hi,
     * mmm_lo/hi), broadcast to both lanes, then build 24 output bytes
     * per lane, which we store as 4× 128-bit pieces (16+8 per lane).
     * -------------------------------------------------------------- */
    for (int ci = 0; ci < safe_chunks_256; ci++) {
        __m256i r    = _mm256_loadu_si256((const __m256i *)(src + 2 * p));
        __m256i r_n1 = _mm256_loadu_si256((const __m256i *)(src + 2 * p + 1));

        /* 256*b + 128, shared by both blends (b is the high byte of
         * each (a,b) pair in r). */
        __m256i t = _mm256_or_si256(_mm256_and_si256(r, bmask_256),
                                    half_256);

        __m256i m1_u16 = _mm256_srli_epi16(
            _mm256_add_epi16(_mm256_maddubs_epi16(r, w_ab_256), t), 8);
        __m256i m2_u16 = _mm256_srli_epi16(
            _mm256_add_epi16(_mm256_maddubs_epi16(r_n1, w_bc_256), t), 8);

        /* Pack the two blend streams side by side: per lane, m1[0..7]
         * in the low 8 bytes, m2[0..7] in the high 8. */
        __m256i mm = _mm256_packus_epi16(m1_u16, m2_u16);

        __m256i out_lo = _mm256_or_si256(
            _mm256_shuffle_epi8(r,  ma_lo_256),
            _mm256_shuffle_epi8(mm, mmm_lo_256));
        __m256i out_hi = _mm256_or_si256(
            _mm256_shuffle_epi8(r,  ma_hi_256),
            _mm256_shuffle_epi8(mm, mmm_hi_256));

        /* Extract each 128-bit lane and store 24 bytes (16 + 8). */
        __m128i ol_lane0 = _mm256_castsi256_si128(out_lo);
        __m128i oh_lane0 = _mm256_castsi256_si128(out_hi);
        __m128i ol_lane1 = _mm256_extracti128_si256(out_lo, 1);
        __m128i oh_lane1 = _mm256_extracti128_si256(out_hi, 1);

        _mm_storeu_si128((__m128i *)(dst + 3 * p),          ol_lane0);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 16),     oh_lane0);
        _mm_storeu_si128((__m128i *)(dst + 3 * p + 24),     ol_lane1);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 40),     oh_lane1);

        p += 16;
    }

    /* 128-bit fast path: handles the residual safe chunk (0 or 1) that
     * the 256-bit loop couldn't fit.  Also handles the whole fast path
     * on CPUs / workloads where the 256-bit loop can't run (w < 34). */
    for (int ci = 2 * safe_chunks_256; ci < safe_chunks; ci++) {
        __m128i r    = _mm_loadu_si128((const __m128i *)(src + 2 * p));
        __m128i r_n1 = _mm_loadu_si128((const __m128i *)(src + 2 * p + 1));

        __m128i t = _mm_or_si128(_mm_and_si128(r, bmask), half);

        __m128i m1_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_maddubs_epi16(r, w_ab), t), 8);
        __m128i m2_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_maddubs_epi16(r_n1, w_bc), t), 8);

        __m128i mm = _mm_packus_epi16(m1_u16, m2_u16);

        __m128i out_lo = _mm_or_si128(
            _mm_shuffle_epi8(r,  ma_lo),
            _mm_shuffle_epi8(mm, mmm_lo));
        __m128i out_hi = _mm_or_si128(
            _mm_shuffle_epi8(r,  ma_hi),
            _mm_shuffle_epi8(mm, mmm_hi));

        _mm_storeu_si128((__m128i *)(dst + 3 * p), out_lo);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 16), out_hi);

        p += 8;
    }

    /* Slow path for the last (unsafe) chunk(s) - builds the (b,c) pairs
     * in-register (shift one byte, insert the next source byte or the
     * replicated edge at the top) so we never load past the row end. */
    for (int ci = safe_chunks; ci < full_chunks; ci++) {
        __m128i r = _mm_loadu_si128((const __m128i *)(src + 2 * p));

        int next_idx = 2 * (p + 8);
        uint8_t next_a = (next_idx < w) ? src[next_idx] : src[w - 1];
        __m128i r_n1 = _mm_insert_epi8(_mm_srli_si128(r, 1), next_a, 15);

        __m128i t = _mm_or_si128(_mm_and_si128(r, bmask), half);

        __m128i m1_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_maddubs_epi16(r, w_ab), t), 8);
        __m128i m2_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_maddubs_epi16(r_n1, w_bc), t), 8);

        __m128i mm = _mm_packus_epi16(m1_u16, m2_u16);

        __m128i out_lo = _mm_or_si128(
            _mm_shuffle_epi8(r,  ma_lo),
            _mm_shuffle_epi8(mm, mmm_lo));
        __m128i out_hi = _mm_or_si128(
            _mm_shuffle_epi8(r,  ma_hi),
            _mm_shuffle_epi8(mm, mmm_hi));

        _mm_storeu_si128((__m128i *)(dst + 3 * p), out_lo);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 16), out_hi);

        p += 8;
    }

    /* Scalar tail for the last (pairs % 8) pairs. */
    for (; p < pairs; p++) {
        uint8_t a = src[2 * p];
        uint8_t b = src[2 * p + 1];
        uint8_t c = (2 * p + 2 < w) ? src[2 * p + 2] : src[w - 1];
        dst[3 * p + 0] = a;
        dst[3 * p + 1] = up_blend_21_u8(a, b);
        dst[3 * p + 2] = up_blend_21_u8(c, b);
    }
}


/* -----------------------------------------------------------------------
 * 1.5x (2->3) plane upscale - fully vectorized (AVX2)
 * ----------------------------------------------------------------------- */
static void up_1_5x_plane_avx2(const uint8_t *src, int src_w, int src_h,
                               int src_stride, uint8_t *dst, int dst_stride,
                               uint8_t *scratch)
{
    if (!scratch) return;

    int pairs_v = src_h / 2;
    for (int j = 0; j < pairs_v; j++) {
        const uint8_t *r2j  = src + (size_t)(2 * j)     * src_stride;
        const uint8_t *r2j1 = src + (size_t)(2 * j + 1) * src_stride;
        const uint8_t *r2j2 = (2 * j + 2 < src_h)
                                    ? src + (size_t)(2 * j + 2) * src_stride
                                    : r2j1;

        up_h_1_5x_row_avx2(r2j, src_w,
                           dst + (size_t)(3 * j + 0) * dst_stride);

        up_vblend_21_row_avx2(r2j, r2j1, src_w, scratch);
        up_h_1_5x_row_avx2(scratch, src_w,
                           dst + (size_t)(3 * j + 1) * dst_stride);

        up_vblend_21_row_avx2(r2j2, r2j1, src_w, scratch);
        up_h_1_5x_row_avx2(scratch, src_w,
                           dst + (size_t)(3 * j + 2) * dst_stride);
    }
}


/* -----------------------------------------------------------------------
 * Driver: upscale one plane through the cascade + tail
 * ----------------------------------------------------------------------- */

static void upscale_plane_avx2(const fused_kernel_params_t *p,
                               const uint8_t *src,
                               int src_w, int src_h, int src_stride,
                               int is_chroma)
{
    int N    = p->upscale_cascade_depth;
    int tail = p->upscale_tail_1_5x;

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
        if (dst) up_2x_plane_avx2(src, src_w, src_h, src_stride, dst, dst_stride,
                                   p->upscale_scratch);
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
            up_2x_plane_avx2(src_up, up_w, up_h, src_up_stride, dst, dst_stride,
                             p->upscale_scratch);
        }
    }

    if (tail && (p->upscale_active & (1u << FUSED_UP_IDX_TAIL))) {
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
            up_1_5x_plane_avx2(tail_src, tail_src_w, tail_src_h, tail_src_stride,
                               dst, dst_stride, p->upscale_scratch);
        }
    }
}


/* -----------------------------------------------------------------------
 * Public entry points - SDR
 * ----------------------------------------------------------------------- */

void fused_kernel_upscale_avx2(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v)
{
    upscale_plane_avx2(p, src_y, p->src_width, p->src_height,
                       p->src_y_stride, 0);
    upscale_plane_avx2(p, src_u, p->src_width / 2, p->src_height / 2,
                       p->src_uv_stride, 1);
    upscale_plane_avx2(p, src_v, p->src_width / 2, p->src_height / 2,
                       p->src_uv_stride, 2);
    _mm256_zeroupper();
}

void fused_kernel_thirds_up_avx2(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    /* TODO: integrate upscale into the per-chunk loop of the existing
     * thirds AVX2 kernel for true single-pass over source. */
    if (p->active_outputs != 0) {
        fused_kernel_thirds_avx2(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_avx2(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_avx2(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_avx2(p, src_y, src_u, src_v);
    }
}


/* =======================================================================
 * HDR (10-bit) AVX2 upscale
 * =======================================================================
 *
 * Same structure as the SDR path but operating on uint16_t planes.  The
 * 85/171 blend uses _mm_madd_epi16 with interleaved (a, b) pairs - this
 * handles the u16×u16->u32 widening in a single instruction.  The
 * vertical 2x primitive uses _mm256_avg_epu16, and the horizontal
 * primitives use 128-bit SSE for the same "avoid permute2x128" reasons
 * as the SDR path.
 */

static inline uint16_t up_avg_u16_scalar(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a + (uint32_t)b + 1) >> 1);
}

static inline uint16_t up_blend_21_u16_scalar(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a * 85 + (uint32_t)b * 171 + 128) >> 8);
}

/* ---- Horizontal 2x upscale of one HDR row (256-bit AVX2) ----
 *
 * Same structure as the SDR up_h_2x_row_avx2: full-width stores halve the
 * store count per output sample (the binding resource for a doubling
 * kernel), the shifted source comes from an overlapping unaligned load,
 * and two cross-lane permutes reassemble the per-lane u16 interleaves
 * into linear order.
 *
 * Explicit 64-byte alignment keeps the hot loop entry out of the tail end
 * of a 64-byte icache line.  We observed this path being very sensitive to
 * function placement: when the HDR downscale kernel's text size changes,
 * the unrelated HDR 2x upscale can regress by 10-60% simply because the
 * first iteration's decode straddles an icache-line boundary. */
static void __attribute__((aligned(64), hot))
up_h_2x_row_avx2_u16(const uint16_t *src, int src_w, uint16_t *dst)
{
    int x = 0;
    int full_chunks = src_w / 16;

    if (full_chunks > 0) {
        for (int c = 0; c + 1 < full_chunks; c++) {
            __m256i r = _mm256_loadu_si256((const __m256i *)(src + x));
            /* Samples x+1 .. x+16; in bounds because another full chunk
             * follows this one. */
            __m256i r_shifted =
                _mm256_loadu_si256((const __m256i *)(src + x + 1));
            __m256i r_mid = _mm256_avg_epu16(r, r_shifted);

            __m256i ilo = _mm256_unpacklo_epi16(r, r_mid);
            __m256i ihi = _mm256_unpackhi_epi16(r, r_mid);
            __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
            __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

            _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
            _mm256_storeu_si256((__m256i *)(dst + 2 * x + 16), out1);
            x += 16;
        }

        /* Last full chunk: no overlapping load past the row end; the
         * shifted vector is assembled in-register.  The shifted-in element
         * may be the scalar tail's first source sample or the replicated
         * row edge. */
        __m256i r = _mm256_loadu_si256((const __m256i *)(src + x));
        uint16_t edge = (x + 16 < src_w) ? src[x + 16] : src[src_w - 1];
        __m256i edge_v = _mm256_set1_epi16((short)edge);
        __m256i hi_edge = _mm256_permute2x128_si256(r, edge_v, 0x21);
        __m256i r_shifted = _mm256_alignr_epi8(hi_edge, r, 2);
        __m256i r_mid = _mm256_avg_epu16(r, r_shifted);

        __m256i ilo = _mm256_unpacklo_epi16(r, r_mid);
        __m256i ihi = _mm256_unpackhi_epi16(r, r_mid);
        __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

        _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
        _mm256_storeu_si256((__m256i *)(dst + 2 * x + 16), out1);
        x += 16;
    }

    /* One 8-sample SSE chunk if at least 8 samples remain, so narrow
     * planes (chroma rows of small sources) stay off the scalar tail. */
    if (x + 8 <= src_w) {
        __m128i r = _mm_loadu_si128((const __m128i *)(src + x));
        uint16_t edge = (x + 8 < src_w) ? src[x + 8] : src[src_w - 1];
        __m128i edge_v = _mm_set1_epi16((short)edge);
        __m128i r_shifted = _mm_alignr_epi8(edge_v, r, 2);
        __m128i r_mid = _mm_avg_epu16(r, r_shifted);

        __m128i out0 = _mm_unpacklo_epi16(r, r_mid);
        __m128i out1 = _mm_unpackhi_epi16(r, r_mid);

        _mm_storeu_si128((__m128i *)(dst + 2 * x + 0), out0);
        _mm_storeu_si128((__m128i *)(dst + 2 * x + 8), out1);
        x += 8;
    }

    /* Scalar tail */
    for (; x < src_w; x++) {
        uint16_t a = src[x];
        uint16_t b = (x + 1 < src_w) ? src[x + 1] : a;
        dst[2 * x + 0] = a;
        dst[2 * x + 1] = up_avg_u16_scalar(a, b);
    }
}

/* -----------------------------------------------------------------------
 * Fused vertical 2x average + horizontal doubling for one HDR row.
 *
 * Produces the bilinearly-interpolated ("odd") output row of a 2x HDR
 * upscale.  For each 16-element chunk it computes the vertical average of
 * the two source rows ((a + b + 1) >> 1) in registers and feeds it straight
 * into the horizontal doubling, so the averaged row never has to be written
 * to a scratch buffer and read back.  The horizontal doubling matches
 * up_h_2x_row_avx2_u16, including the overlapping-load trick: the shifted
 * stream is the vertical average of the +1-offset loads, which equals
 * shifting the averaged row because the average is pointwise.  aligned(64)
 * mirrors up_h_2x_row_avx2_u16: this HDR 2x path is icache-placement
 * sensitive.
 * ----------------------------------------------------------------------- */
static void __attribute__((aligned(64), hot))
up_h_2x_vblend_row_avx2_u16(const uint16_t *a_row, const uint16_t *b_row,
                            int src_w, uint16_t *dst)
{
    int x = 0;
    int full_chunks = src_w / 16;

    if (full_chunks > 0) {
        for (int c = 0; c + 1 < full_chunks; c++) {
            __m256i r = _mm256_avg_epu16(
                _mm256_loadu_si256((const __m256i *)(a_row + x)),
                _mm256_loadu_si256((const __m256i *)(b_row + x)));
            __m256i r_shifted = _mm256_avg_epu16(
                _mm256_loadu_si256((const __m256i *)(a_row + x + 1)),
                _mm256_loadu_si256((const __m256i *)(b_row + x + 1)));
            __m256i r_mid = _mm256_avg_epu16(r, r_shifted);

            __m256i ilo = _mm256_unpacklo_epi16(r, r_mid);
            __m256i ihi = _mm256_unpackhi_epi16(r, r_mid);
            __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
            __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

            _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
            _mm256_storeu_si256((__m256i *)(dst + 2 * x + 16), out1);
            x += 16;
        }

        /* Last full chunk: shifted vector assembled in-register, exactly
         * as in up_h_2x_row_avx2_u16. */
        __m256i r = _mm256_avg_epu16(
            _mm256_loadu_si256((const __m256i *)(a_row + x)),
            _mm256_loadu_si256((const __m256i *)(b_row + x)));
        uint16_t edge = (x + 16 < src_w)
                          ? up_avg_u16_scalar(a_row[x + 16], b_row[x + 16])
                          : up_avg_u16_scalar(a_row[src_w - 1], b_row[src_w - 1]);
        __m256i edge_v = _mm256_set1_epi16((short)edge);
        __m256i hi_edge = _mm256_permute2x128_si256(r, edge_v, 0x21);
        __m256i r_shifted = _mm256_alignr_epi8(hi_edge, r, 2);
        __m256i r_mid = _mm256_avg_epu16(r, r_shifted);

        __m256i ilo = _mm256_unpacklo_epi16(r, r_mid);
        __m256i ihi = _mm256_unpackhi_epi16(r, r_mid);
        __m256i out0 = _mm256_permute2x128_si256(ilo, ihi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(ilo, ihi, 0x31);

        _mm256_storeu_si256((__m256i *)(dst + 2 * x +  0), out0);
        _mm256_storeu_si256((__m256i *)(dst + 2 * x + 16), out1);
        x += 16;
    }

    /* One 8-sample SSE chunk if at least 8 samples remain, so narrow
     * planes (chroma rows of small sources) stay off the scalar tail. */
    if (x + 8 <= src_w) {
        __m128i r = _mm_avg_epu16(
            _mm_loadu_si128((const __m128i *)(a_row + x)),
            _mm_loadu_si128((const __m128i *)(b_row + x)));
        uint16_t edge = (x + 8 < src_w)
                          ? up_avg_u16_scalar(a_row[x + 8], b_row[x + 8])
                          : up_avg_u16_scalar(a_row[src_w - 1], b_row[src_w - 1]);
        __m128i edge_v = _mm_set1_epi16((short)edge);
        __m128i r_shifted = _mm_alignr_epi8(edge_v, r, 2);
        __m128i r_mid = _mm_avg_epu16(r, r_shifted);

        __m128i out0 = _mm_unpacklo_epi16(r, r_mid);
        __m128i out1 = _mm_unpackhi_epi16(r, r_mid);

        _mm_storeu_si128((__m128i *)(dst + 2 * x + 0), out0);
        _mm_storeu_si128((__m128i *)(dst + 2 * x + 8), out1);
        x += 8;
    }

    /* Scalar tail */
    for (; x < src_w; x++) {
        uint16_t a = up_avg_u16_scalar(a_row[x], b_row[x]);
        uint16_t b = (x + 1 < src_w) ? up_avg_u16_scalar(a_row[x + 1], b_row[x + 1])
                                     : a;
        dst[2 * x + 0] = a;
        dst[2 * x + 1] = up_avg_u16_scalar(a, b);
    }
}

/* ---- Vertical 2x upscale of one HDR plane ---- */
static void up_2x_plane_avx2_u16(const uint16_t *src, int src_w, int src_h,
                                 int src_el_stride, uint16_t *dst,
                                 int dst_el_stride, uint16_t *scratch)
{
    if (!scratch) return;

    for (int i = 0; i < src_h; i++) {
        const uint16_t *row_cur = src + (size_t)i * src_el_stride;
        const uint16_t *row_nxt = (i + 1 < src_h)
                                    ? src + (size_t)(i + 1) * src_el_stride
                                    : row_cur;

        /* Even output row: horizontal doubling of the source row directly. */
        up_h_2x_row_avx2_u16(row_cur, src_w,
                             dst + (size_t)(2 * i)     * dst_el_stride);

        /* Odd output row: the vertical average of (row_cur, row_nxt) is
         * computed inline and fused with the horizontal doubling, so it does
         * not need a scratch row between the two steps. */
        up_h_2x_vblend_row_avx2_u16(row_cur, row_nxt, src_w,
                                    dst + (size_t)(2 * i + 1) * dst_el_stride);
    }
}

/* ---- Vertical 85/171 blend of two HDR rows ----
 *
 * Uses _mm256_madd_epi16(interleaved_ab, {85,171,85,171,...}) to compute
 * 85*a + 171*b at 32-bit precision in a single instruction per 8 pairs. */
static inline void up_vblend_21_row_avx2_u16(const uint16_t *a_row,
                                             const uint16_t *b_row,
                                             int w, uint16_t *out)
{
    int x = 0;
    const __m256i weights = _mm256_set1_epi32((171 << 16) | 85);
    const __m256i r128    = _mm256_set1_epi32(128);

    for (; x + 16 <= w; x += 16) {
        __m256i av = _mm256_loadu_si256((const __m256i *)(a_row + x));
        __m256i bv = _mm256_loadu_si256((const __m256i *)(b_row + x));

        /* Interleave a and b at u16 granularity.  unpacklo/hi are per-lane
         * and so is madd_epi16 and packus_epi32, so the per-lane pipeline
         * stays consistent and no cross-lane permute is needed. */
        __m256i ab_lo = _mm256_unpacklo_epi16(av, bv);
        __m256i ab_hi = _mm256_unpackhi_epi16(av, bv);

        /* madd_epi16 pair-sums adjacent i16 products into i32 results. */
        __m256i m_lo = _mm256_madd_epi16(ab_lo, weights);
        __m256i m_hi = _mm256_madd_epi16(ab_hi, weights);

        m_lo = _mm256_srli_epi32(_mm256_add_epi32(m_lo, r128), 8);
        m_hi = _mm256_srli_epi32(_mm256_add_epi32(m_hi, r128), 8);

        /* Pack i32 -> u16.  packus_epi32 is per-lane, matching the
         * per-lane unpacklo/hi above -> linear result. */
        __m256i packed = _mm256_packus_epi32(m_lo, m_hi);
        _mm256_storeu_si256((__m256i *)(out + x), packed);
    }

    for (; x < w; x++) {
        out[x] = up_blend_21_u16_scalar(a_row[x], b_row[x]);
    }
}

/* ---- Horizontal 1.5x (2->3) HDR upscale of one row (AVX2, 256-bit
 *      fast path + 128-bit slow-path residual) ----
 *
 * Same raw-pair strategy as the SDR up_h_1_5x_row_avx2, and even
 * simpler: both blend weights fit in a signed 16-bit lane, so
 * madd_epi16 computes 85*a + 171*b straight from the interleaved
 * (a,b) u16 pairs of the raw load - no deinterleave, no decomposition.
 * The (b,c) pairs come from the same row loaded one element later,
 * weighted {171,85}.  m1/m2 are packed side by side (packus_epi32)
 * and the output assembly indexes the raw load and the packed blends
 * directly. */
static void up_h_1_5x_row_avx2_u16(const uint16_t *src, int w, uint16_t *dst)
{
    int pairs = w / 2;
    int full_chunks = pairs / 4;   /* 4 pairs = 8 src u16 = 16 src bytes =
                                    * 12 dst u16 = 24 dst bytes per iter */
    int p = 0;

    /* 256-bit fast path: 8 pairs -> 24 dst u16 per chunk.  Reads 16 u16
     * source elements starting at 2p (32 bytes) into r, and another 16
     * starting at 2p+1 into r_n1 - last element accessed is at index
     * 2p + 16.  The (w - 18)/16 bound keeps one element of margin.  The
     * residual falls through to the 128-bit loop (which has its own
     * bounds-safe srli+insert slow path). */
    int safe_chunks_256 = (w >= 18) ? ((w - 18) / 16 + 1) : 0;
    if (safe_chunks_256 * 2 > full_chunks) safe_chunks_256 = full_chunks / 2;

    /* Byte-level shuffle masks - they operate on registers holding u16
     * values (2 bytes each). */

    /* Output u16 positions 0..7: a[0], m1[0], m2[0], a[1], m1[1], m2[1],
     *   a[2], m1[2]
     * a[i] sits at bytes 4i,4i+1 of the raw load r.
     * mm = packus_epi32(m1_u32, m2_u32) holds m1[i] at bytes 2i,2i+1 and
     * m2[i] at bytes 8+2i,9+2i (of each 128-bit lane). */
    static const int8_t a_lo_u16_mask[16] = {
        0, 1, -1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 8, 9, -1, -1
    };
    static const int8_t mm_lo_u16_mask[16] = {
        -1, -1, 0, 1, 8, 9, -1, -1, 2, 3, 10, 11, -1, -1, 4, 5
    };

    /* Next 4 output u16: m2[2], a[3], m1[3], m2[3] */
    static const int8_t a_hi_u16_mask[16] = {
        -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int8_t mm_hi_u16_mask[16] = {
        12, 13, -1, -1, 6, 7, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1
    };

    const __m128i ma_lo    = _mm_loadu_si128((const __m128i *)a_lo_u16_mask);
    const __m128i mmm_lo   = _mm_loadu_si128((const __m128i *)mm_lo_u16_mask);
    const __m128i ma_hi    = _mm_loadu_si128((const __m128i *)a_hi_u16_mask);
    const __m128i mmm_hi   = _mm_loadu_si128((const __m128i *)mm_hi_u16_mask);

    /* {85, 171, ...} matches the (a,b) layout of r; {171, 85, ...}
     * matches the (b,c) layout of the +1-offset load. */
    const __m128i w_ab     = _mm_set1_epi32((171 << 16) | 85);
    const __m128i w_bc     = _mm_set1_epi32((85 << 16) | 171);
    const __m128i r128_32  = _mm_set1_epi32(128);

    /* 256-bit broadcasts of the same masks/weights.  vpshufb, vpmaddwd,
     * vpackusdw and the add/shift ops we use are all lane-independent,
     * so each 128-bit lane independently computes its own 4 pairs. */
    const __m256i ma_lo_256    = _mm256_broadcastsi128_si256(ma_lo);
    const __m256i mmm_lo_256   = _mm256_broadcastsi128_si256(mmm_lo);
    const __m256i ma_hi_256    = _mm256_broadcastsi128_si256(ma_hi);
    const __m256i mmm_hi_256   = _mm256_broadcastsi128_si256(mmm_hi);
    const __m256i w_ab_256     = _mm256_set1_epi32((171 << 16) | 85);
    const __m256i w_bc_256     = _mm256_set1_epi32((85 << 16) | 171);
    const __m256i r128_32_256  = _mm256_set1_epi32(128);

    /* --------------------------------------------------------------
     * 256-bit HDR fast path.  Lane 0 handles 4 pairs [p..p+3], lane 1
     * handles [p+4..p+7].
     * -------------------------------------------------------------- */
    for (int ci = 0; ci < safe_chunks_256; ci++) {
        __m256i r    = _mm256_loadu_si256((const __m256i *)(src + 2 * p));
        __m256i r_n1 = _mm256_loadu_si256((const __m256i *)(src + 2 * p + 1));

        __m256i m1_u32 = _mm256_srli_epi32(_mm256_add_epi32(
            _mm256_madd_epi16(r, w_ab_256), r128_32_256), 8);
        __m256i m2_u32 = _mm256_srli_epi32(_mm256_add_epi32(
            _mm256_madd_epi16(r_n1, w_bc_256), r128_32_256), 8);

        /* Pack the two blend streams side by side: per lane, m1[0..3]
         * in the low 8 bytes, m2[0..3] in the high 8. */
        __m256i mm = _mm256_packus_epi32(m1_u32, m2_u32);

        __m256i out_lo = _mm256_or_si256(
            _mm256_shuffle_epi8(r,  ma_lo_256),
            _mm256_shuffle_epi8(mm, mmm_lo_256));
        __m256i out_hi = _mm256_or_si256(
            _mm256_shuffle_epi8(r,  ma_hi_256),
            _mm256_shuffle_epi8(mm, mmm_hi_256));

        /* Extract each 128-bit lane and store 12 u16 (16 + 8 bytes). */
        __m128i ol_lane0 = _mm256_castsi256_si128(out_lo);
        __m128i oh_lane0 = _mm256_castsi256_si128(out_hi);
        __m128i ol_lane1 = _mm256_extracti128_si256(out_lo, 1);
        __m128i oh_lane1 = _mm256_extracti128_si256(out_hi, 1);

        /* dst is uint16_t* - 3*p + k addresses u16 element 3p+k.
         * Lane 0 produces u16 indices 0..11, lane 1 produces 12..23. */
        _mm_storeu_si128((__m128i *)(dst + 3 * p + 0),  ol_lane0);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 8),  oh_lane0);
        _mm_storeu_si128((__m128i *)(dst + 3 * p + 12), ol_lane1);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 20), oh_lane1);

        p += 8;
    }

    for (int c = 2 * safe_chunks_256; c < full_chunks; c++) {
        /* Load 16 bytes = 8 u16 source elements covering 4 pairs. */
        __m128i r = _mm_loadu_si128((const __m128i *)(src + 2 * p));

        /* (b,c) pairs built in-register: shift one u16 element, insert
         * the next source element (or the replicated edge) at the top -
         * this path never loads past the row end. */
        int next_idx = 2 * (p + 4);
        uint16_t next_a = (next_idx < w) ? src[next_idx] : src[w - 1];
        __m128i r_n1 = _mm_insert_epi16(_mm_srli_si128(r, 2), next_a, 7);

        /* 85*a + 171*b and 171*b + 85*c -> 4 i32 results each. */
        __m128i m1_u32 = _mm_srli_epi32(_mm_add_epi32(
            _mm_madd_epi16(r, w_ab), r128_32), 8);
        __m128i m2_u32 = _mm_srli_epi32(_mm_add_epi32(
            _mm_madd_epi16(r_n1, w_bc), r128_32), 8);

        __m128i mm = _mm_packus_epi32(m1_u32, m2_u32);

        /* Assemble output via two shuffle + OR for each 16/8 slice. */
        __m128i out_lo = _mm_or_si128(
            _mm_shuffle_epi8(r,  ma_lo),
            _mm_shuffle_epi8(mm, mmm_lo));
        __m128i out_hi = _mm_or_si128(
            _mm_shuffle_epi8(r,  ma_hi),
            _mm_shuffle_epi8(mm, mmm_hi));

        _mm_storeu_si128((__m128i *)(dst + 3 * p), out_lo);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 8), out_hi);

        p += 4;
    }

    /* Scalar tail */
    for (; p < pairs; p++) {
        uint16_t a = src[2 * p];
        uint16_t b = src[2 * p + 1];
        uint16_t c = (2 * p + 2 < w) ? src[2 * p + 2] : src[w - 1];
        dst[3 * p + 0] = a;
        dst[3 * p + 1] = up_blend_21_u16_scalar(a, b);
        dst[3 * p + 2] = up_blend_21_u16_scalar(c, b);
    }
}

/* ---- 1.5x (2->3) plane upscale for HDR ---- */
static void up_1_5x_plane_avx2_u16(const uint16_t *src, int src_w, int src_h,
                                   int src_el_stride, uint16_t *dst,
                                   int dst_el_stride, uint16_t *scratch)
{
    if (!scratch) return;

    int pairs_v = src_h / 2;
    for (int j = 0; j < pairs_v; j++) {
        const uint16_t *r2j  = src + (size_t)(2 * j)     * src_el_stride;
        const uint16_t *r2j1 = src + (size_t)(2 * j + 1) * src_el_stride;
        const uint16_t *r2j2 = (2 * j + 2 < src_h)
                                    ? src + (size_t)(2 * j + 2) * src_el_stride
                                    : r2j1;

        up_h_1_5x_row_avx2_u16(r2j, src_w,
                               dst + (size_t)(3 * j + 0) * dst_el_stride);

        up_vblend_21_row_avx2_u16(r2j, r2j1, src_w, scratch);
        up_h_1_5x_row_avx2_u16(scratch, src_w,
                               dst + (size_t)(3 * j + 1) * dst_el_stride);

        up_vblend_21_row_avx2_u16(r2j2, r2j1, src_w, scratch);
        up_h_1_5x_row_avx2_u16(scratch, src_w,
                               dst + (size_t)(3 * j + 2) * dst_el_stride);
    }
}

/* ---- Driver: upscale one HDR plane through the cascade + tail ---- */
static void upscale_plane_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src,
                                   int src_w, int src_h, int src_el_stride,
                                   int is_chroma)
{
    int N    = p->upscale_cascade_depth;
    int tail = p->upscale_tail_1_5x;

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
        if (dst) up_2x_plane_avx2_u16(src, src_w, src_h, src_el_stride,
                                       dst, dst_el_stride,
                                       p->upscale_scratch_hdr);
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
            up_2x_plane_avx2_u16(src_up, up_w, up_h, src_up_el_stride,
                                 dst, dst_el_stride, p->upscale_scratch_hdr);
        }
    }

    if (tail && (p->upscale_hdr_active & (1u << FUSED_UP_IDX_TAIL))) {
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
            up_1_5x_plane_avx2_u16(tail_src, tail_src_w, tail_src_h,
                                   tail_src_el_stride, dst, dst_el_stride,
                                   p->upscale_scratch_hdr);
        }
    }
}

void fused_kernel_upscale_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v)
{
    const uint16_t *up_src_u = src_u;
    const uint16_t *up_src_v = src_v;
    int up_src_uv_el_stride = p->src_uv_el_stride;

    if (p->is_p010) {
        if (fused_hdr_deinterleave_p010(p, src_u) != 0) return;
        up_src_u = p->p010_tmp_u;
        up_src_v = p->p010_tmp_v;
        up_src_uv_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);
    }

    upscale_plane_hdr_avx2(p, src_y, p->src_width, p->src_height,
                           p->src_y_el_stride, 0);
    upscale_plane_hdr_avx2(p, up_src_u, p->src_width / 2, p->src_height / 2,
                           up_src_uv_el_stride, 1);
    upscale_plane_hdr_avx2(p, up_src_v, p->src_width / 2, p->src_height / 2,
                           up_src_uv_el_stride, 2);
    _mm256_zeroupper();
}

void fused_kernel_thirds_up_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_hdr_avx2(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_hdr_avx2(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_hdr_avx2(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_avx2(p, src_y, src_u, src_v);
    }
}

#endif /* __x86_64__ */
