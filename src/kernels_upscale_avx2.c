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
static void up_h_2x_row_avx2(const uint8_t *src, int src_w, uint8_t *dst)
{
    /* 128-bit SSE inner loop.  We intentionally avoid 256-bit ops here
     * because AVX2 _mm256_unpacklo/hi_epi8 are per-lane, and assembling
     * linear output would require an extra _mm256_permute2x128_si256
     * per chunk.  permute2x128 is expensive on Zen 1 (where it runs at
     * roughly 1/3 throughput) and the savings from going to 256-bit
     * are wiped out by the permute.  On Zen 2+ and Intel Skylake+,
     * 128-bit is only slightly slower than a well-tuned 256-bit path
     * because the workload is memory-bandwidth bound at typical sizes. */
    int x = 0;
    int full_chunks = src_w / 16;

    for (int c = 0; c < full_chunks; c++) {
        __m128i r = _mm_loadu_si128((const __m128i *)(src + x));

        /* "Shifted by one byte" version of r, with the 17th byte being
         * either the first byte of the next chunk or the row's last
         * byte (edge replication). */
        uint8_t edge = (x + 16 < src_w) ? src[x + 16] : src[src_w - 1];
        __m128i edge_v = _mm_set1_epi8((char)edge);
        __m128i r_shifted = _mm_alignr_epi8(edge_v, r, 1);
        __m128i r_mid = _mm_avg_epu8(r, r_shifted);

        /* Interleave src and mid.  unpacklo/hi_epi8 on 128-bit produce
         * linearly-ordered output (no lane crossing) so no permute is
         * needed. */
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
 * For each 16-byte chunk it computes the vertical average of the two source
 * rows ((a + b + 1) >> 1) in registers and feeds it straight into the
 * horizontal doubling, so the averaged row never has to be written to a
 * scratch buffer and read back.  The horizontal doubling matches
 * up_h_2x_row_avx2.
 * ----------------------------------------------------------------------- */
static void up_h_2x_vblend_row_avx2(const uint8_t *a_row, const uint8_t *b_row,
                                    int src_w, uint8_t *dst)
{
    int x = 0;
    int full_chunks = src_w / 16;

    for (int c = 0; c < full_chunks; c++) {
        /* r = vertical average of the two source rows for this 16-byte
         * chunk. */
        __m128i r = _mm_avg_epu8(
            _mm_loadu_si128((const __m128i *)(a_row + x)),
            _mm_loadu_si128((const __m128i *)(b_row + x)));

        /* Edge byte: the averaged value one column past this chunk, or the
         * last averaged column replicated at the row end. */
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
 * ----------------------------------------------------------------------- */
static inline void up_vblend_21_row_avx2(const uint8_t *a_row,
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
 * Vectorized horizontal 1.5x (2->3) upscale (AVX2 via 128-bit shuffles)
 * -----------------------------------------------------------------------
 *
 * Processes 8 source pairs (16 source bytes) -> 24 destination bytes per
 * iteration using SSE/AVX2 128-bit operations.  SSE has no direct 3-way
 * interleave store, so the output is assembled via pshufb + OR with
 * precomputed masks.  128-bit is used because AVX2 256-bit shuffles are
 * per-lane and the 3-way interleave crosses lane boundaries.
 *
 * Optimizations over a naive version:
 *   1. c_bytes (= src[2i+2]) is computed by loading src + 2*p + 2 and
 *      applying the same even-byte deinterleave mask.  This saves the
 *      srli_si128 + insert_epi8 dependency chain on the hot path.  The
 *      last chunk uses the slower srli+insert path because src+2 would
 *      walk past the end of the row.
 *   2. 171 * b is computed once and shared between m1 and m2 - saves a
 *      mullo_epi16 per chunk.
 *   3. m1 and m2 are pre-interleaved via unpacklo_epi8 into a single
 *      `mm` vector ({m1[0], m2[0], m1[1], m2[1], ...}) so the 3-way
 *      output assembly uses 4 pshufb (not 6) - saves 2 shuffle-port
 *      ops per chunk, which is significant on Zen 1 where pshufb is
 *      1/cycle throughput.
 */
static void up_h_1_5x_row_avx2(const uint8_t *src, int w, uint8_t *dst)
{
    int pairs = w / 2;
    int full_chunks = pairs / 8;   /* 8 pairs = 16 src bytes = 24 dst bytes */
    int p = 0;

    /* 128-bit fast path reads src[2p .. 2p+17] (17 bytes starting from 2p).
     * We need 2p + 17 < w, i.e. p * 16 < w - 17, so the max safe number
     * of chunks is (w - 2) / 16 (clamped to full_chunks).  The last
     * remaining chunk (if any) falls through to the slow path. */
    int safe_chunks = (w > 1) ? (w - 2) / 16 : 0;
    if (safe_chunks > full_chunks) safe_chunks = full_chunks;
    if (safe_chunks < 0)           safe_chunks = 0;

    /* 256-bit fast path processes 16 pairs -> 48 dst bytes per chunk.
     * Reads src[2p .. 2p+33] (r from 2p, r_n2 from 2p+2, both 32 bytes).
     * Need 2p + 33 < w, so ci_256 <= (w - 34)/32.  Number of 256-bit
     * safe chunks = (w - 34)/32 + 1 when w >= 34, else 0.  Capped at
     * half of safe_chunks (128-bit) so the 128-bit fast-path residual
     * stays <= 1 chunk. */
    int safe_chunks_256 = (w >= 34) ? ((w - 34) / 32 + 1) : 0;
    if (safe_chunks_256 * 2 > safe_chunks)
        safe_chunks_256 = safe_chunks / 2;

    /* Deinterleave masks: even bytes -> low 8 bytes, odd bytes -> low 8 bytes. */
    static const int8_t even_mask_bytes[16] = {
        0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int8_t odd_mask_bytes[16] = {
        1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1
    };

    /* Interleave masks for the first 16 output bytes.
     *
     * Output position layout for 8 pairs (bytes 0..15 of the stream):
     *   0=a[0], 1=m1[0], 2=m2[0], 3=a[1], 4=m1[1], 5=m2[1],
     *   6=a[2], 7=m1[2], 8=m2[2], 9=a[3], 10=m1[3], 11=m2[3],
     *   12=a[4], 13=m1[4], 14=m2[4], 15=a[5]
     *
     * Source a vector: a[0..7] in low 8 bytes.
     * Source "mm" vector: unpacklo(m1, m2) = {m1[0], m2[0], m1[1], m2[1],
     *   m1[2], m2[2], m1[3], m2[3], m1[4], m2[4], m1[5], m2[5], ...}
     * Indices into mm: m1[i] at mm[2i], m2[i] at mm[2i+1].
     *
     * So:
     *   a_lo mask selects a_bytes for positions {0, 3, 6, 9, 12, 15}:
     *     indices 0, 1, 2, 3, 4, 5
     *   mm_lo mask selects mm for positions {1, 2, 4, 5, 7, 8, 10, 11, 13, 14}:
     *     mm indices 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (= m1[0], m2[0], m1[1],
     *     m2[1], m1[2], m2[2], m1[3], m2[3], m1[4], m2[4]) */
    static const int8_t a_lo_bytes[16] = {
        0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5
    };
    static const int8_t mm_lo_bytes[16] = {
        -1, 0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1, 8, 9, -1
    };

    /* Interleave masks for the next 8 output bytes (positions 16..23):
     *   16=m1[5], 17=m2[5], 18=a[6], 19=m1[6], 20=m2[6], 21=a[7],
     *   22=m1[7], 23=m2[7]
     *
     *   a_hi mask: positions 2 and 5 from a_bytes indices 6 and 7
     *   mm_hi mask: positions 0, 1, 3, 4, 6, 7 from mm indices 10..15
     *     (mm[10]=m1[5], mm[11]=m2[5], mm[12]=m1[6], mm[13]=m2[6],
     *      mm[14]=m1[7], mm[15]=m2[7]) */
    static const int8_t a_hi_bytes[16] = {
        -1, -1, 6, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int8_t mm_hi_bytes[16] = {
        10, 11, -1, 12, 13, -1, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1
    };

    const __m128i em     = _mm_loadu_si128((const __m128i *)even_mask_bytes);
    const __m128i om     = _mm_loadu_si128((const __m128i *)odd_mask_bytes);
    const __m128i ma_lo  = _mm_loadu_si128((const __m128i *)a_lo_bytes);
    const __m128i mmm_lo = _mm_loadu_si128((const __m128i *)mm_lo_bytes);
    const __m128i ma_hi  = _mm_loadu_si128((const __m128i *)a_hi_bytes);
    const __m128i mmm_hi = _mm_loadu_si128((const __m128i *)mm_hi_bytes);

    const __m128i w85_16  = _mm_set1_epi16(85);
    const __m128i w171_16 = _mm_set1_epi16(171);
    const __m128i r128_16 = _mm_set1_epi16(128);
    const __m128i zero    = _mm_setzero_si128();

    /* 256-bit mask/constant broadcasts.  vpshufb and the unpack/pack/add/
     * mul ops we use are all lane-independent, so the same per-lane mask
     * is broadcast to both 128-bit lanes of the __m256i. */
    const __m256i em_256     = _mm256_broadcastsi128_si256(em);
    const __m256i om_256     = _mm256_broadcastsi128_si256(om);
    const __m256i ma_lo_256  = _mm256_broadcastsi128_si256(ma_lo);
    const __m256i mmm_lo_256 = _mm256_broadcastsi128_si256(mmm_lo);
    const __m256i ma_hi_256  = _mm256_broadcastsi128_si256(ma_hi);
    const __m256i mmm_hi_256 = _mm256_broadcastsi128_si256(mmm_hi);
    const __m256i w85_256    = _mm256_set1_epi16(85);
    const __m256i w171_256   = _mm256_set1_epi16(171);
    const __m256i r128_256   = _mm256_set1_epi16(128);
    const __m256i zero_256   = _mm256_setzero_si256();

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
        __m256i r_n2 = _mm256_loadu_si256((const __m256i *)(src + 2 * p + 2));

        __m256i a_bytes = _mm256_shuffle_epi8(r,    em_256);
        __m256i b_bytes = _mm256_shuffle_epi8(r,    om_256);
        __m256i c_bytes = _mm256_shuffle_epi8(r_n2, em_256);

        __m256i a_u16 = _mm256_unpacklo_epi8(a_bytes, zero_256);
        __m256i b_u16 = _mm256_unpacklo_epi8(b_bytes, zero_256);
        __m256i c_u16 = _mm256_unpacklo_epi8(c_bytes, zero_256);

        __m256i b171 = _mm256_mullo_epi16(b_u16, w171_256);
        __m256i a85  = _mm256_mullo_epi16(a_u16, w85_256);
        __m256i c85  = _mm256_mullo_epi16(c_u16, w85_256);

        __m256i m1_u16 = _mm256_srli_epi16(
            _mm256_add_epi16(_mm256_add_epi16(a85, b171), r128_256), 8);
        __m256i m2_u16 = _mm256_srli_epi16(
            _mm256_add_epi16(_mm256_add_epi16(c85, b171), r128_256), 8);

        __m256i m1 = _mm256_packus_epi16(m1_u16, zero_256);
        __m256i m2 = _mm256_packus_epi16(m2_u16, zero_256);

        __m256i mm = _mm256_unpacklo_epi8(m1, m2);

        __m256i out_lo = _mm256_or_si256(
            _mm256_shuffle_epi8(a_bytes, ma_lo_256),
            _mm256_shuffle_epi8(mm,      mmm_lo_256));
        __m256i out_hi = _mm256_or_si256(
            _mm256_shuffle_epi8(a_bytes, ma_hi_256),
            _mm256_shuffle_epi8(mm,      mmm_hi_256));

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
        __m128i r_n2 = _mm_loadu_si128((const __m128i *)(src + 2 * p + 2));

        __m128i a_bytes = _mm_shuffle_epi8(r,    em);
        __m128i b_bytes = _mm_shuffle_epi8(r,    om);
        __m128i c_bytes = _mm_shuffle_epi8(r_n2, em);

        __m128i a_u16 = _mm_unpacklo_epi8(a_bytes, zero);
        __m128i b_u16 = _mm_unpacklo_epi8(b_bytes, zero);
        __m128i c_u16 = _mm_unpacklo_epi8(c_bytes, zero);

        /* Share 171*b between m1 and m2. */
        __m128i b171 = _mm_mullo_epi16(b_u16, w171_16);
        __m128i a85  = _mm_mullo_epi16(a_u16, w85_16);
        __m128i c85  = _mm_mullo_epi16(c_u16, w85_16);

        __m128i m1_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_add_epi16(a85, b171), r128_16), 8);
        __m128i m2_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_add_epi16(c85, b171), r128_16), 8);

        __m128i m1 = _mm_packus_epi16(m1_u16, zero);
        __m128i m2 = _mm_packus_epi16(m2_u16, zero);

        /* Pre-interleave m1 and m2 so the output assembly needs only
         * 4 shuffles (2 for lo, 2 for hi) instead of 6. */
        __m128i mm = _mm_unpacklo_epi8(m1, m2);  /* {m1[0], m2[0], m1[1], m2[1], ...} */

        __m128i out_lo = _mm_or_si128(
            _mm_shuffle_epi8(a_bytes, ma_lo),
            _mm_shuffle_epi8(mm,      mmm_lo));
        __m128i out_hi = _mm_or_si128(
            _mm_shuffle_epi8(a_bytes, ma_hi),
            _mm_shuffle_epi8(mm,      mmm_hi));

        _mm_storeu_si128((__m128i *)(dst + 3 * p), out_lo);
        _mm_storel_epi64((__m128i *)(dst + 3 * p + 16), out_hi);

        p += 8;
    }

    /* Slow path for the last (unsafe) chunk(s) - uses srli+insert for
     * the c vector so we don't walk past the end of the row. */
    for (int ci = safe_chunks; ci < full_chunks; ci++) {
        __m128i r = _mm_loadu_si128((const __m128i *)(src + 2 * p));

        __m128i a_bytes = _mm_shuffle_epi8(r, em);
        __m128i b_bytes = _mm_shuffle_epi8(r, om);

        int next_idx = 2 * (p + 8);
        uint8_t next_a = (next_idx < w) ? src[next_idx] : src[w - 1];
        __m128i c_raw  = _mm_srli_si128(a_bytes, 1);
        __m128i c_bytes = _mm_insert_epi8(c_raw, next_a, 7);

        __m128i a_u16 = _mm_unpacklo_epi8(a_bytes, zero);
        __m128i b_u16 = _mm_unpacklo_epi8(b_bytes, zero);
        __m128i c_u16 = _mm_unpacklo_epi8(c_bytes, zero);

        __m128i b171 = _mm_mullo_epi16(b_u16, w171_16);
        __m128i a85  = _mm_mullo_epi16(a_u16, w85_16);
        __m128i c85  = _mm_mullo_epi16(c_u16, w85_16);

        __m128i m1_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_add_epi16(a85, b171), r128_16), 8);
        __m128i m2_u16 = _mm_srli_epi16(
            _mm_add_epi16(_mm_add_epi16(c85, b171), r128_16), 8);

        __m128i m1 = _mm_packus_epi16(m1_u16, zero);
        __m128i m2 = _mm_packus_epi16(m2_u16, zero);

        __m128i mm = _mm_unpacklo_epi8(m1, m2);

        __m128i out_lo = _mm_or_si128(
            _mm_shuffle_epi8(a_bytes, ma_lo),
            _mm_shuffle_epi8(mm,      mmm_lo));
        __m128i out_hi = _mm_or_si128(
            _mm_shuffle_epi8(a_bytes, ma_hi),
            _mm_shuffle_epi8(mm,      mmm_hi));

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

/* ---- Horizontal 2x upscale of one HDR row (128-bit SSE) ----
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
    int full_chunks = src_w / 8;

    for (int c = 0; c < full_chunks; c++) {
        __m128i r = _mm_loadu_si128((const __m128i *)(src + x));
        /* Edge: next chunk's first u16 (or replicated row end). */
        uint16_t edge = (x + 8 < src_w) ? src[x + 8] : src[src_w - 1];
        __m128i edge_v = _mm_set1_epi16((short)edge);
        /* Shift right by one u16 = 2 bytes via alignr. */
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

        /* Vertical midpoint via _mm256_avg_epu16. */
        int x = 0;
        for (; x + 16 <= src_w; x += 16) {
            __m256i a = _mm256_loadu_si256((const __m256i *)(row_cur + x));
            __m256i b = _mm256_loadu_si256((const __m256i *)(row_nxt + x));
            _mm256_storeu_si256((__m256i *)(scratch + x), _mm256_avg_epu16(a, b));
        }
        for (; x < src_w; x++) {
            scratch[x] = up_avg_u16_scalar(row_cur[x], row_nxt[x]);
        }

        up_h_2x_row_avx2_u16(row_cur, src_w,
                             dst + (size_t)(2 * i)     * dst_el_stride);
        up_h_2x_row_avx2_u16(scratch, src_w,
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
 *      fast path + 128-bit slow-path residual) ---- */
static void up_h_1_5x_row_avx2_u16(const uint16_t *src, int w, uint16_t *dst)
{
    int pairs = w / 2;
    int full_chunks = pairs / 4;   /* 4 pairs = 8 src u16 = 16 src bytes =
                                    * 12 dst u16 = 24 dst bytes per iter */
    int p = 0;

    /* 256-bit fast path: 8 pairs -> 24 dst u16 per chunk.  Reads 16 u16
     * source elements starting at 2p (32 bytes) into r, and another 16
     * starting at 2p+2 into r_n2 - last element accessed is at index
     * 2p + 17.  Need 2p + 17 < w, so ci_256 <= (w - 18)/16.  The residual
     * falls through to the existing 128-bit loop (which has its own
     * bounds-safe srli+insert slow path for c). */
    int safe_chunks_256 = (w >= 18) ? ((w - 18) / 16 + 1) : 0;
    if (safe_chunks_256 * 2 > full_chunks) safe_chunks_256 = full_chunks / 2;

    /* Byte-level shuffle masks - they operate on the 16-byte registers
     * holding u16 values (2 bytes each). */

    /* Deinterleave masks: even u16 -> low 8 bytes, odd u16 -> low 8 bytes. */
    static const int8_t even_u16_mask[16] = {
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int8_t odd_u16_mask[16] = {
        2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1
    };

    /* Interleave masks for the first 16 output bytes (8 u16 = a[0,1,2],
     * m1[0,1,2], m2[0,1]).  mm = _mm_unpacklo_epi16(m1, m2) = {m1[0],
     * m2[0], m1[1], m2[1], m1[2], m2[2], m1[3], m2[3]} - 8 u16 = 16 bytes.
     *
     * Output u16 positions 0..7: a[0], m1[0], m2[0], a[1], m1[1], m2[1],
     *   a[2], m1[2]
     *   -> a bytes at u16 pos 0, 3, 6     = a_bytes indices 0,1  2,3  4,5
     *   -> mm bytes at u16 pos 1, 2, 4, 5, 7  = mm indices 0..5 and 8..9 */
    static const int8_t a_lo_u16_mask[16] = {
        0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1
    };
    static const int8_t mm_lo_u16_mask[16] = {
        -1, -1, 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1, -1, 8, 9
    };

    /* Next 8 output bytes (4 u16): m2[2], a[3], m1[3], m2[3]
     *   mm[10,11] = m2[2]  -> position 0
     *   a[3]      = a bytes 6,7
     *   mm[12,13] = m1[3]  -> position 4
     *   mm[14,15] = m2[3]  -> position 6 */
    static const int8_t a_hi_u16_mask[16] = {
        -1, -1, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };
    static const int8_t mm_hi_u16_mask[16] = {
        10, 11, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1
    };

    const __m128i em       = _mm_loadu_si128((const __m128i *)even_u16_mask);
    const __m128i om       = _mm_loadu_si128((const __m128i *)odd_u16_mask);
    const __m128i ma_lo    = _mm_loadu_si128((const __m128i *)a_lo_u16_mask);
    const __m128i mmm_lo   = _mm_loadu_si128((const __m128i *)mm_lo_u16_mask);
    const __m128i ma_hi    = _mm_loadu_si128((const __m128i *)a_hi_u16_mask);
    const __m128i mmm_hi   = _mm_loadu_si128((const __m128i *)mm_hi_u16_mask);

    /* {85, 171, 85, 171, ...} as 8 i16 - used with _mm_madd_epi16 on
     * interleaved (a, b) u16 pairs to get 85*a + 171*b per pair. */
    const __m128i weights  = _mm_set1_epi32((171 << 16) | 85);
    const __m128i r128_32  = _mm_set1_epi32(128);
    const __m128i zero     = _mm_setzero_si128();

    /* 256-bit broadcasts of the same masks/weights.  vpshufb,
     * vpmaddwd, vpunpcklwd, vpackusdw and the add/shift ops we use are
     * all lane-independent, so each 128-bit lane independently
     * computes its own 4 pairs. */
    const __m256i em_256       = _mm256_broadcastsi128_si256(em);
    const __m256i om_256       = _mm256_broadcastsi128_si256(om);
    const __m256i ma_lo_256    = _mm256_broadcastsi128_si256(ma_lo);
    const __m256i mmm_lo_256   = _mm256_broadcastsi128_si256(mmm_lo);
    const __m256i ma_hi_256    = _mm256_broadcastsi128_si256(ma_hi);
    const __m256i mmm_hi_256   = _mm256_broadcastsi128_si256(mmm_hi);
    const __m256i weights_256  = _mm256_set1_epi32((171 << 16) | 85);
    const __m256i r128_32_256  = _mm256_set1_epi32(128);
    const __m256i zero_256     = _mm256_setzero_si256();

    /* --------------------------------------------------------------
     * 256-bit HDR fast path.  Lane 0 handles 4 pairs [p..p+3], lane 1
     * handles [p+4..p+7].  Uses the same "load r_n2 = src + 2p + 2"
     * trick as the SDR fast path to avoid a per-chunk srli+insert for
     * the c vector - simpler and faster, but requires an extra 16 u16
     * of headroom beyond the chunk's end.
     * -------------------------------------------------------------- */
    for (int ci = 0; ci < safe_chunks_256; ci++) {
        __m256i r    = _mm256_loadu_si256((const __m256i *)(src + 2 * p));
        __m256i r_n2 = _mm256_loadu_si256((const __m256i *)(src + 2 * p + 2));

        __m256i a_bytes = _mm256_shuffle_epi8(r,    em_256);
        __m256i b_bytes = _mm256_shuffle_epi8(r,    om_256);
        __m256i c_bytes = _mm256_shuffle_epi8(r_n2, em_256);

        __m256i ab = _mm256_unpacklo_epi16(a_bytes, b_bytes);
        __m256i cb = _mm256_unpacklo_epi16(c_bytes, b_bytes);

        __m256i m1_u32 = _mm256_madd_epi16(ab, weights_256);
        __m256i m2_u32 = _mm256_madd_epi16(cb, weights_256);

        m1_u32 = _mm256_srli_epi32(_mm256_add_epi32(m1_u32, r128_32_256), 8);
        m2_u32 = _mm256_srli_epi32(_mm256_add_epi32(m2_u32, r128_32_256), 8);

        __m256i m1 = _mm256_packus_epi32(m1_u32, zero_256);
        __m256i m2 = _mm256_packus_epi32(m2_u32, zero_256);

        __m256i mm = _mm256_unpacklo_epi16(m1, m2);

        __m256i out_lo = _mm256_or_si256(
            _mm256_shuffle_epi8(a_bytes, ma_lo_256),
            _mm256_shuffle_epi8(mm,      mmm_lo_256));
        __m256i out_hi = _mm256_or_si256(
            _mm256_shuffle_epi8(a_bytes, ma_hi_256),
            _mm256_shuffle_epi8(mm,      mmm_hi_256));

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

        /* Deinterleave even/odd u16 values. */
        __m128i a_bytes = _mm_shuffle_epi8(r, em);  /* a[0..3] in low 8 bytes */
        __m128i b_bytes = _mm_shuffle_epi8(r, om);  /* b[0..3] in low 8 bytes */

        /* c_bytes = a shifted by one u16 element, with next_a at index 3. */
        int next_idx = 2 * (p + 4);
        uint16_t next_a = (next_idx < w) ? src[next_idx] : src[w - 1];
        __m128i c_raw = _mm_srli_si128(a_bytes, 2);            /* shift a by 1 u16 */
        __m128i c_bytes = _mm_insert_epi16(c_raw, next_a, 3);  /* u16 at pos 3 */

        /* Interleave a with b and c with b for madd_epi16. */
        __m128i ab = _mm_unpacklo_epi16(a_bytes, b_bytes);
        __m128i cb = _mm_unpacklo_epi16(c_bytes, b_bytes);

        /* 85*a + 171*b -> 4 i32 results */
        __m128i m1_u32 = _mm_madd_epi16(ab, weights);
        __m128i m2_u32 = _mm_madd_epi16(cb, weights);

        m1_u32 = _mm_srli_epi32(_mm_add_epi32(m1_u32, r128_32), 8);
        m2_u32 = _mm_srli_epi32(_mm_add_epi32(m2_u32, r128_32), 8);

        /* Pack i32 -> u16 (4 values in low 8 bytes). */
        __m128i m1 = _mm_packus_epi32(m1_u32, zero);
        __m128i m2 = _mm_packus_epi32(m2_u32, zero);

        /* Interleave m1 and m2 at u16 granularity:
         * mm = {m1[0], m2[0], m1[1], m2[1], m1[2], m2[2], m1[3], m2[3]} */
        __m128i mm = _mm_unpacklo_epi16(m1, m2);

        /* Assemble output via two shuffle + OR for each 16/8 slice. */
        __m128i out_lo = _mm_or_si128(
            _mm_shuffle_epi8(a_bytes, ma_lo),
            _mm_shuffle_epi8(mm,      mmm_lo));
        __m128i out_hi = _mm_or_si128(
            _mm_shuffle_epi8(a_bytes, ma_hi),
            _mm_shuffle_epi8(mm,      mmm_hi));

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
    upscale_plane_hdr_avx2(p, src_y, p->src_width, p->src_height,
                           p->src_y_el_stride, 0);
    upscale_plane_hdr_avx2(p, src_u, p->src_width / 2, p->src_height / 2,
                           p->src_uv_el_stride, 1);
    upscale_plane_hdr_avx2(p, src_v, p->src_width / 2, p->src_height / 2,
                           p->src_uv_el_stride, 2);
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
