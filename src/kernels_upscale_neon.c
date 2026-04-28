/*
 * kernels_upscale_neon.c - NEON upscale kernels.
 *
 * Entry points:
 *   fused_kernel_upscale_neon       - upscale only
 *   fused_kernel_thirds_up_neon     - combined thirds downscale + upscale
 *   fused_kernel_pow2_up_neon       - combined pow2 downscale + upscale
 *
 * The upscale primitives are vectorized with NEON intrinsics:
 *   - Horizontal 2x: vrhaddq_u8 + vextq_u8 + vzip1q_u8/vzip2q_u8.
 *     16 source bytes -> 32 destination bytes per iteration.
 *   - Vertical 2x: vrhaddq_u8 to compute the midpoint row.
 *   - Horizontal 1.5x (2->3): vld2q_u8 even/odd deinterleave -> vmull_u8
 *     + vmlal_u8 with 85/171 weights -> vst3q_u8 interleaved store.
 *     32 source bytes -> 48 destination bytes per iteration.
 *   - Vertical 1.5x blend row: vmull_u8 + vmlal_u8 with 85/171 weights.
 *
 * HDR (10-bit) path uses the same structural approach with uint16_t data:
 *   - Horizontal/vertical 2x: vrhaddq_u16 + vzip1q_u16/vzip2q_u16 +
 *     vld1q_u16/vst1q_u16, 8 u16 per iteration.
 *   - 85/171 blends: vmull_u16 / vmlal_u16 + vshrn_n_u32 for u16×u16->u32
 *     intermediates (required because 85*max_u16 overflows 16 bits).
 *   - Horizontal 1.5x: vld2q_u16 deinterleave -> vmull/vmlal -> vst3q_u16
 *     interleaved store, 8 pairs per iter.
 *
 * The cascade levels each read from the previous level's output buffer.
 * Source memory is read exactly once at level 0; subsequent levels read
 * from L1/L2-hot output buffers that level k-1 just wrote.
 *
 * The combined down+up kernels currently call the existing downscale
 * NEON kernel followed by the upscale-only kernel.  The downscale phase
 * touches source memory once and the upscale phase touches it again.
 * On Apple Silicon (16 MB L2 per cluster) the second pass is L2-resident
 * for typical 1080p sources so the cost is small.  Genuine inner-loop
 * fusion (loading source rows once into NEON registers and emitting both
 * directions before moving to the next chunk) is left as a follow-up
 * optimization - see the per-kernel comments below.
 *
 */

#if defined(__aarch64__) || defined(_M_ARM64)

#include "internal.h"
#include "upscale_chunk.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>


/* -----------------------------------------------------------------------
 * Vectorized 2x horizontal doubling for one row
 * -----------------------------------------------------------------------
 *
 * Reads `src_w` source bytes and writes `2*src_w` destination bytes.
 * Each 16-byte source chunk produces 32 destination bytes:
 *   dst[2i+0] = src[i]
 *   dst[2i+1] = (src[i] + src[i+1] + 1) >> 1
 * The "next" sample for the right edge of the row replicates src[w-1].
 */
static void up_h_2x_row_neon(const uint8_t *src, int src_w, uint8_t *dst)
{
    int x = 0;
    int full_chunks = src_w / 16;

    for (int c = 0; c < full_chunks; c++) {
        uint8x16_t r = vld1q_u8(src + x);
        /* Build the "shifted by one" vector. The 17th byte comes from the
         * next chunk (or replicated last byte at the row's right edge). */
        uint8_t edge = (x + 16 < src_w) ? src[x + 16] : src[src_w - 1];
        uint8x16_t r_next = vextq_u8(r, vdupq_n_u8(edge), 1);
        uint8x16_t r_mid  = vrhaddq_u8(r, r_next);
        uint8x16x2_t zipped = { { vzip1q_u8(r, r_mid), vzip2q_u8(r, r_mid) } };
        vst1q_u8(dst + 2 * x +  0, zipped.val[0]);
        vst1q_u8(dst + 2 * x + 16, zipped.val[1]);
        x += 16;
    }

    /* Tail (less than 16 leftover bytes) - fall back to scalar. */
    for (; x < src_w; x++) {
        uint8_t a = src[x];
        uint8_t b = (x + 1 < src_w) ? src[x + 1] : a;
        dst[2 * x + 0] = a;
        dst[2 * x + 1] = up_avg_u8(a, b);
    }
}


/* -----------------------------------------------------------------------
 * Vertical 2x upscale of one plane (with horizontal doubling fused)
 * -----------------------------------------------------------------------
 *
 * Produces dst of size (2*src_w × 2*src_h).  For each source row i:
 *   - Output row 2i+0 = horizontally-doubled src[i]
 *   - Output row 2i+1 = horizontally-doubled vert_avg(src[i], src[i+1])
 *
 * The vertical midpoint row is computed in-place (vrhaddq_u8 over the
 * full row width) before being passed to up_h_2x_row_neon.  Bottom edge
 * replicates src[h-1] as src[h].
 */
static void up_2x_plane_neon(const uint8_t *src, int src_w, int src_h,
                             int src_stride, uint8_t *dst, int dst_stride,
                             uint8_t *scratch)
{
    /* `scratch` is a persistent row buffer provided by the caller,
     * sized at init.  Avoids per-frame aligned_alloc. */
    if (!scratch) return;

    for (int i = 0; i < src_h; i++) {
        const uint8_t *row_cur = src + (size_t)i * src_stride;
        const uint8_t *row_nxt = (i + 1 < src_h)
                                    ? src + (size_t)(i + 1) * src_stride
                                    : row_cur;

        /* Compute vertical midpoint row in-place via NEON pairwise average. */
        int x = 0;
        for (; x + 16 <= src_w; x += 16) {
            uint8x16_t a = vld1q_u8(row_cur + x);
            uint8x16_t b = vld1q_u8(row_nxt + x);
            vst1q_u8(scratch + x, vrhaddq_u8(a, b));
        }
        for (; x < src_w; x++) {
            scratch[x] = up_avg_u8(row_cur[x], row_nxt[x]);
        }

        /* Horizontally double both rows. */
        up_h_2x_row_neon(row_cur, src_w, dst + (size_t)(2 * i)     * dst_stride);
        up_h_2x_row_neon(scratch, src_w, dst + (size_t)(2 * i + 1) * dst_stride);
    }
}


/* -----------------------------------------------------------------------
 * Vectorized vertical 85/171 blend of two rows
 * -----------------------------------------------------------------------
 *
 * out[i] = (85 * a_row[i] + 171 * b_row[i] + 128) >> 8
 *
 * This is the vertical half of the 1.5x upscale - produces the intermediate
 * row between two source rows using the same 171/85 weights as the
 * existing horizontal thirds blend.
 */
static inline void up_vblend_21_row_neon(const uint8_t *a_row,
                                         const uint8_t *b_row,
                                         int w, uint8_t *out)
{
    int x = 0;
    const uint8x8_t w85_8  = vdup_n_u8(85);
    const uint8x8_t w171_8 = vdup_n_u8(171);
    const uint16x8_t r128  = vdupq_n_u16(128);

    for (; x + 16 <= w; x += 16) {
        uint8x16_t av = vld1q_u8(a_row + x);
        uint8x16_t bv = vld1q_u8(b_row + x);
        uint16x8_t lo = vmull_u8(vget_low_u8(av),  w85_8);
        lo = vmlal_u8(lo, vget_low_u8(bv),  w171_8);
        lo = vaddq_u16(lo, r128);
        uint16x8_t hi = vmull_u8(vget_high_u8(av), w85_8);
        hi = vmlal_u8(hi, vget_high_u8(bv), w171_8);
        hi = vaddq_u16(hi, r128);
        uint8x8_t  lo_u8 = vshrn_n_u16(lo, 8);
        uint8x8_t  hi_u8 = vshrn_n_u16(hi, 8);
        vst1q_u8(out + x, vcombine_u8(lo_u8, hi_u8));
    }
    for (; x < w; x++) {
        out[x] = up_blend_21_u8(a_row[x], b_row[x]);
    }
}

/* -----------------------------------------------------------------------
 * Vectorized horizontal 1.5x (2->3) upscale
 * -----------------------------------------------------------------------
 *
 * For each pair (src[2i], src[2i+1]) of source samples, emit three output
 * samples:
 *   dst[3i+0] = src[2i]
 *   dst[3i+1] = (src[2i]   * 85 + src[2i+1] * 171 + 128) >> 8
 *   dst[3i+2] = (src[2i+2] * 85 + src[2i+1] * 171 + 128) >> 8
 * where src[w] is replicated from src[w-1] at the right edge.
 *
 * Vectorized path: vld2q_u8 deinterleaves 32 source bytes into 16 "a"
 * bytes (even indices, i.e. src[2i]) and 16 "b" bytes (odd indices, i.e.
 * src[2i+1]).  The "c" vector for the second midpoint is the "a" vector
 * shifted left by one lane, with the next chunk's first "a" at the tail.
 * vst3q_u8 then stores (a, m1, m2) interleaved as 48 consecutive bytes.
 */
static void up_h_1_5x_row_neon(const uint8_t *src, int w, uint8_t *dst)
{
    int pairs = w / 2;
    int full_chunks = pairs / 16;   /* 16 pairs = 32 src bytes = 48 dst bytes */
    int p = 0;

    const uint8x8_t  w85_8  = vdup_n_u8(85);
    const uint8x8_t  w171_8 = vdup_n_u8(171);
    const uint16x8_t r128   = vdupq_n_u16(128);

    for (int c = 0; c < full_chunks; c++) {
        uint8x16x2_t ab = vld2q_u8(src + 2 * p);
        uint8x16_t a = ab.val[0];   /* src[2i] for i in [0..15] */
        uint8x16_t b = ab.val[1];   /* src[2i+1] for i in [0..15] */

        /* Next "a" at the end: either the first byte of the next chunk
         * or the row's last byte (edge replication). */
        int next_idx = 2 * (p + 16);
        uint8_t next_a = (next_idx < w) ? src[next_idx] : src[w - 1];
        uint8x16_t c_vec = vextq_u8(a, vdupq_n_u8(next_a), 1);

        /* m1 = (a*85 + b*171 + 128) >> 8 */
        uint16x8_t m1_lo = vmull_u8(vget_low_u8(a),  w85_8);
        m1_lo = vmlal_u8(m1_lo, vget_low_u8(b),  w171_8);
        m1_lo = vaddq_u16(m1_lo, r128);
        uint16x8_t m1_hi = vmull_u8(vget_high_u8(a), w85_8);
        m1_hi = vmlal_u8(m1_hi, vget_high_u8(b), w171_8);
        m1_hi = vaddq_u16(m1_hi, r128);
        uint8x16_t m1 = vcombine_u8(vshrn_n_u16(m1_lo, 8),
                                    vshrn_n_u16(m1_hi, 8));

        /* m2 = (c*85 + b*171 + 128) >> 8 */
        uint16x8_t m2_lo = vmull_u8(vget_low_u8(c_vec),  w85_8);
        m2_lo = vmlal_u8(m2_lo, vget_low_u8(b),  w171_8);
        m2_lo = vaddq_u16(m2_lo, r128);
        uint16x8_t m2_hi = vmull_u8(vget_high_u8(c_vec), w85_8);
        m2_hi = vmlal_u8(m2_hi, vget_high_u8(b), w171_8);
        m2_hi = vaddq_u16(m2_hi, r128);
        uint8x16_t m2 = vcombine_u8(vshrn_n_u16(m2_lo, 8),
                                    vshrn_n_u16(m2_hi, 8));

        /* Store a, m1, m2 interleaved as 48 consecutive bytes. */
        uint8x16x3_t triple = { { a, m1, m2 } };
        vst3q_u8(dst + 3 * p, triple);

        p += 16;
    }

    /* Scalar tail for the last (pairs % 16) pairs. */
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
 * 1.5x (2->3) plane upscale - fully vectorized
 * ----------------------------------------------------------------------- */
static void up_1_5x_plane_neon(const uint8_t *src, int src_w, int src_h,
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

        /* Row 3j+0 = horizontally-1.5x of r2j (exact copy vertically). */
        up_h_1_5x_row_neon(r2j, src_w,
                           dst + (size_t)(3 * j + 0) * dst_stride);

        /* Row 3j+1 = h_1_5x(vblend(r2j, r2j1)) - weights (85, 171) means
         * 1/3 of the top row and 2/3 of the bottom row. */
        up_vblend_21_row_neon(r2j, r2j1, src_w, scratch);
        up_h_1_5x_row_neon(scratch, src_w,
                           dst + (size_t)(3 * j + 1) * dst_stride);

        /* Row 3j+2 = h_1_5x(vblend(r2j2, r2j1)) - 2/3 of the bottom row
         * (r2j1 weighted 171) and 1/3 of the next row (r2j2 weighted 85). */
        up_vblend_21_row_neon(r2j2, r2j1, src_w, scratch);
        up_h_1_5x_row_neon(scratch, src_w,
                           dst + (size_t)(3 * j + 2) * dst_stride);
    }
}


/* -----------------------------------------------------------------------
 * Driver: upscale one plane through the cascade + tail
 * -----------------------------------------------------------------------
 *
 * Mirrors upscale_plane_scalar in kernels_upscale_scalar.c but calls the
 * vectorized 2x kernel above.
 */
static void upscale_plane_neon(const fused_kernel_params_t *p,
                               const uint8_t *src,
                               int src_w, int src_h, int src_stride,
                               int is_chroma)
{
    int N    = p->upscale_cascade_depth;
    int tail = p->upscale_tail_1_5x;

    /* Level 0 (2x) reads source */
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
        if (dst) up_2x_plane_neon(src, src_w, src_h, src_stride, dst, dst_stride,
                                   p->upscale_scratch);
    }

    /* Levels 1..N-1 cascade from up_out[k-1] */
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
            up_2x_plane_neon(src_up, up_w, up_h, src_up_stride, dst, dst_stride,
                             p->upscale_scratch);
        }
    }

    /* 1.5x tail */
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
            up_1_5x_plane_neon(tail_src, tail_src_w, tail_src_h, tail_src_stride,
                               dst, dst_stride, p->upscale_scratch);
        }
    }
}


/* -----------------------------------------------------------------------
 * Public entry points - SDR
 * ----------------------------------------------------------------------- */

void fused_kernel_upscale_neon(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v)
{
    upscale_plane_neon(p, src_y, p->src_width, p->src_height,
                       p->src_y_stride, 0);
    upscale_plane_neon(p, src_u, p->src_width / 2, p->src_height / 2,
                       p->src_uv_stride, 1);
    upscale_plane_neon(p, src_v, p->src_width / 2, p->src_height / 2,
                       p->src_uv_stride, 2);
}

void fused_kernel_thirds_up_neon(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    /* TODO: integrate the upscale emit into the per-chunk loop of the
     * existing thirds NEON kernel for true single-pass over source. The
     * current implementation is back-to-back which on Apple Silicon hits
     * L2 for the second pass and is fast in practice. */
    if (p->active_outputs != 0) {
        fused_kernel_thirds_neon(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_neon(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_neon(const fused_kernel_params_t *p,
                               const uint8_t *src_y,
                               const uint8_t *src_u,
                               const uint8_t *src_v)
{
    /* Same TODO as fused_kernel_thirds_up_neon. */
    if (p->active_outputs != 0) {
        fused_kernel_pow2_neon(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_neon(p, src_y, src_u, src_v);
    }
}


/* =======================================================================
 * HDR (10-bit) NEON upscale
 * =======================================================================
 *
 * Same structure as the SDR path above but operating on uint16_t planes.
 * The key difference is that 85 * max_u16 overflows 16 bits, so the
 * 85/171 blend must use u32 intermediates via vmull_u16 + vmlal_u16.
 * The 2x primitives still use u16-native avg (vrhaddq_u16) and
 * interleave (vzip1q_u16 / vzip2q_u16).
 *
 * Element strides: the HDR kernel params store byte strides.  Inside
 * these helpers we convert to element strides (divide by 2) for clean
 * uint16_t * arithmetic.
 */

/* Rounded u16 average (matches vrhaddq_u16 semantics). */
static inline uint16_t up_avg_u16_scalar(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a + (uint32_t)b + 1) >> 1);
}

/* Scalar (85*a + 171*b + 128) >> 8 for the tail handling. */
static inline uint16_t up_blend_21_u16_scalar(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a * 85 + (uint32_t)b * 171 + 128) >> 8);
}

/* ---- Horizontal 2x upscale of one HDR row ---- */
static void up_h_2x_row_neon_u16(const uint16_t *src, int src_w, uint16_t *dst)
{
    int x = 0;
    int full_chunks = src_w / 8;

    for (int c = 0; c < full_chunks; c++) {
        uint16x8_t r = vld1q_u16(src + x);
        /* Next element for the chunk's right edge: first element of the
         * next chunk or the row's last element (edge replication). */
        uint16_t edge = (x + 8 < src_w) ? src[x + 8] : src[src_w - 1];
        uint16x8_t r_next = vextq_u16(r, vdupq_n_u16(edge), 1);
        uint16x8_t r_mid  = vrhaddq_u16(r, r_next);
        /* Interleave src and mid: (r[0], r_mid[0], r[1], r_mid[1], ...). */
        uint16x8_t out_lo = vzip1q_u16(r, r_mid);
        uint16x8_t out_hi = vzip2q_u16(r, r_mid);
        vst1q_u16(dst + 2 * x +  0, out_lo);
        vst1q_u16(dst + 2 * x +  8, out_hi);
        x += 8;
    }

    /* Scalar tail (less than 8 leftover elements). */
    for (; x < src_w; x++) {
        uint16_t a = src[x];
        uint16_t b = (x + 1 < src_w) ? src[x + 1] : a;
        dst[2 * x + 0] = a;
        dst[2 * x + 1] = up_avg_u16_scalar(a, b);
    }
}

/* ---- Vertical 2x upscale of one HDR plane ---- */
static void up_2x_plane_neon_u16(const uint16_t *src, int src_w, int src_h,
                                 int src_el_stride, uint16_t *dst,
                                 int dst_el_stride, uint16_t *scratch)
{
    if (!scratch) return;

    for (int i = 0; i < src_h; i++) {
        const uint16_t *row_cur = src + (size_t)i * src_el_stride;
        const uint16_t *row_nxt = (i + 1 < src_h)
                                    ? src + (size_t)(i + 1) * src_el_stride
                                    : row_cur;

        /* Vertical midpoint row via vrhaddq_u16. */
        int x = 0;
        for (; x + 8 <= src_w; x += 8) {
            uint16x8_t a = vld1q_u16(row_cur + x);
            uint16x8_t b = vld1q_u16(row_nxt + x);
            vst1q_u16(scratch + x, vrhaddq_u16(a, b));
        }
        for (; x < src_w; x++) {
            scratch[x] = up_avg_u16_scalar(row_cur[x], row_nxt[x]);
        }

        up_h_2x_row_neon_u16(row_cur, src_w,
                             dst + (size_t)(2 * i)     * dst_el_stride);
        up_h_2x_row_neon_u16(scratch, src_w,
                             dst + (size_t)(2 * i + 1) * dst_el_stride);
    }
}

/* ---- Vertical 85/171 blend of two HDR rows ---- */
static inline void up_vblend_21_row_neon_u16(const uint16_t *a_row,
                                             const uint16_t *b_row,
                                             int w, uint16_t *out)
{
    int x = 0;
    const uint16x4_t w85_4   = vdup_n_u16(85);
    const uint16x4_t w171_4  = vdup_n_u16(171);
    const uint32x4_t r128_32 = vdupq_n_u32(128);

    for (; x + 8 <= w; x += 8) {
        uint16x8_t av = vld1q_u16(a_row + x);
        uint16x8_t bv = vld1q_u16(b_row + x);
        /* +128 folds into the first vmlal; saves a separate vaddq_u32. */
        uint32x4_t lo = vmlal_u16(r128_32, vget_low_u16(av), w85_4);
        lo = vmlal_u16(lo, vget_low_u16(bv), w171_4);
        uint32x4_t hi = vmlal_u16(r128_32, vget_high_u16(av), w85_4);
        hi = vmlal_u16(hi, vget_high_u16(bv), w171_4);
        /* Shift right 8 and narrow back to u16. */
        uint16x4_t lo_u16 = vshrn_n_u32(lo, 8);
        uint16x4_t hi_u16 = vshrn_n_u32(hi, 8);
        vst1q_u16(out + x, vcombine_u16(lo_u16, hi_u16));
    }
    for (; x < w; x++) {
        out[x] = up_blend_21_u16_scalar(a_row[x], b_row[x]);
    }
}

/* ---- Horizontal 1.5x (2->3) HDR upscale of one row ---- */
static void up_h_1_5x_row_neon_u16(const uint16_t *src, int w, uint16_t *dst)
{
    int pairs = w / 2;
    int full_chunks = pairs / 8;   /* 8 pairs per chunk */
    int p = 0;

    const uint16x4_t w85_4   = vdup_n_u16(85);
    const uint16x4_t w171_4  = vdup_n_u16(171);
    const uint32x4_t r128_32 = vdupq_n_u32(128);

    for (int c = 0; c < full_chunks; c++) {
        /* vld2q_u16 loads 16 u16 values and deinterleaves into two 8-lane
         * vectors: even positions -> a, odd positions -> b. */
        uint16x8x2_t ab = vld2q_u16(src + 2 * p);
        uint16x8_t a = ab.val[0];
        uint16x8_t b = ab.val[1];

        /* Next "a" for the c-vector shift. */
        int next_idx = 2 * (p + 8);
        uint16_t next_a = (next_idx < w) ? src[next_idx] : src[w - 1];
        uint16x8_t c_vec = vextq_u16(a, vdupq_n_u16(next_a), 1);

        /* +128 folds into the first vmlal of each half; saves 4 vaddq_u32. */
        /* m1 = (a*85 + b*171 + 128) >> 8 */
        uint32x4_t m1_lo = vmlal_u16(r128_32, vget_low_u16(a), w85_4);
        m1_lo = vmlal_u16(m1_lo, vget_low_u16(b), w171_4);
        uint32x4_t m1_hi = vmlal_u16(r128_32, vget_high_u16(a), w85_4);
        m1_hi = vmlal_u16(m1_hi, vget_high_u16(b), w171_4);
        uint16x8_t m1 = vcombine_u16(vshrn_n_u32(m1_lo, 8),
                                     vshrn_n_u32(m1_hi, 8));

        /* m2 = (c*85 + b*171 + 128) >> 8 */
        uint32x4_t m2_lo = vmlal_u16(r128_32, vget_low_u16(c_vec), w85_4);
        m2_lo = vmlal_u16(m2_lo, vget_low_u16(b), w171_4);
        uint32x4_t m2_hi = vmlal_u16(r128_32, vget_high_u16(c_vec), w85_4);
        m2_hi = vmlal_u16(m2_hi, vget_high_u16(b), w171_4);
        uint16x8_t m2 = vcombine_u16(vshrn_n_u32(m2_lo, 8),
                                     vshrn_n_u32(m2_hi, 8));

        /* vst3q_u16 stores three 8-lane vectors interleaved as 24 u16
         * values: {a[0], m1[0], m2[0], a[1], m1[1], m2[1], ...}. */
        uint16x8x3_t triple = { { a, m1, m2 } };
        vst3q_u16(dst + 3 * p, triple);

        p += 8;
    }

    /* Scalar tail. */
    for (; p < pairs; p++) {
        uint16_t a = src[2 * p];
        uint16_t b = src[2 * p + 1];
        uint16_t c = (2 * p + 2 < w) ? src[2 * p + 2] : src[w - 1];
        dst[3 * p + 0] = a;
        dst[3 * p + 1] = up_blend_21_u16_scalar(a, b);
        dst[3 * p + 2] = up_blend_21_u16_scalar(c, b);
    }
}

/* ---- 1.5x plane upscale for HDR ---- */
static void up_1_5x_plane_neon_u16(const uint16_t *src, int src_w, int src_h,
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

        up_h_1_5x_row_neon_u16(r2j, src_w,
                               dst + (size_t)(3 * j + 0) * dst_el_stride);

        up_vblend_21_row_neon_u16(r2j, r2j1, src_w, scratch);
        up_h_1_5x_row_neon_u16(scratch, src_w,
                               dst + (size_t)(3 * j + 1) * dst_el_stride);

        up_vblend_21_row_neon_u16(r2j2, r2j1, src_w, scratch);
        up_h_1_5x_row_neon_u16(scratch, src_w,
                               dst + (size_t)(3 * j + 2) * dst_el_stride);
    }
}

/* ---- Driver: upscale one HDR plane through the cascade + tail ---- */
static void upscale_plane_hdr_neon(const fused_hdr_kernel_params_t *p,
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
        if (dst) up_2x_plane_neon_u16(src, src_w, src_h, src_el_stride,
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
            up_2x_plane_neon_u16(src_up, up_w, up_h, src_up_el_stride,
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
            up_1_5x_plane_neon_u16(tail_src, tail_src_w, tail_src_h,
                                   tail_src_el_stride, dst, dst_el_stride,
                                   p->upscale_scratch_hdr);
        }
    }
}

void fused_kernel_upscale_hdr_neon(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v)
{
    upscale_plane_hdr_neon(p, src_y, p->src_width, p->src_height,
                           p->src_y_el_stride, 0);
    upscale_plane_hdr_neon(p, src_u, p->src_width / 2, p->src_height / 2,
                           p->src_uv_el_stride, 1);
    upscale_plane_hdr_neon(p, src_v, p->src_width / 2, p->src_height / 2,
                           p->src_uv_el_stride, 2);
}

void fused_kernel_thirds_up_hdr_neon(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_hdr_neon(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_neon(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_hdr_neon(const fused_hdr_kernel_params_t *p,
                                   const uint16_t *src_y,
                                   const uint16_t *src_u,
                                   const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_hdr_neon(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_neon(p, src_y, src_u, src_v);
    }
}

#endif /* __aarch64__ */
