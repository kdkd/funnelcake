/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * kernels_upscale_scalar.c - scalar upscale kernels.
 *
 * Four entry points:
 *   fused_kernel_upscale_scalar       - upscale only (no downscale)
 *   fused_kernel_thirds_up_scalar     - combined thirds downscale + upscale
 *   fused_kernel_pow2_up_scalar       - combined pow2 downscale + upscale
 *   (HDR parallels in the HDR section at the end of this file)
 *
 * The upscale cascade is:
 *   Level 0 (2x): reads source, writes up_out[0]           (bilinear)
 *   Level k >= 1: reads up_out[k-1], writes up_out[k]       (bilinear 2x)
 *   Tail  (1.5x): reads source if cascade_depth==0,         (bilinear 2->3)
 *                 else reads up_out[cascade_depth-1]
 *
 * The scalar combined kernels run the existing downscale kernel and then
 * the upscale-only kernel back-to-back.  This reads source twice in the
 * combined case, which is fine for the scalar path because it is only
 * used as a misaligned-source fallback - performance is not a goal, and
 * keeping the existing downscale kernels untouched avoids risk.  The
 * genuinely fused single-pass behaviour lives in the SIMD kernels where
 * it matters for throughput.
 */

#include "internal.h"
#include "upscale_chunk.h"
#include <stdlib.h>
#include <string.h>


/* -----------------------------------------------------------------------
 * Bilinear 2x plane upscale from an arbitrary source buffer
 * -----------------------------------------------------------------------
 *
 * Produces dst of size (2*src_w × 2*src_h) from src (src_w × src_h).
 * - Row 2i+0 = horizontally-doubled copy of src[i]
 * - Row 2i+1 = horizontally-doubled copy of vert_avg(src[i], src[i+1])
 *              (src[h-1] replicated as "row h" at the bottom edge)
 */
/* Two-pass horizontal 2x expand of one row.  Pass 1 writes the bilinear
 * midpoints (avg of adjacent samples) contiguously into `tmp`; pass 2
 * interleaves the source row with `tmp` into `dst`:
 *      dst[2i] = src[i],  dst[2i+1] = avg(src[i], src[i+1])
 *
 * The obvious single fused loop (compute-and-interleave in one go) makes
 * the compiler fall back to an 8-wide widening sequence (punpcklbw->pavgw
 * ->pack) for the interleaved store.  Splitting into two contiguous passes
 * instead lets it emit a 16-wide pavgb pass and a punpck-based interleave
 * pass - roughly 2-4x the per-row throughput on x86-64 SSE2, which is what
 * closes the gap to libswscale's bilinear on AVX2-less CPUs.  `tmp` must be
 * distinct from both `src` and `dst`. */
static inline void up_h_2x_row_2pass_u8(const uint8_t *restrict src, int w,
                                        uint8_t *restrict dst,
                                        uint8_t *restrict tmp)
{
    int i;
    for (i = 0; i + 1 < w; i++)
        tmp[i] = up_avg_u8(src[i], src[i + 1]);
    if (w > 0)
        tmp[w - 1] = src[w - 1];          /* edge: avg(a,a) == a */
    for (i = 0; i < w; i++) {
        dst[2 * i + 0] = src[i];
        dst[2 * i + 1] = tmp[i];
    }
}

static void up_2x_plane_u8(const uint8_t *restrict src, int src_w, int src_h,
                           int src_stride, uint8_t *restrict dst, int dst_stride,
                           uint8_t *restrict vbuf, uint8_t *restrict hbuf)
{
    /* `vbuf`/`hbuf` are two persistent row buffers sized at init to the
     * widest row any upscale helper will process: vbuf holds the vertical
     * midpoint row, hbuf the horizontal two-pass temp.  Avoids per-frame
     * malloc. */
    if (!vbuf || !hbuf) return;

    for (int i = 0; i < src_h; i++) {
        const uint8_t *row_cur = src + (size_t)i * src_stride;
        const uint8_t *row_nxt = (i + 1 < src_h)
                                    ? src + (size_t)(i + 1) * src_stride
                                    : row_cur;

        /* Output row 2i+0 = horizontally-doubled row_cur */
        up_h_2x_row_2pass_u8(row_cur, src_w,
                             dst + (size_t)(2 * i) * dst_stride, hbuf);

        /* Vertical midpoint of row_cur and row_nxt */
        for (int x = 0; x < src_w; x++) {
            vbuf[x] = up_avg_u8(row_cur[x], row_nxt[x]);
        }

        /* Output row 2i+1 = horizontally-doubled midrow */
        up_h_2x_row_2pass_u8(vbuf, src_w,
                             dst + (size_t)(2 * i + 1) * dst_stride, hbuf);
    }
}


/* -----------------------------------------------------------------------
 * Bilinear 1.5x (2->3) plane upscale
 * -----------------------------------------------------------------------
 *
 * Produces dst of size (src_w*3/2 × src_h*3/2).  Both dimensions must be
 * even. The vertical pattern mirrors the horizontal 1.5x:
 *   output row 3j+0 = horizontally-1.5x of src[2j]
 *   output row 3j+1 = horizontally-1.5x of blend(src[2j], src[2j+1])  (1/3, 2/3)
 *   output row 3j+2 = horizontally-1.5x of blend(src[2j+1], src[2j+2]) (2/3, 1/3)
 *                     with src[h] replicated to src[h-1] at bottom edge
 */
static void up_1_5x_plane_u8(const uint8_t *restrict src, int src_w, int src_h,
                             int src_stride, uint8_t *restrict dst, int dst_stride,
                             uint8_t *restrict scratch)
{
    if (!scratch) return;

    int pairs_v = src_h / 2;
    for (int j = 0; j < pairs_v; j++) {
        const uint8_t *r2j   = src + (size_t)(2 * j)     * src_stride;
        const uint8_t *r2j1  = src + (size_t)(2 * j + 1) * src_stride;
        const uint8_t *r2j2  = (2 * j + 2 < src_h)
                                    ? src + (size_t)(2 * j + 2) * src_stride
                                    : r2j1;

        /* Row 3j+0: horizontally-1.5x of r2j (exact copy vertically) */
        up_h_1_5x_row_u8(r2j, src_w, dst + (size_t)(3 * j + 0) * dst_stride);

        /* Row 3j+1: vertical blend (1/3, 2/3) = (85, 171) of r2j and r2j1 */
        for (int x = 0; x < src_w; x++) {
            scratch[x] = up_blend_21_u8(r2j[x], r2j1[x]);
        }
        up_h_1_5x_row_u8(scratch, src_w, dst + (size_t)(3 * j + 1) * dst_stride);

        /* Row 3j+2: vertical blend (2/3, 1/3) of r2j1 and r2j2 */
        for (int x = 0; x < src_w; x++) {
            scratch[x] = up_blend_21_u8(r2j2[x], r2j1[x]);
        }
        up_h_1_5x_row_u8(scratch, src_w, dst + (size_t)(3 * j + 2) * dst_stride);
    }
}


/* -----------------------------------------------------------------------
 * Upscale one plane (Y, U, or V) through the full cascade + tail
 * -----------------------------------------------------------------------
 *
 * src is the source plane of size (src_w × src_h).
 * plane_idx: 0 = luma (Y), 1 = chroma (U or V); chroma widths are halved.
 *
 * For each of the N pow2 cascade levels, writes to the corresponding
 * p->up_out[k] buffer (luma or chroma plane as appropriate).  For the
 * optional 1.5x tail, writes to p->up_out[FUSED_UP_IDX_TAIL] reading from
 * src (if N==0) or from up_out[N-1] (if N>=1).
 */
static void upscale_plane_scalar(const fused_kernel_params_t *p,
                                 const uint8_t *src,
                                 int src_w, int src_h, int src_stride,
                                 int is_chroma)
{
    int N = p->upscale_cascade_depth;
    int tail = p->upscale_tail_1_5x;

    /* Level 0 (2x) reads source; writes up_out[0] */
    if (N >= 1 && (p->upscale_active & (1u << FUSED_UP_IDX_2X))) {
        uint8_t *dst;
        int dst_stride;
        if (!is_chroma) {
            dst = p->up_out[FUSED_UP_IDX_2X].plane_y;
            dst_stride = p->up_out[FUSED_UP_IDX_2X].y_stride;
        } else {
            dst = (is_chroma == 1) ? p->up_out[FUSED_UP_IDX_2X].plane_u
                                   : p->up_out[FUSED_UP_IDX_2X].plane_v;
            dst_stride = p->up_out[FUSED_UP_IDX_2X].uv_stride;
        }
        if (dst) up_2x_plane_u8(src, src_w, src_h, src_stride, dst, dst_stride,
                                p->upscale_scratch, p->upscale_scratch2);
    }

    /* Levels 1..N-1 cascade from up_out[k-1] -> up_out[k] */
    for (int k = 1; k < N; k++) {
        int src_up_w = src_w << k;         /* level k-1 width */
        int src_up_h = src_h << k;
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
            up_2x_plane_u8(src_up, src_up_w, src_up_h, src_up_stride,
                           dst, dst_stride, p->upscale_scratch, p->upscale_scratch2);
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
            up_1_5x_plane_u8(tail_src, tail_src_w, tail_src_h, tail_src_stride,
                             dst, dst_stride, p->upscale_scratch);
        }
    }
}


/* -----------------------------------------------------------------------
 * Public entry points - SDR
 * ----------------------------------------------------------------------- */

void fused_kernel_upscale_scalar(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    /* Luma plane */
    upscale_plane_scalar(p, src_y, p->src_width, p->src_height,
                         p->src_y_stride, 0);
    /* Chroma planes (half width and half height) */
    upscale_plane_scalar(p, src_u, p->src_width / 2, p->src_height / 2,
                         p->src_uv_stride, 1);
    upscale_plane_scalar(p, src_v, p->src_width / 2, p->src_height / 2,
                         p->src_uv_stride, 2);
}

void fused_kernel_thirds_up_scalar(const fused_kernel_params_t *p,
                                   const uint8_t *src_y,
                                   const uint8_t *src_u,
                                   const uint8_t *src_v)
{
    /* Run the existing thirds downscale kernel, then the upscale kernel.
     * See file header for the rationale: scalar is the misaligned-source
     * fallback path only; single-pass source reads are not a performance
     * concern here.  The SIMD combined kernels are genuinely fused. */
    if (p->active_outputs != 0) {
        fused_kernel_thirds_scalar(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_scalar(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_scalar(const fused_kernel_params_t *p,
                                 const uint8_t *src_y,
                                 const uint8_t *src_u,
                                 const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_scalar(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_scalar(p, src_y, src_u, src_v);
    }
}


/* =======================================================================
 * HDR - same structure with uint16_t planes and 10-bit-aware helpers.
 * ======================================================================= */

/* Two-pass horizontal 2x expand for 10-bit rows.  See up_h_2x_row_2pass_u8
 * for the rationale; here the vectorized passes use pavgw / 16-bit punpck. */
static inline void up_h_2x_row_2pass_u16(const uint16_t *restrict src, int w,
                                         uint16_t *restrict dst,
                                         uint16_t *restrict tmp)
{
    int i;
    for (i = 0; i + 1 < w; i++)
        tmp[i] = up_avg_u16(src[i], src[i + 1]);
    if (w > 0)
        tmp[w - 1] = src[w - 1];
    for (i = 0; i < w; i++) {
        dst[2 * i + 0] = src[i];
        dst[2 * i + 1] = tmp[i];
    }
}

static void up_2x_plane_u16(const uint16_t *restrict src, int src_w, int src_h,
                            int src_el_stride, uint16_t *restrict dst, int dst_el_stride,
                            uint16_t *restrict vbuf, uint16_t *restrict hbuf)
{
    if (!vbuf || !hbuf) return;

    for (int i = 0; i < src_h; i++) {
        const uint16_t *row_cur = src + (size_t)i * src_el_stride;
        const uint16_t *row_nxt = (i + 1 < src_h)
                                    ? src + (size_t)(i + 1) * src_el_stride
                                    : row_cur;

        up_h_2x_row_2pass_u16(row_cur, src_w,
                              dst + (size_t)(2 * i) * dst_el_stride, hbuf);

        for (int x = 0; x < src_w; x++) {
            vbuf[x] = up_avg_u16(row_cur[x], row_nxt[x]);
        }

        up_h_2x_row_2pass_u16(vbuf, src_w,
                              dst + (size_t)(2 * i + 1) * dst_el_stride, hbuf);
    }
}

static void up_1_5x_plane_u16(const uint16_t *restrict src, int src_w, int src_h,
                              int src_el_stride, uint16_t *restrict dst, int dst_el_stride,
                              uint16_t *restrict scratch)
{
    if (!scratch) return;

    int pairs_v = src_h / 2;
    for (int j = 0; j < pairs_v; j++) {
        const uint16_t *r2j  = src + (size_t)(2 * j)     * src_el_stride;
        const uint16_t *r2j1 = src + (size_t)(2 * j + 1) * src_el_stride;
        const uint16_t *r2j2 = (2 * j + 2 < src_h)
                                    ? src + (size_t)(2 * j + 2) * src_el_stride
                                    : r2j1;

        up_h_1_5x_row_u16(r2j, src_w,
                          dst + (size_t)(3 * j + 0) * dst_el_stride);

        for (int x = 0; x < src_w; x++) {
            scratch[x] = up_blend_21_u16(r2j[x], r2j1[x]);
        }
        up_h_1_5x_row_u16(scratch, src_w,
                          dst + (size_t)(3 * j + 1) * dst_el_stride);

        for (int x = 0; x < src_w; x++) {
            scratch[x] = up_blend_21_u16(r2j2[x], r2j1[x]);
        }
        up_h_1_5x_row_u16(scratch, src_w,
                          dst + (size_t)(3 * j + 2) * dst_el_stride);
    }
}

static void upscale_plane_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src,
                                     int src_w, int src_h, int src_el_stride,
                                     int is_chroma)
{
    int N = p->upscale_cascade_depth;
    int tail = p->upscale_tail_1_5x;

    /* Level 0 (2x) */
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
        if (dst) up_2x_plane_u16(src, src_w, src_h, src_el_stride, dst, dst_el_stride,
                                 p->upscale_scratch_hdr, p->upscale_scratch_hdr2);
    }

    /* Levels 1..N-1 */
    for (int k = 1; k < N; k++) {
        int src_up_w = src_w << k;
        int src_up_h = src_h << k;
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
            up_2x_plane_u16(src_up, src_up_w, src_up_h, src_up_el_stride,
                            dst, dst_el_stride, p->upscale_scratch_hdr, p->upscale_scratch_hdr2);
        }
    }

    /* 1.5x tail */
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
            up_1_5x_plane_u16(tail_src, tail_src_w, tail_src_h, tail_src_el_stride,
                              dst, dst_el_stride, p->upscale_scratch_hdr);
        }
    }
}

void fused_kernel_upscale_hdr_scalar(const fused_hdr_kernel_params_t *p,
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

    upscale_plane_hdr_scalar(p, src_y, p->src_width, p->src_height,
                             p->src_y_el_stride, 0);
    upscale_plane_hdr_scalar(p, up_src_u, p->src_width / 2, p->src_height / 2,
                             up_src_uv_el_stride, 1);
    upscale_plane_hdr_scalar(p, up_src_v, p->src_width / 2, p->src_height / 2,
                             up_src_uv_el_stride, 2);
}

void fused_kernel_thirds_up_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                       const uint16_t *src_y,
                                       const uint16_t *src_u,
                                       const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_hdr_scalar(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_scalar(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                     const uint16_t *src_y,
                                     const uint16_t *src_u,
                                     const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_hdr_scalar(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_scalar(p, src_y, src_u, src_v);
    }
}
