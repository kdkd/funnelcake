/*
 * kernels_hdr_scalar.c -- scalar (non-SIMD) fused downscale kernels for
 *                         10-bit HDR content (uint16_t samples).
 *
 * Two entry points:
 *   fused_kernel_pow2_hdr_scalar   -- power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_hdr_scalar -- thirds family (1.5x/3x/6x/12x)
 *
 * Both process YUV420 frames plane-by-plane (I010) or with interleaved
 * chroma deinterleaving (P010).  All arithmetic uses uint32_t intermediates
 * and multiply-and-shift in place of division; 10-bit values (max 1023)
 * require wider intermediates than the 8-bit kernels because products like
 * 1023 * 171 = 174,933 overflow uint16_t.
 *
 * Pointer parameters are restrict-qualified to aid auto-vectorization.
 */

#include "internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */

/* Pairwise average of two uint16 values, with rounding (matches vpavgw).
 * Uses uint32_t intermediate because (1023 + 1023 + 1) = 2047 fits in
 * uint16_t, but we widen for safety and consistency with the rest. */
static inline uint16_t avg_u16(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a + (uint32_t)b + 1) >> 1);
}

/* Bilinear blend: (a * 171 + b * 85 + 128) >> 8  ~  a*2/3 + b*1/3
 * Must use uint32_t: 10-bit a * 171 = max 174,933 which overflows uint16_t. */
static inline uint16_t blend_2_1_u16(uint16_t a, uint16_t b)
{
    return (uint16_t)(((uint32_t)a * 171 + (uint32_t)b * 85 + 128) >> 8);
}

/* Integer divide-by-3 for a sum of 3 uint16 values (max sum = 3069).
 * 21846 / 65536 ~ 1/3.  Exact for sum <= 3069 (max 3 * 1023).
 * Uses uint32_t multiply: 3069 * 21846 = 67,027,674 which fits in uint32_t. */
static inline uint16_t div3_u32(uint32_t sum)
{
    return (uint16_t)(((uint32_t)sum * 21846u) >> 16);
}


/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single 10-bit plane
 *
 * Same algorithm as the 8-bit scale_plane_pow2 but operating on uint16_t
 * samples with uint32_t arithmetic.  Strides are in bytes; element
 * indexing uses stride/2 to convert to uint16_t offsets.
 *
 * The vertical cascade processes groups of source rows:
 *   2 source rows -> 1 row at 2x
 *   4 source rows -> 1 row at 4x  (average two 2x rows)
 *   8 source rows -> 1 row at 8x  (average two 4x rows)
 *  16 source rows -> 1 row at 16x (average two 8x rows)
 *
 * Each vertically-reduced row is then horizontally reduced by cascading
 * pairwise column averages to produce the output width.
 * ----------------------------------------------------------------------- */

static void scale_plane_pow2_hdr(
    const uint16_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    /* Arrays indexed by cascade level 0..3, corresponding to:
     *   level 0 = 2x  (bit 1)
     *   level 1 = 4x  (bit 3)
     *   level 2 = 8x  (bit 5)
     *   level 3 = 16x (bit 7)
     */
    uint16_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    (void)dst_heights;

    /* Convert byte stride to element stride for source indexing */
    int src_el_stride = src_stride / (int)sizeof(uint16_t);

    /* Determine deepest active level (0=2x .. 3=16x). */
    int deepest = -1;
    /* Map cascade level -> bit position:  level k -> bit (2k+1) */
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) {
            deepest = k;
            break;
        }
    }
    if (deepest < 0) return;  /* nothing to do */

    /* How many source rows per output-row group at the deepest level.
     * 2x -> 2, 4x -> 4, 8x -> 8, 16x -> 16. */
    int group_rows = (2 << deepest);
    int num_groups = src_h / group_rows;

    /* Carve scratch buffers from the persistent pool (init-time alloc). */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint16_t *vert_buf[4] = { NULL, NULL, NULL, NULL };
    int vert_rows[4];

    for (int k = 0; k <= deepest; k++) {
        vert_rows[k] = group_rows >> (k + 1);
        vert_buf[k] = (uint16_t *)fused_scratch_alloc(
            &scratch, (size_t)vert_rows[k] * (size_t)src_w * sizeof(uint16_t));
        if (!vert_buf[k]) return;
    }

    /* Horizontal cascade buffer: max input width = src_w elements. */
    uint16_t *h_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)src_w * sizeof(uint16_t));
    if (!h_buf) return;

    /* Output row cursor per level */
    int out_row[4] = { 0, 0, 0, 0 };

    for (int g = 0; g < num_groups; g++) {
        const uint16_t *grp_base = src + (size_t)g * (size_t)group_rows * (size_t)src_el_stride;

        /* -- Vertical cascade ----------------------------------------- */

        /* Level 0 (2x vertical): pair-average source rows */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint16_t *restrict ra = grp_base + (size_t)(2 * r)     * (size_t)src_el_stride;
            const uint16_t *restrict rb = grp_base + (size_t)(2 * r + 1) * (size_t)src_el_stride;
            uint16_t *restrict dst_row  = vert_buf[0] + (size_t)r * (size_t)src_w;
            for (int x = 0; x < src_w; x++) {
                dst_row[x] = avg_u16(ra[x], rb[x]);
            }
        }

        /* Deeper levels: pair-average previous level */
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict ra = vert_buf[k - 1] + (size_t)(2 * r)     * (size_t)src_w;
                const uint16_t *restrict rb = vert_buf[k - 1] + (size_t)(2 * r + 1) * (size_t)src_w;
                uint16_t *restrict dst_row  = vert_buf[k] + (size_t)r * (size_t)src_w;
                for (int x = 0; x < src_w; x++) {
                    dst_row[x] = avg_u16(ra[x], rb[x]);
                }
            }
        }

        /* -- Horizontal cascade + output write ------------------------ */
        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;

            /* Convert destination byte stride to element stride */
            int dst_el_stride = dst_strides[k] / (int)sizeof(uint16_t);

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *restrict vert_row = vert_buf[k] + (size_t)r * (size_t)src_w;

                /* Horizontal cascade: (k+1) halvings from src_w to dst_widths[k]. */
                int cur_w = src_w;
                const uint16_t *cur_src = vert_row;

                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    int next_w = cur_w >> 1;
                    for (int x = 0; x < next_w; x++) {
                        h_buf[x] = avg_u16(cur_src[2 * x], cur_src[2 * x + 1]);
                    }
                    cur_w = next_w;
                    cur_src = h_buf;
                }

                /* Write to output plane */
                uint16_t *restrict out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_el_stride;
                memcpy(out, h_buf, (size_t)dst_widths[k] * sizeof(uint16_t));
                out_row[k]++;
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}


/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single 10-bit plane
 *
 * Same algorithm as the 8-bit scale_plane_thirds but operating on
 * uint16_t samples with uint32_t arithmetic.
 *
 * Vertical cascade (per 6 source rows, or 12 if 12x active):
 *   Step 1: Pairwise row averages of 6 rows -> 3 intermediates (v01, v23, v45)
 *   Step 2: 1.5x output = bilinear from 3 intermediates -> 4 rows
 *   Step 3: 3x output = avg pairs of intermediates -> 2 rows
 *   Step 4: 6x output = avg the two 3x rows -> 1 row
 *   Step 5: 12x output = avg two 6x rows from consecutive 6-row groups -> 1 row
 *
 * Horizontal:
 *   1.5x: 3:2 bilinear (every 3 src pixels -> 2 dst pixels)
 *   3x:   3:1 box average (every 3 src pixels -> 1 dst pixel)
 *   6x:   cascade from 3x (pairwise average 3x result)
 *   12x:  cascade from 6x (pairwise average 6x result)
 * ----------------------------------------------------------------------- */

/* Horizontal 1.5x filter: 3:2 bilinear reduction.
 * For every 3 source pixels, produce 2 output pixels.
 *   out[0] = blend(src[0], src[1]) = src[0]*171/256 + src[1]*85/256
 *   out[1] = blend(src[2], src[1]) = src[2]*171/256 + src[1]*85/256
 * Uses uint32_t intermediates via blend_2_1_u16. */
static void h_filter_1_5x_hdr(
    const uint16_t *restrict src, int src_w,
    uint16_t *restrict dst, int dst_w)
{
    int x_out = 0;
    for (int x_in = 0; x_in < src_w - 2 && x_out < dst_w - 1; x_in += 3, x_out += 2) {
        dst[x_out]     = blend_2_1_u16(src[x_in],     src[x_in + 1]);
        dst[x_out + 1] = blend_2_1_u16(src[x_in + 2], src[x_in + 1]);
    }
}

/* Horizontal 3x filter: box average of 3 source pixels.
 * Sum uses uint32_t because 3 * 1023 = 3069 fits in uint16_t, but we use
 * uint32_t for the multiply in div3_u32. */
static void h_filter_3x_hdr(
    const uint16_t *restrict src, int src_w,
    uint16_t *restrict dst, int dst_w)
{
    (void)src_w;
    for (int x = 0; x < dst_w; x++) {
        uint32_t sum = (uint32_t)src[3 * x]
                     + (uint32_t)src[3 * x + 1]
                     + (uint32_t)src[3 * x + 2];
        dst[x] = div3_u32(sum);
    }
}

/* Horizontal pairwise halving (used for 6x from 3x, and 12x from 6x). */
static void h_filter_halve_hdr(
    const uint16_t *restrict src,
    uint16_t *restrict dst, int dst_w)
{
    for (int x = 0; x < dst_w; x++) {
        dst[x] = avg_u16(src[2 * x], src[2 * x + 1]);
    }
}

static void scale_plane_thirds_hdr(
    const uint16_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    /* Indexed by thirds cascade level:
     *   level 0 = 1.5x (bit 0)
     *   level 1 = 3x   (bit 2)
     *   level 2 = 6x   (bit 4)
     *   level 3 = 12x  (bit 6)
     */
    uint16_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    (void)dst_heights;

    /* Convert byte stride to element stride */
    int src_el_stride = src_stride / (int)sizeof(uint16_t);

    static const int bit_pos[4] = { 0, 2, 4, 6 };

    /* Determine deepest active level. */
    int deepest = -1;
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) {
            deepest = k;
            break;
        }
    }
    if (deepest < 0) return;

    int need_12x = (deepest >= 3);

    /* The base processing unit is 6 source rows. */
    int base6_groups = src_h / 6;

    /* Carve scratch buffers from the persistent pool (init-time alloc). */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    size_t row_bytes = (size_t)src_w * sizeof(uint16_t);

    /* Vertical intermediates */
    uint16_t *v01  = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v23  = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v45  = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);

    /* 3x vertical intermediates (2 rows) */
    uint16_t *v3x_0 = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v3x_1 = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);

    /* 6x vertical intermediate (1 row) */
    uint16_t *v6x = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);

    /* Horizontal scratch buffers */
    int w_3x = src_w / 3;
    uint16_t *h_3x_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)w_3x * sizeof(uint16_t));

    int w_6x = w_3x / 2;
    uint16_t *h_6x_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)(w_6x > 0 ? w_6x : 1) * sizeof(uint16_t));

    /* For 12x: need to save a 6x row from the first half of the 12-row group */
    uint16_t *v6x_prev = need_12x
        ? (uint16_t *)fused_scratch_alloc(&scratch, row_bytes) : NULL;

    /* For 1.5x blended rows (rows 1 and 2 of the 4-row output) */
    int need_1_5x = (active_outputs & (1u << 0)) != 0;
    uint16_t *blend_tmp = need_1_5x
        ? (uint16_t *)fused_scratch_alloc(&scratch, row_bytes) : NULL;

    if (!v01 || !v23 || !v45 || !v3x_0 || !v3x_1 || !v6x ||
        !h_3x_buf || !h_6x_buf ||
        (need_12x && !v6x_prev) || (need_1_5x && !blend_tmp)) {
        return;
    }

    /* Output row cursors */
    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint16_t *grp = src + (size_t)g6 * 6 * (size_t)src_el_stride;

        const uint16_t *restrict row0 = grp;
        const uint16_t *restrict row1 = grp + (size_t)src_el_stride;
        const uint16_t *restrict row2 = grp + (size_t)2 * (size_t)src_el_stride;
        const uint16_t *restrict row3 = grp + (size_t)3 * (size_t)src_el_stride;
        const uint16_t *restrict row4 = grp + (size_t)4 * (size_t)src_el_stride;
        const uint16_t *restrict row5 = grp + (size_t)5 * (size_t)src_el_stride;

        /* Step 1: pairwise row averages */
        for (int x = 0; x < src_w; x++) {
            v01[x] = avg_u16(row0[x], row1[x]);
            v23[x] = avg_u16(row2[x], row3[x]);
            v45[x] = avg_u16(row4[x], row5[x]);
        }

        /* Step 3: 3x vertical (always computed if deepest >= 1) */
        if (deepest >= 1) {
            for (int x = 0; x < src_w; x++) {
                v3x_0[x] = avg_u16(v01[x], v23[x]);
                v3x_1[x] = avg_u16(v23[x], v45[x]);
            }
        }

        /* Step 4: 6x vertical (always computed if deepest >= 2) */
        if (deepest >= 2) {
            for (int x = 0; x < src_w; x++) {
                v6x[x] = avg_u16(v3x_0[x], v3x_1[x]);
            }
        }

        /* -- Output 1.5x (bit 0) ----------------------------------- */
        if (active_outputs & (1u << 0)) {
            /* 1.5x vertical: 6 source rows -> 4 output rows (ratio 3:2).
             *   Row 0: v01
             *   Row 1: blend_2_1(v01, v23)
             *   Row 2: blend_2_1(v23, v45)
             *   Row 3: v45
             * Each row is then horizontally 1.5x-filtered. */
            int dw = dst_widths[0];
            int ds_el = dst_strides[0] / (int)sizeof(uint16_t);

            /* Row 0: v01 -> horizontal 1.5x */
            h_filter_1_5x_hdr(v01, src_w,
                              dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el, dw);
            out_row[0]++;

            /* Row 1: blend(v01, v23) -> horizontal 1.5x */
            for (int x = 0; x < src_w; x++)
                blend_tmp[x] = blend_2_1_u16(v01[x], v23[x]);
            h_filter_1_5x_hdr(blend_tmp, src_w,
                              dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el, dw);
            out_row[0]++;

            /* Row 2: blend(v23, v45) -> horizontal 1.5x */
            for (int x = 0; x < src_w; x++)
                blend_tmp[x] = blend_2_1_u16(v23[x], v45[x]);
            h_filter_1_5x_hdr(blend_tmp, src_w,
                              dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el, dw);
            out_row[0]++;

            /* Row 3: v45 -> horizontal 1.5x */
            h_filter_1_5x_hdr(v45, src_w,
                              dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el, dw);
            out_row[0]++;
        }

        /* -- Output 3x (bit 2) ------------------------------------- */
        if (active_outputs & (1u << 2)) {
            int dw = dst_widths[1];
            int ds_el = dst_strides[1] / (int)sizeof(uint16_t);

            /* 3x vertical produced 2 rows: v3x_0 and v3x_1 (at src_w).
             * Horizontal: 3:1 box average -> dw pixels */
            h_filter_3x_hdr(v3x_0, src_w,
                            dst_planes[1] + (size_t)out_row[1] * (size_t)ds_el, dw);
            out_row[1]++;

            h_filter_3x_hdr(v3x_1, src_w,
                            dst_planes[1] + (size_t)out_row[1] * (size_t)ds_el, dw);
            out_row[1]++;
        }

        /* -- Output 6x (bit 4) ------------------------------------- */
        if (active_outputs & (1u << 4)) {
            int dw = dst_widths[2];
            int ds_el = dst_strides[2] / (int)sizeof(uint16_t);

            /* 6x vertical produced 1 row: v6x (at src_w).
             * Horizontal: first do 3x horizontal (3:1 box avg -> w_3x),
             * then pairwise halve (-> w_6x = dw). */
            h_filter_3x_hdr(v6x, src_w, h_3x_buf, w_3x);
            h_filter_halve_hdr(h_3x_buf,
                               dst_planes[2] + (size_t)out_row[2] * (size_t)ds_el, dw);
            out_row[2]++;
        }

        /* -- 12x handling: save 6x row from this 6-row group ------- */
        if (need_12x) {
            if ((g6 & 1) == 0) {
                /* Save the 6x vertical row */
                memcpy(v6x_prev, v6x, row_bytes);
            } else {
                /* Average v6x_prev and v6x, then horizontal filter */
                uint16_t *v12x_row = v6x_prev;  /* reuse as output */
                for (int x = 0; x < src_w; x++) {
                    v12x_row[x] = avg_u16(v6x_prev[x], v6x[x]);
                }

                if (active_outputs & (1u << 6)) {
                    int dw = dst_widths[3];
                    int ds_el = dst_strides[3] / (int)sizeof(uint16_t);

                    /* Horizontal: 3x box avg -> halve -> halve = 12:1 reduction */
                    h_filter_3x_hdr(v12x_row, src_w, h_3x_buf, w_3x);
                    h_filter_halve_hdr(h_3x_buf, h_6x_buf, w_6x);
                    h_filter_halve_hdr(h_6x_buf,
                                       dst_planes[3] + (size_t)out_row[3] * (size_t)ds_el, dw);
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
 * For I010 (planar): call scale_plane_*_hdr() three times (Y, U, V).
 * For P010 (semi-planar): Y is still a separate plane; U and V are
 * interleaved in src_u as UVUV... pairs.  We deinterleave one chroma
 * row at a time into stack-allocated scratch buffers before calling
 * the per-plane scale function.
 *
 * The deinterleaving approach: allocate a temporary planar U and V
 * buffer at the chroma source dimensions, deinterleave the entire
 * chroma plane up front, then process U and V identically to I010.
 * This is simpler than modifying the inner loops and costs only one
 * extra pass over the chroma data.
 * ----------------------------------------------------------------------- */

void fused_kernel_pow2_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src_y,
                                  const uint16_t *src_u,
                                  const uint16_t *src_v)
{
    /* Map bit positions to cascade level arrays.
     * bit 1 = 2x  -> level 0
     * bit 3 = 4x  -> level 1
     * bit 5 = 8x  -> level 2
     * bit 7 = 16x -> level 3
     */
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    uint16_t *y_planes[4], *u_planes[4], *v_planes[4];
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
    scale_plane_pow2_hdr(src_y,
                         p->src_width, p->src_height, p->src_y_stride,
                         p->active_outputs,
                         y_planes, y_widths, y_strides, y_heights,
                         p->scratch_pool, p->scratch_pool_size);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        /* P010: src_u points to interleaved UV plane (UVUV...).
         * Deinterleave into pre-allocated planar buffers, then process
         * each plane normally. */
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        for (int y = 0; y < chroma_h; y++) {
            const uint16_t *row = src_u + (size_t)y * (size_t)uv_el_stride;
            uint16_t *u_row = p->p010_tmp_u + (size_t)y * (size_t)planar_el_stride;
            uint16_t *v_row = p->p010_tmp_v + (size_t)y * (size_t)planar_el_stride;
            for (int x = 0; x < chroma_w; x++) {
                u_row[x] = row[2 * x];
                v_row[x] = row[2 * x + 1];
            }
        }

        scale_plane_pow2_hdr(p->p010_tmp_u,
                             chroma_w, chroma_h, p->p010_tmp_stride,
                             p->active_outputs,
                             u_planes, uv_widths, uv_strides, uv_heights,
                             p->scratch_pool, p->scratch_pool_size);

        scale_plane_pow2_hdr(p->p010_tmp_v,
                             chroma_w, chroma_h, p->p010_tmp_stride,
                             p->active_outputs,
                             v_planes, uv_widths, uv_strides, uv_heights,
                             p->scratch_pool, p->scratch_pool_size);
        /* No free - buffers are owned by the context */
    } else {
        /* I010: separate U and V planes */
        scale_plane_pow2_hdr(src_u,
                             chroma_w, chroma_h, p->src_uv_stride,
                             p->active_outputs,
                             u_planes, uv_widths, uv_strides, uv_heights,
                             p->scratch_pool, p->scratch_pool_size);

        scale_plane_pow2_hdr(src_v,
                             chroma_w, chroma_h, p->src_uv_stride,
                             p->active_outputs,
                             v_planes, uv_widths, uv_strides, uv_heights,
                             p->scratch_pool, p->scratch_pool_size);
    }
}


void fused_kernel_thirds_hdr_scalar(const fused_hdr_kernel_params_t *p,
                                    const uint16_t *src_y,
                                    const uint16_t *src_u,
                                    const uint16_t *src_v)
{
    /* Map bit positions to cascade level arrays.
     * bit 0 = 1.5x -> level 0
     * bit 2 = 3x   -> level 1
     * bit 4 = 6x   -> level 2
     * bit 6 = 12x  -> level 3
     */
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    uint16_t *y_planes[4], *u_planes[4], *v_planes[4];
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
    scale_plane_thirds_hdr(src_y,
                           p->src_width, p->src_height, p->src_y_stride,
                           p->active_outputs,
                           y_planes, y_widths, y_strides, y_heights,
                           p->scratch_pool, p->scratch_pool_size);

    /* Chroma planes */
    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        /* P010: deinterleave UV plane into pre-allocated planar buffers */
        int uv_el_stride = p->src_uv_el_stride;
        int planar_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);

        for (int y = 0; y < chroma_h; y++) {
            const uint16_t *row = src_u + (size_t)y * (size_t)uv_el_stride;
            uint16_t *u_row = p->p010_tmp_u + (size_t)y * (size_t)planar_el_stride;
            uint16_t *v_row = p->p010_tmp_v + (size_t)y * (size_t)planar_el_stride;
            for (int x = 0; x < chroma_w; x++) {
                u_row[x] = row[2 * x];
                v_row[x] = row[2 * x + 1];
            }
        }

        scale_plane_thirds_hdr(p->p010_tmp_u,
                               chroma_w, chroma_h, p->p010_tmp_stride,
                               p->active_outputs,
                               u_planes, uv_widths, uv_strides, uv_heights,
                               p->scratch_pool, p->scratch_pool_size);

        scale_plane_thirds_hdr(p->p010_tmp_v,
                               chroma_w, chroma_h, p->p010_tmp_stride,
                               p->active_outputs,
                               v_planes, uv_widths, uv_strides, uv_heights,
                               p->scratch_pool, p->scratch_pool_size);
        /* No free - buffers are owned by the context */
    } else {
        /* I010: separate U and V planes */
        scale_plane_thirds_hdr(src_u,
                               chroma_w, chroma_h, p->src_uv_stride,
                               p->active_outputs,
                               u_planes, uv_widths, uv_strides, uv_heights,
                               p->scratch_pool, p->scratch_pool_size);

        scale_plane_thirds_hdr(src_v,
                               chroma_w, chroma_h, p->src_uv_stride,
                               p->active_outputs,
                               v_planes, uv_widths, uv_strides, uv_heights,
                               p->scratch_pool, p->scratch_pool_size);
    }
}
