/*
 * kernels_scalar.c -- scalar (non-SIMD) fused downscale kernels.
 *
 * Two entry points:
 *   fused_kernel_pow2_scalar   -- power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_scalar -- thirds family (1.5x/3x/6x/12x)
 *
 * Both process YUV420 I420 frames plane-by-plane.  All arithmetic uses
 * uint16_t intermediates and multiply-and-shift in place of division.
 * Pointer parameters are restrict-qualified to aid auto-vectorization.
 */

#include "internal.h"
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */

/* Pairwise average of two uint8 values, with rounding (matches vpavgb). */
static inline uint8_t avg_u8(uint8_t a, uint8_t b)
{
    return (uint8_t)(((uint16_t)a + (uint16_t)b + 1) >> 1);
}

/* Bilinear blend: (a * 171 + b * 85 + 128) >> 8  ~  a*2/3 + b*1/3 */
static inline uint8_t blend_2_1(uint8_t a, uint8_t b)
{
    return (uint8_t)(((uint16_t)a * 171 + (uint16_t)b * 85 + 128) >> 8);
}

/* Integer divide-by-3 for a sum of 3 uint8 values (max sum = 765). */
static inline uint8_t div3_u16(uint16_t sum)
{
    return (uint8_t)((sum * (uint32_t)0x5556) >> 16);
}


/* -----------------------------------------------------------------------
 * Power-of-two kernel: scale a single plane
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

static void scale_plane_pow2(
    const uint8_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    /* Arrays indexed by cascade level 0..3, corresponding to:
     *   level 0 = 2x  (bit 1)
     *   level 1 = 4x  (bit 3)
     *   level 2 = 8x  (bit 5)
     *   level 3 = 16x (bit 7)
     */
    uint8_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    (void)dst_heights;

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

    /* Allocate row buffers for vertical intermediates.
     * We need up to 4 cascade levels, each halving the row count.
     * Level 0 (2x): group_rows/2 rows
     * Level 1 (4x): group_rows/4 rows
     * Level 2 (8x): group_rows/8 rows (=1)
     * Level 3 (16x): group_rows/16 rows (=1) -- only if group_rows=16
     *
     * Actually, the cascade processes:
     *   Step 1: pair-average group_rows source rows -> group_rows/2 rows (2x vert)
     *   Step 2: pair-average those -> group_rows/4 rows (4x vert)
     *   ...
     * We only need 2 row-buffers: one for current level, one for next.
     * But it's simpler to allocate one buffer per level.
     *
     * Max rows at level 0 = group_rows/2 = at most 8 (when group_rows=16).
     * Each row is src_w bytes.
     */
    uint8_t *vert_buf[4] = { NULL, NULL, NULL, NULL };
    int vert_rows[4];  /* how many rows at each level */

    /* Carve scratch buffers from the pool allocated at init time. */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    for (int k = 0; k <= deepest; k++) {
        vert_rows[k] = group_rows >> (k + 1);  /* 2x: group/2, 4x: group/4, etc. */
        vert_buf[k] = (uint8_t *)fused_scratch_alloc(
            &scratch, (size_t)vert_rows[k] * (size_t)src_w);
        if (!vert_buf[k]) return;
    }

    /* Horizontal cascade buffer: we do horizontal halving in-place using a
     * scratch buffer.  Max input width = src_w. */
    uint8_t *h_buf = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)src_w);
    if (!h_buf) return;

    /* Output row cursor per level */
    int out_row[4] = { 0, 0, 0, 0 };

    for (int g = 0; g < num_groups; g++) {
        const uint8_t *grp_base = src + (size_t)g * (size_t)group_rows * (size_t)src_stride;

        /* -- Vertical cascade ----------------------------------------- */

        /* Level 0 (2x vertical): pair-average source rows */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint8_t *restrict ra = grp_base + (size_t)(2 * r)     * (size_t)src_stride;
            const uint8_t *restrict rb = grp_base + (size_t)(2 * r + 1) * (size_t)src_stride;
            uint8_t *restrict dst_row  = vert_buf[0] + (size_t)r * (size_t)src_w;
            for (int x = 0; x < src_w; x++) {
                dst_row[x] = avg_u8(ra[x], rb[x]);
            }
        }

        /* Deeper levels: pair-average previous level */
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint8_t *restrict ra = vert_buf[k - 1] + (size_t)(2 * r)     * (size_t)src_w;
                const uint8_t *restrict rb = vert_buf[k - 1] + (size_t)(2 * r + 1) * (size_t)src_w;
                uint8_t *restrict dst_row  = vert_buf[k] + (size_t)r * (size_t)src_w;
                for (int x = 0; x < src_w; x++) {
                    dst_row[x] = avg_u8(ra[x], rb[x]);
                }
            }
        }

        /* -- Horizontal cascade + output write ------------------------ */
        /* For each level that is active, take its vertical intermediate
         * rows (at full source width) and cascade horizontally to produce
         * the output at that level's width.
         *
         * Horizontal cascade: level k needs (k+1) halvings from src_w.
         *   2x output: 1 halving
         *   4x output: 2 halvings
         *   8x output: 3 halvings
         *   16x output: 4 halvings
         *
         * We can share horizontal work: first halving produces 2x width,
         * then halving that produces 4x width, etc.  But that only helps
         * if multiple levels are active.  For simplicity and to keep the
         * code auto-vectorizable (no branching in inner loop), we process
         * each active level independently.
         *
         * Actually, let's cascade: for each vertical row at each level,
         * do horizontal halvings starting from the vertical result.
         */
        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint8_t *restrict vert_row = vert_buf[k] + (size_t)r * (size_t)src_w;

                /* Horizontal cascade: (k+1) halvings from src_w to dst_widths[k].
                 * Each halving: out[x] = avg(in[2x], in[2x+1]).
                 * After the first halving, subsequent halvings read and
                 * write h_buf in-place (safe because reads are always ahead
                 * of writes: read position 2*x >= write position x). */
                int cur_w = src_w;
                const uint8_t *cur_src = vert_row;

                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    int next_w = cur_w >> 1;
                    for (int x = 0; x < next_w; x++) {
                        h_buf[x] = avg_u8(cur_src[2 * x], cur_src[2 * x + 1]);
                    }
                    cur_w = next_w;
                    cur_src = h_buf;
                }

                /* Write to output plane */
                uint8_t *restrict out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_strides[k];
                memcpy(out, h_buf, (size_t)dst_widths[k]);
                out_row[k]++;
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}


/* -----------------------------------------------------------------------
 * Thirds kernel: scale a single plane
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
 */
static void h_filter_1_5x(
    const uint8_t *restrict src, int src_w,
    uint8_t *restrict dst, int dst_w)
{
    int x_out = 0;
    for (int x_in = 0; x_in < src_w - 2 && x_out < dst_w - 1; x_in += 3, x_out += 2) {
        dst[x_out]     = blend_2_1(src[x_in],     src[x_in + 1]);
        dst[x_out + 1] = blend_2_1(src[x_in + 2], src[x_in + 1]);
    }
}

/* Horizontal 3x filter: box average of 3 source pixels. */
static void h_filter_3x(
    const uint8_t *restrict src, int src_w,
    uint8_t *restrict dst, int dst_w)
{
    (void)src_w;
    for (int x = 0; x < dst_w; x++) {
        uint16_t sum = (uint16_t)src[3 * x]
                     + (uint16_t)src[3 * x + 1]
                     + (uint16_t)src[3 * x + 2];
        dst[x] = div3_u16(sum);
    }
}

/* Horizontal pairwise halving (used for 6x from 3x, and 12x from 6x). */
static void h_filter_halve(
    const uint8_t *restrict src,
    uint8_t *restrict dst, int dst_w)
{
    for (int x = 0; x < dst_w; x++) {
        dst[x] = avg_u8(src[2 * x], src[2 * x + 1]);
    }
}

static void scale_plane_thirds(
    const uint8_t *restrict src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    /* Indexed by thirds cascade level:
     *   level 0 = 1.5x (bit 0)
     *   level 1 = 3x   (bit 2)
     *   level 2 = 6x   (bit 4)
     *   level 3 = 12x  (bit 6)
     */
    uint8_t *restrict dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    int dst_heights[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    (void)dst_heights;

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

    /* The base processing unit is 6 source rows.
     * If 12x is active, we pair two 6-row groups (12 rows total). */
    int base6_groups = src_h / 6;

    /* Allocate scratch rows at full source width. */
    /* Vertical intermediates per 6-row group:
     *   v01, v23, v45 (3 rows) -- pairwise averages
     *   1.5x: 4 output rows (v_15x[0..3])
     *   3x:   2 output rows (v_3x[0..1])
     *   6x:   1 output row  (v_6x[0])
     *
     * We also need h_buf for horizontal filtering scratch.
     * For 1.5x horizontal we go from src_w -> dst_widths[0] (2/3 * src_w).
     * For 3x we go from src_w -> dst_widths[1] (1/3 * src_w).
     * For 6x we first do 3x horizontal, then halve.
     * For 12x we do 6x horizontal, then halve.
     */
    size_t row_bytes = (size_t)src_w;

    /* Carve scratch buffers from the pool allocated at init time. */
    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    /* Vertical intermediates */
    uint8_t *v01  = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v23  = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v45  = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);

    /* 3x vertical intermediates (2 rows) */
    uint8_t *v3x_0 = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v3x_1 = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);

    /* 6x vertical intermediate (1 row) */
    uint8_t *v6x = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);

    /* Horizontal scratch buffers */
    /* 3x horiz output width = src_w / 3 */
    int w_3x = src_w / 3;
    uint8_t *h_3x_buf = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)w_3x);

    /* 6x horiz output width = w_3x / 2 */
    int w_6x = w_3x / 2;
    uint8_t *h_6x_buf = (uint8_t *)fused_scratch_alloc(
        &scratch, (size_t)(w_6x > 0 ? w_6x : 1));

    /* For 12x: need to save a 6x row from the first half of the 12-row group */
    uint8_t *v6x_prev = need_12x
        ? (uint8_t *)fused_scratch_alloc(&scratch, row_bytes) : NULL;

    /* For 1.5x blended rows (rows 1 and 2 of the 4-row output) */
    int need_1_5x = (active_outputs & (1u << 0)) != 0;
    uint8_t *blend_tmp = need_1_5x
        ? (uint8_t *)fused_scratch_alloc(&scratch, row_bytes) : NULL;

    if (!v01 || !v23 || !v45 || !v3x_0 || !v3x_1 || !v6x ||
        !h_3x_buf || !h_6x_buf ||
        (need_12x && !v6x_prev) || (need_1_5x && !blend_tmp)) {
        return;  /* scratch pool exhausted - shouldn't happen if init sized correctly */
    }

    /* Output row cursors */
    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint8_t *grp = src + (size_t)g6 * 6 * (size_t)src_stride;

        const uint8_t *restrict row0 = grp;
        const uint8_t *restrict row1 = grp + (size_t)src_stride;
        const uint8_t *restrict row2 = grp + (size_t)2 * (size_t)src_stride;
        const uint8_t *restrict row3 = grp + (size_t)3 * (size_t)src_stride;
        const uint8_t *restrict row4 = grp + (size_t)4 * (size_t)src_stride;
        const uint8_t *restrict row5 = grp + (size_t)5 * (size_t)src_stride;

        /* Step 1: pairwise row averages */
        for (int x = 0; x < src_w; x++) {
            v01[x] = avg_u8(row0[x], row1[x]);
            v23[x] = avg_u8(row2[x], row3[x]);
            v45[x] = avg_u8(row4[x], row5[x]);
        }

        /* Step 3: 3x vertical (always computed if deepest >= 1) */
        if (deepest >= 1) {
            for (int x = 0; x < src_w; x++) {
                v3x_0[x] = avg_u8(v01[x], v23[x]);
                v3x_1[x] = avg_u8(v23[x], v45[x]);
            }
        }

        /* Step 4: 6x vertical (always computed if deepest >= 2) */
        if (deepest >= 2) {
            for (int x = 0; x < src_w; x++) {
                v6x[x] = avg_u8(v3x_0[x], v3x_1[x]);
            }
        }

        /* -- Output 1.5x (bit 0) ----------------------------------- */
        if (active_outputs & (1u << 0)) {
            /* 1.5x vertical: 6 source rows -> 4 output rows (ratio 3:2).
             *   Row 0: v01
             *   Row 1: blend_2_1(v01, v23)  = v01*171/256 + v23*85/256
             *   Row 2: blend_2_1(v23, v45)  = v23*171/256 + v45*85/256
             *   Row 3: v45
             * Each row is then horizontally 1.5x-filtered. */
            int dw = dst_widths[0];
            int ds = dst_strides[0];

            /* Row 0: v01 -> horizontal 1.5x */
            h_filter_1_5x(v01, src_w,
                          dst_planes[0] + (size_t)out_row[0] * (size_t)ds, dw);
            out_row[0]++;

            /* Row 1: blend(v01, v23) -> horizontal 1.5x */
            for (int x = 0; x < src_w; x++)
                blend_tmp[x] = blend_2_1(v01[x], v23[x]);
            h_filter_1_5x(blend_tmp, src_w,
                          dst_planes[0] + (size_t)out_row[0] * (size_t)ds, dw);
            out_row[0]++;

            /* Row 2: blend(v23, v45) -> horizontal 1.5x */
            for (int x = 0; x < src_w; x++)
                blend_tmp[x] = blend_2_1(v23[x], v45[x]);
            h_filter_1_5x(blend_tmp, src_w,
                          dst_planes[0] + (size_t)out_row[0] * (size_t)ds, dw);
            out_row[0]++;

            /* Row 3: v45 -> horizontal 1.5x */
            h_filter_1_5x(v45, src_w,
                          dst_planes[0] + (size_t)out_row[0] * (size_t)ds, dw);
            out_row[0]++;
        }

        /* -- Output 3x (bit 2) ------------------------------------- */
        if (active_outputs & (1u << 2)) {
            int dw = dst_widths[1];
            int ds = dst_strides[1];

            /* 3x vertical produced 2 rows: v3x_0 and v3x_1 (at src_w).
             * Horizontal: 3:1 box average -> dw pixels */
            h_filter_3x(v3x_0, src_w,
                        dst_planes[1] + (size_t)out_row[1] * (size_t)ds, dw);
            out_row[1]++;

            h_filter_3x(v3x_1, src_w,
                        dst_planes[1] + (size_t)out_row[1] * (size_t)ds, dw);
            out_row[1]++;
        }

        /* -- Output 6x (bit 4) ------------------------------------- */
        if (active_outputs & (1u << 4)) {
            int dw = dst_widths[2];
            int ds = dst_strides[2];

            /* 6x vertical produced 1 row: v6x (at src_w).
             * Horizontal: first do 3x horizontal (3:1 box avg -> w_3x),
             * then pairwise halve (-> w_6x = dw). */
            h_filter_3x(v6x, src_w, h_3x_buf, w_3x);
            h_filter_halve(h_3x_buf,
                           dst_planes[2] + (size_t)out_row[2] * (size_t)ds, dw);
            out_row[2]++;
        }

        /* -- 12x handling: save 6x row from this 6-row group ------- */
        if (need_12x) {
            /* For 12x, we need to average two consecutive 6x rows
             * (from two 6-row groups = 12 source rows).
             * On even groups (g6 % 2 == 0), save v6x.
             * On odd groups, average saved v6x with current v6x and output. */
            if ((g6 & 1) == 0) {
                /* Save the 6x vertical row */
                memcpy(v6x_prev, v6x, row_bytes);
            } else {
                /* Average v6x_prev and v6x, then horizontal filter */
                /* 12x vertical: 1 row from 12 source rows */
                uint8_t *v12x_row = v6x_prev;  /* reuse as output */
                for (int x = 0; x < src_w; x++) {
                    v12x_row[x] = avg_u8(v6x_prev[x], v6x[x]);
                }

                if (active_outputs & (1u << 6)) {
                    int dw = dst_widths[3];
                    int ds = dst_strides[3];

                    /* Horizontal: 3x box avg -> halve -> halve
                     * = 12:1 reduction */
                    h_filter_3x(v12x_row, src_w, h_3x_buf, w_3x);
                    h_filter_halve(h_3x_buf, h_6x_buf, w_6x);
                    h_filter_halve(h_6x_buf,
                                   dst_planes[3] + (size_t)out_row[3] * (size_t)ds, dw);
                    out_row[3]++;
                }
            }
        }
    }

    /* Scratch buffers are carved from the persistent pool - nothing to free. */
}


/* -----------------------------------------------------------------------
 * Public entry points
 * ----------------------------------------------------------------------- */

void fused_kernel_pow2_scalar(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v)
{
    /* Map bit positions to cascade level arrays.
     * bit 1 = 2x  -> level 0
     * bit 3 = 4x  -> level 1
     * bit 5 = 8x  -> level 2
     * bit 7 = 16x -> level 3
     */
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    uint8_t *y_planes[4], *u_planes[4], *v_planes[4];
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
    scale_plane_pow2(src_y,
                     p->src_width, p->src_height, p->src_y_stride,
                     p->active_outputs,
                     y_planes, y_widths, y_strides, y_heights,
                     p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_pow2(src_u,
                     p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                     p->active_outputs,
                     u_planes, uv_widths, uv_strides, uv_heights,
                     p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_pow2(src_v,
                     p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                     p->active_outputs,
                     v_planes, uv_widths, uv_strides, uv_heights,
                     p->scratch_pool, p->scratch_pool_size);
}


void fused_kernel_thirds_scalar(const fused_kernel_params_t *p,
                                const uint8_t *src_y,
                                const uint8_t *src_u,
                                const uint8_t *src_v)
{
    /* Map bit positions to cascade level arrays.
     * bit 0 = 1.5x -> level 0
     * bit 2 = 3x   -> level 1
     * bit 4 = 6x   -> level 2
     * bit 6 = 12x  -> level 3
     */
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    uint8_t *y_planes[4], *u_planes[4], *v_planes[4];
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
    scale_plane_thirds(src_y,
                       p->src_width, p->src_height, p->src_y_stride,
                       p->active_outputs,
                       y_planes, y_widths, y_strides, y_heights,
                       p->scratch_pool, p->scratch_pool_size);

    /* U plane (half dimensions) */
    scale_plane_thirds(src_u,
                       p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                       p->active_outputs,
                       u_planes, uv_widths, uv_strides, uv_heights,
                       p->scratch_pool, p->scratch_pool_size);

    /* V plane (half dimensions) */
    scale_plane_thirds(src_v,
                       p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                       p->active_outputs,
                       v_planes, uv_widths, uv_strides, uv_heights,
                       p->scratch_pool, p->scratch_pool_size);
}
