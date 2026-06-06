/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/* --------------------------------------------------------------------------
 * kernels_rvv.c - SDR downscale kernels for RISC-V (RVV 1.0)
 *
 * Provides:
 *   fused_kernel_pow2_rvv   - power-of-two family (2x/4x/8x/16x)
 *   fused_kernel_thirds_rvv - thirds family (1.5x/3x/6x/12x)  [stub for now]
 *
 * Targets RVV 1.0 (the V extension).  Kernels are vector-length-agnostic:
 * each loop iteration calls __riscv_vsetvl_e8m1 with the remaining element
 * count and adapts to whatever the hardware reports back.  Tail handling
 * falls out of the loop naturally - the last iteration's vl is just smaller.
 *
 * Rounding-mode contract: every kernel writes vxrm = 0 (round-to-nearest-up)
 * at entry.  This makes the vaaddu averaging instruction compute
 * (a + b + 1) >> 1, matching the scalar avg_u8 helper bit-for-bit.  The
 * parity test in test/test_parity.c verifies this against the scalar path.
 *
 * Targeted at SpacemiT X60 (Ky X1) initially: VLEN=256, DLEN=128, full V
 * including Zfh/Zvfh.  Code is portable to any V-capable chip; X60-specific
 * tuning (LMUL=1 with manual unrolling rather than LMUL>1 grouping) is the
 * default because that part has DLEN=VLEN/2, so LMUL>1 buys no throughput.
 * -------------------------------------------------------------------------- */

#include "internal.h"
#include "kernels_rvv_compat.h"

#if defined(__riscv) && (__riscv_xlen == 64)

/* --------------------------------------------------------------------------
 * Primitive 1: vertical pair-average.
 *
 *   for x in 0..n: dst[x] = (rowA[x] + rowB[x] + 1) >> 1
 *
 * One vsetvl-driven loop; tail handled by the final shrunken vl.
 * -------------------------------------------------------------------------- */
static inline void vavg_row_u8(const uint8_t *rowA,
                               const uint8_t *rowB,
                               uint8_t *dst,
                               size_t n)
{
    size_t x = 0;
    while (x < n) {
        size_t vl = __riscv_vsetvl_e8m1(n - x);
        vuint8m1_t va = __riscv_vle8_v_u8m1(rowA + x, vl);
        vuint8m1_t vb = __riscv_vle8_v_u8m1(rowB + x, vl);
        vuint8m1_t vavg = fused_vaaddu_vv_u8m1(va, vb, vl);
        __riscv_vse8_v_u8m1(dst + x, vavg, vl);
        x += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive 2: horizontal pair-halve.
 *
 *   for x in 0..dst_n: dst[x] = (src[2x] + src[2x+1] + 1) >> 1
 *
 * Implemented via two stride-2 loads (even, odd) + vaaddu.  GCC 13 doesn't
 * ship segment-load intrinsics (the v0.11 -> v1.0 transition), and on the
 * X60 the perf gap between strided-stride-2 and a hypothetical vlseg2e8 is
 * small enough not to matter for this case.
 *
 * Safe to use with src == dst (the in-place horizontal cascade in the scalar
 * kernel relies on read positions always being ahead of write positions).
 * -------------------------------------------------------------------------- */
static inline void vhalve_row_u8(const uint8_t *src,
                                 uint8_t *dst,
                                 size_t dst_n)
{
    size_t x = 0;
    while (x < dst_n) {
        size_t vl = __riscv_vsetvl_e8m1(dst_n - x);
        vuint8m1_t even, odd;
        fused_load2_u8m1(src + 2 * x, vl, &even, &odd);
        vuint8m1_t avg = fused_vaaddu_vv_u8m1(even, odd, vl);
        __riscv_vse8_v_u8m1(dst + x, avg, vl);
        x += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive 2b: fused 2x2-box downscale row (the pure-2x case).
 *
 *   for x in 0..dst_n:
 *     dst[x] = avg( avg(rowA[2x],   rowB[2x]),
 *                   avg(rowA[2x+1], rowB[2x+1]) )
 *
 * A 2x downscale is a vertical pair-average followed by a horizontal halve.
 * Done separately, the vertical average has to write a full-width row out and
 * read it straight back for the halve.  Here both rows arrive as even/odd
 * pairs from a single stride-2 load each, the vertical averages happen in
 * registers, and one more vaaddu folds them into the halved output - so the
 * full-width middle row never exists.  Bit-exact with the two-step path: the
 * operand pairing and the (a + b + 1) >> 1 rounding at each step are identical.
 * -------------------------------------------------------------------------- */
static inline void vdown_2x2_row_u8(const uint8_t *rowA,
                                    const uint8_t *rowB,
                                    uint8_t *dst,
                                    size_t dst_n)
{
    size_t x = 0;
    while (x < dst_n) {
        size_t vl = __riscv_vsetvl_e8m1(dst_n - x);
        vuint8m1_t ae, ao, be, bo;
        fused_load2_u8m1(rowA + 2 * x, vl, &ae, &ao);
        fused_load2_u8m1(rowB + 2 * x, vl, &be, &bo);
        vuint8m1_t ve = fused_vaaddu_vv_u8m1(ae, be, vl);  /* even columns, vertical */
        vuint8m1_t vo = fused_vaaddu_vv_u8m1(ao, bo, vl);  /* odd columns, vertical  */
        vuint8m1_t h  = fused_vaaddu_vv_u8m1(ve, vo, vl);  /* fold pair -> halved out */
        __riscv_vse8_v_u8m1(dst + x, h, vl);
        x += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive 3: 2:1 bilinear blend of two vectors.
 *
 *   blend(a, b) = (171*a + 85*b + 128) >> 8   ~= a * 2/3 + b * 1/3
 *
 * This one operation is the whole vocabulary of the 1.5x thirds path: the
 * vertical row blend and the horizontal 1.5x filter are both just this 2:1
 * blend applied to different neighbours, so the kernels below are written in
 * terms of it.  Operating on vectors (rather than memory rows) lets callers
 * keep intermediates in registers.
 *
 * The widening multiply-accumulate builds 171*a + 85*b (max 65280, comfortably
 * inside u16), then vnclipu does the round-and-shift in a single shot: with
 * vxrm = RNU it adds the 128 bias and shifts out 8 bits for us, so the whole
 * blend stays on the e8 vtype with no detour through a separate +128 add.
 * -------------------------------------------------------------------------- */
static inline vuint8m1_t vblend_2_1_u8m1(vuint8m1_t a, vuint8m1_t b, size_t vl)
{
    vuint16m2_t s = __riscv_vwmulu_vx_u16m2(a, 171, vl);
    s = __riscv_vwmaccu_vx_u16m2(s, 85, b, vl);
    return fused_vnclipu_wx_u8m1(s, 8, vl);
}

/* --------------------------------------------------------------------------
 * Primitive 4: horizontal 3:1 box average.
 *
 *   for x in 0..dst_n: dst[x] = (src[3x] + src[3x+1] + src[3x+2] + 0) / 3
 *
 * Bit-exact with scalar div3_u16 which uses the (sum * 0x5556) >> 16 trick.
 * Here we keep the math at u16: vmulhu(sum, 0x5556) gives the same upper-16
 * bits result as a u32 (sum * 0x5556) >> 16 for sums in [0, 765].
 * -------------------------------------------------------------------------- */
static inline void vh_filter_3x_row_u8(const uint8_t *src,
                                       uint8_t *dst,
                                       size_t dst_n)
{
    size_t i = 0;
    while (i < dst_n) {
        size_t vl = __riscv_vsetvl_e8m1(dst_n - i);
        vuint8m1_t a, b, c;
        fused_load3_u8m1(src + 3 * i, vl, &a, &b, &c);
        vuint16m2_t s = __riscv_vwaddu_vv_u16m2(a, b, vl);
        s = __riscv_vwaddu_wv_u16m2(s, c, vl);
        vuint16m2_t q = __riscv_vmulhu_vx_u16m2(s, 0x5556, vl);
        vuint8m1_t r = __riscv_vnsrl_wx_u8m1(q, 0, vl);
        __riscv_vse8_v_u8m1(dst + i, r, vl);
        i += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive 5: horizontal 3:2 bilinear (the 1.5x horizontal filter).
 *
 *   for i in 0..pairs:
 *     dst[2i + 0] = blend_2_1(src[3i],   src[3i + 1])
 *     dst[2i + 1] = blend_2_1(src[3i+2], src[3i + 1])
 *
 * Loads three vectors (a = src[3i], b = src[3i+1], c = src[3i+2]) via
 * stride-3 loads, computes the two blends, and writes them via two stride-2
 * stores at offsets 0 and 1.  This matches the scalar h_filter_1_5x that
 * processes pairs and leaves any odd-out tail untouched (input width is
 * always a multiple of 3 for valid configurations).
 * -------------------------------------------------------------------------- */
static inline void vh_filter_1_5x_row_u8(const uint8_t *src,
                                         uint8_t *dst,
                                         size_t pairs)
{
    size_t i = 0;
    while (i < pairs) {
        size_t vl = __riscv_vsetvl_e8m1(pairs - i);
        vuint8m1_t a, b, c;
        fused_load3_u8m1(src + 3 * i, vl, &a, &b, &c);
        vuint8m1_t r0 = vblend_2_1_u8m1(a, b, vl);   /* dst[2i]   */
        vuint8m1_t r1 = vblend_2_1_u8m1(c, b, vl);   /* dst[2i+1] */
        fused_store2_u8m1(dst + 2 * i, vl, r0, r1);
        i += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive 6: vertical 2:1 blend folded into the horizontal 1.5x filter.
 *
 * The two interior rows of each 1.5x output group are a vertical 2:1 blend of
 * two reduced rows (ra, rb) that then goes through the 1.5x horizontal filter.
 * Both steps are the same vblend_2_1, so fuse them: blend ra and rb on the fly
 * as each stride-3 chunk arrives, then blend horizontally.  The blended row
 * lives entirely in registers - it never gets written out and read back, which
 * drops a full-row store and reload (and a whole loop) per interior row.
 * -------------------------------------------------------------------------- */
static inline void vh_filter_1_5x_blend_row_u8(const uint8_t *ra,
                                               const uint8_t *rb,
                                               uint8_t *dst,
                                               size_t pairs)
{
    size_t i = 0;
    while (i < pairs) {
        size_t vl = __riscv_vsetvl_e8m1(pairs - i);
        vuint8m1_t a0, a1, a2, b0, b1, b2;
        fused_load3_u8m1(ra + 3 * i, vl, &a0, &a1, &a2);
        fused_load3_u8m1(rb + 3 * i, vl, &b0, &b1, &b2);

        /* Vertical blend of the two rows at each of the three taps... */
        vuint8m1_t t0 = vblend_2_1_u8m1(a0, b0, vl);
        vuint8m1_t t1 = vblend_2_1_u8m1(a1, b1, vl);
        vuint8m1_t t2 = vblend_2_1_u8m1(a2, b2, vl);

        /* ...then the horizontal 1.5x of those freshly blended taps. */
        vuint8m1_t r0 = vblend_2_1_u8m1(t0, t1, vl);   /* dst[2i]   */
        vuint8m1_t r1 = vblend_2_1_u8m1(t2, t1, vl);   /* dst[2i+1] */
        fused_store2_u8m1(dst + 2 * i, vl, r0, r1);
        i += vl;
    }
}

/* --------------------------------------------------------------------------
 * scale_plane_pow2_rvv
 *
 * Single-plane pow2 downscale.  Same algorithm as scale_plane_pow2 in
 * kernels_scalar.c (process source rows in groups of 2^(deepest+1), do
 * vertical pair-averaging in cascade, then for each active level do
 * (k+1) horizontal halvings).  The only difference is that the inner
 * per-byte loops are replaced with vavg_row_u8 / vhalve_row_u8.
 * -------------------------------------------------------------------------- */
static void scale_plane_pow2_rvv(
    const uint8_t *src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint8_t *dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    /* Map cascade level -> bit position in active_outputs. */
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    int deepest = -1;
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) { deepest = k; break; }
    }
    if (deepest < 0) return;

    int group_rows = (2 << deepest);
    int num_groups = src_h / group_rows;

    /* Pure 2x downscale: every group is just two source rows collapsing to one
     * output row, so fuse the vertical average and the horizontal halve in one
     * pass and skip the scratch buffers entirely.  Deeper cascades keep the
     * general path below, where vert_buf[0] is a shared input to the next
     * level and so genuinely has to be materialized. */
    if (deepest == 0) {
        for (int g = 0; g < num_groups; g++) {
            const uint8_t *ra = src + (size_t)(2 * g)     * (size_t)src_stride;
            const uint8_t *rb = src + (size_t)(2 * g + 1) * (size_t)src_stride;
            uint8_t *out = dst_planes[0] + (size_t)g * (size_t)dst_strides[0];
            vdown_2x2_row_u8(ra, rb, out, (size_t)dst_widths[0]);
        }
        return;
    }

    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint8_t *vert_buf[4] = { NULL, NULL, NULL, NULL };
    int      vert_rows[4];

    for (int k = 0; k <= deepest; k++)
        vert_rows[k] = group_rows >> (k + 1);

    /* Materialize a vertical buffer for every level EXCEPT the deepest.  No
     * deeper level reads the deepest level's vertical average, so it is folded
     * straight into that level's first horizontal halve below (vdown_2x2_row)
     * and the full-width row never has to land in memory. */
    for (int k = 0; k < deepest; k++) {
        vert_buf[k] = (uint8_t *)fused_scratch_alloc(
            &scratch, (size_t)vert_rows[k] * (size_t)src_w);
        if (!vert_buf[k]) return;
    }

    uint8_t *h_buf = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)src_w);
    if (!h_buf) return;

    int out_row[4] = { 0, 0, 0, 0 };

    for (int g = 0; g < num_groups; g++) {
        const uint8_t *grp_base = src
            + (size_t)g * (size_t)group_rows * (size_t)src_stride;

        /* Vertical cascade: level 0 averages source row pairs; each subsequent
         * level averages pairs from the level below.  It stops one short of the
         * deepest level - that one is produced on the fly inside the horizontal
         * pass. */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint8_t *ra = grp_base + (size_t)(2 * r)     * (size_t)src_stride;
            const uint8_t *rb = grp_base + (size_t)(2 * r + 1) * (size_t)src_stride;
            uint8_t *dst_row  = vert_buf[0] + (size_t)r * (size_t)src_w;
            vavg_row_u8(ra, rb, dst_row, (size_t)src_w);
        }
        for (int k = 1; k < deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint8_t *ra = vert_buf[k - 1]
                    + (size_t)(2 * r)     * (size_t)src_w;
                const uint8_t *rb = vert_buf[k - 1]
                    + (size_t)(2 * r + 1) * (size_t)src_w;
                uint8_t *dst_row  = vert_buf[k] + (size_t)r * (size_t)src_w;
                vavg_row_u8(ra, rb, dst_row, (size_t)src_w);
            }
        }

        /* Horizontal cascade per active level. */
        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;

            int steps = k + 1;
            for (int r = 0; r < vert_rows[k]; r++) {
                uint8_t *out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_strides[k];

                /* Reduce the row to dst_widths[k] by halving (k+1) times.
                 * Intermediate halvings narrow through h_buf - safe in place,
                 * since output element x consumes source 2x and 2x+1 so the read
                 * cursor stays a step ahead of the write - and the last halving
                 * stores straight into the destination plane (clamped to
                 * dst_widths[k], which can sit just under the halved width when
                 * the source isn't an exact multiple of the output).
                 *
                 * For the deepest level there is no materialized vert_buf: its
                 * vertical average is fused into the first halve.  vdown_2x2_row
                 * reads the two parent rows directly and lands the first halve
                 * in h_buf, then the cascade simply resumes at the second step.
                 * deepest >= 1 here (pure 2x returned early), so there is always
                 * at least one more halve to carry the result to the output. */
                int cur_w;
                const uint8_t *cur_src;
                int hstep;
                if (k == deepest) {
                    const uint8_t *ra = vert_buf[deepest - 1]
                        + (size_t)(2 * r)     * (size_t)src_w;
                    const uint8_t *rb = vert_buf[deepest - 1]
                        + (size_t)(2 * r + 1) * (size_t)src_w;
                    int half_w = src_w >> 1;
                    vdown_2x2_row_u8(ra, rb, h_buf, (size_t)half_w);
                    cur_src = h_buf;
                    cur_w   = half_w;
                    hstep   = 1;
                } else {
                    cur_src = vert_buf[k] + (size_t)r * (size_t)src_w;
                    cur_w   = src_w;
                    hstep   = 0;
                }
                for (; hstep < steps; hstep++) {
                    int next_w = cur_w >> 1;
                    if (hstep == steps - 1) {
                        vhalve_row_u8(cur_src, out, (size_t)dst_widths[k]);
                    } else {
                        vhalve_row_u8(cur_src, h_buf, (size_t)next_w);
                        cur_src = h_buf;
                    }
                    cur_w = next_w;
                }
                out_row[k]++;
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Public entry points
 * -------------------------------------------------------------------------- */

void fused_kernel_pow2_rvv(const fused_kernel_params_t *p,
                           const uint8_t *src_y,
                           const uint8_t *src_u,
                           const uint8_t *src_v)
{
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    /* vxrm = 0 (RNU) makes vaaddu compute (a + b + 1) >> 1, matching the
     * scalar avg_u8.  Set once at the top of the kernel - it persists for
     * the whole call.  GCC 13's RVV intrinsic spec puts vxrm in a global
     * CSR (the per-instruction-form arg list arrived in v1.0 / GCC 14). */
    FUSED_RVV_SET_VXRM_RNU();

    uint8_t *y_planes[4], *u_planes[4], *v_planes[4];
    int y_widths[4], y_strides[4];
    int uv_widths[4], uv_strides[4];

    for (int k = 0; k < 4; k++) {
        int b = bit_pos[k];
        if (p->active_outputs & (1u << b)) {
            y_planes[k]  = p->out[b].plane_y;
            y_widths[k]  = p->out[b].width;
            y_strides[k] = p->out[b].y_stride;

            u_planes[k]  = p->out[b].plane_u;
            v_planes[k]  = p->out[b].plane_v;
            uv_widths[k]  = p->out[b].width / 2;
            uv_strides[k] = p->out[b].uv_stride;
        } else {
            y_planes[k] = u_planes[k] = v_planes[k] = NULL;
            y_widths[k] = y_strides[k] = 0;
            uv_widths[k] = uv_strides[k] = 0;
        }
    }

    scale_plane_pow2_rvv(src_y,
                         p->src_width, p->src_height, p->src_y_stride,
                         p->active_outputs,
                         y_planes, y_widths, y_strides,
                         p->scratch_pool, p->scratch_pool_size);

    scale_plane_pow2_rvv(src_u,
                         p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                         p->active_outputs,
                         u_planes, uv_widths, uv_strides,
                         p->scratch_pool, p->scratch_pool_size);

    scale_plane_pow2_rvv(src_v,
                         p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                         p->active_outputs,
                         v_planes, uv_widths, uv_strides,
                         p->scratch_pool, p->scratch_pool_size);
}

/* --------------------------------------------------------------------------
 * scale_plane_thirds_rvv
 *
 * Single-plane thirds downscale.  Same algorithm as scale_plane_thirds in
 * kernels_scalar.c: process source rows in groups of 6 (or pairs of 6 for
 * 12x), compute three pairwise vertical averages (v01, v23, v45) at full
 * source width, then for each active level produce the corresponding
 * vertically-reduced row(s) and apply the appropriate horizontal filter.
 *
 * Output cascade per 6-row source group:
 *   1.5x: 4 output rows.  Rows 0/3 are v01/v45; rows 1/2 are vertical
 *         blends.  Each goes through h_filter_1_5x.
 *   3x:   2 output rows from v3x_0 = avg(v01, v23) and v3x_1 = avg(v23, v45);
 *         each through h_filter_3x.
 *   6x:   1 output row from v6x = avg(v3x_0, v3x_1); h_filter_3x then halve.
 *   12x:  emitted every other 6-row group.  Vertical: avg(v6x_prev, v6x).
 *         Horizontal: 3x box -> halve -> halve.
 * -------------------------------------------------------------------------- */
static void scale_plane_thirds_rvv(
    const uint8_t *src,
    int src_w, int src_h, int src_stride,
    uint32_t active_outputs,
    uint8_t *dst_planes[4],
    int dst_widths[4],
    int dst_strides[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    int deepest = -1;
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) { deepest = k; break; }
    }
    if (deepest < 0) return;

    int need_12x = (deepest >= 3);
    int need_1_5x = (active_outputs & (1u << 0)) != 0;
    int base6_groups = src_h / 6;

    size_t row_bytes = (size_t)src_w;

    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint8_t *v01   = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v23   = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v45   = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v3x_0 = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v3x_1 = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint8_t *v6x   = (uint8_t *)fused_scratch_alloc(&scratch, row_bytes);

    int w_3x = src_w / 3;
    int w_6x = w_3x / 2;
    uint8_t *h_3x_buf = (uint8_t *)fused_scratch_alloc(&scratch, (size_t)w_3x);
    uint8_t *h_6x_buf = (uint8_t *)fused_scratch_alloc(
        &scratch, (size_t)(w_6x > 0 ? w_6x : 1));

    uint8_t *v6x_prev = need_12x
        ? (uint8_t *)fused_scratch_alloc(&scratch, row_bytes) : NULL;

    if (!v01 || !v23 || !v45 || !v3x_0 || !v3x_1 || !v6x ||
        !h_3x_buf || !h_6x_buf ||
        (need_12x && !v6x_prev)) {
        return;
    }

    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint8_t *grp = src + (size_t)g6 * 6 * (size_t)src_stride;
        const uint8_t *row0 = grp;
        const uint8_t *row1 = grp + (size_t)src_stride;
        const uint8_t *row2 = grp + (size_t)2 * (size_t)src_stride;
        const uint8_t *row3 = grp + (size_t)3 * (size_t)src_stride;
        const uint8_t *row4 = grp + (size_t)4 * (size_t)src_stride;
        const uint8_t *row5 = grp + (size_t)5 * (size_t)src_stride;

        /* Vertical pair-averages of the 6 rows. */
        vavg_row_u8(row0, row1, v01, row_bytes);
        vavg_row_u8(row2, row3, v23, row_bytes);
        vavg_row_u8(row4, row5, v45, row_bytes);

        /* 3x vertical (pair-averages of v01..v45). */
        if (deepest >= 1) {
            vavg_row_u8(v01, v23, v3x_0, row_bytes);
            vavg_row_u8(v23, v45, v3x_1, row_bytes);
        }

        /* 6x vertical. */
        if (deepest >= 2) {
            vavg_row_u8(v3x_0, v3x_1, v6x, row_bytes);
        }

        /* 1.5x output: 4 rows per group, each at dst_widths[0] (~ 2/3 src_w).
         * Rows 0 and 3 filter v01 and v45 directly; rows 1 and 2 fold their
         * vertical 2:1 blend straight into the filter, so no blend scratch. */
        if (need_1_5x) {
            int dw = dst_widths[0];
            int ds = dst_strides[0];
            size_t pairs = (size_t)dw / 2;

            uint8_t *out;

            /* Row 0: v01 */
            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds;
            vh_filter_1_5x_row_u8(v01, out, pairs);
            out_row[0]++;

            /* Row 1: blend(v01, v23) -> filter */
            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds;
            vh_filter_1_5x_blend_row_u8(v01, v23, out, pairs);
            out_row[0]++;

            /* Row 2: blend(v23, v45) -> filter */
            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds;
            vh_filter_1_5x_blend_row_u8(v23, v45, out, pairs);
            out_row[0]++;

            /* Row 3: v45 */
            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds;
            vh_filter_1_5x_row_u8(v45, out, pairs);
            out_row[0]++;
        }

        /* 3x output: 2 rows per group, each at dst_widths[1] (~ 1/3 src_w). */
        if (active_outputs & (1u << 2)) {
            int dw = dst_widths[1];
            int ds = dst_strides[1];

            vh_filter_3x_row_u8(v3x_0,
                dst_planes[1] + (size_t)out_row[1] * (size_t)ds, (size_t)dw);
            out_row[1]++;
            vh_filter_3x_row_u8(v3x_1,
                dst_planes[1] + (size_t)out_row[1] * (size_t)ds, (size_t)dw);
            out_row[1]++;
        }

        /* 6x output: 1 row per group; 3x horizontal then halve. */
        if (active_outputs & (1u << 4)) {
            int dw = dst_widths[2];
            int ds = dst_strides[2];

            vh_filter_3x_row_u8(v6x, h_3x_buf, (size_t)w_3x);
            vhalve_row_u8(h_3x_buf,
                dst_planes[2] + (size_t)out_row[2] * (size_t)ds, (size_t)dw);
            out_row[2]++;
        }

        /* 12x output: 1 row per *pair* of 6-row groups. */
        if (need_12x) {
            if ((g6 & 1) == 0) {
                /* Save v6x for next iteration to pair with. */
                size_t i = 0;
                while (i < row_bytes) {
                    size_t vl = __riscv_vsetvl_e8m1(row_bytes - i);
                    vuint8m1_t v = __riscv_vle8_v_u8m1(v6x + i, vl);
                    __riscv_vse8_v_u8m1(v6x_prev + i, v, vl);
                    i += vl;
                }
            } else {
                /* avg(v6x_prev, v6x) -> in-place into v6x_prev,
                 * then 3x box -> halve -> halve. */
                vavg_row_u8(v6x_prev, v6x, v6x_prev, row_bytes);

                if (active_outputs & (1u << 6)) {
                    int dw = dst_widths[3];
                    int ds = dst_strides[3];

                    vh_filter_3x_row_u8(v6x_prev, h_3x_buf, (size_t)w_3x);
                    vhalve_row_u8(h_3x_buf, h_6x_buf, (size_t)w_6x);
                    vhalve_row_u8(h_6x_buf,
                        dst_planes[3] + (size_t)out_row[3] * (size_t)ds, (size_t)dw);
                    out_row[3]++;
                }
            }
        }
    }
}

void fused_kernel_thirds_rvv(const fused_kernel_params_t *p,
                             const uint8_t *src_y,
                             const uint8_t *src_u,
                             const uint8_t *src_v)
{
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    FUSED_RVV_SET_VXRM_RNU();

    uint8_t *y_planes[4], *u_planes[4], *v_planes[4];
    int y_widths[4], y_strides[4];
    int uv_widths[4], uv_strides[4];

    for (int k = 0; k < 4; k++) {
        int b = bit_pos[k];
        if (p->active_outputs & (1u << b)) {
            y_planes[k]  = p->out[b].plane_y;
            y_widths[k]  = p->out[b].width;
            y_strides[k] = p->out[b].y_stride;

            u_planes[k]  = p->out[b].plane_u;
            v_planes[k]  = p->out[b].plane_v;
            uv_widths[k]  = p->out[b].width / 2;
            uv_strides[k] = p->out[b].uv_stride;
        } else {
            y_planes[k] = u_planes[k] = v_planes[k] = NULL;
            y_widths[k] = y_strides[k] = 0;
            uv_widths[k] = uv_strides[k] = 0;
        }
    }

    scale_plane_thirds_rvv(src_y,
                           p->src_width, p->src_height, p->src_y_stride,
                           p->active_outputs,
                           y_planes, y_widths, y_strides,
                           p->scratch_pool, p->scratch_pool_size);

    scale_plane_thirds_rvv(src_u,
                           p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                           p->active_outputs,
                           u_planes, uv_widths, uv_strides,
                           p->scratch_pool, p->scratch_pool_size);

    scale_plane_thirds_rvv(src_v,
                           p->src_width / 2, p->src_height / 2, p->src_uv_stride,
                           p->active_outputs,
                           v_planes, uv_widths, uv_strides,
                           p->scratch_pool, p->scratch_pool_size);
}

#endif /* __riscv */
