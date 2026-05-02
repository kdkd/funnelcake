/* --------------------------------------------------------------------------
 * kernels_hdr_rvv.c - HDR (10-bit) downscale kernels for RISC-V (RVV 1.0).
 *
 * Provides:
 *   fused_kernel_pow2_hdr_rvv   - power-of-two family
 *   fused_kernel_thirds_hdr_rvv - thirds family  [stub for now]
 *
 * Same algorithmic shape as kernels_rvv.c, but operating on uint16_t
 * elements (10-bit values stored in 16 bits).  vsetvl uses e16m1 instead
 * of e8m1 - on the X60 with VLEN=256, that gives 16 u16 elements per
 * vector vs 32 u8 elements.
 *
 * Strides arrive in BYTES (matching the SDR kernel's convention) and are
 * converted to element strides via "/ sizeof(uint16_t)" at the entry to
 * each per-plane worker.
 * -------------------------------------------------------------------------- */

#include "internal.h"
#include <riscv_vector.h>

#if defined(__riscv) && (__riscv_xlen == 64)

/* --------------------------------------------------------------------------
 * Primitive: vertical pair-average for u16.
 *
 *   for x in 0..n: dst[x] = (a[x] + b[x] + 1) >> 1   matching scalar avg_u16.
 *
 * 10-bit values fit fine in u16 throughout: max sum 1023 + 1023 + 1 = 2047
 * is still 11 bits, so vaaddu_vv_u16m1 with vxrm=0 produces the same result
 * as the scalar (u32-cast just-for-safety) path.
 * -------------------------------------------------------------------------- */
static inline void vavg_row_u16(const uint16_t *a, const uint16_t *b,
                                uint16_t *dst, size_t n)
{
    size_t x = 0;
    while (x < n) {
        size_t vl = __riscv_vsetvl_e16m1(n - x);
        vuint16m1_t va = __riscv_vle16_v_u16m1(a + x, vl);
        vuint16m1_t vb = __riscv_vle16_v_u16m1(b + x, vl);
        vuint16m1_t vavg = __riscv_vaaddu_vv_u16m1(va, vb, vl);
        __riscv_vse16_v_u16m1(dst + x, vavg, vl);
        x += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive: 2:1 bilinear blend for u16 (the thirds 1.5x vertical row blend).
 *
 *   for x in 0..n: dst[x] = (a[x] * 171 + b[x] * 85 + 128) >> 8
 *
 * Inputs are 10-bit (max 1023), so a*171 = max 174,933 overflows u16.
 * Widen to u32 for the multiply-accumulate, then narrow back to u16.
 * -------------------------------------------------------------------------- */
static inline void vblend_2_1_row_u16(const uint16_t *a, const uint16_t *b,
                                      uint16_t *dst, size_t n)
{
    size_t x = 0;
    while (x < n) {
        size_t vl = __riscv_vsetvl_e16m1(n - x);
        vuint16m1_t va = __riscv_vle16_v_u16m1(a + x, vl);
        vuint16m1_t vb = __riscv_vle16_v_u16m1(b + x, vl);
        vuint32m2_t s = __riscv_vwmulu_vx_u32m2(va, 171, vl);
        s = __riscv_vwmaccu_vx_u32m2(s, 85, vb, vl);
        s = __riscv_vadd_vx_u32m2(s, 128, vl);
        vuint16m1_t r = __riscv_vnsrl_wx_u16m1(s, 8, vl);
        __riscv_vse16_v_u16m1(dst + x, r, vl);
        x += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive: horizontal 3:1 box average for u16.
 *
 *   for x in 0..dst_n: dst[x] = (src[3x] + src[3x+1] + src[3x+2]) / 3
 *
 * Bit-exact with scalar div3_u32 (sum * 21846) >> 16 - we just stay at
 * u16 since the sum (max 3069) fits, and vmulhu gives the high 16 bits of
 * the u16 * u16 = u32 product.
 * -------------------------------------------------------------------------- */
static inline void vh_filter_3x_row_u16(const uint16_t *src,
                                        uint16_t *dst, size_t dst_n)
{
    size_t i = 0;
    while (i < dst_n) {
        size_t vl = __riscv_vsetvl_e16m1(dst_n - i);
        size_t stride = sizeof(uint16_t) * 3;
        vuint16m1_t a = __riscv_vlse16_v_u16m1(src + 3 * i + 0, stride, vl);
        vuint16m1_t b = __riscv_vlse16_v_u16m1(src + 3 * i + 1, stride, vl);
        vuint16m1_t c = __riscv_vlse16_v_u16m1(src + 3 * i + 2, stride, vl);
        vuint16m1_t s = __riscv_vadd_vv_u16m1(a, b, vl);
        s = __riscv_vadd_vv_u16m1(s, c, vl);
        vuint16m1_t r = __riscv_vmulhu_vx_u16m1(s, 21846, vl);
        __riscv_vse16_v_u16m1(dst + i, r, vl);
        i += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive: horizontal 3:2 bilinear for u16.
 *
 *   for i in 0..pairs:
 *     dst[2i + 0] = blend_2_1(src[3i],   src[3i + 1])
 *     dst[2i + 1] = blend_2_1(src[3i+2], src[3i + 1])
 *
 * Same blend-with-widening as vblend_2_1_row_u16, run twice per chunk
 * with stride-2 stores at offsets 0 and 1.
 * -------------------------------------------------------------------------- */
static inline void vh_filter_1_5x_row_u16(const uint16_t *src,
                                          uint16_t *dst, size_t pairs)
{
    size_t i = 0;
    while (i < pairs) {
        size_t vl = __riscv_vsetvl_e16m1(pairs - i);
        size_t stride3 = sizeof(uint16_t) * 3;
        size_t stride2 = sizeof(uint16_t) * 2;
        vuint16m1_t a = __riscv_vlse16_v_u16m1(src + 3 * i + 0, stride3, vl);
        vuint16m1_t b = __riscv_vlse16_v_u16m1(src + 3 * i + 1, stride3, vl);
        vuint16m1_t c = __riscv_vlse16_v_u16m1(src + 3 * i + 2, stride3, vl);

        /* dst[2i]   = (171*a + 85*b + 128) >> 8 */
        vuint32m2_t s0 = __riscv_vwmulu_vx_u32m2(a, 171, vl);
        s0 = __riscv_vwmaccu_vx_u32m2(s0, 85, b, vl);
        s0 = __riscv_vadd_vx_u32m2(s0, 128, vl);
        vuint16m1_t r0 = __riscv_vnsrl_wx_u16m1(s0, 8, vl);

        /* dst[2i+1] = (171*c + 85*b + 128) >> 8 */
        vuint32m2_t s1 = __riscv_vwmulu_vx_u32m2(c, 171, vl);
        s1 = __riscv_vwmaccu_vx_u32m2(s1, 85, b, vl);
        s1 = __riscv_vadd_vx_u32m2(s1, 128, vl);
        vuint16m1_t r1 = __riscv_vnsrl_wx_u16m1(s1, 8, vl);

        __riscv_vsse16_v_u16m1(dst + 2 * i + 0, stride2, r0, vl);
        __riscv_vsse16_v_u16m1(dst + 2 * i + 1, stride2, r1, vl);
        i += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive: horizontal pair-halve for u16.
 *
 *   for x in 0..dst_n: dst[x] = (src[2x] + src[2x+1] + 1) >> 1
 *
 * Two stride-2 (in elements; the vlse intrinsic stride is in BYTES, so 4
 * for u16) loads + vaaddu.  Safe with src == dst.
 * -------------------------------------------------------------------------- */
static inline void vhalve_row_u16(const uint16_t *src, uint16_t *dst, size_t dst_n)
{
    size_t x = 0;
    while (x < dst_n) {
        size_t vl = __riscv_vsetvl_e16m1(dst_n - x);
        vuint16m1_t even = __riscv_vlse16_v_u16m1(src + 2 * x + 0,
                                                   sizeof(uint16_t) * 2, vl);
        vuint16m1_t odd  = __riscv_vlse16_v_u16m1(src + 2 * x + 1,
                                                   sizeof(uint16_t) * 2, vl);
        vuint16m1_t avg  = __riscv_vaaddu_vv_u16m1(even, odd, vl);
        __riscv_vse16_v_u16m1(dst + x, avg, vl);
        x += vl;
    }
}

/* --------------------------------------------------------------------------
 * Primitive: P010 chroma deinterleave.
 *
 * P010/P210 ship chroma as a single interleaved UVUVUV... plane.  Split it
 * into the two planar buffers the scale kernel expects.  Strides are in
 * bytes; the source stride is the row stride of the interleaved plane and
 * the destination stride is the row stride of the planar U/V buffers.
 * -------------------------------------------------------------------------- */
static void p010_deinterleave_uv_rvv(const uint16_t *src_uv,
                                     int chroma_w, int chroma_h,
                                     int src_uv_el_stride,
                                     uint16_t *dst_u, uint16_t *dst_v,
                                     int dst_planar_el_stride)
{
    for (int y = 0; y < chroma_h; y++) {
        const uint16_t *row = src_uv + (size_t)y * (size_t)src_uv_el_stride;
        uint16_t *u_row = dst_u + (size_t)y * (size_t)dst_planar_el_stride;
        uint16_t *v_row = dst_v + (size_t)y * (size_t)dst_planar_el_stride;

        size_t x = 0;
        size_t n = (size_t)chroma_w;
        size_t stride_b = sizeof(uint16_t) * 2; /* 4 bytes between same-component samples */
        while (x < n) {
            size_t vl = __riscv_vsetvl_e16m1(n - x);
            vuint16m1_t u = __riscv_vlse16_v_u16m1(row + 2 * x + 0, stride_b, vl);
            vuint16m1_t v = __riscv_vlse16_v_u16m1(row + 2 * x + 1, stride_b, vl);
            __riscv_vse16_v_u16m1(u_row + x, u, vl);
            __riscv_vse16_v_u16m1(v_row + x, v, vl);
            x += vl;
        }
    }
}

/* --------------------------------------------------------------------------
 * scale_plane_pow2_hdr_rvv
 *
 * Mirrors scale_plane_pow2_hdr in kernels_hdr_scalar.c.  Process source
 * rows in groups of 2^(deepest+1), pair-average rows in a vertical
 * cascade, then for each active level k do (k+1) horizontal pair-halvings.
 * -------------------------------------------------------------------------- */
static void scale_plane_pow2_hdr_rvv(
    const uint16_t *src,
    int src_w, int src_h, int src_stride_bytes,
    uint32_t active_outputs,
    uint16_t *dst_planes[4],
    int dst_widths[4],
    int dst_strides_bytes[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    int src_el_stride = src_stride_bytes / (int)sizeof(uint16_t);

    int deepest = -1;
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) { deepest = k; break; }
    }
    if (deepest < 0) return;

    int group_rows = (2 << deepest);
    int num_groups = src_h / group_rows;

    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint16_t *vert_buf[4] = { NULL, NULL, NULL, NULL };
    int       vert_rows[4];

    for (int k = 0; k <= deepest; k++) {
        vert_rows[k] = group_rows >> (k + 1);
        vert_buf[k]  = (uint16_t *)fused_scratch_alloc(
            &scratch,
            (size_t)vert_rows[k] * (size_t)src_w * sizeof(uint16_t));
        if (!vert_buf[k]) return;
    }

    uint16_t *h_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)src_w * sizeof(uint16_t));
    if (!h_buf) return;

    int out_row[4] = { 0, 0, 0, 0 };

    for (int g = 0; g < num_groups; g++) {
        const uint16_t *grp_base = src
            + (size_t)g * (size_t)group_rows * (size_t)src_el_stride;

        /* Vertical cascade. */
        for (int r = 0; r < vert_rows[0]; r++) {
            const uint16_t *ra = grp_base + (size_t)(2 * r)     * (size_t)src_el_stride;
            const uint16_t *rb = grp_base + (size_t)(2 * r + 1) * (size_t)src_el_stride;
            uint16_t *dst_row  = vert_buf[0] + (size_t)r * (size_t)src_w;
            vavg_row_u16(ra, rb, dst_row, (size_t)src_w);
        }
        for (int k = 1; k <= deepest; k++) {
            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *ra = vert_buf[k - 1]
                    + (size_t)(2 * r)     * (size_t)src_w;
                const uint16_t *rb = vert_buf[k - 1]
                    + (size_t)(2 * r + 1) * (size_t)src_w;
                uint16_t *dst_row  = vert_buf[k] + (size_t)r * (size_t)src_w;
                vavg_row_u16(ra, rb, dst_row, (size_t)src_w);
            }
        }

        /* Horizontal cascade per active level. */
        for (int k = 0; k <= deepest; k++) {
            if (!(active_outputs & (1u << bit_pos[k]))) continue;
            int dst_el_stride = dst_strides_bytes[k] / (int)sizeof(uint16_t);

            for (int r = 0; r < vert_rows[k]; r++) {
                const uint16_t *vert_row = vert_buf[k]
                    + (size_t)r * (size_t)src_w;

                int cur_w = src_w;
                const uint16_t *cur_src = vert_row;
                uint16_t *cur_dst = h_buf;
                for (int hstep = 0; hstep < (k + 1); hstep++) {
                    int next_w = cur_w >> 1;
                    vhalve_row_u16(cur_src, cur_dst, (size_t)next_w);
                    cur_w = next_w;
                    cur_src = h_buf;
                }

                uint16_t *out = dst_planes[k]
                    + (size_t)out_row[k] * (size_t)dst_el_stride;
                size_t n = (size_t)dst_widths[k];
                size_t i = 0;
                while (i < n) {
                    size_t vl = __riscv_vsetvl_e16m1(n - i);
                    vuint16m1_t v = __riscv_vle16_v_u16m1(h_buf + i, vl);
                    __riscv_vse16_v_u16m1(out + i, v, vl);
                    i += vl;
                }
                out_row[k]++;
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Public entry points
 * -------------------------------------------------------------------------- */

void fused_kernel_pow2_hdr_rvv(const fused_hdr_kernel_params_t *p,
                               const uint16_t *src_y,
                               const uint16_t *src_u,
                               const uint16_t *src_v)
{
    static const int bit_pos[4] = { 1, 3, 5, 7 };

    vwrite_csr(RVV_VXRM, 0);

    uint16_t *y_planes[4], *u_planes[4], *v_planes[4];
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

    scale_plane_pow2_hdr_rvv(src_y,
                             p->src_width, p->src_height, p->src_y_stride,
                             p->active_outputs,
                             y_planes, y_widths, y_strides,
                             p->scratch_pool, p->scratch_pool_size);

    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        /* P010/P210: src_u is the interleaved UVUV plane; src_v is unused
         * (typically NULL).  Deinterleave into the pre-allocated planar
         * tmp buffers, then process U and V as if they were I010. */
        p010_deinterleave_uv_rvv(src_u, chroma_w, chroma_h,
                                 p->src_uv_el_stride,
                                 p->p010_tmp_u, p->p010_tmp_v,
                                 p->p010_tmp_stride / (int)sizeof(uint16_t));

        scale_plane_pow2_hdr_rvv(p->p010_tmp_u,
                                 chroma_w, chroma_h, p->p010_tmp_stride,
                                 p->active_outputs,
                                 u_planes, uv_widths, uv_strides,
                                 p->scratch_pool, p->scratch_pool_size);

        scale_plane_pow2_hdr_rvv(p->p010_tmp_v,
                                 chroma_w, chroma_h, p->p010_tmp_stride,
                                 p->active_outputs,
                                 v_planes, uv_widths, uv_strides,
                                 p->scratch_pool, p->scratch_pool_size);
    } else {
        scale_plane_pow2_hdr_rvv(src_u,
                                 chroma_w, chroma_h, p->src_uv_stride,
                                 p->active_outputs,
                                 u_planes, uv_widths, uv_strides,
                                 p->scratch_pool, p->scratch_pool_size);

        scale_plane_pow2_hdr_rvv(src_v,
                                 chroma_w, chroma_h, p->src_uv_stride,
                                 p->active_outputs,
                                 v_planes, uv_widths, uv_strides,
                                 p->scratch_pool, p->scratch_pool_size);
    }
}

/* --------------------------------------------------------------------------
 * scale_plane_thirds_hdr_rvv
 *
 * Mirrors scale_plane_thirds_hdr in kernels_hdr_scalar.c.  6-row groups
 * (or 12-row pairs for 12x), three pairwise vertical averages
 * (v01/v23/v45), and per-active-level horizontal filter.
 * -------------------------------------------------------------------------- */
static void scale_plane_thirds_hdr_rvv(
    const uint16_t *src,
    int src_w, int src_h, int src_stride_bytes,
    uint32_t active_outputs,
    uint16_t *dst_planes[4],
    int dst_widths[4],
    int dst_strides_bytes[4],
    uint8_t *scratch_pool_base,
    size_t scratch_pool_size)
{
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    int src_el_stride = src_stride_bytes / (int)sizeof(uint16_t);

    int deepest = -1;
    for (int k = 3; k >= 0; k--) {
        if (active_outputs & (1u << bit_pos[k])) { deepest = k; break; }
    }
    if (deepest < 0) return;

    int need_12x = (deepest >= 3);
    int need_1_5x = (active_outputs & (1u << 0)) != 0;
    int base6_groups = src_h / 6;

    size_t row_bytes = (size_t)src_w * sizeof(uint16_t);

    fused_scratch_t scratch;
    fused_scratch_init(&scratch, scratch_pool_base, scratch_pool_size);

    uint16_t *v01   = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v23   = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v45   = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v3x_0 = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v3x_1 = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);
    uint16_t *v6x   = (uint16_t *)fused_scratch_alloc(&scratch, row_bytes);

    int w_3x = src_w / 3;
    int w_6x = w_3x / 2;
    uint16_t *h_3x_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)w_3x * sizeof(uint16_t));
    uint16_t *h_6x_buf = (uint16_t *)fused_scratch_alloc(
        &scratch, (size_t)(w_6x > 0 ? w_6x : 1) * sizeof(uint16_t));

    uint16_t *v6x_prev = need_12x
        ? (uint16_t *)fused_scratch_alloc(&scratch, row_bytes) : NULL;
    uint16_t *blend_tmp = need_1_5x
        ? (uint16_t *)fused_scratch_alloc(&scratch, row_bytes) : NULL;

    if (!v01 || !v23 || !v45 || !v3x_0 || !v3x_1 || !v6x ||
        !h_3x_buf || !h_6x_buf ||
        (need_12x && !v6x_prev) || (need_1_5x && !blend_tmp)) {
        return;
    }

    int out_row[4] = { 0, 0, 0, 0 };

    for (int g6 = 0; g6 < base6_groups; g6++) {
        const uint16_t *grp = src + (size_t)g6 * 6 * (size_t)src_el_stride;
        const uint16_t *row0 = grp;
        const uint16_t *row1 = grp + (size_t)src_el_stride;
        const uint16_t *row2 = grp + (size_t)2 * (size_t)src_el_stride;
        const uint16_t *row3 = grp + (size_t)3 * (size_t)src_el_stride;
        const uint16_t *row4 = grp + (size_t)4 * (size_t)src_el_stride;
        const uint16_t *row5 = grp + (size_t)5 * (size_t)src_el_stride;

        vavg_row_u16(row0, row1, v01, (size_t)src_w);
        vavg_row_u16(row2, row3, v23, (size_t)src_w);
        vavg_row_u16(row4, row5, v45, (size_t)src_w);

        if (deepest >= 1) {
            vavg_row_u16(v01, v23, v3x_0, (size_t)src_w);
            vavg_row_u16(v23, v45, v3x_1, (size_t)src_w);
        }
        if (deepest >= 2) {
            vavg_row_u16(v3x_0, v3x_1, v6x, (size_t)src_w);
        }

        /* 1.5x output */
        if (need_1_5x) {
            int dw = dst_widths[0];
            int ds_el = dst_strides_bytes[0] / (int)sizeof(uint16_t);
            size_t pairs = (size_t)dw / 2;
            uint16_t *out;

            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el;
            vh_filter_1_5x_row_u16(v01, out, pairs);
            out_row[0]++;

            vblend_2_1_row_u16(v01, v23, blend_tmp, (size_t)src_w);
            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el;
            vh_filter_1_5x_row_u16(blend_tmp, out, pairs);
            out_row[0]++;

            vblend_2_1_row_u16(v23, v45, blend_tmp, (size_t)src_w);
            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el;
            vh_filter_1_5x_row_u16(blend_tmp, out, pairs);
            out_row[0]++;

            out = dst_planes[0] + (size_t)out_row[0] * (size_t)ds_el;
            vh_filter_1_5x_row_u16(v45, out, pairs);
            out_row[0]++;
        }

        /* 3x output */
        if (active_outputs & (1u << 2)) {
            int dw = dst_widths[1];
            int ds_el = dst_strides_bytes[1] / (int)sizeof(uint16_t);

            vh_filter_3x_row_u16(v3x_0,
                dst_planes[1] + (size_t)out_row[1] * (size_t)ds_el, (size_t)dw);
            out_row[1]++;
            vh_filter_3x_row_u16(v3x_1,
                dst_planes[1] + (size_t)out_row[1] * (size_t)ds_el, (size_t)dw);
            out_row[1]++;
        }

        /* 6x output */
        if (active_outputs & (1u << 4)) {
            int dw = dst_widths[2];
            int ds_el = dst_strides_bytes[2] / (int)sizeof(uint16_t);

            vh_filter_3x_row_u16(v6x, h_3x_buf, (size_t)w_3x);
            vhalve_row_u16(h_3x_buf,
                dst_planes[2] + (size_t)out_row[2] * (size_t)ds_el, (size_t)dw);
            out_row[2]++;
        }

        /* 12x output */
        if (need_12x) {
            if ((g6 & 1) == 0) {
                /* Save v6x for next group. */
                size_t i = 0;
                while (i < (size_t)src_w) {
                    size_t vl = __riscv_vsetvl_e16m1((size_t)src_w - i);
                    vuint16m1_t v = __riscv_vle16_v_u16m1(v6x + i, vl);
                    __riscv_vse16_v_u16m1(v6x_prev + i, v, vl);
                    i += vl;
                }
            } else {
                vavg_row_u16(v6x_prev, v6x, v6x_prev, (size_t)src_w);

                if (active_outputs & (1u << 6)) {
                    int dw = dst_widths[3];
                    int ds_el = dst_strides_bytes[3] / (int)sizeof(uint16_t);

                    vh_filter_3x_row_u16(v6x_prev, h_3x_buf, (size_t)w_3x);
                    vhalve_row_u16(h_3x_buf, h_6x_buf, (size_t)w_6x);
                    vhalve_row_u16(h_6x_buf,
                        dst_planes[3] + (size_t)out_row[3] * (size_t)ds_el, (size_t)dw);
                    out_row[3]++;
                }
            }
        }
    }
}

void fused_kernel_thirds_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                 const uint16_t *src_y,
                                 const uint16_t *src_u,
                                 const uint16_t *src_v)
{
    static const int bit_pos[4] = { 0, 2, 4, 6 };

    vwrite_csr(RVV_VXRM, 0);

    uint16_t *y_planes[4], *u_planes[4], *v_planes[4];
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

    scale_plane_thirds_hdr_rvv(src_y,
                               p->src_width, p->src_height, p->src_y_stride,
                               p->active_outputs,
                               y_planes, y_widths, y_strides,
                               p->scratch_pool, p->scratch_pool_size);

    int chroma_w = p->src_width / 2;
    int chroma_h = p->src_height / 2;

    if (p->is_p010 && p->p010_tmp_u) {
        p010_deinterleave_uv_rvv(src_u, chroma_w, chroma_h,
                                 p->src_uv_el_stride,
                                 p->p010_tmp_u, p->p010_tmp_v,
                                 p->p010_tmp_stride / (int)sizeof(uint16_t));

        scale_plane_thirds_hdr_rvv(p->p010_tmp_u,
                                   chroma_w, chroma_h, p->p010_tmp_stride,
                                   p->active_outputs,
                                   u_planes, uv_widths, uv_strides,
                                   p->scratch_pool, p->scratch_pool_size);

        scale_plane_thirds_hdr_rvv(p->p010_tmp_v,
                                   chroma_w, chroma_h, p->p010_tmp_stride,
                                   p->active_outputs,
                                   v_planes, uv_widths, uv_strides,
                                   p->scratch_pool, p->scratch_pool_size);
    } else {
        scale_plane_thirds_hdr_rvv(src_u,
                                   chroma_w, chroma_h, p->src_uv_stride,
                                   p->active_outputs,
                                   u_planes, uv_widths, uv_strides,
                                   p->scratch_pool, p->scratch_pool_size);

        scale_plane_thirds_hdr_rvv(src_v,
                                   chroma_w, chroma_h, p->src_uv_stride,
                                   p->active_outputs,
                                   v_planes, uv_widths, uv_strides,
                                   p->scratch_pool, p->scratch_pool_size);
    }
}

#endif /* __riscv */
