/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/* --------------------------------------------------------------------------
 * kernels_upscale_rvv.c - SDR + HDR upscale kernels for RISC-V (RVV 1.0).
 *
 * Provides:
 *   fused_kernel_upscale_rvv          - SDR upscale only (no downscale)
 *   fused_kernel_thirds_up_rvv        - SDR thirds downscale + upscale
 *   fused_kernel_pow2_up_rvv          - SDR pow2 downscale + upscale
 *   fused_kernel_upscale_hdr_rvv      - HDR upscale only [stub]
 *   fused_kernel_thirds_up_hdr_rvv    - HDR thirds downscale + upscale [stub]
 *   fused_kernel_pow2_up_hdr_rvv      - HDR pow2 downscale + upscale [stub]
 *
 * The SDR combined (down + up) variants use the same sequential strategy as
 * the scalar reference: invoke the SDR down kernel and then the SDR upscale
 * kernel.  This re-reads source once - the genuinely-fused single-pass
 * variant in AVX2/NEON is more code than was worth landing in this first
 * cut on the X60, where compute throughput rather than memory bandwidth is
 * the binding constraint.  May revisit if benches motivate it.
 * -------------------------------------------------------------------------- */

#include "internal.h"
#include "kernels_rvv_compat.h"

#if defined(__riscv) && (__riscv_xlen == 64)

/* --------------------------------------------------------------------------
 * Row-level RVV primitives
 *
 * These small helpers are TU-local static inlines; the compiler folds them
 * straight into the plane loops below.  A few overlap in spirit with
 * primitives in kernels_rvv.c, but static-inline visibility is per-TU, so the
 * handful of lines is repeated here rather than shared through a header.
 * -------------------------------------------------------------------------- */

/* Bilinear vertical blend with 85/171 weights:
 *   dst[x] = (a[x] * 85 + b[x] * 171 + 128) >> 8
 *
 * Used by the upscale 1.5x vertical pattern, where `a` is the further-from-
 * sample row and gets the smaller weight.  Note this is the inverse weight
 * order from the downscale blend_2_1 helper - the upscale traverses the
 * 2->3 sample pattern rather than 3->2.
 */
static inline void vblend_85_171_row_u8(const uint8_t *a, const uint8_t *b,
                                        uint8_t *dst, size_t n)
{
    size_t x = 0;
    while (x < n) {
        size_t vl = __riscv_vsetvl_e8m1(n - x);
        vuint8m1_t va = __riscv_vle8_v_u8m1(a + x, vl);
        vuint8m1_t vb = __riscv_vle8_v_u8m1(b + x, vl);
        vuint16m2_t s = __riscv_vwmulu_vx_u16m2(va, 85, vl);
        s = __riscv_vwmaccu_vx_u16m2(s, 171, vb, vl);
        vuint8m1_t r = fused_vnclipu_wx_u8m1(s, 8, vl);
        __riscv_vse8_v_u8m1(dst + x, r, vl);
        x += vl;
    }
}

/* Horizontal 2x: w source pixels -> 2w output pixels.
 *   dst[2i]   = src[i]
 *   dst[2i+1] = avg(src[i], src[i+1])         (replicate src[w-1] at edge)
 *
 * Main loop processes positions 0..w-2 where src[i+1] is in-bounds.  The
 * final source pixel becomes dst[2(w-1)] = dst[2(w-1)+1] = src[w-1] since
 * the replicated "next" pixel equals it.
 */
static inline void vup_h_2x_row_u8(const uint8_t *src, int w, uint8_t *dst)
{
    int main_n = w - 1;
    int i = 0;
    while (i < main_n) {
        size_t vl = __riscv_vsetvl_e8m1(main_n - i);
        vuint8m1_t a = __riscv_vle8_v_u8m1(src + i, vl);
        vuint8m1_t b = __riscv_vle8_v_u8m1(src + i + 1, vl);
        vuint8m1_t avg = fused_vaaddu_vv_u8m1(a, b, vl);
        fused_store2_u8m1(dst + 2 * i, vl, a, avg);
        i += vl;
    }
    if (w > 0) {
        dst[2 * (w - 1) + 0] = src[w - 1];
        dst[2 * (w - 1) + 1] = src[w - 1];
    }
}

/* Odd output row of a 2x upscale - the row that lands halfway between two
 * source rows.  It is built straight from cur and nxt so the vertical
 * pair-average never has to land in memory:
 *   dst[2i]   = avg(cur[i],   nxt[i])
 *   dst[2i+1] = avg( avg(cur[i],   nxt[i]),
 *                    avg(cur[i+1], nxt[i+1]) )
 * The vertical average flows directly into the horizontal 2x as three cheap
 * vaaddu passes, so the intermediate row is gone before it ever costs a store.
 * The right edge replicates the last averaged sample, matching the rounding of
 * vup_h_2x_row_u8 and the scalar reference.
 */
static inline void vup_2x_oddrow_u8(const uint8_t *cur, const uint8_t *nxt,
                                    int w, uint8_t *dst)
{
    int main_n = w - 1;
    int i = 0;
    while (i < main_n) {
        size_t vl = __riscv_vsetvl_e8m1(main_n - i);
        vuint8m1_t ca = __riscv_vle8_v_u8m1(cur + i,     vl);
        vuint8m1_t na = __riscv_vle8_v_u8m1(nxt + i,     vl);
        vuint8m1_t cb = __riscv_vle8_v_u8m1(cur + i + 1, vl);
        vuint8m1_t nb = __riscv_vle8_v_u8m1(nxt + i + 1, vl);
        vuint8m1_t v0 = fused_vaaddu_vv_u8m1(ca, na, vl);   /* vertical avg at i   */
        vuint8m1_t v1 = fused_vaaddu_vv_u8m1(cb, nb, vl);   /* vertical avg at i+1 */
        vuint8m1_t h  = fused_vaaddu_vv_u8m1(v0, v1, vl);   /* horizontal blend    */
        fused_store2_u8m1(dst + 2 * i, vl, v0, h);
        i += vl;
    }
    if (w > 0) {
        uint8_t s = (uint8_t)(((uint16_t)cur[w - 1] + nxt[w - 1] + 1) >> 1);
        dst[2 * (w - 1) + 0] = s;
        dst[2 * (w - 1) + 1] = s;
    }
}

/* Horizontal 1.5x (2:3): w source pixels -> w*3/2 output pixels.
 *   dst[3i+0] = src[2i]
 *   dst[3i+1] = (src[2i]   * 85  + src[2i+1] * 171 + 128) >> 8
 *   dst[3i+2] = (src[2i+1] * 171 + src[2i+2] * 85  + 128) >> 8
 *
 * Source width must be even.  Main loop processes pairs where src[2i+2]
 * is in-bounds (i.e. i < pairs - 1).  Final pair handled scalar-fashion
 * with c = b (edge replication).
 */
static inline void vup_h_1_5x_row_u8(const uint8_t *src, int w, uint8_t *dst)
{
    int pairs = w / 2;
    if (pairs <= 0) return;

    int main_pairs = pairs - 1;
    int i = 0;
    while (i < main_pairs) {
        size_t vl = __riscv_vsetvl_e8m1(main_pairs - i);
        /* a = src[2i, 2i+2, ...], b = src[2i+1, 2i+3, ...] - perfect vlseg2.
         * c = src[2i+2, 2i+4, ...] is just a shifted by one pair, so it
         * stays as a strided load (no segment-op equivalent for that one). */
        vuint8m1_t a, b;
        fused_load2_u8m1(src + 2 * i, vl, &a, &b);
        vuint8m1_t c = __riscv_vlse8_v_u8m1(src + 2 * i + 2, 2, vl);

        /* dst[3i+1] = (a*85 + b*171 + 128) >> 8 */
        vuint16m2_t s1 = __riscv_vwmulu_vx_u16m2(a, 85, vl);
        s1 = __riscv_vwmaccu_vx_u16m2(s1, 171, b, vl);
        vuint8m1_t r1 = fused_vnclipu_wx_u8m1(s1, 8, vl);

        /* dst[3i+2] = (b*171 + c*85 + 128) >> 8 */
        vuint16m2_t s2 = __riscv_vwmulu_vx_u16m2(b, 171, vl);
        s2 = __riscv_vwmaccu_vx_u16m2(s2, 85, c, vl);
        vuint8m1_t r2 = fused_vnclipu_wx_u8m1(s2, 8, vl);

        fused_store3_u8m1(dst + 3 * i, vl, a, r1, r2);
        i += vl;
    }

    /* Final pair: c is replicated to b. */
    {
        int last = pairs - 1;
        uint8_t a = src[2 * last + 0];
        uint8_t b = src[2 * last + 1];
        uint8_t c = b;
        dst[3 * last + 0] = a;
        dst[3 * last + 1] = (uint8_t)(((uint16_t)a * 85  + (uint16_t)b * 171 + 128) >> 8);
        dst[3 * last + 2] = (uint8_t)(((uint16_t)b * 171 + (uint16_t)c * 85  + 128) >> 8);
    }
}

/* --------------------------------------------------------------------------
 * Plane-level upscales
 * -------------------------------------------------------------------------- */

/* 2x plane upscale: src (sw x sh) -> dst (2sw x 2sh).  Each source row yields
 * two output rows: the even one is a plain horizontal 2x of the row itself, the
 * odd one blends into the next row vertically and horizontally in a single
 * fused pass (vup_2x_oddrow_u8), so no per-row scratch is needed. */
static void vup_2x_plane_u8(const uint8_t *src, int sw, int sh,
                            int sstride, uint8_t *dst, int dstride)
{
    for (int i = 0; i < sh; i++) {
        const uint8_t *cur = src + (size_t)i * sstride;
        const uint8_t *nxt = (i + 1 < sh) ? src + (size_t)(i + 1) * sstride
                                          : cur;
        vup_h_2x_row_u8(cur, sw, dst + (size_t)(2 * i) * dstride);
        vup_2x_oddrow_u8(cur, nxt, sw, dst + (size_t)(2 * i + 1) * dstride);
    }
}

/* 1.5x plane upscale: src (sw x sh) -> dst (sw*3/2 x sh*3/2). */
static void vup_1_5x_plane_u8(const uint8_t *src, int sw, int sh,
                              int sstride, uint8_t *dst, int dstride,
                              uint8_t *scratch)
{
    if (!scratch) return;
    int pairs_v = sh / 2;
    for (int j = 0; j < pairs_v; j++) {
        const uint8_t *r2j  = src + (size_t)(2 * j)     * sstride;
        const uint8_t *r2j1 = src + (size_t)(2 * j + 1) * sstride;
        const uint8_t *r2j2 = (2 * j + 2 < sh) ? src + (size_t)(2 * j + 2) * sstride
                                               : r2j1;

        vup_h_1_5x_row_u8(r2j, sw, dst + (size_t)(3 * j + 0) * dstride);

        /* Vertical blend (1/3, 2/3) of r2j and r2j1 */
        vblend_85_171_row_u8(r2j, r2j1, scratch, (size_t)sw);
        vup_h_1_5x_row_u8(scratch, sw, dst + (size_t)(3 * j + 1) * dstride);

        /* Vertical blend (2/3, 1/3) of r2j1 and r2j2 - equivalent to
         * blend(r2j2, r2j1) with the (85, 171) weights. */
        vblend_85_171_row_u8(r2j2, r2j1, scratch, (size_t)sw);
        vup_h_1_5x_row_u8(scratch, sw, dst + (size_t)(3 * j + 2) * dstride);
    }
}

/* --------------------------------------------------------------------------
 * Per-plane upscale dispatch (mirrors upscale_plane_scalar)
 * -------------------------------------------------------------------------- */
static void upscale_plane_rvv(const fused_kernel_params_t *p,
                              const uint8_t *src,
                              int sw, int sh, int sstride,
                              int is_chroma)
{
    int N    = p->upscale_cascade_depth;
    int tail = p->upscale_tail_1_5x;

    /* Level 0 (2x): source -> up_out[0] */
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
        if (dst) vup_2x_plane_u8(src, sw, sh, sstride, dst, dst_stride);
    }

    /* Levels 1..N-1: up_out[k-1] -> up_out[k] */
    for (int k = 1; k < N; k++) {
        int src_up_w = sw << k;
        int src_up_h = sh << k;
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
            vup_2x_plane_u8(src_up, src_up_w, src_up_h, src_up_stride,
                            dst, dst_stride);
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
            tail_src_w      = sw;
            tail_src_h      = sh;
            tail_src_stride = sstride;
        } else {
            if (!is_chroma) {
                tail_src        = p->up_out[N - 1].plane_y;
                tail_src_stride = p->up_out[N - 1].y_stride;
            } else {
                tail_src        = (is_chroma == 1) ? p->up_out[N - 1].plane_u
                                                   : p->up_out[N - 1].plane_v;
                tail_src_stride = p->up_out[N - 1].uv_stride;
            }
            tail_src_w = sw << N;
            tail_src_h = sh << N;
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
            vup_1_5x_plane_u8(tail_src, tail_src_w, tail_src_h, tail_src_stride,
                              dst, dst_stride, p->upscale_scratch);
        }
    }
}

/* --------------------------------------------------------------------------
 * Public SDR entry points
 * -------------------------------------------------------------------------- */

void fused_kernel_upscale_rvv(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v)
{
    FUSED_RVV_SET_VXRM_RNU();

    upscale_plane_rvv(p, src_y, p->src_width, p->src_height,
                      p->src_y_stride, 0);
    upscale_plane_rvv(p, src_u, p->src_width / 2, p->src_height / 2,
                      p->src_uv_stride, 1);
    upscale_plane_rvv(p, src_v, p->src_width / 2, p->src_height / 2,
                      p->src_uv_stride, 2);
}

void fused_kernel_thirds_up_rvv(const fused_kernel_params_t *p,
                                const uint8_t *src_y,
                                const uint8_t *src_u,
                                const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_rvv(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_rvv(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_rvv(const fused_kernel_params_t *p,
                              const uint8_t *src_y,
                              const uint8_t *src_u,
                              const uint8_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_rvv(p, src_y, src_u, src_v);
    }
    if (p->upscale_active != 0) {
        fused_kernel_upscale_rvv(p, src_y, src_u, src_v);
    }
}

/* --------------------------------------------------------------------------
 * HDR upscale - RVV implementation
 *
 * Same shape as the SDR upscale path above but operating on uint16_t
 * elements with vsetvl_e16m1.  The blend helper uses u32-widened
 * multiplies because 10-bit values multiplied by 171 overflow u16.
 *
 * P010/P210 chroma is deinterleaved into the persistent planar temp buffers
 * before the U/V upscale passes, matching the HDR downscale entry points.
 * -------------------------------------------------------------------------- */

/* Bilinear vertical blend with 85/171 weights for u16:
 *   dst[x] = (a[x] * 85 + b[x] * 171 + 128) >> 8
 * Inputs are 10-bit; 1023*171 = 174933 overflows u16, so widen to u32. */
static inline void vblend_85_171_row_u16(const uint16_t *a, const uint16_t *b,
                                         uint16_t *dst, size_t n)
{
    size_t x = 0;
    while (x < n) {
        size_t vl = __riscv_vsetvl_e16m1(n - x);
        vuint16m1_t va = __riscv_vle16_v_u16m1(a + x, vl);
        vuint16m1_t vb = __riscv_vle16_v_u16m1(b + x, vl);
        vuint32m2_t s = __riscv_vwmulu_vx_u32m2(va, 85, vl);
        s = __riscv_vwmaccu_vx_u32m2(s, 171, vb, vl);
        vuint16m1_t r = fused_vnclipu_wx_u16m1(s, 8, vl);
        __riscv_vse16_v_u16m1(dst + x, r, vl);
        x += vl;
    }
}

/* Horizontal 2x for u16:
 *   dst[2i]   = src[i]
 *   dst[2i+1] = avg(src[i], src[i+1])    (replicate src[w-1] at right edge)
 */
static inline void vup_h_2x_row_u16(const uint16_t *src, int w, uint16_t *dst)
{
    int main_n = w - 1;
    int i = 0;
    while (i < main_n) {
        size_t vl = __riscv_vsetvl_e16m1(main_n - i);
        vuint16m1_t a = __riscv_vle16_v_u16m1(src + i, vl);
        vuint16m1_t b = __riscv_vle16_v_u16m1(src + i + 1, vl);
        vuint16m1_t avg = fused_vaaddu_vv_u16m1(a, b, vl);
        fused_store2_u16m1(dst + 2 * i, vl, a, avg);
        i += vl;
    }
    if (w > 0) {
        dst[2 * (w - 1) + 0] = src[w - 1];
        dst[2 * (w - 1) + 1] = src[w - 1];
    }
}

/* Odd output row of a 2x upscale, u16 - twin of vup_2x_oddrow_u8.  The vertical
 * pair-average of cur and nxt feeds straight into the horizontal 2x, so the
 * halfway row is produced without staging it in memory:
 *   dst[2i]   = avg(cur[i],   nxt[i])
 *   dst[2i+1] = avg( avg(cur[i],   nxt[i]),
 *                    avg(cur[i+1], nxt[i+1]) )
 * 10-bit samples leave plenty of headroom in u16, so the averages stay in
 * u16m1 the whole way. */
static inline void vup_2x_oddrow_u16(const uint16_t *cur, const uint16_t *nxt,
                                     int w, uint16_t *dst)
{
    int main_n = w - 1;
    int i = 0;
    while (i < main_n) {
        size_t vl = __riscv_vsetvl_e16m1(main_n - i);
        vuint16m1_t ca = __riscv_vle16_v_u16m1(cur + i,     vl);
        vuint16m1_t na = __riscv_vle16_v_u16m1(nxt + i,     vl);
        vuint16m1_t cb = __riscv_vle16_v_u16m1(cur + i + 1, vl);
        vuint16m1_t nb = __riscv_vle16_v_u16m1(nxt + i + 1, vl);
        vuint16m1_t v0 = fused_vaaddu_vv_u16m1(ca, na, vl);   /* vertical avg at i   */
        vuint16m1_t v1 = fused_vaaddu_vv_u16m1(cb, nb, vl);   /* vertical avg at i+1 */
        vuint16m1_t h  = fused_vaaddu_vv_u16m1(v0, v1, vl);   /* horizontal blend    */
        fused_store2_u16m1(dst + 2 * i, vl, v0, h);
        i += vl;
    }
    if (w > 0) {
        uint16_t s = (uint16_t)(((uint32_t)cur[w - 1] + nxt[w - 1] + 1) >> 1);
        dst[2 * (w - 1) + 0] = s;
        dst[2 * (w - 1) + 1] = s;
    }
}

/* Horizontal 1.5x (2:3) for u16:
 *   dst[3i+0] = src[2i]
 *   dst[3i+1] = (src[2i]   * 85  + src[2i+1] * 171 + 128) >> 8
 *   dst[3i+2] = (src[2i+1] * 171 + src[2i+2] * 85  + 128) >> 8
 */
static inline void vup_h_1_5x_row_u16(const uint16_t *src, int w, uint16_t *dst)
{
    int pairs = w / 2;
    if (pairs <= 0) return;

    int main_pairs = pairs - 1;
    int i = 0;
    while (i < main_pairs) {
        size_t vl = __riscv_vsetvl_e16m1(main_pairs - i);
        vuint16m1_t a, b;
        fused_load2_u16m1(src + 2 * i, vl, &a, &b);
        vuint16m1_t c = __riscv_vlse16_v_u16m1(src + 2 * i + 2,
                                               sizeof(uint16_t) * 2, vl);

        vuint32m2_t s1 = __riscv_vwmulu_vx_u32m2(a, 85, vl);
        s1 = __riscv_vwmaccu_vx_u32m2(s1, 171, b, vl);
        vuint16m1_t r1 = fused_vnclipu_wx_u16m1(s1, 8, vl);

        vuint32m2_t s2 = __riscv_vwmulu_vx_u32m2(b, 171, vl);
        s2 = __riscv_vwmaccu_vx_u32m2(s2, 85, c, vl);
        vuint16m1_t r2 = fused_vnclipu_wx_u16m1(s2, 8, vl);

        fused_store3_u16m1(dst + 3 * i, vl, a, r1, r2);
        i += vl;
    }

    /* Final pair: c is replicated to b. */
    {
        int last = pairs - 1;
        uint16_t a = src[2 * last + 0];
        uint16_t b = src[2 * last + 1];
        uint16_t c = b;
        dst[3 * last + 0] = a;
        dst[3 * last + 1] = (uint16_t)(((uint32_t)a * 85  + (uint32_t)b * 171 + 128) >> 8);
        dst[3 * last + 2] = (uint16_t)(((uint32_t)b * 171 + (uint32_t)c * 85  + 128) >> 8);
    }
}

/* 2x plane upscale for u16.  Even output row is a horizontal 2x of the source
 * row; the odd row is the fused vertical+horizontal blend into the next row,
 * so no per-row scratch is needed. */
static void vup_2x_plane_u16(const uint16_t *src, int sw, int sh,
                             int sstride_el, uint16_t *dst, int dstride_el)
{
    for (int i = 0; i < sh; i++) {
        const uint16_t *cur = src + (size_t)i * sstride_el;
        const uint16_t *nxt = (i + 1 < sh) ? src + (size_t)(i + 1) * sstride_el
                                           : cur;
        vup_h_2x_row_u16(cur, sw, dst + (size_t)(2 * i) * dstride_el);
        vup_2x_oddrow_u16(cur, nxt, sw, dst + (size_t)(2 * i + 1) * dstride_el);
    }
}

/* 1.5x plane upscale for u16. */
static void vup_1_5x_plane_u16(const uint16_t *src, int sw, int sh,
                               int sstride_el, uint16_t *dst, int dstride_el,
                               uint16_t *scratch)
{
    if (!scratch) return;
    int pairs_v = sh / 2;
    for (int j = 0; j < pairs_v; j++) {
        const uint16_t *r2j  = src + (size_t)(2 * j)     * sstride_el;
        const uint16_t *r2j1 = src + (size_t)(2 * j + 1) * sstride_el;
        const uint16_t *r2j2 = (2 * j + 2 < sh) ? src + (size_t)(2 * j + 2) * sstride_el
                                                : r2j1;

        vup_h_1_5x_row_u16(r2j, sw, dst + (size_t)(3 * j + 0) * dstride_el);

        vblend_85_171_row_u16(r2j, r2j1, scratch, (size_t)sw);
        vup_h_1_5x_row_u16(scratch, sw, dst + (size_t)(3 * j + 1) * dstride_el);

        vblend_85_171_row_u16(r2j2, r2j1, scratch, (size_t)sw);
        vup_h_1_5x_row_u16(scratch, sw, dst + (size_t)(3 * j + 2) * dstride_el);
    }
}

/* Per-plane HDR upscale dispatch (mirrors upscale_plane_hdr_scalar). */
static void upscale_plane_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src,
                                  int sw, int sh, int sstride_el,
                                  int is_chroma)
{
    int N    = p->upscale_cascade_depth;
    int tail = p->upscale_tail_1_5x;

    if (N >= 1 && (p->upscale_hdr_active & (1u << FUSED_UP_IDX_2X))) {
        uint16_t *dst;
        int dst_el_stride;
        if (!is_chroma) {
            dst = p->hdr_up_out[FUSED_UP_IDX_2X].plane_y;
            dst_el_stride = p->hdr_up_out[FUSED_UP_IDX_2X].y_stride / (int)sizeof(uint16_t);
        } else {
            dst = (is_chroma == 1) ? p->hdr_up_out[FUSED_UP_IDX_2X].plane_u
                                   : p->hdr_up_out[FUSED_UP_IDX_2X].plane_v;
            dst_el_stride = p->hdr_up_out[FUSED_UP_IDX_2X].uv_stride / (int)sizeof(uint16_t);
        }
        if (dst) vup_2x_plane_u16(src, sw, sh, sstride_el, dst, dst_el_stride);
    }

    for (int k = 1; k < N; k++) {
        int src_up_w = sw << k;
        int src_up_h = sh << k;
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
            vup_2x_plane_u16(src_up, src_up_w, src_up_h, src_up_el_stride,
                             dst, dst_el_stride);
        }
    }

    if (tail && (p->upscale_hdr_active & (1u << FUSED_UP_IDX_TAIL))) {
        const uint16_t *tail_src;
        int tail_src_w, tail_src_h, tail_src_el_stride;
        uint16_t *dst;
        int dst_el_stride;

        if (N == 0) {
            tail_src           = src;
            tail_src_w         = sw;
            tail_src_h         = sh;
            tail_src_el_stride = sstride_el;
        } else {
            if (!is_chroma) {
                tail_src           = p->hdr_up_out[N - 1].plane_y;
                tail_src_el_stride = p->hdr_up_out[N - 1].y_stride / (int)sizeof(uint16_t);
            } else {
                tail_src           = (is_chroma == 1) ? p->hdr_up_out[N - 1].plane_u
                                                      : p->hdr_up_out[N - 1].plane_v;
                tail_src_el_stride = p->hdr_up_out[N - 1].uv_stride / (int)sizeof(uint16_t);
            }
            tail_src_w = sw << N;
            tail_src_h = sh << N;
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
            vup_1_5x_plane_u16(tail_src, tail_src_w, tail_src_h, tail_src_el_stride,
                               dst, dst_el_stride, p->upscale_scratch_hdr);
        }
    }
}

void fused_kernel_upscale_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src_y,
                                  const uint16_t *src_u,
                                  const uint16_t *src_v)
{
    FUSED_RVV_SET_VXRM_RNU();

    const uint16_t *up_src_u = src_u;
    const uint16_t *up_src_v = src_v;
    int up_src_uv_el_stride = p->src_uv_el_stride;

    if (p->is_p010) {
        if (fused_hdr_deinterleave_p010(p, src_u) != 0) return;
        up_src_u = p->p010_tmp_u;
        up_src_v = p->p010_tmp_v;
        up_src_uv_el_stride = p->p010_tmp_stride / (int)sizeof(uint16_t);
    }

    upscale_plane_hdr_rvv(p, src_y, p->src_width, p->src_height,
                          p->src_y_el_stride, 0);
    upscale_plane_hdr_rvv(p, up_src_u, p->src_width / 2, p->src_height / 2,
                          up_src_uv_el_stride, 1);
    upscale_plane_hdr_rvv(p, up_src_v, p->src_width / 2, p->src_height / 2,
                          up_src_uv_el_stride, 2);
}

void fused_kernel_thirds_up_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                    const uint16_t *src_y,
                                    const uint16_t *src_u,
                                    const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_thirds_hdr_rvv(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_rvv(p, src_y, src_u, src_v);
    }
}

void fused_kernel_pow2_up_hdr_rvv(const fused_hdr_kernel_params_t *p,
                                  const uint16_t *src_y,
                                  const uint16_t *src_u,
                                  const uint16_t *src_v)
{
    if (p->active_outputs != 0) {
        fused_kernel_pow2_hdr_rvv(p, src_y, src_u, src_v);
    }
    if (p->upscale_hdr_active != 0) {
        fused_kernel_upscale_hdr_rvv(p, src_y, src_u, src_v);
    }
}

#endif /* __riscv */
