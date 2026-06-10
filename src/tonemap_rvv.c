/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * tonemap_rvv.c - RVV (riscv64) built-in-curve tone mapping.
 *
 * Entry points mirror the scalar reference in tonemap.c:
 *   fused_tonemap_apply_rvv      - planar I010 chroma
 *   fused_tonemap_apply_p010_rvv - interleaved P010 chroma
 *   fused_tonemap_luma_rvv       - custom-LUT luma pass
 *
 * Same pipeline as the x86/NEON kernels, with the strategy adapted to
 * what RVV does well:
 *
 *  - The three pq_to_sdr lookups per pixel use vluxei16, a true indexed
 *    gather from the L1-resident 1KB table.  An in-register vrgather is
 *    not an option: at VLEN=256 a vrgatherei16 over e8 elements tops out
 *    at LMUL=4, which is 128 bytes of table - an eighth of what we need.
 *
 *  - Luma rows load through a segment load (vlseg2e16), which splits the
 *    2x-rate luma into even/odd streams that line up element-for-element
 *    with the chroma samples.  The chroma-rate deltas apply to both
 *    streams directly - no zip/duplicate step - and the even stream of
 *    row 0 is exactly the pixel set the chroma re-encode wants, so no
 *    compaction step either.  Both are real work the fixed-width kernels
 *    have to do that simply vanishes here.
 *
 *  - vsetvl stripmining absorbs ragged row widths natively: every chunk
 *    including the last runs the same vector code, so there is no
 *    fused_tm_block tail loop.
 *
 * All arithmetic uses the scalar reference's widths and shifts: deltas
 * evaluated from the shared Q15 coefficients (state->delta_coef_*), the
 * luma dot in 16-bit (max value 65408), chroma dots in 32-bit.
 * Bit-exactness against fused_tonemap_apply_scalar is enforced by the
 * parity test.
 *
 * Everything runs at LMUL=1 for the 16-bit elements (32-bit intermediates
 * at m2, byte results at mf2).  On the X60 the datapath is half the
 * register width, so larger LMUL buys no throughput - it only adds
 * register pressure.
 *
 * Compiled with -march=rv64gcv and selected at runtime behind
 * caps->has_rvv.
 */

#include "internal.h"
#include "tonemap.h"
#include "kernels_rvv_compat.h"

#if defined(__riscv) && (__riscv_xlen == 64)

/* -----------------------------------------------------------------------
 * Q15 delta evaluation for one chroma-rate vector of 10-bit codes:
 *   delta = ((code - 512) * coef + 16384) >> 15
 * ----------------------------------------------------------------------- */

static inline vint16m1_t tm_delta_rvv(vuint16m1_t code, int32_t coef,
                                      size_t vl)
{
    vint32m2_t x = __riscv_vsub_vx_i32m2(
        __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(code, vl)),
        512, vl);
    vint32m2_t d = __riscv_vsra_vx_i32m2(
        __riscv_vadd_vx_i32m2(__riscv_vmul_vx_i32m2(x, coef, vl), 16384, vl),
        15, vl);
    return __riscv_vncvt_x_x_w_i16m1(d, vl);
}

/* dG sums its two Q15 terms in 32-bit before narrowing. */
static inline vint16m1_t tm_delta_g_rvv(vuint16m1_t cb, vuint16m1_t cr,
                                        int32_t gcb, int32_t gcr, size_t vl)
{
    vint32m2_t xb = __riscv_vsub_vx_i32m2(
        __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(cb, vl)),
        512, vl);
    vint32m2_t xr = __riscv_vsub_vx_i32m2(
        __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vzext_vf2_u32m2(cr, vl)),
        512, vl);
    vint32m2_t tb = __riscv_vsra_vx_i32m2(
        __riscv_vadd_vx_i32m2(__riscv_vmul_vx_i32m2(xb, gcb, vl), 16384, vl),
        15, vl);
    vint32m2_t tr = __riscv_vsra_vx_i32m2(
        __riscv_vadd_vx_i32m2(__riscv_vmul_vx_i32m2(xr, gcr, vl), 16384, vl),
        15, vl);
    return __riscv_vncvt_x_x_w_i16m1(__riscv_vadd_vv_i32m2(tr, tb, vl), vl);
}

/* -----------------------------------------------------------------------
 * clamp(Y' + delta, 0, 1023), then gather the tone-mapped bytes.  The
 * clamped index doubles as the byte offset into the table.
 * ----------------------------------------------------------------------- */

static inline vuint8mf2_t tm_lookup_rvv(const uint8_t *lut, vuint16m1_t y,
                                        vint16m1_t d, size_t vl)
{
    vint16m1_t idx = __riscv_vadd_vv_i16m1(
        __riscv_vreinterpret_v_u16m1_i16m1(y), d, vl);
    idx = __riscv_vmax_vx_i16m1(idx, 0, vl);
    idx = __riscv_vmin_vx_i16m1(idx, 1023, vl);
    return __riscv_vluxei16_v_u8mf2(
        lut, __riscv_vreinterpret_v_i16m1_u16m1(idx), vl);
}

/* (67r + 174g + 15b + 128) >> 8, in 16-bit (max value 65408). */
static inline vuint16m1_t tm_luma_dot_rvv(vuint8mf2_t r, vuint8mf2_t g,
                                          vuint8mf2_t b, size_t vl)
{
    vuint16m1_t acc = __riscv_vwmulu_vx_u16m1(r, 67, vl);
    acc = __riscv_vwmaccu_vx_u16m1(acc, 174, g, vl);
    acc = __riscv_vwmaccu_vx_u16m1(acc, 15, b, vl);
    return __riscv_vsrl_vx_u16m1(__riscv_vadd_vx_u16m1(acc, 128, vl), 8, vl);
}

/* -----------------------------------------------------------------------
 * Chroma-rate gamut + BT.709 encode dots.
 * Mirrors fused_tm_rgb_to_chroma_sdr exactly (32-bit, same rounding).
 * Inputs are row 0's even-position SDR bytes and luma words.
 * ----------------------------------------------------------------------- */

static inline void tm_chroma_dots_rvv(const fused_hdr_internal_t *state,
                                      vuint8mf2_t r, vuint8mf2_t g,
                                      vuint8mf2_t b, vuint16m1_t y,
                                      uint8_t *du, uint8_t *dv, size_t vl)
{
    vint16m1_t rw = __riscv_vreinterpret_v_u16m1_i16m1(
        __riscv_vzext_vf2_u16m1(r, vl));
    vint16m1_t gw = __riscv_vreinterpret_v_u16m1_i16m1(
        __riscv_vzext_vf2_u16m1(g, vl));
    vint16m1_t bw = __riscv_vreinterpret_v_u16m1_i16m1(
        __riscv_vzext_vf2_u16m1(b, vl));
    vint32m2_t yw = __riscv_vreinterpret_v_u32m2_i32m2(
        __riscv_vzext_vf2_u32m2(y, vl));

    /* Gamut rows * 256 (each sums to 256), widening MACs from 16-bit:
     * see tonemap.h */
    vint32m2_t r709 = __riscv_vwmul_vx_i32m2(rw, 425, vl);
    r709 = __riscv_vwmacc_vx_i32m2(r709, -150, gw, vl);
    r709 = __riscv_vwmacc_vx_i32m2(r709, -19, bw, vl);
    r709 = __riscv_vsra_vx_i32m2(__riscv_vadd_vx_i32m2(r709, 128, vl), 8, vl);
    r709 = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(r709, 0, vl), 255, vl);

    vint32m2_t b709 = __riscv_vwmul_vx_i32m2(rw, -5, vl);
    b709 = __riscv_vwmacc_vx_i32m2(b709, -26, gw, vl);
    b709 = __riscv_vwmacc_vx_i32m2(b709, 287, bw, vl);
    b709 = __riscv_vsra_vx_i32m2(__riscv_vadd_vx_i32m2(b709, 128, vl), 8, vl);
    b709 = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(b709, 0, vl), 255, vl);

    /* cb = clamp(128 + ((b709 - y)*cbs + 128 >> 8), cmin, cmax) */
    vint32m2_t cb = __riscv_vadd_vx_i32m2(__riscv_vsra_vx_i32m2(
        __riscv_vadd_vx_i32m2(__riscv_vmul_vx_i32m2(
            __riscv_vsub_vv_i32m2(b709, yw, vl),
            state->cb_out_scale, vl), 128, vl), 8, vl), 128, vl);
    vint32m2_t cr = __riscv_vadd_vx_i32m2(__riscv_vsra_vx_i32m2(
        __riscv_vadd_vx_i32m2(__riscv_vmul_vx_i32m2(
            __riscv_vsub_vv_i32m2(r709, yw, vl),
            state->cr_out_scale, vl), 128, vl), 8, vl), 128, vl);

    cb = __riscv_vmin_vx_i32m2(
        __riscv_vmax_vx_i32m2(cb, state->chroma_out_min, vl),
        state->chroma_out_max, vl);
    cr = __riscv_vmin_vx_i32m2(
        __riscv_vmax_vx_i32m2(cr, state->chroma_out_min, vl),
        state->chroma_out_max, vl);

    __riscv_vse8_v_u8mf2(du, __riscv_vncvt_x_x_w_u8mf2(
        __riscv_vreinterpret_v_i16m1_u16m1(
            __riscv_vncvt_x_x_w_i16m1(cb, vl)), vl), vl);
    __riscv_vse8_v_u8mf2(dv, __riscv_vncvt_x_x_w_u8mf2(
        __riscv_vreinterpret_v_i16m1_u16m1(
            __riscv_vncvt_x_x_w_i16m1(cr, vl)), vl), vl);
}

/* -----------------------------------------------------------------------
 * One chunk: vl chroma samples = 2 rows x 2*vl luma pixels.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_chunk_rvv(const fused_hdr_internal_t *state,
             const uint16_t *su, const uint16_t *sv,
             const uint16_t *suv, int p010,
             const uint16_t *sy0, const uint16_t *sy1,
             uint8_t *dy0, uint8_t *dy1,
             uint8_t *du, uint8_t *dv, size_t vl)
{
    const uint8_t *lut = state->pq_to_sdr;

    /* ---- chroma load ---- */
    vuint16m1_t cb, cr;
    if (p010) {
        fused_load2_u16m1(suv, vl, &cb, &cr);
    } else {
        cb = __riscv_vle16_v_u16m1(su, vl);
        cr = __riscv_vle16_v_u16m1(sv, vl);
    }
    cb = __riscv_vand_vx_u16m1(cb, 0x3FF, vl);
    cr = __riscv_vand_vx_u16m1(cr, 0x3FF, vl);

    /* ---- Q15 deltas at chroma rate ---- */
    vint16m1_t dr = tm_delta_rvv(cr, state->delta_coef_r, vl);
    vint16m1_t db = tm_delta_rvv(cb, state->delta_coef_b, vl);
    vint16m1_t dg = tm_delta_g_rvv(cb, cr, state->delta_coef_g_cb,
                                   state->delta_coef_g_cr, vl);

    /* ---- both rows: even/odd luma streams share the deltas as-is ---- */
    vuint8mf2_t r0e = __riscv_vundefined_u8mf2();
    vuint8mf2_t g0e = r0e, b0e = r0e;
    vuint16m1_t y0e = __riscv_vundefined_u16m1();

    for (int row = 0; row < 2; row++) {
        vuint16m1_t ye, yo;
        fused_load2_u16m1(row ? sy1 : sy0, vl, &ye, &yo);
        ye = __riscv_vand_vx_u16m1(ye, 0x3FF, vl);
        yo = __riscv_vand_vx_u16m1(yo, 0x3FF, vl);

        vuint8mf2_t re = tm_lookup_rvv(lut, ye, dr, vl);
        vuint8mf2_t ge = tm_lookup_rvv(lut, ye, dg, vl);
        vuint8mf2_t be = tm_lookup_rvv(lut, ye, db, vl);
        vuint8mf2_t ro = tm_lookup_rvv(lut, yo, dr, vl);
        vuint8mf2_t go = tm_lookup_rvv(lut, yo, dg, vl);
        vuint8mf2_t bo = tm_lookup_rvv(lut, yo, db, vl);

        vuint16m1_t yse = tm_luma_dot_rvv(re, ge, be, vl);
        vuint16m1_t yso = tm_luma_dot_rvv(ro, go, bo, vl);

        /* Re-interleave even/odd: odd into the high byte of each even
         * word, then store the words as one little-endian byte run. */
        vuint16m1_t packed = __riscv_vor_vv_u16m1(
            yse, __riscv_vsll_vx_u16m1(yso, 8, vl), vl);
        __riscv_vse8_v_u8m1(row ? dy1 : dy0,
                            __riscv_vreinterpret_v_u16m1_u8m1(packed),
                            vl * 2);

        if (row == 0) {
            r0e = re;
            g0e = ge;
            b0e = be;
            y0e = yse;
        }
    }

    /* ---- chroma outputs from row 0's top-left pixels ---- */
    tm_chroma_dots_rvv(state, r0e, g0e, b0e, y0e, du, dv, vl);
}

/* -----------------------------------------------------------------------
 * Frame driver.  p010 is a compile-time literal at both call sites.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
tm_frame_rvv(const fused_hdr_internal_t *state,
             const uint16_t *src_y, int src_y_stride,
             const uint16_t *src_u, const uint16_t *src_v,
             const uint16_t *src_uv, int src_uv_stride, int p010,
             uint8_t *dst_y, int dst_y_stride,
             uint8_t *dst_u, int dst_uv_stride, uint8_t *dst_v,
             int width, int height)
{
    size_t chroma_w = (size_t)(width / 2);
    int chroma_h = height / 2;

    int src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    int src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);

    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *su  = src_u  ? src_u  + cy * src_uv_pitch : 0;
        const uint16_t *sv  = src_v  ? src_v  + cy * src_uv_pitch : 0;
        const uint16_t *suv = src_uv ? src_uv + cy * src_uv_pitch : 0;
        const uint16_t *sy0 = src_y + (cy * 2) * src_y_pitch;
        const uint16_t *sy1 = sy0 + src_y_pitch;
        uint8_t        *dy0 = dst_y + (cy * 2) * dst_y_stride;
        uint8_t        *dy1 = dy0 + dst_y_stride;
        uint8_t        *du  = dst_u + cy * dst_uv_stride;
        uint8_t        *dv  = dst_v + cy * dst_uv_stride;

        size_t cx = 0;
        while (cx < chroma_w) {
            size_t vl = __riscv_vsetvl_e16m1(chroma_w - cx);
            tm_chunk_rvv(state,
                         p010 ? 0 : su + cx, p010 ? 0 : sv + cx,
                         p010 ? suv + cx * 2 : 0, p010,
                         sy0 + cx * 2, sy1 + cx * 2,
                         dy0 + cx * 2, dy1 + cx * 2,
                         du + cx, dv + cx, vl);
            cx += vl;
        }
    }
}

__attribute__((hot))
void fused_tonemap_apply_rvv(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_rvv(state, src_y, src_y_stride, src_u, src_v,
                 0, src_uv_stride, 0,
                 dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                 width, height);
}

__attribute__((hot))
void fused_tonemap_apply_p010_rvv(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_rvv(state, src_y, src_y_stride, 0, 0,
                 src_uv, src_uv_stride, 1,
                 dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                 width, height);
}

/* -----------------------------------------------------------------------
 * Custom-LUT luma pass: dst[x] = lut_y[src[x] & 0x3FF], gathered.
 * Lives here (not tonemap.c) because the dispatcher TU is not compiled
 * with -march=rv64gcv.
 * ----------------------------------------------------------------------- */

void fused_tonemap_luma_rvv(const uint8_t *lut_y,
                            const uint16_t *src_y, int src_y_stride,
                            uint8_t *dst_y, int dst_y_stride,
                            int width, int height)
{
    int src_y_pitch = src_y_stride / (int)sizeof(uint16_t);

    for (int y = 0; y < height; y++) {
        const uint16_t *sy = src_y + y * src_y_pitch;
        uint8_t        *dy = dst_y + y * dst_y_stride;

        size_t x = 0;
        while (x < (size_t)width) {
            size_t vl = __riscv_vsetvl_e16m1((size_t)width - x);
            vuint16m1_t v = __riscv_vand_vx_u16m1(
                __riscv_vle16_v_u16m1(sy + x, vl), 0x3FF, vl);
            __riscv_vse8_v_u8mf2(dy + x,
                                 __riscv_vluxei16_v_u8mf2(lut_y, v, vl), vl);
            x += vl;
        }
    }
}

#endif /* __riscv && __riscv_xlen == 64 */
