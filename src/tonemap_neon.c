/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * tonemap_neon.c - NEON (aarch64) built-in-curve tone mapping.
 *
 * Entry points mirror the scalar reference in tonemap.c:
 *   fused_tonemap_apply_neon      - planar I010 chroma
 *   fused_tonemap_apply_p010_neon - interleaved P010 chroma
 *
 * Same pipeline as the x86 kernels, with the lookup strategy adapted to
 * NEON's constraints.  A tbl-based in-register lookup does NOT work
 * here: the 1024-byte pq_to_sdr table would need 64 q registers to be
 * resident (NEON has 32), so a vqtbl4q chain would reload four table
 * registers per 64-byte group per channel - far more memory traffic
 * than the lookups themselves.  Instead this kernel vectorizes all the
 * arithmetic (Q10 chroma deltas, index clamps, luma dot, chroma-rate
 * gamut/encode dots) and performs the three byte lookups per pixel as
 * scalar L1 loads through small stack buffers.  Apple and Neoverse
 * cores sustain 3+ loads per cycle with deep out-of-order windows, so
 * the 48 independent lookups per 16-pixel group pipeline well - the
 * same idiom the custom-LUT luma pass (tonemap_luma_neon) already uses.
 *
 * All arithmetic uses the scalar reference's widths and shifts: deltas
 * evaluated from the shared Q10 coefficients (state->delta_coef_*), the
 * luma dot in modular 16-bit (max value 65408), chroma dots in 32-bit.
 * Bit-exactness against fused_tonemap_apply_scalar is enforced by the
 * parity test.
 *
 * NEON has no element masks, so ragged edges (chroma_w % 8) fall back to
 * fused_tm_block - the same shared inline the scalar loops use.
 *
 * Compiled unconditionally on aarch64 (NEON is baseline) and selected at
 * runtime behind caps->has_neon.
 */

#if defined(__aarch64__) || defined(_M_ARM64)

#include "internal.h"
#include "tonemap.h"

#include <arm_neon.h>

/* -----------------------------------------------------------------------
 * Q10 delta evaluation for 8 chroma codes (one uint16x8) -> 8 deltas.
 *
 * vqrdmulh of (code - 512) << 5 by the Q10 coefficient computes
 *   ((x*32) * coef + 16384) >> 15  ==  (x*coef + 512) >> 10
 * which is exactly the scalar table definition, entirely in 16-bit.
 * (The saturating case of vqrdmulh needs both operands at -32768;
 * x << 5 never goes below -16384, so it cannot fire.)
 * ----------------------------------------------------------------------- */

static inline FUSED_ALWAYS_INLINE int16x8_t
tm_delta8_neon(uint16x8_t codes, int16x8_t coef)
{
    int16x8_t x = vshlq_n_s16(
        vsubq_s16(vreinterpretq_s16_u16(codes), vdupq_n_s16(512)), 5);
    return vqrdmulhq_s16(x, coef);
}

/* dG = g_cr(Cr) + g_cb(Cb): each term rounds exactly like its scalar
 * table entry, and the sum (magnitude < 1024) stays in 16-bit. */
static inline FUSED_ALWAYS_INLINE int16x8_t
tm_delta8_g_neon(uint16x8_t cb, uint16x8_t cr, int16x8_t gcb, int16x8_t gcr)
{
    int16x8_t c512 = vdupq_n_s16(512);
    int16x8_t xb = vshlq_n_s16(
        vsubq_s16(vreinterpretq_s16_u16(cb), c512), 5);
    int16x8_t xr = vshlq_n_s16(
        vsubq_s16(vreinterpretq_s16_u16(cr), c512), 5);
    return vaddq_s16(vqrdmulhq_s16(xr, gcr), vqrdmulhq_s16(xb, gcb));
}

/* -----------------------------------------------------------------------
 * Index computation for one 16-pixel row segment: clamp(Y' + delta).
 * Deltas arrive already expanded to luma rate (d0 = px 0..7, d1 = 8..15).
 * ----------------------------------------------------------------------- */

static inline FUSED_ALWAYS_INLINE void
tm_indices_neon(uint16x8_t y0, uint16x8_t y1, int16x8_t d0, int16x8_t d1,
                uint16x8_t *i0_out, uint16x8_t *i1_out)
{
    int16x8_t zero = vdupq_n_s16(0);
    int16x8_t top  = vdupq_n_s16(1023);
    *i0_out = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(
        vaddq_s16(vreinterpretq_s16_u16(y0), d0), zero), top));
    *i1_out = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(
        vaddq_s16(vreinterpretq_s16_u16(y1), d1), zero), top));
}

/* 16 lookups from a pair of index vectors into a byte buffer.  Lane
 * extraction goes through GPRs (umov) rather than a stack round-trip;
 * the 16 table loads are independent and pipeline behind each other. */
static inline FUSED_ALWAYS_INLINE void
tm_lookup16_neon(const uint8_t *lut, uint16x8_t i0, uint16x8_t i1,
                 uint8_t *out)
{
    out[ 0] = lut[vgetq_lane_u16(i0, 0)];
    out[ 1] = lut[vgetq_lane_u16(i0, 1)];
    out[ 2] = lut[vgetq_lane_u16(i0, 2)];
    out[ 3] = lut[vgetq_lane_u16(i0, 3)];
    out[ 4] = lut[vgetq_lane_u16(i0, 4)];
    out[ 5] = lut[vgetq_lane_u16(i0, 5)];
    out[ 6] = lut[vgetq_lane_u16(i0, 6)];
    out[ 7] = lut[vgetq_lane_u16(i0, 7)];
    out[ 8] = lut[vgetq_lane_u16(i1, 0)];
    out[ 9] = lut[vgetq_lane_u16(i1, 1)];
    out[10] = lut[vgetq_lane_u16(i1, 2)];
    out[11] = lut[vgetq_lane_u16(i1, 3)];
    out[12] = lut[vgetq_lane_u16(i1, 4)];
    out[13] = lut[vgetq_lane_u16(i1, 5)];
    out[14] = lut[vgetq_lane_u16(i1, 6)];
    out[15] = lut[vgetq_lane_u16(i1, 7)];
}

/* -----------------------------------------------------------------------
 * Chroma-rate gamut + BT.709 encode dots for 8 samples.
 * Mirrors fused_tm_rgb_to_chroma_sdr exactly (32-bit, same rounding).
 * Inputs are the even-position r/g/b/y values widened to 16-bit.
 * ----------------------------------------------------------------------- */

static inline FUSED_ALWAYS_INLINE void
tm_chroma_dots_neon(uint16x8_t r, uint16x8_t g, uint16x8_t b, uint16x8_t y,
                    int16x8_t cbs, int16x8_t crs,
                    int16x8_t cmin, int16x8_t cmax,
                    uint8_t *du, uint8_t *dv)
{
    int32x4_t c128 = vdupq_n_s32(128);
    uint16x8_t c255 = vdupq_n_u16(255);
    int16x4_t g425 = vdup_n_s16(425),  gm150 = vdup_n_s16(-150);
    int16x4_t gm19 = vdup_n_s16(-19),  gm5   = vdup_n_s16(-5);
    int16x4_t gm26 = vdup_n_s16(-26),  g287  = vdup_n_s16(287);

    /* Values are 0..255, so the u16 halves reinterpret as s16.  The
     * saturating rounded narrows perform the gamut rounding, lower clamp,
     * and 32-to-16 conversion together. */
    int16x4_t rl = vreinterpret_s16_u16(vget_low_u16(r));
    int16x4_t gl = vreinterpret_s16_u16(vget_low_u16(g));
    int16x4_t bl = vreinterpret_s16_u16(vget_low_u16(b));
    int16x4_t rh = vreinterpret_s16_u16(vget_high_u16(r));
    int16x4_t gh = vreinterpret_s16_u16(vget_high_u16(g));
    int16x4_t bh = vreinterpret_s16_u16(vget_high_u16(b));

    int32x4_t r709l = vmlal_s16(
        vmlal_s16(vmull_s16(rl, g425), gl, gm150), bl, gm19);
    int32x4_t r709h = vmlal_s16(
        vmlal_s16(vmull_s16(rh, g425), gh, gm150), bh, gm19);
    int32x4_t b709l = vmlal_s16(
        vmlal_s16(vmull_s16(rl, gm5), gl, gm26), bl, g287);
    int32x4_t b709h = vmlal_s16(
        vmlal_s16(vmull_s16(rh, gm5), gh, gm26), bh, g287);

    uint16x8_t r709 = vqrshrun_high_n_s32(
        vqrshrun_n_s32(r709l, 8), r709h, 8);
    uint16x8_t b709 = vqrshrun_high_n_s32(
        vqrshrun_n_s32(b709l, 8), b709h, 8);
    r709 = vminq_u16(r709, c255);
    b709 = vminq_u16(b709, c255);

    /* cb = clamp(128 + ((b709 - y)*cbs + 128 >> 8), cmin, cmax).
     * Differences and scales fit signed 16-bit; widening multiplies retain
     * the exact 32-bit products. */
    int16x8_t bd = vsubq_s16(vreinterpretq_s16_u16(b709),
                             vreinterpretq_s16_u16(y));
    int16x8_t rd = vsubq_s16(vreinterpretq_s16_u16(r709),
                             vreinterpretq_s16_u16(y));
    int32x4_t cbl = vmull_s16(vget_low_s16(bd), vget_low_s16(cbs));
    int32x4_t cbh = vmull_high_s16(bd, cbs);
    int32x4_t crl = vmull_s16(vget_low_s16(rd), vget_low_s16(crs));
    int32x4_t crh = vmull_high_s16(rd, crs);

    cbl = vrsraq_n_s32(c128, cbl, 8);
    cbh = vrsraq_n_s32(c128, cbh, 8);
    crl = vrsraq_n_s32(c128, crl, 8);
    crh = vrsraq_n_s32(c128, crh, 8);

    int16x8_t cbw = vcombine_s16(vmovn_s32(cbl), vmovn_s32(cbh));
    int16x8_t crw = vcombine_s16(vmovn_s32(crl), vmovn_s32(crh));
    cbw = vminq_s16(vmaxq_s16(cbw, cmin), cmax);
    crw = vminq_s16(vmaxq_s16(crw, cmin), cmax);
    vst1_u8(du, vmovn_u16(vreinterpretq_u16_s16(cbw)));
    vst1_u8(dv, vmovn_u16(vreinterpretq_u16_s16(crw)));
}

/* -----------------------------------------------------------------------
 * One chunk: 8 chroma samples = 2 rows x 16 luma pixels (full only;
 * ragged edges are handled by the caller with fused_tm_block).
 * ----------------------------------------------------------------------- */

static inline FUSED_ALWAYS_INLINE void
tm_chunk_neon(const fused_hdr_internal_t *state,
              const uint16_t *su, const uint16_t *sv,
              const uint16_t *suv, int p010,
              const uint16_t *sy0, const uint16_t *sy1,
              uint8_t *dy0, uint8_t *dy1,
              uint8_t *du, uint8_t *dv,
              int16x8_t cbs, int16x8_t crs, int16x8_t cmin, int16x8_t cmax)
{
    const uint8_t *lut = state->pq_to_sdr;
    uint16x8_t mask10 = vdupq_n_u16(0x3FF);

    /* ---- chroma load ---- */
    uint16x8_t cb, cr;
    if (p010) {
        uint16x8x2_t uv = vld2q_u16(suv);   /* structured deinterleave */
        cb = vandq_u16(uv.val[0], mask10);
        cr = vandq_u16(uv.val[1], mask10);
    } else {
        cb = vandq_u16(vld1q_u16(su), mask10);
        cr = vandq_u16(vld1q_u16(sv), mask10);
    }

    /* ---- Q10 deltas at chroma rate ---- */
    int16x8_t dr = tm_delta8_neon(
        cr, vdupq_n_s16((int16_t)state->delta_coef_r));
    int16x8_t db = tm_delta8_neon(
        cb, vdupq_n_s16((int16_t)state->delta_coef_b));
    int16x8_t dg = tm_delta8_g_neon(
        cb, cr,
        vdupq_n_s16((int16_t)state->delta_coef_g_cb),
        vdupq_n_s16((int16_t)state->delta_coef_g_cr));

    /* ---- expand to luma rate: zip with itself duplicates each word ---- */
    int16x8_t dr0 = vzip1q_s16(dr, dr), dr1 = vzip2q_s16(dr, dr);
    int16x8_t dg0 = vzip1q_s16(dg, dg), dg1 = vzip2q_s16(dg, dg);
    int16x8_t db0 = vzip1q_s16(db, db), db1 = vzip2q_s16(db, db);

    /* ---- both rows: compute all 96 indices, then look them all up ---- */
    uint8_t obuf[6][16];

    {
        uint16x8_t i0, i1;
        for (int row = 0; row < 2; row++) {
            const uint16_t *sy = row ? sy1 : sy0;
            uint16x8_t y0 = vandq_u16(vld1q_u16(sy),     mask10);
            uint16x8_t y1 = vandq_u16(vld1q_u16(sy + 8), mask10);
            tm_indices_neon(y0, y1, dr0, dr1, &i0, &i1);
            tm_lookup16_neon(lut, i0, i1, obuf[row * 3 + 0]);
            tm_indices_neon(y0, y1, dg0, dg1, &i0, &i1);
            tm_lookup16_neon(lut, i0, i1, obuf[row * 3 + 1]);
            tm_indices_neon(y0, y1, db0, db1, &i0, &i1);
            tm_lookup16_neon(lut, i0, i1, obuf[row * 3 + 2]);
        }
    }

    /* ---- luma dot + store, both rows ----
     * Widening multiply-accumulate straight from the bytes:
     * (67r + 174g + 15b + 128) >> 8 in modular 16-bit (max 65408). */
    uint8x8_t c67  = vdup_n_u8(67);
    uint8x8_t c174 = vdup_n_u8(174);
    uint8x8_t c15  = vdup_n_u8(15);
    uint8x16_t rgb[6];
    for (int i = 0; i < 6; i++)
        rgb[i] = vld1q_u8(obuf[i]);

    uint16x8_t ye[2];       /* even-extract source: y words of row 0 */
    uint8x16_t reb = rgb[0], geb = rgb[1], beb = rgb[2];

    for (int row = 0; row < 2; row++) {
        uint8x16_t rb = rgb[row * 3 + 0];
        uint8x16_t gb = rgb[row * 3 + 1];
        uint8x16_t bb = rgb[row * 3 + 2];

        uint16x8_t ys[2];
        for (int h = 0; h < 2; h++) {
            uint8x8_t r8 = h ? vget_high_u8(rb) : vget_low_u8(rb);
            uint8x8_t g8 = h ? vget_high_u8(gb) : vget_low_u8(gb);
            uint8x8_t b8 = h ? vget_high_u8(bb) : vget_low_u8(bb);
            uint16x8_t acc = vmull_u8(r8, c67);
            acc = vmlal_u8(acc, g8, c174);
            acc = vmlal_u8(acc, b8, c15);
            ys[h] = vshrq_n_u16(vaddq_u16(acc, vdupq_n_u16(128)), 8);
        }
        vst1q_u8(row ? dy1 : dy0,
                 vcombine_u8(vmovn_u16(ys[0]), vmovn_u16(ys[1])));
        if (row == 0) {
            ye[0] = ys[0];
            ye[1] = ys[1];
        }
    }

    /* ---- chroma outputs from row 0's top-left pixels ----
     * uzp1 of the byte vector with itself compacts the even bytes into
     * the low half; the y values are still words, so uzp1 on the word
     * pair picks the even-position words directly. */
    uint8x16_t re = vuzp1q_u8(reb, reb);
    uint8x16_t ge = vuzp1q_u8(geb, geb);
    uint8x16_t be = vuzp1q_u8(beb, beb);
    uint16x8_t yev = vuzp1q_u16(ye[0], ye[1]);

    tm_chroma_dots_neon(vmovl_u8(vget_low_u8(re)),
                        vmovl_u8(vget_low_u8(ge)),
                        vmovl_u8(vget_low_u8(be)),
                        yev,
                        cbs, crs, cmin, cmax, du, dv);
}

/* -----------------------------------------------------------------------
 * Frame driver.  p010 is a compile-time literal at both call sites.
 * ----------------------------------------------------------------------- */

static inline FUSED_ALWAYS_INLINE void
tm_frame_neon(const fused_hdr_internal_t *state,
              const uint16_t *src_y, int src_y_stride,
              const uint16_t *src_u, const uint16_t *src_v,
              const uint16_t *src_uv, int src_uv_stride, int p010,
              uint8_t *dst_y, int dst_y_stride,
              uint8_t *dst_u, int dst_uv_stride, uint8_t *dst_v,
              int width, int height)
{
    int chroma_w = width  / 2;
    int chroma_h = height / 2;

    int src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    int src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);

    int simd_cw = chroma_w & ~7;

    int16x8_t cbs  = vdupq_n_s16((int16_t)state->cb_out_scale);
    int16x8_t crs  = vdupq_n_s16((int16_t)state->cr_out_scale);
    int16x8_t cmin = vdupq_n_s16((int16_t)state->chroma_out_min);
    int16x8_t cmax = vdupq_n_s16((int16_t)state->chroma_out_max);

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

        int cx = 0;
        for (; cx < simd_cw; cx += 8) {
            tm_chunk_neon(state,
                          p010 ? 0 : su + cx, p010 ? 0 : sv + cx,
                          p010 ? suv + cx * 2 : 0, p010,
                          sy0 + cx * 2, sy1 + cx * 2,
                          dy0 + cx * 2, dy1 + cx * 2,
                          du + cx, dv + cx,
                          cbs, crs, cmin, cmax);
        }
        for (; cx < chroma_w; cx++) {
            int cb_10 = p010 ? (suv[cx * 2] & 0x3FF) : (su[cx] & 0x3FF);
            int cr_10 = p010 ? (suv[cx * 2 + 1] & 0x3FF) : (sv[cx] & 0x3FF);
            fused_tm_block(state, cb_10, cr_10,
                           sy0 + cx * 2, sy1 + cx * 2,
                           dy0 + cx * 2, dy1 + cx * 2,
                           &du[cx], &dv[cx]);
        }
    }
}

FUSED_HOT
void fused_tonemap_apply_neon(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_neon(state, src_y, src_y_stride, src_u, src_v,
                  0, src_uv_stride, 0,
                  dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                  width, height);
}

FUSED_HOT
void fused_tonemap_apply_p010_neon(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    tm_frame_neon(state, src_y, src_y_stride, 0, 0,
                  src_uv, src_uv_stride, 1,
                  dst_y, dst_y_stride, dst_u, dst_uv_stride, dst_v,
                  width, height);
}

#endif /* __aarch64__ || _M_ARM64 */
