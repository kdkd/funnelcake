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
 * arithmetic (Q10 chroma deltas, index adds, luma dot, chroma-rate
 * gamut/encode dots) and performs the three byte lookups per pixel as
 * scalar L1 loads.  Index vectors leave the register file as 64-bit
 * halves (two moves per eight indices; the vector-to-GPR path is a
 * single port on Apple and Arm big cores, so it is kept lightly loaded),
 * each 16-bit index is peeled off with a shift or bitfield extract, and
 * the gathered bytes are reassembled into a vector without touching the
 * stack.  The 48 independent lookups per 16-pixel row segment pipeline
 * well behind deep out-of-order windows.  The table is a 4096-entry
 * padded copy so that no clamp is needed on the indices (see
 * TM_LUT_BIAS below).
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

#if defined(__aarch64__)

#include "internal.h"
#include "tonemap.h"

#include <arm_neon.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Q10 delta evaluation for 8 chroma codes (one uint16x8) -> 8 deltas.
 *
 * vqrdmulh of (code - 512) << 5 by the Q10 coefficient computes
 *   ((x*32) * coef + 16384) >> 15  ==  (x*coef + 512) >> 10
 * which is exactly the scalar table definition, entirely in 16-bit.
 * (The saturating case of vqrdmulh needs both operands at -32768;
 * x << 5 never goes below -16384, so it cannot fire.)
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) int16x8_t
tm_delta8_neon(uint16x8_t codes, int16x8_t coef, int16x8_t bias)
{
    int16x8_t x = vshlq_n_s16(
        vsubq_s16(vreinterpretq_s16_u16(codes), vdupq_n_s16(512)), 5);
    return vaddq_s16(vqrdmulhq_s16(x, coef), bias);
}

/* dG = g_cr(Cr) + g_cb(Cb): each term rounds exactly like its scalar
 * table entry, and the sum (magnitude < 1024) stays in 16-bit. */
static inline __attribute__((always_inline)) int16x8_t
tm_delta8_g_neon(uint16x8_t cb, uint16x8_t cr, int16x8_t gcb, int16x8_t gcr,
                 int16x8_t bias)
{
    int16x8_t c512 = vdupq_n_s16(512);
    int16x8_t xb = vshlq_n_s16(
        vsubq_s16(vreinterpretq_s16_u16(cb), c512), 5);
    int16x8_t xr = vshlq_n_s16(
        vsubq_s16(vreinterpretq_s16_u16(cr), c512), 5);
    return vaddq_s16(vaddq_s16(vqrdmulhq_s16(xr, gcr), bias),
                     vqrdmulhq_s16(xb, gcb));
}

/* -----------------------------------------------------------------------
 * Clamp-free indexing.
 *
 * The scalar reference clamps Y' + delta to [0, 1023] before indexing the
 * 1024-entry pq_to_sdr table.  |delta| is bounded by 512 * 1.8814 < 1024
 * (the largest Q10 NCL coefficient, full-range Cb - see tonemap.c), so
 * Y' + delta + TM_LUT_BIAS always lands in [60, 3011].  The kernel
 * therefore indexes a 4096-entry copy of the table whose entries outside
 * the central 1024 replicate the end values, which is exactly what the
 * clamp would have selected.  The bias is folded into the deltas at
 * chroma rate (3 adds per 16 pixels), so each per-channel index costs a
 * single add.
 * ----------------------------------------------------------------------- */

#define TM_LUT_BIAS 1024
#define TM_LUT_PAD  4096

static void tm_build_padded_lut(const uint8_t *lut, uint8_t *pad)
{
    memset(pad, lut[0], TM_LUT_BIAS);
    memcpy(pad + TM_LUT_BIAS, lut, 1024);
    memset(pad + TM_LUT_BIAS + 1024, lut[1023],
           TM_LUT_PAD - TM_LUT_BIAS - 1024);
}

static inline __attribute__((always_inline)) void
tm_indices_neon(uint16x8_t y0, uint16x8_t y1, int16x8_t d0, int16x8_t d1,
                uint16x8_t *i0_out, uint16x8_t *i1_out)
{
    *i0_out = vaddq_u16(y0, vreinterpretq_u16_s16(d0));
    *i1_out = vaddq_u16(y1, vreinterpretq_u16_s16(d1));
}

/* 8 lookups from one index vector into a byte table, packed into a
 * 64-bit GPR.  The index vector leaves the register file as two 64-bit
 * halves (one fmov/umov each); the vector-to-GPR path is a single port
 * on Apple and Arm big cores, so the fewer lane moves the better.  The
 * gathered bytes are OR-shifted into a GPR, so no stack buffer is
 * involved and there is no wide reload of narrow stores to fail
 * store forwarding.  clang lowers the packing to ld1 lane loads and gcc
 * keeps the ldrb + orr form; both measure well. */
static inline __attribute__((always_inline)) uint64_t
tm_gather4_neon(const uint8_t *lut, uint64_t q)
{
    return (uint64_t)lut[q & 0xFFFF]
         | (uint64_t)lut[(q >> 16) & 0xFFFF] << 8
         | (uint64_t)lut[(q >> 32) & 0xFFFF] << 16
         | (uint64_t)lut[q >> 48] << 24;
}

static inline __attribute__((always_inline)) uint8x8_t
tm_lookup8_neon(const uint8_t *lut, uint16x8_t idx)
{
    uint64x2_t q = vreinterpretq_u64_u16(idx);
    uint64_t lo = tm_gather4_neon(lut, vgetq_lane_u64(q, 0));
    uint64_t hi = tm_gather4_neon(lut, vgetq_lane_u64(q, 1));
    return vcreate_u8(lo | hi << 32);
}

static inline __attribute__((always_inline)) uint8x16_t
tm_lookup16_neon(const uint8_t *lut, uint16x8_t i0, uint16x8_t i1)
{
    return vcombine_u8(tm_lookup8_neon(lut, i0), tm_lookup8_neon(lut, i1));
}

/* -----------------------------------------------------------------------
 * Chroma-rate gamut + BT.709 encode dots for 8 samples.
 * Mirrors fused_tm_rgb_to_chroma_sdr exactly (32-bit, same rounding).
 * Inputs are the even-position r/g/b/y values widened to 16-bit.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
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

static inline __attribute__((always_inline)) void
tm_chunk_neon(const fused_hdr_internal_t *state, const uint8_t *lut,
              const uint16_t *su, const uint16_t *sv,
              const uint16_t *suv, int p010,
              const uint16_t *sy0, const uint16_t *sy1,
              uint8_t *dy0, uint8_t *dy1,
              uint8_t *du, uint8_t *dv,
              int16x8_t cbs, int16x8_t crs, int16x8_t cmin, int16x8_t cmax)
{
    uint16x8_t mask10 = vdupq_n_u16(0x3FF);
    int16x8_t bias = vdupq_n_s16(TM_LUT_BIAS);

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
        cr, vdupq_n_s16((int16_t)state->delta_coef_r), bias);
    int16x8_t db = tm_delta8_neon(
        cb, vdupq_n_s16((int16_t)state->delta_coef_b), bias);
    int16x8_t dg = tm_delta8_g_neon(
        cb, cr,
        vdupq_n_s16((int16_t)state->delta_coef_g_cb),
        vdupq_n_s16((int16_t)state->delta_coef_g_cr), bias);

    /* ---- expand to luma rate: zip with itself duplicates each word ---- */
    int16x8_t dr0 = vzip1q_s16(dr, dr), dr1 = vzip2q_s16(dr, dr);
    int16x8_t dg0 = vzip1q_s16(dg, dg), dg1 = vzip2q_s16(dg, dg);
    int16x8_t db0 = vzip1q_s16(db, db), db1 = vzip2q_s16(db, db);

    /* ---- both rows: indices, lookups, luma dot, store ---- */
    uint8x8_t c67  = vdup_n_u8(67);
    uint8x8_t c174 = vdup_n_u8(174);
    uint8x8_t c15  = vdup_n_u8(15);
    uint8x16_t reb = vdupq_n_u8(0), geb = reb, beb = reb, yeb = reb;

    for (int row = 0; row < 2; row++) {
        const uint16_t *sy = row ? sy1 : sy0;
        uint16x8_t y0 = vandq_u16(vld1q_u16(sy),     mask10);
        uint16x8_t y1 = vandq_u16(vld1q_u16(sy + 8), mask10);
        uint16x8_t i0, i1;

        tm_indices_neon(y0, y1, dr0, dr1, &i0, &i1);
        uint8x16_t rb = tm_lookup16_neon(lut, i0, i1);
        tm_indices_neon(y0, y1, dg0, dg1, &i0, &i1);
        uint8x16_t gb = tm_lookup16_neon(lut, i0, i1);
        tm_indices_neon(y0, y1, db0, db1, &i0, &i1);
        uint8x16_t bb = tm_lookup16_neon(lut, i0, i1);

        /* (67r + 174g + 15b + 128) >> 8: widening multiply-accumulate
         * straight from the bytes.  The accumulator peaks at 65280, so
         * the rounding narrow's +128 cannot wrap 16 bits. */
        uint16x8_t acc_lo = vmull_u8(vget_low_u8(rb), c67);
        acc_lo = vmlal_u8(acc_lo, vget_low_u8(gb), c174);
        acc_lo = vmlal_u8(acc_lo, vget_low_u8(bb), c15);
        uint16x8_t acc_hi = vmull_u8(vget_high_u8(rb), c67);
        acc_hi = vmlal_u8(acc_hi, vget_high_u8(gb), c174);
        acc_hi = vmlal_u8(acc_hi, vget_high_u8(bb), c15);
        uint8x16_t yb = vcombine_u8(vrshrn_n_u16(acc_lo, 8),
                                    vrshrn_n_u16(acc_hi, 8));
        vst1q_u8(row ? dy1 : dy0, yb);
        if (row == 0) {
            reb = rb; geb = gb; beb = bb; yeb = yb;
        }
    }

    /* ---- chroma outputs from row 0's top-left pixels ----
     * uzp1 of a byte vector with itself compacts the even bytes into the
     * low half. */
    uint8x16_t re = vuzp1q_u8(reb, reb);
    uint8x16_t ge = vuzp1q_u8(geb, geb);
    uint8x16_t be = vuzp1q_u8(beb, beb);
    uint8x16_t ye = vuzp1q_u8(yeb, yeb);

    tm_chroma_dots_neon(vmovl_u8(vget_low_u8(re)),
                        vmovl_u8(vget_low_u8(ge)),
                        vmovl_u8(vget_low_u8(be)),
                        vmovl_u8(vget_low_u8(ye)),
                        cbs, crs, cmin, cmax, du, dv);
}

/* -----------------------------------------------------------------------
 * Frame driver.  p010 is a compile-time literal at both call sites.
 * ----------------------------------------------------------------------- */

static inline __attribute__((always_inline)) void
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

    uint8_t lut_pad[TM_LUT_PAD] __attribute__((aligned(64)));
    tm_build_padded_lut(state->pq_to_sdr, lut_pad);

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
            tm_chunk_neon(state, lut_pad,
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

__attribute__((hot))
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

__attribute__((hot))
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

#endif /* __aarch64__ */
