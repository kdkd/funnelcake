/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#ifndef FUNNELCAKE_TONEMAP_H
#define FUNNELCAKE_TONEMAP_H

#include "internal.h"
#include "funnelcake.h"

/* --------------------------------------------------------------------------
 * Shared per-block scalar pipeline
 *
 * One 4:2:0 block: reconstruct gamma-domain R'G'B' for the four luma
 * pixels (three delta adds each - the exact BT.2020 NCL inverse), tone
 * map every channel through the shared pq_to_sdr table, emit luma from
 * the BT.2020-coefficient dot (== BT.709 luma of the gamut-converted
 * RGB), and encode U/V from the top-left pixel's RGB via the gamut rows.
 *
 * This is the reference implementation the SIMD kernels must match bit
 * for bit; it is inline here so the scalar loops in tonemap.c and the
 * SIMD kernels' ragged-edge tails are the same code by construction.
 * -------------------------------------------------------------------------- */

static inline int fused_tm_clamp_i(int x, int lo, int hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline void fused_tm_pixel_rgb(
    const fused_hdr_internal_t *state,
    int y_pq, int cb_10, int cr_10,
    int *r_out, int *g_out, int *b_out)
{
    int pq_r = y_pq + state->pq_r_delta[cr_10];
    int pq_g = y_pq + state->pq_g_delta_cr[cr_10] + state->pq_g_delta_cb[cb_10];
    int pq_b = y_pq + state->pq_b_delta[cb_10];
    if (pq_r < 0) pq_r = 0;
    if (pq_r > 1023) pq_r = 1023;
    if (pq_g < 0) pq_g = 0;
    if (pq_g > 1023) pq_g = 1023;
    if (pq_b < 0) pq_b = 0;
    if (pq_b > 1023) pq_b = 1023;

    *r_out = state->pq_to_sdr[pq_r];
    *g_out = state->pq_to_sdr[pq_g];
    *b_out = state->pq_to_sdr[pq_b];
}

static inline int fused_tm_rgb_to_y_sdr(int r_sdr, int g_sdr, int b_sdr)
{
    /* BT.2020 luma coefficients * 256: 67 + 174 + 15 == 256.  Equals the
     * BT.709 luma of the gamut-converted RGB - see tonemap.c. */
    return (67 * r_sdr + 174 * g_sdr + 15 * b_sdr + 128) >> 8;
}

/*
 * BT.709 YCbCr re-encode with BT.2020 -> BT.709 gamut conversion.
 *
 * The reconstructed SDR R, G, B are BT.2020-primary.  The gamut
 * conversion is a 3x3 matrix M (applied in gamma domain - the standard
 * fast-path approximation), followed by the BT.709 YCbCr encode.  Both
 * are dot products, so they compose: the BT.709 RGB is never
 * materialized.
 *
 * Luma comes out especially clean: the BT.709 luma row applied to M
 * algebraically equals the BT.2020 luma coefficients (both reduce to the
 * Y row of the RGB->XYZ matrix for their primaries), so
 *   Y709(M * rgb2020) = [0.2627 0.6780 0.0593] . rgb2020
 * exactly - the gamut conversion costs luma nothing (see
 * fused_tm_rgb_to_y_sdr above).
 *
 * Chroma needs the R and B rows of M (BT.2020 -> BT.709 linear-domain
 * matrix, applied gamma-domain):
 *   R709 = 1.6605*R - 0.5876*G - 0.0728*B
 *   B709 = -0.0182*R - 0.1006*G + 1.1187*B
 * Fixed-point 8.8 with rows tweaked to sum to exactly 256 so neutral
 * (R=G=B) stays neutral, then Cb/Cr from the usual difference form with
 * dst_range scaling baked into cb/cr_out_scale at init.
 */
static inline void fused_tm_rgb_to_chroma_sdr(
    const fused_hdr_internal_t *state,
    int r_sdr, int g_sdr, int b_sdr, int y_sdr,
    uint8_t *u_out, uint8_t *v_out)
{
    /* BT.2020->BT.709 gamut rows * 256, each summing to exactly 256 */
    int r709 = fused_tm_clamp_i((425 * r_sdr - 150 * g_sdr - 19 * b_sdr + 128) >> 8,
                                0, 255);
    int b709 = fused_tm_clamp_i((-5 * r_sdr - 26 * g_sdr + 287 * b_sdr + 128) >> 8,
                                0, 255);

    /* +128 then arithmetic >>8 is round-half-up for either sign */
    int cb_sdr = 128 + (((b709 - y_sdr) * state->cb_out_scale + 128) >> 8);
    int cr_sdr = 128 + (((r709 - y_sdr) * state->cr_out_scale + 128) >> 8);

    *u_out = (uint8_t)fused_tm_clamp_i(cb_sdr, state->chroma_out_min,
                                       state->chroma_out_max);
    *v_out = (uint8_t)fused_tm_clamp_i(cr_sdr, state->chroma_out_min,
                                       state->chroma_out_max);
}

static inline void fused_tm_block(
    const fused_hdr_internal_t *state,
    int cb_10, int cr_10,
    const uint16_t *sy0, const uint16_t *sy1,   /* already offset to x */
    uint8_t *dy0, uint8_t *dy1,
    uint8_t *du, uint8_t *dv)
{
    int r_sdr, g_sdr, b_sdr;

    int y_pq = sy0[0] & 0x3FF;
    fused_tm_pixel_rgb(state, y_pq, cb_10, cr_10, &r_sdr, &g_sdr, &b_sdr);
    int y_sdr = fused_tm_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr);
    dy0[0] = (uint8_t)fused_tm_clamp_i(y_sdr, 0, 255);
    fused_tm_rgb_to_chroma_sdr(state, r_sdr, g_sdr, b_sdr, y_sdr, du, dv);

    y_pq = sy0[1] & 0x3FF;
    fused_tm_pixel_rgb(state, y_pq, cb_10, cr_10, &r_sdr, &g_sdr, &b_sdr);
    dy0[1] = (uint8_t)fused_tm_clamp_i(
        fused_tm_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);

    y_pq = sy1[0] & 0x3FF;
    fused_tm_pixel_rgb(state, y_pq, cb_10, cr_10, &r_sdr, &g_sdr, &b_sdr);
    dy1[0] = (uint8_t)fused_tm_clamp_i(
        fused_tm_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);

    y_pq = sy1[1] & 0x3FF;
    fused_tm_pixel_rgb(state, y_pq, cb_10, cr_10, &r_sdr, &g_sdr, &b_sdr);
    dy1[1] = (uint8_t)fused_tm_clamp_i(
        fused_tm_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);
}

/* --------------------------------------------------------------------------
 * Tone mapping LUT generation
 *
 * Fills the tone mapping LUT arrays (lut_y, pq_to_sdr, the chroma delta
 * tables) and the SDR re-encode constants inside fused_hdr_internal_t
 * based on the source transfer function and tone mapping configuration.
 * All color science - EOTF, tone curve, SDR OETF, limited/full range -
 * is folded into the tables here; the per-pixel path is integer-only.
 * Called once at init time by fused_hdr_init.
 *
 * hdr       : internal state whose LUT arrays will be written.
 * src_transfer : FUSED_TRC_PQ or FUSED_TRC_HLG.
 * tm        : tone mapping configuration (curve preset, peak/target nits,
 *             src/dst range, optional custom LUT pointer).
 * log_warn  : logging config for diagnostic messages (may be NULL).
 * -------------------------------------------------------------------------- */

void fused_tonemap_generate_luts(fused_hdr_internal_t *hdr,
                                 int src_transfer,
                                 const fused_tonemap_config_t *tm,
                                 const fused_log_config_t *log_warn);


/* --------------------------------------------------------------------------
 * Tone mapping application - planar I010 chroma
 *
 * Applies the precomputed LUTs to 10-bit YUV420 planes, producing 8-bit
 * SDR output planes.
 *
 * Custom LUT: fast luma lookup plus simple chroma scaling.
 * Built-in curves: reconstructs linear-light R, G, B from YCbCr for each
 *   luma pixel, tone-maps each channel through the LUTs, then recomputes
 *   BT.709 Y.  Chroma is written once per 2x2 block from the top-left
 *   reconstructed RGB sample, matching the previous sampling position.
 *
 * Width and height are luma dimensions (chroma is width/2 x height/2).
 * Strides are in bytes.
 * -------------------------------------------------------------------------- */

void fused_tonemap_apply(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);


/* --------------------------------------------------------------------------
 * Tone mapping application - interleaved P010 chroma
 *
 * Same as fused_tonemap_apply but reads chroma from a single interleaved
 * uint16_t UV plane (even indices = U, odd indices = V) instead of
 * separate U/V planes.  Used for the tonemap_1x path when the source
 * format is P010.
 * -------------------------------------------------------------------------- */

void fused_tonemap_apply_p010(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);


/* --------------------------------------------------------------------------
 * Built-in-curve implementations behind the dispatch above.
 *
 * The scalar functions are the reference; SIMD kernels must match them bit
 * for bit (the whole pipeline is integer LUTs and fixed-point dots, so
 * exact parity is achievable and enforced by tests).
 * -------------------------------------------------------------------------- */

void fused_tonemap_apply_scalar(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);

void fused_tonemap_apply_p010_scalar(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);

#if defined(__x86_64__)

/* AVX-512 (VBMI) kernels in tonemap_avx512.c.  Self-stubbed when the
 * compiler lacks AVX-512 support: fused_tonemap_avx512_compiled() returns
 * 0 and the entry points delegate to the scalar reference. */

int fused_tonemap_avx512_compiled(void);

void fused_tonemap_apply_avx512(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);

void fused_tonemap_apply_p010_avx512(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);

/* AVX2 kernels in tonemap_avx2.c (compiled unconditionally with -mavx2,
 * selected at runtime behind has_avx2). */

void fused_tonemap_apply_avx2(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);

void fused_tonemap_apply_p010_avx2(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height);

#endif /* __x86_64__ */

#endif /* FUNNELCAKE_TONEMAP_H */
