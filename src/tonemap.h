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
 * Tone mapping LUT generation
 *
 * Fills the tone mapping LUT arrays (lut_y, pq_to_linear, linear_to_sdr)
 * inside fused_hdr_internal_t based on the source transfer function and
 * tone mapping configuration.  Called once at init time by fused_hdr_init.
 *
 * hdr       : internal state whose LUT arrays will be written.
 * src_transfer : FUSED_TRC_PQ or FUSED_TRC_HLG.
 * tm        : tone mapping configuration (curve preset, peak/target nits,
 *             optional custom LUT pointer).
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
 * Luma pass:   fast LUT lookup: dst_y[x] = lut_y[src_y[x] & 0x3FF]
 * Chroma pass: reconstructs linear-light R, G, B from YCbCr using the
 *   pq_to_linear LUT, tone-maps each channel individually through the
 *   linear_to_sdr LUT, then recomputes BT.709 YCbCr from the results.
 *   Also overwrites the luma output at chroma resolution (2×2 blocks)
 *   to keep Y consistent with the per-channel tone mapping.
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


#endif /* FUNNELCAKE_TONEMAP_H */
