/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#include "funnelcake.h"
#include "funnelcake_helpers.h"
#include <assert.h>
#include <limits.h>
int main(void)
{
    fused_hdr_ctx_t h = {0};
    h.src_width = 1073741824;
    h.src_height = 2;
    h.src_y_stride = h.src_uv_stride = 32;
    assert(fused_hdr_init(&h) == FUSED_ERR_BAD_DIMENSIONS);
    fused_scaler_ctx_t s = {0};
    s.src_width = 128;
    s.src_height = 64;
    s.src_y_stride = s.src_uv_stride = INT_MAX & ~31;
    s.requested_flags = FUSED_SCALE_2X;
    assert(fused_scaler_init(&s) == FUSED_ERR_BAD_DIMENSIONS);
    int y, uv;
    fused_plane_strides_16(INT_MAX - 1, &y, &uv);
    assert(y == 0);
    fused_plane_strides(128, &y, &uv);
    assert(y == 128 && uv == 64);
    return 0;
}
