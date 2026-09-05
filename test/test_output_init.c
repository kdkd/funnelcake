/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#include "funnelcake.h"
#include <assert.h>
static void zeros(const unsigned char *p, size_t n)
{
    assert(p);
    for(size_t i=0;i<n;i++) assert(p[i]==0);
}
int main(void)
{
    fused_scaler_ctx_t c={0};
    c.src_width=136;c.src_height=64;c.src_y_stride=160;c.src_uv_stride=96;
    c.requested_flags=FUSED_SCALE_2X;c.log_warnings.target=FUSED_LOG_SUPPRESS;
    assert(fused_scaler_init(&c)>=0);
    fused_scale_output_t *o=&c.outputs[1];
    zeros(o->plane_y,(size_t)o->y_stride*o->height);
    zeros(o->plane_u,(size_t)o->uv_stride*(o->height/2));
    zeros(o->plane_v,(size_t)o->uv_stride*(o->height/2));
    fused_scaler_free(&c);
    fused_hdr_ctx_t h={0};
    h.src_width=136;h.src_height=64;h.src_y_stride=288;h.src_uv_stride=160;
    h.requested_flags=h.hdr_flags=FUSED_SCALE_2X;h.log_warnings.target=FUSED_LOG_SUPPRESS;
    assert(fused_hdr_init(&h)>=0);
    fused_hdr_output_t *p=&h.hdr_outputs[1];
    zeros((const unsigned char*)p->plane_y,(size_t)p->y_stride*p->height);
    zeros((const unsigned char*)p->plane_u,(size_t)p->uv_stride*(p->height/2));
    fused_hdr_free(&h);
    return 0;
}
