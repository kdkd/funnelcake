/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#include "funnelcake.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
static void geometry(const fused_scale_output_t *a, const fused_scale_output_t *b)
{
    assert(a->width == b->width && a->height == b->height);
    assert(a->y_stride == b->y_stride && a->uv_stride == b->uv_stride);
    assert(a->fallback == b->fallback);
    assert(!a->plane_y && !a->plane_u && !a->plane_v);
}
int main(void)
{
    for (int flags = 1; flags < 256; ++flags) {
        fused_scaler_ctx_t c = {0}, q = {0};
        c.src_width=256; c.src_height=192; c.src_y_stride=256; c.src_uv_stride=128;
        c.requested_flags=flags;
        c.log_errors.target=c.log_warnings.target=FUSED_LOG_SUPPRESS;
        fused_scaler_ctx_t before=c;
        size_t bytes=0;
        int qr=fused_scaler_query(&c,&q,&bytes);
        assert(!memcmp(&before,&c,sizeof(c)));
        int ir=fused_scaler_init(&c);
        assert(qr==ir);
        if (qr>=0) {
            assert(bytes>0 && !q._internal);
            assert(q.achieved_flags==c.achieved_flags && q.rejected_flags==c.rejected_flags);
            for (int i=0;i<8;i++) geometry(&q.outputs[i],&c.outputs[i]);
        }
        fused_scaler_free(&c);
    }
    for (int format=0;format<4;format++) for (int mode=0;mode<4;mode++) {
        fused_hdr_ctx_t c={0},q={0};size_t bytes=0;
        c.src_width=128;c.src_height=96;c.src_y_stride=256;c.src_uv_stride=256;
        c.src_format=format;c.requested_flags=FUSED_SCALE_2X;
        c.hdr_flags=(mode&1)?FUSED_SCALE_2X:0;
        c.sdr_flags=(mode&2)?FUSED_SCALE_2X:0;
        c.tonemap_1x=1;c.upscale_flags=FUSED_UPSCALE_2X;
        c.upscale_tail_1_5x=1;c.upscale_sdr_flags=FUSED_UPSCALE_2X;
        c.log_errors.target=c.log_warnings.target=FUSED_LOG_SUPPRESS;
        fused_hdr_ctx_t before=c;
        int qr=fused_hdr_query(&c,&q,&bytes);
        assert(!memcmp(&before,&c,sizeof(c)));
        int ir=fused_hdr_init(&c);assert(qr==ir && qr>=0);
        assert(bytes>0 && !q._internal);
        assert(q.achieved_hdr_flags==c.achieved_hdr_flags);
        assert(q.achieved_sdr_flags==c.achieved_sdr_flags);
        assert(q.achieved_upscale_flags==c.achieved_upscale_flags);
        for(int i=0;i<8;i++) {
            assert(q.hdr_outputs[i].width==c.hdr_outputs[i].width);
            assert(q.hdr_outputs[i].height==c.hdr_outputs[i].height);
            assert(!q.hdr_outputs[i].plane_y);
            geometry(&q.sdr_outputs[i],&c.sdr_outputs[i]);
        }
        geometry(&q.output_1x,&c.output_1x);
        for(int i=0;i<FUSED_MAX_UPSCALE_STEPS;i++) {
            assert(q.upscale_hdr_outputs[i].width==c.upscale_hdr_outputs[i].width);
            assert(!q.upscale_hdr_outputs[i].plane_y);
            geometry(&q.upscale_sdr_outputs[i],&c.upscale_sdr_outputs[i]);
        }
        fused_hdr_free(&c);
    }
    puts("query geometry matches initialization");
    return 0;
}
