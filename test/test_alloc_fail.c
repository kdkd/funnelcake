/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#include "funnelcake.h"
#include "internal.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

/* No actual memory exhaustion or signals: return ENOMEM at each allocation
 * site and account for every allocation made by init/free. */
static void *live[512];
static size_t calls, fail_at, live_count;
static int persistent;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "allocation test line %d, failure %zu, persistent %d: %s\n", \
            __LINE__, fail_at, persistent, #c); exit(1); } } while (0)

static int fail(void)
{
    calls++;
    return fail_at && (calls == fail_at || (persistent && calls > fail_at));
}
static void remember(void *p)
{
    CHECK(p && live_count < sizeof(live) / sizeof(live[0]));
    live[live_count++] = p;
}
void *test_calloc(size_t count, size_t size)
{
    if (fail()) return NULL;
    void *p = calloc(count, size);
    if (p) remember(p);
    return p;
}
int test_posix_memalign(void **out, size_t alignment, size_t size)
{
    if (fail()) return ENOMEM;
    int rc = posix_memalign(out, alignment, size);
    if (!rc) remember(*out);
    return rc;
}
void test_free(void *p)
{
    if (!p) return;
    size_t i;
    for (i = 0; i < live_count; i++) if (live[i] == p) break;
    CHECK(i < live_count);
    live[i] = live[--live_count];
    free(p);
}

static void exercise(int hdr, int mode, unsigned flags, unsigned up)
{
    if (!hdr) {
        fused_scaler_ctx_t c = {0};
        c.src_width = 192; c.src_height = 96;
        c.src_y_stride = 192; c.src_uv_stride = 96;
        c.requested_flags = flags;
        c.upscale_flags = up;
        c.upscale_tail_1_5x = !!up;
        c.log_errors.target = c.log_warnings.target = FUSED_LOG_SUPPRESS;
        int rc = fused_scaler_init(&c);
        if (rc < 0) CHECK(!c._internal && !live_count);
        if (rc >= 0) {
            fused_internal_t *s = c._internal;
            CHECK(s);
            if (c.achieved_flags) CHECK(s->params.scratch_pool);
            if (c.achieved_upscale_flags || c.achieved_upscale_tail)
                CHECK(s->params.upscale_scratch);
        }
        fused_scaler_free(&c);
        fused_scaler_free(&c);
    } else {
        fused_hdr_ctx_t c = {0};
        c.src_width = 192; c.src_height = 96;
        c.src_y_stride = c.src_uv_stride = 384;
        c.src_format = (mode & 1) ? FUSED_PIX_P010 : FUSED_PIX_I010;
        c.src_transfer = FUSED_TRC_PQ;
        c.requested_flags = flags;
        c.hdr_flags = (mode & 2) ? flags : 0;
        c.sdr_flags = flags;
        c.tonemap_1x = 1;
        c.upscale_flags = c.upscale_sdr_flags = up;
        c.upscale_tail_1_5x = !!up;
        c.log_errors.target = c.log_warnings.target = FUSED_LOG_SUPPRESS;
        int rc = fused_hdr_init(&c);
        if (rc < 0) CHECK(!c._internal && !live_count);
        if (rc >= 0) {
            fused_hdr_internal_t *s = c._internal;
            CHECK(s);
            if (c.achieved_hdr_flags || c.achieved_sdr_flags)
                CHECK(s->params.scratch_pool);
            if (c.achieved_upscale_flags || c.achieved_upscale_tail)
                CHECK(s->params.upscale_scratch_hdr);
            if (s->params.is_p010)
                CHECK(s->params.p010_tmp_u && s->params.p010_tmp_v);
        }
        fused_hdr_free(&c);
        fused_hdr_free(&c);
    }
    CHECK(!live_count);
}

int main(void)
{
    size_t trials = 0;
    const unsigned masks[] = {FUSED_SCALE_16X, FUSED_SCALE_12X, 0xaa, 0x55};
    for (int hdr = 0; hdr < 2; hdr++)
    for (int mode = 0; mode < (hdr ? 4 : 1); mode++)
    for (size_t m = 0; m < sizeof(masks) / sizeof(masks[0]); m++) {
        unsigned up = FUSED_UPSCALE_2X | FUSED_UPSCALE_4X;
        fail_at = calls = 0;
        exercise(hdr, mode, masks[m], up);
        size_t total = calls;
        CHECK(total > 0);
        for (persistent = 0; persistent < 2; persistent++)
        for (size_t n = 1; n <= total; n++) {
            calls = 0; fail_at = n;
            exercise(hdr, mode, masks[m], up);
            fail_at = calls = 0;
            exercise(hdr, mode, masks[m], up);
            trials++;
        }
    }
    printf("Allocation failure and recovery: %zu cases passed\n", trials);
    return 0;
}
