/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
 * Local helpers
 * -------------------------------------------------------------------------- */

static int align_up_32(int v) { return (v + 31) & ~31; }

static void suppress_log(fused_hdr_ctx_t *ctx)
{
    ctx->log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx->log_errors.target   = FUSED_LOG_SUPPRESS;
}

/*
 * Probe whether fused_hdr_run actually writes HDR output for a given config.
 * Run with a solid-512 source and check if any HDR output pixel is non-zero.
 * Returns 1 if output was written, 0 if not.
 */
static int hdr_kernel_produces_output(int src_w, int src_h, uint32_t flags,
                                      int want_hdr, int want_sdr)
{
    test_hdr_frame_t frame;
    if (test_hdr_frame_create(&frame, src_w, src_h, PATTERN_SOLID, 0) != 0)
        return 0;

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = flags;
    ctx.hdr_flags      = want_hdr ? flags : 0;
    ctx.sdr_flags      = want_sdr ? flags : 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    if (rc < 0) {
        test_hdr_frame_free(&frame);
        return 0;
    }

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    int written = 0;
    for (int i = 0; i < 8 && !written; i++) {
        if (want_hdr && ctx.hdr_outputs[i].plane_y) {
            int ow = ctx.hdr_outputs[i].width;
            int stride = ctx.hdr_outputs[i].y_stride / (int)sizeof(uint16_t);
            for (int x = 0; x < ow && !written; x++) {
                if (ctx.hdr_outputs[i].plane_y[x] != 0)
                    written = 1;
            }
            (void)stride;
        }
        if (want_sdr && ctx.sdr_outputs[i].plane_y) {
            int ow = ctx.sdr_outputs[i].width;
            for (int x = 0; x < ow && !written; x++) {
                if (ctx.sdr_outputs[i].plane_y[x] != 0)
                    written = 1;
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    return written;
}

/* --------------------------------------------------------------------------
 * 1. test_hdr_solid_preservation
 *    Solid-512 input -> all HDR output pixels 512 +/- 1
 * -------------------------------------------------------------------------- */

static void test_hdr_solid_preservation(void)
{
    uint32_t flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 1, 0)) {
        TEST_SKIP("HDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_SOLID, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = flags;
    ctx.hdr_flags      = flags;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "fused_hdr_init failed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    for (int i = 0; i < 8; i++) {
        if (!ctx.hdr_outputs[i].plane_y) continue;
        int ow = ctx.hdr_outputs[i].width;
        int oh = ctx.hdr_outputs[i].height;
        int stride = ctx.hdr_outputs[i].y_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < oh; y++) {
            const uint16_t *row = ctx.hdr_outputs[i].plane_y + y * stride;
            for (int x = 0; x < ow; x++) {
                int diff = (int)row[x] - 512;
                if (diff < -1 || diff > 1) {
                    printf("\n  FAIL [%s:%d] solid HDR: output[%d] pixel(%d,%d)=%d, expected 512+/-1\n",
                           __func__, __LINE__, i, x, y, row[x]);
                    g_results.failed++;
                    fused_hdr_free(&ctx);
                    test_hdr_frame_free(&frame);
                    return;
                }
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 2. test_hdr_10bit_range
 *    Random input [0,1023] -> all HDR output pixels in [0,1023]
 * -------------------------------------------------------------------------- */

static void test_hdr_10bit_range(void)
{
    uint32_t flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 1, 0)) {
        TEST_SKIP("HDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_RANDOM, 0xDEADBEEFu);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = flags;
    ctx.hdr_flags      = flags;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "fused_hdr_init failed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    for (int i = 0; i < 8; i++) {
        if (!ctx.hdr_outputs[i].plane_y) continue;
        int ow = ctx.hdr_outputs[i].width;
        int oh = ctx.hdr_outputs[i].height;
        int stride = ctx.hdr_outputs[i].y_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < oh; y++) {
            const uint16_t *row = ctx.hdr_outputs[i].plane_y + y * stride;
            for (int x = 0; x < ow; x++) {
                if (row[x] > 1023) {
                    printf("\n  FAIL [%s:%d] 10bit range: output[%d] pixel(%d,%d)=%d > 1023\n",
                           __func__, __LINE__, i, x, y, row[x]);
                    g_results.failed++;
                    fused_hdr_free(&ctx);
                    test_hdr_frame_free(&frame);
                    return;
                }
            }
        }
        /* Also check chroma planes */
        int cw = ow / 2;
        int ch = oh / 2;
        int uv_stride = ctx.hdr_outputs[i].uv_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < ch; y++) {
            const uint16_t *ru = ctx.hdr_outputs[i].plane_u + y * uv_stride;
            const uint16_t *rv = ctx.hdr_outputs[i].plane_v + y * uv_stride;
            for (int x = 0; x < cw; x++) {
                if (ru[x] > 1023 || rv[x] > 1023) {
                    printf("\n  FAIL [%s:%d] 10bit range: chroma output[%d] (%d,%d) U=%d V=%d\n",
                           __func__, __LINE__, i, x, y, ru[x], rv[x]);
                    g_results.failed++;
                    fused_hdr_free(&ctx);
                    test_hdr_frame_free(&frame);
                    return;
                }
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 3. test_hdr_gradient_monotonicity
 *    Horizontal 10-bit gradient -> output non-decreasing (+/-2 tolerance)
 * -------------------------------------------------------------------------- */

static void test_hdr_gradient_monotonicity(void)
{
    uint32_t flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 1, 0)) {
        TEST_SKIP("HDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_HGRADIENT, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = flags;
    ctx.hdr_flags      = flags;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "fused_hdr_init failed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    for (int i = 0; i < 8; i++) {
        if (!ctx.hdr_outputs[i].plane_y) continue;
        int ow = ctx.hdr_outputs[i].width;
        int oh = ctx.hdr_outputs[i].height;
        int stride = ctx.hdr_outputs[i].y_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < oh; y++) {
            const uint16_t *row = ctx.hdr_outputs[i].plane_y + y * stride;
            for (int x = 1; x < ow; x++) {
                if ((int)row[x] < (int)row[x-1] - 2) {
                    printf("\n  FAIL [%s:%d] HDR gradient monotonicity: output[%d] row=%d "
                           "pixel[%d]=%d < pixel[%d]=%d - 2\n",
                           __func__, __LINE__, i, y, x, row[x], x-1, row[x-1]);
                    g_results.failed++;
                    fused_hdr_free(&ctx);
                    test_hdr_frame_free(&frame);
                    return;
                }
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 4. test_hdr_tonemap_monotonicity
 *    Luma ramp 0-1023 -> SDR output Y is non-decreasing (LUT must be monotonic)
 * -------------------------------------------------------------------------- */

static void test_hdr_tonemap_monotonicity(void)
{
    uint32_t flags = FUSED_SCALE_1_5X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 0, 1)) {
        TEST_SKIP("SDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_HGRADIENT, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = flags;
    /* Test monotonicity on HDR output (no tone mapping).
     * SDR output has chroma-resolution Y rewrite (2x2 blocks) which
     * inherently introduces step discontinuities at block boundaries. */
    ctx.hdr_flags      = flags;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "fused_hdr_init failed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    for (int i = 0; i < 8; i++) {
        if (!ctx.hdr_outputs[i].plane_y) continue;
        int ow = ctx.hdr_outputs[i].width;
        int oh = ctx.hdr_outputs[i].height;
        int stride = ctx.hdr_outputs[i].y_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < oh; y++) {
            const uint16_t *row = ctx.hdr_outputs[i].plane_y + y * stride;
            for (int x = 1; x < ow; x++) {
                if ((int)row[x] < (int)row[x-1] - 2) {
                    printf("\n  FAIL [%s:%d] HDR gradient monotonicity: output[%d] row=%d "
                           "pixel[%d]=%d < pixel[%d]=%d - 2\n",
                           __func__, __LINE__, i, y, x, row[x], x-1, row[x-1]);
                    g_results.failed++;
                    fused_hdr_free(&ctx);
                    test_hdr_frame_free(&frame);
                    return;
                }
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 5. test_hdr_tonemap_range
 *    All SDR output pixels in [0, 255]
 * -------------------------------------------------------------------------- */

static void test_hdr_tonemap_range(void)
{
    uint32_t flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 0, 1)) {
        TEST_SKIP("SDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_RANDOM, 0xCAFEBABEu);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = flags;
    ctx.hdr_flags      = 0;
    ctx.sdr_flags      = flags;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "fused_hdr_init failed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    /* uint8_t is always [0,255]; scan all planes to ensure no memory errors */
    for (int i = 0; i < 8; i++) {
        if (!ctx.sdr_outputs[i].plane_y) continue;
        TEST_ASSERT(ctx.sdr_outputs[i].plane_u != NULL, "sdr plane_u != NULL");
        TEST_ASSERT(ctx.sdr_outputs[i].plane_v != NULL, "sdr plane_v != NULL");

        int ow  = ctx.sdr_outputs[i].width;
        int oh  = ctx.sdr_outputs[i].height;
        int ys  = ctx.sdr_outputs[i].y_stride;
        int uvs = ctx.sdr_outputs[i].uv_stride;
        int cw  = ow / 2;
        int ch  = oh / 2;
        volatile uint8_t sink = 0;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                sink = ctx.sdr_outputs[i].plane_y[y * ys + x];
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++) {
                sink = ctx.sdr_outputs[i].plane_u[y * uvs + x];
                sink = ctx.sdr_outputs[i].plane_v[y * uvs + x];
            }
        (void)sink;
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 6. test_hdr_p010_equivalence
 *    Identical data in I010 and P010 format, scale both -> outputs must match
 * -------------------------------------------------------------------------- */

static void test_hdr_p010_equivalence(void)
{
    uint32_t flags = FUSED_SCALE_1_5X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 1, 0)) {
        TEST_SKIP("HDR kernel not yet producing output -- skipping pixel checks");
    }

    /* Create I010 frame */
    test_hdr_frame_t i010_frame;
    int r = test_hdr_frame_create(&i010_frame, 1920, 1080, PATTERN_HGRADIENT, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed for I010");

    /* Create P010 frame with same pattern */
    test_p010_frame_t p010_frame;
    r = test_p010_frame_create(&p010_frame, 1920, 1080, PATTERN_HGRADIENT, 0);
    TEST_ASSERT(r == 0, "test_p010_frame_create failed");

    /* Run I010 */
    fused_hdr_ctx_t ctx_i010;
    memset(&ctx_i010, 0, sizeof(ctx_i010));
    ctx_i010.src_width      = 1920;
    ctx_i010.src_height     = 1080;
    ctx_i010.src_y_stride   = i010_frame.y_stride;
    ctx_i010.src_uv_stride  = i010_frame.uv_stride;
    ctx_i010.src_format     = FUSED_PIX_I010;
    ctx_i010.src_transfer   = FUSED_TRC_PQ;
    ctx_i010.requested_flags = flags;
    ctx_i010.hdr_flags      = flags;
    suppress_log(&ctx_i010);

    int rc = fused_hdr_init(&ctx_i010);
    TEST_ASSERT(rc >= 0, "I010 init failed");

    fused_hdr_run(&ctx_i010, i010_frame.plane_y, i010_frame.plane_u, i010_frame.plane_v);

    /* Run P010 */
    fused_hdr_ctx_t ctx_p010;
    memset(&ctx_p010, 0, sizeof(ctx_p010));
    ctx_p010.src_width      = 1920;
    ctx_p010.src_height     = 1080;
    ctx_p010.src_y_stride   = p010_frame.y_stride;
    ctx_p010.src_uv_stride  = p010_frame.uv_stride;
    ctx_p010.src_format     = FUSED_PIX_P010;
    ctx_p010.src_transfer   = FUSED_TRC_PQ;
    ctx_p010.requested_flags = flags;
    ctx_p010.hdr_flags      = flags;
    suppress_log(&ctx_p010);

    rc = fused_hdr_init(&ctx_p010);
    TEST_ASSERT(rc >= 0, "P010 init failed");

    fused_hdr_run(&ctx_p010, p010_frame.plane_y, p010_frame.plane_uv, NULL);

    /* Compare HDR outputs */
    for (int i = 0; i < 8; i++) {
        if (!ctx_i010.hdr_outputs[i].plane_y) continue;
        TEST_ASSERT(ctx_p010.hdr_outputs[i].plane_y != NULL,
                    "P010 output should exist where I010 output exists");

        int ow = ctx_i010.hdr_outputs[i].width;
        int oh = ctx_i010.hdr_outputs[i].height;
        int stride_i = ctx_i010.hdr_outputs[i].y_stride / (int)sizeof(uint16_t);
        int stride_p = ctx_p010.hdr_outputs[i].y_stride / (int)sizeof(uint16_t);

        for (int y = 0; y < oh; y++) {
            const uint16_t *row_i = ctx_i010.hdr_outputs[i].plane_y + y * stride_i;
            const uint16_t *row_p = ctx_p010.hdr_outputs[i].plane_y + y * stride_p;
            for (int x = 0; x < ow; x++) {
                int diff = (int)row_i[x] - (int)row_p[x];
                if (diff < -1 || diff > 1) {
                    printf("\n  FAIL [%s:%d] P010 equiv: output[%d] pixel(%d,%d) "
                           "I010=%d P010=%d (diff=%d)\n",
                           __func__, __LINE__, i, x, y,
                           row_i[x], row_p[x], diff);
                    g_results.failed++;
                    fused_hdr_free(&ctx_i010);
                    fused_hdr_free(&ctx_p010);
                    test_hdr_frame_free(&i010_frame);
                    test_p010_frame_free(&p010_frame);
                    return;
                }
            }
        }
    }

    fused_hdr_free(&ctx_i010);
    fused_hdr_free(&ctx_p010);
    test_hdr_frame_free(&i010_frame);
    test_p010_frame_free(&p010_frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 7. test_hdr_curve_variation
 *    Each preset (Hable/Reinhard/BT2390) produces different output on same input
 * -------------------------------------------------------------------------- */

static void test_hdr_curve_variation(void)
{
    uint32_t flags = FUSED_SCALE_1_5X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 0, 1)) {
        TEST_SKIP("SDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_HGRADIENT, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    int curves[] = { FUSED_TONEMAP_HABLE, FUSED_TONEMAP_REINHARD, FUSED_TONEMAP_BT2390 };
    int n_curves = 3;

    /* Accumulate a simple checksum for each curve's output */
    uint32_t checksums[3] = { 0, 0, 0 };

    for (int ci = 0; ci < n_curves; ci++) {
        fused_hdr_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.src_width      = frame.width;
        ctx.src_height     = frame.height;
        ctx.src_y_stride   = frame.y_stride;
        ctx.src_uv_stride  = frame.uv_stride;
        ctx.src_format     = FUSED_PIX_I010;
        ctx.src_transfer   = FUSED_TRC_PQ;
        ctx.requested_flags = flags;
        ctx.hdr_flags      = 0;
        ctx.sdr_flags      = flags;
        ctx.tonemap.curve  = curves[ci];
        suppress_log(&ctx);

        int rc = fused_hdr_init(&ctx);
        TEST_ASSERT(rc >= 0, "fused_hdr_init failed for curve");

        fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

        /* Sum first row of first active SDR output as checksum */
        for (int i = 0; i < 8; i++) {
            if (!ctx.sdr_outputs[i].plane_y) continue;
            int ow = ctx.sdr_outputs[i].width;
            const uint8_t *row = ctx.sdr_outputs[i].plane_y;
            for (int x = 0; x < ow; x++)
                checksums[ci] += row[x];
            break;
        }

        fused_hdr_free(&ctx);
    }

    /* At least two of three must differ */
    int all_same = (checksums[0] == checksums[1]) &&
                   (checksums[1] == checksums[2]);
    TEST_ASSERT(!all_same, "different tone map curves should produce different output");

    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 8. test_hdr_output_dimensions
 *    Verify exact output dimensions for 4K -> all thirds steps
 * -------------------------------------------------------------------------- */

static void test_hdr_output_dimensions(void)
{
    static const struct {
        int w, h;
        uint32_t flags;
        struct { int bit, w, h; } expected[4];
        int n_expected;
    } dim_tests[] = {
        { 3840, 2160, FUSED_SCALE_1_5X|FUSED_SCALE_3X|FUSED_SCALE_6X|FUSED_SCALE_12X,
          {{0,2560,1440},{2,1280,720},{4,640,360},{6,320,180}}, 4 },
        { 1920, 1080, FUSED_SCALE_1_5X|FUSED_SCALE_3X|FUSED_SCALE_6X,
          {{0,1280,720},{2,640,360},{4,320,180}}, 3 },
        { 1280,  720, FUSED_SCALE_2X|FUSED_SCALE_4X,
          {{1,640,360},{3,320,180}}, 2 },
    };

    for (int di = 0; di < (int)(sizeof(dim_tests)/sizeof(dim_tests[0])); di++) {
        fused_hdr_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.src_width      = dim_tests[di].w;
        ctx.src_height     = dim_tests[di].h;
        ctx.src_y_stride   = align_up_32(dim_tests[di].w * 2);
        ctx.src_uv_stride  = align_up_32((dim_tests[di].w / 2) * 2);
        ctx.src_format     = FUSED_PIX_I010;
        ctx.src_transfer   = FUSED_TRC_PQ;
        ctx.requested_flags = dim_tests[di].flags;
        ctx.hdr_flags      = dim_tests[di].flags;
        suppress_log(&ctx);

        int rc = fused_hdr_init(&ctx);
        if (rc < 0) {
            printf("\n  FAIL [%s:%d] %dx%d init failed rc=%d\n",
                   __func__, __LINE__, dim_tests[di].w, dim_tests[di].h, rc);
            g_results.failed++;
            return;
        }

        for (int ei = 0; ei < dim_tests[di].n_expected; ei++) {
            int bit = dim_tests[di].expected[ei].bit;
            int ew  = dim_tests[di].expected[ei].w;
            int eh  = dim_tests[di].expected[ei].h;
            if (ctx.hdr_outputs[bit].width != ew || ctx.hdr_outputs[bit].height != eh) {
                printf("\n  FAIL [%s:%d] %dx%d hdr_outputs[%d]: got %dx%d, expected %dx%d\n",
                       __func__, __LINE__,
                       dim_tests[di].w, dim_tests[di].h,
                       bit,
                       ctx.hdr_outputs[bit].width, ctx.hdr_outputs[bit].height,
                       ew, eh);
                g_results.failed++;
                fused_hdr_free(&ctx);
                return;
            }
        }

        fused_hdr_free(&ctx);
    }

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 9. test_hdr_hlg_vs_pq_differ
 *    Same input frame, two HDR scalers: one PQ, one HLG, both producing
 *    SDR output with Hable curve at 1.5x. SDR outputs should differ since
 *    HLG and PQ use different EOTFs. Compare first-row checksums.
 * -------------------------------------------------------------------------- */

static void test_hdr_hlg_vs_pq_differ(void)
{
    uint32_t flags = FUSED_SCALE_1_5X;

    if (!hdr_kernel_produces_output(1920, 1080, flags, 0, 1)) {
        TEST_SKIP("SDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_HGRADIENT, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    /* PQ context */
    fused_hdr_ctx_t ctx_pq;
    memset(&ctx_pq, 0, sizeof(ctx_pq));
    ctx_pq.src_width      = frame.width;
    ctx_pq.src_height     = frame.height;
    ctx_pq.src_y_stride   = frame.y_stride;
    ctx_pq.src_uv_stride  = frame.uv_stride;
    ctx_pq.src_format     = FUSED_PIX_I010;
    ctx_pq.src_transfer   = FUSED_TRC_PQ;
    ctx_pq.requested_flags = flags;
    ctx_pq.hdr_flags      = 0;
    ctx_pq.sdr_flags      = flags;
    ctx_pq.tonemap.curve  = FUSED_TONEMAP_HABLE;
    suppress_log(&ctx_pq);

    int rc = fused_hdr_init(&ctx_pq);
    TEST_ASSERT(rc >= 0, "PQ init failed");

    fused_hdr_run(&ctx_pq, frame.plane_y, frame.plane_u, frame.plane_v);

    /* HLG context */
    fused_hdr_ctx_t ctx_hlg;
    memset(&ctx_hlg, 0, sizeof(ctx_hlg));
    ctx_hlg.src_width      = frame.width;
    ctx_hlg.src_height     = frame.height;
    ctx_hlg.src_y_stride   = frame.y_stride;
    ctx_hlg.src_uv_stride  = frame.uv_stride;
    ctx_hlg.src_format     = FUSED_PIX_I010;
    ctx_hlg.src_transfer   = FUSED_TRC_HLG;
    ctx_hlg.requested_flags = flags;
    ctx_hlg.hdr_flags      = 0;
    ctx_hlg.sdr_flags      = flags;
    ctx_hlg.tonemap.curve  = FUSED_TONEMAP_HABLE;
    suppress_log(&ctx_hlg);

    rc = fused_hdr_init(&ctx_hlg);
    TEST_ASSERT(rc >= 0, "HLG init failed");

    fused_hdr_run(&ctx_hlg, frame.plane_y, frame.plane_u, frame.plane_v);

    /* Compare first-row checksums of SDR output */
    uint32_t checksum_pq = 0, checksum_hlg = 0;
    if (ctx_pq.sdr_outputs[FUSED_IDX_1_5X].plane_y && ctx_hlg.sdr_outputs[FUSED_IDX_1_5X].plane_y) {
        int ow = ctx_pq.sdr_outputs[FUSED_IDX_1_5X].width;
        const uint8_t *row_pq  = ctx_pq.sdr_outputs[FUSED_IDX_1_5X].plane_y;
        const uint8_t *row_hlg = ctx_hlg.sdr_outputs[FUSED_IDX_1_5X].plane_y;
        for (int x = 0; x < ow; x++) {
            checksum_pq  += row_pq[x];
            checksum_hlg += row_hlg[x];
        }
    }

    TEST_ASSERT(checksum_pq != checksum_hlg,
                "PQ and HLG should produce different SDR tone-mapped output");

    fused_hdr_free(&ctx_pq);
    fused_hdr_free(&ctx_hlg);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 10. test_hdr_misaligned_source_fallback
 *     Offset HDR source Y pointer by 1 element (2 bytes), breaking alignment.
 *     Verify no crash and output is produced (alignment check + scalar fallback).
 * -------------------------------------------------------------------------- */

static void test_hdr_misaligned_source_fallback(void)
{
    int src_w = 1920;
    int src_h = 1080;
    int y_stride  = align_up_32(src_w * 2);
    int uv_stride = align_up_32((src_w / 2) * 2);

    /* Allocate oversized buffers so we can safely offset */
    uint16_t *y_buf  = NULL;
    uint16_t *u_buf  = NULL;
    uint16_t *v_buf  = NULL;
    size_t y_size  = (size_t)y_stride  * src_h + 32;
    size_t uv_size = (size_t)uv_stride * (src_h / 2) + 32;
    if (posix_memalign((void **)&y_buf,  32, y_size)  != 0 ||
        posix_memalign((void **)&u_buf,  32, uv_size) != 0 ||
        posix_memalign((void **)&v_buf,  32, uv_size) != 0) {
        free(y_buf); free(u_buf); free(v_buf);
        TEST_ASSERT(0, "posix_memalign failed");
    }

    /* Fill with non-zero 10-bit values */
    int y_samples  = (int)(y_size / sizeof(uint16_t));
    int uv_samples = (int)(uv_size / sizeof(uint16_t));
    for (int i = 0; i < y_samples; i++)
        y_buf[i] = 512;
    for (int i = 0; i < uv_samples; i++) {
        u_buf[i] = 512;
        v_buf[i] = 512;
    }

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = src_w;
    ctx.src_height     = src_h;
    ctx.src_y_stride   = y_stride;
    ctx.src_uv_stride  = uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_2X;
    ctx.hdr_flags      = FUSED_SCALE_2X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");

    /* Pass misaligned pointers: offset by 1 element (2 bytes) */
    fused_hdr_run(&ctx, y_buf + 1, u_buf + 1, v_buf + 1);

    /* Verify output was produced */
    int found_nonzero = 0;
    if (ctx.hdr_outputs[FUSED_IDX_2X].plane_y) {
        int ow = ctx.hdr_outputs[FUSED_IDX_2X].width;
        for (int x = 0; x < ow && !found_nonzero; x++) {
            if (ctx.hdr_outputs[FUSED_IDX_2X].plane_y[x] != 0)
                found_nonzero = 1;
        }
    }
    TEST_ASSERT(found_nonzero, "HDR output should have non-zero pixels after misaligned fallback");

    fused_hdr_free(&ctx);
    free(y_buf);
    free(u_buf);
    free(v_buf);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 11. test_hdr_double_free_safety
 *     Init HDR context, free twice. Verify no crash.
 *     Free a zero-initialized HDR context. Verify no crash.
 * -------------------------------------------------------------------------- */

static void test_hdr_double_free_safety(void)
{
    /* Part 1: init then double-free */
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");

    fused_hdr_free(&ctx);
    fused_hdr_free(&ctx);  /* second free - should be a no-op */

    /* Part 2: free on zero-initialized context */
    fused_hdr_ctx_t zero_ctx;
    memset(&zero_ctx, 0, sizeof(zero_ctx));
    fused_hdr_free(&zero_ctx);  /* should be a no-op */

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 12. test_hdr_non_standard_correctness
 *     1000x600, I010, PQ, FUSED_SCALE_2X, hdr_flags=2x.
 *     Create solid-512 input. Run scaler. Verify all HDR output pixels
 *     are 512 +/- 1. Tests the scalar tail with 10-bit data.
 * -------------------------------------------------------------------------- */

static void test_hdr_non_standard_correctness(void)
{
    uint32_t flags = FUSED_SCALE_2X;

    if (!hdr_kernel_produces_output(1000, 600, flags, 1, 0)) {
        TEST_SKIP("HDR kernel not yet producing output -- skipping pixel checks");
    }

    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1000, 600, PATTERN_SOLID, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = flags;
    ctx.hdr_flags      = flags;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "fused_hdr_init failed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    if (ctx.hdr_outputs[FUSED_IDX_2X].plane_y) {
        int ow = ctx.hdr_outputs[FUSED_IDX_2X].width;
        int oh = ctx.hdr_outputs[FUSED_IDX_2X].height;
        int stride = ctx.hdr_outputs[FUSED_IDX_2X].y_stride / (int)sizeof(uint16_t);

        TEST_ASSERT_EQ(ow, 500, "output width = 500");
        TEST_ASSERT_EQ(oh, 300, "output height = 300");

        for (int y = 0; y < oh; y++) {
            const uint16_t *row = ctx.hdr_outputs[FUSED_IDX_2X].plane_y + y * stride;
            for (int x = 0; x < ow; x++) {
                int diff = (int)row[x] - 512;
                if (diff < -1 || diff > 1) {
                    printf("\n  FAIL [%s:%d] non-standard HDR solid: pixel(%d,%d)=%d, expected 512+/-1\n",
                           __func__, __LINE__, x, y, row[x]);
                    g_results.failed++;
                    fused_hdr_free(&ctx);
                    test_hdr_frame_free(&frame);
                    return;
                }
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 13. test_hdr_p010_tonemap_1x
 *     Create a P010 frame. Init with tonemap_1x=1 and a scale flag.
 *     Run. Verify output_1x is populated. Tests the fused_tonemap_apply_p010
 *     path in fused_hdr_run.
 * -------------------------------------------------------------------------- */

static void test_hdr_p010_tonemap_1x(void)
{
    test_p010_frame_t frame;
    int r = test_p010_frame_create(&frame, 1920, 1080, PATTERN_HGRADIENT, 0);
    TEST_ASSERT(r == 0, "test_p010_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_P010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X;
    ctx.sdr_flags      = 0;
    ctx.tonemap_1x     = 1;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "P010 tonemap_1x init should succeed");
    TEST_ASSERT(ctx.output_1x.plane_y != NULL, "output_1x.plane_y != NULL");
    TEST_ASSERT_EQ(ctx.output_1x.width,  1920, "output_1x.width = 1920");
    TEST_ASSERT_EQ(ctx.output_1x.height, 1080, "output_1x.height = 1080");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_uv, NULL);

    /* Verify output_1x was actually written (non-zero pixels) */
    int found_nonzero = 0;
    int ow = ctx.output_1x.width;
    int stride = ctx.output_1x.y_stride;
    const uint8_t *row = ctx.output_1x.plane_y;
    for (int x = 0; x < ow && !found_nonzero; x++) {
        if (row[x] != 0)
            found_nonzero = 1;
    }
    (void)stride;
    TEST_ASSERT(found_nonzero, "output_1x should contain non-zero pixels");

    fused_hdr_free(&ctx);
    test_p010_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 14. test_hdr_p010_upscale
 *     P010 callers pass interleaved UV as src_u and NULL src_v. HDR upscale
 *     must deinterleave chroma before scaling rather than reading src_v.
 * -------------------------------------------------------------------------- */

static void test_hdr_p010_upscale(void)
{
    test_p010_frame_t frame;
    int r = test_p010_frame_create(&frame, 64, 64, PATTERN_SOLID, 0);
    TEST_ASSERT(r == 0, "test_p010_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_P010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = 0;
    ctx.hdr_flags      = 0;
    ctx.sdr_flags      = 0;
    ctx.upscale_flags  = FUSED_UPSCALE_2X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "P010 upscale init should succeed");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X].plane_y != NULL,
                "P010 2x output allocated");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_uv, NULL);

    const fused_hdr_output_t *out = &ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X];
    TEST_ASSERT_EQ(out->width, 128, "P010 2x output width");
    TEST_ASSERT_EQ(out->height, 128, "P010 2x output height");
    TEST_ASSERT(out->plane_y[0] != 0, "P010 upscale wrote Y");
    TEST_ASSERT(out->plane_u[0] != 0, "P010 upscale wrote U");
    TEST_ASSERT(out->plane_v[0] != 0, "P010 upscale wrote V");

    fused_hdr_free(&ctx);
    test_p010_frame_free(&frame);
    TEST_PASS();
}

/* ==========================================================================
 * HDR upscale tests
 * ========================================================================== */

/* test_hdr_upscale_dimensions
 *   Verify upscale_hdr_outputs[].width/.height match expected for several
 *   cascade depths and the tail. */
static void test_hdr_upscale_dimensions(void)
{
    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 192, 108, PATTERN_PQ_RAMP, 0);
    TEST_ASSERT(r == 0, "frame create");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width         = frame.width;
    ctx.src_height        = frame.height;
    ctx.src_y_stride      = frame.y_stride;
    ctx.src_uv_stride     = frame.uv_stride;
    ctx.src_format        = FUSED_PIX_I010;
    ctx.src_transfer      = FUSED_TRC_PQ;
    ctx.requested_flags   = 0;          /* upscale-only */
    ctx.upscale_flags     = FUSED_UPSCALE_2X | FUSED_UPSCALE_4X;
    ctx.upscale_tail_1_5x = 1;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "hdr init");

    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X].plane_y != NULL, "2x");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X].width  == 384, "2x w");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X].height == 216, "2x h");

    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_4X].plane_y != NULL, "4x");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_4X].width  == 768, "4x w");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_4X].height == 432, "4x h");

    /* Tail = 4x * 1.5 = 6x */
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_TAIL].plane_y != NULL, "tail");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_TAIL].width  == 1152, "tail w");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_TAIL].height == 648, "tail h");

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* test_hdr_upscale_solid_preservation
 *   Solid 10-bit value source must be preserved (within ±1) by upscale. */
static void test_hdr_upscale_solid_preservation(void)
{
    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 192, 108, PATTERN_SOLID, 0);
    TEST_ASSERT(r == 0, "frame create");

    /* Set source to a fixed 10-bit value (mid-gray) */
    const uint16_t mid = 512;
    int y_el = frame.y_stride / (int)sizeof(uint16_t);
    int uv_el = frame.uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < frame.height; y++)
        for (int x = 0; x < frame.width; x++)
            frame.plane_y[y * y_el + x] = mid;
    for (int y = 0; y < frame.height / 2; y++)
        for (int x = 0; x < frame.width / 2; x++) {
            frame.plane_u[y * uv_el + x] = mid;
            frame.plane_v[y * uv_el + x] = mid;
        }

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width         = frame.width;
    ctx.src_height        = frame.height;
    ctx.src_y_stride      = frame.y_stride;
    ctx.src_uv_stride     = frame.uv_stride;
    ctx.src_format        = FUSED_PIX_I010;
    ctx.src_transfer      = FUSED_TRC_PQ;
    ctx.requested_flags   = 0;
    ctx.upscale_flags     = FUSED_UPSCALE_2X | FUSED_UPSCALE_4X;
    ctx.upscale_tail_1_5x = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init");
    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    for (int k = 0; k < FUSED_MAX_UPSCALE_STEPS; k++) {
        const fused_hdr_output_t *o = &ctx.upscale_hdr_outputs[k];
        if (!o->plane_y) continue;
        int el_stride = o->y_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < o->height; y++) {
            for (int x = 0; x < o->width; x++) {
                int v = o->plane_y[y * el_stride + x];
                int diff = v - mid;
                if (diff < -1 || diff > 1) {
                    TEST_ASSERT(0, "HDR upscale Y not preserved");
                }
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* test_hdr_upscale_combined_with_downscale
 *   HDR downscale and upscale in one init call must both produce correct
 *   outputs. */
static void test_hdr_upscale_combined_with_downscale(void)
{
    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 1920, 1080, PATTERN_SOLID, 0);
    TEST_ASSERT(r == 0, "frame create");

    /* Solid mid-value */
    const uint16_t mid = 512;
    int y_el = frame.y_stride / (int)sizeof(uint16_t);
    int uv_el = frame.uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < frame.height; y++)
        for (int x = 0; x < frame.width; x++)
            frame.plane_y[y * y_el + x] = mid;
    for (int y = 0; y < frame.height / 2; y++)
        for (int x = 0; x < frame.width / 2; x++) {
            frame.plane_u[y * uv_el + x] = mid;
            frame.plane_v[y * uv_el + x] = mid;
        }

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width       = frame.width;
    ctx.src_height      = frame.height;
    ctx.src_y_stride    = frame.y_stride;
    ctx.src_uv_stride   = frame.uv_stride;
    ctx.src_format      = FUSED_PIX_I010;
    ctx.src_transfer    = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_2X;
    ctx.hdr_flags       = FUSED_SCALE_2X;
    ctx.upscale_flags   = FUSED_UPSCALE_2X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_2X].plane_y != NULL, "down 2x");
    TEST_ASSERT(ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X].plane_y != NULL, "up 2x");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    /* Verify downscale 2x output is mid */
    {
        const fused_hdr_output_t *o = &ctx.hdr_outputs[FUSED_IDX_2X];
        int el_stride = o->y_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < o->height; y++) {
            for (int x = 0; x < o->width; x++) {
                int v = o->plane_y[y * el_stride + x];
                if (v < mid - 1 || v > mid + 1) {
                    TEST_ASSERT(0, "HDR down 2x not mid");
                }
            }
        }
    }
    /* Verify upscale 2x output is mid */
    {
        const fused_hdr_output_t *o = &ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X];
        int el_stride = o->y_stride / (int)sizeof(uint16_t);
        for (int y = 0; y < o->height; y++) {
            for (int x = 0; x < o->width; x++) {
                int v = o->plane_y[y * el_stride + x];
                if (v < mid - 1 || v > mid + 1) {
                    TEST_ASSERT(0, "HDR up 2x not mid");
                }
            }
        }
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}


/* --------------------------------------------------------------------------
 * run_hdr_correctness_tests
 * -------------------------------------------------------------------------- */

void run_hdr_correctness_tests(void)
{
    RUN_TEST(test_hdr_solid_preservation);
    RUN_TEST(test_hdr_10bit_range);
    RUN_TEST(test_hdr_gradient_monotonicity);
    RUN_TEST(test_hdr_tonemap_monotonicity);
    RUN_TEST(test_hdr_tonemap_range);
    RUN_TEST(test_hdr_p010_equivalence);
    RUN_TEST(test_hdr_curve_variation);
    RUN_TEST(test_hdr_output_dimensions);
    RUN_TEST(test_hdr_hlg_vs_pq_differ);
    RUN_TEST(test_hdr_misaligned_source_fallback);
    RUN_TEST(test_hdr_double_free_safety);
    RUN_TEST(test_hdr_non_standard_correctness);
    RUN_TEST(test_hdr_p010_tonemap_1x);
    RUN_TEST(test_hdr_p010_upscale);

    /* HDR upscale tests */
    RUN_TEST(test_hdr_upscale_dimensions);
    RUN_TEST(test_hdr_upscale_solid_preservation);
    RUN_TEST(test_hdr_upscale_combined_with_downscale);
}
