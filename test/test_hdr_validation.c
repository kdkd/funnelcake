/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include "detect.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
 * Local helpers
 * -------------------------------------------------------------------------- */

static int align_up_32(int x)
{
    return (x + 31) & ~31;
}

static void suppress_log(fused_hdr_ctx_t *ctx)
{
    ctx->log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx->log_errors.target   = FUSED_LOG_SUPPRESS;
}

/* --------------------------------------------------------------------------
 * 1. test_hdr_valid_i010_thirds
 *    1920x1080, I010, PQ, thirds (1.5x|3x|6x), hdr_flags=all -> FUSED_OK
 * -------------------------------------------------------------------------- */

static void test_hdr_valid_i010_thirds(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "rc should be FUSED_OK");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0].plane_y != NULL");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_1_5X].width,  1280, "hdr_outputs[0].width");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_1_5X].height, 720,  "hdr_outputs[0].height");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_3X].width, 640, "hdr_outputs[2].width");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_6X].width, 320, "hdr_outputs[4].width");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 2. test_hdr_valid_i010_pow2
 *    1280x720, I010, PQ, pow2 (2x|4x), hdr_flags=all -> FUSED_OK
 * -------------------------------------------------------------------------- */

static void test_hdr_valid_i010_pow2(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1280;
    ctx.src_height     = 720;
    ctx.src_y_stride   = align_up_32(1280 * 2);
    ctx.src_uv_stride  = align_up_32(640 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_2X | FUSED_SCALE_4X;
    ctx.hdr_flags      = FUSED_SCALE_2X | FUSED_SCALE_4X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "rc should be FUSED_OK");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_2X].width, 640, "hdr_outputs[1].width");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_4X].width, 320, "hdr_outputs[3].width");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 3. test_hdr_valid_p010
 *    1920x1080, P010, PQ, thirds, hdr_flags=all -> FUSED_OK
 * -------------------------------------------------------------------------- */

static void test_hdr_valid_p010(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2 * 2); /* interleaved UV: cw * 2 * sizeof(uint16_t) */
    ctx.src_format     = FUSED_PIX_P010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "rc should be FUSED_OK");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0].plane_y != NULL");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 4. test_hdr_valid_i210
 *    1920x1080, I210, PQ, thirds, hdr_flags=all -> FUSED_OK (4:2:2 accepted)
 * -------------------------------------------------------------------------- */

static void test_hdr_valid_i210(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I210;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "rc should be FUSED_OK");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0].plane_y != NULL");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 5. test_hdr_valid_p210
 *    1920x1080, P210, HLG, thirds, hdr_flags=all -> FUSED_OK
 * -------------------------------------------------------------------------- */

static void test_hdr_valid_p210(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2 * 2); /* P210: interleaved UV, full height chroma */
    ctx.src_format     = FUSED_PIX_P210;
    ctx.src_transfer   = FUSED_TRC_HLG;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "rc should be FUSED_OK");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0].plane_y != NULL");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 6. test_hdr_mixed_outputs
 *    hdr_flags=1.5x, sdr_flags=3x|6x -> both output sets populated
 * -------------------------------------------------------------------------- */

static void test_hdr_mixed_outputs(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X;
    ctx.sdr_flags      = FUSED_SCALE_3X | FUSED_SCALE_6X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0] populated");
    TEST_ASSERT(ctx.sdr_outputs[FUSED_IDX_3X].plane_y != NULL, "sdr_outputs[2] populated");
    TEST_ASSERT(ctx.sdr_outputs[FUSED_IDX_6X].plane_y != NULL, "sdr_outputs[4] populated");
    /* HDR outputs not requested for 3x/6x */
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_3X].plane_y == NULL, "hdr_outputs[2] NULL (not requested)");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_6X].plane_y == NULL, "hdr_outputs[4] NULL (not requested)");
    /* SDR output not requested for 1.5x */
    TEST_ASSERT(ctx.sdr_outputs[FUSED_IDX_1_5X].plane_y == NULL, "sdr_outputs[0] NULL (not requested)");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 7. test_hdr_sdr_only
 *    hdr_flags=0, sdr_flags=1.5x|3x -> SDR outputs populated, HDR NULL
 * -------------------------------------------------------------------------- */

static void test_hdr_sdr_only(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X;
    ctx.hdr_flags      = 0;
    ctx.sdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");
    TEST_ASSERT(ctx.sdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "sdr_outputs[0] populated");
    TEST_ASSERT(ctx.sdr_outputs[FUSED_IDX_3X].plane_y != NULL, "sdr_outputs[2] populated");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y == NULL, "hdr_outputs[0] NULL");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_3X].plane_y == NULL, "hdr_outputs[2] NULL");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 8. test_hdr_hdr_only
 *    sdr_flags=0, hdr_flags=1.5x|3x -> HDR outputs populated, SDR NULL
 * -------------------------------------------------------------------------- */

static void test_hdr_hdr_only(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0] populated");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_3X].plane_y != NULL, "hdr_outputs[2] populated");
    TEST_ASSERT(ctx.sdr_outputs[FUSED_IDX_1_5X].plane_y == NULL, "sdr_outputs[0] NULL");
    TEST_ASSERT(ctx.sdr_outputs[FUSED_IDX_3X].plane_y == NULL, "sdr_outputs[2] NULL");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 9. test_hdr_tonemap_1x
 *    tonemap_1x=1 -> output_1x populated at source resolution
 * -------------------------------------------------------------------------- */

static void test_hdr_tonemap_1x(void)
{
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
    ctx.tonemap_1x     = 1;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");
    TEST_ASSERT(ctx.output_1x.plane_y != NULL, "output_1x.plane_y != NULL");
    TEST_ASSERT_EQ(ctx.output_1x.width,  1920, "output_1x.width = 1920");
    TEST_ASSERT_EQ(ctx.output_1x.height, 1080, "output_1x.height = 1080");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 9b. test_hdr_upscale_sdr_flags
 *     upscale_sdr_flags must be a subset of upscale_flags, and the SDR
 *     tail copy requires the HDR tail; valid requests allocate 8-bit
 *     outputs at the upscaled dimensions.
 * -------------------------------------------------------------------------- */

static void test_hdr_upscale_sdr_flags(void)
{
    fused_hdr_ctx_t ctx;

    /* (a) SDR level without the matching 10-bit level */
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.upscale_flags     = FUSED_UPSCALE_2X;
    ctx.upscale_sdr_flags = FUSED_UPSCALE_2X | FUSED_UPSCALE_4X;
    suppress_log(&ctx);
    TEST_ASSERT_EQ(fused_hdr_init(&ctx), FUSED_ERR_INVALID_FLAGS,
                   "upscale_sdr_flags not subset -> FUSED_ERR_INVALID_FLAGS");

    /* (b) SDR tail without the HDR tail */
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.upscale_flags         = FUSED_UPSCALE_2X;
    ctx.upscale_sdr_tail_1_5x = 1;
    suppress_log(&ctx);
    TEST_ASSERT_EQ(fused_hdr_init(&ctx), FUSED_ERR_INVALID_FLAGS,
                   "SDR tail without HDR tail -> FUSED_ERR_INVALID_FLAGS");

    /* (c) valid: SDR copy of the 2x level at upscaled dimensions */
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.upscale_flags     = FUSED_UPSCALE_2X;
    ctx.upscale_sdr_flags = FUSED_UPSCALE_2X;
    suppress_log(&ctx);
    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");
    TEST_ASSERT_EQ((int)ctx.achieved_upscale_sdr_flags, (int)FUSED_UPSCALE_2X,
                   "achieved_upscale_sdr_flags = 2x");
    TEST_ASSERT(ctx.upscale_sdr_outputs[0].plane_y != NULL,
                "upscale_sdr_outputs[0].plane_y != NULL");
    TEST_ASSERT_EQ(ctx.upscale_sdr_outputs[0].width,  3840,
                   "SDR upscale width = 3840");
    TEST_ASSERT_EQ(ctx.upscale_sdr_outputs[0].height, 2160,
                   "SDR upscale height = 2160");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 10. test_hdr_flags_not_subset
 *     sdr_flags has bits not in requested_flags -> FUSED_ERR_INVALID_FLAGS
 * -------------------------------------------------------------------------- */

static void test_hdr_flags_not_subset(void)
{
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
    ctx.sdr_flags      = FUSED_SCALE_3X;  /* not in requested_flags */
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_ERR_INVALID_FLAGS, "sdr_flags not subset -> FUSED_ERR_INVALID_FLAGS");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 11. test_hdr_invalid_format
 *     src_format=99 -> error
 * -------------------------------------------------------------------------- */

static void test_hdr_invalid_format(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = 99;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc < 0, "invalid format -> negative error");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 12. test_hdr_invalid_transfer
 *     src_transfer=99 -> error
 * -------------------------------------------------------------------------- */

static void test_hdr_invalid_transfer(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = 99;
    ctx.requested_flags = FUSED_SCALE_1_5X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc < 0, "invalid transfer -> negative error");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 13. test_hdr_custom_null_lut
 *     FUSED_TONEMAP_CUSTOM with custom_lut=NULL -> error
 * -------------------------------------------------------------------------- */

static void test_hdr_custom_null_lut(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X;
    ctx.hdr_flags      = 0;
    ctx.sdr_flags      = FUSED_SCALE_1_5X;
    ctx.tonemap.curve  = FUSED_TONEMAP_CUSTOM;
    ctx.tonemap.custom_lut = NULL;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc < 0, "CUSTOM with NULL lut -> negative error");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 14. test_hdr_all_curves
 *     Each of HABLE, REINHARD, BT2390 -> init succeeds
 * -------------------------------------------------------------------------- */

static void test_hdr_all_curves(void)
{
    int curves[] = { FUSED_TONEMAP_HABLE, FUSED_TONEMAP_REINHARD, FUSED_TONEMAP_BT2390 };
    int n = (int)(sizeof(curves) / sizeof(curves[0]));

    for (int i = 0; i < n; i++) {
        fused_hdr_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.src_width      = 1920;
        ctx.src_height     = 1080;
        ctx.src_y_stride   = align_up_32(1920 * 2);
        ctx.src_uv_stride  = align_up_32(960 * 2);
        ctx.src_format     = FUSED_PIX_I010;
        ctx.src_transfer   = FUSED_TRC_PQ;
        ctx.requested_flags = FUSED_SCALE_1_5X;
        ctx.hdr_flags      = 0;
        ctx.sdr_flags      = FUSED_SCALE_1_5X;
        ctx.tonemap.curve  = curves[i];
        suppress_log(&ctx);

        int rc = fused_hdr_init(&ctx);
        if (rc < 0) {
            printf("\n  FAIL [%s:%d] curve %d: init returned %d\n",
                   __func__, __LINE__, curves[i], rc);
            g_results.failed++;
            return;
        }
        fused_hdr_free(&ctx);
    }

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 15. test_hdr_output_alignment
 *     All planes 32-byte aligned, strides %32 == 0
 * -------------------------------------------------------------------------- */

static void test_hdr_output_alignment(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");

    for (int i = 0; i < 8; i++) {
        if (ctx.hdr_outputs[i].plane_y) {
            TEST_ASSERT(((uintptr_t)ctx.hdr_outputs[i].plane_y % 32) == 0,
                        "hdr plane_y 32-byte aligned");
            TEST_ASSERT(((uintptr_t)ctx.hdr_outputs[i].plane_u % 32) == 0,
                        "hdr plane_u 32-byte aligned");
            TEST_ASSERT(((uintptr_t)ctx.hdr_outputs[i].plane_v % 32) == 0,
                        "hdr plane_v 32-byte aligned");
            TEST_ASSERT((ctx.hdr_outputs[i].y_stride  % 32) == 0,
                        "hdr y_stride multiple of 32");
            TEST_ASSERT((ctx.hdr_outputs[i].uv_stride % 32) == 0,
                        "hdr uv_stride multiple of 32");
        }
        if (ctx.sdr_outputs[i].plane_y) {
            TEST_ASSERT(((uintptr_t)ctx.sdr_outputs[i].plane_y % 32) == 0,
                        "sdr plane_y 32-byte aligned");
            TEST_ASSERT(((uintptr_t)ctx.sdr_outputs[i].plane_u % 32) == 0,
                        "sdr plane_u 32-byte aligned");
            TEST_ASSERT(((uintptr_t)ctx.sdr_outputs[i].plane_v % 32) == 0,
                        "sdr plane_v 32-byte aligned");
            TEST_ASSERT((ctx.sdr_outputs[i].y_stride  % 32) == 0,
                        "sdr y_stride multiple of 32");
            TEST_ASSERT((ctx.sdr_outputs[i].uv_stride % 32) == 0,
                        "sdr uv_stride multiple of 32");
        }
    }

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 16. test_hdr_free_cleans_up
 *     init then free -> all pointers become NULL
 * -------------------------------------------------------------------------- */

static void test_hdr_free_cleans_up(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = FUSED_SCALE_1_5X;
    ctx.tonemap_1x     = 1;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed before free test");

    fused_hdr_free(&ctx);

    for (int i = 0; i < 8; i++) {
        TEST_ASSERT(ctx.hdr_outputs[i].plane_y == NULL, "hdr plane_y NULL after free");
        TEST_ASSERT(ctx.hdr_outputs[i].plane_u == NULL, "hdr plane_u NULL after free");
        TEST_ASSERT(ctx.hdr_outputs[i].plane_v == NULL, "hdr plane_v NULL after free");
        TEST_ASSERT(ctx.sdr_outputs[i].plane_y == NULL, "sdr plane_y NULL after free");
        TEST_ASSERT(ctx.sdr_outputs[i].plane_u == NULL, "sdr plane_u NULL after free");
        TEST_ASSERT(ctx.sdr_outputs[i].plane_v == NULL, "sdr plane_v NULL after free");
    }
    TEST_ASSERT(ctx.output_1x.plane_y == NULL, "output_1x.plane_y NULL after free");
    TEST_ASSERT(ctx._internal == NULL, "_internal NULL after free");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 17. test_hdr_i010_hlg
 *     1920x1080, I010, HLG, thirds (1.5x|3x|6x), hdr_flags=all.
 *     Tests I010+HLG combination which was previously uncovered.
 * -------------------------------------------------------------------------- */

static void test_hdr_i010_hlg(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_HLG;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "I010+HLG thirds -> FUSED_OK");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0] allocated");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 18. test_hdr_p010_hlg
 *     1920x1080, P010, HLG, thirds. Tests P010+HLG combination.
 * -------------------------------------------------------------------------- */

static void test_hdr_p010_hlg(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2 * 2);
    ctx.src_format     = FUSED_PIX_P010;
    ctx.src_transfer   = FUSED_TRC_HLG;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT_OK(rc, "P010+HLG thirds -> FUSED_OK");
    TEST_ASSERT(ctx.hdr_outputs[FUSED_IDX_1_5X].plane_y != NULL, "hdr_outputs[0] allocated");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 19. test_hdr_16x_pow2
 *     1920x1088 (divisible by 32), I010, PQ, FUSED_SCALE_16X only,
 *     hdr_flags=16x. Output 120x68.
 * -------------------------------------------------------------------------- */

static void test_hdr_16x_pow2(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1088;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_16X;
    ctx.hdr_flags      = FUSED_SCALE_16X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "1920x1088 16x HDR init should succeed");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_16X].width,  120, "output width = 120");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_16X].height,  68, "output height = 68");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 20. test_hdr_12x_only
 *     1920x1080, I010, PQ, FUSED_SCALE_12X only, hdr_flags=12x.
 *     Tests 12x in isolation.
 * -------------------------------------------------------------------------- */

static void test_hdr_12x_only(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_12X;
    ctx.hdr_flags      = FUSED_SCALE_12X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "1920x1080 12x HDR init should succeed");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_12X].width,  160, "output width = 160");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_12X].height,  90, "output height = 90");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 21. test_hdr_tonemap_1x_only
 *     No requested_flags, no hdr_flags, no sdr_flags, just tonemap_1x=1.
 *     Verify init succeeds and output_1x is populated at source resolution.
 *     Tests the tonemap-only path without any scaling.
 * -------------------------------------------------------------------------- */

static void test_hdr_tonemap_1x_only(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1920;
    ctx.src_height     = 1080;
    ctx.src_y_stride   = align_up_32(1920 * 2);
    ctx.src_uv_stride  = align_up_32(960 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = 0;
    ctx.hdr_flags      = 0;
    ctx.sdr_flags      = 0;
    ctx.tonemap_1x     = 1;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "tonemap_1x only init should succeed");
    TEST_ASSERT(ctx.output_1x.plane_y != NULL, "output_1x.plane_y != NULL");
    TEST_ASSERT_EQ(ctx.output_1x.width,  1920, "output_1x.width = 1920");
    TEST_ASSERT_EQ(ctx.output_1x.height, 1080, "output_1x.height = 1080");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 22. test_hdr_custom_lut_output
 *     Create a custom LUT mapping every input to 42. Init with
 *     FUSED_TONEMAP_CUSTOM and that LUT. Run on solid-512 frame with
 *     sdr_flags. Verify all SDR luma output pixels are 42.
 * -------------------------------------------------------------------------- */

static void test_hdr_custom_lut_output(void)
{
    uint8_t custom_lut[1024];
    memset(custom_lut, 42, 1024);

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
    ctx.requested_flags = FUSED_SCALE_1_5X;
    ctx.hdr_flags      = 0;
    ctx.sdr_flags      = FUSED_SCALE_1_5X;
    ctx.tonemap.curve      = FUSED_TONEMAP_CUSTOM;
    ctx.tonemap.custom_lut = custom_lut;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "custom LUT init should succeed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    /* Verify all SDR luma output pixels are 42 */
    if (ctx.sdr_outputs[FUSED_IDX_1_5X].plane_y) {
        int ow = ctx.sdr_outputs[FUSED_IDX_1_5X].width;
        int oh = ctx.sdr_outputs[FUSED_IDX_1_5X].height;
        int stride = ctx.sdr_outputs[FUSED_IDX_1_5X].y_stride;
        for (int y = 0; y < oh; y++) {
            const uint8_t *row = ctx.sdr_outputs[FUSED_IDX_1_5X].plane_y + y * stride;
            for (int x = 0; x < ow; x++) {
                if (row[x] != 42) {
                    printf("\n  FAIL [%s:%d] custom LUT: pixel(%d,%d)=%d, expected 42\n",
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
 * 23. test_hdr_non_standard_width
 *     1000x600, I010, PQ, FUSED_SCALE_2X, hdr_flags=2x.
 *     Verify init succeeds and output is 500x300.
 * -------------------------------------------------------------------------- */

static void test_hdr_non_standard_width(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 1000;
    ctx.src_height     = 600;
    ctx.src_y_stride   = align_up_32(1000 * 2);
    ctx.src_uv_stride  = align_up_32(500 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_2X;
    ctx.hdr_flags      = FUSED_SCALE_2X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "1000x600 HDR 2x init should succeed");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_2X].width,  500, "output width = 500");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_2X].height, 300, "output height = 300");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 24. test_hdr_minimum_dimensions
 *     64x4, I010, PQ, FUSED_SCALE_2X, hdr_flags=2x.
 *     Minimum valid HDR configuration.
 * -------------------------------------------------------------------------- */

static void test_hdr_minimum_dimensions(void)
{
    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 64;
    ctx.src_height     = 4;
    ctx.src_y_stride   = align_up_32(64 * 2);
    ctx.src_uv_stride  = align_up_32(32 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_2X;
    ctx.hdr_flags      = FUSED_SCALE_2X;
    ctx.sdr_flags      = 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "64x4 HDR 2x init should succeed");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_2X].width,  32, "output width = 32");
    TEST_ASSERT_EQ(ctx.hdr_outputs[FUSED_IDX_2X].height,  2, "output height = 2");

    fused_hdr_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 25. test_hdr_forced_scalar_fallback_fields
 *     With FUNNELCAKE_FORCE_SCALAR set, every produced HDR descriptor should
 *     report fallback=1, matching the public fallback contract.
 * -------------------------------------------------------------------------- */

static void test_hdr_forced_scalar_fallback_fields(void)
{
    setenv("FUNNELCAKE_FORCE_SCALAR", "1", 1);
    fused_detect_cpu_reset();

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = 192;
    ctx.src_height     = 108;
    ctx.src_y_stride   = align_up_32(192 * 2);
    ctx.src_uv_stride  = align_up_32(96 * 2);
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X;
    ctx.hdr_flags      = FUSED_SCALE_1_5X;
    ctx.sdr_flags      = FUSED_SCALE_3X;
    ctx.tonemap_1x     = 1;
    ctx.upscale_flags  = FUSED_UPSCALE_2X;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    int failed = 0;

    if (fused_simd_available() != 0) {
        printf("  FAIL [%s:%d] forced scalar should disable SIMD\n", __func__, __LINE__);
        failed = 1;
    } else if ((rc & FUSED_WARN_BIT_SCALAR) == 0 || rc < 0) {
        printf("  FAIL [%s:%d] forced scalar init rc=%d, expected scalar warning\n",
               __func__, __LINE__, rc);
        failed = 1;
    } else if (ctx.hdr_outputs[FUSED_IDX_1_5X].fallback != 1) {
        printf("  FAIL [%s:%d] hdr output fallback=%d, expected 1\n",
               __func__, __LINE__, ctx.hdr_outputs[FUSED_IDX_1_5X].fallback);
        failed = 1;
    } else if (ctx.sdr_outputs[FUSED_IDX_3X].fallback != 1) {
        printf("  FAIL [%s:%d] sdr output fallback=%d, expected 1\n",
               __func__, __LINE__, ctx.sdr_outputs[FUSED_IDX_3X].fallback);
        failed = 1;
    } else if (ctx.output_1x.fallback != 1) {
        printf("  FAIL [%s:%d] 1x output fallback=%d, expected 1\n",
               __func__, __LINE__, ctx.output_1x.fallback);
        failed = 1;
    } else if (ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X].fallback != 1) {
        printf("  FAIL [%s:%d] upscale output fallback=%d, expected 1\n",
               __func__, __LINE__, ctx.upscale_hdr_outputs[FUSED_UP_IDX_2X].fallback);
        failed = 1;
    }

    fused_hdr_free(&ctx);
    unsetenv("FUNNELCAKE_FORCE_SCALAR");
    fused_detect_cpu_reset();

    if (failed) {
        g_results.failed++;
        return;
    }

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * run_hdr_validation_tests
 * -------------------------------------------------------------------------- */

void run_hdr_validation_tests(void)
{
    RUN_TEST(test_hdr_valid_i010_thirds);
    RUN_TEST(test_hdr_valid_i010_pow2);
    RUN_TEST(test_hdr_valid_p010);
    RUN_TEST(test_hdr_valid_i210);
    RUN_TEST(test_hdr_valid_p210);
    RUN_TEST(test_hdr_mixed_outputs);
    RUN_TEST(test_hdr_sdr_only);
    RUN_TEST(test_hdr_hdr_only);
    RUN_TEST(test_hdr_tonemap_1x);
    RUN_TEST(test_hdr_upscale_sdr_flags);
    RUN_TEST(test_hdr_flags_not_subset);
    RUN_TEST(test_hdr_invalid_format);
    RUN_TEST(test_hdr_invalid_transfer);
    RUN_TEST(test_hdr_custom_null_lut);
    RUN_TEST(test_hdr_all_curves);
    RUN_TEST(test_hdr_output_alignment);
    RUN_TEST(test_hdr_free_cleans_up);
    RUN_TEST(test_hdr_i010_hlg);
    RUN_TEST(test_hdr_p010_hlg);
    RUN_TEST(test_hdr_16x_pow2);
    RUN_TEST(test_hdr_12x_only);
    RUN_TEST(test_hdr_tonemap_1x_only);
    RUN_TEST(test_hdr_custom_lut_output);
    RUN_TEST(test_hdr_non_standard_width);
    RUN_TEST(test_hdr_minimum_dimensions);
    RUN_TEST(test_hdr_forced_scalar_fallback_fields);
}
