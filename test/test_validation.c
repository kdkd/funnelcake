#include "test_main.h"
#include "funnelcake.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * Local helpers
 * -------------------------------------------------------------------------- */

static int align_up_32(int x)
{
    return (x + 31) & ~31;
}

/* Suppress all log output for a context. */
static void suppress_log(fused_scaler_ctx_t *ctx)
{
    ctx->log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx->log_errors.target   = FUSED_LOG_SUPPRESS;
}

/* --------------------------------------------------------------------------
 * 1. test_valid_1080p_thirds
 *    1920x1080, flags 1.5x|3x|6x → FUSED_OK
 * -------------------------------------------------------------------------- */

static void test_valid_1080p_thirds(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1920;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = align_up_32(1920);
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_OK, "rc should be FUSED_OK");
    /* outputs[0] = FUSED_SCALE_1_5X (bit 0) */
    TEST_ASSERT_EQ(ctx.outputs[0].width,  1280, "outputs[0].width");
    TEST_ASSERT_EQ(ctx.outputs[0].height, 720,  "outputs[0].height");
    /* outputs[2] = FUSED_SCALE_3X (bit 2) */
    TEST_ASSERT_EQ(ctx.outputs[2].width, 640, "outputs[2].width");
    /* outputs[4] = FUSED_SCALE_6X (bit 4) */
    TEST_ASSERT_EQ(ctx.outputs[4].width, 320, "outputs[4].width");
    TEST_ASSERT(ctx.outputs[0].plane_y != NULL, "outputs[0].plane_y != NULL");

    fused_scaler_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 2. test_valid_720p_pow2
 *    1280x720, flags 2x|4x → FUSED_OK
 * -------------------------------------------------------------------------- */

static void test_valid_720p_pow2(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1280;
    ctx.src_height    = 720;
    ctx.src_y_stride  = align_up_32(1280);
    ctx.src_uv_stride = align_up_32(640);
    ctx.requested_flags = FUSED_SCALE_2X | FUSED_SCALE_4X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_OK, "rc should be FUSED_OK");
    /* outputs[1] = FUSED_SCALE_2X (bit 1) */
    TEST_ASSERT_EQ(ctx.outputs[1].width, 640, "outputs[1].width");
    /* outputs[3] = FUSED_SCALE_4X (bit 3) */
    TEST_ASSERT_EQ(ctx.outputs[3].width, 320, "outputs[3].width");

    fused_scaler_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 3. test_valid_4k_full_thirds
 *    3840x2160, flags 1.5x|3x|6x|12x → FUSED_OK
 * -------------------------------------------------------------------------- */

static void test_valid_4k_full_thirds(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 3840;
    ctx.src_height    = 2160;
    ctx.src_y_stride  = align_up_32(3840);
    ctx.src_uv_stride = align_up_32(1920);
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X |
                          FUSED_SCALE_6X  | FUSED_SCALE_12X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_OK, "rc should be FUSED_OK");
    TEST_ASSERT(ctx.outputs[0].plane_y != NULL, "outputs[0] allocated");
    TEST_ASSERT(ctx.outputs[6].plane_y != NULL, "outputs[6] allocated");

    fused_scaler_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 4. test_mixed_families
 *    1920x1080, flags 1.5x|2x → FUSED_ERR_INVALID_FLAGS
 * -------------------------------------------------------------------------- */

static void test_mixed_families(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1920;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = align_up_32(1920);
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_2X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_ERR_INVALID_FLAGS, "mixed families → FUSED_ERR_INVALID_FLAGS");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 5. test_empty_flags
 *    flags=0 → FUSED_ERR_INVALID_FLAGS
 * -------------------------------------------------------------------------- */

static void test_empty_flags(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1920;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = align_up_32(1920);
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = 0;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_ERR_INVALID_FLAGS, "empty flags → FUSED_ERR_INVALID_FLAGS");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 6. test_bad_dimensions_zero
 *    width=0 → FUSED_ERR_BAD_DIMENSIONS
 * -------------------------------------------------------------------------- */

static void test_bad_dimensions_zero(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 0;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = align_up_32(1920);
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = FUSED_SCALE_1_5X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_ERR_BAD_DIMENSIONS, "zero width → FUSED_ERR_BAD_DIMENSIONS");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 7. test_bad_dimensions_odd
 *    width=1921, height=1081 → FUSED_ERR_BAD_DIMENSIONS
 * -------------------------------------------------------------------------- */

static void test_bad_dimensions_odd(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1921;
    ctx.src_height    = 1081;
    ctx.src_y_stride  = align_up_32(1921);
    ctx.src_uv_stride = align_up_32(961);
    ctx.requested_flags = FUSED_SCALE_1_5X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_ERR_BAD_DIMENSIONS, "odd dims → FUSED_ERR_BAD_DIMENSIONS");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 8. test_bad_alignment
 *    stride=100 (not 32-byte aligned) → FUSED_ERR_BAD_ALIGNMENT
 * -------------------------------------------------------------------------- */

static void test_bad_alignment(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1920;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = 100;   /* deliberately misaligned */
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = FUSED_SCALE_1_5X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT_EQ(rc, FUSED_ERR_BAD_ALIGNMENT, "bad stride → FUSED_ERR_BAD_ALIGNMENT");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 9. test_crop_to_fit
 *    1360x762, flags=2x (no options)
 *
 *    Crop-to-fit rounds down to multiple of 2*max_ratio=4 to ensure even
 *    output: eff_h = round_down(762, 4) = 760. Output = 760/2 = 380 (even).
 *    Width 1360 is already divisible by 4, so eff_w = 1360.
 * -------------------------------------------------------------------------- */

static void test_crop_to_fit(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1360;
    ctx.src_height    = 762;
    ctx.src_y_stride  = align_up_32(1360);
    ctx.src_uv_stride = align_up_32(680);
    ctx.requested_flags = FUSED_SCALE_2X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT(rc >= 0, "1360x762 2x crop: should succeed");
    TEST_ASSERT(rc & FUSED_WARN_BIT_CROPPED, "should have CROPPED warning");
    TEST_ASSERT_EQ(ctx.effective_height, 760, "effective height cropped to 760");
    TEST_ASSERT_EQ(ctx.effective_width, 1360, "effective width unchanged");
    TEST_ASSERT_EQ(ctx.outputs[1].height, 380, "2x output height = 380");
    TEST_ASSERT_EQ(ctx.outputs[1].width, 680, "2x output width = 680");

    fused_scaler_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 10. test_no_crop_rejects
 *     1360x762, flags=2x, FUSED_OPT_NO_CROP
 *
 *     Without crop-to-fit, eff_h=762. out_h=381 (odd) → step rejected.
 *     achieved=0 → FUSED_ERR_NO_STEPS (negative, not FUSED_WARN_BIT_PARTIAL).
 * -------------------------------------------------------------------------- */

static void test_no_crop_rejects(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1360;
    ctx.src_height    = 762;
    ctx.src_y_stride  = align_up_32(1360);
    ctx.src_uv_stride = align_up_32(680);
    ctx.requested_flags = FUSED_SCALE_2X;
    ctx.options = FUSED_OPT_NO_CROP;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    /* Step rejected, achieved=0 → hard error */
    TEST_ASSERT(rc < 0, "1360x762 2x NO_CROP: step rejected → negative return code");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 11. test_scalar_fallback_oddball
 *     1184x660, flags=2x, y_stride=1216, uv_stride=608 (no fallback opt)
 *     chroma output width = 592/2 = 296; 296%32 != 0 → scalar fallback.
 *     Should succeed with FUSED_WARN_BIT_SCALAR, outputs[1].fallback=1.
 * -------------------------------------------------------------------------- */

static void test_scalar_fallback_oddball(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1184;
    ctx.src_height    = 660;
    ctx.src_y_stride  = 1216;   /* next multiple of 32 >= 1184 */
    ctx.src_uv_stride = 608;    /* next multiple of 32 >= 592  */
    ctx.requested_flags = FUSED_SCALE_2X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT(rc >= 0, "scalar fallback case should succeed");
    TEST_ASSERT((rc & FUSED_WARN_BIT_SCALAR) != 0, "FUSED_WARN_BIT_SCALAR should be set");
    TEST_ASSERT_EQ(ctx.outputs[1].fallback, 1, "outputs[1].fallback should be 1");
    TEST_ASSERT_EQ(ctx.outputs[1].width, 592, "outputs[1].width");

    fused_scaler_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 12. test_no_fallback_rejects
 *     Same as above but with FUSED_OPT_NO_FALLBACK.
 *     chroma output width 296 not %32, fallback rejected → achieved=0 →
 *     FUSED_ERR_NO_STEPS (negative).
 * -------------------------------------------------------------------------- */

static void test_no_fallback_rejects(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1184;
    ctx.src_height    = 660;
    ctx.src_y_stride  = 1216;
    ctx.src_uv_stride = 608;
    ctx.requested_flags = FUSED_SCALE_2X;
    ctx.options = FUSED_OPT_NO_FALLBACK;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    /* Step rejected (chroma width not %32), achieved=0 → hard error */
    TEST_ASSERT(rc < 0, "NO_FALLBACK oddball: step rejected → negative return code");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 13. test_deep_step_only
 *     1920x1080, flags=6x only (bit 4).
 *     Shallow slots (0, 2) not requested → width=0, plane_y=NULL.
 *     outputs[4] (6x) → width=320.
 * -------------------------------------------------------------------------- */

static void test_deep_step_only(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1920;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = align_up_32(1920);
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = FUSED_SCALE_6X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT(rc >= 0, "deep step only should succeed");
    TEST_ASSERT_EQ(ctx.outputs[0].width, 0, "outputs[0].width should be 0 (not requested)");
    TEST_ASSERT(ctx.outputs[0].plane_y == NULL, "outputs[0].plane_y should be NULL");
    TEST_ASSERT_EQ(ctx.outputs[2].width, 0, "outputs[2].width should be 0 (not requested)");
    TEST_ASSERT_EQ(ctx.outputs[4].width, 320, "outputs[4].width should be 320");

    fused_scaler_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 14. test_output_alignment
 *     Verify all allocated output planes are 32-byte aligned and all strides
 *     are multiples of 32.
 * -------------------------------------------------------------------------- */

static void test_output_alignment(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1920;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = align_up_32(1920);
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed");

    for (int i = 0; i < 8; i++) {
        if (ctx.outputs[i].plane_y) {
            TEST_ASSERT(((uintptr_t)ctx.outputs[i].plane_y % 32) == 0,
                        "plane_y 32-byte aligned");
            TEST_ASSERT(((uintptr_t)ctx.outputs[i].plane_u % 32) == 0,
                        "plane_u 32-byte aligned");
            TEST_ASSERT(((uintptr_t)ctx.outputs[i].plane_v % 32) == 0,
                        "plane_v 32-byte aligned");
            TEST_ASSERT((ctx.outputs[i].y_stride  % 32) == 0, "y_stride multiple of 32");
            TEST_ASSERT((ctx.outputs[i].uv_stride % 32) == 0, "uv_stride multiple of 32");
        }
    }

    fused_scaler_free(&ctx);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 15. test_all_production_ladders
 *     All 6 standard ladders should return FUSED_OK.
 * -------------------------------------------------------------------------- */

static void test_all_production_ladders(void)
{
    struct {
        int w, h;
        uint32_t flags;
        const char *name;
    } ladders[] = {
        { 3840, 2160, FUSED_SCALE_1_5X|FUSED_SCALE_3X|FUSED_SCALE_6X|FUSED_SCALE_12X,
          "3840x2160 thirds" },
        { 2560, 1440, FUSED_SCALE_2X|FUSED_SCALE_4X|FUSED_SCALE_8X,
          "2560x1440 pow2"  },
        { 1920, 1080, FUSED_SCALE_1_5X|FUSED_SCALE_3X|FUSED_SCALE_6X,
          "1920x1080 thirds" },
        { 1280,  720, FUSED_SCALE_2X|FUSED_SCALE_4X,
          "1280x720 pow2"   },
        {  960,  540, FUSED_SCALE_1_5X|FUSED_SCALE_3X,
          "960x540 thirds"  },
        {  640,  360, FUSED_SCALE_2X,
          "640x360 pow2"    },
    };
    int n = (int)(sizeof(ladders) / sizeof(ladders[0]));

    for (int i = 0; i < n; i++) {
        fused_scaler_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.src_width     = ladders[i].w;
        ctx.src_height    = ladders[i].h;
        ctx.src_y_stride  = align_up_32(ladders[i].w);
        ctx.src_uv_stride = align_up_32(ladders[i].w / 2);
        ctx.requested_flags = ladders[i].flags;
        suppress_log(&ctx);

        int rc = fused_scaler_init(&ctx);
        if (rc != FUSED_OK) {
            printf("\n  FAIL [%s:%d] ladder %s: rc=%d, expected FUSED_OK\n",
                   __func__, __LINE__, ladders[i].name, rc);
            g_results.failed++;
            fused_scaler_free(&ctx);
            return;
        }
        fused_scaler_free(&ctx);
    }

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 16. test_free_cleans_up
 *     init then free → all output plane pointers should be NULL.
 * -------------------------------------------------------------------------- */

static void test_free_cleans_up(void)
{
    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1920;
    ctx.src_height    = 1080;
    ctx.src_y_stride  = align_up_32(1920);
    ctx.src_uv_stride = align_up_32(960);
    ctx.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
    suppress_log(&ctx);

    int rc = fused_scaler_init(&ctx);
    TEST_ASSERT(rc >= 0, "init should succeed before free test");

    fused_scaler_free(&ctx);

    for (int i = 0; i < 8; i++) {
        TEST_ASSERT(ctx.outputs[i].plane_y == NULL, "plane_y NULL after free");
        TEST_ASSERT(ctx.outputs[i].plane_u == NULL, "plane_u NULL after free");
        TEST_ASSERT(ctx.outputs[i].plane_v == NULL, "plane_v NULL after free");
    }
    TEST_ASSERT(ctx._internal == NULL, "_internal NULL after free");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 17. test_diagnostic_callback
 *     Use FUSED_LOG_CALLBACK for log_warnings. Use a resolution/option combo
 *     that triggers a warning-level diagnostic (NO_FALLBACK with oddball chroma).
 *     Verify callback is called and message is non-empty.
 * -------------------------------------------------------------------------- */

static int   s_cb_call_count = 0;
static char  s_cb_last_msg[4096];

static void diag_callback(int level, const char *msg, void *user_ctx)
{
    (void)level;
    (void)user_ctx;
    s_cb_call_count++;
    if (msg)
        snprintf(s_cb_last_msg, sizeof(s_cb_last_msg), "%s", msg);
}

static void test_diagnostic_callback(void)
{
    s_cb_call_count = 0;
    s_cb_last_msg[0] = '\0';

    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = 1184;
    ctx.src_height    = 660;
    ctx.src_y_stride  = 1216;
    ctx.src_uv_stride = 608;
    ctx.requested_flags = FUSED_SCALE_2X;
    ctx.options = FUSED_OPT_NO_FALLBACK;
    /* Route warnings to our callback */
    ctx.log_warnings.target   = FUSED_LOG_CALLBACK;
    ctx.log_warnings.callback = diag_callback;
    ctx.log_warnings.callback_ctx = NULL;
    /* Suppress errors (init returns hard error here) */
    ctx.log_errors.target = FUSED_LOG_SUPPRESS;

    fused_scaler_init(&ctx);
    fused_scaler_free(&ctx);

    TEST_ASSERT(s_cb_call_count > 0, "diagnostic callback was called");
    TEST_ASSERT(s_cb_last_msg[0] != '\0', "callback received non-empty message");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * run_validation_tests
 * -------------------------------------------------------------------------- */

void run_validation_tests(void)
{
    RUN_TEST(test_valid_1080p_thirds);
    RUN_TEST(test_valid_720p_pow2);
    RUN_TEST(test_valid_4k_full_thirds);
    RUN_TEST(test_mixed_families);
    RUN_TEST(test_empty_flags);
    RUN_TEST(test_bad_dimensions_zero);
    RUN_TEST(test_bad_dimensions_odd);
    RUN_TEST(test_bad_alignment);
    RUN_TEST(test_crop_to_fit);
    RUN_TEST(test_no_crop_rejects);
    RUN_TEST(test_scalar_fallback_oddball);
    RUN_TEST(test_no_fallback_rejects);
    RUN_TEST(test_deep_step_only);
    RUN_TEST(test_output_alignment);
    RUN_TEST(test_all_production_ladders);
    RUN_TEST(test_free_cleans_up);
    RUN_TEST(test_diagnostic_callback);
}
