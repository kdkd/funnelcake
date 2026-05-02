/* --------------------------------------------------------------------------
 * test_parity.c - SIMD-vs-scalar bit-exact output parity check.
 *
 * Runs each (workload x frame size) twice on the same deterministic input:
 *   A) with FUNNELCAKE_FORCE_SCALAR=1   -> scalar kernel
 *   B) with FUNNELCAKE_FORCE_SCALAR unset -> native SIMD kernel (if any)
 * and bit-compares every output plane.  The detection cache is flipped via
 * fused_detect_cpu_reset() between runs.
 *
 * The existing AVX2 / NEON / RVV kernels are designed to be bit-exact
 * against scalar (the scalar avg_u8 helper rounds the same way as
 * vpavgb / vrhadd / vaaddu).  Any divergence is a real bug, not a tolerance
 * issue.
 *
 * On a build with no SIMD path (UNAME_M without an arch branch in the
 * Makefile, or a CPU that fails its arch's detection), both runs use the
 * scalar kernel and the test trivially passes - still useful as a smoke
 * test for the FORCE_SCALAR plumbing.
 * -------------------------------------------------------------------------- */

#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include "detect.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
 * Captured-output containers
 *
 * After each scaler run we copy the planes out into our own malloc'd
 * buffers so the second run (which calls fused_scaler_init/free again,
 * reallocating the internal output buffers) doesn't clobber them.
 * -------------------------------------------------------------------------- */

typedef struct {
    int      present;
    int      w, h;
    int      y_stride, uv_stride;  /* in bytes  (SDR) or elements*2 (HDR) */
    void    *y, *u, *v;            /* uint8_t*  (SDR) or uint16_t*  (HDR) */
} captured_plane_t;

typedef struct {
    captured_plane_t down[8];
    captured_plane_t up[FUSED_MAX_UPSCALE_STEPS];
} captured_sdr_t;

typedef struct {
    captured_plane_t hdr_down[8];
    captured_plane_t hdr_up[FUSED_MAX_UPSCALE_STEPS];
} captured_hdr_t;

static void capture_init(captured_sdr_t *c) { memset(c, 0, sizeof(*c)); }
static void capture_init_hdr(captured_hdr_t *c) { memset(c, 0, sizeof(*c)); }

static void plane_free(captured_plane_t *p)
{
    free(p->y); free(p->u); free(p->v);
    memset(p, 0, sizeof(*p));
}

static void capture_free(captured_sdr_t *c)
{
    for (int i = 0; i < 8; i++) plane_free(&c->down[i]);
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) plane_free(&c->up[i]);
}

static void capture_free_hdr(captured_hdr_t *c)
{
    for (int i = 0; i < 8; i++) plane_free(&c->hdr_down[i]);
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) plane_free(&c->hdr_up[i]);
}

/* Copy a single SDR plane (w x h bytes, with stride) into a freshly-malloc'd
 * tightly-packed buffer.  We pack to dst_w bytes per row so the comparison
 * doesn't depend on internal stride padding (which can differ if the
 * allocator returns differently-aligned base pointers between runs). */
static void *copy_plane_packed(const void *src, int w, int h, int src_stride,
                               int el_size)
{
    if (!src || w <= 0 || h <= 0) return NULL;
    size_t row_bytes = (size_t)w * (size_t)el_size;
    size_t total = row_bytes * (size_t)h;
    void *dst = malloc(total);
    if (!dst) return NULL;
    for (int y = 0; y < h; y++) {
        memcpy((char *)dst + (size_t)y * row_bytes,
               (const char *)src + (size_t)y * (size_t)src_stride,
               row_bytes);
    }
    return dst;
}

/* --------------------------------------------------------------------------
 * SDR scaler runner
 * -------------------------------------------------------------------------- */

static int run_sdr_capture(int src_w, int src_h, uint32_t down_flags,
                           uint32_t up_flags, int up_tail,
                           uint32_t seed, captured_sdr_t *out)
{
    test_frame_t frame;
    if (test_frame_create(&frame, src_w, src_h, PATTERN_RANDOM, seed) != 0) {
        return -1;
    }

    fused_scaler_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width        = frame.width;
    ctx.src_height       = frame.height;
    ctx.src_y_stride     = frame.y_stride;
    ctx.src_uv_stride    = frame.uv_stride;
    ctx.requested_flags  = down_flags;
    ctx.upscale_flags    = up_flags;
    ctx.upscale_tail_1_5x = up_tail;
    ctx.log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx.log_errors.target   = FUSED_LOG_SUPPRESS;

    int rc = fused_scaler_init(&ctx);
    if (rc < 0) {
        test_frame_free(&frame);
        return rc;
    }

    fused_scaler_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    /* Capture downscale outputs */
    for (int i = 0; i < 8; i++) {
        if (!ctx.outputs[i].plane_y) continue;
        captured_plane_t *p = &out->down[i];
        p->present   = 1;
        p->w         = ctx.outputs[i].width;
        p->h         = ctx.outputs[i].height;
        p->y_stride  = ctx.outputs[i].y_stride;
        p->uv_stride = ctx.outputs[i].uv_stride;
        p->y = copy_plane_packed(ctx.outputs[i].plane_y, p->w, p->h,
                                  p->y_stride, 1);
        p->u = copy_plane_packed(ctx.outputs[i].plane_u, p->w / 2, p->h / 2,
                                  p->uv_stride, 1);
        p->v = copy_plane_packed(ctx.outputs[i].plane_v, p->w / 2, p->h / 2,
                                  p->uv_stride, 1);
    }
    /* Capture upscale outputs */
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) {
        if (!ctx.upscale_outputs[i].plane_y) continue;
        captured_plane_t *p = &out->up[i];
        p->present   = 1;
        p->w         = ctx.upscale_outputs[i].width;
        p->h         = ctx.upscale_outputs[i].height;
        p->y_stride  = ctx.upscale_outputs[i].y_stride;
        p->uv_stride = ctx.upscale_outputs[i].uv_stride;
        p->y = copy_plane_packed(ctx.upscale_outputs[i].plane_y, p->w, p->h,
                                  p->y_stride, 1);
        p->u = copy_plane_packed(ctx.upscale_outputs[i].plane_u, p->w / 2, p->h / 2,
                                  p->uv_stride, 1);
        p->v = copy_plane_packed(ctx.upscale_outputs[i].plane_v, p->w / 2, p->h / 2,
                                  p->uv_stride, 1);
    }

    fused_scaler_free(&ctx);
    test_frame_free(&frame);
    return 0;
}

/* --------------------------------------------------------------------------
 * HDR scaler runner
 *
 * 10-bit HDR-only, no tone mapping - we want to compare scaling kernels in
 * isolation, not the LUTs.
 * -------------------------------------------------------------------------- */

static int run_hdr_capture(int src_w, int src_h, uint32_t down_flags,
                           uint32_t up_flags, int up_tail,
                           uint32_t seed, captured_hdr_t *out)
{
    test_hdr_frame_t frame;
    if (test_hdr_frame_create(&frame, src_w, src_h, PATTERN_RANDOM, seed) != 0) {
        return -1;
    }

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width         = frame.width;
    ctx.src_height        = frame.height;
    ctx.src_y_stride      = frame.y_stride;
    ctx.src_uv_stride     = frame.uv_stride;
    ctx.src_format        = FUSED_PIX_I010;
    ctx.src_transfer      = FUSED_TRC_PQ;
    ctx.requested_flags   = down_flags;
    ctx.hdr_flags         = down_flags;
    ctx.sdr_flags         = 0;
    ctx.upscale_flags     = up_flags;
    ctx.upscale_tail_1_5x = up_tail;
    ctx.log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx.log_errors.target   = FUSED_LOG_SUPPRESS;

    int rc = fused_hdr_init(&ctx);
    if (rc < 0) {
        test_hdr_frame_free(&frame);
        return rc;
    }

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    for (int i = 0; i < 8; i++) {
        if (!ctx.hdr_outputs[i].plane_y) continue;
        captured_plane_t *p = &out->hdr_down[i];
        p->present   = 1;
        p->w         = ctx.hdr_outputs[i].width;
        p->h         = ctx.hdr_outputs[i].height;
        p->y_stride  = ctx.hdr_outputs[i].y_stride;
        p->uv_stride = ctx.hdr_outputs[i].uv_stride;
        p->y = copy_plane_packed(ctx.hdr_outputs[i].plane_y, p->w, p->h,
                                  p->y_stride, 2);
        p->u = copy_plane_packed(ctx.hdr_outputs[i].plane_u, p->w / 2, p->h / 2,
                                  p->uv_stride, 2);
        p->v = copy_plane_packed(ctx.hdr_outputs[i].plane_v, p->w / 2, p->h / 2,
                                  p->uv_stride, 2);
    }
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) {
        if (!ctx.upscale_hdr_outputs[i].plane_y) continue;
        captured_plane_t *p = &out->hdr_up[i];
        p->present   = 1;
        p->w         = ctx.upscale_hdr_outputs[i].width;
        p->h         = ctx.upscale_hdr_outputs[i].height;
        p->y_stride  = ctx.upscale_hdr_outputs[i].y_stride;
        p->uv_stride = ctx.upscale_hdr_outputs[i].uv_stride;
        p->y = copy_plane_packed(ctx.upscale_hdr_outputs[i].plane_y, p->w, p->h,
                                  p->y_stride, 2);
        p->u = copy_plane_packed(ctx.upscale_hdr_outputs[i].plane_u, p->w / 2, p->h / 2,
                                  p->uv_stride, 2);
        p->v = copy_plane_packed(ctx.upscale_hdr_outputs[i].plane_v, p->w / 2, p->h / 2,
                                  p->uv_stride, 2);
    }

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    return 0;
}

/* --------------------------------------------------------------------------
 * Plane comparison - returns 0 if identical, sets *first_diff and returns 1
 * on the first byte mismatch.
 * -------------------------------------------------------------------------- */

typedef struct {
    int   plane_idx;     /* 0..7 (down) or 0..5 (up) */
    char  plane_label;   /* 'Y' / 'U' / 'V' */
    int   x, y;
    int   scalar_val;
    int   simd_val;
} parity_diff_t;

static int compare_plane_bytes(const void *a, const void *b, int w, int h,
                               int el_size, int plane_idx, char label,
                               parity_diff_t *first_diff)
{
    if (a == NULL && b == NULL) return 0;
    if (a == NULL || b == NULL) {
        first_diff->plane_idx = plane_idx;
        first_diff->plane_label = label;
        first_diff->x = first_diff->y = -1;
        first_diff->scalar_val = a ? 1 : 0;
        first_diff->simd_val   = b ? 1 : 0;
        return 1;
    }
    size_t row_bytes = (size_t)w * (size_t)el_size;
    for (int y = 0; y < h; y++) {
        const uint8_t *ra = (const uint8_t *)a + (size_t)y * row_bytes;
        const uint8_t *rb = (const uint8_t *)b + (size_t)y * row_bytes;
        if (memcmp(ra, rb, row_bytes) == 0) continue;
        for (int x = 0; x < w; x++) {
            int va, vb;
            if (el_size == 1) {
                va = ((const uint8_t *)ra)[x];
                vb = ((const uint8_t *)rb)[x];
            } else {
                va = ((const uint16_t *)ra)[x];
                vb = ((const uint16_t *)rb)[x];
            }
            if (va != vb) {
                first_diff->plane_idx = plane_idx;
                first_diff->plane_label = label;
                first_diff->x = x;
                first_diff->y = y;
                first_diff->scalar_val = va;
                first_diff->simd_val   = vb;
                return 1;
            }
        }
    }
    return 0;
}

static int compare_captured(const captured_plane_t *a,
                            const captured_plane_t *b,
                            int el_size, int plane_idx,
                            parity_diff_t *first_diff)
{
    if (a->present != b->present) {
        first_diff->plane_idx = plane_idx;
        first_diff->plane_label = '?';
        first_diff->x = first_diff->y = -1;
        first_diff->scalar_val = a->present;
        first_diff->simd_val   = b->present;
        return 1;
    }
    if (!a->present) return 0;
    if (a->w != b->w || a->h != b->h) {
        first_diff->plane_idx = plane_idx;
        first_diff->plane_label = '!';
        first_diff->x = a->w;
        first_diff->y = a->h;
        first_diff->scalar_val = b->w;
        first_diff->simd_val   = b->h;
        return 1;
    }
    if (compare_plane_bytes(a->y, b->y, a->w, a->h, el_size,
                            plane_idx, 'Y', first_diff)) return 1;
    if (compare_plane_bytes(a->u, b->u, a->w / 2, a->h / 2, el_size,
                            plane_idx, 'U', first_diff)) return 1;
    if (compare_plane_bytes(a->v, b->v, a->w / 2, a->h / 2, el_size,
                            plane_idx, 'V', first_diff)) return 1;
    return 0;
}

/* --------------------------------------------------------------------------
 * Detection-state toggle
 * -------------------------------------------------------------------------- */

static void force_scalar_on(void)
{
    setenv("FUNNELCAKE_FORCE_SCALAR", "1", 1);
    fused_detect_cpu_reset();
}

static void force_scalar_off(void)
{
    unsetenv("FUNNELCAKE_FORCE_SCALAR");
    fused_detect_cpu_reset();
}

/* --------------------------------------------------------------------------
 * Test workloads
 * -------------------------------------------------------------------------- */

typedef struct {
    const char *label;
    int         w, h;
    uint32_t    down_flags;
    uint32_t    up_flags;
    int         up_tail;
} sdr_workload_t;

static const sdr_workload_t k_sdr_workloads[] = {
    /* Pow2 down only */
    { "1280x720 down:2x,4x",
      1280, 720, FUSED_SCALE_2X | FUSED_SCALE_4X, 0, 0 },
    /* Thirds down only */
    { "1280x720 down:1.5x,3x",
      1280, 720, FUSED_SCALE_1_5X | FUSED_SCALE_3X, 0, 0 },
    /* Plain upscale (1.5x tail) */
    { "640x360  up:1.5x",
      640, 360, 0, 0, 1 },
    /* Pow2 upscale chain */
    { "640x360  up:2x,4x",
      640, 360, 0, FUSED_UPSCALE_2X | FUSED_UPSCALE_4X, 0 },
    /* Pow2 down + up combined */
    { "1280x720 down:2x up:2x",
      1280, 720, FUSED_SCALE_2X, FUSED_UPSCALE_2X, 0 },
    /* Thirds down + up combined */
    { "1280x720 down:1.5x,3x up:2x",
      1280, 720, FUSED_SCALE_1_5X | FUSED_SCALE_3X, FUSED_UPSCALE_2X, 0 },
};

typedef struct {
    const char *label;
    int         w, h;
    uint32_t    down_flags;
    uint32_t    up_flags;
    int         up_tail;
} hdr_workload_t;

static const hdr_workload_t k_hdr_workloads[] = {
    { "1280x720 HDR down:2x,4x",
      1280, 720, FUSED_SCALE_2X | FUSED_SCALE_4X, 0, 0 },
    { "1280x720 HDR down:1.5x,3x",
      1280, 720, FUSED_SCALE_1_5X | FUSED_SCALE_3X, 0, 0 },
    { "640x360  HDR up:1.5x",
      640, 360, 0, 0, 1 },
    { "640x360  HDR up:2x,4x",
      640, 360, 0, FUSED_UPSCALE_2X | FUSED_UPSCALE_4X, 0 },
    { "1280x720 HDR down:2x up:2x",
      1280, 720, FUSED_SCALE_2X, FUSED_UPSCALE_2X, 0 },
    { "1280x720 HDR down:1.5x,3x up:2x",
      1280, 720, FUSED_SCALE_1_5X | FUSED_SCALE_3X, FUSED_UPSCALE_2X, 0 },
};

/* --------------------------------------------------------------------------
 * SDR parity test
 * -------------------------------------------------------------------------- */

static void report_diff(const char *label, const parity_diff_t *d, int is_up)
{
    const char *kind = is_up ? "up" : "down";
    if (d->plane_label == '?') {
        printf("\n  FAIL [%s] %s output[%d] presence mismatch: scalar=%d simd=%d\n",
               label, kind, d->plane_idx, d->scalar_val, d->simd_val);
    } else if (d->plane_label == '!') {
        printf("\n  FAIL [%s] %s output[%d] dimension mismatch: scalar=%dx%d simd=%dx%d\n",
               label, kind, d->plane_idx, d->x, d->y, d->scalar_val, d->simd_val);
    } else {
        printf("\n  FAIL [%s] %s output[%d] plane %c at (%d,%d): scalar=%d simd=%d\n",
               label, kind, d->plane_idx, d->plane_label, d->x, d->y,
               d->scalar_val, d->simd_val);
    }
}

static int test_sdr_workload(const sdr_workload_t *w)
{
    captured_sdr_t scalar_out, simd_out;
    capture_init(&scalar_out);
    capture_init(&simd_out);

    /* Run A: scalar */
    force_scalar_on();
    int rc = run_sdr_capture(w->w, w->h, w->down_flags, w->up_flags,
                             w->up_tail, /*seed=*/0xC0FFEE, &scalar_out);
    if (rc < 0) {
        printf("\n  FAIL [%s] scalar run init failed: rc=%d\n", w->label, rc);
        capture_free(&scalar_out);
        force_scalar_off();
        return -1;
    }

    /* Run B: native SIMD (or scalar again if no SIMD on this build/cpu) */
    force_scalar_off();
    rc = run_sdr_capture(w->w, w->h, w->down_flags, w->up_flags,
                         w->up_tail, /*seed=*/0xC0FFEE, &simd_out);
    if (rc < 0) {
        printf("\n  FAIL [%s] simd run init failed: rc=%d\n", w->label, rc);
        capture_free(&scalar_out);
        capture_free(&simd_out);
        return -1;
    }

    parity_diff_t diff;
    memset(&diff, 0, sizeof(diff));
    int failed = 0;
    for (int i = 0; i < 8 && !failed; i++) {
        if (compare_captured(&scalar_out.down[i], &simd_out.down[i],
                             1, i, &diff)) {
            report_diff(w->label, &diff, /*is_up=*/0);
            failed = 1;
        }
    }
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS && !failed; i++) {
        if (compare_captured(&scalar_out.up[i], &simd_out.up[i],
                             1, i, &diff)) {
            report_diff(w->label, &diff, /*is_up=*/1);
            failed = 1;
        }
    }

    capture_free(&scalar_out);
    capture_free(&simd_out);
    return failed ? -1 : 0;
}

static void test_sdr_parity_all(void)
{
    int n_pass = 0, n_fail = 0;
    for (size_t i = 0; i < sizeof(k_sdr_workloads)/sizeof(k_sdr_workloads[0]); i++) {
        if (test_sdr_workload(&k_sdr_workloads[i]) == 0) {
            n_pass++;
        } else {
            n_fail++;
        }
    }
    if (n_fail == 0) {
        TEST_PASS();
    } else {
        printf("  (%d workloads passed, %d failed)\n", n_pass, n_fail);
        g_results.failed++;
    }
    /* Restore detection state for any tests that follow. */
    force_scalar_off();
}

/* --------------------------------------------------------------------------
 * HDR parity test
 * -------------------------------------------------------------------------- */

static int test_hdr_workload(const hdr_workload_t *w)
{
    captured_hdr_t scalar_out, simd_out;
    capture_init_hdr(&scalar_out);
    capture_init_hdr(&simd_out);

    force_scalar_on();
    int rc = run_hdr_capture(w->w, w->h, w->down_flags, w->up_flags,
                             w->up_tail, /*seed=*/0xC0FFEE, &scalar_out);
    if (rc < 0) {
        printf("\n  FAIL [%s] scalar run init failed: rc=%d\n", w->label, rc);
        capture_free_hdr(&scalar_out);
        force_scalar_off();
        return -1;
    }

    force_scalar_off();
    rc = run_hdr_capture(w->w, w->h, w->down_flags, w->up_flags,
                         w->up_tail, /*seed=*/0xC0FFEE, &simd_out);
    if (rc < 0) {
        printf("\n  FAIL [%s] simd run init failed: rc=%d\n", w->label, rc);
        capture_free_hdr(&scalar_out);
        capture_free_hdr(&simd_out);
        return -1;
    }

    parity_diff_t diff;
    memset(&diff, 0, sizeof(diff));
    int failed = 0;
    for (int i = 0; i < 8 && !failed; i++) {
        if (compare_captured(&scalar_out.hdr_down[i], &simd_out.hdr_down[i],
                             2, i, &diff)) {
            report_diff(w->label, &diff, /*is_up=*/0);
            failed = 1;
        }
    }
    for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS && !failed; i++) {
        if (compare_captured(&scalar_out.hdr_up[i], &simd_out.hdr_up[i],
                             2, i, &diff)) {
            report_diff(w->label, &diff, /*is_up=*/1);
            failed = 1;
        }
    }

    capture_free_hdr(&scalar_out);
    capture_free_hdr(&simd_out);
    return failed ? -1 : 0;
}

static void test_hdr_parity_all(void)
{
    int n_pass = 0, n_fail = 0;
    for (size_t i = 0; i < sizeof(k_hdr_workloads)/sizeof(k_hdr_workloads[0]); i++) {
        if (test_hdr_workload(&k_hdr_workloads[i]) == 0) {
            n_pass++;
        } else {
            n_fail++;
        }
    }
    if (n_fail == 0) {
        TEST_PASS();
    } else {
        printf("  (%d workloads passed, %d failed)\n", n_pass, n_fail);
        g_results.failed++;
    }
    force_scalar_off();
}

/* --------------------------------------------------------------------------
 * Entry point
 * -------------------------------------------------------------------------- */

void run_parity_tests(void)
{
    RUN_TEST(test_sdr_parity_all);
    RUN_TEST(test_hdr_parity_all);
}
