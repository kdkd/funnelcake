/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Bench iteration policy:
 *   - BENCH_WARMUP_FRAMES is a small fixed number used to warm caches and
 *     get a rough per-iteration timing estimate.
 *   - BENCH_MEASURE_FRAMES_MAX is the upper bound on measurement iterations.
 *   - BENCH_MEASURE_TARGET_US is the target total measurement time - the
 *     actual iteration count is scaled to hit roughly this total.
 *   - BENCH_MEASURE_MIN is the minimum iteration count even for very slow
 *     kernels, to keep the median/percentile stats meaningful.
 * This keeps fast benches precise (~1000 samples) while capping slow
 * benches at ~1 second of measurement work. */
#define BENCH_WARMUP_FRAMES       5
#define BENCH_MEASURE_FRAMES_MAX  1000
#define BENCH_MEASURE_TARGET_US   800000.0
#define BENCH_MEASURE_MIN         50

/* Legacy alias for existing array declarations. */
#define BENCH_MEASURE_FRAMES      BENCH_MEASURE_FRAMES_MAX

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

static double time_diff_us(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1e6 +
           (end->tv_nsec - start->tv_nsec) / 1e3;
}

static int compare_doubles(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

/* --------------------------------------------------------------------------
 * bench_config: run one benchmark configuration
 * -------------------------------------------------------------------------- */

static void bench_config(const char *label, int width, int height,
                         uint32_t flags, uint32_t up_flags, int up_tail)
{
    test_frame_t frame;
    fused_scaler_ctx_t ctx;
    double times[BENCH_MEASURE_FRAMES];
    int rc;

    /* Build a random test frame */
    if (test_frame_create(&frame, width, height, PATTERN_RANDOM, 0xdeadbeef) != 0) {
        printf("  %-44s  ERROR: could not allocate test frame\n", label);
        return;
    }

    /* Init scaler - suppress log output during benchmarks */
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width    = width;
    ctx.src_height   = height;
    ctx.src_y_stride  = frame.y_stride;
    ctx.src_uv_stride = frame.uv_stride;
    ctx.requested_flags   = flags;
    ctx.upscale_flags     = up_flags;
    ctx.upscale_tail_1_5x = up_tail;
    ctx.log_errors.target   = FUSED_LOG_SUPPRESS;
    ctx.log_warnings.target = FUSED_LOG_SUPPRESS;

    rc = fused_scaler_init(&ctx);
    if (rc < 0) {
        printf("  %-44s  ERROR: fused_scaler_init returned %d\n", label, rc);
        test_frame_free(&frame);
        return;
    }

    /* Determine whether any step used SIMD or all fell back to scalar */
    int any_simd = 0;
    int any_scalar = 0;
    for (int b = 0; b < 8; b++) {
        if (ctx.outputs[b].plane_y != NULL) {
            if (ctx.outputs[b].fallback)
                any_scalar = 1;
            else
                any_simd = 1;
        }
    }
    /* Upscale outputs are always SIMD-eligible (no per-step alignment) */
    for (int b = 0; b < FUSED_MAX_UPSCALE_STEPS; b++) {
        if (ctx.upscale_outputs[b].plane_y != NULL) {
            if (ctx.upscale_outputs[b].fallback)
                any_scalar = 1;
            else
                any_simd = 1;
        }
    }
    const char *kernel_tag = (any_simd && !any_scalar) ? "SIMD  " :
                             (!any_simd && any_scalar)  ? "scalar" :
                                                          "mixed ";

    /* Warmup - also used as a rough per-iter timing probe so we can
     * size the measurement loop to a fixed total time budget. */
    struct timespec wt0, wt1;
    clock_gettime(CLOCK_MONOTONIC, &wt0);
    for (int i = 0; i < BENCH_WARMUP_FRAMES; i++) {
        fused_scaler_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);
    }
    clock_gettime(CLOCK_MONOTONIC, &wt1);
    double per_iter_est = time_diff_us(&wt0, &wt1) / (double)BENCH_WARMUP_FRAMES;
    if (per_iter_est <= 0.0) per_iter_est = 1.0;

    int iters = (int)(BENCH_MEASURE_TARGET_US / per_iter_est);
    if (iters < BENCH_MEASURE_MIN)       iters = BENCH_MEASURE_MIN;
    if (iters > BENCH_MEASURE_FRAMES_MAX) iters = BENCH_MEASURE_FRAMES_MAX;

    /* Measure */
    for (int i = 0; i < iters; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fused_scaler_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = time_diff_us(&t0, &t1);
    }

    /* Sort to find min / median / max */
    qsort(times, iters, sizeof(double), compare_doubles);

    double t_min    = times[0];
    double t_max    = times[iters - 1];
    double t_median = times[iters / 2];

    double t_sum = 0.0;
    for (int i = 0; i < iters; i++)
        t_sum += times[i];
    double t_mean = t_sum / (double)iters;

    /* 60 fps budget = 16.7 ms = 16700 us */
    double budget_pct = (t_median / 16700.0) * 100.0;

    printf("  %-44s [%s]  min=%-6.0f med=%-6.0f mean=%-6.0f max=%-6.0f us  (%.1f%% of 60fps budget)\n",
           label, kernel_tag,
           t_min, t_median, t_mean, t_max,
           budget_pct);

    /* Record median for comparison table */
    if (g_bench_comparison_count < BENCH_MAX_CONFIGS) {
        bench_comparison_t *c = &g_bench_comparison[g_bench_comparison_count];
        /* Check if this label already exists (from a previous run) */
        int found = -1;
        for (int ci = 0; ci < g_bench_comparison_count; ci++) {
            if (strcmp(g_bench_comparison[ci].label, label) == 0) { found = ci; break; }
        }
        if (found >= 0) {
            g_bench_comparison[found].funnelcake_med = t_median;
        } else {
            c->label = label;
            c->funnelcake_med = t_median;
            c->swscale_indep_med = 0;
            c->swscale_cascade_med = 0;
            g_bench_comparison_count++;
        }
    }

    fused_scaler_free(&ctx);
    test_frame_free(&frame);
}

/* --------------------------------------------------------------------------
 * Benchmark configurations
 * -------------------------------------------------------------------------- */

typedef struct {
    const char *label;
    int         width;
    int         height;
    uint32_t    flags;          /* downscale flags */
    uint32_t    up_flags;       /* upscale flags */
    int         up_tail;        /* 1.5x tail */
} bench_entry_t;

/* Label format:
 *   {WxH} [down:scales] [up:scales]
 *
 * "down" and "up" are omitted if that direction is not active.  Scales
 * are the actual output ratios in comma-separated form.  The 1.5x tail
 * is expressed as its actual cascade ratio (e.g. 2x + 1.5x tail -> the
 * tail output is at 3x, so the label reads "up:2x,3x").  Labels must
 * match the corresponding entries in test/test_swscale_bench.c so the
 * comparison table rows line up. */
static const bench_entry_t bench_configs[] = {
    /* Downscale-only */
    { "640x360 down:2x",                   640,  360,
      FUSED_SCALE_2X, 0, 0 },
    { "960x540 down:1.5x,3x",              960,  540,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X, 0, 0 },
    { "1280x720 down:2x,4x",              1280,  720,
      FUSED_SCALE_2X  | FUSED_SCALE_4X, 0, 0 },
    { "1920x1080 down:1.5x,3x,6x",        1920, 1080,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X, 0, 0 },
    { "2560x1440 down:2x,4x,8x",          2560, 1440,
      FUSED_SCALE_2X  | FUSED_SCALE_4X | FUSED_SCALE_8X, 0, 0 },
    { "3840x2160 down:1.5x,3x,6x,12x",    3840, 2160,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X | FUSED_SCALE_12X, 0, 0 },

    /* Upscale-only.  Memory-aware: deep cascades only on small sources. */
    { "480x270 up:2x",                     480,  270,
      0, FUSED_UPSCALE_2X, 0 },
    { "480x270 up:2x,4x",                  480,  270,
      0, FUSED_UPSCALE_2X|FUSED_UPSCALE_4X, 0 },
    { "960x540 up:2x",                     960,  540,
      0, FUSED_UPSCALE_2X, 0 },
    { "960x540 up:2x,3x",                  960,  540,
      0, FUSED_UPSCALE_2X, 1 },
    { "1920x1080 up:2x",                  1920, 1080,
      0, FUSED_UPSCALE_2X, 0 },
    { "1920x1080 up:1.5x",                1920, 1080,
      0, 0, 1 },
    { "240x136 up:2x,4x,8x,16x",           240,  136,
      0, FUSED_UPSCALE_2X|FUSED_UPSCALE_4X|FUSED_UPSCALE_8X|FUSED_UPSCALE_16X, 0 },
    { "120x68 up:2x,4x,8x,16x,32x",        120,   68,
      0, FUSED_UPSCALE_POW2_MASK, 0 },

    /* Combined down + up - single-pass stress tests.
     * Source dimensions must be divisible by 8 for the pow2 4x cascade
     * to produce even outputs at every level. */
    { "1920x1080 down:2x up:2x",          1920, 1080,
      FUSED_SCALE_2X, FUSED_UPSCALE_2X, 0 },
    { "1920x1080 down:1.5x,3x up:2x",     1920, 1080,
      FUSED_SCALE_1_5X|FUSED_SCALE_3X, FUSED_UPSCALE_2X, 0 },
    { "1280x720 down:2x,4x up:2x,4x",     1280,  720,
      FUSED_SCALE_2X|FUSED_SCALE_4X, FUSED_UPSCALE_2X|FUSED_UPSCALE_4X, 0 },
};

#define BENCH_CONFIG_COUNT ((int)(sizeof(bench_configs) / sizeof(bench_configs[0])))

/* --------------------------------------------------------------------------
 * Entry point
 * -------------------------------------------------------------------------- */

void run_bench_tests(const char *filter)
{
    for (int i = 0; i < BENCH_CONFIG_COUNT; i++) {
        const bench_entry_t *e = &bench_configs[i];
        if (filter != NULL && strstr(e->label, filter) == NULL)
            continue;
        bench_config(e->label, e->width, e->height, e->flags,
                     e->up_flags, e->up_tail);
    }
}
