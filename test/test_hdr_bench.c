#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define BENCH_WARMUP_FRAMES  100
#define BENCH_MEASURE_FRAMES 1000

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

static void suppress_log(fused_hdr_ctx_t *ctx)
{
    ctx->log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx->log_errors.target   = FUSED_LOG_SUPPRESS;
}

/* --------------------------------------------------------------------------
 * Benchmark configuration entry
 * -------------------------------------------------------------------------- */

typedef struct {
    const char *label;
    int         width;
    int         height;
    uint32_t    flags;
    int         want_hdr;     /* produce HDR outputs */
    int         want_sdr;     /* produce SDR (tone-mapped) outputs */
    int         is_p010;      /* use P010 format instead of I010 */
    int         tonemap_1x;   /* special: 1:1 tonemap only */
} hdr_bench_entry_t;

static const hdr_bench_entry_t hdr_bench_configs[] = {
    /* Resolution sweeps -- I010 HDR-only */
    { "640x360 I010 pow2 HDR",      640,  360,
      FUSED_SCALE_2X,
      1, 0, 0, 0 },
    { "960x540 I010 thirds HDR",    960,  540,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X,
      1, 0, 0, 0 },
    { "1280x720 I010 pow2 HDR",     1280, 720,
      FUSED_SCALE_2X | FUSED_SCALE_4X,
      1, 0, 0, 0 },
    { "1920x1080 I010 thirds HDR",  1920, 1080,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X,
      1, 0, 0, 0 },
    { "2560x1440 I010 pow2 HDR",    2560, 1440,
      FUSED_SCALE_2X | FUSED_SCALE_4X | FUSED_SCALE_8X,
      1, 0, 0, 0 },
    { "3840x2160 I010 thirds HDR",  3840, 2160,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X | FUSED_SCALE_12X,
      1, 0, 0, 0 },

    /* SDR-only (includes tone mapping cost) */
    { "1920x1080 I010 thirds SDR",  1920, 1080,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X,
      0, 1, 0, 0 },
    { "3840x2160 I010 thirds SDR",  3840, 2160,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X | FUSED_SCALE_12X,
      0, 1, 0, 0 },

    /* HDR+SDR combined */
    { "1920x1080 I010 thirds H+S",  1920, 1080,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X,
      1, 1, 0, 0 },

    /* P010 overhead measurement */
    { "3840x2160 P010 thirds HDR",  3840, 2160,
      FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X | FUSED_SCALE_12X,
      1, 0, 1, 0 },

    /* 1:1 tonemap only */
    { "3840x2160 I010 tonemap 1x",  3840, 2160,
      0,
      0, 0, 0, 1 },
};

#define HDR_BENCH_CONFIG_COUNT ((int)(sizeof(hdr_bench_configs) / sizeof(hdr_bench_configs[0])))

/* --------------------------------------------------------------------------
 * hdr_bench_config: run one HDR benchmark configuration (I010)
 * -------------------------------------------------------------------------- */

static void hdr_bench_i010(const hdr_bench_entry_t *e)
{
    test_hdr_frame_t frame;
    double times[BENCH_MEASURE_FRAMES];

    if (test_hdr_frame_create(&frame, e->width, e->height, PATTERN_RANDOM, 0xdeadbeef) != 0) {
        printf("  %-36s  ERROR: could not allocate test frame\n", e->label);
        return;
    }

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = e->width;
    ctx.src_height     = e->height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;

    if (e->tonemap_1x) {
        /* Special: tonemap_1x only, no scale flags needed beyond what the API requires */
        ctx.requested_flags = FUSED_SCALE_1_5X;  /* need at least one flag for init */
        ctx.hdr_flags       = 0;
        ctx.sdr_flags       = 0;
        ctx.tonemap_1x      = 1;
    } else {
        ctx.requested_flags = e->flags;
        ctx.hdr_flags       = e->want_hdr ? e->flags : 0;
        ctx.sdr_flags       = e->want_sdr ? e->flags : 0;
    }
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    if (rc < 0) {
        printf("  %-36s  ERROR: fused_hdr_init returned %d\n", e->label, rc);
        test_hdr_frame_free(&frame);
        return;
    }

    /* Determine kernel type */
    int any_simd = 0, any_scalar = 0;
    for (int b = 0; b < 8; b++) {
        if (ctx.hdr_outputs[b].plane_y != NULL) {
            if (ctx.hdr_outputs[b].fallback) any_scalar = 1;
            else                              any_simd = 1;
        }
        if (ctx.sdr_outputs[b].plane_y != NULL) {
            if (ctx.sdr_outputs[b].fallback) any_scalar = 1;
            else                              any_simd = 1;
        }
    }
    const char *kernel_tag = (any_simd && !any_scalar) ? "SIMD  " :
                             (!any_simd && any_scalar)  ? "scalar" :
                                                          "mixed ";

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP_FRAMES; i++)
        fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    /* Measure */
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = time_diff_us(&t0, &t1);
    }

    qsort(times, BENCH_MEASURE_FRAMES, sizeof(double), compare_doubles);

    double t_min    = times[0];
    double t_max    = times[BENCH_MEASURE_FRAMES - 1];
    double t_median = times[BENCH_MEASURE_FRAMES / 2];
    double t_sum    = 0.0;
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++)
        t_sum += times[i];
    double t_mean = t_sum / BENCH_MEASURE_FRAMES;

    double budget_pct = (t_median / 16700.0) * 100.0;

    printf("  %-36s [%s]  min=%-6.0f med=%-6.0f mean=%-6.0f max=%-6.0f us  (%.1f%% of 60fps budget)\n",
           e->label, kernel_tag,
           t_min, t_median, t_mean, t_max,
           budget_pct);

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
}

/* --------------------------------------------------------------------------
 * hdr_bench_p010: run one HDR benchmark with P010 format
 * -------------------------------------------------------------------------- */

static void hdr_bench_p010(const hdr_bench_entry_t *e)
{
    test_p010_frame_t frame;
    double times[BENCH_MEASURE_FRAMES];

    if (test_p010_frame_create(&frame, e->width, e->height, PATTERN_RANDOM, 0xdeadbeef) != 0) {
        printf("  %-36s  ERROR: could not allocate P010 test frame\n", e->label);
        return;
    }

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = e->width;
    ctx.src_height     = e->height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_P010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.requested_flags = e->flags;
    ctx.hdr_flags       = e->want_hdr ? e->flags : 0;
    ctx.sdr_flags       = e->want_sdr ? e->flags : 0;
    suppress_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    if (rc < 0) {
        printf("  %-36s  ERROR: fused_hdr_init returned %d\n", e->label, rc);
        test_p010_frame_free(&frame);
        return;
    }

    /* Determine kernel type */
    int any_simd = 0, any_scalar = 0;
    for (int b = 0; b < 8; b++) {
        if (ctx.hdr_outputs[b].plane_y != NULL) {
            if (ctx.hdr_outputs[b].fallback) any_scalar = 1;
            else                              any_simd = 1;
        }
    }
    const char *kernel_tag = (any_simd && !any_scalar) ? "SIMD  " :
                             (!any_simd && any_scalar)  ? "scalar" :
                                                          "mixed ";

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP_FRAMES; i++)
        fused_hdr_run(&ctx, frame.plane_y, frame.plane_uv, NULL);

    /* Measure */
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fused_hdr_run(&ctx, frame.plane_y, frame.plane_uv, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = time_diff_us(&t0, &t1);
    }

    qsort(times, BENCH_MEASURE_FRAMES, sizeof(double), compare_doubles);

    double t_min    = times[0];
    double t_max    = times[BENCH_MEASURE_FRAMES - 1];
    double t_median = times[BENCH_MEASURE_FRAMES / 2];
    double t_sum    = 0.0;
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++)
        t_sum += times[i];
    double t_mean = t_sum / BENCH_MEASURE_FRAMES;

    double budget_pct = (t_median / 16700.0) * 100.0;

    printf("  %-36s [%s]  min=%-6.0f med=%-6.0f mean=%-6.0f max=%-6.0f us  (%.1f%% of 60fps budget)\n",
           e->label, kernel_tag,
           t_min, t_median, t_mean, t_max,
           budget_pct);

    fused_hdr_free(&ctx);
    test_p010_frame_free(&frame);
}

/* --------------------------------------------------------------------------
 * Entry point
 * -------------------------------------------------------------------------- */

void run_hdr_bench_tests(const char *filter)
{
    for (int i = 0; i < HDR_BENCH_CONFIG_COUNT; i++) {
        const hdr_bench_entry_t *e = &hdr_bench_configs[i];
        if (filter != NULL && strstr(e->label, filter) == NULL)
            continue;

        if (e->is_p010)
            hdr_bench_p010(e);
        else
            hdr_bench_i010(e);
    }
}
