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

/* --------------------------------------------------------------------------
 * bench_config: run one benchmark configuration
 * -------------------------------------------------------------------------- */

static void bench_config(const char *label, int width, int height,
                         int chroma_format, uint32_t flags)
{
    test_frame_t frame;
    fused_scaler_ctx_t ctx;
    double times[BENCH_MEASURE_FRAMES];
    int rc;

    /* Build a random test frame */
    if (test_frame_create_ex(&frame, width, height, chroma_format,
                             PATTERN_RANDOM, 0xdeadbeef) != 0) {
        printf("  %-28s  ERROR: could not allocate test frame\n", label);
        return;
    }

    /* Init scaler — suppress log output during benchmarks */
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width    = width;
    ctx.src_height   = height;
    ctx.src_y_stride  = frame.y_stride;
    ctx.src_uv_stride = frame.uv_stride;
    ctx.chroma_format = chroma_format;
    ctx.requested_flags = flags;
    ctx.log_errors.target   = FUSED_LOG_SUPPRESS;
    ctx.log_warnings.target = FUSED_LOG_SUPPRESS;

    rc = fused_scaler_init(&ctx);
    if (rc < 0) {
        printf("  %-28s  ERROR: fused_scaler_init returned %d\n", label, rc);
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
    const char *kernel_tag = (any_simd && !any_scalar) ? "SIMD  " :
                             (!any_simd && any_scalar)  ? "scalar" :
                                                          "mixed ";

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP_FRAMES; i++) {
        fused_scaler_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);
    }

    /* Measure */
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fused_scaler_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = time_diff_us(&t0, &t1);
    }

    /* Sort to find min / median / max */
    qsort(times, BENCH_MEASURE_FRAMES, sizeof(double), compare_doubles);

    double t_min    = times[0];
    double t_max    = times[BENCH_MEASURE_FRAMES - 1];
    double t_median = times[BENCH_MEASURE_FRAMES / 2];

    double t_sum = 0.0;
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++)
        t_sum += times[i];
    double t_mean = t_sum / BENCH_MEASURE_FRAMES;

    /* 60 fps budget = 16.7 ms = 16700 us */
    double budget_pct = (t_median / 16700.0) * 100.0;

    printf("  %-28s [%s]  min=%-6.0f med=%-6.0f mean=%-6.0f max=%-6.0f us  (%.1f%% of 60fps budget)\n",
           label, kernel_tag,
           t_min, t_median, t_mean, t_max,
           budget_pct);

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
    int         chroma_format;
    uint32_t    flags;
} bench_entry_t;

static const bench_entry_t bench_configs[] = {
    { "640x360 pow2 2x 420",     640,  360,  FUSED_CHROMA_420, FUSED_SCALE_2X },
    { "960x540 thirds 420",      960,  540,  FUSED_CHROMA_420, FUSED_SCALE_1_5X | FUSED_SCALE_3X },
    { "1280x720 pow2 420",       1280, 720,  FUSED_CHROMA_420, FUSED_SCALE_2X  | FUSED_SCALE_4X },
    { "1280x720 pow2 422",       1280, 720,  FUSED_CHROMA_422, FUSED_SCALE_2X  | FUSED_SCALE_4X },
    { "1920x1080 thirds 420",    1920, 1080, FUSED_CHROMA_420, FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X },
    { "1920x1080 thirds 422",    1920, 1080, FUSED_CHROMA_422, FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X },
    { "2560x1440 pow2 420",      2560, 1440, FUSED_CHROMA_420, FUSED_SCALE_2X  | FUSED_SCALE_4X | FUSED_SCALE_8X },
    { "3840x2160 thirds 420",    3840, 2160, FUSED_CHROMA_420, FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X | FUSED_SCALE_12X },
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
        bench_config(e->label, e->width, e->height, e->chroma_format, e->flags);
    }
}
