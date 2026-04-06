#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef HAVE_LIBSWSCALE

#include <libswscale/swscale.h>
#include <libavutil/pixfmt.h>

#define BENCH_WARMUP_FRAMES  100
#define BENCH_MEASURE_FRAMES 1000

/* --------------------------------------------------------------------------
 * Step descriptor table
 * -------------------------------------------------------------------------- */

static const struct {
    uint32_t    flag;
    int         ratio_n;
    int         ratio_d;
    const char *name;
} steps[] = {
    { FUSED_SCALE_1_5X,  2, 3,  "1.5x" },
    { FUSED_SCALE_2X,    1, 2,  "2x"   },
    { FUSED_SCALE_3X,    1, 3,  "3x"   },
    { FUSED_SCALE_4X,    1, 4,  "4x"   },
    { FUSED_SCALE_6X,    1, 6,  "6x"   },
    { FUSED_SCALE_8X,    1, 8,  "8x"   },
    { FUSED_SCALE_12X,   1, 12, "12x"  },
    { FUSED_SCALE_16X,   1, 16, "16x"  },
};

#define NUM_STEPS ((int)(sizeof(steps) / sizeof(steps[0])))
#define MAX_OUTPUTS 8

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
 * Output buffer for one scale step
 * -------------------------------------------------------------------------- */

typedef struct {
    int      width;
    int      height;
    int      y_stride;
    int      uv_stride;
    uint8_t *plane_y;
    uint8_t *plane_u;
    uint8_t *plane_v;
} swscale_output_t;

static int output_alloc(swscale_output_t *out, int w, int h)
{
    out->width     = w;
    out->height    = h;
    out->y_stride  = (w + 31) & ~31;
    out->uv_stride = (w / 2 + 31) & ~31;

    if (posix_memalign((void **)&out->plane_y, 32,
                       (size_t)out->y_stride * h) != 0)
        return -1;
    if (posix_memalign((void **)&out->plane_u, 32,
                       (size_t)out->uv_stride * (h / 2)) != 0)
        return -1;
    if (posix_memalign((void **)&out->plane_v, 32,
                       (size_t)out->uv_stride * (h / 2)) != 0)
        return -1;

    return 0;
}

static void output_free(swscale_output_t *out)
{
    free(out->plane_y);
    free(out->plane_u);
    free(out->plane_v);
    memset(out, 0, sizeof(*out));
}

/* --------------------------------------------------------------------------
 * Collect active steps for a given flag bitmask
 * -------------------------------------------------------------------------- */

typedef struct {
    int step_idx;   /* index into steps[] */
    int out_w;
    int out_h;
} active_step_t;

static int collect_active_steps(int src_w, int src_h, uint32_t flags,
                                active_step_t *out, int max_out)
{
    int n = 0;
    for (int i = 0; i < NUM_STEPS && n < max_out; i++) {
        if (flags & steps[i].flag) {
            out[n].step_idx = i;
            out[n].out_w = src_w * steps[i].ratio_n / steps[i].ratio_d;
            out[n].out_h = src_h * steps[i].ratio_n / steps[i].ratio_d;
            n++;
        }
    }
    return n;
}

/* Compare active steps by reduction ratio — shallowest first (largest output).
 * Used to order the cascade chain. */
static int compare_by_reduction(const void *a, const void *b)
{
    const active_step_t *sa = (const active_step_t *)a;
    const active_step_t *sb = (const active_step_t *)b;
    /* Larger output area = shallower reduction = comes first */
    int area_a = sa->out_w * sa->out_h;
    int area_b = sb->out_w * sb->out_h;
    if (area_a > area_b) return -1;
    if (area_a < area_b) return  1;
    return 0;
}

/* --------------------------------------------------------------------------
 * Print stats in the same format as test_bench.c
 * -------------------------------------------------------------------------- */

/* is_cascade: 0 = independent, 1 = cascaded.
 * base_label: the config label (e.g. "1920x1080 thirds") for comparison table lookup. */
static void print_stats(const char *label, const char *base_label,
                         int is_cascade, double *times)
{
    qsort(times, BENCH_MEASURE_FRAMES, sizeof(double), compare_doubles);

    double t_min    = times[0];
    double t_max    = times[BENCH_MEASURE_FRAMES - 1];
    double t_median = times[BENCH_MEASURE_FRAMES / 2];

    double t_sum = 0.0;
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++)
        t_sum += times[i];
    double t_mean = t_sum / BENCH_MEASURE_FRAMES;

    double budget_pct = (t_median / 16700.0) * 100.0;

    printf("  %-35s [swscale]  min=%-6.0f med=%-6.0f mean=%-6.0f max=%-6.0f us  (%.1f%% of 60fps budget)\n",
           label, t_min, t_median, t_mean, t_max, budget_pct);

    /* Record in comparison table */
    for (int ci = 0; ci < g_bench_comparison_count; ci++) {
        if (strcmp(g_bench_comparison[ci].label, base_label) == 0) {
            if (is_cascade)
                g_bench_comparison[ci].swscale_cascade_med = t_median;
            else
                g_bench_comparison[ci].swscale_indep_med = t_median;
            return;
        }
    }
    /* If no funnelcake entry exists yet, create one */
    if (g_bench_comparison_count < BENCH_MAX_CONFIGS) {
        bench_comparison_t *c = &g_bench_comparison[g_bench_comparison_count++];
        c->label = base_label;
        c->funnelcake_med = 0;
        if (is_cascade) {
            c->swscale_indep_med = 0;
            c->swscale_cascade_med = t_median;
        } else {
            c->swscale_indep_med = t_median;
            c->swscale_cascade_med = 0;
        }
    }
}

/* --------------------------------------------------------------------------
 * bench_swscale_independent: each output scales from the source directly
 * -------------------------------------------------------------------------- */

static void bench_swscale_independent(const char *label, const char *base_label,
                                      const test_frame_t *frame,
                                      active_step_t *asteps, int num_steps)
{
    int src_w = frame->width;
    int src_h = frame->height;

    struct SwsContext *sws_ctx[MAX_OUTPUTS] = {0};
    swscale_output_t  outputs[MAX_OUTPUTS];
    memset(outputs, 0, sizeof(outputs));

    /* Create contexts and allocate output buffers */
    for (int s = 0; s < num_steps; s++) {
        if (output_alloc(&outputs[s], asteps[s].out_w, asteps[s].out_h) != 0) {
            printf("  %-35s  ERROR: output alloc failed\n", label);
            goto cleanup;
        }

        sws_ctx[s] = sws_getContext(src_w, src_h, AV_PIX_FMT_YUV420P,
                                    asteps[s].out_w, asteps[s].out_h,
                                    AV_PIX_FMT_YUV420P,
                                    SWS_BILINEAR, NULL, NULL, NULL);
        if (!sws_ctx[s]) {
            printf("  %-35s  ERROR: sws_getContext failed\n", label);
            goto cleanup;
        }
    }

    /* Set up source slices/strides */
    const uint8_t *src_slices[4] = {
        frame->plane_y, frame->plane_u, frame->plane_v, NULL
    };
    int src_strides[4] = {
        frame->y_stride, frame->uv_stride, frame->uv_stride, 0
    };

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP_FRAMES; i++) {
        for (int s = 0; s < num_steps; s++) {
            uint8_t *dst_slices[4] = {
                outputs[s].plane_y, outputs[s].plane_u,
                outputs[s].plane_v, NULL
            };
            int dst_strides[4] = {
                outputs[s].y_stride, outputs[s].uv_stride,
                outputs[s].uv_stride, 0
            };
            sws_scale(sws_ctx[s], src_slices, src_strides, 0, src_h,
                      dst_slices, dst_strides);
        }
    }

    /* Measure */
    double times[BENCH_MEASURE_FRAMES];
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int s = 0; s < num_steps; s++) {
            uint8_t *dst_slices[4] = {
                outputs[s].plane_y, outputs[s].plane_u,
                outputs[s].plane_v, NULL
            };
            int dst_strides[4] = {
                outputs[s].y_stride, outputs[s].uv_stride,
                outputs[s].uv_stride, 0
            };
            sws_scale(sws_ctx[s], src_slices, src_strides, 0, src_h,
                      dst_slices, dst_strides);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = time_diff_us(&t0, &t1);
    }

    print_stats(label, base_label, 0, times);

cleanup:
    for (int s = 0; s < num_steps; s++) {
        if (sws_ctx[s])
            sws_freeContext(sws_ctx[s]);
        output_free(&outputs[s]);
    }
}

/* --------------------------------------------------------------------------
 * bench_swscale_cascaded: each step reads from the previous step's output
 * -------------------------------------------------------------------------- */

static void bench_swscale_cascaded(const char *label, const char *base_label,
                                   const test_frame_t *frame,
                                   active_step_t *asteps, int num_steps)
{
    int src_w = frame->width;
    int src_h = frame->height;

    /* Sort steps from shallowest to deepest reduction (largest output first) */
    qsort(asteps, (size_t)num_steps, sizeof(active_step_t), compare_by_reduction);

    struct SwsContext *sws_ctx[MAX_OUTPUTS] = {0};
    swscale_output_t  outputs[MAX_OUTPUTS];
    memset(outputs, 0, sizeof(outputs));

    /* Allocate output buffers */
    for (int s = 0; s < num_steps; s++) {
        if (output_alloc(&outputs[s], asteps[s].out_w, asteps[s].out_h) != 0) {
            printf("  %-35s  ERROR: output alloc failed\n", label);
            goto cleanup;
        }
    }

    /* Create SwsContext for each cascade step.
     * Step 0 reads from source; step N reads from output N-1. */
    for (int s = 0; s < num_steps; s++) {
        int in_w = (s == 0) ? src_w : outputs[s - 1].width;
        int in_h = (s == 0) ? src_h : outputs[s - 1].height;

        sws_ctx[s] = sws_getContext(in_w, in_h, AV_PIX_FMT_YUV420P,
                                    asteps[s].out_w, asteps[s].out_h,
                                    AV_PIX_FMT_YUV420P,
                                    SWS_BILINEAR, NULL, NULL, NULL);
        if (!sws_ctx[s]) {
            printf("  %-35s  ERROR: sws_getContext failed\n", label);
            goto cleanup;
        }
    }

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP_FRAMES; i++) {
        for (int s = 0; s < num_steps; s++) {
            const uint8_t *in_y, *in_u, *in_v;
            int in_y_stride, in_uv_stride, in_h;

            if (s == 0) {
                in_y  = frame->plane_y;
                in_u  = frame->plane_u;
                in_v  = frame->plane_v;
                in_y_stride  = frame->y_stride;
                in_uv_stride = frame->uv_stride;
                in_h  = frame->height;
            } else {
                in_y  = outputs[s - 1].plane_y;
                in_u  = outputs[s - 1].plane_u;
                in_v  = outputs[s - 1].plane_v;
                in_y_stride  = outputs[s - 1].y_stride;
                in_uv_stride = outputs[s - 1].uv_stride;
                in_h  = outputs[s - 1].height;
            }

            const uint8_t *in_slices[4] = { in_y, in_u, in_v, NULL };
            int in_strides[4] = { in_y_stride, in_uv_stride, in_uv_stride, 0 };
            uint8_t *dst_slices[4] = {
                outputs[s].plane_y, outputs[s].plane_u,
                outputs[s].plane_v, NULL
            };
            int dst_strides[4] = {
                outputs[s].y_stride, outputs[s].uv_stride,
                outputs[s].uv_stride, 0
            };
            sws_scale(sws_ctx[s], in_slices, in_strides, 0, in_h,
                      dst_slices, dst_strides);
        }
    }

    /* Measure */
    double times[BENCH_MEASURE_FRAMES];
    for (int i = 0; i < BENCH_MEASURE_FRAMES; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int s = 0; s < num_steps; s++) {
            const uint8_t *in_y, *in_u, *in_v;
            int in_y_stride, in_uv_stride, in_h;

            if (s == 0) {
                in_y  = frame->plane_y;
                in_u  = frame->plane_u;
                in_v  = frame->plane_v;
                in_y_stride  = frame->y_stride;
                in_uv_stride = frame->uv_stride;
                in_h  = frame->height;
            } else {
                in_y  = outputs[s - 1].plane_y;
                in_u  = outputs[s - 1].plane_u;
                in_v  = outputs[s - 1].plane_v;
                in_y_stride  = outputs[s - 1].y_stride;
                in_uv_stride = outputs[s - 1].uv_stride;
                in_h  = outputs[s - 1].height;
            }

            const uint8_t *in_slices[4] = { in_y, in_u, in_v, NULL };
            int in_strides[4] = { in_y_stride, in_uv_stride, in_uv_stride, 0 };
            uint8_t *dst_slices[4] = {
                outputs[s].plane_y, outputs[s].plane_u,
                outputs[s].plane_v, NULL
            };
            int dst_strides[4] = {
                outputs[s].y_stride, outputs[s].uv_stride,
                outputs[s].uv_stride, 0
            };
            sws_scale(sws_ctx[s], in_slices, in_strides, 0, in_h,
                      dst_slices, dst_strides);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[i] = time_diff_us(&t0, &t1);
    }

    print_stats(label, base_label, 1, times);

cleanup:
    for (int s = 0; s < num_steps; s++) {
        if (sws_ctx[s])
            sws_freeContext(sws_ctx[s]);
        output_free(&outputs[s]);
    }
}

/* --------------------------------------------------------------------------
 * Benchmark configurations
 * -------------------------------------------------------------------------- */

typedef struct {
    const char *label;
    int         width;
    int         height;
    uint32_t    flags;
} bench_entry_t;

static const bench_entry_t bench_configs[] = {
    { "640x360 pow2 2x",     640,  360,  FUSED_SCALE_2X },
    { "960x540 thirds",      960,  540,  FUSED_SCALE_1_5X | FUSED_SCALE_3X },
    { "1280x720 pow2",       1280, 720,  FUSED_SCALE_2X  | FUSED_SCALE_4X },
    { "1920x1080 thirds",    1920, 1080, FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X },
    { "2560x1440 pow2",      2560, 1440, FUSED_SCALE_2X  | FUSED_SCALE_4X | FUSED_SCALE_8X },
    { "3840x2160 thirds",    3840, 2160, FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X | FUSED_SCALE_12X },
};

#define BENCH_CONFIG_COUNT ((int)(sizeof(bench_configs) / sizeof(bench_configs[0])))

/* --------------------------------------------------------------------------
 * Determine family name from flags
 * -------------------------------------------------------------------------- */

static const char *family_name(uint32_t flags)
{
    if (flags & FUSED_SCALE_POW2_MASK)
        return "pow2";
    return "thirds";
}

/* --------------------------------------------------------------------------
 * Entry point
 * -------------------------------------------------------------------------- */

void run_swscale_bench_tests(const char *filter)
{
    for (int i = 0; i < BENCH_CONFIG_COUNT; i++) {
        const bench_entry_t *e = &bench_configs[i];

        /* Build labels for independent and cascaded variants */
        char label_indep[128];
        char label_cascade[128];
        snprintf(label_indep,   sizeof(label_indep),
                 "%dx%d %s (indep)", e->width, e->height, family_name(e->flags));
        snprintf(label_cascade, sizeof(label_cascade),
                 "%dx%d %s (cascade)", e->width, e->height, family_name(e->flags));

        /* Apply filter */
        int match_indep   = (filter == NULL || strstr(label_indep, filter) != NULL);
        int match_cascade = (filter == NULL || strstr(label_cascade, filter) != NULL);
        if (!match_indep && !match_cascade)
            continue;

        /* Create test frame */
        test_frame_t frame;
        if (test_frame_create(&frame, e->width, e->height,
                              PATTERN_RANDOM, 0xdeadbeef) != 0) {
            printf("  %-35s  ERROR: could not allocate test frame\n", e->label);
            continue;
        }

        /* Collect active steps */
        active_step_t asteps[MAX_OUTPUTS];
        int num_steps = collect_active_steps(e->width, e->height, e->flags,
                                             asteps, MAX_OUTPUTS);
        if (num_steps == 0) {
            test_frame_free(&frame);
            continue;
        }

        /* Independent benchmark */
        if (match_indep) {
            active_step_t indep_steps[MAX_OUTPUTS];
            memcpy(indep_steps, asteps, sizeof(active_step_t) * (size_t)num_steps);
            bench_swscale_independent(label_indep, e->label, &frame, indep_steps, num_steps);
        }

        /* Cascaded benchmark (only when more than one step) */
        if (match_cascade && num_steps > 1) {
            active_step_t cascade_steps[MAX_OUTPUTS];
            memcpy(cascade_steps, asteps, sizeof(active_step_t) * (size_t)num_steps);
            bench_swscale_cascaded(label_cascade, e->label, &frame, cascade_steps, num_steps);
        }

        test_frame_free(&frame);
    }
}

#else /* !HAVE_LIBSWSCALE */

void run_swscale_bench_tests(const char *filter __attribute__((unused)))
{
    (void)filter;
    printf("  (libswscale not available — skipping comparison benchmarks)\n");
    printf("  Install ffmpeg dev libraries and rebuild to enable.\n");
}

#endif /* HAVE_LIBSWSCALE */
