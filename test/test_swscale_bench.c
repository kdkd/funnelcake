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

#define BENCH_WARMUP_FRAMES       5
#define BENCH_MEASURE_FRAMES_MAX  1000
#define BENCH_MEASURE_TARGET_US   800000.0
#define BENCH_MEASURE_MIN         50
#define BENCH_MEASURE_FRAMES      BENCH_MEASURE_FRAMES_MAX

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
#define MAX_OUTPUTS 12

/* Upscale steps: pow2 cascade × 5 levels, same struct shape but ratio_n > ratio_d. */
static const struct {
    uint32_t    flag;
    int         ratio_n;
    int         ratio_d;
    const char *name;
} up_steps[] = {
    { FUSED_UPSCALE_2X,   2,  1, "up2x"  },
    { FUSED_UPSCALE_4X,   4,  1, "up4x"  },
    { FUSED_UPSCALE_8X,   8,  1, "up8x"  },
    { FUSED_UPSCALE_16X, 16,  1, "up16x" },
    { FUSED_UPSCALE_32X, 32,  1, "up32x" },
};

#define NUM_UP_STEPS ((int)(sizeof(up_steps) / sizeof(up_steps[0])))

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
    /* Defensive rounding for YUV420P: chroma plane must hold ceil(w/2) x
     * ceil(h/2) samples.  libswscale writes ceil(h/2) rows when given an
     * odd luma height, so allocating h/2 under-allocates by one row and
     * corrupts the heap on the next free(). */
    int chroma_w = (w + 1) / 2;
    int chroma_h = (h + 1) / 2;

    out->width     = w;
    out->height    = h;
    out->y_stride  = (w + 31) & ~31;
    out->uv_stride = (chroma_w + 31) & ~31;

    if (posix_memalign((void **)&out->plane_y, 32,
                       (size_t)out->y_stride * h) != 0)
        return -1;
    if (posix_memalign((void **)&out->plane_u, 32,
                       (size_t)out->uv_stride * chroma_h) != 0)
        return -1;
    if (posix_memalign((void **)&out->plane_v, 32,
                       (size_t)out->uv_stride * chroma_h) != 0)
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
    int step_idx;   /* index into steps[] (downscale) or -1 for upscale */
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

/* Collect upscale active steps (and optional 1.5x tail) into the same array.
 * Upscale outputs are larger than source.  The tail is appended after the pow2
 * cascade if up_tail is set. */
static int collect_upscale_steps(int src_w, int src_h, uint32_t up_flags,
                                 int up_tail, active_step_t *out, int max_out)
{
    int n = 0;
    int up_n = 0;
    for (int i = 0; i < NUM_UP_STEPS && n < max_out; i++) {
        if (up_flags & up_steps[i].flag) {
            out[n].step_idx = -1;     /* marker: this is an upscale step */
            out[n].out_w = src_w * up_steps[i].ratio_n;
            out[n].out_h = src_h * up_steps[i].ratio_n;
            n++;
            up_n++;
        }
    }
    if (up_tail && n < max_out) {
        int tail_in_w = (up_n == 0) ? src_w : out[n - 1].out_w;
        int tail_in_h = (up_n == 0) ? src_h : out[n - 1].out_h;
        out[n].step_idx = -1;
        out[n].out_w = tail_in_w * 3 / 2;
        out[n].out_h = tail_in_h * 3 / 2;
        n++;
    }
    return n;
}

/* Compare active steps by reduction ratio - shallowest first (largest output).
 * Used to order the downscale cascade chain. */
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
                         int is_cascade, double *times, int n)
{
    qsort(times, (size_t)n, sizeof(double), compare_doubles);

    double t_min    = times[0];
    double t_max    = times[n - 1];
    double t_median = times[n / 2];

    double t_sum = 0.0;
    for (int i = 0; i < n; i++)
        t_sum += times[i];
    double t_mean = t_sum / (double)n;

    double budget_pct = (t_median / 16700.0) * 100.0;

    printf("  %-54s [swscale]  min=%-6.0f med=%-6.0f mean=%-6.0f max=%-6.0f us  (%.1f%% of 60fps budget)\n",
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
 * Sanity-check helper: sum Y-plane bytes as a quick checksum so we can
 * confirm that each swscale step is actually producing output.  Returns 0
 * if the plane is NULL.
 * -------------------------------------------------------------------------- */
static uint64_t plane_y_checksum(const swscale_output_t *out)
{
    if (!out->plane_y) return 0;
    uint64_t sum = 0;
    for (int y = 0; y < out->height; y++) {
        const uint8_t *row = out->plane_y + (size_t)y * out->y_stride;
        for (int x = 0; x < out->width; x++) sum += row[x];
    }
    return sum;
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
            printf("  %-54s  ERROR: output alloc failed\n", label);
            goto cleanup;
        }

        sws_ctx[s] = sws_getContext(src_w, src_h, AV_PIX_FMT_YUV420P,
                                    asteps[s].out_w, asteps[s].out_h,
                                    AV_PIX_FMT_YUV420P,
                                    SWS_BILINEAR, NULL, NULL, NULL);
        if (!sws_ctx[s]) {
            printf("  %-54s  ERROR: sws_getContext failed\n", label);
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

    /* Verification pass: run each step once, print the number of rows
     * sws_scale claims to have produced and the Y-plane checksum. Enabled
     * by setting the FUNNELCAKE_SWSCALE_VERIFY environment variable. */
    if (getenv("FUNNELCAKE_SWSCALE_VERIFY")) {
        printf("    [verify] %s: %d step(s), src=%dx%d\n",
               label, num_steps, src_w, src_h);
        for (int s = 0; s < num_steps; s++) {
            uint8_t *dst_slices[4] = {
                outputs[s].plane_y, outputs[s].plane_u,
                outputs[s].plane_v, NULL
            };
            int dst_strides[4] = {
                outputs[s].y_stride, outputs[s].uv_stride,
                outputs[s].uv_stride, 0
            };
            int rows = sws_scale(sws_ctx[s], src_slices, src_strides,
                                 0, src_h, dst_slices, dst_strides);
            uint64_t sum = plane_y_checksum(&outputs[s]);
            double avg = (double)sum /
                         ((double)outputs[s].width * outputs[s].height);
            printf("    [verify]   step %d: %dx%d, sws_scale returned %d, "
                   "Y checksum=%llu (avg=%.1f)\n",
                   s, outputs[s].width, outputs[s].height, rows,
                   (unsigned long long)sum, avg);
        }
    }

    /* Warmup and probe */
    struct timespec wt0, wt1;
    clock_gettime(CLOCK_MONOTONIC, &wt0);
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
    clock_gettime(CLOCK_MONOTONIC, &wt1);
    double per_iter_est = time_diff_us(&wt0, &wt1) / (double)BENCH_WARMUP_FRAMES;
    if (per_iter_est <= 0.0) per_iter_est = 1.0;
    int iters = (int)(BENCH_MEASURE_TARGET_US / per_iter_est);
    if (iters < BENCH_MEASURE_MIN)       iters = BENCH_MEASURE_MIN;
    if (iters > BENCH_MEASURE_FRAMES_MAX) iters = BENCH_MEASURE_FRAMES_MAX;

    /* Measure */
    double times[BENCH_MEASURE_FRAMES];
    for (int i = 0; i < iters; i++) {
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

    print_stats(label, base_label, 0, times, iters);

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

    /* Caller is responsible for pre-sorting asteps in cascade order:
     *   downscale -> largest output first (shallowest reduction first)
     *   upscale   -> smallest output first (shallowest growth first) */

    struct SwsContext *sws_ctx[MAX_OUTPUTS] = {0};
    swscale_output_t  outputs[MAX_OUTPUTS];
    memset(outputs, 0, sizeof(outputs));

    /* Allocate output buffers */
    for (int s = 0; s < num_steps; s++) {
        if (output_alloc(&outputs[s], asteps[s].out_w, asteps[s].out_h) != 0) {
            printf("  %-54s  ERROR: output alloc failed\n", label);
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
            printf("  %-54s  ERROR: sws_getContext failed\n", label);
            goto cleanup;
        }
    }

    /* Verification pass (see independent variant above) */
    if (getenv("FUNNELCAKE_SWSCALE_VERIFY")) {
        printf("    [verify] %s: %d step(s) cascaded, src=%dx%d\n",
               label, num_steps, src_w, src_h);
        for (int s = 0; s < num_steps; s++) {
            const uint8_t *in_y, *in_u, *in_v;
            int in_y_stride, in_uv_stride, in_h_local;
            int in_w_local;

            if (s == 0) {
                in_y  = frame->plane_y;
                in_u  = frame->plane_u;
                in_v  = frame->plane_v;
                in_y_stride  = frame->y_stride;
                in_uv_stride = frame->uv_stride;
                in_h_local   = frame->height;
                in_w_local   = frame->width;
            } else {
                in_y  = outputs[s - 1].plane_y;
                in_u  = outputs[s - 1].plane_u;
                in_v  = outputs[s - 1].plane_v;
                in_y_stride  = outputs[s - 1].y_stride;
                in_uv_stride = outputs[s - 1].uv_stride;
                in_h_local   = outputs[s - 1].height;
                in_w_local   = outputs[s - 1].width;
            }

            const uint8_t *in_slices[4]  = { in_y, in_u, in_v, NULL };
            int           in_strides[4]  = { in_y_stride, in_uv_stride, in_uv_stride, 0 };
            uint8_t       *dst_slices[4] = {
                outputs[s].plane_y, outputs[s].plane_u,
                outputs[s].plane_v, NULL
            };
            int dst_strides[4] = {
                outputs[s].y_stride, outputs[s].uv_stride,
                outputs[s].uv_stride, 0
            };
            int rows = sws_scale(sws_ctx[s], in_slices, in_strides,
                                 0, in_h_local, dst_slices, dst_strides);
            uint64_t sum = plane_y_checksum(&outputs[s]);
            double avg = (double)sum /
                         ((double)outputs[s].width * outputs[s].height);
            printf("    [verify]   step %d: in=%dx%d out=%dx%d, "
                   "sws_scale returned %d, Y checksum=%llu (avg=%.1f)\n",
                   s, in_w_local, in_h_local, outputs[s].width, outputs[s].height,
                   rows, (unsigned long long)sum, avg);
        }
    }

    /* Warmup and probe */
    struct timespec wt0, wt1;
    clock_gettime(CLOCK_MONOTONIC, &wt0);
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
    clock_gettime(CLOCK_MONOTONIC, &wt1);
    double per_iter_est = time_diff_us(&wt0, &wt1) / (double)BENCH_WARMUP_FRAMES;
    if (per_iter_est <= 0.0) per_iter_est = 1.0;
    int iters = (int)(BENCH_MEASURE_TARGET_US / per_iter_est);
    if (iters < BENCH_MEASURE_MIN)       iters = BENCH_MEASURE_MIN;
    if (iters > BENCH_MEASURE_FRAMES_MAX) iters = BENCH_MEASURE_FRAMES_MAX;

    /* Measure */
    double times[BENCH_MEASURE_FRAMES];
    for (int i = 0; i < iters; i++) {
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

    print_stats(label, base_label, 1, times, iters);

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
    uint32_t    up_flags;
    int         up_tail;
} bench_entry_t;

static const bench_entry_t bench_configs[] = {
    /* Labels must match the corresponding entries in test/test_bench.c
     * so the comparison table rows line up.  See the comment above the
     * bench_configs array in test_bench.c for the label format. */

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

    /* Upscale-only */
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

    /* Combined down + up */
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

void run_swscale_bench_tests(const char *filter)
{
    for (int i = 0; i < BENCH_CONFIG_COUNT; i++) {
        const bench_entry_t *e = &bench_configs[i];

        int has_down = (e->flags != 0);
        int has_up   = (e->up_flags != 0) || e->up_tail;

        /* Build labels for the variants we will run.  The base label
         * comes from the config (already in the descriptive format used
         * by test_bench.c), and we append "(indep)" / "(cascade)" for
         * the detail-row printouts.
         *
         * Downscale-only: run both independent and cascade - cascade is
         *   the natural pipeline for downscale and shows the real-world
         *   "reuse intermediates" savings.
         * Upscale-only: run only independent.  Cascading an upscale is
         *   strictly worse in both speed and quality, so no sensible
         *   caller would do it.
         * Combined down + up: independent only (cascade ordering doesn't
         *   apply when both directions are mixed in one pipeline). */
        char label_indep[128];
        char label_cascade[128];

        snprintf(label_indep, sizeof(label_indep), "%s (indep)", e->label);
        if (has_down && !has_up) {
            snprintf(label_cascade, sizeof(label_cascade),
                     "%s (cascade)", e->label);
        } else {
            label_cascade[0] = 0;  /* upscale-only or combined */
        }

        int match_indep   = (filter == NULL || strstr(label_indep, filter) != NULL);
        int match_cascade = (label_cascade[0] != 0) &&
                            (filter == NULL || strstr(label_cascade, filter) != NULL);
        if (!match_indep && !match_cascade)
            continue;

        /* Create test frame */
        test_frame_t frame;
        if (test_frame_create(&frame, e->width, e->height,
                              PATTERN_RANDOM, 0xdeadbeef) != 0) {
            printf("  %-54s  ERROR: could not allocate test frame\n", e->label);
            continue;
        }

        /* Collect active steps from both downscale and upscale flags. */
        active_step_t asteps[MAX_OUTPUTS];
        int num_steps = 0;
        if (has_down) {
            num_steps += collect_active_steps(e->width, e->height, e->flags,
                                              asteps + num_steps,
                                              MAX_OUTPUTS - num_steps);
        }
        if (has_up) {
            num_steps += collect_upscale_steps(e->width, e->height,
                                               e->up_flags, e->up_tail,
                                               asteps + num_steps,
                                               MAX_OUTPUTS - num_steps);
        }
        if (num_steps == 0) {
            test_frame_free(&frame);
            continue;
        }

        /* Independent benchmark (each output reads source) */
        if (match_indep) {
            active_step_t indep_steps[MAX_OUTPUTS];
            memcpy(indep_steps, asteps, sizeof(active_step_t) * (size_t)num_steps);
            bench_swscale_independent(label_indep, e->label, &frame,
                                      indep_steps, num_steps);
        }

        /* Cascaded benchmark - downscale only.  Upscale and combined
         * cases have label_cascade[0]=0 and skip this branch. */
        if (match_cascade && num_steps > 1) {
            active_step_t cascade_steps[MAX_OUTPUTS];
            memcpy(cascade_steps, asteps, sizeof(active_step_t) * (size_t)num_steps);
            qsort(cascade_steps, (size_t)num_steps, sizeof(active_step_t),
                  compare_by_reduction);
            bench_swscale_cascaded(label_cascade, e->label, &frame,
                                   cascade_steps, num_steps);
        }

        test_frame_free(&frame);
    }
}

#else /* !HAVE_LIBSWSCALE */

void run_swscale_bench_tests(const char *filter __attribute__((unused)))
{
    (void)filter;
    printf("  (libswscale not available - skipping comparison benchmarks)\n");
    printf("  Install ffmpeg dev libraries and rebuild to enable.\n");
}

#endif /* HAVE_LIBSWSCALE */
