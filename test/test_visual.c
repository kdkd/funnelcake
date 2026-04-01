#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* --------------------------------------------------------------------------
 * YUV -> RGB conversion (BT.601)
 * -------------------------------------------------------------------------- */

static void yuv_to_rgb(uint8_t y, uint8_t u, uint8_t v,
                        uint8_t *r, uint8_t *g, uint8_t *b)
{
    int c = (int)y - 16;
    int d = (int)u - 128;
    int e = (int)v - 128;
    int ri = (298 * c + 409 * e + 128) >> 8;
    int gi = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int bi = (298 * c + 516 * d + 128) >> 8;
    *r = (uint8_t)(ri < 0 ? 0 : ri > 255 ? 255 : ri);
    *g = (uint8_t)(gi < 0 ? 0 : gi > 255 ? 255 : gi);
    *b = (uint8_t)(bi < 0 ? 0 : bi > 255 ? 255 : bi);
}

/* --------------------------------------------------------------------------
 * Save a grayscale Y plane as a 1-channel PNG
 * -------------------------------------------------------------------------- */

static void save_plane_png(const char *path, const uint8_t *plane,
                            int w, int h, int stride)
{
    uint8_t *buf = malloc((size_t)w * (size_t)h);
    if (!buf) {
        fprintf(stderr, "  [visual] malloc failed for %s\n", path);
        return;
    }
    for (int row = 0; row < h; row++)
        memcpy(buf + row * w, plane + row * stride, (size_t)w);
    stbi_write_png(path, w, h, 1, buf, w);
    free(buf);
}

/* --------------------------------------------------------------------------
 * Save a YUV420 frame as an RGB PNG (3-channel)
 * -------------------------------------------------------------------------- */

static void save_rgb_png(const char *path,
                          const uint8_t *y_plane, int y_stride,
                          const uint8_t *u_plane, const uint8_t *v_plane, int uv_stride,
                          int width, int height)
{
    uint8_t *rgb = malloc((size_t)width * (size_t)height * 3);
    if (!rgb) {
        fprintf(stderr, "  [visual] malloc failed for %s\n", path);
        return;
    }
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            uint8_t yv = y_plane[row * y_stride + col];
            uint8_t uv = u_plane[(row / 2) * uv_stride + col / 2];
            uint8_t vv = v_plane[(row / 2) * uv_stride + col / 2];
            uint8_t r, g, b;
            yuv_to_rgb(yv, uv, vv, &r, &g, &b);
            int idx = (row * width + col) * 3;
            rgb[idx]     = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }
    stbi_write_png(path, width, height, 3, rgb, width * 3);
    free(rgb);
}

/* --------------------------------------------------------------------------
 * Ratio name from output bit index
 * -------------------------------------------------------------------------- */

static const char *ratio_name(int bit)
{
    static const char *names[] = {
        "1.5x", "2x", "3x", "4x", "6x", "8x", "12x", "16x"
    };
    if (bit >= 0 && bit < 8)
        return names[bit];
    return "?x";
}

/* --------------------------------------------------------------------------
 * Suppress logging helper
 * -------------------------------------------------------------------------- */

static void suppress_log(fused_scaler_ctx_t *ctx)
{
    ctx->log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx->log_errors.target   = FUSED_LOG_SUPPRESS;
}

/* --------------------------------------------------------------------------
 * run_visual_tests
 *
 * Generate PNGs in output/ for representative configs × patterns.
 * -------------------------------------------------------------------------- */

void run_visual_tests(void)
{
    /* Ensure output directory exists */
    mkdir("output", 0755);

    static const struct {
        int w, h;
        uint32_t flags;
    } configs[] = {
        { 1920, 1080, FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X },
        { 1280,  720, FUSED_SCALE_2X   | FUSED_SCALE_4X                   },
    };

    static const test_pattern_t patterns[] = {
        PATTERN_HGRADIENT,
        PATTERN_CHECKERBOARD,
    };

    int total_files = 0;

    for (int ci = 0; ci < (int)(sizeof(configs)/sizeof(configs[0])); ci++) {
        for (int pi = 0; pi < (int)(sizeof(patterns)/sizeof(patterns[0])); pi++) {
            test_pattern_t pat = patterns[pi];
            const char *pat_name = pattern_names[pat];

            test_frame_t frame;
            if (test_frame_create(&frame, configs[ci].w, configs[ci].h, pat, 0) != 0) {
                fprintf(stderr, "  [visual] test_frame_create failed for %dx%d %s\n",
                        configs[ci].w, configs[ci].h, pat_name);
                continue;
            }

            fused_scaler_ctx_t ctx;
            memset(&ctx, 0, sizeof(ctx));
            ctx.src_width     = frame.width;
            ctx.src_height    = frame.height;
            ctx.src_y_stride  = frame.y_stride;
            ctx.src_uv_stride = frame.uv_stride;
            ctx.requested_flags = configs[ci].flags;
            suppress_log(&ctx);

            int rc = fused_scaler_init(&ctx);
            if (rc < 0) {
                fprintf(stderr, "  [visual] fused_scaler_init failed rc=%d for %dx%d\n",
                        rc, configs[ci].w, configs[ci].h);
                test_frame_free(&frame);
                continue;
            }

            fused_scaler_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

            for (int i = 0; i < 8; i++) {
                if (!ctx.outputs[i].plane_y) continue;

                int ow     = ctx.outputs[i].width;
                int oh     = ctx.outputs[i].height;
                int ys     = ctx.outputs[i].y_stride;
                int uvs    = ctx.outputs[i].uv_stride;
                const char *ratio = ratio_name(i);

                char path_y[256];
                char path_rgb[256];

                snprintf(path_y,   sizeof(path_y),
                         "output/%dx%d_%s_Y_%s.png",
                         configs[ci].w, configs[ci].h, ratio, pat_name);
                snprintf(path_rgb, sizeof(path_rgb),
                         "output/%dx%d_%s_RGB_%s.png",
                         configs[ci].w, configs[ci].h, ratio, pat_name);

                save_plane_png(path_y,
                               ctx.outputs[i].plane_y, ow, oh, ys);
                printf("  %s\n", path_y);
                total_files++;

                save_rgb_png(path_rgb,
                             ctx.outputs[i].plane_y, ys,
                             ctx.outputs[i].plane_u,
                             ctx.outputs[i].plane_v, uvs,
                             ow, oh);
                printf("  %s\n", path_rgb);
                total_files++;
            }

            fused_scaler_free(&ctx);
            test_frame_free(&frame);
        }
    }

    printf("\n  Generated %d PNG files in output/\n", total_files);
}
