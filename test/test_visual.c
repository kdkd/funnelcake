#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* --------------------------------------------------------------------------
 * All PNG generation uses ffmpeg for correct color space handling.
 * This avoids manual YUV->RGB matrix conversion and ensures the display
 * colorimetry matches the actual data:
 *   SDR 8-bit:  yuv420p, BT.709 primaries/transfer/matrix, full range
 *   SDR (tone-mapped): yuv420p, BT.709, full range
 *   HDR 10-bit: yuv420p10le, BT.2020/PQ
 * -------------------------------------------------------------------------- */

/* Write raw 8-bit I420 planes to a temp file, convert to PNG via ffmpeg.
 * color_params is the ffmpeg color space flags string. */
static int save_sdr_png_via_ffmpeg(const char *png_path,
                                    const uint8_t *y_plane, int y_stride,
                                    const uint8_t *u_plane, const uint8_t *v_plane,
                                    int uv_stride, int width, int height,
                                    const char *color_params)
{
    char tmp_path[280];
    snprintf(tmp_path, sizeof(tmp_path), "%s.yuv", png_path);

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) return 0;

    int chroma_w = width / 2;
    int chroma_h = height / 2;

    for (int row = 0; row < height; row++)
        fwrite(y_plane + row * y_stride, 1, (size_t)width, fp);
    for (int row = 0; row < chroma_h; row++)
        fwrite(u_plane + row * uv_stride, 1, (size_t)chroma_w, fp);
    for (int row = 0; row < chroma_h; row++)
        fwrite(v_plane + row * uv_stride, 1, (size_t)chroma_w, fp);
    fclose(fp);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -f rawvideo -pix_fmt yuv420p -s %dx%d "
             "-color_range pc %s "
             "-i \"%s\" -pix_fmt rgb24 \"%s\" 2>/dev/null",
             width, height, color_params, tmp_path, png_path);

    int ret = system(cmd);
    remove(tmp_path);
    return ret == 0;
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

static void suppress_hdr_log(fused_hdr_ctx_t *ctx)
{
    ctx->log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx->log_errors.target   = FUSED_LOG_SUPPRESS;
}

/* --------------------------------------------------------------------------
 * Check if ffmpeg is available (needed for HDR visual output)
 * -------------------------------------------------------------------------- */

static int have_ffmpeg(void)
{
    return system("command -v ffmpeg >/dev/null 2>&1") == 0;
}


/* --------------------------------------------------------------------------
 * Load a raw I010 frame from a .yuv file (must know dimensions).
 * Returns 0 on success, -1 on failure.
 * -------------------------------------------------------------------------- */

static int load_raw_i010(const char *path, int width, int height,
                          test_hdr_frame_t *frame)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    if (test_hdr_frame_create(frame, width, height, PATTERN_SOLID, 0) != 0) {
        fclose(fp);
        return -1;
    }

    int chroma_w = width / 2;
    int chroma_h = height / 2;
    int y_el_stride = frame->y_stride / 2;
    int uv_el_stride = frame->uv_stride / 2;

    /* Read Y */
    for (int row = 0; row < height; row++) {
        if (fread(frame->plane_y + row * y_el_stride,
                  sizeof(uint16_t), (size_t)width, fp) != (size_t)width) {
            test_hdr_frame_free(frame);
            fclose(fp);
            return -1;
        }
    }
    /* Read U */
    for (int row = 0; row < chroma_h; row++) {
        if (fread(frame->plane_u + row * uv_el_stride,
                  sizeof(uint16_t), (size_t)chroma_w, fp) != (size_t)chroma_w) {
            test_hdr_frame_free(frame);
            fclose(fp);
            return -1;
        }
    }
    /* Read V */
    for (int row = 0; row < chroma_h; row++) {
        if (fread(frame->plane_v + row * uv_el_stride,
                  sizeof(uint16_t), (size_t)chroma_w, fp) != (size_t)chroma_w) {
            test_hdr_frame_free(frame);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

/* --------------------------------------------------------------------------
 * Run one HDR visual test: scale an HDR frame, save HDR outputs as 16-bit
 * PNGs via ffmpeg, save SDR (tone-mapped) outputs as 8-bit PNGs via stb.
 * -------------------------------------------------------------------------- */

static int run_hdr_visual_config(const char *label,
                                  test_hdr_frame_t *frame,
                                  uint32_t flags, int src_transfer)
{
    int total = 0;

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width     = frame->width;
    ctx.src_height    = frame->height;
    ctx.src_y_stride  = frame->y_stride;
    ctx.src_uv_stride = frame->uv_stride;
    ctx.src_format    = FUSED_PIX_I010;
    ctx.src_transfer  = src_transfer;
    ctx.requested_flags = flags;
    ctx.hdr_flags       = 0;      /* No HDR output (PNG can't display HDR) */
    ctx.sdr_flags       = flags;  /* SDR tone-mapped output for visual verification */
    ctx.tonemap.curve   = FUSED_TONEMAP_HABLE;
    suppress_hdr_log(&ctx);

    int rc = fused_hdr_init(&ctx);
    if (rc < 0) {
        fprintf(stderr, "  [visual] fused_hdr_init failed rc=%d for %s\n", rc, label);
        return 0;
    }

    fused_hdr_run(&ctx, frame->plane_y, frame->plane_u, frame->plane_v);

    for (int i = 0; i < 8; i++) {
        const char *ratio = ratio_name(i);

        /* SDR (tone-mapped) output only.  HDR output is skipped because
         * PNG is an SDR format - any conversion from PQ to displayable
         * RGB is itself a tone mapping, making the "HDR" PNG misleading.
         * The SDR output demonstrates the actual tone mapping result. */
        if (ctx.sdr_outputs[i].plane_y) {
            char path[256];
            snprintf(path, sizeof(path), "output/SDR_%s_%s.png", label, ratio);
            if (save_sdr_png_via_ffmpeg(path,
                    ctx.sdr_outputs[i].plane_y, ctx.sdr_outputs[i].y_stride,
                    ctx.sdr_outputs[i].plane_u,
                    ctx.sdr_outputs[i].plane_v, ctx.sdr_outputs[i].uv_stride,
                    ctx.sdr_outputs[i].width, ctx.sdr_outputs[i].height,
                    "-color_primaries bt709 -color_trc bt709 -colorspace bt709")) {
                printf("  %s\n", path);
                total++;
            }
        }
    }

    fused_hdr_free(&ctx);
    return total;
}

/* --------------------------------------------------------------------------
 * run_visual_tests
 *
 * Generate PNGs in output/ for representative configs × patterns.
 * All output uses ffmpeg for correct colorimetry.
 * HDR tests require ffmpeg - skipped entirely if ffmpeg is not found.
 * -------------------------------------------------------------------------- */

void run_visual_tests(void)
{
    if (!have_ffmpeg()) {
        printf("\n  ffmpeg not found - skipping all visual output.\n");
        printf("  Install ffmpeg and re-run 'make visual' for PNG output.\n");
        return;
    }

    /* Ensure output directory exists */
    mkdir("output", 0755);

    int total_files = 0;

    /* ======================================================================
     * SDR visual tests (existing)
     * ====================================================================== */

    printf("\n  --- SDR visual output ---\n\n");

    static const struct {
        int w, h;
        uint32_t flags;
    } sdr_configs[] = {
        { 1920, 1080, FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X },
        { 1280,  720, FUSED_SCALE_2X   | FUSED_SCALE_4X                   },
    };

    static const test_pattern_t sdr_patterns[] = {
        PATTERN_HGRADIENT,
        PATTERN_CHECKERBOARD,
    };

    for (int ci = 0; ci < (int)(sizeof(sdr_configs)/sizeof(sdr_configs[0])); ci++) {
        for (int pi = 0; pi < (int)(sizeof(sdr_patterns)/sizeof(sdr_patterns[0])); pi++) {
            test_pattern_t pat = sdr_patterns[pi];
            const char *pat_name = pattern_names[pat];

            test_frame_t frame;
            if (test_frame_create(&frame, sdr_configs[ci].w, sdr_configs[ci].h, pat, 0) != 0) {
                fprintf(stderr, "  [visual] test_frame_create failed for %dx%d %s\n",
                        sdr_configs[ci].w, sdr_configs[ci].h, pat_name);
                continue;
            }

            fused_scaler_ctx_t ctx;
            memset(&ctx, 0, sizeof(ctx));
            ctx.src_width     = frame.width;
            ctx.src_height    = frame.height;
            ctx.src_y_stride  = frame.y_stride;
            ctx.src_uv_stride = frame.uv_stride;
            ctx.requested_flags = sdr_configs[ci].flags;
            suppress_log(&ctx);

            int rc = fused_scaler_init(&ctx);
            if (rc < 0) {
                fprintf(stderr, "  [visual] fused_scaler_init failed rc=%d for %dx%d\n",
                        rc, sdr_configs[ci].w, sdr_configs[ci].h);
                test_frame_free(&frame);
                continue;
            }

            fused_scaler_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

            for (int i = 0; i < 8; i++) {
                if (!ctx.outputs[i].plane_y) continue;

                int ow  = ctx.outputs[i].width;
                int oh  = ctx.outputs[i].height;
                int ys  = ctx.outputs[i].y_stride;
                int uvs = ctx.outputs[i].uv_stride;
                const char *ratio = ratio_name(i);

                char path_rgb[256];
                snprintf(path_rgb, sizeof(path_rgb),
                         "output/%dx%d_%s_%s.png",
                         sdr_configs[ci].w, sdr_configs[ci].h, ratio, pat_name);

                if (save_sdr_png_via_ffmpeg(path_rgb,
                        ctx.outputs[i].plane_y, ys,
                        ctx.outputs[i].plane_u,
                        ctx.outputs[i].plane_v, uvs,
                        ow, oh,
                        "-color_primaries bt709 -color_trc bt709 -colorspace bt709")) {
                    printf("  %s\n", path_rgb);
                    total_files++;
                }
            }

            fused_scaler_free(&ctx);
            test_frame_free(&frame);
        }
    }

    /* ======================================================================
     * HDR visual tests (requires ffmpeg)
     * ====================================================================== */

    {
        printf("\n  --- SDR tone-mapped output (generated HDR patterns) ---\n\n");

        uint32_t hdr_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
        static const struct {
            test_pattern_t pat;
            const char *label;
        } hdr_patterns[] = {
            { PATTERN_PQ_RAMP,         "1080p_pq_ramp" },
            { PATTERN_HDR_HIGHLIGHT,   "1080p_highlight" },
        };

        for (int pi = 0; pi < (int)(sizeof(hdr_patterns)/sizeof(hdr_patterns[0])); pi++) {
            test_hdr_frame_t frame;
            if (test_hdr_frame_create(&frame, 1920, 1080, hdr_patterns[pi].pat, 0) != 0) {
                fprintf(stderr, "  [visual] test_hdr_frame_create failed for %s\n",
                        hdr_patterns[pi].label);
                continue;
            }
            total_files += run_hdr_visual_config(hdr_patterns[pi].label,
                                                  &frame, hdr_flags, FUSED_TRC_PQ);
            test_hdr_frame_free(&frame);
        }

        /* Sample files from test/samples/ (if present from 'make fetch-samples').
         *
         * The synthetic ffmpeg samples (bars, gradients, mandelbrot) are
         * gamma-space 10-bit - NOT actually PQ/HLG encoded.  They're useful
         * for verifying the 10-bit scaling pipeline (HDR-only output) but
         * tone mapping them produces wrong colors because the PQ EOTF
         * misinterprets gamma values.
         *
         * Only truly PQ-encoded content (like the Jellyfin HDR10 clip or
         * DVB test content) produces correct tone-mapped SDR output.
         *
         * Synthetic samples: scale with HDR-only output (no tone mapping).
         * Real HDR samples: scale with both HDR and SDR output. */
        static const struct {
            const char *filename;
            int w, h;
            int transfer;
            const char *label;
        } samples[] = {
            /* Synthetic PQ-encoded via zscale */
            { "test/samples/bars_1080p_i010_pq.yuv",          1920, 1080, FUSED_TRC_PQ,  "sample_bars" },
            { "test/samples/gradients_1080p_i010_pq.yuv",     1920, 1080, FUSED_TRC_PQ,  "sample_gradients" },
            { "test/samples/mandelbrot_1080p_i010_pq.yuv",    1920, 1080, FUSED_TRC_PQ,  "sample_mandelbrot" },
            { "test/samples/testsrc2_1080p_i010_pq.yuv",      1920, 1080, FUSED_TRC_PQ,  "sample_testsrc2" },
            /* Real HDR content (haasn/hdr-tests) */
            { "test/samples/hdr_colorbars_1080p_i010_pq.yuv", 1920, 1080, FUSED_TRC_PQ,  "sample_hdr_colorbars" },
            { "test/samples/hdr_snow_i010_pq.yuv",            3840, 2160, FUSED_TRC_HLG, "sample_hdr_snow_hlg" },
            /* OpenEXR StillLife (EXR linear -> PQ via zscale) */
            { "test/samples/stilllife_i010_pq.yuv",            1240,  846, FUSED_TRC_PQ,  "sample_stilllife" },
            /* Real-world photographs (JPEG -> PQ via zscale) */
            { "test/samples/photo_panorama_i010_pq.yuv",      1000,  666, FUSED_TRC_PQ,  "photo_panorama" },
            { "test/samples/photo_landscape_i010_pq.yuv",     1000,  666, FUSED_TRC_PQ,  "photo_landscape" },
            { "test/samples/photo_nature_i010_pq.yuv",        1000,  666, FUSED_TRC_PQ,  "photo_nature" },
            { "test/samples/photo_triad_i010_pq.yuv",          800,  534, FUSED_TRC_PQ,  "photo_triad" },
        };

        int found_samples = 0;
        for (int si = 0; si < (int)(sizeof(samples)/sizeof(samples[0])); si++) {
            /* Skip entries with unknown dimensions (need to probe) */
            if (samples[si].w == 0) continue;

            test_hdr_frame_t frame;
            if (load_raw_i010(samples[si].filename, samples[si].w, samples[si].h, &frame) != 0) {
                printf("  (skipped: %s not found)\n", samples[si].filename);
                continue;
            }

            if (!found_samples) {
                printf("\n  --- SDR tone-mapped output (sample files) ---\n\n");
                found_samples = 1;
            }

            total_files += run_hdr_visual_config(samples[si].label,
                                                  &frame, hdr_flags,
                                                  samples[si].transfer);

            test_hdr_frame_free(&frame);
        }

        if (!found_samples) {
            printf("\n  No sample files found in test/samples/.\n");
            printf("  Run 'make fetch-samples' to download HDR test content.\n");
        }
    }

    /* ======================================================================
     * Video file demos - extract frame from .mov/.mp4/.MP4, scale and
     * tone-map through funnelcake, write output as .mov files:
     *   HDR: HEVC Main10 with HLG/PQ metadata (displays in HDR on macOS)
     *   SDR: H.264 BT.709 (standard SDR display)
     * ====================================================================== */

    {
        static const struct {
            const char *filename;
        } known_videos[] = {
            { "test/samples/C0008.MP4" },
            { "test/samples/IMG_1467.mov" },
        };

        int found_videos = 0;
        for (int vi = 0; vi < (int)(sizeof(known_videos)/sizeof(known_videos[0])); vi++) {
            const char *vpath = known_videos[vi].filename;

            FILE *vf = fopen(vpath, "rb");
            if (!vf) {
                printf("  (skipped: %s not found)\n", vpath);
                continue;
            }
            fclose(vf);

            if (!found_videos) {
                printf("\n  --- Video file demos ---\n\n");
                found_videos = 1;
            }

            /* Probe dimensions with ffprobe */
            char probe_cmd[512];
            snprintf(probe_cmd, sizeof(probe_cmd),
                     "ffprobe -v error -select_streams v:0 "
                     "-show_entries stream=width,height "
                     "-of csv=s=x:p=0 \"%s\" 2>/dev/null",
                     vpath);
            FILE *pp = popen(probe_cmd, "r");
            if (!pp) continue;
            int vw = 0, vh = 0;
            if (fscanf(pp, "%dx%d", &vw, &vh) != 2) { pclose(pp); continue; }
            pclose(pp);

            if (vw <= 0 || vh <= 0 || (vw & 1) || (vh & 1)) continue;

            /* Extract one frame as I010 (10-bit 4:2:0 planar) */
            char raw_path[256];
            snprintf(raw_path, sizeof(raw_path), "/tmp/funnelcake_demo_%d.yuv", vi);
            char extract_cmd[1024];
            snprintf(extract_cmd, sizeof(extract_cmd),
                     "ffmpeg -y -i \"%s\" -frames:v 1 -pix_fmt yuv420p10le "
                     "-f rawvideo \"%s\" 2>/dev/null", vpath, raw_path);
            if (system(extract_cmd) != 0) continue;

            /* Load the frame */
            test_hdr_frame_t frame;
            if (load_raw_i010(raw_path, vw, vh, &frame) != 0) {
                remove(raw_path);
                continue;
            }
            remove(raw_path);

            /* Derive a short label from the filename */
            const char *base = strrchr(vpath, '/');
            base = base ? base + 1 : vpath;
            char label[64];
            snprintf(label, sizeof(label), "%.*s",
                     (int)(strchr(base, '.') ? strchr(base, '.') - base : 32), base);

            printf("  %s (%dx%d):\n", vpath, vw, vh);

            /* Set up HDR scaler: produce both HDR and SDR at each thirds step */
            uint32_t flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
            fused_hdr_ctx_t ctx;
            memset(&ctx, 0, sizeof(ctx));
            ctx.src_width      = frame.width;
            ctx.src_height     = frame.height;
            ctx.src_y_stride   = frame.y_stride;
            ctx.src_uv_stride  = frame.uv_stride;
            ctx.src_format     = FUSED_PIX_I010;
            ctx.src_transfer   = FUSED_TRC_HLG;  /* both files are HLG */
            ctx.requested_flags = flags;
            ctx.hdr_flags       = flags;
            ctx.sdr_flags       = flags;
            ctx.tonemap.curve   = FUSED_TONEMAP_HABLE;
            suppress_hdr_log(&ctx);

            int rc = fused_hdr_init(&ctx);
            if (rc < 0) {
                fprintf(stderr, "    fused_hdr_init failed rc=%d\n", rc);
                test_hdr_frame_free(&frame);
                continue;
            }

            fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

            /* Write output .mov files for each scale step */
            static const char *ratio_names[] = {
                "1.5x", "2x", "3x", "4x", "6x", "8x", "12x", "16x"
            };

            for (int i = 0; i < 8; i++) {
                const char *ratio = (i < 8) ? ratio_names[i] : "?x";

                /* HDR output -> HEVC Main10 .mov with HLG metadata */
                if (ctx.hdr_outputs[i].plane_y) {
                    int ow = ctx.hdr_outputs[i].width;
                    int oh = ctx.hdr_outputs[i].height;
                    int ys = ctx.hdr_outputs[i].y_stride / 2;
                    int uvs = ctx.hdr_outputs[i].uv_stride / 2;
                    int cw = ow / 2, ch = oh / 2;

                    char tmp[256], out[256];
                    snprintf(tmp, sizeof(tmp), "/tmp/funnelcake_%s_hdr_%s.yuv", label, ratio);
                    snprintf(out, sizeof(out), "output/%s_HDR_%s.mov", label, ratio);

                    FILE *fp = fopen(tmp, "wb");
                    if (fp) {
                        for (int r = 0; r < oh; r++)
                            fwrite(ctx.hdr_outputs[i].plane_y + r * ys,
                                   sizeof(uint16_t), (size_t)ow, fp);
                        for (int r = 0; r < ch; r++)
                            fwrite(ctx.hdr_outputs[i].plane_u + r * uvs,
                                   sizeof(uint16_t), (size_t)cw, fp);
                        for (int r = 0; r < ch; r++)
                            fwrite(ctx.hdr_outputs[i].plane_v + r * uvs,
                                   sizeof(uint16_t), (size_t)cw, fp);
                        fclose(fp);

                        char cmd[1024];
                        snprintf(cmd, sizeof(cmd),
                                 "ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s %dx%d "
                                 "-color_primaries bt2020 -color_trc arib-std-b67 "
                                 "-colorspace bt2020nc -color_range tv "
                                 "-i \"%s\" -c:v libx265 -tag:v hvc1 -pix_fmt yuv420p10le "
                                 "-x265-params \"colorprim=bt2020:transfer=arib-std-b67"
                                 ":colormatrix=bt2020nc:range=limited\" "
                                 "\"%s\" 2>/dev/null",
                                 ow, oh, tmp, out);
                        if (system(cmd) == 0) {
                            printf("    %s\n", out);
                            total_files++;
                        }
                        remove(tmp);
                    }
                }

                /* SDR output -> H.264 .mov with BT.709 metadata */
                if (ctx.sdr_outputs[i].plane_y) {
                    int ow = ctx.sdr_outputs[i].width;
                    int oh = ctx.sdr_outputs[i].height;
                    int ys = ctx.sdr_outputs[i].y_stride;
                    int uvs = ctx.sdr_outputs[i].uv_stride;
                    int cw = ow / 2, ch = oh / 2;

                    char tmp[256], out[256];
                    snprintf(tmp, sizeof(tmp), "/tmp/funnelcake_%s_sdr_%s.yuv", label, ratio);
                    snprintf(out, sizeof(out), "output/%s_SDR_%s.mov", label, ratio);

                    FILE *fp = fopen(tmp, "wb");
                    if (fp) {
                        for (int r = 0; r < oh; r++)
                            fwrite(ctx.sdr_outputs[i].plane_y + r * ys, 1, (size_t)ow, fp);
                        for (int r = 0; r < ch; r++)
                            fwrite(ctx.sdr_outputs[i].plane_u + r * uvs, 1, (size_t)cw, fp);
                        for (int r = 0; r < ch; r++)
                            fwrite(ctx.sdr_outputs[i].plane_v + r * uvs, 1, (size_t)cw, fp);
                        fclose(fp);

                        char cmd[1024];
                        snprintf(cmd, sizeof(cmd),
                                 "ffmpeg -y -f rawvideo -pix_fmt yuv420p -s %dx%d "
                                 "-color_primaries bt709 -color_trc bt709 "
                                 "-colorspace bt709 -color_range pc "
                                 "-i \"%s\" -c:v libx264 -pix_fmt yuv420p "
                                 "\"%s\" 2>/dev/null",
                                 ow, oh, tmp, out);
                        if (system(cmd) == 0) {
                            printf("    %s\n", out);
                            total_files++;
                        }
                        remove(tmp);
                    }
                }
            }

            fused_hdr_free(&ctx);
            test_hdr_frame_free(&frame);
        }
    }

    printf("\n  Generated %d files in output/\n", total_files);
}
