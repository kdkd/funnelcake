/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "test_patterns.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* --------------------------------------------------------------------------
 * Pattern names table
 * -------------------------------------------------------------------------- */

const char *pattern_names[] = {
    "solid",
    "hgradient",
    "vgradient",
    "checkerboard",
    "random"
};

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

/* Round up x to the next multiple of 32. */
static int align_up_32(int x)
{
    return (x + 31) & ~31;
}

/* Allocate a 32-byte aligned plane of (stride * rows) bytes. */
static uint8_t *alloc_plane(int stride, int rows)
{
    void *ptr = NULL;
    size_t sz = (size_t)stride * (size_t)rows;
    if (posix_memalign(&ptr, 32, sz) != 0)
        return NULL;
    memset(ptr, 0, sz);
    return (uint8_t *)ptr;
}

/* Simple xorshift32 PRNG - returns next state and writes value via *val. */
static uint32_t xorshift32(uint32_t state, uint8_t *val)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    *val = (uint8_t)(state & 0xFF);
    return state;
}

/* --------------------------------------------------------------------------
 * Fill functions
 * -------------------------------------------------------------------------- */

static void fill_solid(test_frame_t *f)
{
    /* Y=128, U=128, V=128 - neutral gray */
    for (int y = 0; y < f->height; y++)
        memset(f->plane_y + y * f->y_stride, 128, (size_t)f->width);

    int ch = f->height / 2;
    int cw = f->width  / 2;
    for (int y = 0; y < ch; y++) {
        memset(f->plane_u + y * f->uv_stride, 128, (size_t)cw);
        memset(f->plane_v + y * f->uv_stride, 128, (size_t)cw);
    }
}

static void fill_hgradient(test_frame_t *f)
{
    /* Luma: x gradient 0-255 across width */
    for (int y = 0; y < f->height; y++) {
        uint8_t *row = f->plane_y + y * f->y_stride;
        for (int x = 0; x < f->width; x++) {
            row[x] = (uint8_t)((x * 255) / (f->width > 1 ? f->width - 1 : 1));
        }
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    for (int y = 0; y < ch; y++) {
        uint8_t *ru = f->plane_u + y * f->uv_stride;
        uint8_t *rv = f->plane_v + y * f->uv_stride;
        for (int x = 0; x < cw; x++) {
            uint8_t v = (uint8_t)((x * 255) / (cw > 1 ? cw - 1 : 1));
            ru[x] = v;
            rv[x] = v;
        }
    }
}

static void fill_vgradient(test_frame_t *f)
{
    /* Luma: y gradient 0-255 down height */
    for (int y = 0; y < f->height; y++) {
        uint8_t val = (uint8_t)((y * 255) / (f->height > 1 ? f->height - 1 : 1));
        memset(f->plane_y + y * f->y_stride, val, (size_t)f->width);
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    for (int y = 0; y < ch; y++) {
        uint8_t val = (uint8_t)((y * 255) / (ch > 1 ? ch - 1 : 1));
        memset(f->plane_u + y * f->uv_stride, val, (size_t)cw);
        memset(f->plane_v + y * f->uv_stride, val, (size_t)cw);
    }
}

static void fill_checkerboard(test_frame_t *f)
{
    /* Luma block_size=8, values 16 and 240 */
    int ybs = 8;
    for (int y = 0; y < f->height; y++) {
        uint8_t *row = f->plane_y + y * f->y_stride;
        for (int x = 0; x < f->width; x++) {
            int tile = (x / ybs) + (y / ybs);
            row[x] = (tile & 1) ? 240 : 16;
        }
    }

    /* Chroma block_size=4, values 16 and 240 */
    int cbs = 4;
    int ch = f->height / 2;
    int cw = f->width  / 2;
    for (int y = 0; y < ch; y++) {
        uint8_t *ru = f->plane_u + y * f->uv_stride;
        uint8_t *rv = f->plane_v + y * f->uv_stride;
        for (int x = 0; x < cw; x++) {
            int tile = (x / cbs) + (y / cbs);
            uint8_t val = (tile & 1) ? 240 : 16;
            ru[x] = val;
            rv[x] = val;
        }
    }
}

static void fill_random(test_frame_t *f, uint32_t seed)
{
    uint32_t st = (seed == 0) ? 1 : seed;  /* xorshift32 must not have state=0 */

    for (int y = 0; y < f->height; y++) {
        uint8_t *row = f->plane_y + y * f->y_stride;
        for (int x = 0; x < f->width; x++) {
            st = xorshift32(st, &row[x]);
        }
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    for (int y = 0; y < ch; y++) {
        uint8_t *ru = f->plane_u + y * f->uv_stride;
        uint8_t *rv = f->plane_v + y * f->uv_stride;
        for (int x = 0; x < cw; x++) {
            st = xorshift32(st, &ru[x]);
            st = xorshift32(st, &rv[x]);
        }
    }
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

int test_frame_create(test_frame_t *frame, int width, int height,
                      test_pattern_t pattern, uint32_t seed)
{
    if (!frame || width <= 0 || height <= 0) return -1;

    memset(frame, 0, sizeof(*frame));

    frame->width     = width;
    frame->height    = height;
    frame->y_stride  = align_up_32(width);
    frame->uv_stride = align_up_32(width / 2);

    frame->plane_y = alloc_plane(frame->y_stride, height);
    frame->plane_u = alloc_plane(frame->uv_stride, height / 2);
    frame->plane_v = alloc_plane(frame->uv_stride, height / 2);

    if (!frame->plane_y || !frame->plane_u || !frame->plane_v) {
        test_frame_free(frame);
        return -1;
    }

    switch (pattern) {
        case PATTERN_SOLID:        fill_solid(frame);             break;
        case PATTERN_HGRADIENT:    fill_hgradient(frame);         break;
        case PATTERN_VGRADIENT:    fill_vgradient(frame);         break;
        case PATTERN_CHECKERBOARD: fill_checkerboard(frame);      break;
        case PATTERN_RANDOM:       fill_random(frame, seed);      break;
        default:                   fill_solid(frame);             break;
    }

    return 0;
}

void test_frame_free(test_frame_t *frame)
{
    if (!frame) return;
    free(frame->plane_y);
    free(frame->plane_u);
    free(frame->plane_v);
    memset(frame, 0, sizeof(*frame));
}

/* ==========================================================================
 * 10-bit HDR frame creation
 * ========================================================================== */

/* Allocate a 32-byte aligned plane of (stride * rows) bytes for uint16_t. */
static uint16_t *alloc_plane_16(int stride_bytes, int rows)
{
    void *ptr = NULL;
    size_t sz = (size_t)stride_bytes * (size_t)rows;
    if (posix_memalign(&ptr, 32, sz) != 0)
        return NULL;
    memset(ptr, 0, sz);
    return (uint16_t *)ptr;
}

/* xorshift32 returning a 10-bit value */
static uint32_t xorshift32_16(uint32_t state, uint16_t *val)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    *val = (uint16_t)(state & 0x3FF);
    return state;
}

/* --------------------------------------------------------------------------
 * PQ (ST 2084) inverse EOTF
 * Map linear luminance L (nits) to a limited-range 10-bit PQ code
 * (Y 64..940), matching what real HDR10 streams and the ffmpeg/zscale
 * sample files carry.
 * -------------------------------------------------------------------------- */

#include <math.h>

static uint16_t linear_to_pq(double L)
{
    /* ST 2084 constants */
    static const double c1 = 0.8359375;          /* 3424/4096   */
    static const double c2 = 18.8515625;         /* 2413/128    */
    static const double c3 = 18.6875;            /* 2392/128    */
    static const double m1 = 0.1593017578125;    /* 2610/16384  */
    static const double m2 = 78.84375;           /* 2523/32     */

    double Y = L / 10000.0;
    if (Y < 0.0) Y = 0.0;
    double Ym1 = pow(Y, m1);
    double num = c1 + c2 * Ym1;
    double den = 1.0 + c3 * Ym1;
    double E = pow(num / den, m2);
    int code = 64 + (int)(E * 876.0 + 0.5);   /* limited range */
    if (code < 64)  code = 64;
    if (code > 940) code = 940;
    return (uint16_t)code;
}

/* --------------------------------------------------------------------------
 * HDR fill functions
 * -------------------------------------------------------------------------- */

static void fill_hdr_solid(test_hdr_frame_t *f)
{
    /* Y=512, U=512, V=512 - mid-range 10-bit */
    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++)
            row[x] = 512;
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            ru[x] = 512;
            rv[x] = 512;
        }
    }
}

static void fill_hdr_hgradient(test_hdr_frame_t *f)
{
    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++)
            row[x] = (uint16_t)((x * 1023) / (f->width > 1 ? f->width - 1 : 1));
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            uint16_t v = (uint16_t)((x * 1023) / (cw > 1 ? cw - 1 : 1));
            ru[x] = v;
            rv[x] = v;
        }
    }
}

static void fill_hdr_vgradient(test_hdr_frame_t *f)
{
    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < f->height; y++) {
        uint16_t val = (uint16_t)((y * 1023) / (f->height > 1 ? f->height - 1 : 1));
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++)
            row[x] = val;
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t val = (uint16_t)((y * 1023) / (ch > 1 ? ch - 1 : 1));
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            ru[x] = val;
            rv[x] = val;
        }
    }
}

static void fill_hdr_checkerboard(test_hdr_frame_t *f)
{
    int ybs = 8;
    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++) {
            int tile = (x / ybs) + (y / ybs);
            row[x] = (tile & 1) ? 960 : 64;
        }
    }

    int cbs = 4;
    int ch = f->height / 2;
    int cw = f->width  / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            int tile = (x / cbs) + (y / cbs);
            uint16_t val = (tile & 1) ? 960 : 64;
            ru[x] = val;
            rv[x] = val;
        }
    }
}

static void fill_hdr_random(test_hdr_frame_t *f, uint32_t seed)
{
    uint32_t st = (seed == 0) ? 1 : seed;
    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);

    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++)
            st = xorshift32_16(st, &row[x]);
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            st = xorshift32_16(st, &ru[x]);
            st = xorshift32_16(st, &rv[x]);
        }
    }
}

static void fill_hdr_pq_ramp(test_hdr_frame_t *f)
{
    /* Y values follow the PQ curve: map linear 0-1000 nits through ST 2084 */
    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++) {
            double L = (double)x * 1000.0 / (double)(f->width > 1 ? f->width - 1 : 1);
            row[x] = linear_to_pq(L);
        }
    }

    /* Chroma at mid-level */
    int ch = f->height / 2;
    int cw = f->width  / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            ru[x] = 512;
            rv[x] = 512;
        }
    }
}

static void fill_hdr_highlight(test_hdr_frame_t *f)
{
    /* Background at ~50 nits PQ value, center rectangle at ~500 nits PQ value */
    uint16_t bg_y = linear_to_pq(50.0);
    uint16_t fg_y = linear_to_pq(500.0);

    int x0 = f->width  / 4;
    int x1 = f->width  * 3 / 4;
    int y0 = f->height / 4;
    int y1 = f->height * 3 / 4;

    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++) {
            if (x >= x0 && x < x1 && y >= y0 && y < y1)
                row[x] = fg_y;
            else
                row[x] = bg_y;
        }
    }

    /* Chroma at mid-level */
    int ch = f->height / 2;
    int cw = f->width  / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            ru[x] = 512;
            rv[x] = 512;
        }
    }
}

static void fill_hdr_saturated_2020(test_hdr_frame_t *f)
{
    /* Y=512 everywhere; top-half U=800,V=512; bottom-half U=512,V=800 */
    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++)
            row[x] = 512;
    }

    int ch = f->height / 2;
    int cw = f->width  / 2;
    int half_ch = ch / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            if (y < half_ch) {
                ru[x] = 800;
                rv[x] = 512;
            } else {
                ru[x] = 512;
                rv[x] = 800;
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * PQ-encoded SMPTE color bars in BT.2020 YCbCr
 *
 * Creates proper PQ-encoded color bars that can be tone-mapped to SDR to
 * visually verify the tone mapping pipeline.  Each bar is defined as a
 * linear-light RGB color (in nits, BT.2020 primaries), converted through
 * the BT.2020 RGB->YCbCr matrix, and then PQ-encoded.
 *
 * The 7 standard bars at 75% intensity (~200 nits peak for SDR-equivalent
 * content represented in HDR):
 *   White, Yellow, Cyan, Green, Magenta, Red, Blue
 * -------------------------------------------------------------------------- */

static void fill_hdr_pq_colorbars(test_hdr_frame_t *f)
{
    /* 75% bars in linear light (nits).  200 nits ≈ 75% SDR on a 1000-nit display.
     * These are approximate - the exact values depend on the display, but
     * they're standard enough to produce recognizable bars after tone mapping. */
    static const struct { double r, g, b; } bars[8] = {
        /* White   */ { 200.0, 200.0, 200.0 },
        /* Yellow  */ { 200.0, 200.0,   0.0 },
        /* Cyan    */ {   0.0, 200.0, 200.0 },
        /* Green   */ {   0.0, 200.0,   0.0 },
        /* Magenta */ { 200.0,   0.0, 200.0 },
        /* Red     */ { 200.0,   0.0,   0.0 },
        /* Blue    */ {   0.0,   0.0, 200.0 },
        /* Black   */ {   0.0,   0.0,   0.0 },
    };

    /* BT.2020 non-constant-luminance YCbCr (from BT.2020 spec), entirely
     * in the gamma (PQ signal) domain, like real HDR10 streams:
     * Y' =  0.2627*R' + 0.6780*G' + 0.0593*B'   (R' = PQ(R), etc.)
     * Cb = (B' - Y') / 1.8814  (scaled to [-0.5, 0.5])
     * Cr = (R' - Y') / 1.4746  (scaled to [-0.5, 0.5]) */

    /* Pre-compute Y/Cb/Cr for each bar in PQ domain */
    uint16_t bar_y[8], bar_cb[8], bar_cr[8];
    for (int i = 0; i < 8; i++) {
        double R = bars[i].r, G = bars[i].g, B = bars[i].b;

        /* PQ-encode each component, then take the NCL weighted sum for
         * luma.  Codes are limited range, so the normalized signal is
         * (code - 64) / 876. */
        double R_pq = ((double)linear_to_pq(R) - 64.0) / 876.0;
        double G_pq = ((double)linear_to_pq(G) - 64.0) / 876.0;
        double B_pq = ((double)linear_to_pq(B) - 64.0) / 876.0;
        double Y_pq = 0.2627 * R_pq + 0.6780 * G_pq + 0.0593 * B_pq;

        bar_y[i] = (uint16_t)(64 + (int)(Y_pq * 876.0 + 0.5));
        Y_pq = ((double)bar_y[i] - 64.0) / 876.0;   /* re-read quantized Y' */

        /* Cb = (B' - Y') / 1.8814, Cr = (R' - Y') / 1.4746
         * Quantize to limited-range 10-bit chroma (center 512, span 896) */
        double Cb = (B_pq - Y_pq) / 1.8814;
        double Cr = (R_pq - Y_pq) / 1.4746;

        int cb_code = (int)(Cb * 896.0 + 512.0 + 0.5);
        int cr_code = (int)(Cr * 896.0 + 512.0 + 0.5);
        if (cb_code < 64) cb_code = 64;
        if (cb_code > 960) cb_code = 960;
        if (cr_code < 64) cr_code = 64;
        if (cr_code > 960) cr_code = 960;
        bar_cb[i] = (uint16_t)cb_code;
        bar_cr[i] = (uint16_t)cr_code;
    }

    int samples_per_row = f->y_stride / (int)sizeof(uint16_t);
    int bar_width = f->width / 8;

    /* Fill luma */
    for (int y = 0; y < f->height; y++) {
        uint16_t *row = f->plane_y + y * samples_per_row;
        for (int x = 0; x < f->width; x++) {
            int bar_idx = x / bar_width;
            if (bar_idx > 7) bar_idx = 7;
            row[x] = bar_y[bar_idx];
        }
    }

    /* Fill chroma (4:2:0) */
    int ch = f->height / 2;
    int cw = f->width / 2;
    int uv_samples_per_row = f->uv_stride / (int)sizeof(uint16_t);
    int bar_cw = bar_width / 2;
    if (bar_cw < 1) bar_cw = 1;

    for (int y = 0; y < ch; y++) {
        uint16_t *ru = f->plane_u + y * uv_samples_per_row;
        uint16_t *rv = f->plane_v + y * uv_samples_per_row;
        for (int x = 0; x < cw; x++) {
            int bar_idx = x / bar_cw;
            if (bar_idx > 7) bar_idx = 7;
            ru[x] = bar_cb[bar_idx];
            rv[x] = bar_cr[bar_idx];
        }
    }
}

static void fill_hdr_pattern(test_hdr_frame_t *f, test_pattern_t pattern,
                             uint32_t seed)
{
    switch (pattern) {
        case PATTERN_SOLID:        fill_hdr_solid(f);             break;
        case PATTERN_HGRADIENT:    fill_hdr_hgradient(f);         break;
        case PATTERN_VGRADIENT:    fill_hdr_vgradient(f);         break;
        case PATTERN_CHECKERBOARD: fill_hdr_checkerboard(f);      break;
        case PATTERN_RANDOM:       fill_hdr_random(f, seed);      break;
        default:
            /* HDR-specific patterns */
            if (pattern == PATTERN_PQ_RAMP)
                fill_hdr_pq_ramp(f);
            else if (pattern == PATTERN_HDR_HIGHLIGHT)
                fill_hdr_highlight(f);
            else if (pattern == PATTERN_SATURATED_2020)
                fill_hdr_saturated_2020(f);
            else if (pattern == PATTERN_PQ_COLORBARS)
                fill_hdr_pq_colorbars(f);
            else
                fill_hdr_solid(f);
            break;
    }
}

/* --------------------------------------------------------------------------
 * HDR frame public API
 * -------------------------------------------------------------------------- */

int test_hdr_frame_create(test_hdr_frame_t *frame, int width, int height,
                          test_pattern_t pattern, uint32_t seed)
{
    if (!frame || width <= 0 || height <= 0) return -1;

    memset(frame, 0, sizeof(*frame));

    frame->width     = width;
    frame->height    = height;
    frame->y_stride  = align_up_32(width * 2);         /* width * sizeof(uint16_t) rounded up */
    frame->uv_stride = align_up_32((width / 2) * 2);

    frame->plane_y = alloc_plane_16(frame->y_stride, height);
    frame->plane_u = alloc_plane_16(frame->uv_stride, height / 2);
    frame->plane_v = alloc_plane_16(frame->uv_stride, height / 2);

    if (!frame->plane_y || !frame->plane_u || !frame->plane_v) {
        test_hdr_frame_free(frame);
        return -1;
    }

    fill_hdr_pattern(frame, pattern, seed);
    return 0;
}

void test_hdr_frame_free(test_hdr_frame_t *frame)
{
    if (!frame) return;
    free(frame->plane_y);
    free(frame->plane_u);
    free(frame->plane_v);
    memset(frame, 0, sizeof(*frame));
}

/* --------------------------------------------------------------------------
 * P010 frame public API
 *
 * Creates an I010 frame internally, then interleaves U/V into a single plane.
 * -------------------------------------------------------------------------- */

int test_p010_frame_create(test_p010_frame_t *frame, int width, int height,
                           test_pattern_t pattern, uint32_t seed)
{
    if (!frame || width <= 0 || height <= 0) return -1;

    memset(frame, 0, sizeof(*frame));

    /* Create a temporary I010 frame to get pattern data */
    test_hdr_frame_t i010;
    if (test_hdr_frame_create(&i010, width, height, pattern, seed) != 0)
        return -1;

    frame->width     = width;
    frame->height    = height;
    frame->y_stride  = i010.y_stride;
    /* UV interleaved: two samples (U+V) per chroma pixel pair */
    frame->uv_stride = align_up_32((width / 2) * 2 * 2);  /* (cw * 2 * sizeof(uint16_t)) aligned */

    /* Copy Y plane */
    size_t y_size = (size_t)frame->y_stride * height;
    if (posix_memalign((void **)&frame->plane_y, 32, y_size) != 0) {
        test_hdr_frame_free(&i010);
        return -1;
    }
    memcpy(frame->plane_y, i010.plane_y, y_size);

    /* Interleave U and V into UV plane */
    int ch = height / 2;
    int cw = width  / 2;
    size_t uv_size = (size_t)frame->uv_stride * ch;
    if (posix_memalign((void **)&frame->plane_uv, 32, uv_size) != 0) {
        free(frame->plane_y);
        test_hdr_frame_free(&i010);
        memset(frame, 0, sizeof(*frame));
        return -1;
    }
    memset(frame->plane_uv, 0, uv_size);

    int i010_uv_samples = i010.uv_stride / (int)sizeof(uint16_t);
    int p010_uv_samples = frame->uv_stride / (int)sizeof(uint16_t);
    for (int y = 0; y < ch; y++) {
        const uint16_t *src_u = i010.plane_u + y * i010_uv_samples;
        const uint16_t *src_v = i010.plane_v + y * i010_uv_samples;
        uint16_t       *dst   = frame->plane_uv + y * p010_uv_samples;
        for (int x = 0; x < cw; x++) {
            dst[x * 2 + 0] = src_u[x];
            dst[x * 2 + 1] = src_v[x];
        }
    }

    test_hdr_frame_free(&i010);
    return 0;
}

void test_p010_frame_free(test_p010_frame_t *frame)
{
    if (!frame) return;
    free(frame->plane_y);
    free(frame->plane_uv);
    memset(frame, 0, sizeof(*frame));
}
