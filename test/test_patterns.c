#include "test_patterns.h"
#include "funnelcake.h"
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

static int chroma_height_for(const test_frame_t *f)
{
    return (f->chroma_format == FUSED_CHROMA_422) ? f->height : (f->height / 2);
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

/* Simple xorshift32 PRNG — returns next state and writes value via *val. */
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
    /* Y=128, U=128, V=128 — neutral gray */
    for (int y = 0; y < f->height; y++)
        memset(f->plane_y + y * f->y_stride, 128, (size_t)f->width);

    int ch = chroma_height_for(f);
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

    int ch = chroma_height_for(f);
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

    int ch = chroma_height_for(f);
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
    int ch = chroma_height_for(f);
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

    int ch = chroma_height_for(f);
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
    return test_frame_create_ex(frame, width, height, FUSED_CHROMA_420, pattern, seed);
}

int test_frame_create_ex(test_frame_t *frame, int width, int height,
                         int chroma_format, test_pattern_t pattern, uint32_t seed)
{
    if (!frame || width <= 0 || height <= 0) return -1;
    if (chroma_format != FUSED_CHROMA_420 && chroma_format != FUSED_CHROMA_422) return -1;

    memset(frame, 0, sizeof(*frame));

    frame->width     = width;
    frame->height    = height;
    frame->y_stride  = align_up_32(width);
    frame->uv_stride = align_up_32(width / 2);
    frame->chroma_format = chroma_format;

    frame->plane_y = alloc_plane(frame->y_stride, height);
    frame->plane_u = alloc_plane(frame->uv_stride, chroma_height_for(frame));
    frame->plane_v = alloc_plane(frame->uv_stride, chroma_height_for(frame));

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
