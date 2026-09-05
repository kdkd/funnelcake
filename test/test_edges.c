/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#include "funnelcake.h"
#include "internal.h"
#include "detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Deliberately no signal handlers or expected faults. Exact heap boundaries
 * become red zones under ASan; snapshots detect input writes in normal builds.
 * Different padding and output poison catch accidental reads and missed writes. */
static int case_w, case_h, case_format, case_layout;
static unsigned case_mask, case_up, case_tail, case_frame;
static int case_superset;
static size_t cases, simd_cases, rejected_cases;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "edge test line %d: %dx%d format=%d layout=%d mask=%x up=%x tail=%u frame=%u: %s\n", \
        __LINE__, case_w, case_h, case_format, case_layout, case_mask, \
        case_up, case_tail, case_frame, #c); exit(1); } } while (0)

typedef struct {
    unsigned char *base, *data, *saved;
    size_t size;
    int stride, width, height, bytes;
} plane;

static void alloc_plane(plane *p, int width, int height, int bytes, int layout)
{
    p->width = width; p->height = height; p->bytes = bytes;
    p->stride = ((width * bytes + 31) & ~31) + ((layout & 1) ? 64 : 0);
    int offset = (layout & 2) ? bytes : 0;
    /* End precisely at the final active sample, with no trailing row padding. */
    p->size = (size_t)offset + (size_t)(height - 1) * p->stride + width * bytes;
    CHECK(posix_memalign((void **)&p->base, 64, p->size) == 0);
    p->data = p->base + offset;
    p->saved = malloc(p->size);
    CHECK(p->saved);
}

static uint32_t next_random(uint32_t *s)
{
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    return *s;
}

static void fill_plane(plane *p, int component, int poison)
{
    uint32_t seed = 0xc0ffeeu + (unsigned)component * 101 + case_frame * 31337;
    memset(p->base, poison, p->size);
    for (int y = 0; y < p->height; y++) for (int x = 0; x < p->width; x++) {
        unsigned max = p->bytes == 1 ? 255 : 1023;
        unsigned v;
        if (case_frame == 0) v = next_random(&seed) & max;
        else {
            /* Adjacent even/odd codes, extremes, and impulses exercise ties. */
            static const unsigned values[] = {0, 1, 2, 3, 126, 127, 254, 255};
            v = values[(x + 3 * y + component) & 7];
            if (p->bytes == 2) v = (v * 4 + (x & 3)) & max;
        }
        unsigned char *dst = p->data + (size_t)y * p->stride + x * p->bytes;
        if (p->bytes == 1) *dst = (unsigned char)v;
        else { uint16_t u = (uint16_t)v; memcpy(dst, &u, sizeof(u)); }
    }
    memcpy(p->saved, p->base, p->size);
}

static void free_plane(plane *p)
{
    CHECK(memcmp(p->saved, p->base, p->size) == 0);
    free(p->base); free(p->saved);
}

static void select_scalar(int scalar)
{
    CHECK((scalar ? setenv("FUNNELCAKE_FORCE_SCALAR", "1", 1)
                  : unsetenv("FUNNELCAKE_FORCE_SCALAR")) == 0);
    fused_detect_cpu_reset();
}

static void compare_plane(const void *a, const void *b, int width, int height,
                          int stride_a, int stride_b, int bytes)
{
    CHECK(a && b);
    for (int y = 0; y < height; y++) {
        const unsigned char *ra = (const unsigned char *)a + (size_t)y * stride_a;
        const unsigned char *rb = (const unsigned char *)b + (size_t)y * stride_b;
        if (memcmp(ra, rb, (size_t)width * bytes)) {
            for (int x = 0; x < width * bytes; x++) if (ra[x] != rb[x]) {
                fprintf(stderr, "row=%d byte=%d scalar=%u native=%u\n", y, x, ra[x], rb[x]);
                break;
            }
            CHECK(0);
        }
    }
}

#define OUTPUT_HELPERS(name, type, bytes) \
static void poison_##name(type *o, int value) \
{ \
    if (!o->plane_y) return; \
    memset(o->plane_y, value, (size_t)o->y_stride * o->height); \
    memset(o->plane_u, value, (size_t)o->uv_stride * (o->height / 2)); \
    memset(o->plane_v, value, (size_t)o->uv_stride * (o->height / 2)); \
} \
static void compare_##name(const type *a, const type *b) \
{ \
    if (case_superset && !b->plane_y) return; \
    CHECK(!!a->plane_y == !!b->plane_y); \
    if (!a->plane_y) return; \
    CHECK(a->width == b->width && a->height == b->height); \
    compare_plane(a->plane_y, b->plane_y, a->width, a->height, a->y_stride, b->y_stride, bytes); \
    compare_plane(a->plane_u, b->plane_u, a->width/2, a->height/2, a->uv_stride, b->uv_stride, bytes); \
    compare_plane(a->plane_v, b->plane_v, a->width/2, a->height/2, a->uv_stride, b->uv_stride, bytes); \
}
OUTPUT_HELPERS(sdr, fused_scale_output_t, 1)
OUTPUT_HELPERS(hdr, fused_hdr_output_t, 2)

static void run_case(void)
{
    plane input[2][3] = {{{0}}};
    fused_scaler_ctx_t s[2] = {{0}};
    fused_hdr_ctx_t h[2] = {{0}};
    int rc[2];
    int hdr = case_format >= 0;
    int packed = hdr && (case_format & 1);
    int chroma_height = hdr && case_format >= FUSED_PIX_I210 ? case_h : case_h / 2;
    for (int side = 0; side < 2; side++) {
        alloc_plane(&input[side][0], case_w, case_h, hdr ? 2 : 1, side ? case_layout : 0);
        alloc_plane(&input[side][1], packed ? case_w : case_w/2, chroma_height,
                    hdr ? 2 : 1, side ? case_layout : 0);
        if (!packed) alloc_plane(&input[side][2], case_w/2, chroma_height,
                                 hdr ? 2 : 1, side ? case_layout : 0);
        select_scalar(!side);
        unsigned requested = case_superset && !side
            ? ((case_mask & 0xaa) ? 0xaa : 0x55) : case_mask;
        if (!hdr) {
            s[side].src_width = case_w; s[side].src_height = case_h;
            s[side].src_y_stride = input[side][0].stride;
            s[side].src_uv_stride = input[side][1].stride;
            s[side].requested_flags = requested;
            s[side].upscale_flags = case_up;
            s[side].upscale_tail_1_5x = case_tail;
            s[side].log_errors.target = s[side].log_warnings.target = FUSED_LOG_SUPPRESS;
            rc[side] = fused_scaler_init(&s[side]);
        } else {
            h[side].src_width = case_w; h[side].src_height = case_h;
            h[side].src_y_stride = input[side][0].stride;
            h[side].src_uv_stride = input[side][1].stride;
            h[side].src_format = case_format;
            h[side].src_transfer = (case_layout & 1) ? FUSED_TRC_HLG : FUSED_TRC_PQ;
            h[side].requested_flags = h[side].sdr_flags = requested;
            h[side].hdr_flags = (case_layout & 1) ? 0 : requested;
            h[side].tonemap_1x = 1;
            h[side].tonemap.curve = case_layout % 3;
            h[side].upscale_flags = h[side].upscale_sdr_flags = case_up;
            h[side].upscale_tail_1_5x = case_tail;
            h[side].upscale_sdr_tail_1_5x = case_tail;
            h[side].log_errors.target = h[side].log_warnings.target = FUSED_LOG_SUPPRESS;
            rc[side] = fused_hdr_init(&h[side]);
        }
    }
    CHECK((rc[0] < 0) == (rc[1] < 0));
    CHECK(!case_superset || (rc[0] >= 0 && rc[1] >= 0));
    if (rc[0] >= 0) {
        CHECK((rc[0] & ~FUSED_WARN_BIT_SCALAR) == (rc[1] & ~FUSED_WARN_BIT_SCALAR));
        if (hdr) {
            if (case_superset) {
                CHECK(h[1].achieved_sdr_flags == case_mask);
                CHECK(h[0].achieved_sdr_flags == ((case_mask & 0xaa) ? 0xaa : 0x55));
                CHECK(h[1].achieved_hdr_flags == ((case_layout & 1) ? 0 : case_mask));
            } else {
                CHECK(h[0].achieved_hdr_flags == h[1].achieved_hdr_flags);
                CHECK(h[0].achieved_sdr_flags == h[1].achieved_sdr_flags);
            }
            if (case_layout < 2 && ((fused_hdr_internal_t *)h[0]._internal)->kernel_fn !=
                ((fused_hdr_internal_t *)h[1]._internal)->kernel_fn) simd_cases++;
        } else {
            if (case_superset) {
                CHECK(s[1].achieved_flags == case_mask);
                CHECK(s[0].achieved_flags == ((case_mask & 0xaa) ? 0xaa : 0x55));
            } else CHECK(s[0].achieved_flags == s[1].achieved_flags);
            if (case_layout < 2 && ((fused_internal_t *)s[0]._internal)->kernel_fn !=
                ((fused_internal_t *)s[1]._internal)->kernel_fn) simd_cases++;
        }
    } else rejected_cases++;
    /* Two different frames through each context detect stale scratch/output. */
    for (case_frame = 0; case_frame < 2; case_frame++) {
        for (int side = 0; side < 2; side++) {
            for (int p = 0; p < (packed ? 2 : 3); p++)
                fill_plane(&input[side][p], p, side ? 0xa5 : 0x5a);
            if (rc[side] < 0) continue;
            select_scalar(!side);
            int poison = side ? 0x55 : 0xaa;
            if (!hdr) {
                for (int i = 0; i < 8; i++) poison_sdr(&s[side].outputs[i], poison);
                for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) poison_sdr(&s[side].upscale_outputs[i], poison);
                fused_scaler_run(&s[side], input[side][0].data, input[side][1].data, input[side][2].data);
            } else {
                for (int i = 0; i < 8; i++) {
                    poison_hdr(&h[side].hdr_outputs[i], poison);
                    poison_sdr(&h[side].sdr_outputs[i], poison);
                }
                for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) {
                    poison_hdr(&h[side].upscale_hdr_outputs[i], poison);
                    poison_sdr(&h[side].upscale_sdr_outputs[i], poison);
                }
                poison_sdr(&h[side].output_1x, poison);
                fused_hdr_run(&h[side], (uint16_t *)input[side][0].data,
                              (uint16_t *)input[side][1].data, (uint16_t *)input[side][2].data);
            }
            for (int p = 0; p < (packed ? 2 : 3); p++)
                CHECK(memcmp(input[side][p].base, input[side][p].saved, input[side][p].size) == 0);
        }
        if (rc[0] < 0) continue;
        for (int i = 0; i < 8; i++) {
            if (!hdr) compare_sdr(&s[0].outputs[i], &s[1].outputs[i]);
            else {
                compare_hdr(&h[0].hdr_outputs[i], &h[1].hdr_outputs[i]);
                compare_sdr(&h[0].sdr_outputs[i], &h[1].sdr_outputs[i]);
            }
        }
        for (int i = 0; i < FUSED_MAX_UPSCALE_STEPS; i++) {
            if (!hdr) compare_sdr(&s[0].upscale_outputs[i], &s[1].upscale_outputs[i]);
            else {
                compare_hdr(&h[0].upscale_hdr_outputs[i], &h[1].upscale_hdr_outputs[i]);
                compare_sdr(&h[0].upscale_sdr_outputs[i], &h[1].upscale_sdr_outputs[i]);
            }
        }
        if (hdr) compare_sdr(&h[0].output_1x, &h[1].output_1x);
    }
    for (int side = 0; side < 2; side++) {
        fused_scaler_free(&s[side]); fused_hdr_free(&h[side]);
        for (int p = 0; p < (packed ? 2 : 3); p++) free_plane(&input[side][p]);
    }
    cases++;
}

static void matrix(void)
{
    static const int widths[] = {2, 4, 30, 32, 34, 46, 48, 50, 62, 64, 66,
        94, 96, 98, 126, 128, 130, 190, 192, 194, 254, 256, 258,
        510, 512, 514, 1022, 1024, 1026, 1536, 1560};
    static const int heights[] = {2, 4, 6, 12, 16, 24, 32, 34, 48, 64, 96};
    static const unsigned masks[] = {0, FUSED_SCALE_2X, FUSED_SCALE_16X,
        FUSED_SCALE_12X, 0xaa, 0x55, FUSED_SCALE_1_5X | FUSED_SCALE_6X};
    cases = simd_cases = rejected_cases = 0;
    select_scalar(0);
    const char *backend = fused_backend();
    int available = fused_simd_available();
    for (size_t w = 0; w < sizeof(widths)/sizeof(widths[0]); w++)
    for (size_t m = 0; m < sizeof(masks)/sizeof(masks[0]); m++)
    for (case_layout = 0; case_layout < 4; case_layout++)
    for (case_format = -1; case_format <= FUSED_PIX_P210; case_format++) {
        case_w = widths[w]; case_h = heights[(w + m) % (sizeof(heights)/sizeof(heights[0]))];
        case_mask = masks[m];
        case_up = (m % 3 == 0) ? FUSED_UPSCALE_2X | FUSED_UPSCALE_4X : 0;
        case_tail = (m % 3 == 1);
        run_case();
    }
    /* Adding other outputs must not change a requested output. Use sizes
     * divisible by the deepest cascade so cropping cannot change geometry. */
    case_superset = 1;
    for (int family = 0; family < 2; family++)
    for (unsigned subset = 1; subset < 16; subset++)
    for (case_layout = 0; case_layout < 2; case_layout++)
    for (case_format = -1; case_format <= FUSED_PIX_P210; case_format++) {
        case_w = family ? 1536 : 1560; case_h = 96;
        case_mask = 0;
        for (int bit = 0; bit < 4; bit++)
            if (subset & (1u << bit)) case_mask |= 1u << (2 * bit + family);
        case_up = case_tail = 0;
        run_case();
    }
    case_superset = 0;
    CHECK(!available || simd_cases > 0);
    printf("Boundary matrix (%s): %zu cases, %zu aligned SIMD cases, %zu rejected configurations\n",
           backend, cases, simd_cases, rejected_cases);
    if (!available) puts("SIMD coverage skipped: no supported SIMD backend in this build/CPU");
}

int main(void)
{
    select_scalar(0);
    int avx512 = strcmp(fused_backend(), "avx512") == 0;
    matrix();
    if (avx512) {
        CHECK(setenv("FUNNELCAKE_NO_AVX512", "1", 1) == 0);
        select_scalar(0);
        CHECK(strcmp(fused_backend(), "avx2") == 0);
        matrix();
    }
    return 0;
}
