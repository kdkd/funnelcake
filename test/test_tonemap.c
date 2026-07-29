/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/*
 * Tone mapping correctness tests.
 *
 * These exist because the original tone mapping shipped with several
 * defects that every existing test passed over: the tone curves
 * degenerated to a hard clip at target_nits (and peak_nits had no effect
 * at all for Hable), a linearly-indexed shadow LUT crushed everything
 * below ~1.2 nits to black in 55-code steps, G was reconstructed in the
 * wrong (linear) domain, and limited-range video was read as full range.
 *
 * Each test below targets one of those failure classes directly on the
 * generated LUTs or on fused_tonemap_apply output, so a regression of the
 * same nature cannot pass unnoticed again.
 */

#include "test_main.h"
#include "test_patterns.h"
#include "funnelcake.h"
#include "internal.h"
#include "tonemap.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

/* Independent PQ inverse EOTF (ST 2084), written from the spec. */
static double ref_pq_encode(double nits)
{
    static const double c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    static const double m1 = 0.1593017578125, m2 = 78.84375;
    double Y = nits / 10000.0;
    if (Y < 0.0) Y = 0.0;
    double Ym1 = pow(Y, m1);
    return pow((c1 + c2 * Ym1) / (1.0 + c3 * Ym1), m2);
}

static double ref_pq_decode(double N)
{
    static const double c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    static const double m1 = 0.1593017578125, m2 = 78.84375;
    double Nm1 = pow(N, 1.0 / m2);
    double num = Nm1 - c1;
    if (num < 0.0) num = 0.0;
    return pow(num / (c2 - c3 * Nm1), 1.0 / m1) * 10000.0;
}

/* nits -> 10-bit code for the given range */
static int pq_code(double nits, int full)
{
    double E = ref_pq_encode(nits);
    if (full)
        return (int)(E * 1023.0 + 0.5);
    return 64 + (int)(E * 876.0 + 0.5);
}

/* Generate LUTs into a zeroed internal state. */
static void gen_luts(fused_hdr_internal_t *h, int curve, int peak, int target,
                     int transfer, int src_range, int dst_range)
{
    fused_tonemap_config_t tm;
    fused_log_config_t lg;
    memset(&tm, 0, sizeof(tm));
    memset(&lg, 0, sizeof(lg));
    lg.target = FUSED_LOG_SUPPRESS;
    tm.curve = curve;
    tm.peak_nits = peak;
    tm.target_nits = target;
    tm.src_range = src_range;
    tm.dst_range = dst_range;
    memset(h, 0, sizeof(*h));
    fused_tonemap_generate_luts(h, transfer, &tm, &lg);
}

static const int g_curves[3] = {
    FUSED_TONEMAP_HABLE, FUSED_TONEMAP_REINHARD, FUSED_TONEMAP_BT2390
};
static const char *g_curve_names[3] = { "hable", "reinhard", "bt2390" };

/* One static state keeps the test stack small. */
static fused_hdr_internal_t g_state, g_state2;

/* --------------------------------------------------------------------------
 * 1. test_tonemap_lut_monotonic
 *    pq_to_sdr must be non-decreasing for every curve, transfer and range.
 * -------------------------------------------------------------------------- */

static void test_tonemap_lut_monotonic(void)
{
    for (int c = 0; c < 3; c++)
    for (int trc = 0; trc < 2; trc++)
    for (int rng = 0; rng < 2; rng++) {
        gen_luts(&g_state, g_curves[c], 1000, 100,
                 trc ? FUSED_TRC_HLG : FUSED_TRC_PQ, rng, rng);
        for (int i = 1; i < 1024; i++) {
            if (g_state.pq_to_sdr[i] < g_state.pq_to_sdr[i - 1]) {
                printf("\n  FAIL [%s:%d] %s trc=%d rng=%d: lut[%d]=%d < lut[%d]=%d\n",
                       __func__, __LINE__, g_curve_names[c], trc, rng,
                       i, g_state.pq_to_sdr[i], i - 1, g_state.pq_to_sdr[i - 1]);
                g_results.failed++;
                return;
            }
        }
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 2. test_tonemap_highlights_compress
 *    Anti-regression for the normalize-to-target defect: the curve must
 *    keep compressing above target_nits, not clip there.  With peak=1000
 *    and target=100, content at 150/200/500/950 nits must remain strictly
 *    ordered - a plateau anywhere below peak means the tone map collapsed
 *    to identity-then-clip again.
 * -------------------------------------------------------------------------- */

static void test_tonemap_highlights_compress(void)
{
    for (int c = 0; c < 3; c++) {
        gen_luts(&g_state, g_curves[c], 1000, 100, FUSED_TRC_PQ,
                 FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED);
        const uint8_t *lut = g_state.pq_to_sdr;
        int v100 = lut[pq_code(100.0, 0)];
        int v150 = lut[pq_code(150.0, 0)];
        int v200 = lut[pq_code(200.0, 0)];
        int v500 = lut[pq_code(500.0, 0)];
        int v950 = lut[pq_code(950.0, 0)];
        if (!(v100 < v150 && v150 < v200 && v200 < v500 && v500 < v950)) {
            printf("\n  FAIL [%s:%d] %s: highlight plateau: "
                   "100:%d 150:%d 200:%d 500:%d 950:%d\n",
                   __func__, __LINE__, g_curve_names[c],
                   v100, v150, v200, v500, v950);
            g_results.failed++;
            return;
        }
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 3. test_tonemap_peak_sensitivity
 *    Anti-regression for the Hable normalization cancellation that made
 *    peak_nits a complete no-op: changing peak_nits must change the LUT
 *    for every curve (PQ and HLG).
 * -------------------------------------------------------------------------- */

static void test_tonemap_peak_sensitivity(void)
{
    for (int c = 0; c < 3; c++)
    for (int trc = 0; trc < 2; trc++) {
        int t = trc ? FUSED_TRC_HLG : FUSED_TRC_PQ;
        gen_luts(&g_state,  g_curves[c], 1000, 100, t,
                 FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED);
        gen_luts(&g_state2, g_curves[c], 4000, 100, t,
                 FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED);
        if (memcmp(g_state.pq_to_sdr, g_state2.pq_to_sdr, 1024) == 0) {
            printf("\n  FAIL [%s:%d] %s trc=%d: peak=1000 and peak=4000 "
                   "produce identical LUTs (peak_nits is a no-op)\n",
                   __func__, __LINE__, g_curve_names[c], trc);
            g_results.failed++;
            return;
        }
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 4. test_tonemap_shadow_detail
 *    Anti-regression for the linearly-indexed shadow LUT: 0.5 nits must
 *    stay above black, and no two adjacent codes may jump more than a few
 *    output codes (the old table jumped 55 between its first two entries).
 * -------------------------------------------------------------------------- */

static void test_tonemap_shadow_detail(void)
{
    for (int c = 0; c < 3; c++)
    for (int rng = 0; rng < 2; rng++) {
        gen_luts(&g_state, g_curves[c], 1000, 100, FUSED_TRC_PQ, rng, rng);
        const uint8_t *lut = g_state.pq_to_sdr;
        int black = rng ? 0 : 16;

        if (lut[pq_code(0.5, rng)] <= black) {
            printf("\n  FAIL [%s:%d] %s rng=%d: 0.5 nits maps to black (%d)\n",
                   __func__, __LINE__, g_curve_names[c], rng,
                   lut[pq_code(0.5, rng)]);
            g_results.failed++;
            return;
        }

        for (int i = 1; i < 1024; i++) {
            int step = lut[i] - lut[i - 1];
            if (step > 3) {
                printf("\n  FAIL [%s:%d] %s rng=%d: lut[%d]->%d jumps %d codes "
                       "(shadow posterization)\n",
                       __func__, __LINE__, g_curve_names[c], rng, i - 1, i, step);
                g_results.failed++;
                return;
            }
        }
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 5. test_tonemap_range_flags
 *    Limited-range input must put black at code 64 and clip at 940;
 *    limited-range output must land on 16..235.  Full/full must use the
 *    whole code space.  The two must differ.
 * -------------------------------------------------------------------------- */

static void test_tonemap_range_flags(void)
{
    gen_luts(&g_state, FUSED_TONEMAP_HABLE, 1000, 100, FUSED_TRC_PQ,
             FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED);
    gen_luts(&g_state2, FUSED_TONEMAP_HABLE, 1000, 100, FUSED_TRC_PQ,
             FUSED_RANGE_FULL, FUSED_RANGE_FULL);

    const uint8_t *lim = g_state.pq_to_sdr;
    const uint8_t *ful = g_state2.pq_to_sdr;

    TEST_ASSERT_EQ(lim[0], 16,  "limited: sub-black codes map to black level 16");
    TEST_ASSERT_EQ(lim[64], 16, "limited: code 64 is black level 16");
    TEST_ASSERT_EQ(lim[940], lim[1023], "limited: codes above 940 clip");
    TEST_ASSERT_EQ(lim[1023], 235, "limited: white is 235");
    TEST_ASSERT_EQ(ful[0], 0,   "full: code 0 is 0");
    TEST_ASSERT_EQ(ful[1023], 255, "full: top code is 255");
    TEST_ASSERT(memcmp(lim, ful, 1024) != 0, "limited and full LUTs differ");

    /* Chroma encode constants must track dst_range */
    TEST_ASSERT_EQ(g_state.chroma_out_min, 16,   "limited chroma min");
    TEST_ASSERT_EQ(g_state.chroma_out_max, 240,  "limited chroma max");
    TEST_ASSERT_EQ(g_state2.chroma_out_min, 0,   "full chroma min");
    TEST_ASSERT_EQ(g_state2.chroma_out_max, 255, "full chroma max");

    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 6. test_tonemap_gray_neutral
 *    Anti-regression for chroma reconstruction sign/scale errors: a gray
 *    ramp (Cb = Cr = 512) must come out with U = V = exactly 128 and a
 *    non-decreasing luma row.  Exercises fused_tonemap_apply directly.
 * -------------------------------------------------------------------------- */

static void test_tonemap_gray_neutral(void)
{
    enum { W = 64, H = 4 };
    static uint16_t src_y[W * H], src_u[(W / 2) * (H / 2)], src_v[(W / 2) * (H / 2)];
    static uint8_t  dst_y[W * H], dst_u[(W / 2) * (H / 2)], dst_v[(W / 2) * (H / 2)];

    gen_luts(&g_state, FUSED_TONEMAP_HABLE, 1000, 100, FUSED_TRC_PQ,
             FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED);

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            src_y[y * W + x] = (uint16_t)(64 + (876 * x) / (W - 1));
    for (int i = 0; i < (W / 2) * (H / 2); i++) {
        src_u[i] = 512;
        src_v[i] = 512;
    }

    fused_tonemap_apply(&g_state,
                        src_y, W * 2, src_u, (W / 2) * 2, src_v,
                        dst_y, W, dst_u, W / 2, dst_v,
                        W, H);

    for (int i = 0; i < (W / 2) * (H / 2); i++) {
        if (dst_u[i] != 128 || dst_v[i] != 128) {
            printf("\n  FAIL [%s:%d] gray input: U=%d V=%d at %d (expected 128/128)\n",
                   __func__, __LINE__, dst_u[i], dst_v[i], i);
            g_results.failed++;
            return;
        }
    }
    for (int y = 0; y < H; y++) {
        for (int x = 1; x < W; x++) {
            if (dst_y[y * W + x] < dst_y[y * W + x - 1]) {
                printf("\n  FAIL [%s:%d] gray ramp luma not monotonic at (%d,%d)\n",
                       __func__, __LINE__, x, y);
                g_results.failed++;
                return;
            }
        }
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 7. test_tonemap_golden_pixels
 *    Full-pipeline golden reference: encode known linear BT.2020 colors to
 *    limited-range NCL YCbCr, tone map them through fused_tonemap_apply,
 *    and compare against an independent double-precision implementation of
 *    the same pipeline (gamma-domain NCL inverse, per-channel BT.2390
 *    EETF, gamma-domain gamut conversion, BT.709 re-encode).
 * -------------------------------------------------------------------------- */

static double ref_bt2390(double nits, double peak, double target)
{
    double src_max = ref_pq_encode(peak);
    double dst_max = ref_pq_encode(target);
    double max_lum = dst_max / src_max;
    double e1 = ref_pq_encode(nits) / src_max;
    if (e1 > 1.0) e1 = 1.0;
    double ks = 1.5 * max_lum - 0.5;
    double e2 = e1;
    if (e1 > ks && ks < 1.0) {
        double t = (e1 - ks) / (1.0 - ks), t2 = t * t, t3 = t2 * t;
        e2 = (2*t3 - 3*t2 + 1) * ks + (t3 - 2*t2 + t) * (1.0 - ks)
           + (-2*t3 + 3*t2) * max_lum;
    }
    if (e2 > max_lum) e2 = max_lum;
    double v = ref_pq_decode(e2 * src_max) / target;
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

static double ref_sdr_oetf(double L)
{
    return (L >= 0.0031308) ? 1.055 * pow(L, 1.0 / 2.4) - 0.055 : 12.92 * L;
}

static void test_tonemap_golden_pixels(void)
{
    /* Linear-light BT.2020 RGB test colors, in nits */
    static const struct { double r, g, b; } colors[] = {
        { 100, 100, 100 },   /* SDR white */
        { 200, 200, 200 },   /* 2x target gray */
        { 200,   0,   0 },   /* red */
        {   0, 200,   0 },   /* green */
        {   0,   0, 200 },   /* blue */
        { 200, 200,   0 },   /* yellow */
        {  10,  30,  60 },   /* dim blue-ish mix */
        { 800, 400, 100 },   /* bright highlight mix */
    };
    const double peak = 1000.0, target = 100.0;
    const double KR = 0.2627, KG = 0.6780, KB = 0.0593;

    gen_luts(&g_state, FUSED_TONEMAP_BT2390, (int)peak, (int)target,
             FUSED_TRC_PQ, FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED);

    for (unsigned ci = 0; ci < sizeof(colors) / sizeof(colors[0]); ci++) {
        double R = colors[ci].r, G = colors[ci].g, B = colors[ci].b;

        /* Encode: gamma-domain BT.2020 NCL, limited range 10-bit */
        double Rp = ref_pq_encode(R), Gp = ref_pq_encode(G), Bp = ref_pq_encode(B);
        double Yp = KR * Rp + KG * Gp + KB * Bp;
        int y_code  = 64 + (int)(Yp * 876.0 + 0.5);
        int cb_code = 512 + (int)((Bp - Yp) / 1.8814 * 896.0 +
                                  ((Bp >= Yp) ? 0.5 : -0.5));
        int cr_code = 512 + (int)((Rp - Yp) / 1.4746 * 896.0 +
                                  ((Rp >= Yp) ? 0.5 : -0.5));

        /* Reference pipeline, double precision all the way */
        double yn  = (y_code - 64) / 876.0;
        double rn  = yn + 1.4746 * (cr_code - 512) / 896.0;
        double bn  = yn + 1.8814 * (cb_code - 512) / 896.0;
        double gn  = (yn - KR * rn - KB * bn) / KG;
        double ch3[3] = { rn, gn, bn };
        double sdr[3];
        for (int k = 0; k < 3; k++) {
            double n = ch3[k] < 0.0 ? 0.0 : (ch3[k] > 1.0 ? 1.0 : ch3[k]);
            double nits = ref_pq_decode(n);
            sdr[k] = ref_sdr_oetf(ref_bt2390(nits, peak, target));
        }
        double r8 = 16.0 + 219.0 * sdr[0];
        double g8 = 16.0 + 219.0 * sdr[1];
        double b8 = 16.0 + 219.0 * sdr[2];
        double y_ref  = KR * r8 + KG * g8 + KB * b8;   /* == BT.709 luma of
                                                          gamut-mapped RGB */
        double r709 = 1.6605 * r8 - 0.5876 * g8 - 0.0728 * b8;
        double b709 = -0.0182 * r8 - 0.1006 * g8 + 1.1187 * b8;
        if (r709 < 0) r709 = 0;
        if (r709 > 255) r709 = 255;
        if (b709 < 0) b709 = 0;
        if (b709 > 255) b709 = 255;
        double cscale = 224.0 / 219.0;
        double u_ref = 128.0 + (b709 - y_ref) * cscale * 0.5 / (1.0 - 0.0722);
        double v_ref = 128.0 + (r709 - y_ref) * cscale * 0.5 / (1.0 - 0.2126);
        /* limited-range chroma legal excursion */
        if (u_ref < 16) u_ref = 16;
        if (u_ref > 240) u_ref = 240;
        if (v_ref < 16) v_ref = 16;
        if (v_ref > 240) v_ref = 240;

        /* Run the implementation on a 2x2 frame of this color */
        uint16_t src_y[4], src_u[1], src_v[1];
        uint8_t  dst_y[4], dst_u[1], dst_v[1];
        for (int i = 0; i < 4; i++) src_y[i] = (uint16_t)y_code;
        src_u[0] = (uint16_t)cb_code;
        src_v[0] = (uint16_t)cr_code;

        fused_tonemap_apply(&g_state,
                            src_y, 2 * 2, src_u, 1 * 2, src_v,
                            dst_y, 2, dst_u, 1, dst_v,
                            2, 2);

        if (fabs((double)dst_y[0] - y_ref) > 3.0 ||
            fabs((double)dst_u[0] - u_ref) > 3.0 ||
            fabs((double)dst_v[0] - v_ref) > 3.0) {
            printf("\n  FAIL [%s:%d] color %u (%.0f/%.0f/%.0f nits): "
                   "got Y=%d U=%d V=%d, reference Y=%.1f U=%.1f V=%.1f\n",
                   __func__, __LINE__, ci, R, G, B,
                   dst_y[0], dst_u[0], dst_v[0], y_ref, u_ref, v_ref);
            g_results.failed++;
            return;
        }
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 8. test_tonemap_hue_sanity
 *    Anti-regression for wrong-domain G reconstruction and gamut sign
 *    errors, which preserve grays but corrupt saturated colors: primary
 *    colors must land on the correct side of the U/V axes.
 * -------------------------------------------------------------------------- */

static void test_tonemap_hue_sanity(void)
{
    static const struct {
        double r, g, b;
        const char *name;
        int u_sign;   /* expected sign of U - 128 */
        int v_sign;   /* expected sign of V - 128 */
    } cases[] = {
        { 200,   0,   0, "red",    -1, +1 },
        {   0, 200,   0, "green",  -1, -1 },
        {   0,   0, 200, "blue",   +1, -1 },
        { 200, 200,   0, "yellow", -1, +1 },
    };
    const double KR = 0.2627, KG = 0.6780, KB = 0.0593;

    gen_luts(&g_state, FUSED_TONEMAP_HABLE, 1000, 100, FUSED_TRC_PQ,
             FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED);

    for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        double Rp = ref_pq_encode(cases[ci].r);
        double Gp = ref_pq_encode(cases[ci].g);
        double Bp = ref_pq_encode(cases[ci].b);
        double Yp = KR * Rp + KG * Gp + KB * Bp;
        int y_code  = 64 + (int)(Yp * 876.0 + 0.5);
        int cb_code = 512 + (int)((Bp - Yp) / 1.8814 * 896.0 +
                                  ((Bp >= Yp) ? 0.5 : -0.5));
        int cr_code = 512 + (int)((Rp - Yp) / 1.4746 * 896.0 +
                                  ((Rp >= Yp) ? 0.5 : -0.5));

        uint16_t src_y[4], src_u[1], src_v[1];
        uint8_t  dst_y[4], dst_u[1], dst_v[1];
        for (int i = 0; i < 4; i++) src_y[i] = (uint16_t)y_code;
        src_u[0] = (uint16_t)cb_code;
        src_v[0] = (uint16_t)cr_code;

        fused_tonemap_apply(&g_state,
                            src_y, 2 * 2, src_u, 1 * 2, src_v,
                            dst_y, 2, dst_u, 1, dst_v,
                            2, 2);

        int du = (int)dst_u[0] - 128;
        int dv = (int)dst_v[0] - 128;
        /* Saturated primaries should be well clear of neutral */
        if (du * cases[ci].u_sign < 10 || dv * cases[ci].v_sign < 10) {
            printf("\n  FAIL [%s:%d] %s: U=%d V=%d (expected U%c, V%c, "
                   "magnitude >= 10)\n",
                   __func__, __LINE__, cases[ci].name, dst_u[0], dst_v[0],
                   cases[ci].u_sign > 0 ? '+' : '-',
                   cases[ci].v_sign > 0 ? '+' : '-');
            g_results.failed++;
            return;
        }
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 9. test_tonemap_e2e_colorbars
 *    End-to-end through the public API: tone-map PQ color bars with
 *    tonemap_1x and check the classic luminance ordering of the bars plus
 *    neutral chroma on the white and black bars.
 * -------------------------------------------------------------------------- */

static void test_tonemap_e2e_colorbars(void)
{
    test_hdr_frame_t frame;
    int r = test_hdr_frame_create(&frame, 256, 144, PATTERN_PQ_COLORBARS, 0);
    TEST_ASSERT(r == 0, "test_hdr_frame_create failed");

    fused_hdr_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src_width      = frame.width;
    ctx.src_height     = frame.height;
    ctx.src_y_stride   = frame.y_stride;
    ctx.src_uv_stride  = frame.uv_stride;
    ctx.src_format     = FUSED_PIX_I010;
    ctx.src_transfer   = FUSED_TRC_PQ;
    ctx.tonemap_1x     = 1;
    ctx.tonemap.curve  = FUSED_TONEMAP_BT2390;
    ctx.log_warnings.target = FUSED_LOG_SUPPRESS;
    ctx.log_errors.target   = FUSED_LOG_SUPPRESS;

    int rc = fused_hdr_init(&ctx);
    TEST_ASSERT(rc >= 0, "fused_hdr_init failed");

    fused_hdr_run(&ctx, frame.plane_y, frame.plane_u, frame.plane_v);

    TEST_ASSERT(ctx.output_1x.plane_y != NULL, "output_1x populated");

    /* Sample the center of each bar: white yellow cyan green magenta red
     * blue black, sampled mid-height. */
    int bar_w = frame.width / 8;
    int row = frame.height / 2;
    int y_smp[8], u_smp[8], v_smp[8];
    for (int b = 0; b < 8; b++) {
        int x = b * bar_w + bar_w / 2;
        y_smp[b] = ctx.output_1x.plane_y[row * ctx.output_1x.y_stride + x];
        u_smp[b] = ctx.output_1x.plane_u[(row / 2) * ctx.output_1x.uv_stride + x / 2];
        v_smp[b] = ctx.output_1x.plane_v[(row / 2) * ctx.output_1x.uv_stride + x / 2];
    }

    /* Standard bar luminance ordering */
    int ordered = 1;
    for (int b = 1; b < 8; b++)
        if (y_smp[b] >= y_smp[b - 1]) ordered = 0;
    if (!ordered) {
        printf("\n  FAIL [%s:%d] bar luma not in descending order: "
               "%d %d %d %d %d %d %d %d\n", __func__, __LINE__,
               y_smp[0], y_smp[1], y_smp[2], y_smp[3],
               y_smp[4], y_smp[5], y_smp[6], y_smp[7]);
        g_results.failed++;
        fused_hdr_free(&ctx);
        test_hdr_frame_free(&frame);
        return;
    }

    /* White and black bars are neutral */
    TEST_ASSERT(abs(u_smp[0] - 128) <= 2 && abs(v_smp[0] - 128) <= 2,
                "white bar chroma neutral");
    TEST_ASSERT(abs(u_smp[7] - 128) <= 2 && abs(v_smp[7] - 128) <= 2,
                "black bar chroma neutral");
    /* Black bar at black level (limited-range default) */
    TEST_ASSERT(y_smp[7] <= 17, "black bar at black level");
    /* 200-nit white bar above SDR mid-gray but below clip */
    TEST_ASSERT(y_smp[0] > 180 && y_smp[0] < 235, "white bar in SDR highlights");

    fused_hdr_free(&ctx);
    test_hdr_frame_free(&frame);
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * 10. test_tonemap_simd_parity
 *     The dispatched implementation (AVX-512/AVX2 when available) must
 *     match the scalar reference bit for bit, for I010 and P010, across
 *     chunk-multiple and ragged widths, on random data including
 *     out-of-range codes.  On platforms without a SIMD tone map kernel
 *     dispatch falls through to scalar and this passes trivially.
 * -------------------------------------------------------------------------- */

static uint32_t tm_rand(uint32_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

static void test_tonemap_simd_parity(void)
{
    static const struct { int w, h; } sizes[] = {
        { 64, 2 },      /* single full chunk, minimum height */
        { 66, 4 },      /* 1 chunk + 1-sample tail */
        { 250, 62 },    /* 3 chunks + 29-sample tail */
        { 1920, 32 },   /* chunk-multiple row */
    };
    static const struct { int curve, src_range, dst_range; } cfgs[] = {
        { FUSED_TONEMAP_HABLE,  FUSED_RANGE_LIMITED, FUSED_RANGE_LIMITED },
        { FUSED_TONEMAP_BT2390, FUSED_RANGE_FULL,    FUSED_RANGE_FULL    },
    };
    uint32_t seed = 0xC0FFEEu;

    for (unsigned si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++)
    for (unsigned ci = 0; ci < sizeof(cfgs) / sizeof(cfgs[0]); ci++) {
        int w = sizes[si].w, h = sizes[si].h;
        int cw = w / 2, ch = h / 2;
        /* strides with padding to catch pitch handling */
        int sy_pitch  = w + 8;            /* uint16 elements */
        int suv_pitch = w + 8;            /* covers planar (cw) and P010 (2*cw) */
        int dy_stride  = w + 16;          /* bytes */
        int duv_stride = cw + 16;

        uint16_t *src_y  = malloc((size_t)sy_pitch * h * 2);
        uint16_t *src_u  = malloc((size_t)suv_pitch * ch * 2);
        uint16_t *src_v  = malloc((size_t)suv_pitch * ch * 2);
        uint16_t *src_uv = malloc((size_t)suv_pitch * ch * 2);
        uint8_t  *ref_y = malloc((size_t)dy_stride * h);
        uint8_t  *ref_u = malloc((size_t)duv_stride * ch);
        uint8_t  *ref_v = malloc((size_t)duv_stride * ch);
        uint8_t  *got_y = malloc((size_t)dy_stride * h);
        uint8_t  *got_u = malloc((size_t)duv_stride * ch);
        uint8_t  *got_v = malloc((size_t)duv_stride * ch);
        if (!src_y || !src_u || !src_v || !src_uv ||
            !ref_y || !ref_u || !ref_v || !got_y || !got_u || !got_v) {
            printf("\n  FAIL [%s:%d] allocation failed\n", __func__, __LINE__);
            g_results.failed++;
            return;
        }

        /* Random data, full 16-bit (both paths mask to 10 bits) */
        for (int i = 0; i < sy_pitch * h; i++)
            src_y[i] = (uint16_t)tm_rand(&seed);
        for (int i = 0; i < suv_pitch * ch; i++) {
            src_u[i]  = (uint16_t)tm_rand(&seed);
            src_v[i]  = (uint16_t)tm_rand(&seed);
            src_uv[i] = (uint16_t)tm_rand(&seed);
        }

        gen_luts(&g_state, cfgs[ci].curve, 1000, 100, FUSED_TRC_PQ,
                 cfgs[ci].src_range, cfgs[ci].dst_range);

        int fail = 0;

        /* I010 */
        memset(ref_y, 0xAA, (size_t)dy_stride * h);
        memset(got_y, 0x55, (size_t)dy_stride * h);
        memset(ref_u, 0xAA, (size_t)duv_stride * ch);
        memset(got_u, 0x55, (size_t)duv_stride * ch);
        memset(ref_v, 0xAA, (size_t)duv_stride * ch);
        memset(got_v, 0x55, (size_t)duv_stride * ch);
        fused_tonemap_apply_scalar(&g_state,
                                   src_y, sy_pitch * 2,
                                   src_u, suv_pitch * 2, src_v,
                                   ref_y, dy_stride, ref_u, duv_stride, ref_v,
                                   w, h);
        fused_tonemap_apply(&g_state,
                            src_y, sy_pitch * 2,
                            src_u, suv_pitch * 2, src_v,
                            got_y, dy_stride, got_u, duv_stride, got_v,
                            w, h);
        for (int y = 0; y < h && !fail; y++)
            if (memcmp(ref_y + y * dy_stride, got_y + y * dy_stride, w))
                fail = 1;
        for (int y = 0; y < ch && !fail; y++)
            if (memcmp(ref_u + y * duv_stride, got_u + y * duv_stride, cw) ||
                memcmp(ref_v + y * duv_stride, got_v + y * duv_stride, cw))
                fail = 2;
        if (fail) {
            printf("\n  FAIL [%s:%d] I010 %dx%d cfg=%u: scalar/SIMD mismatch "
                   "(plane %s)\n", __func__, __LINE__, w, h, ci,
                   fail == 1 ? "Y" : "U/V");
            g_results.failed++;
            goto cleanup;
        }

        /* P010 */
        memset(ref_y, 0xAA, (size_t)dy_stride * h);
        memset(got_y, 0x55, (size_t)dy_stride * h);
        memset(ref_u, 0xAA, (size_t)duv_stride * ch);
        memset(got_u, 0x55, (size_t)duv_stride * ch);
        memset(ref_v, 0xAA, (size_t)duv_stride * ch);
        memset(got_v, 0x55, (size_t)duv_stride * ch);
        fused_tonemap_apply_p010_scalar(&g_state,
                                        src_y, sy_pitch * 2,
                                        src_uv, suv_pitch * 2,
                                        ref_y, dy_stride, ref_u, duv_stride,
                                        ref_v, w, h);
        fused_tonemap_apply_p010(&g_state,
                                 src_y, sy_pitch * 2,
                                 src_uv, suv_pitch * 2,
                                 got_y, dy_stride, got_u, duv_stride, got_v,
                                 w, h);
        for (int y = 0; y < h && !fail; y++)
            if (memcmp(ref_y + y * dy_stride, got_y + y * dy_stride, w))
                fail = 1;
        for (int y = 0; y < ch && !fail; y++)
            if (memcmp(ref_u + y * duv_stride, got_u + y * duv_stride, cw) ||
                memcmp(ref_v + y * duv_stride, got_v + y * duv_stride, cw))
                fail = 2;
        if (fail) {
            printf("\n  FAIL [%s:%d] P010 %dx%d cfg=%u: scalar/SIMD mismatch "
                   "(plane %s)\n", __func__, __LINE__, w, h, ci,
                   fail == 1 ? "Y" : "U/V");
            g_results.failed++;
            goto cleanup;
        }

        free(src_y); free(src_u); free(src_v); free(src_uv);
        free(ref_y); free(ref_u); free(ref_v);
        free(got_y); free(got_u); free(got_v);
        continue;

cleanup:
        free(src_y); free(src_u); free(src_v); free(src_uv);
        free(ref_y); free(ref_u); free(ref_v);
        free(got_y); free(got_u); free(got_v);
        return;
    }
    TEST_PASS();
}

/* --------------------------------------------------------------------------
 * run_tonemap_tests
 * -------------------------------------------------------------------------- */

void run_tonemap_tests(void)
{
    RUN_TEST(test_tonemap_lut_monotonic);
    RUN_TEST(test_tonemap_highlights_compress);
    RUN_TEST(test_tonemap_peak_sensitivity);
    RUN_TEST(test_tonemap_shadow_detail);
    RUN_TEST(test_tonemap_range_flags);
    RUN_TEST(test_tonemap_gray_neutral);
    RUN_TEST(test_tonemap_golden_pixels);
    RUN_TEST(test_tonemap_hue_sanity);
    RUN_TEST(test_tonemap_e2e_colorbars);
    RUN_TEST(test_tonemap_simd_parity);
}
