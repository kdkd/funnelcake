/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "tonemap.h"
#include "internal.h"
#include "log.h"
#include "funnelcake.h"
#include "detect.h"

#include <math.h>
#include <string.h>
#include <stdint.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif


/* --------------------------------------------------------------------------
 * ST 2084 (PQ) EOTF constants
 * -------------------------------------------------------------------------- */

static const double PQ_M1 = 0.1593017578125;       /* 2610 / 16384  */
static const double PQ_M2 = 78.84375;              /* 2523 / 32 * 128 */
static const double PQ_C1 = 0.8359375;             /* 3424 / 4096    */
static const double PQ_C2 = 18.8515625;            /* 2413 / 128     */
static const double PQ_C3 = 18.6875;               /* 2392 / 128     */


/* --------------------------------------------------------------------------
 * HLG OETF inverse constants (ARIB STD-B67)
 * -------------------------------------------------------------------------- */

static const double HLG_A = 0.17883277;
static const double HLG_B = 0.28466892;            /* 1 - 4*a        */
static const double HLG_C = 0.55991073;            /* 0.5 - a*ln(4a) */


/* --------------------------------------------------------------------------
 * Hable (Uncharted 2) filmic constants
 * -------------------------------------------------------------------------- */

static const double HABLE_A = 0.15;
static const double HABLE_B = 0.50;
static const double HABLE_C = 0.10;
static const double HABLE_D = 0.20;
static const double HABLE_E = 0.02;
static const double HABLE_F = 0.30;


/* --------------------------------------------------------------------------
 * Chroma reconstruction constants
 *
 * BT.2020 non-constant-luminance (what HDR10/HLG streams actually carry)
 * defines, entirely in the gamma (PQ/HLG signal) domain:
 *   Y'  = Kr*R' + Kg*G' + Kb*B'
 *   Cb  = (B' - Y') / 1.8814
 *   Cr  = (R' - Y') / 1.4746
 *
 * The exact inverse, also gamma-domain:
 *   R'  = Y' + 1.4746 * Cr
 *   B'  = Y' + 1.8814 * Cb
 *   G'  = (Y' - Kr*R' - Kb*B') / Kg
 *       = Y' - (Kr*1.4746*Cr + Kb*1.8814*Cb) / Kg     (1 - Kr - Kb = Kg)
 *
 * All three channels are affine in (Y', Cr, Cb), so reconstruction is three
 * adds of precomputed per-chroma-code deltas - no linear-light detour.
 * -------------------------------------------------------------------------- */

#define NCL_CR_SCALE  1.4746   /* (1 - Kr) * 2 where Kr = 0.2627 */
#define NCL_CB_SCALE  1.8814   /* (1 - Kb) * 2 where Kb = 0.0593 */
#define BT2020_KR     0.2627
#define BT2020_KG     0.6780
#define BT2020_KB     0.0593

/* BT.709 YCbCr encoding constants */
#define BT709_KR      0.2126
#define BT709_KG      0.7152
#define BT709_KB      0.0722


/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

static inline double clamp_d(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline int clamp_i(int x, int lo, int hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}


/* --------------------------------------------------------------------------
 * EOTF: PQ (ST 2084)
 *
 * Input:  normalized PQ signal N in [0, 1]
 * Output: linear luminance in [0, 1] representing [0, 10000] nits
 * -------------------------------------------------------------------------- */

static inline double eotf_pq(double N)
{
    double Nm1 = pow(N, 1.0 / PQ_M2);
    double num = Nm1 - PQ_C1;
    if (num < 0.0) num = 0.0;
    double den = PQ_C2 - PQ_C3 * Nm1;
    return pow(num / den, 1.0 / PQ_M1);
}


/* --------------------------------------------------------------------------
 * Inverse EOTF: PQ (ST 2084)
 *
 * Input:  linear luminance in [0, 1] representing [0, 10000] nits
 * Output: normalized PQ signal in [0, 1]
 * -------------------------------------------------------------------------- */

static inline double oetf_pq(double L)
{
    if (L < 0.0) L = 0.0;
    double Lm1 = pow(L, PQ_M1);
    return pow((PQ_C1 + PQ_C2 * Lm1) / (1.0 + PQ_C3 * Lm1), PQ_M2);
}


/* --------------------------------------------------------------------------
 * EOTF: HLG (ARIB STD-B67 inverse OETF + OOTF)
 *
 * Input:  normalized HLG signal E in [0, 1]
 * Output: display luminance in [0, 1] (after system gamma 1.2)
 *
 * Note: the BT.2100 HLG OOTF applies the system gamma to the luminance of
 * the frame and scales R, G, B by the resulting ratio.  Applying it per
 * channel/per signal as done here is the common fast approximation (it is
 * exact for achromatic content and slightly oversaturates strong colors).
 * -------------------------------------------------------------------------- */

static inline double eotf_hlg(double E)
{
    double L;
    if (E <= 0.5) {
        L = (E * E) / 3.0;
    } else {
        L = (exp((E - HLG_C) / HLG_A) + HLG_B) / 12.0;
    }
    /* OOTF system gamma */
    return pow(L, 1.2);
}


/* --------------------------------------------------------------------------
 * Hable filmic tone curve
 * -------------------------------------------------------------------------- */

static inline double hable_curve(double x)
{
    return ((x * (HABLE_A * x + HABLE_C * HABLE_B) + HABLE_D * HABLE_E) /
            (x * (HABLE_A * x + HABLE_B) + HABLE_D * HABLE_F)) - HABLE_E / HABLE_F;
}


/* --------------------------------------------------------------------------
 * SDR output OETF (linear -> gamma)
 *
 * Input:  display-relative linear luminance in [0, 1]
 * Output: gamma-corrected signal in [0, 1]
 *
 * This is the sRGB encoding curve (IEC 61966-2-1), not the BT.709 camera
 * OETF.  For display-referred SDR output - encoding values that should
 * reproduce a given relative luminance on a typical SDR display - the sRGB
 * curve is the standard practical choice; the BT.709 camera OETF assumes a
 * BT.1886 display plus rendering-intent gamma and would look washed out.
 * -------------------------------------------------------------------------- */

static inline double sdr_oetf(double L)
{
    if (L >= 0.0031308)
        return 1.055 * pow(L, 1.0 / 2.4) - 0.055;
    else
        return 12.92 * L;
}


/* --------------------------------------------------------------------------
 * Tone curves: absolute nits -> display-relative linear [0, 1]
 *
 * Each curve compresses [0, peak_nits] smoothly onto [0, 1] where 1.0 is
 * SDR reference white (target_nits).  None of them clip below peak_nits,
 * and peak_nits genuinely shapes all of them - content at peak maps to
 * (or asymptotically approaches) 1.0, content below the curve's knee
 * passes through at or near its true brightness.
 *
 * BT2390: the ITU-R BT.2390 EETF evaluated in PQ space.  Identity below
 *   the knee at KS = 1.5*maxLum - 0.5 (normalized PQ), Hermite-spline
 *   roll-off above.  Reference broadcast behavior: midtones are exact.
 *
 * HABLE: filmic curve with white point at peak.  hable(2x)/hable(2W)
 *   where W = peak/target; the x2 exposure factor limits the filmic
 *   midtone dimming to about -1 stop at default settings.
 *
 * REINHARD: extended Reinhard, x*(1 + x/W^2)/(1 + x).  Maps W exactly to
 *   1.0 and degenerates to the identity when peak == target.
 * -------------------------------------------------------------------------- */

static double tone_curve_bt2390(double nits, double peak_f, double target_f)
{
    double src_max = oetf_pq(peak_f / 10000.0);     /* source range in PQ */
    double dst_max = oetf_pq(target_f / 10000.0);   /* target range in PQ */
    double max_lum = dst_max / src_max;

    double e1 = oetf_pq(nits / 10000.0) / src_max;  /* normalized PQ [0,1] */
    if (e1 > 1.0) e1 = 1.0;     /* content above declared peak clips at peak
                                 * (the spline is only defined on [0,1]) */
    double ks = 1.5 * max_lum - 0.5;                /* knee start */

    double e2;
    if (e1 <= ks || ks >= 1.0) {
        e2 = e1;                                    /* below knee: identity */
    } else {
        double t  = (e1 - ks) / (1.0 - ks);
        double t2 = t * t, t3 = t2 * t;
        e2 = (2.0*t3 - 3.0*t2 + 1.0) * ks
           + (t3 - 2.0*t2 + t) * (1.0 - ks)
           + (-2.0*t3 + 3.0*t2) * max_lum;
    }
    if (e2 > max_lum) e2 = max_lum;

    double l_out = eotf_pq(e2 * src_max) * 10000.0; /* nits, <= target */
    return clamp_d(l_out / target_f, 0.0, 1.0);
}

static double tone_curve_disp(int curve, double nits,
                              double peak_f, double target_f)
{
    double w = peak_f / target_f;   /* peak in SDR-white-relative units */
    double x = nits / target_f;     /* input in SDR-white-relative units */

    switch (curve) {
    case FUSED_TONEMAP_REINHARD:
        return clamp_d(x * (1.0 + x / (w * w)) / (1.0 + x), 0.0, 1.0);

    case FUSED_TONEMAP_BT2390:
        return tone_curve_bt2390(nits, peak_f, target_f);

    case FUSED_TONEMAP_HABLE:
    default:
        return clamp_d(hable_curve(2.0 * x) / hable_curve(2.0 * w), 0.0, 1.0);
    }
}


/* --------------------------------------------------------------------------
 * fused_tonemap_generate_luts
 * -------------------------------------------------------------------------- */

void fused_tonemap_generate_luts(fused_hdr_internal_t *hdr,
                                 int src_transfer,
                                 const fused_tonemap_config_t *tm,
                                 const fused_log_config_t *log_warn)
{
    int peak_nits   = (tm->peak_nits   > 0) ? tm->peak_nits   : 1000;
    int target_nits = (tm->target_nits > 0) ? tm->target_nits : 100;
    int curve       = tm->curve;
    int src_full    = (tm->src_range == FUSED_RANGE_FULL);
    int dst_full    = (tm->dst_range == FUSED_RANGE_FULL);

    /* ------------------------------------------------------------------ */
    /* Custom LUT: bypass all computation, just copy the caller's table.  */
    /* ------------------------------------------------------------------ */

    if (curve == FUSED_TONEMAP_CUSTOM) {
        if (tm->custom_lut) {
            memcpy(hdr->lut_y, tm->custom_lut, 1024);
        } else {
            fused_log(log_warn, FUSED_LOG_WARN,
                "funnelcake: FUSED_TONEMAP_CUSTOM selected but custom_lut is NULL; "
                "falling back to Hable\n");
            curve = FUSED_TONEMAP_HABLE;
        }
    }

    double peak_f   = (double)peak_nits;
    double target_f = (double)target_nits;

    /* 10-bit input quantization: limited range is Y 64..940 (offset 64,
     * span 876) and chroma 64..960 (center 512, span 896); full range
     * spans the whole code space for both. */
    double y_off   = src_full ? 0.0    : 64.0;
    double y_span  = src_full ? 1023.0 : 876.0;
    double c_span  = src_full ? 1023.0 : 896.0;

    /* 8-bit output quantization: limited is 16 + 219*V, full is 255*V. */
    double out_scale = dst_full ? 255.0 : 219.0;
    int    out_off   = dst_full ? 0     : 16;

    /* ------------------------------------------------------------------ */
    /* pq_to_sdr: the entire per-channel pipeline compiled into one table. */
    /*   10-bit code -> signal -> EOTF -> nits -> tone curve -> SDR OETF   */
    /*   -> 8-bit code.                                                    */
    /* Indexing by the PQ/HLG code keeps shadow resolution perfect by      */
    /* construction (the input axis is already perceptually uniform).      */
    /* ------------------------------------------------------------------ */

    for (int i = 0; i < 1024; i++) {
        double N = clamp_d(((double)i - y_off) / y_span, 0.0, 1.0);

        /* EOTF -> absolute nits.  PQ is absolute (1.0 = 10000 nits);
         * HLG is display-referred (1.0 = display peak). */
        double nits;
        if (src_transfer == FUSED_TRC_HLG)
            nits = eotf_hlg(N) * peak_f;
        else
            nits = eotf_pq(N) * 10000.0;

        double V = tone_curve_disp(curve, nits, peak_f, target_f);

        hdr->pq_to_sdr[i] = (uint8_t)clamp_i(
            (int)(sdr_oetf(V) * out_scale + 0.5) + out_off, 0, 255);
    }

    /* For built-in curves the luma LUT is the same table (the built-in
     * path derives luma from reconstructed R'G'B', but lut_y is kept
     * consistent for diagnostics and possible fast paths). */
    if (curve != FUSED_TONEMAP_CUSTOM)
        memcpy(hdr->lut_y, hdr->pq_to_sdr, 1024);

    /* ------------------------------------------------------------------ */
    /* Chroma delta tables: BT.2020 NCL inverse, gamma domain.             */
    /* Deltas are expressed in Y'-code units, so y_span/c_span folds the   */
    /* limited-range chroma-vs-luma excursion difference in at init.       */
    /* ------------------------------------------------------------------ */

    {
        double k = y_span / c_span;
        double g_cr = -(BT2020_KR / BT2020_KG) * NCL_CR_SCALE; /* -0.5714 */
        double g_cb = -(BT2020_KB / BT2020_KG) * NCL_CB_SCALE; /* -0.1646 */

        /* Q15 coefficients are the canonical definition; tables and SIMD
         * kernels both evaluate ((i-512)*coef + 16384) >> 15 so the two
         * paths agree bit for bit. */
        hdr->delta_coef_r    = (int32_t)floor(NCL_CR_SCALE * k * 32768.0 + 0.5);
        hdr->delta_coef_b    = (int32_t)floor(NCL_CB_SCALE * k * 32768.0 + 0.5);
        hdr->delta_coef_g_cr = (int32_t)floor(g_cr * k * 32768.0 + 0.5);
        hdr->delta_coef_g_cb = (int32_t)floor(g_cb * k * 32768.0 + 0.5);

        for (int i = 0; i < 1024; i++) {
            int x = i - 512;
            hdr->pq_r_delta[i]    = (int16_t)((x * hdr->delta_coef_r    + 16384) >> 15);
            hdr->pq_b_delta[i]    = (int16_t)((x * hdr->delta_coef_b    + 16384) >> 15);
            hdr->pq_g_delta_cr[i] = (int16_t)((x * hdr->delta_coef_g_cr + 16384) >> 15);
            hdr->pq_g_delta_cb[i] = (int16_t)((x * hdr->delta_coef_g_cb + 16384) >> 15);
        }
    }

    /* ------------------------------------------------------------------ */
    /* SDR YCbCr re-encode constants (BT.709 matrix, dst_range scaling).   */
    /* Limited-range 8-bit chroma has a 224-step excursion against luma's  */
    /* 219, hence the 224/219 factor.                                      */
    /* ------------------------------------------------------------------ */

    {
        double cscale = dst_full ? 1.0 : 224.0 / 219.0;
        hdr->cb_out_scale = (int)(256.0 * cscale * 0.5 / (1.0 - BT709_KB) + 0.5);
        hdr->cr_out_scale = (int)(256.0 * cscale * 0.5 / (1.0 - BT709_KR) + 0.5);
        hdr->chroma_out_min = dst_full ? 0   : 16;
        hdr->chroma_out_max = dst_full ? 255 : 240;
    }

    fused_log(log_warn, FUSED_LOG_WARN,
        "funnelcake: tone map LUTs generated - transfer=%s curve=%d "
        "peak=%d target=%d range=%s/%s\n",
        (src_transfer == FUSED_TRC_HLG) ? "HLG" : "PQ",
        curve, peak_nits, target_nits,
        src_full ? "full" : "limited", dst_full ? "full" : "limited");
}


/* --------------------------------------------------------------------------
 * SIMD luma tone mapping - exact LUT with batched lookups
 *
 * The luma pass applies: dst[x] = lut_y[src[x] & 0x3FF]
 *
 * The 1024-entry LUT fits in L1 cache (1 KB).  Each lookup is a dependent
 * memory read, but the CPU's out-of-order engine can overlap independent
 * lookups across pixels.  Processing in batches of 16 (NEON) or 16 (AVX2)
 * provides enough independent loads for the OOO engine to keep the memory
 * pipeline full.  A prefetch of the next source row hides latency when the
 * kernel has evicted the output planes from L1.
 * -------------------------------------------------------------------------- */

#if defined(__x86_64__)

/*
 * tonemap_luma_avx2 - exact LUT tone mapping, 16 pixels per iteration.
 *
 * Loads 16 source values, masks to 10 bits, extracts each lane for scalar
 * LUT lookup, packs 16 result bytes, and stores.  The LUT stays L1-hot
 * across the entire frame.
 */
__attribute__((target("avx2")))
static inline __m128i tonemap_luma_8(const uint8_t *lut, __m128i src8)
{
    uint8_t r[8];
    r[0] = lut[(uint16_t)_mm_extract_epi16(src8, 0)];
    r[1] = lut[(uint16_t)_mm_extract_epi16(src8, 1)];
    r[2] = lut[(uint16_t)_mm_extract_epi16(src8, 2)];
    r[3] = lut[(uint16_t)_mm_extract_epi16(src8, 3)];
    r[4] = lut[(uint16_t)_mm_extract_epi16(src8, 4)];
    r[5] = lut[(uint16_t)_mm_extract_epi16(src8, 5)];
    r[6] = lut[(uint16_t)_mm_extract_epi16(src8, 6)];
    r[7] = lut[(uint16_t)_mm_extract_epi16(src8, 7)];
    return _mm_loadl_epi64((const __m128i *)r);
}

__attribute__((target("avx2")))
static void tonemap_luma_avx2(const uint8_t *lut_y,
                               const uint16_t *src_y, int src_y_stride,
                               uint8_t *dst_y, int dst_y_stride,
                               int width, int height)
{
    int src_y_pitch = src_y_stride / (int)sizeof(uint16_t);
    __m256i mask10 = _mm256_set1_epi16(0x3FF);
    int simd_w = width & ~15;

    for (int y = 0; y < height; y++) {
        const uint16_t *sy = src_y + y * src_y_pitch;
        uint8_t        *dy = dst_y + y * dst_y_stride;
        int x = 0;

        /* Prefetch next row into L2 while processing this one */
        if (y + 1 < height)
            _mm_prefetch((const char *)(sy + src_y_pitch), _MM_HINT_T1);

        for (; x < simd_w; x += 16) {
            __m256i val = _mm256_and_si256(
                _mm256_loadu_si256((const __m256i *)(sy + x)), mask10);

            __m128i lo = _mm256_castsi256_si128(val);
            __m128i hi = _mm256_extracti128_si256(val, 1);

            __m128i r0 = tonemap_luma_8(lut_y, lo);
            __m128i r1 = tonemap_luma_8(lut_y, hi);

            _mm_storeu_si128((__m128i *)(dy + x),
                             _mm_unpacklo_epi64(r0, r1));
        }

        for (; x < width; x++)
            dy[x] = lut_y[sy[x] & 0x3FF];
    }

    _mm256_zeroupper();
}

#endif /* __x86_64__ */


#if defined(__aarch64__)

/*
 * tonemap_luma_neon - exact LUT tone mapping, 16 pixels per iteration.
 *
 * Extracts each lane for scalar LUT lookup, packs results into uint8x16_t.
 */
static void tonemap_luma_neon(const uint8_t *lut_y,
                              const uint16_t *src_y, int src_y_stride,
                              uint8_t *dst_y, int dst_y_stride,
                              int width, int height)
{
    int src_y_pitch = src_y_stride / (int)sizeof(uint16_t);
    uint16x8_t mask10 = vdupq_n_u16(0x3FF);
    int simd_w = width & ~15;

    for (int y = 0; y < height; y++) {
        const uint16_t *sy = src_y + y * src_y_pitch;
        uint8_t        *dy = dst_y + y * dst_y_stride;
        int x = 0;

        if (y + 1 < height)
            __builtin_prefetch(sy + src_y_pitch, 0, 2);

        for (; x < simd_w; x += 16) {
            uint16x8_t v0 = vandq_u16(vld1q_u16(sy + x),     mask10);
            uint16x8_t v1 = vandq_u16(vld1q_u16(sy + x + 8), mask10);

            uint8_t buf[16];
            buf[0]  = lut_y[vgetq_lane_u16(v0, 0)];
            buf[1]  = lut_y[vgetq_lane_u16(v0, 1)];
            buf[2]  = lut_y[vgetq_lane_u16(v0, 2)];
            buf[3]  = lut_y[vgetq_lane_u16(v0, 3)];
            buf[4]  = lut_y[vgetq_lane_u16(v0, 4)];
            buf[5]  = lut_y[vgetq_lane_u16(v0, 5)];
            buf[6]  = lut_y[vgetq_lane_u16(v0, 6)];
            buf[7]  = lut_y[vgetq_lane_u16(v0, 7)];
            buf[8]  = lut_y[vgetq_lane_u16(v1, 0)];
            buf[9]  = lut_y[vgetq_lane_u16(v1, 1)];
            buf[10] = lut_y[vgetq_lane_u16(v1, 2)];
            buf[11] = lut_y[vgetq_lane_u16(v1, 3)];
            buf[12] = lut_y[vgetq_lane_u16(v1, 4)];
            buf[13] = lut_y[vgetq_lane_u16(v1, 5)];
            buf[14] = lut_y[vgetq_lane_u16(v1, 6)];
            buf[15] = lut_y[vgetq_lane_u16(v1, 7)];

            vst1q_u8(dy + x, vld1q_u8(buf));
        }

        for (; x < width; x++)
            dy[x] = lut_y[sy[x] & 0x3FF];
    }
}

#endif /* __aarch64__ */


/* --------------------------------------------------------------------------
 * Chroma reconstruction helpers
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * Per-pixel RGB reconstruction helper
 *
 * Given a 10-bit Y' and chroma (Cb, Cr) at a single pixel, reconstruct the
 * gamma-domain BT.2020 R', G', B' (three delta adds - the exact NCL
 * inverse), and tone-map each channel through the shared pq_to_sdr table.
 * Returns 8-bit tone-mapped SDR values, still BT.2020 primaries.
 * -------------------------------------------------------------------------- */

static inline void tonemap_pixel_rgb(
    const fused_hdr_internal_t *state,
    int y_pq, int cb_10, int cr_10,
    int *r_out, int *g_out, int *b_out)
{
    int pq_r = y_pq + state->pq_r_delta[cr_10];
    int pq_g = y_pq + state->pq_g_delta_cr[cr_10] + state->pq_g_delta_cb[cb_10];
    int pq_b = y_pq + state->pq_b_delta[cb_10];
    if (pq_r < 0) pq_r = 0;
    if (pq_r > 1023) pq_r = 1023;
    if (pq_g < 0) pq_g = 0;
    if (pq_g > 1023) pq_g = 1023;
    if (pq_b < 0) pq_b = 0;
    if (pq_b > 1023) pq_b = 1023;

    *r_out = state->pq_to_sdr[pq_r];
    *g_out = state->pq_to_sdr[pq_g];
    *b_out = state->pq_to_sdr[pq_b];
}


/* --------------------------------------------------------------------------
 * BT.709 YCbCr re-encode with BT.2020 -> BT.709 gamut conversion
 *
 * The reconstructed SDR R, G, B above are BT.2020-primary.  The gamut
 * conversion is a 3x3 matrix M (applied here in gamma domain - the
 * standard fast-path approximation), followed by the BT.709 YCbCr encode.
 * Both are dot products, so they compose: we never materialize the
 * BT.709 RGB.
 *
 * Luma comes out especially clean: the BT.709 luma row applied to M
 * algebraically equals the BT.2020 luma coefficients (both reduce to the
 * Y row of the RGB->XYZ matrix for their primaries), so
 *   Y709(M * rgb2020) = [0.2627 0.6780 0.0593] . rgb2020
 * exactly - the gamut conversion costs luma nothing.
 *
 * Chroma needs the R and B rows of M (BT.2020 -> BT.709, linear-domain
 * matrix, applied gamma-domain):
 *   R709 = 1.6605*R - 0.5876*G - 0.0728*B
 *   B709 = -0.0182*R - 0.1006*G + 1.1187*B
 * Fixed-point 8.8 with rows tweaked to sum to exactly 256 so neutral
 * (R=G=B) stays neutral, then Cb/Cr from the usual difference form with
 * dst_range scaling baked into cb/cr_out_scale at init.
 * -------------------------------------------------------------------------- */

static inline int tonemap_rgb_to_y_sdr(int r_sdr, int g_sdr, int b_sdr)
{
    /* BT.2020 luma coefficients * 256: 67 + 174 + 15 == 256 */
    return (67 * r_sdr + 174 * g_sdr + 15 * b_sdr + 128) >> 8;
}

static inline void tonemap_rgb_to_chroma_sdr(
    const fused_hdr_internal_t *state,
    int r_sdr, int g_sdr, int b_sdr, int y_sdr,
    uint8_t *u_out, uint8_t *v_out)
{
    /* Gamut rows * 256, each summing to exactly 256 */
    int r709 = clamp_i((425 * r_sdr - 150 * g_sdr - 19 * b_sdr + 128) >> 8,
                       0, 255);
    int b709 = clamp_i((-5 * r_sdr - 26 * g_sdr + 287 * b_sdr + 128) >> 8,
                       0, 255);

    /* +128 then arithmetic >>8 is round-half-up for either sign */
    int cb_sdr = 128 + (((b709 - y_sdr) * state->cb_out_scale + 128) >> 8);
    int cr_sdr = 128 + (((r709 - y_sdr) * state->cr_out_scale + 128) >> 8);

    *u_out = (uint8_t)clamp_i(cb_sdr, state->chroma_out_min, state->chroma_out_max);
    *v_out = (uint8_t)clamp_i(cr_sdr, state->chroma_out_min, state->chroma_out_max);
}


/* --------------------------------------------------------------------------
 * fused_tonemap_apply - planar I010 chroma
 * -------------------------------------------------------------------------- */


__attribute__((hot))
void fused_tonemap_apply(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    const uint8_t *lut_y          = state->lut_y;

    int chroma_w = width  / 2;
    int chroma_h = height / 2;

    int src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    int src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);

    /* ------------------------------------------------------------------ */
    /* Custom LUT: fast Y-only LUT + simple chroma scaling.                */
    /* Can't do per-channel RGB reconstruction without a known curve.      */
    /* ------------------------------------------------------------------ */

    if (state->is_custom_lut) {
        /* Luma pass */
#if defined(__x86_64__)
        if (fused_detect_cpu()->has_avx2) {
            tonemap_luma_avx2(lut_y, src_y, src_y_stride, dst_y, dst_y_stride,
                              width, height);
        } else
#elif defined(__aarch64__)
        if (fused_detect_cpu()->has_neon) {
            tonemap_luma_neon(lut_y, src_y, src_y_stride, dst_y, dst_y_stride,
                              width, height);
        } else
#endif
        {
            for (int y = 0; y < height; y++) {
                const uint16_t *sy = src_y + y * src_y_pitch;
                uint8_t        *dy = dst_y + y * dst_y_stride;
                for (int x = 0; x < width; x++)
                    dy[x] = lut_y[sy[x] & 0x3FF];
            }
        }
        /* Chroma pass - simple scaling */
        for (int cy = 0; cy < chroma_h; cy++) {
            const uint16_t *su = src_u + cy * src_uv_pitch;
            const uint16_t *sv = src_v + cy * src_uv_pitch;
            uint8_t        *du = dst_u + cy * dst_uv_stride;
            uint8_t        *dv = dst_v + cy * dst_uv_stride;
            for (int cx = 0; cx < chroma_w; cx++) {
                du[cx] = (uint8_t)clamp_i((((int)(su[cx] & 0x3FF) - 512) * 242) >> 8, -128, 127) + 128;
                dv[cx] = (uint8_t)clamp_i((((int)(sv[cx] & 0x3FF) - 512) * 235) >> 8, -128, 127) + 128;
            }
        }
        return;
    }

    /* ------------------------------------------------------------------ */
    /* Built-in curves: SIMD when available, scalar otherwise.             */
    /* ------------------------------------------------------------------ */

#if defined(__x86_64__)
    if (fused_detect_cpu()->has_avx512 && fused_tonemap_avx512_compiled()) {
        fused_tonemap_apply_avx512(state, src_y, src_y_stride,
                                   src_u, src_uv_stride, src_v,
                                   dst_y, dst_y_stride,
                                   dst_u, dst_uv_stride, dst_v,
                                   width, height);
        return;
    }
#endif

    fused_tonemap_apply_scalar(state, src_y, src_y_stride,
                               src_u, src_uv_stride, src_v,
                               dst_y, dst_y_stride,
                               dst_u, dst_uv_stride, dst_v,
                               width, height);
}


/* --------------------------------------------------------------------------
 * fused_tonemap_apply_scalar - built-in-curve reference path
 *
 * RGB reconstruction and tone mapping per luma pixel, then BT.709 YCbCr
 * re-encode.  Chroma is nearest-neighbor shared across each 2x2 luma
 * block, and U/V are written from the block's top-left reconstructed RGB.
 * The SIMD kernels replicate this integer pipeline bit for bit; parity
 * tests compare against this function.
 * -------------------------------------------------------------------------- */

__attribute__((hot))
void fused_tonemap_apply_scalar(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_u,  int src_uv_stride,
    const uint16_t *src_v,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    int chroma_w = width  / 2;
    int chroma_h = height / 2;

    int src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    int src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);

    /* Process one 4:2:0 block at a time, doing chroma while the top-left
     * reconstruction is hot. */
    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *su = src_u + cy * src_uv_pitch;
        const uint16_t *sv = src_v + cy * src_uv_pitch;
        const uint16_t *sy0 = src_y + (cy * 2) * src_y_pitch;
        const uint16_t *sy1 = sy0 + src_y_pitch;
        uint8_t        *dy0 = dst_y + (cy * 2) * dst_y_stride;
        uint8_t        *dy1 = dy0 + dst_y_stride;
        uint8_t        *du = dst_u + cy * dst_uv_stride;
        uint8_t        *dv = dst_v + cy * dst_uv_stride;

        for (int cx = 0; cx < chroma_w; cx++) {
            int cb_10 = su[cx] & 0x3FF;
            int cr_10 = sv[cx] & 0x3FF;
            int x = cx * 2;

            int r_sdr, g_sdr, b_sdr;

            int y_pq = sy0[x] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            int y_sdr = tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr);
            dy0[x] = (uint8_t)clamp_i(y_sdr, 0, 255);
            tonemap_rgb_to_chroma_sdr(state, r_sdr, g_sdr, b_sdr, y_sdr,
                                      &du[cx], &dv[cx]);

            y_pq = sy0[x + 1] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            dy0[x + 1] = (uint8_t)clamp_i(
                tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);

            y_pq = sy1[x] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            dy1[x] = (uint8_t)clamp_i(
                tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);

            y_pq = sy1[x + 1] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            dy1[x + 1] = (uint8_t)clamp_i(
                tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);
        }
    }
}


/* --------------------------------------------------------------------------
 * fused_tonemap_apply_p010 - interleaved P010 chroma
 * -------------------------------------------------------------------------- */

__attribute__((hot))
void fused_tonemap_apply_p010(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    const uint8_t *lut_y          = state->lut_y;

    int chroma_w = width  / 2;
    int chroma_h = height / 2;

    int src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    int src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);

    if (state->is_custom_lut) {
        /* Custom LUT: luma LUT + simple chroma scaling */
#if defined(__x86_64__)
        if (fused_detect_cpu()->has_avx2) {
            tonemap_luma_avx2(lut_y, src_y, src_y_stride, dst_y, dst_y_stride,
                              width, height);
        } else
#elif defined(__aarch64__)
        if (fused_detect_cpu()->has_neon) {
            tonemap_luma_neon(lut_y, src_y, src_y_stride, dst_y, dst_y_stride,
                              width, height);
        } else
#endif
        {
            for (int y = 0; y < height; y++) {
                const uint16_t *sy = src_y + y * src_y_pitch;
                uint8_t *dy = dst_y + y * dst_y_stride;
                for (int x = 0; x < width; x++)
                    dy[x] = lut_y[sy[x] & 0x3FF];
            }
        }
        for (int cy = 0; cy < chroma_h; cy++) {
            const uint16_t *suv = src_uv + cy * src_uv_pitch;
            uint8_t *du = dst_u + cy * dst_uv_stride;
            uint8_t *dv = dst_v + cy * dst_uv_stride;
            for (int cx = 0; cx < chroma_w; cx++) {
                du[cx] = (uint8_t)clamp_i((((int)(suv[cx*2] & 0x3FF) - 512) * 242) >> 8, -128, 127) + 128;
                dv[cx] = (uint8_t)clamp_i((((int)(suv[cx*2+1] & 0x3FF) - 512) * 235) >> 8, -128, 127) + 128;
            }
        }
        return;
    }

#if defined(__x86_64__)
    if (fused_detect_cpu()->has_avx512 && fused_tonemap_avx512_compiled()) {
        fused_tonemap_apply_p010_avx512(state, src_y, src_y_stride,
                                        src_uv, src_uv_stride,
                                        dst_y, dst_y_stride,
                                        dst_u, dst_uv_stride, dst_v,
                                        width, height);
        return;
    }
#endif

    fused_tonemap_apply_p010_scalar(state, src_y, src_y_stride,
                                    src_uv, src_uv_stride,
                                    dst_y, dst_y_stride,
                                    dst_u, dst_uv_stride, dst_v,
                                    width, height);
}


/* --------------------------------------------------------------------------
 * fused_tonemap_apply_p010_scalar - built-in-curve reference path (P010)
 * -------------------------------------------------------------------------- */

__attribute__((hot))
void fused_tonemap_apply_p010_scalar(
    const fused_hdr_internal_t *state,
    const uint16_t *src_y,  int src_y_stride,
    const uint16_t *src_uv, int src_uv_stride,
    uint8_t *dst_y, int dst_y_stride,
    uint8_t *dst_u, int dst_uv_stride,
    uint8_t *dst_v,
    int width, int height)
{
    int chroma_w = width  / 2;
    int chroma_h = height / 2;

    int src_y_pitch  = src_y_stride  / (int)sizeof(uint16_t);
    int src_uv_pitch = src_uv_stride / (int)sizeof(uint16_t);

    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *suv = src_uv + cy * src_uv_pitch;
        const uint16_t *sy0 = src_y  + (cy * 2) * src_y_pitch;
        const uint16_t *sy1 = sy0 + src_y_pitch;
        uint8_t        *dy0 = dst_y  + (cy * 2) * dst_y_stride;
        uint8_t        *dy1 = dy0 + dst_y_stride;
        uint8_t        *du  = dst_u  + cy * dst_uv_stride;
        uint8_t        *dv  = dst_v  + cy * dst_uv_stride;

        for (int cx = 0; cx < chroma_w; cx++) {
            int cb_10 = suv[cx * 2]     & 0x3FF;
            int cr_10 = suv[cx * 2 + 1] & 0x3FF;
            int x = cx * 2;

            int r_sdr, g_sdr, b_sdr;

            int y_pq = sy0[x] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            int y_sdr = tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr);
            dy0[x] = (uint8_t)clamp_i(y_sdr, 0, 255);
            tonemap_rgb_to_chroma_sdr(state, r_sdr, g_sdr, b_sdr, y_sdr,
                                      &du[cx], &dv[cx]);

            y_pq = sy0[x + 1] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            dy0[x + 1] = (uint8_t)clamp_i(
                tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);

            y_pq = sy1[x] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            dy1[x] = (uint8_t)clamp_i(
                tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);

            y_pq = sy1[x + 1] & 0x3FF;
            tonemap_pixel_rgb(state, y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);
            dy1[x + 1] = (uint8_t)clamp_i(
                tonemap_rgb_to_y_sdr(r_sdr, g_sdr, b_sdr), 0, 255);
        }
    }
}
