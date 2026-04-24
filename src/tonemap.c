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

#if defined(__GNUC__) || defined(__clang__)
#define FUSED_HOT __attribute__((hot))
#else
#define FUSED_HOT
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
 * BT.2020 -> BT.709 chroma scale factors (fixed-point 8.8)
 *
 * Approximate YCbCr-domain gamut mapping derived from the BT.2020->BT.709
 * RGB conversion matrix.  U (Cb) compresses by ~0.945, V (Cr) by ~0.918.
 * -------------------------------------------------------------------------- */

/* BT.2020->BT.709 chroma scale constants removed - chroma tone mapping now
 * uses per-channel RGB reconstruction instead of YCbCr-domain scaling. */


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
 * EOTF: HLG (ARIB STD-B67 inverse OETF + OOTF)
 *
 * Input:  normalized HLG signal E in [0, 1]
 * Output: display luminance in [0, 1] (after system gamma 1.2)
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
 * BT.709 OETF (linear -> gamma)
 *
 * Input:  linear luminance in [0, 1]
 * Output: gamma-corrected signal in [0, 1]
 * -------------------------------------------------------------------------- */

static inline double bt709_oetf(double L)
{
    if (L >= 0.0031308)
        return 1.055 * pow(L, 1.0 / 2.4) - 0.055;
    else
        return 12.92 * L;
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

    /* ------------------------------------------------------------------ */
    /* Luma LUT generation (skipped for valid custom LUT)                  */
    /* ------------------------------------------------------------------ */

    double peak_f   = (double)peak_nits;
    double target_f = (double)target_nits;

    /* Precompute tone curve constants (used by both luma LUT and linear_to_sdr LUT).
     *
     * Each curve is normalized so that target_nits maps to 1.0 (full SDR white).
     * Content at or below target_nits passes through at approximately full
     * brightness; content above target_nits (true HDR highlights) gets compressed.
     * Without this normalization, SDR-level content appears dimmed because the
     * curve treats target_nits as mid-range. */
    double hable_W     = peak_f / target_f;
    double hable_exposure = 2.0;
    double hable_denom = hable_curve(hable_W * hable_exposure);
    /* Reference point: what the curve gives at target_nits (= SDR white) */
    double hable_ref   = hable_curve(1.0 * hable_exposure) / hable_denom;

    double bt2390_ks     = 1.5 * target_f / peak_f;
    double bt2390_maxlum = peak_f / target_f;
    /* BT.2390 reference: compute what the spline gives at target_nits (x_tc=1.0) */
    double bt2390_ref;
    {
        double x_tc = 1.0;  /* target_nits / target_nits */
        if (x_tc <= bt2390_ks) {
            bt2390_ref = x_tc / bt2390_maxlum;
        } else {
            double t = clamp_d((x_tc - bt2390_ks) / (bt2390_maxlum - bt2390_ks), 0.0, 1.0);
            double t2 = t * t, t3 = t2 * t;
            double p0 = bt2390_ks / bt2390_maxlum, p1 = 1.0;
            double m0 = 1.0 * (bt2390_maxlum - bt2390_ks) / bt2390_maxlum, m1 = 0.0;
            bt2390_ref = (2*t3-3*t2+1)*p0 + (t3-2*t2+t)*m0 + (-2*t3+3*t2)*p1 + (t3-t2)*m1;
        }
    }

    if (curve != FUSED_TONEMAP_CUSTOM) {

        for (int i = 0; i < 1024; i++) {
            double N = (double)i / 1023.0;

            /* Step 1-2: EOTF -> linear luminance [0, 1] */
            double L;
            if (src_transfer == FUSED_TRC_HLG)
                L = eotf_hlg(N);
            else
                L = eotf_pq(N);

            /* Step 3: convert to absolute nits.
             * PQ EOTF output is normalized to 10000 nits (the ST 2084 absolute
             * reference), so L * 10000 gives the actual luminance in nits.
             * HLG EOTF output is display-referred (normalized to 1.0 = display
             * peak), so L * peak_nits gives the actual luminance. */
            double nits;
            if (src_transfer == FUSED_TRC_HLG)
                nits = L * peak_f;
            else
                nits = L * 10000.0;

            /* Step 4: apply tone curve - compress into [0, 1] */
            double mapped;

            switch (curve) {
            case FUSED_TONEMAP_REINHARD: {
                double x_norm = nits / peak_f;  /* [0, 1] */
                double raw = x_norm / (1.0 + x_norm);
                /* Normalize: target_nits -> 1.0 */
                double ref = (target_f / peak_f) / (1.0 + target_f / peak_f);
                mapped = (ref > 0.0) ? clamp_d(raw / ref, 0.0, 1.0) : raw;
                break;
            }

            case FUSED_TONEMAP_BT2390: {
                /* BT.2390 EETF: spline-based knee function */
                double x_tc = nits / target_f;  /* [0, maxlum] */
                double raw;
                if (x_tc <= bt2390_ks) {
                    raw = x_tc / bt2390_maxlum;
                } else {
                    double t = clamp_d((x_tc - bt2390_ks) / (bt2390_maxlum - bt2390_ks), 0.0, 1.0);
                    double t2 = t * t, t3 = t2 * t;
                    double p0 = bt2390_ks / bt2390_maxlum, p1 = 1.0;
                    double m0 = 1.0 * (bt2390_maxlum - bt2390_ks) / bt2390_maxlum, m1 = 0.0;
                    raw = (2*t3-3*t2+1)*p0 + (t3-2*t2+t)*m0 + (-2*t3+3*t2)*p1 + (t3-t2)*m1;
                }
                /* Normalize: target_nits -> 1.0 */
                mapped = (bt2390_ref > 0.0) ? clamp_d(raw / bt2390_ref, 0.0, 1.0) : raw;
                break;
            }

            case FUSED_TONEMAP_HABLE:
            default: {
                double raw = hable_curve(nits / target_f * hable_exposure) / hable_denom;
                /* Normalize: target_nits -> 1.0 (full SDR white) */
                mapped = clamp_d(raw / hable_ref, 0.0, 1.0);
                break;
            }
            }

            /* Step 5: BT.709 OETF (linear -> gamma) */
            double V = bt709_oetf(mapped);

            /* Step 6: quantize to 8-bit */
            hdr->lut_y[i] = (uint8_t)clamp_i((int)(V * 255.0 + 0.5), 0, 255);
        }
    }

    /* ------------------------------------------------------------------ */
    /* PQ-to-linear LUT: 10-bit PQ code -> linear luminance [0, 1]          */
    /*                                                                     */
    /* Used by the chroma tone mapping pass to reconstruct linear-light    */
    /* R, G, B from YCbCr.  Each entry is the result of the PQ EOTF       */
    /* (or HLG EOTF), stored as float.                                    */
    /* ------------------------------------------------------------------ */

    for (int i = 0; i < 1024; i++) {
        double N = (double)i / 1023.0;
        if (src_transfer == FUSED_TRC_HLG)
            hdr->pq_to_linear[i] = (float)eotf_hlg(N);
        else
            hdr->pq_to_linear[i] = (float)eotf_pq(N);
    }

    /* ------------------------------------------------------------------ */
    /* Linear-to-SDR LUT: linear [0, 1] -> 8-bit SDR gamma output          */
    /*                                                                     */
    /* Incorporates: absolute nits scaling, tone curve, BT.709 OETF.       */
    /* Indexed by (linear_value * 4095).  Used by the chroma pass to       */
    /* tone-map individually reconstructed R, G, B channels.               */
    /* ------------------------------------------------------------------ */

    for (int i = 0; i < 4096; i++) {
        double L = (double)i / 4095.0;  /* linear [0, 1] relative to peak */

        /* Convert to absolute nits */
        double nits;
        if (src_transfer == FUSED_TRC_HLG)
            nits = L * peak_f;
        else
            nits = L * 10000.0;

        /* Apply tone curve with normalization (same as luma LUT).
         * Each curve is normalized so target_nits -> 1.0. */
        double mapped;
        switch (curve) {
        case FUSED_TONEMAP_REINHARD: {
            double x_norm = nits / peak_f;
            double raw = x_norm / (1.0 + x_norm);
            double ref = (target_f / peak_f) / (1.0 + target_f / peak_f);
            mapped = (ref > 0.0) ? clamp_d(raw / ref, 0.0, 1.0) : raw;
            break;
        }
        case FUSED_TONEMAP_BT2390: {
            double x_tc = nits / target_f;
            double raw;
            if (x_tc <= bt2390_ks) {
                raw = x_tc / bt2390_maxlum;
            } else {
                double t = clamp_d((x_tc - bt2390_ks) / (bt2390_maxlum - bt2390_ks), 0.0, 1.0);
                double t2 = t * t, t3 = t2 * t;
                double p0 = bt2390_ks / bt2390_maxlum, p1 = 1.0;
                double m0 = 1.0 * (bt2390_maxlum - bt2390_ks) / bt2390_maxlum, m1 = 0.0;
                raw = (2*t3-3*t2+1)*p0 + (t3-2*t2+t)*m0 + (-2*t3+3*t2)*p1 + (t3-t2)*m1;
            }
            mapped = (bt2390_ref > 0.0) ? clamp_d(raw / bt2390_ref, 0.0, 1.0) : raw;
            break;
        }
        default: { /* Hable */
            double raw = hable_curve(nits / target_f * hable_exposure) / hable_denom;
            mapped = clamp_d(raw / hable_ref, 0.0, 1.0);
            break;
        }
        }

        double V = bt709_oetf(mapped);
        hdr->linear_to_sdr[i] = (uint8_t)clamp_i((int)(V * 255.0 + 0.5), 0, 255);
    }

    fused_log(log_warn, FUSED_LOG_WARN,
        "funnelcake: tone map LUTs generated - transfer=%s curve=%d "
        "peak=%d target=%d\n",
        (src_transfer == FUSED_TRC_HLG) ? "HLG" : "PQ",
        curve, peak_nits, target_nits);
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
 * NCL chroma reconstruction helpers
 *
 * HDR10 uses non-constant-luminance (NCL) YCbCr encoding:
 *   Y' = PQ(0.2627*R + 0.6780*G + 0.0593*B)   <- PQ of linear luma
 *   Cb = (PQ(B) - Y') / 1.8814
 *   Cr = (PQ(R) - Y') / 1.4746
 *
 * To recover linear R, G, B for correct per-channel tone mapping:
 *   PQ(R) = Y' + 1.4746 * Cr
 *   PQ(B) = Y' + 1.8814 * Cb
 *   R_lin = EOTF(PQ(R)),  B_lin = EOTF(PQ(B)),  Y_lin = EOTF(Y')
 *   G_lin = (Y_lin - 0.2627*R_lin - 0.0593*B_lin) / 0.6780
 *
 * BT.2020 NCL coefficients for chroma reconstruction:
 * -------------------------------------------------------------------------- */

#define NCL_CR_SCALE  1.4746   /* (1 - Kr) * 2 where Kr = 0.2627 */
#define NCL_CB_SCALE  1.8814   /* (1 - Kb) * 2 where Kb = 0.0593 */
#define BT2020_KR     0.2627
#define BT2020_KG     0.6780
#define BT2020_KB     0.0593

/* BT.709 YCbCr encoding constants (full range) */
#define BT709_KR      0.2126
#define BT709_KG      0.7152
#define BT709_KB      0.0722


/* --------------------------------------------------------------------------
 * fused_tonemap_apply - planar I010 chroma
 * -------------------------------------------------------------------------- */

/* --------------------------------------------------------------------------
 * Per-pixel RGB reconstruction helper
 *
 * Given a PQ-encoded Y and chroma (Cb, Cr) at a single pixel, recover
 * linear R, G, B, tone-map each channel, and return 8-bit BT.709 R, G, B.
 * -------------------------------------------------------------------------- */

static inline void tonemap_pixel_rgb(
    const float *pq_to_linear, const uint8_t *linear_to_sdr,
    int y_pq, int cb_10, int cr_10,
    int *r_out, int *g_out, int *b_out)
{
    double y_norm  = (double)y_pq / 1023.0;
    double cb_norm = (double)(cb_10 - 512) / 1023.0;
    double cr_norm = (double)(cr_10 - 512) / 1023.0;

    int pq_r = (int)((y_norm + NCL_CR_SCALE * cr_norm) * 1023.0 + 0.5);
    int pq_b = (int)((y_norm + NCL_CB_SCALE * cb_norm) * 1023.0 + 0.5);
    if (pq_r < 0) pq_r = 0;
    if (pq_r > 1023) pq_r = 1023;
    if (pq_b < 0) pq_b = 0;
    if (pq_b > 1023) pq_b = 1023;

    float y_lin = pq_to_linear[y_pq];
    float r_lin = pq_to_linear[pq_r];
    float b_lin = pq_to_linear[pq_b];

    float g_lin = (y_lin - (float)BT2020_KR * r_lin
                         - (float)BT2020_KB * b_lin) / (float)BT2020_KG;
    if (g_lin < 0.0f) g_lin = 0.0f;

    int r_idx = (int)(r_lin * 4095.0f + 0.5f); if (r_idx > 4095) r_idx = 4095;
    int g_idx = (int)(g_lin * 4095.0f + 0.5f); if (g_idx > 4095) g_idx = 4095;
    int b_idx = (int)(b_lin * 4095.0f + 0.5f); if (b_idx > 4095) b_idx = 4095;

    *r_out = linear_to_sdr[r_idx];
    *g_out = linear_to_sdr[g_idx];
    *b_out = linear_to_sdr[b_idx];
}


FUSED_HOT
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
    const float   *pq_to_linear   = state->pq_to_linear;
    const uint8_t *linear_to_sdr  = state->linear_to_sdr;

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
    /* Built-in curves: full per-pixel RGB reconstruction at luma          */
    /* resolution.  For each pixel, recover linear R, G, B from the NCL   */
    /* YCbCr encoding, tone-map each channel individually, and encode     */
    /* as BT.709 YCbCr.  Chroma is upsampled to luma resolution           */
    /* (nearest-neighbor: each chroma sample covers a 2×2 luma block).    */
    /*                                                                     */
    /* This produces consistent Y, Cb, Cr with no washout or banding,     */
    /* at the cost of processing every luma pixel through the RGB          */
    /* reconstruction (~6 LUT reads + float arithmetic per pixel).         */
    /* ------------------------------------------------------------------ */

    for (int y = 0; y < height; y++) {
        const uint16_t *sy = src_y + y * src_y_pitch;
        const uint16_t *su = src_u + (y / 2) * src_uv_pitch;
        const uint16_t *sv = src_v + (y / 2) * src_uv_pitch;
        uint8_t        *dy = dst_y + y * dst_y_stride;

        for (int x = 0; x < width; x++) {
            int y_pq  = sy[x] & 0x3FF;
            int cb_10 = su[x / 2] & 0x3FF;
            int cr_10 = sv[x / 2] & 0x3FF;

            int r_sdr, g_sdr, b_sdr;
            tonemap_pixel_rgb(pq_to_linear, linear_to_sdr,
                              y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);

            /* BT.709 luma from tone-mapped RGB */
            dy[x] = (uint8_t)clamp_i(
                (int)(BT709_KR * r_sdr + BT709_KG * g_sdr
                    + BT709_KB * b_sdr + 0.5), 0, 255);
        }
    }

    /* Chroma at chroma resolution - compute from tone-mapped RGB */
    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *su = src_u + cy * src_uv_pitch;
        const uint16_t *sv = src_v + cy * src_uv_pitch;
        const uint16_t *sy = src_y + (cy * 2) * src_y_pitch;
        uint8_t        *du = dst_u + cy * dst_uv_stride;
        uint8_t        *dv = dst_v + cy * dst_uv_stride;

        for (int cx = 0; cx < chroma_w; cx++) {
            int y_pq  = sy[cx * 2] & 0x3FF;
            int cb_10 = su[cx] & 0x3FF;
            int cr_10 = sv[cx] & 0x3FF;

            int r_sdr, g_sdr, b_sdr;
            tonemap_pixel_rgb(pq_to_linear, linear_to_sdr,
                              y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);

            int y_sdr = (int)(BT709_KR * r_sdr + BT709_KG * g_sdr
                            + BT709_KB * b_sdr + 0.5);
            int cb_sdr = (int)((b_sdr - y_sdr) / (2.0 * (1.0 - BT709_KB)) + 128.5);
            int cr_sdr = (int)((r_sdr - y_sdr) / (2.0 * (1.0 - BT709_KR)) + 128.5);

            du[cx] = (uint8_t)clamp_i(cb_sdr, 0, 255);
            dv[cx] = (uint8_t)clamp_i(cr_sdr, 0, 255);
        }
    }
}


/* --------------------------------------------------------------------------
 * fused_tonemap_apply_p010 - interleaved P010 chroma
 * -------------------------------------------------------------------------- */

FUSED_HOT
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
    const float   *pq_to_linear   = state->pq_to_linear;
    const uint8_t *linear_to_sdr  = state->linear_to_sdr;

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

    /* Full per-pixel RGB reconstruction at luma resolution */
    for (int y = 0; y < height; y++) {
        const uint16_t *sy  = src_y  + y * src_y_pitch;
        const uint16_t *suv = src_uv + (y / 2) * src_uv_pitch;
        uint8_t        *dy  = dst_y  + y * dst_y_stride;

        for (int x = 0; x < width; x++) {
            int y_pq  = sy[x] & 0x3FF;
            int cb_10 = suv[(x / 2) * 2]     & 0x3FF;
            int cr_10 = suv[(x / 2) * 2 + 1] & 0x3FF;

            int r_sdr, g_sdr, b_sdr;
            tonemap_pixel_rgb(pq_to_linear, linear_to_sdr,
                              y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);

            dy[x] = (uint8_t)clamp_i(
                (int)(BT709_KR * r_sdr + BT709_KG * g_sdr
                    + BT709_KB * b_sdr + 0.5), 0, 255);
        }
    }

    /* Chroma at chroma resolution */
    for (int cy = 0; cy < chroma_h; cy++) {
        const uint16_t *suv = src_uv + cy * src_uv_pitch;
        const uint16_t *sy  = src_y  + (cy * 2) * src_y_pitch;
        uint8_t        *du  = dst_u  + cy * dst_uv_stride;
        uint8_t        *dv  = dst_v  + cy * dst_uv_stride;

        for (int cx = 0; cx < chroma_w; cx++) {
            int y_pq  = sy[cx * 2] & 0x3FF;
            int cb_10 = suv[cx * 2]     & 0x3FF;
            int cr_10 = suv[cx * 2 + 1] & 0x3FF;

            int r_sdr, g_sdr, b_sdr;
            tonemap_pixel_rgb(pq_to_linear, linear_to_sdr,
                              y_pq, cb_10, cr_10,
                              &r_sdr, &g_sdr, &b_sdr);

            int y_sdr = (int)(BT709_KR * r_sdr + BT709_KG * g_sdr
                            + BT709_KB * b_sdr + 0.5);
            int cb_sdr = (int)((b_sdr - y_sdr) / (2.0 * (1.0 - BT709_KB)) + 128.5);
            int cr_sdr = (int)((r_sdr - y_sdr) / (2.0 * (1.0 - BT709_KR)) + 128.5);

            du[cx] = (uint8_t)clamp_i(cb_sdr, 0, 255);
            dv[cx] = (uint8_t)clamp_i(cr_sdr, 0, 255);
        }
    }
}
