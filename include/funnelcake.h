/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#ifndef FUNNELCAKE_H
#define FUNNELCAKE_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif


/* --------------------------------------------------------------------------
 * Scale step flags (requested_flags / achieved_flags / rejected_flags)
 * Each bit selects one downscale output. Bits 0-3 are the "thirds" family
 * (steps divisible by 1.5x), bits 4-7 are the "power-of-two" family.
 * A single init call must request flags from only one family.
 * -------------------------------------------------------------------------- */

#define FUSED_SCALE_1_5X    (1u << 0)   /* 3:2 reduction   (thirds family) */
#define FUSED_SCALE_2X      (1u << 1)   /* 2:1 reduction   (pow2 family)   */
#define FUSED_SCALE_3X      (1u << 2)   /* 3:1 reduction   (thirds family) */
#define FUSED_SCALE_4X      (1u << 3)   /* 4:1 reduction   (pow2 family)   */
#define FUSED_SCALE_6X      (1u << 4)   /* 6:1 reduction   (thirds family) */
#define FUSED_SCALE_8X      (1u << 5)   /* 8:1 reduction   (pow2 family)   */
#define FUSED_SCALE_12X     (1u << 6)   /* 12:1 reduction  (thirds family) */
#define FUSED_SCALE_16X     (1u << 7)   /* 16:1 reduction  (pow2 family)   */

/* Convenience masks for each family */
#define FUSED_SCALE_THIRDS_MASK \
    (FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X | FUSED_SCALE_12X)

#define FUSED_SCALE_POW2_MASK \
    (FUSED_SCALE_2X | FUSED_SCALE_4X | FUSED_SCALE_8X | FUSED_SCALE_16X)

/* Output array indices - use these to index into outputs[], hdr_outputs[],
 * and sdr_outputs[] instead of hard-coding bit positions.
 *
 *   fused_scale_output_t *out = &scaler.outputs[FUSED_IDX_3X];
 */
#define FUSED_IDX_1_5X   0
#define FUSED_IDX_2X     1
#define FUSED_IDX_3X     2
#define FUSED_IDX_4X     3
#define FUSED_IDX_6X     4
#define FUSED_IDX_8X     5
#define FUSED_IDX_12X    6
#define FUSED_IDX_16X    7


/* --------------------------------------------------------------------------
 * Upscale flags (upscale_flags)
 *
 * Upscaling is a cascading 2x chain (0..5 levels) optionally followed by a
 * single 1.5x tail that reads the deepest 2x output. Unlike the downscale
 * flags, the upscale flag mask MUST be a contiguous prefix of the pow2
 * cascade - you can request {2x}, {2x,4x}, {2x,4x,8x}, etc. but not {4x}
 * alone or {2x,8x}. This mirrors the cascade dependency: 8x reads from 4x's
 * output buffer.
 *
 * The upscale chain is independent of the downscale family (thirds/pow2):
 * both may be requested in the same init call and are produced in a single
 * pass over the source.
 *
 * Combinations (cascade depth N, tail on/off) produce these outputs:
 *   N=0, tail=1 -> [1.5x]                           (reads source directly)
 *   N=1, tail=0 -> [2x]
 *   N=1, tail=1 -> [2x, 3x]                         (3x = 2x * 1.5)
 *   N=2, tail=0 -> [2x, 4x]
 *   N=2, tail=1 -> [2x, 4x, 6x]
 *   ...
 *   N=5, tail=0 -> [2x, 4x, 8x, 16x, 32x]
 *   N=5, tail=1 -> [2x, 4x, 8x, 16x, 32x, 48x]
 * -------------------------------------------------------------------------- */

#define FUSED_UPSCALE_2X        (1u << 0)
#define FUSED_UPSCALE_4X        (1u << 1)
#define FUSED_UPSCALE_8X        (1u << 2)
#define FUSED_UPSCALE_16X       (1u << 3)
#define FUSED_UPSCALE_32X       (1u << 4)

#define FUSED_UPSCALE_POW2_MASK 0x1Fu

/* Maximum number of upscale output slots (5 pow2 levels + 1.5x tail). */
#define FUSED_MAX_UPSCALE_STEPS 6

/* Upscale output array indices.
 *
 *   fused_scale_output_t *out = &scaler.upscale_outputs[FUSED_UP_IDX_4X];
 *
 * Slots 0..4 hold the 2x..32x outputs, slot 5 holds the 1.5x tail (if
 * upscale_tail_1_5x was set).
 */
#define FUSED_UP_IDX_2X     0
#define FUSED_UP_IDX_4X     1
#define FUSED_UP_IDX_8X     2
#define FUSED_UP_IDX_16X    3
#define FUSED_UP_IDX_32X    4
#define FUSED_UP_IDX_TAIL   5


/* --------------------------------------------------------------------------
 * Option flags (fused_scaler_ctx_t.options)
 * -------------------------------------------------------------------------- */

/* Reject steps that require cropping the source to satisfy dimension constraints.
 * By default, up to (ratio-1) rows/columns are silently cropped from the
 * bottom/right edge and FUSED_WARN_BIT_CROPPED is set in the return code. */
#define FUSED_OPT_NO_CROP       (1u << 0)

/* Reject steps that cannot use the SIMD kernel due to alignment constraints.
 * By default, such steps fall back to the scalar kernel and
 * FUSED_WARN_BIT_SCALAR is set in the return code. */
#define FUSED_OPT_NO_FALLBACK   (1u << 1)


/* --------------------------------------------------------------------------
 * Return codes
 *
 * Zero (FUSED_OK): all requested outputs produced with SIMD, no crop.
 * Positive: warning bits - composable, test with &.
 * Negative: hard errors - not composable, nothing was produced.
 * -------------------------------------------------------------------------- */

#define FUSED_OK                 0

/* Warning bits (composable) */
#define FUSED_WARN_BIT_SCALAR   (1 << 0)  /* scalar fallback used for >=1 step */
#define FUSED_WARN_BIT_PARTIAL  (1 << 1)  /* >=1 requested steps were rejected  */
#define FUSED_WARN_BIT_CROPPED  (1 << 2)  /* source was cropped to fit          */

/* Hard errors (negative) */
#define FUSED_ERR_INVALID_FLAGS  (-1)  /* flags from both families, or unknown bits */
#define FUSED_ERR_NO_STEPS       (-2)  /* no valid step flags set after filtering   */
#define FUSED_ERR_BAD_DIMENSIONS (-3)  /* src_width/height <= 0 or too small        */
#define FUSED_ERR_BAD_ALIGNMENT  (-4)  /* strides not 32-byte aligned               */


/* --------------------------------------------------------------------------
 * Log levels
 * -------------------------------------------------------------------------- */

#define FUSED_LOG_ERROR     0
#define FUSED_LOG_WARN      1


/* --------------------------------------------------------------------------
 * Log targets
 * -------------------------------------------------------------------------- */

#define FUSED_LOG_STDERR    0   /* default: write to stderr                  */
#define FUSED_LOG_STDOUT    1   /* write to stdout                           */
#define FUSED_LOG_FILE      2   /* write to config.file (must be non-NULL)   */
#define FUSED_LOG_SUPPRESS  3   /* discard all messages                      */
#define FUSED_LOG_CALLBACK  4   /* call config.callback with formatted text  */


/* --------------------------------------------------------------------------
 * Types
 * -------------------------------------------------------------------------- */

/*
 * Logging configuration. Zero-initialised struct means FUSED_LOG_STDERR.
 *
 * When target == FUSED_LOG_FILE, file must be a valid open FILE*.
 * When target == FUSED_LOG_CALLBACK, callback must be non-NULL.
 * callback_ctx is passed through opaquely to the callback.
 */
typedef struct {
    int     target;
    FILE   *file;
    void  (*callback)(int level, const char *msg, void *ctx);
    void   *callback_ctx;
} fused_log_config_t;

/*
 * One downscaled output plane set, indexed by bit position:
 *   FUSED_SCALE_1_5X (bit 0) -> outputs[0]
 *   FUSED_SCALE_2X   (bit 1) -> outputs[1]
 *   ...
 *   FUSED_SCALE_16X  (bit 7) -> outputs[7]
 *
 * plane_y/u/v are allocated by fused_scaler_init and freed by
 * fused_scaler_free. They are 32-byte aligned. Slots for steps that were
 * not requested or were rejected have NULL plane pointers.
 */
typedef struct {
    int      width;
    int      height;
    int      y_stride;
    int      uv_stride;
    uint8_t *plane_y;
    uint8_t *plane_u;
    uint8_t *plane_v;
    int      fallback;   /* 0 = SIMD kernel used, 1 = scalar kernel used */
} fused_scale_output_t;

/*
 * Scaler context. The caller fills source description and configuration
 * fields, then calls fused_scaler_init. Result fields are written by init.
 * The _internal pointer is managed by init/free and must not be touched.
 */
typedef struct {
    /* Source description - set by caller before init */
    int      src_width;
    int      src_height;
    int      src_y_stride;
    int      src_uv_stride;

    /* Configuration - set by caller before init */
    uint32_t requested_flags;   /* FUSED_SCALE_* bitmask (one family only) */
    uint32_t options;           /* FUSED_OPT_* bitmask                     */

    /* Logging - set by caller before init; zero struct = stderr defaults */
    fused_log_config_t log_errors;
    fused_log_config_t log_warnings;

    /* Upscale configuration - set by caller before init (all optional; zero
     * values mean "no upscaling").  May be used alone or together with the
     * downscale requested_flags above; in the combined case both direction's
     * outputs are produced in a single pass over the source. */
    uint32_t upscale_flags;     /* FUSED_UPSCALE_* contiguous prefix mask   */
    int      upscale_tail_1_5x; /* 0 or 1: append a 1.5x tail output        */

    /* Results - written by fused_scaler_init */
    uint32_t achieved_flags;    /* downscale steps that will be produced    */
    uint32_t rejected_flags;    /* downscale steps that were rejected       */
    int      effective_width;   /* actual source luma width used            */
    int      effective_height;  /* actual source luma height used           */
    fused_scale_output_t outputs[8];

    /* Upscale results - written by fused_scaler_init */
    uint32_t achieved_upscale_flags;   /* upscale levels that will be produced */
    int      achieved_upscale_tail;    /* 1 if the 1.5x tail will be produced  */
    fused_scale_output_t upscale_outputs[FUSED_MAX_UPSCALE_STEPS];

    /* Internal - opaque, managed by init/free; do not read or write */
    void *_internal;
} fused_scaler_ctx_t;


/* --------------------------------------------------------------------------
 * Functions
 * -------------------------------------------------------------------------- */

/*
 * fused_scaler_init - validate configuration and allocate output buffers.
 *
 * Returns FUSED_OK (0) on perfect success, a positive bitmask of
 * FUSED_WARN_BIT_* on partial success, or a negative FUSED_ERR_* on hard
 * error. On hard error, no resources are allocated and the context is
 * unchanged (except log messages written per log_errors config).
 *
 * Safe to call multiple times on the same context if the previous call
 * returned a hard error (no cleanup needed). If a previous call succeeded
 * or partially succeeded, call fused_scaler_free first.
 */
int fused_scaler_init(fused_scaler_ctx_t *ctx);

/*
 * fused_scaler_run - process one input frame and fill all achieved outputs.
 *
 * src_y, src_u, src_v must point to the start of the YUV420 I420 planes
 * for the current frame. Strides come from ctx->src_y_stride and
 * ctx->src_uv_stride. Buffers must remain valid for the duration of the
 * call. Only the effective region (ctx->effective_width x
 * ctx->effective_height) is read; pixels outside this region are ignored.
 *
 * Must only be called after a successful fused_scaler_init (return >= 0).
 */
void fused_scaler_run(fused_scaler_ctx_t *ctx,
                      const uint8_t *src_y,
                      const uint8_t *src_u,
                      const uint8_t *src_v);

/*
 * fused_scaler_free - release all resources allocated by fused_scaler_init.
 *
 * Safe to call on a zero-initialised context or after a hard-error init
 * (no-op in those cases). After this call, the context may be re-initialised
 * with new parameters.
 */
void fused_scaler_free(fused_scaler_ctx_t *ctx);


/* --------------------------------------------------------------------------
 * Capability query (applies to both the SDR and HDR scalers)
 * -------------------------------------------------------------------------- */

/*
 * fused_simd_available - report whether the scalers will use SIMD kernels.
 *
 * Returns 1 if fused_scaler_init / fused_hdr_init will run the vectorized
 * (AVX2 / NEON / RVV) kernels on this machine, or 0 if they will use the
 * portable scalar kernels instead. Scalar fallback happens when:
 *   - the build target has no SIMD kernels (a CPU family other than x86_64,
 *     aarch64, or riscv64),
 *   - the running CPU lacks the required extension (e.g. a pre-AVX2 x86_64
 *     part), or
 *   - FUNNELCAKE_FORCE_SCALAR is set to a non-empty value in the environment.
 *
 * When this returns 0, an otherwise-perfect init reports FUSED_WARN_BIT_SCALAR
 * (not FUSED_OK) and every produced output's `fallback` field is 1; callers
 * that treat that warning bit as success should consult this function first.
 *
 * Uses the same one-time CPU probe as the scalers; the result is cached, and
 * the call is thread-safe, including the first invocation.
 */
int fused_simd_available(void);

/* Static strings, valid for the lifetime of the loaded library. Backend is
 * the preferred available ISA; individual outputs may use scalar fallback. */
const char *fused_version(void);
const char *fused_backend(void);


/* ==========================================================================
 * HDR10 API - 10-bit scaling with optional tone mapping to SDR
 *
 * Completely separate from the 8-bit API above. The existing SDR types,
 * functions, and behavior are unchanged.
 * ========================================================================== */


/* --------------------------------------------------------------------------
 * Input pixel formats (fused_hdr_ctx_t.src_format)
 *
 * All formats use 10-bit samples stored in the low 10 bits of uint16_t.
 * 4:2:2 formats are accepted but internally decimated to 4:2:0 by
 * skipping every other chroma row (nearest-neighbor). P010/P210 formats
 * deinterleave chroma on-the-fly during the kernel load phase.
 * -------------------------------------------------------------------------- */

#define FUSED_PIX_I010     0   /* 4:2:0 planar: separate Y, U, V planes     */
#define FUSED_PIX_P010     1   /* 4:2:0 semi-planar: Y plane + interleaved UV */
#define FUSED_PIX_I210     2   /* 4:2:2 planar: decimated to 4:2:0 internally */
#define FUSED_PIX_P210     3   /* 4:2:2 semi-planar: decimated to 4:2:0       */


/* --------------------------------------------------------------------------
 * Transfer function (fused_hdr_ctx_t.src_transfer)
 *
 * Determines the EOTF used when generating tone mapping LUTs.
 * BT.2020 and BT.2100 share the same color primaries; only the transfer
 * function differs.
 * -------------------------------------------------------------------------- */

#define FUSED_TRC_PQ       0   /* SMPTE ST 2084 - Perceptual Quantizer (HDR10)  */
#define FUSED_TRC_HLG      1   /* Hybrid Log-Gamma (BBC/NHK, backward-compat)   */


/* --------------------------------------------------------------------------
 * Quantization range (fused_tonemap_config_t.src_range / dst_range)
 *
 * Limited (video/MPEG) range is the default because essentially all real
 * HDR10/HLG video streams are limited range: 10-bit Y 64..940, chroma
 * 64..960.  Full (PC/JPEG) range uses the entire code space (0..1023 in,
 * 0..255 out).  Only the tone mapping stage interprets pixel values, so
 * these flags have no effect on pure scaling outputs.
 * -------------------------------------------------------------------------- */

#define FUSED_RANGE_LIMITED 0   /* video range: 10-bit 64..940/960, 8-bit 16..235/240 */
#define FUSED_RANGE_FULL    1   /* full range: 10-bit 0..1023, 8-bit 0..255            */


/* --------------------------------------------------------------------------
 * Tone mapping curve presets (fused_tonemap_config_t.curve)
 *
 * Applied to SDR outputs and the 1:1 tone map output. The LUT is
 * precomputed at init time from the selected curve, peak_nits, and
 * target_nits.
 *
 * All built-in curves map [0, peak_nits] smoothly onto the SDR range -
 * highlights compress progressively rather than clipping, and peak_nits
 * controls how much headroom the curve reserves.  They differ in look:
 *
 *   HABLE    - filmic shoulder/toe.  Preserves the most highlight detail;
 *              renders midtones noticeably darker than the source
 *              (about -1 stop at default settings), as filmic curves do.
 *   REINHARD - extended Reinhard with white point at peak_nits.  Soft,
 *              lower contrast; midtones at about half the source level.
 *   BT2390   - ITU-R BT.2390 EETF in PQ space.  Reference broadcast
 *              behavior: content below the knee (a few tens of nits at
 *              default settings) passes through at the correct brightness,
 *              only the top of the range is compressed.  Use this when
 *              output should track the source as closely as possible.
 * -------------------------------------------------------------------------- */

#define FUSED_TONEMAP_HABLE    0   /* Hable/Uncharted 2 filmic - good default    */
#define FUSED_TONEMAP_REINHARD 1   /* Reinhard global - simple, lower contrast    */
#define FUSED_TONEMAP_BT2390   2   /* ITU-R BT.2390 EETF - broadcast reference   */
#define FUSED_TONEMAP_CUSTOM   3   /* Caller-supplied 1024-entry Y LUT            */


/* --------------------------------------------------------------------------
 * HDR types
 * -------------------------------------------------------------------------- */

/*
 * 10-bit output plane descriptor.
 * Same structure as fused_scale_output_t but with uint16_t planes.
 * Strides are in bytes. Planes are 32-byte aligned.
 */
typedef struct {
    int       width;
    int       height;
    int       y_stride;
    int       uv_stride;
    uint16_t *plane_y;
    uint16_t *plane_u;
    uint16_t *plane_v;
    int       fallback;     /* 0 = SIMD kernel used, 1 = scalar kernel used */
} fused_hdr_output_t;

/*
 * Tone mapping configuration for SDR outputs.
 * Zero-initialised struct means: Hable curve, 1000-nit peak, 100-nit target,
 * limited (video) range in and out.
 *
 * Range flags apply to the built-in curves only.  FUSED_TONEMAP_CUSTOM
 * passes 10-bit codes straight through the caller's LUT, so range handling
 * is the caller's responsibility there.
 */
typedef struct {
    int            curve;          /* FUSED_TONEMAP_* preset                    */
    int            peak_nits;      /* source peak brightness (0 = default 1000) */
    int            target_nits;    /* SDR target (0 = default 100)              */
    const uint8_t *custom_lut;     /* 1024-entry Y LUT for FUSED_TONEMAP_CUSTOM */
    int            src_range;      /* FUSED_RANGE_* of 10-bit input (default limited) */
    int            dst_range;      /* FUSED_RANGE_* of 8-bit SDR output (default limited) */
} fused_tonemap_config_t;

/*
 * HDR scaler context.
 * Caller-allocated, typically on the stack or as a struct member.
 * Zero-initialise before use, fill source/config fields, then call
 * fused_hdr_init.
 *
 * Each scale step can produce an HDR (10-bit) output, an SDR (8-bit,
 * tone-mapped) output, or both. Set the corresponding bit in hdr_flags
 * and/or sdr_flags. Both must be subsets of requested_flags.
 */
typedef struct {
    /* Source description - set by caller before init */
    int      src_width;
    int      src_height;
    int      src_y_stride;       /* bytes per row of Y plane                   */
    int      src_uv_stride;      /* bytes per row of U/V (or UV interleaved)   */
    int      src_format;         /* FUSED_PIX_* - pixel layout                 */
    int      src_transfer;       /* FUSED_TRC_* - transfer function            */

    /* Configuration - set by caller before init */
    uint32_t requested_flags;    /* FUSED_SCALE_* bitmask (one family only)    */
    uint32_t hdr_flags;          /* subset: produce 10-bit HDR outputs         */
    uint32_t sdr_flags;          /* subset: produce 8-bit tone-mapped outputs  */
    uint32_t options;            /* FUSED_OPT_* bitmask (NO_CROP, NO_FALLBACK) */

    /* 1:1 tone map - if set, produce an SDR copy at source resolution */
    int      tonemap_1x;

    /* Tone mapping configuration (applied to all SDR outputs + tonemap_1x) */
    fused_tonemap_config_t tonemap;

    /* Logging - zero struct = stderr defaults */
    fused_log_config_t log_errors;
    fused_log_config_t log_warnings;

    /* Results - written by fused_hdr_init */
    uint32_t achieved_hdr_flags;     /* HDR steps that will be produced          */
    uint32_t achieved_sdr_flags;     /* SDR steps that will be produced          */
    uint32_t rejected_flags;         /* steps that were rejected                 */
    int      effective_width;        /* actual source width used (after crop)    */
    int      effective_height;       /* actual source height used (after crop)   */

    /* Outputs - indexed by bit position of FUSED_SCALE_* flag */
    fused_hdr_output_t   hdr_outputs[8];  /* 10-bit; NULL planes if not in hdr_flags */
    fused_scale_output_t sdr_outputs[8];  /* 8-bit;  NULL planes if not in sdr_flags */
    fused_scale_output_t output_1x;       /* 8-bit tone-mapped at source res         */

    /* Upscale configuration and results.  Upscale outputs are 10-bit HDR
     * copies scaled from the 10-bit source.  Each achieved level (and the
     * 1.5x tail) can additionally produce an 8-bit tone-mapped SDR copy:
     * set the level's bit in upscale_sdr_flags (must be a subset of
     * upscale_flags - the 10-bit output exists either way) and/or
     * upscale_sdr_tail_1_5x (requires upscale_tail_1_5x).  SDR upscale
     * copies are tone-mapped from the level's 10-bit planes at the
     * upscaled resolution - the same scale-in-10-bit-then-tone-map order
     * as the downscale SDR outputs. */
    uint32_t upscale_flags;             /* FUSED_UPSCALE_* contiguous prefix mask  */
    int      upscale_tail_1_5x;         /* 0 or 1: append 1.5x tail output         */
    uint32_t upscale_sdr_flags;         /* subset of upscale_flags: also produce
                                         * tone-mapped SDR copies of these levels  */
    int      upscale_sdr_tail_1_5x;     /* 0 or 1: SDR copy of the 1.5x tail       */
    uint32_t achieved_upscale_flags;    /* written by init                         */
    int      achieved_upscale_tail;     /* written by init                         */
    uint32_t achieved_upscale_sdr_flags;/* written by init                         */
    int      achieved_upscale_sdr_tail; /* written by init                         */
    fused_hdr_output_t   upscale_hdr_outputs[FUSED_MAX_UPSCALE_STEPS];
    fused_scale_output_t upscale_sdr_outputs[FUSED_MAX_UPSCALE_STEPS];

    /* Internal - opaque, managed by init/free; do not read or write */
    void *_internal;
} fused_hdr_ctx_t;


/* --------------------------------------------------------------------------
 * HDR functions
 * -------------------------------------------------------------------------- */

/*
 * fused_hdr_init - validate configuration, generate tone mapping LUTs,
 * and allocate output buffers.
 *
 * Returns FUSED_OK (0) on perfect success, a positive bitmask of
 * FUSED_WARN_BIT_* on partial success, or a negative FUSED_ERR_* on hard
 * error. Same return code semantics as fused_scaler_init.
 *
 * Additional error: returns FUSED_ERR_INVALID_FLAGS if hdr_flags or
 * sdr_flags contain bits not in requested_flags, or if src_format or
 * src_transfer is invalid.
 */
int fused_hdr_init(fused_hdr_ctx_t *ctx);

/*
 * fused_hdr_run - process one 10-bit input frame, produce all HDR and
 * SDR outputs, and apply tone mapping to SDR outputs.
 *
 * src_y: pointer to luma plane (uint16_t, 10-bit values in low bits).
 * src_u: pointer to U plane (I010/I210) or interleaved UV plane (P010/P210).
 * src_v: pointer to V plane (I010/I210) or NULL (P010/P210).
 *
 * All pointers must be 32-byte aligned for SIMD. Misaligned pointers
 * cause fallback to the scalar kernel with a one-time warning.
 */
void fused_hdr_run(fused_hdr_ctx_t *ctx,
                   const uint16_t *src_y,
                   const uint16_t *src_u,
                   const uint16_t *src_v);

/*
 * fused_hdr_free - release all resources allocated by fused_hdr_init.
 * Same safety semantics as fused_scaler_free.
 */
void fused_hdr_free(fused_hdr_ctx_t *ctx);


/* Allocation-free planning. Uses init's validation, geometry and dispatch.
 * On success, layout contains output sizes, strides, fallback and achieved /
 * rejected flags, with NULL plane and internal pointers. memory_bytes is the
 * total requested heap storage (outputs, intermediates, scratch and state),
 * excluding allocator overhead. No output arguments change on hard error.
 * Do not run a query layout. The input configuration is never modified. */
int fused_scaler_query(const fused_scaler_ctx_t *config,
                       fused_scaler_ctx_t *layout, size_t *memory_bytes);
int fused_hdr_query(const fused_hdr_ctx_t *config,
                    fused_hdr_ctx_t *layout, size_t *memory_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FUNNELCAKE_H */
