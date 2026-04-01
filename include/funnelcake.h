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
 * Positive: warning bits — composable, test with &.
 * Negative: hard errors — not composable, nothing was produced.
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

/* Supported planar YUV chroma layouts.
 * Zero-initialised contexts default to YUV420 for backward compatibility. */
#define FUSED_CHROMA_420 0   /* I420: chroma width = luma/2, height = luma/2 */
#define FUSED_CHROMA_422 1   /* I422: chroma width = luma/2, height = luma    */

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
    /* Source description — set by caller before init */
    int      src_width;
    int      src_height;
    int      src_y_stride;
    int      src_uv_stride;
    int      chroma_format;   /* FUSED_CHROMA_*; zero-init defaults to 420 */

    /* Configuration — set by caller before init */
    uint32_t requested_flags;   /* FUSED_SCALE_* bitmask (one family only) */
    uint32_t options;           /* FUSED_OPT_* bitmask                     */

    /* Logging — set by caller before init; zero struct = stderr defaults */
    fused_log_config_t log_errors;
    fused_log_config_t log_warnings;

    /* Results — written by fused_scaler_init */
    uint32_t achieved_flags;    /* steps that will be produced              */
    uint32_t rejected_flags;    /* steps that were rejected                 */
    int      effective_width;   /* actual source luma width used            */
    int      effective_height;  /* actual source luma height used           */
    fused_scale_output_t outputs[8];

    /* Internal — opaque, managed by init/free; do not read or write */
    void *_internal;
} fused_scaler_ctx_t;


/* --------------------------------------------------------------------------
 * Functions
 * -------------------------------------------------------------------------- */

/*
 * fused_scaler_init — validate configuration and allocate output buffers.
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
 * fused_scaler_run — process one input frame and fill all achieved outputs.
 *
 * src_y, src_u, src_v must point to the start of the planar source frame
 * for the current frame. Supported chroma layouts are FUSED_CHROMA_420
 * (I420) and FUSED_CHROMA_422 (I422). Strides come from
 * ctx->src_y_stride and ctx->src_uv_stride. Buffers must remain valid for
 * the duration of the call. Only the effective region
 * (ctx->effective_width x ctx->effective_height) is read; pixels outside
 * this region are ignored.
 *
 * Must only be called after a successful fused_scaler_init (return >= 0).
 */
void fused_scaler_run(fused_scaler_ctx_t *ctx,
                      const uint8_t *src_y,
                      const uint8_t *src_u,
                      const uint8_t *src_v);

/*
 * fused_scaler_free — release all resources allocated by fused_scaler_init.
 *
 * Safe to call on a zero-initialised context or after a hard-error init
 * (no-op in those cases). After this call, the context may be re-initialised
 * with new parameters.
 */
void fused_scaler_free(fused_scaler_ctx_t *ctx);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FUNNELCAKE_H */
