# Funnelcake API Reference

Funnelcake is a fused multi-resolution YUV420 downscaler. A single call
to `fused_scaler_run` produces up to four downscaled outputs in one pass,
using AVX2 or NEON SIMD kernels with a portable scalar fallback.

Input and output are I420 planar (Y, U, V separate planes), 8-bit.


## Quick Start

```c
#include "funnelcake.h"
#include <stdint.h>
#include <stdio.h>

/* Allocate aligned source buffers (example: 1920x1080 I420) */
int width  = 1920;
int height = 1080;
int y_stride  = (width + 31) & ~31;          /* 1920 */
int uv_stride = (width / 2 + 31) & ~31;      /* 960  */

uint8_t *src_y = aligned_alloc(32, y_stride  * height);
uint8_t *src_u = aligned_alloc(32, uv_stride * (height / 2));
uint8_t *src_v = aligned_alloc(32, uv_stride * (height / 2));

/* Fill src_y/u/v with your frame data here */

/* Configure and initialise the scaler */
fused_scaler_ctx_t scaler = {0};
scaler.src_width    = width;
scaler.src_height   = height;
scaler.src_y_stride  = y_stride;
scaler.src_uv_stride = uv_stride;
scaler.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

int rc = fused_scaler_init(&scaler);
if (rc < 0) {
    fprintf(stderr, "fused_scaler_init failed: %d\n", rc);
    return rc;
}

/* Process a frame */
fused_scaler_run(&scaler, src_y, src_u, src_v);

/* Access outputs — indexed by FUSED_IDX_* constants */
fused_scale_output_t *out_1_5x = &scaler.outputs[FUSED_IDX_1_5X];
fused_scale_output_t *out_3x   = &scaler.outputs[FUSED_IDX_3X];
fused_scale_output_t *out_6x   = &scaler.outputs[FUSED_IDX_6X];

/* out_1_5x->plane_y, plane_u, plane_v are ready */

/* Clean up */
fused_scaler_free(&scaler);
free(src_y); free(src_u); free(src_v);
```


## API Reference

### `fused_scaler_init`

```c
int fused_scaler_init(fused_scaler_ctx_t *ctx);
```

Validates configuration, selects kernel paths, and allocates output
buffers. Must be called before `fused_scaler_run`.

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `ctx` | Pointer to a caller-allocated context. Caller fills source description and configuration fields before calling. |

**Return value**

- `FUSED_OK` (0): all requested outputs will be produced using SIMD, no
  source cropping applied.
- Positive: one or more `FUSED_WARN_BIT_*` bits OR'd together. Processing
  will proceed with the caveats indicated. Test individual bits with `&`.
- Negative: a `FUSED_ERR_*` hard error. No resources are allocated and
  the context is unchanged. The error is logged per `ctx->log_errors`.

On hard error, the call is safe to retry after adjusting parameters (no
cleanup needed). If a previous init succeeded or partially succeeded,
call `fused_scaler_free` before re-initialising.

---

### `fused_scaler_run`

```c
void fused_scaler_run(fused_scaler_ctx_t *ctx,
                      const uint8_t *src_y,
                      const uint8_t *src_u,
                      const uint8_t *src_v);
```

Processes one input frame and writes all achieved outputs. Call once
per frame after a successful `fused_scaler_init`.

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `ctx` | Initialised scaler context. |
| `src_y` | Pointer to the start of the luma plane. |
| `src_u` | Pointer to the start of the Cb (U) chroma plane. |
| `src_v` | Pointer to the start of the Cr (V) chroma plane. |

Strides are taken from `ctx->src_y_stride` and `ctx->src_uv_stride`.
Only the effective region (`ctx->effective_width` x
`ctx->effective_height`) is read; pixels outside are ignored. Buffers
must remain valid for the duration of the call.

Must only be called after `fused_scaler_init` returns `>= 0`.

---

### `fused_scaler_free`

```c
void fused_scaler_free(fused_scaler_ctx_t *ctx);
```

Releases all resources allocated by `fused_scaler_init`. Safe to call
on a zero-initialised context or on a context where init returned a hard
error (no-op in both cases). After this call the context may be
re-initialised with new parameters.


## Data Types

### `fused_scaler_ctx_t`

The main scaler context. Caller-allocated, typically on the stack or as
a struct member. Zero-initialise before use.

**Fields set by caller before `fused_scaler_init`**

| Field | Type | Description |
|-------|------|-------------|
| `src_width` | `int` | Source luma width in pixels. Must be > 0 and large enough for all requested steps. |
| `src_height` | `int` | Source luma height in pixels. Must be > 0 and large enough for all requested steps. |
| `src_y_stride` | `int` | Bytes per row of the luma plane. Must be >= `src_width` and 32-byte aligned. |
| `src_uv_stride` | `int` | Bytes per row of each chroma plane. Must be >= `src_width/2` and 32-byte aligned. |
| `requested_flags` | `uint32_t` | Bitmask of `FUSED_SCALE_*` flags. All set bits must belong to the same family (thirds or pow2). |
| `options` | `uint32_t` | Bitmask of `FUSED_OPT_*` flags. Zero means default (lenient) behavior. |
| `log_errors` | `fused_log_config_t` | Logging target for hard errors. Zero-value = stderr. |
| `log_warnings` | `fused_log_config_t` | Logging target for warnings. Zero-value = stderr. |

**Fields written by `fused_scaler_init`**

| Field | Type | Description |
|-------|------|-------------|
| `achieved_flags` | `uint32_t` | Steps that will be produced on each `fused_scaler_run` call. |
| `rejected_flags` | `uint32_t` | Steps from `requested_flags` that were rejected. |
| `effective_width` | `int` | Actual luma width read from the source (may be <= `src_width` if cropped). |
| `effective_height` | `int` | Actual luma height read from the source (may be <= `src_height` if cropped). |
| `outputs[8]` | `fused_scale_output_t` | Indexed by `FUSED_IDX_*` constants. Slots for steps not in `achieved_flags` have NULL plane pointers. |

The `_internal` field is opaque; do not read or write it.

---

### `fused_scale_output_t`

Describes one downscaled output. Indexed by `FUSED_IDX_*` constants
(which correspond to the bit positions of `FUSED_SCALE_*` flags):

```
outputs[FUSED_IDX_1_5X]  FUSED_SCALE_1_5X
outputs[FUSED_IDX_2X]    FUSED_SCALE_2X
outputs[FUSED_IDX_3X]    FUSED_SCALE_3X
outputs[FUSED_IDX_4X]    FUSED_SCALE_4X
outputs[FUSED_IDX_6X]    FUSED_SCALE_6X
outputs[FUSED_IDX_8X]    FUSED_SCALE_8X
outputs[FUSED_IDX_12X]   FUSED_SCALE_12X
outputs[FUSED_IDX_16X]   FUSED_SCALE_16X
```

| Field | Type | Description |
|-------|------|-------------|
| `width` | `int` | Output luma width in pixels. |
| `height` | `int` | Output luma height in pixels. |
| `y_stride` | `int` | Bytes per row of `plane_y`. 32-byte aligned. |
| `uv_stride` | `int` | Bytes per row of `plane_u` and `plane_v`. 32-byte aligned. |
| `plane_y` | `uint8_t *` | Luma plane. Allocated by init, freed by free. NULL if step not achieved. |
| `plane_u` | `uint8_t *` | Cb chroma plane. NULL if step not achieved. |
| `plane_v` | `uint8_t *` | Cr chroma plane. NULL if step not achieved. |
| `fallback` | `int` | 0 = SIMD kernel used, 1 = scalar kernel used. |

---

### `fused_log_config_t`

Controls where diagnostic messages are written. A zero-initialised
struct means write to stderr.

| Field | Type | Description |
|-------|------|-------------|
| `target` | `int` | One of the `FUSED_LOG_*` constants. |
| `file` | `FILE *` | Used when `target == FUSED_LOG_FILE`. Must be a valid open file. |
| `callback` | `void (*)(int level, const char *msg, void *ctx)` | Used when `target == FUSED_LOG_CALLBACK`. `level` is `FUSED_LOG_ERROR` or `FUSED_LOG_WARN`. |
| `callback_ctx` | `void *` | Passed through opaquely as the `ctx` argument to `callback`. |

Log target constants:

| Constant | Value | Behavior |
|----------|-------|----------|
| `FUSED_LOG_STDERR` | 0 | Write to stderr (default) |
| `FUSED_LOG_STDOUT` | 1 | Write to stdout |
| `FUSED_LOG_FILE` | 2 | Write to `config.file` |
| `FUSED_LOG_SUPPRESS` | 3 | Discard all messages |
| `FUSED_LOG_CALLBACK` | 4 | Call `config.callback` |


## Scale Step Flags

All flags are ORed into `requested_flags`. All set bits must come from
the same family; mixing families returns `FUSED_ERR_INVALID_FLAGS`.

| Flag | Bit | Ratio | Family |
|------|-----|-------|--------|
| `FUSED_SCALE_1_5X` | 0 | 3:2 (1.5x) | Thirds |
| `FUSED_SCALE_2X` | 1 | 2:1 | Pow2 |
| `FUSED_SCALE_3X` | 2 | 3:1 | Thirds |
| `FUSED_SCALE_4X` | 3 | 4:1 | Pow2 |
| `FUSED_SCALE_6X` | 4 | 6:1 | Thirds |
| `FUSED_SCALE_8X` | 5 | 8:1 | Pow2 |
| `FUSED_SCALE_12X` | 6 | 12:1 | Thirds |
| `FUSED_SCALE_16X` | 7 | 16:1 | Pow2 |

Convenience masks:

```c
FUSED_SCALE_THIRDS_MASK  /* 1.5x | 3x | 6x | 12x */
FUSED_SCALE_POW2_MASK    /* 2x | 4x | 8x | 16x   */
```

You do not need to request every step in the cascade. Requesting only
`FUSED_SCALE_6X` is valid; the library performs the 1.5x and 3x
intermediate passes internally without allocating output buffers for
them.


## Option Flags

Set in `ctx->options` before calling `fused_scaler_init`. Default
behavior (options = 0) is to produce every output possible.

### Default behavior

- **Scalar fallback on by default.** Steps that pass dimension
  constraints but fail SIMD alignment constraints (chroma width not a
  multiple of 32) are produced using the scalar kernel. The return code
  includes `FUSED_WARN_BIT_SCALAR`.
- **Crop-to-fit on by default.** Steps that would produce non-integer
  or odd output dimensions are rescued by silently cropping up to
  `ratio - 1` rows/columns from the bottom/right edge of the source.
  The crop is computed once at init time and applies only to the
  kernel's loop bounds — no data is copied. The return code includes
  `FUSED_WARN_BIT_CROPPED`, and the effective source region is reported
  in `ctx->effective_width` / `ctx->effective_height`.

### `FUSED_OPT_NO_CROP` (bit 0)

Disables crop-to-fit. Steps that require dimension adjustment to satisfy
output constraints are rejected instead. Those steps appear in
`ctx->rejected_flags` and the return code includes
`FUSED_WARN_BIT_PARTIAL`.

### `FUSED_OPT_NO_FALLBACK` (bit 1)

Disables the scalar fallback. Steps that cannot use the SIMD kernel due
to alignment constraints are rejected instead of falling back to scalar.
Those steps appear in `ctx->rejected_flags` and the return code includes
`FUSED_WARN_BIT_PARTIAL`.

Combining both flags (`FUSED_OPT_NO_CROP | FUSED_OPT_NO_FALLBACK`) puts
the scaler in strict mode: only steps that can be processed perfectly
with SIMD on exact dimensions are produced.


## Return Codes

### Success

| Code | Value | Meaning |
|------|-------|---------|
| `FUSED_OK` | 0 | All requested outputs produced with SIMD, no crop applied. |

### Warning bits (positive, composable)

Test individual bits with bitwise AND:

```c
int rc = fused_scaler_init(&scaler);
if (rc > 0) {
    if (rc & FUSED_WARN_BIT_SCALAR)  { /* >=1 step used scalar kernel */ }
    if (rc & FUSED_WARN_BIT_PARTIAL) { /* >=1 step was rejected       */ }
    if (rc & FUSED_WARN_BIT_CROPPED) { /* source was cropped          */ }
}
```

| Constant | Bit | Meaning |
|----------|-----|---------|
| `FUSED_WARN_BIT_SCALAR` | 0 | At least one step used the scalar kernel instead of SIMD. |
| `FUSED_WARN_BIT_PARTIAL` | 1 | At least one requested step was rejected and not produced. |
| `FUSED_WARN_BIT_CROPPED` | 2 | Source was cropped to fit dimension constraints. |

### Hard errors (negative, not composable)

On any negative return, no resources are allocated, no output buffers
are valid, and `fused_scaler_run` must not be called.

| Constant | Value | Meaning |
|----------|-------|---------|
| `FUSED_ERR_INVALID_FLAGS` | -1 | `requested_flags` contains bits from both families, or unknown bits. |
| `FUSED_ERR_NO_STEPS` | -2 | No valid step flags remain after filtering (all were rejected or none were set). |
| `FUSED_ERR_BAD_DIMENSIONS` | -3 | `src_width` or `src_height` is <= 0, or too small for the requested steps. |
| `FUSED_ERR_BAD_ALIGNMENT` | -4 | `src_y_stride` or `src_uv_stride` is not 32-byte aligned. |


## Alignment Requirements

All source strides must be 32-byte aligned. Output buffer strides and
pointers are always 32-byte aligned (guaranteed by the library).

To compute a compliant stride from a pixel width:

```c
int y_stride  = (width + 31) & ~31;
int uv_stride = (width / 2 + 31) & ~31;
```

Allocate buffers with 32-byte alignment:

```c
uint8_t *plane_y = aligned_alloc(32, y_stride  * height);
uint8_t *plane_u = aligned_alloc(32, uv_stride * (height / 2));
uint8_t *plane_v = aligned_alloc(32, uv_stride * (height / 2));
```

Misaligned strides cause `fused_scaler_init` to return
`FUSED_ERR_BAD_ALIGNMENT`. Misaligned buffer pointers do not cause a
hard error at init time but will produce incorrect results or faults
at runtime on steps that use the SIMD kernel.

SIMD steps additionally require the chroma width to be a multiple of 32
(i.e., `src_width / 2` must be a multiple of 32, meaning `src_width`
must be a multiple of 64). Steps that fail this constraint are handled
by the scalar fallback unless `FUSED_OPT_NO_FALLBACK` is set.


## libavcodec Integration

AVFrame planes map directly to the scaler's source parameters:

```c
#include "funnelcake.h"
#include <libavcodec/avcodec.h>

/* frame is an AVFrame* with format AV_PIX_FMT_YUV420P */

fused_scaler_ctx_t scaler = {0};
scaler.src_width     = frame->width;
scaler.src_height    = frame->height;
scaler.src_y_stride  = frame->linesize[0];
scaler.src_uv_stride = frame->linesize[1];
scaler.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

int rc = fused_scaler_init(&scaler);
if (rc < 0) {
    /* Hard error — log and abort */
    fprintf(stderr, "fused_scaler_init: error %d\n", rc);
    return rc;
}
if (rc & FUSED_WARN_BIT_PARTIAL) {
    fprintf(stderr, "warning: some steps rejected, check scaler.rejected_flags\n");
}

/* Process each decoded frame */
fused_scaler_run(&scaler, frame->data[0], frame->data[1], frame->data[2]);

/* Access outputs */
fused_scale_output_t *half = &scaler.outputs[FUSED_IDX_1_5X];
/* half->plane_y, half->plane_u, half->plane_v are ready to encode */

fused_scaler_free(&scaler);
```

Note: `frame->linesize[0]` must be 32-byte aligned. Frames decoded from
most production H.264 streams at standard resolutions (1080p, 720p, etc.)
already satisfy this. If they do not, copy the planes into aligned
buffers before calling `fused_scaler_run`.


## Handling Rejected Steps

When `FUSED_WARN_BIT_PARTIAL` is set, one or more requested steps were
not produced. The rejected steps are recorded in `ctx->rejected_flags`.
A common approach is to use the deepest achieved output as the source
for a libswscale fallback to reach the remaining targets:

```c
#include "funnelcake.h"
#include <libswscale/swscale.h>

int rc = fused_scaler_init(&scaler);
/* ... handle hard errors ... */

fused_scaler_run(&scaler, src_y, src_u, src_v);

if (scaler.rejected_flags) {
    /* Find the deepest achieved output to use as the swscale source */
    fused_scale_output_t *deepest = NULL;
    for (int i = 7; i >= 0; i--) {
        if ((scaler.achieved_flags >> i) & 1) {
            deepest = &scaler.outputs[i];
            break;
        }
    }

    if (deepest && deepest->plane_y) {
        /* Use swscale to reach each rejected target from deepest */
        uint32_t remaining = scaler.rejected_flags;
        while (remaining) {
            int bit = __builtin_ctz(remaining);
            remaining &= remaining - 1;

            /* Compute target dimensions and call sws_scale here */
            /* ... */
        }
    }
}
```

This keeps the fast fused path for the common case while providing a
reliable fallback for edge-case resolutions or strict-mode rejections.


## Logging Configuration

By default, both `log_errors` and `log_warnings` write to stderr. Set
either field before calling `fused_scaler_init` to override.

**Suppress all output** (handle everything via return codes):

```c
fused_scaler_ctx_t scaler = {0};
scaler.log_errors.target   = FUSED_LOG_SUPPRESS;
scaler.log_warnings.target = FUSED_LOG_SUPPRESS;
```

**Redirect to an open file**:

```c
FILE *logfile = fopen("scaler.log", "a");
scaler.log_errors.target   = FUSED_LOG_FILE;
scaler.log_errors.file     = logfile;
scaler.log_warnings.target = FUSED_LOG_FILE;
scaler.log_warnings.file   = logfile;
```

**Use a callback to integrate with your logging framework**:

```c
static void my_log(int level, const char *msg, void *ctx) {
    my_logger_t *log = ctx;
    if (level == FUSED_LOG_ERROR)
        my_logger_error(log, "funnelcake: %s", msg);
    else
        my_logger_warn(log, "funnelcake: %s", msg);
}

scaler.log_errors.target       = FUSED_LOG_CALLBACK;
scaler.log_errors.callback     = my_log;
scaler.log_errors.callback_ctx = my_logger_instance;

scaler.log_warnings.target       = FUSED_LOG_CALLBACK;
scaler.log_warnings.callback     = my_log;
scaler.log_warnings.callback_ctx = my_logger_instance;
```

The `level` argument to the callback is `FUSED_LOG_ERROR` (0) or
`FUSED_LOG_WARN` (1). The `msg` string is a complete formatted message;
do not call `fused_scaler_*` functions from within the callback.


## HDR10 API Reference

The HDR API is a separate set of types and functions for 10-bit PQ/HLG
content. It shares the same scale step flags, option flags, return codes,
and logging infrastructure as the 8-bit API. Each scale step can produce
a 10-bit HDR output, an 8-bit tone-mapped SDR output, or both.


### Quick Start (HDR)

```c
#include "funnelcake.h"
#include <stdint.h>
#include <stdio.h>

/* Allocate aligned 10-bit source buffers (example: 3840x2160 I010) */
int width  = 3840;
int height = 2160;
int y_stride  = ((width * 2) + 31) & ~31;          /* bytes per row, 32-byte aligned */
int uv_stride = ((width / 2 * 2) + 31) & ~31;

uint16_t *src_y = aligned_alloc(32, y_stride  * height);
uint16_t *src_u = aligned_alloc(32, uv_stride * (height / 2));
uint16_t *src_v = aligned_alloc(32, uv_stride * (height / 2));

/* Fill src_y/u/v with your 10-bit frame data here */

/* Configure the HDR scaler */
fused_hdr_ctx_t hdr = {0};
hdr.src_width      = width;
hdr.src_height     = height;
hdr.src_y_stride   = y_stride;
hdr.src_uv_stride  = uv_stride;
hdr.src_format     = FUSED_PIX_I010;
hdr.src_transfer   = FUSED_TRC_PQ;

/* Request thirds cascade with mixed HDR + SDR outputs */
hdr.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
hdr.hdr_flags       = FUSED_SCALE_1_5X;                   /* 2560x1440 HDR */
hdr.sdr_flags       = FUSED_SCALE_1_5X | FUSED_SCALE_3X;  /* 2560x1440 + 1280x720 SDR */

/* Also produce a 1:1 SDR tone-mapped copy at source resolution */
hdr.tonemap_1x = 1;

/* Tone mapping: BT.2390 curve, 1000-nit source, 100-nit SDR target */
hdr.tonemap.curve       = FUSED_TONEMAP_BT2390;
hdr.tonemap.peak_nits   = 1000;
hdr.tonemap.target_nits = 100;

int rc = fused_hdr_init(&hdr);
if (rc < 0) {
    fprintf(stderr, "fused_hdr_init failed: %d\n", rc);
    return rc;
}

/* Process a frame */
fused_hdr_run(&hdr, src_y, src_u, src_v);

/* Access HDR outputs (10-bit, uint16_t planes) */
fused_hdr_output_t *hdr_1440p = &hdr.hdr_outputs[FUSED_IDX_1_5X];
/* hdr_1440p->plane_y, plane_u, plane_v are uint16_t* */

/* Access SDR outputs (8-bit, uint8_t planes) */
fused_scale_output_t *sdr_1440p = &hdr.sdr_outputs[FUSED_IDX_1_5X];
fused_scale_output_t *sdr_720p  = &hdr.sdr_outputs[FUSED_IDX_3X];
fused_scale_output_t *sdr_4k    = &hdr.output_1x;      /* 1:1 tone-mapped  */

/* Clean up */
fused_hdr_free(&hdr);
free(src_y); free(src_u); free(src_v);
```

---

### `fused_hdr_init`

```c
int fused_hdr_init(fused_hdr_ctx_t *ctx);
```

Validates configuration, generates tone mapping LUTs from the selected
curve and transfer function, selects kernel paths, and allocates output
buffers. Must be called before `fused_hdr_run`.

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `ctx` | Pointer to a caller-allocated HDR context. Caller fills source description and configuration fields before calling. |

**Return value**

- `FUSED_OK` (0): all requested outputs will be produced using SIMD, no
  source cropping applied.
- Positive: one or more `FUSED_WARN_BIT_*` bits OR'd together. Processing
  will proceed with the caveats indicated. Test individual bits with `&`.
- Negative: a `FUSED_ERR_*` hard error. No resources are allocated and
  the context is unchanged.

Additional validation beyond the 8-bit API: returns
`FUSED_ERR_INVALID_FLAGS` if `hdr_flags` or `sdr_flags` contain bits not
present in `requested_flags`, or if `src_format` or `src_transfer` is
not a recognized constant.

On hard error, the call is safe to retry after adjusting parameters (no
cleanup needed). If a previous init succeeded or partially succeeded,
call `fused_hdr_free` before re-initialising.

---

### `fused_hdr_run`

```c
void fused_hdr_run(fused_hdr_ctx_t *ctx,
                   const uint16_t *src_y,
                   const uint16_t *src_u,
                   const uint16_t *src_v);
```

Processes one 10-bit input frame, produces all achieved HDR and SDR
outputs, and applies tone mapping to SDR outputs. Call once per frame
after a successful `fused_hdr_init`.

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `ctx` | Initialised HDR scaler context. |
| `src_y` | Pointer to the start of the 10-bit luma plane (`uint16_t`). |
| `src_u` | Pointer to the U plane (I010/I210) or interleaved UV plane (P010/P210). |
| `src_v` | Pointer to the V plane (I010/I210) or `NULL` (P010/P210). |

**P010/P210 convention:** pass the interleaved UV plane as `src_u` and
set `src_v` to `NULL`. The kernel deinterleaves U and V on-the-fly.

All pointers must be **32-byte aligned** for the SIMD kernel. Misaligned
pointers cause fallback to the scalar kernel with a one-time warning.

Strides are taken from `ctx->src_y_stride` and `ctx->src_uv_stride`
(both in bytes). Only the effective region is read.

Must only be called after `fused_hdr_init` returns `>= 0`.

---

### `fused_hdr_free`

```c
void fused_hdr_free(fused_hdr_ctx_t *ctx);
```

Releases all resources allocated by `fused_hdr_init`, including tone
mapping LUTs and all output buffers. Safe to call on a zero-initialised
context or on a context where init returned a hard error (no-op in both
cases). After this call the context may be re-initialised with new
parameters.


## HDR Data Types

### `fused_hdr_output_t`

10-bit output plane descriptor. Same structure as `fused_scale_output_t`
but with `uint16_t` planes. Indexed by `FUSED_IDX_*` constants
(same scheme as the 8-bit outputs):

```
hdr_outputs[FUSED_IDX_1_5X]  FUSED_SCALE_1_5X
hdr_outputs[FUSED_IDX_2X]    FUSED_SCALE_2X
hdr_outputs[FUSED_IDX_3X]    FUSED_SCALE_3X
hdr_outputs[FUSED_IDX_4X]    FUSED_SCALE_4X
hdr_outputs[FUSED_IDX_6X]    FUSED_SCALE_6X
hdr_outputs[FUSED_IDX_8X]    FUSED_SCALE_8X
hdr_outputs[FUSED_IDX_12X]   FUSED_SCALE_12X
hdr_outputs[FUSED_IDX_16X]   FUSED_SCALE_16X
```

| Field | Type | Description |
|-------|------|-------------|
| `width` | `int` | Output luma width in pixels. |
| `height` | `int` | Output luma height in pixels. |
| `y_stride` | `int` | Bytes per row of `plane_y`. 32-byte aligned. |
| `uv_stride` | `int` | Bytes per row of `plane_u` and `plane_v`. 32-byte aligned. |
| `plane_y` | `uint16_t *` | Luma plane (10-bit). Allocated by init, freed by free. NULL if step not achieved. |
| `plane_u` | `uint16_t *` | Cb chroma plane (10-bit). NULL if step not achieved. |
| `plane_v` | `uint16_t *` | Cr chroma plane (10-bit). NULL if step not achieved. |
| `fallback` | `int` | 0 = SIMD kernel used, 1 = scalar kernel used. |

---

### `fused_tonemap_config_t`

Tone mapping configuration. Applied to all SDR outputs and the 1:1 tone
map output. A zero-initialised struct uses the default: Hable curve,
1000-nit peak, 100-nit target.

| Field | Type | Description |
|-------|------|-------------|
| `curve` | `int` | `FUSED_TONEMAP_*` preset constant. Default (0) = Hable. |
| `peak_nits` | `int` | Source peak brightness in nits. 0 = default (1000). |
| `target_nits` | `int` | SDR target brightness in nits. 0 = default (100). |
| `custom_lut` | `const uint8_t *` | 1024-entry Y lookup table for `FUSED_TONEMAP_CUSTOM`. Ignored for other curves. Maps 10-bit input luma [0..1023] to 8-bit output luma [0..255]. |

---

### `fused_hdr_ctx_t`

The HDR scaler context. Caller-allocated, typically on the stack or as a
struct member. Zero-initialise before use.

**Fields set by caller before `fused_hdr_init`**

| Field | Type | Description |
|-------|------|-------------|
| `src_width` | `int` | Source luma width in pixels. |
| `src_height` | `int` | Source luma height in pixels. |
| `src_y_stride` | `int` | Bytes per row of the luma plane. Must be 32-byte aligned. |
| `src_uv_stride` | `int` | Bytes per row of the U/V or interleaved UV plane. Must be 32-byte aligned. |
| `src_format` | `int` | `FUSED_PIX_*` constant — input pixel layout. |
| `src_transfer` | `int` | `FUSED_TRC_*` constant — transfer function (PQ or HLG). |
| `requested_flags` | `uint32_t` | Bitmask of `FUSED_SCALE_*` flags. One family only. |
| `hdr_flags` | `uint32_t` | Subset of `requested_flags` — produce 10-bit HDR outputs for these steps. |
| `sdr_flags` | `uint32_t` | Subset of `requested_flags` — produce 8-bit tone-mapped SDR outputs for these steps. |
| `options` | `uint32_t` | `FUSED_OPT_*` bitmask (same as 8-bit API). |
| `tonemap_1x` | `int` | If non-zero, produce an 8-bit tone-mapped copy at source resolution. |
| `tonemap` | `fused_tonemap_config_t` | Tone mapping curve and parameters. |
| `log_errors` | `fused_log_config_t` | Logging target for hard errors. Zero = stderr. |
| `log_warnings` | `fused_log_config_t` | Logging target for warnings. Zero = stderr. |

**Fields written by `fused_hdr_init`**

| Field | Type | Description |
|-------|------|-------------|
| `achieved_hdr_flags` | `uint32_t` | HDR steps that will be produced. |
| `achieved_sdr_flags` | `uint32_t` | SDR steps that will be produced. |
| `rejected_flags` | `uint32_t` | Steps from `requested_flags` that were rejected. |
| `effective_width` | `int` | Actual luma width read (may be <= `src_width` if cropped). |
| `effective_height` | `int` | Actual luma height read (may be <= `src_height` if cropped). |
| `hdr_outputs[8]` | `fused_hdr_output_t` | 10-bit outputs. Slots not in `achieved_hdr_flags` have NULL planes. |
| `sdr_outputs[8]` | `fused_scale_output_t` | 8-bit tone-mapped outputs. Slots not in `achieved_sdr_flags` have NULL planes. |
| `output_1x` | `fused_scale_output_t` | 8-bit tone-mapped output at source resolution. Only valid if `tonemap_1x` was set. |

The `_internal` field is opaque; do not read or write it.


## Input Pixel Formats

All formats use 10-bit samples stored in the low 10 bits of `uint16_t`.
Set `ctx->src_format` before calling `fused_hdr_init`.

| Constant | Value | Subsampling | Plane layout | Notes |
|----------|-------|-------------|--------------|-------|
| `FUSED_PIX_I010` | 0 | 4:2:0 | Separate Y, U, V planes | Preferred format. No deinterleave overhead. Pass Y, U, V to `fused_hdr_run`. |
| `FUSED_PIX_P010` | 1 | 4:2:0 | Y plane + interleaved UV | Pass UV plane as `src_u`, set `src_v = NULL`. UV is deinterleaved on-the-fly during the kernel load phase (slight performance penalty). |
| `FUSED_PIX_I210` | 2 | 4:2:2 | Separate Y, U, V planes | Chroma is internally decimated to 4:2:0 by skipping every other chroma row (nearest-neighbor). |
| `FUSED_PIX_P210` | 3 | 4:2:2 | Y plane + interleaved UV | Combines P010 deinterleave and I210 row-skipping. |


## Transfer Functions

Set `ctx->src_transfer` before calling `fused_hdr_init`. The transfer
function determines the EOTF used when generating tone mapping LUTs.

| Constant | Value | Description |
|----------|-------|-------------|
| `FUSED_TRC_PQ` | 0 | SMPTE ST 2084 Perceptual Quantizer. Used by HDR10, HDR10+, and Dolby Vision. The standard choice for mastered HDR content. |
| `FUSED_TRC_HLG` | 1 | Hybrid Log-Gamma (BBC/NHK). Backward-compatible with SDR displays. Common in live broadcast and streaming where SDR fallback is needed without tone mapping. |


## Tone Mapping Curves

Set `ctx->tonemap.curve` before calling `fused_hdr_init`. The LUT is
precomputed at init time from the selected curve, `peak_nits`, and
`target_nits`.

### `FUSED_TONEMAP_HABLE` (0) — default

Hable/Uncharted 2 filmic curve. Provides natural highlight rolloff with
good shadow detail. A solid general-purpose default for most content.

### `FUSED_TONEMAP_REINHARD` (1)

Reinhard global operator. Simpler curve with lower contrast than Hable.
Preserves more highlight detail at the cost of a flatter midtone range.
Useful when the source is already conservatively graded.

### `FUSED_TONEMAP_BT2390` (2)

ITU-R BT.2390 EETF (Electro-Optical Transfer Function). The ITU
reference standard for HDR-to-SDR conversion. Preferred for broadcast
and regulatory compliance workflows.

### `FUSED_TONEMAP_CUSTOM` (3)

Caller-supplied lookup table. Set `ctx->tonemap.custom_lut` to a
1024-entry `uint8_t` array that maps 10-bit input luma values [0..1023]
to 8-bit output luma values [0..255]. The library applies this LUT
directly without modification. Chroma is scaled proportionally based on
the luma mapping.


## Per-Step Output Selection

The HDR API adds `hdr_flags` and `sdr_flags` to control which output
type is produced at each scale step.

### Requesting outputs

Both `hdr_flags` and `sdr_flags` must be subsets of `requested_flags`.
Set bits in one or both to control what each step produces:

| hdr_flags bit | sdr_flags bit | Effect |
|---------------|---------------|--------|
| Set | Clear | HDR output only (10-bit, no tone mapping) |
| Clear | Set | SDR output only (8-bit, tone-mapped) |
| Set | Set | Both HDR and SDR outputs |
| Clear | Clear | Step is computed internally but no output is stored |

```c
/* 1080p HDR + SDR, 720p SDR only, 360p SDR only */
hdr.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
hdr.hdr_flags       = FUSED_SCALE_1_5X;
hdr.sdr_flags       = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
```

### The `tonemap_1x` flag

Set `ctx->tonemap_1x = 1` to produce an 8-bit tone-mapped copy at the
**original source resolution** (no scaling). The result is written to
`ctx->output_1x`. This is useful for generating an SDR proxy of the
full-resolution source without downscaling.

### Output indexing

Outputs are indexed by `FUSED_IDX_*` constants, the same scheme used
by the 8-bit API:

```c
fused_hdr_output_t   *hdr_out = &ctx->hdr_outputs[FUSED_IDX_1_5X];
fused_scale_output_t *sdr_out = &ctx->sdr_outputs[FUSED_IDX_1_5X];
fused_scale_output_t *sdr_3x  = &ctx->sdr_outputs[FUSED_IDX_3X];
```

Slots for steps not in `achieved_hdr_flags` or `achieved_sdr_flags`
have NULL plane pointers.


## libavcodec Integration (HDR)

HEVC Main10 decode produces frames in `AV_PIX_FMT_YUV420P10LE` (I010)
or `AV_PIX_FMT_P010LE` (P010). Map directly to funnelcake:

```c
#include "funnelcake.h"
#include <libavcodec/avcodec.h>

/* frame is an AVFrame* from HEVC Main10 decode */

fused_hdr_ctx_t hdr = {0};
hdr.src_width     = frame->width;
hdr.src_height    = frame->height;
hdr.src_y_stride  = frame->linesize[0];
hdr.src_uv_stride = frame->linesize[1];

/* Detect format from AVFrame */
if (frame->format == AV_PIX_FMT_YUV420P10LE) {
    hdr.src_format = FUSED_PIX_I010;
} else if (frame->format == AV_PIX_FMT_P010LE) {
    hdr.src_format = FUSED_PIX_P010;
}

hdr.src_transfer    = FUSED_TRC_PQ;
hdr.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
hdr.hdr_flags       = FUSED_SCALE_1_5X;
hdr.sdr_flags       = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

hdr.tonemap.curve       = FUSED_TONEMAP_BT2390;
hdr.tonemap.peak_nits   = 1000;
hdr.tonemap.target_nits = 100;

int rc = fused_hdr_init(&hdr);
if (rc < 0) {
    fprintf(stderr, "fused_hdr_init: error %d\n", rc);
    return rc;
}

/* Process each decoded frame */
if (hdr.src_format == FUSED_PIX_I010) {
    fused_hdr_run(&hdr,
                  (const uint16_t *)frame->data[0],
                  (const uint16_t *)frame->data[1],
                  (const uint16_t *)frame->data[2]);
} else {
    /* P010: UV interleaved in data[1], data[2] is NULL */
    fused_hdr_run(&hdr,
                  (const uint16_t *)frame->data[0],
                  (const uint16_t *)frame->data[1],
                  NULL);
}

/* hdr.hdr_outputs[FUSED_IDX_1_5X] = 10-bit 1.5x, etc. */

fused_hdr_free(&hdr);
```

Note: `frame->linesize[0]` must be 32-byte aligned. Frames decoded from
HEVC Main10 streams at standard resolutions are typically already aligned.
