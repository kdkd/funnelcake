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

/* Access outputs — indexed by bit position */
fused_scale_output_t *out_1_5x = &scaler.outputs[0]; /* FUSED_SCALE_1_5X bit 0 */
fused_scale_output_t *out_3x   = &scaler.outputs[2]; /* FUSED_SCALE_3X   bit 2 */
fused_scale_output_t *out_6x   = &scaler.outputs[4]; /* FUSED_SCALE_6X   bit 4 */

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
| `outputs[8]` | `fused_scale_output_t` | One slot per bit position. Slots for steps not in `achieved_flags` have NULL plane pointers. |

The `_internal` field is opaque; do not read or write it.

---

### `fused_scale_output_t`

Describes one downscaled output. Indexed by the bit position of the
corresponding `FUSED_SCALE_*` flag:

```
outputs[0]  FUSED_SCALE_1_5X
outputs[1]  FUSED_SCALE_2X
outputs[2]  FUSED_SCALE_3X
outputs[3]  FUSED_SCALE_4X
outputs[4]  FUSED_SCALE_6X
outputs[5]  FUSED_SCALE_8X
outputs[6]  FUSED_SCALE_12X
outputs[7]  FUSED_SCALE_16X
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
/* outputs[0] -> 1.5x,  outputs[2] -> 3x,  outputs[4] -> 6x  */
fused_scale_output_t *half = &scaler.outputs[0];
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
