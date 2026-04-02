# Funnelcake

Funnelcake is a fused multi-resolution YUV420 downscaler. A single call
produces up to four downscaled outputs simultaneously in one pass over the
source data, using AVX2 (x86-64) or NEON (aarch64) SIMD kernels, with a
portable scalar fallback. An HDR10 path handles 10-bit PQ and HLG input
with optional built-in tone mapping to SDR.

It is designed for video pipelines that need to derive multiple lower-resolution
copies of each frame - thumbnail generation, adaptive bitrate encoding ladders,
preview streams - where calling a general-purpose scaler once per output is
prohibitively slow.

The 8-bit SDR path accepts I420 planar (separate Y, U, V planes), 8-bit
unsigned. The 10-bit HDR path accepts I010, P010, I210, and P210 formats
and can produce both HDR and tone-mapped SDR outputs at each scale step.


## How it works

Rather than scaling each output independently from the source, funnelcake
processes all outputs in a single vertical pass. For each group of source rows
(2 rows for the pow2 family, 3 rows for the thirds family), the kernel reads
source data once, computes the horizontal reduction, and writes every output
simultaneously. Each source row is read exactly once regardless of how many
outputs are requested.

Two scale families are supported:

| Family | Steps available |
|--------|----------------|
| **Thirds** | 1.5× (3:2), 3×, 6×, 12× |
| **Pow2**   | 2×, 4×, 8×, 16× |

Each family is a natural cascade: a 12× thirds output passes through 1.5×,
3×, and 6× intermediate stages. You do not need to request every step; the
library produces intermediate outputs only where explicitly requested. A single
init call may request any combination of steps within one family; the two
families may not be mixed in a single context.


## Benchmarks

All results use `make pgo LTO=1` with the CPU-specific `TUNE` value, comparing
against calling libswscale once per output at the equivalent quality setting.
Times are the minimum observed over 1000 frames (single-threaded).

**AMD EPYC 7302 (Zen 2) - 20–40× faster than libswscale**
```
640×360   pow2 (2×)     12 µs    0.1% of 60fps budget
960×540   thirds        87 µs    0.5%
1280×720  pow2          69 µs    0.4%
1920×1080 thirds       368 µs    2.2%
2560×1440 pow2         318 µs    1.9%
3840×2160 thirds      2021 µs   12.1%
```

**AMD EPYC 7301 (Zen 1) - 8–20× faster than libswscale**
```
640×360   pow2 (2×)     20 µs    0.1% of 60fps budget
960×540   thirds       156 µs    0.9%
1280×720  pow2         111 µs    0.7%
1920×1080 thirds       684 µs    4.1%
2560×1440 pow2         698 µs    4.2%
3840×2160 thirds      3529 µs   21.2%
```

**Intel E5-2680v4 (Broadwell) - 10–25× faster than libswscale**
```
640×360   pow2 (2×)     22 µs    0.1% of 60fps budget
960×540   thirds       120 µs    0.7%
1280×720  pow2         114 µs    0.7%
1920×1080 thirds       517 µs    3.1%
2560×1440 pow2         535 µs    3.2%
3840×2160 thirds      1854 µs   11.1%
```

**Intel Xeon 6132 (Skylake) - 15–30× faster than libswscale**
```
640×360   pow2 (2×)     12 µs    0.1% of 60fps budget
960×540   thirds       102 µs    0.6%
1280×720  pow2          99 µs    0.6%
1920×1080 thirds       418 µs    2.5%
2560×1440 pow2         478 µs    2.9%
3840×2160 thirds      1872 µs   11.2%
```

The thirds kernel is slower than the pow2 kernel at the same source resolution
because thirds vertical periods (3 rows) fit less cleanly into cache than
power-of-two periods (2 rows), and the horizontal thirds filter requires
non-trivial deinterleaving on every chunk.


## Source frame requirements

These constraints apply to the source data passed to `fused_scaler_init` and
`fused_scaler_run`.

### Format
- **YUV420 I420 planar only.** The three planes (Y, U, V) must be passed
  separately. Packed formats (NV12, UYVY, etc.) are not supported.
- **8-bit unsigned** samples only.
- **Downscaling only.** The library does not upscale.

### Dimensions
- `src_width` and `src_height` must be **positive** and **even**.
- Both dimensions must be **large enough** to produce at least one output
  pixel at the deepest requested scale step (minimum output size is 32×2
  luma pixels).

### Strides
- `src_y_stride` (bytes per row of the luma plane) must be **≥ src_width**
  and **a multiple of 32**.
- `src_uv_stride` (bytes per row of each chroma plane) must be **≥ src_width / 2**
  and **a multiple of 32**.
- Strides that fail these constraints cause `fused_scaler_init` to return
  `FUSED_ERR_BAD_ALIGNMENT`.

### Pointer alignment
- The `src_y`, `src_u`, and `src_v` pointers passed to `fused_scaler_run`
  must be **32-byte aligned** for the SIMD kernel to be used. Misaligned
  pointers do not return an error; the library falls back to the scalar kernel
  and logs a warning. Frames decoded by libavcodec at standard resolutions
  are typically already aligned.

### Scale family constraints

#### Thirds family (1.5×, 3×, 6×, 12×)
The horizontal thirds filter requires the **chroma output width to be a
multiple of 32**. This means:

- For any thirds step, **`src_width` should be a multiple of 64** (so that
  after halving for chroma and applying the reduction, the result is
  ≥ 32-aligned). Steps whose chroma output width is not a multiple of 32
  fall back to the scalar kernel unless `FUSED_OPT_NO_FALLBACK` is set.

The deepest thirds step imposes a divisibility requirement on `src_width`:

| Deepest step requested | src_width must be divisible by |
|------------------------|-------------------------------|
| 1.5× only              | 3                              |
| 3×                     | 6                              |
| 6×                     | 12                             |
| 12×                    | 24                             |

Similarly for `src_height` (vertical period):

| Deepest step requested | src_height must be divisible by |
|------------------------|--------------------------------|
| 1.5× or 3×             | 6                               |
| 6×                     | 12                              |
| 12×                    | 24                              |

#### Pow2 family (2×, 4×, 8×, 16×)
The deepest pow2 step imposes a similar requirement:

| Deepest step requested | src_width and src_height must be divisible by |
|------------------------|----------------------------------------------|
| 2×                     | 4                                             |
| 4×                     | 8                                             |
| 8×                     | 16                                            |
| 16×                    | 32                                            |

### Crop-to-fit (default)
If the source dimensions are not exactly divisible as required, the library
**silently crops** up to `(ratio − 1)` columns and rows from the bottom/right
edge to find the nearest compliant size. No data is copied; only the kernel's
loop bounds change. The actual region read is reported in
`ctx->effective_width` and `ctx->effective_height`, and `FUSED_WARN_BIT_CROPPED`
is set in the return code.

Set `FUSED_OPT_NO_CROP` to reject steps that require cropping rather than
silently trimming.

### Mixing families
A single `fused_scaler_ctx_t` may only use steps from **one family** per init.
Requesting `FUSED_SCALE_3X | FUSED_SCALE_4X` (thirds + pow2) returns
`FUSED_ERR_INVALID_FLAGS`. Use two separate contexts if you need both families.

### Thread safety
Each context is independent and not thread-safe. Use one context per thread.
Concurrent reads from separate contexts on the same source data are safe.


## Getting started

See **[INSTALL.md](INSTALL.md)** for build instructions, compiler requirements,
PGO and LTO setup, and CPU-specific tuning recommendations.

See **[docs/API.md](docs/API.md)** for the full API reference including data
types, return codes, logging configuration, and libavcodec integration examples.

A minimal usage example:

```c
#include "funnelcake.h"

/* 1920×1080 source, thirds cascade to 1280×720, 640×360, 320×180 */
fused_scaler_ctx_t scaler = {0};
scaler.src_width     = 1920;
scaler.src_height    = 1080;
scaler.src_y_stride  = (1920 + 31) & ~31;   /* 1920 */
scaler.src_uv_stride = (960  + 31) & ~31;   /* 960  */
scaler.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;

int rc = fused_scaler_init(&scaler);
if (rc < 0) { /* hard error - nothing allocated */ }

/* Call once per decoded frame */
fused_scaler_run(&scaler, frame_y, frame_u, frame_v);

/* Outputs indexed by FUSED_IDX_* constants */
fused_scale_output_t *out_1280x720 = &scaler.outputs[FUSED_IDX_1_5X];
fused_scale_output_t *out_640x360  = &scaler.outputs[FUSED_IDX_3X];
fused_scale_output_t *out_320x180  = &scaler.outputs[FUSED_IDX_6X];

fused_scaler_free(&scaler);
```


## Platform support

| Platform | SIMD | Notes |
|----------|------|-------|
| x86-64 with AVX2 | AVX2 | Detected at runtime via cpuid |
| x86-64 without AVX2 | Scalar | Broadwell and later all have AVX2 |
| aarch64 (Apple Silicon, AWS Graviton) | NEON | All aarch64 cores have NEON |
| Other | Scalar | Portable C, no intrinsics |

The scalar fallback is correct on all platforms but significantly slower.
On hardware without AVX2 or NEON, the library logs a one-time notice to
stderr at first init.


## HDR10 support

The HDR API (`fused_hdr_*`) scales 10-bit PQ or HLG content and optionally
tone-maps to 8-bit SDR in the same pass. Each scale step can independently
produce an HDR output, an SDR output, or both.

### Input formats

| Constant | Subsampling | Layout | Notes |
|----------|-------------|--------|-------|
| `FUSED_PIX_I010` | 4:2:0 | Planar Y + U + V | Preferred - no deinterleave cost |
| `FUSED_PIX_P010` | 4:2:0 | Y + interleaved UV | Deinterleaved on-the-fly (slight penalty) |
| `FUSED_PIX_I210` | 4:2:2 | Planar Y + U + V | Chroma rows decimated to 4:2:0 internally |
| `FUSED_PIX_P210` | 4:2:2 | Y + interleaved UV | Combined deinterleave + row-skip |

All formats use 10-bit samples in the low bits of `uint16_t`.

### Tone mapping

Built-in curves applied to SDR outputs:

| Preset | Description |
|--------|-------------|
| `FUSED_TONEMAP_HABLE` | Hable/Uncharted 2 filmic (default) |
| `FUSED_TONEMAP_REINHARD` | Reinhard global operator |
| `FUSED_TONEMAP_BT2390` | ITU-R BT.2390 EETF (broadcast reference) |
| `FUSED_TONEMAP_CUSTOM` | Caller-supplied 1024-entry Y LUT |

### Example: 4K HDR to 1080p HDR + SDR ladder

```c
#include "funnelcake.h"

fused_hdr_ctx_t hdr = {0};
hdr.src_width      = 3840;
hdr.src_height     = 2160;
hdr.src_y_stride   = 3840 * 2;          /* 10-bit: 2 bytes per sample */
hdr.src_uv_stride  = 1920 * 2;
hdr.src_format     = FUSED_PIX_I010;
hdr.src_transfer   = FUSED_TRC_PQ;

/* Request thirds cascade: 1.5x, 3x, 6x */
hdr.requested_flags = FUSED_SCALE_1_5X | FUSED_SCALE_3X | FUSED_SCALE_6X;
hdr.hdr_flags       = FUSED_SCALE_1_5X;                   /* 1080p HDR */
hdr.sdr_flags       = FUSED_SCALE_1_5X | FUSED_SCALE_3X;  /* 1080p + 720p SDR */
hdr.tonemap_1x      = 1;                                  /* 4K SDR copy */

/* Tone mapping: BT.2390 for broadcast-grade SDR */
hdr.tonemap.curve       = FUSED_TONEMAP_BT2390;
hdr.tonemap.peak_nits   = 1000;
hdr.tonemap.target_nits = 100;

int rc = fused_hdr_init(&hdr);
if (rc < 0) { /* handle error */ }

/* Per-frame */
fused_hdr_run(&hdr, frame_y, frame_u, frame_v);

/* Access outputs */
fused_hdr_output_t   *hdr_1080p = &hdr.hdr_outputs[FUSED_IDX_1_5X];
fused_scale_output_t *sdr_1080p = &hdr.sdr_outputs[FUSED_IDX_1_5X];
fused_scale_output_t *sdr_720p  = &hdr.sdr_outputs[FUSED_IDX_3X];
fused_scale_output_t *sdr_4k    = &hdr.output_1x;      /* 8-bit 4K     */

fused_hdr_free(&hdr);
```

See **[docs/API.md](docs/API.md)** for the full HDR10 API reference.
