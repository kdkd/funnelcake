# Funnelcake

Funnelcake is a fused multi-resolution YUV420 scaler. A single call
produces up to four downscaled outputs and up to six upscaled outputs
simultaneously in one pass over the source data, using AVX2 (x86-64) or
NEON (aarch64) SIMD kernels with a portable scalar fallback. An HDR10
path handles 10-bit PQ and HLG input with optional built-in tone mapping
to SDR.

It is designed for video pipelines that need to derive multiple
alternate-resolution copies of each frame - thumbnail generation,
adaptive bitrate encoding ladders, preview streams, super-resolution
ladders - where calling a general-purpose scaler once per output is
prohibitively slow.

The 8-bit SDR path accepts I420 planar (separate Y, U, V planes), 8-bit
unsigned. The 10-bit HDR path accepts I010, P010, I210, and P210 formats
and can produce both HDR and tone-mapped SDR outputs at each downscale
step. Upscaling is available in both paths; upscale outputs on the HDR
path are 10-bit only (no tone-mapping stage).


## How it works

Rather than scaling each output independently from the source, funnelcake
processes all outputs in a single vertical pass. For each group of source rows
(2 rows for the pow2 family, 3 rows for the thirds family), the kernel reads
source data once, computes the horizontal reduction, and writes every output
simultaneously. Each source row is read exactly once regardless of how many
outputs are requested.

Two downscale families are supported:

| Family | Steps available |
|--------|----------------|
| **Thirds** | 1.5× (3:2), 3×, 6×, 12× |
| **Pow2**   | 2×, 4×, 8×, 16× |

Each family is a natural cascade: a 12× thirds output passes through 1.5×,
3×, and 6× intermediate stages. You do not need to request every step; the
library produces intermediate outputs only where explicitly requested. A single
init call may request any combination of steps within one family; the two
families may not be mixed in a single context.

**Upscaling** is a cascading 2× chain of up to five levels (2×, 4×, 8×,
16×, 32×) with an optional 1.5× tail. The tail reads either the source
(when no 2× levels are requested) or the deepest 2× output, producing a
single additional step at 1.5× of that width. A 1080p source can be
upscaled all the way to 8× (15360×8640) in one call; deeper levels are
soft-rejected if they exceed the 16384×16384 size cap.

Upscale and downscale may be requested in the same `fused_scaler_init`
call. Both directions' outputs are produced from a single vertical walk
over the source. See the [Upscale Step Flags](docs/API.md#upscale-step-flags)
section of the API reference for the full permutation table and size
constraints.


## Benchmarks

All measurements are single-threaded median latency over ~1000 iterations
per workload. Each system was built with `make pgo LTO=1 TUNE=native`.
Source frames contain pseudo-random pixel data so the benchmark is not
cache-hot from pattern repetition. libswscale is invoked with
`SWS_BILINEAR` and one `SwsContext` per output target - the "independent"
configuration a naive multi-output libswscale consumer would use. For
downscale workloads libswscale also supports a "cascade" mode where each
output feeds the next, which is roughly 1.5–2× faster than independent
mode on multi-level ladders; even against cascaded libswscale, funnelcake
remains 3–10× faster on every tested CPU. 

Each workload label spells out the exact scales being produced. For
example:

- `down:1.5x,3x,6x` - three downscale outputs at 1.5×, 3×, and 6×
  reduction of the source dimensions
- `up:2x,3x` - a 2× upscale with the optional 1.5× tail applied on top
  (producing an additional 3× output, since 2 × 1.5 = 3)
- `up:2x,4x,8x,16x,32x` - a five-level pow2 upscale cascade
- `down:2x up:2x` - a combined call that produces one 2× downscale AND
  one 2× upscale from the same source frame in a single
  `fused_scaler_run`

Cells below show `funnelcake median time (speedup vs libswscale)`.
Smaller time is better; larger speedup is better.

### SDR downscale v.s. libswscale

**x86_64 / AVX2**

| Workload | Epyc 7302 (Zen 2) | Xeon 6132 (Skylake) | Xeon E5v4 (Broadwell) |
|---|---|---|---|
| 640×360 down:2x                   |   11 µs  (9.9×) |   12 µs (11.8×) |   43 µs  (3.6×) |
| 960×540 down:1.5x,3x              |   86 µs  (6.0×) |   96 µs  (7.1×) |  274 µs  (2.7×) |
| 1280×720 down:2x,4x               |   68 µs (10.0×) |   96 µs  (9.2×) |  252 µs  (3.8×) |
| 1920×1080 down:1.5x,3x,6x         |  372 µs  (7.4×) |  424 µs  (8.3×) |  816 µs  (4.8×) |
| 2560×1440 down:2x,4x,8x           |  315 µs (11.8×) |  449 µs (10.7×) |  640 µs  (8.3×) |
| 3840×2160 down:1.5x,3x,6x,12x     | 2023 µs  (6.6×) | 2046 µs  (8.3×) | 2615 µs  (7.1×) |

**aarch64 / NEON**

| Workload | Graviton 4 (Neoverse V2) | Apple M3 Ultra | Raspberry Pi 5 |
|---|---|---|---|
| 640×360 down:2x                   |   13 µs (13.2×) |   22 µs  (2.6×) |   26 µs (10.4×) |
| 960×540 down:1.5x,3x              |   86 µs  (8.7×) |   47 µs  (6.1×) |  170 µs  (8.3×) |
| 1280×720 down:2x,4x               |   68 µs (14.5×) |   47 µs  (7.6×) |  141 µs (12.5×) |
| 1920×1080 down:1.5x,3x,6x         |  393 µs (16.2×) |  126 µs (11.3×) |  940 µs (14.0×) |
| 2560×1440 down:2x,4x,8x           |  302 µs (17.1×) |  240 µs  (7.7×) | 1139 µs  (8.2×) |
| 3840×2160 down:1.5x,3x,6x,12x     | 1774 µs (15.7×) |  561 µs (12.0×) | 5032 µs (11.2×) |

### SDR upscale v.s. libswscale

**x86_64 / AVX2**

| Workload | Epyc 7302 (Zen 2) | Xeon 6132 (Skylake) | Xeon E5v4 (Broadwell) |
|---|---|---|---|
| 480×270 up:2x                   |   27 µs (10.2×) |   35 µs (10.2×) |   54 µs  (7.0×) |
| 480×270 up:2x,4x                |  128 µs  (7.1×) |  189 µs  (6.1×) |  251 µs  (5.1×) |
| 960×540 up:2x                   |  102 µs  (9.8×) |  156 µs  (8.1×) |  200 µs  (7.0×) |
| 960×540 up:2x,3x                |  936 µs  (2.9×) | 1125 µs  (3.1×) | 1281 µs  (3.0×) |
| 1920×1080 up:2x                 |  679 µs  (6.0×) |  843 µs  (6.0×) |  754 µs  (7.3×) |
| 1920×1080 up:1.5x               |  828 µs  (3.4×) |  953 µs  (3.7×) | 1100 µs  (3.7×) |
| 240×136 up:2x,4x,8x,16x         |  933 µs  (2.5×) | 1140 µs  (2.5×) | 1011 µs  (3.0×) |
| 120×68 up:2x,4x,8x,16x,32x      |  939 µs  (2.1×) | 1133 µs  (2.1×) | 1038 µs  (2.4×) |

**aarch64 / NEON**

| Workload | Graviton 4 (Neoverse V2) | Apple M3 Ultra | Raspberry Pi 5 |
|---|---|---|---|
| 480×270 up:2x                   |   21 µs (22.0×) |   18 µs  (8.6×) |   71 µs (11.1×) |
| 480×270 up:2x,4x                |  107 µs (16.2×) |   89 µs  (6.2×) |  366 µs  (8.6×) |
| 960×540 up:2x                   |   88 µs (22.8×) |   71 µs  (8.5×) |  307 µs (10.3×) |
| 960×540 up:2x,3x                | 1003 µs  (5.4×) |  309 µs  (5.3×) | 1933 µs  (4.7×) |
| 1920×1080 up:2x                 |  360 µs (21.0×) |  276 µs  (8.5×) | 1677 µs  (7.6×) |
| 1920×1080 up:1.5x               |  909 µs  (5.8×) |  238 µs  (6.9×) | 1727 µs  (4.9×) |
| 240×136 up:2x,4x,8x,16x         |  480 µs (10.3×) |  382 µs  (3.7×) | 2053 µs  (4.8×) |
| 120×68 up:2x,4x,8x,16x,32x      |  480 µs  (9.3×) |  386 µs  (3.2×) | 2063 µs  (4.5×) |

On x86 the 1.5× upscale tail is materially slower per byte than the
pure 2× steps because the AVX2 implementation is shuffle-port throughput
limited in its deinterleave → weighted-blend → interleave-store path.
NEON does not have this bottleneck because the 2→3 bilinear maps cleanly
onto `vld2q_u8` / `vst3q_u8`. See [docs/API.md](docs/API.md#performance-notes)
for a longer discussion.

### SDR combined downscale + upscale (single pass) v.s. libswscale

**x86_64 / AVX2**

| Workload | Epyc 7302 (Zen 2) | Xeon 6132 (Skylake) | Xeon E5v4 (Broadwell) |
|---|---|---|---|
| 1920×1080 down:2x up:2x             |  848 µs (5.9×) | 1050 µs (6.0×) |  924 µs (7.2×) |
| 1920×1080 down:1.5x,3x up:2x        | 1180 µs (5.4×) | 1328 µs (5.9×) | 1217 µs (7.0×) |
| 1280×720 down:2x,4x up:2x,4x        | 3037 µs (2.7×) | 2535 µs (3.7×) | 2527 µs (4.2×) |

**aarch64 / NEON**

| Workload | Graviton 4 (Neoverse V2) | Apple M3 Ultra | Raspberry Pi 5 |
|---|---|---|---|
| 1920×1080 down:2x up:2x             |  454 µs (20.0×) |  345 µs (8.2×) | 2031 µs (7.4×) |
| 1920×1080 down:1.5x,3x up:2x        |  695 µs (15.5×) |  393 µs (8.9×) | 2585 µs (7.1×) |
| 1280×720 down:2x,4x up:2x,4x        |  891 µs (15.4×) |  724 µs (5.8×) | 3886 µs (6.2×) |

### HDR10 (10-bit PQ / HLG)

The bench suite does not include a libswscale HDR comparison path, so HDR
numbers are funnelcake's absolute time only. Tone-mapping benchmarks are
omitted. Tone map correctness is being rewritten and the current timings
aren't representative.

**x86_64 / AVX2**

| Workload | Epyc 7302 | Xeon 6132 | Xeon E5v4 |
|---|---|---|---|
| 1920×1080 I010 down:1.5x,3x,6x        |  765 µs |  865 µs |  997 µs |
| 3840×2160 I010 down:1.5x,3x,6x,12x    | 4266 µs | 4243 µs | 5681 µs |
| 3840×2160 P010 down:1.5x,3x,6x,12x    | 5446 µs | 5209 µs | 6734 µs |
| 1920×1080 I010 up:2x                  | 2845 µs | 2474 µs | 2286 µs |
| 1920×1080 I010 down:1.5x,3x up:2x     | 3840 µs | 3409 µs | 3594 µs |

**aarch64 / NEON**

| Workload | Graviton 4 | Apple M3 Ultra | Raspberry Pi 5 |
|---|---|---|---|
| 1920×1080 I010 down:1.5x,3x,6x        |  734 µs |  234 µs |  2147 µs |
| 3840×2160 I010 down:1.5x,3x,6x,12x    | 3233 µs | 1193 µs | 11790 µs |
| 3840×2160 P010 down:1.5x,3x,6x,12x    | 3563 µs | 1420 µs | 13417 µs |
| 1920×1080 I010 up:2x                  |  707 µs |  626 µs |  3393 µs |
| 1920×1080 I010 down:1.5x,3x up:2x     | 1392 µs |  871 µs |  5500 µs |

The P010 row uses the Y + interleaved-UV layout that most HEVC Main10
encoders emit natively; the P010 vs I010 gap on the matching 4K
workload (e.g. 5446 vs 4266 µs on Epyc 7302) is the on-the-fly UV
deinterleave cost, not a fundamental difference in scaling work.

HDR kernels are roughly 2–4× slower per byte than their SDR
counterparts because 10-bit samples halve the number of pixels per
SIMD register and because several per-lane operations (notably 16-bit
averaging on AVX2) lack a single-instruction form and must be expanded
to add-and-shift sequences.

### Graviton 4 is the standout deployment target

The Graviton 4 column deserves calling out explicitly. Against
libswscale on the same hardware, funnelcake's SDR speedups on Graviton
cluster around **15–22× on the pow2 workloads** - the 2× upscales,
downscale ladders from 1080p through 4K, and single-pass combined
down+up calls. For comparison, the same set of workloads sits around
6–12× on Apple M3 Ultra, 7–14× on Raspberry Pi 5, and 5–10× on the
x86 server CPUs in the tables above. The one exception is the 1.5×
upscale tail (`up:2x,3x`, `up:1.5x`): that kernel is compute-bound on
every platform and settles at ~5–6× everywhere, Graviton included.

The most dramatic rows:

- **Pure 2× upscales** (`480×270 up:2x`, `960×540 up:2x`,
  `1920×1080 up:2x`): **21–23×** faster than libswscale.
- **Single-pass combined downscale + upscale**
  (`1920×1080 down:2x up:2x`, `down:1.5x,3x up:2x`,
  `1280×720 down:2x,4x up:2x,4x`): **15–20×** faster.
- **Downscale ladders** at 1080p through 4K: **15–17×** faster against
  independent libswscale, still **~7×** faster even against libswscale's
  cascade mode.

In absolute numbers, a `c8g.2xlarge` instance (one Graviton 4 vCPU)
processes a 1920×1080 thirds-family downscale ladder
(`down:1.5x,3x,6x`) in **393 µs**, a complete 4K thirds ladder
(`down:1.5x,3x,6x,12x`) in **1.77 ms**, and a combined 1080p
downscale + 2× upscale in **454 µs**. At 60 fps each of those consumes
less than 11% of a single core's frame budget - meaning a single
Graviton 4 core can run the 1080p ladder for **~42 live streams** in
parallel, or the full 4K ladder for **~9 streams**, with headroom left
over.

We don't have a single smoking-gun explanation for why Graviton's
relative advantage is so much larger than other aarch64 parts. The
likely contributors are that libswscale's ARM64 bilinear path is less
aggressively hand-tuned than its x86 AVX2 path, the Neoverse V2 cores
in Graviton 4 have generous SIMD throughput that funnelcake's
`vld2q` / `vst3q` / `vrhaddq_u8` inner loops fully exploit, and
libswscale's more cache-unfriendly memory access pattern interacts
badly with the platform's memory subsystem. Whatever the exact cause,
Graviton 4 is by a clear margin the deployment target where using
funnelcake instead of libswscale produces the largest absolute savings
per core for real-time multi-resolution video pipelines.

### A note on the memory wall

Several of the workloads in these tables have been profiled down to
effectively one load + one pair-average + one store per output byte, and
at that point the kernel is doing the minimum useful work per byte and
no amount of further SIMD cleverness will make them faster on current
CPU/memory architectures. On systems profiled while developing
funnelcake, the following configurations were observed to hit the
single-core memory bandwidth ceiling - funnelcake already runs at that
ceiling, so any further speedup in these specific cases would require
wider memory buses or multi-channel striping, not a better kernel:

- **Straight 2× upscale at 1080p on DDR5 systems**: on a Zen 5 system
  this workload is ~15 MB of source read + output write, and funnelcake
  completes it in roughly the time it takes the memory controller to
  physically move that amount of data (~82 GB/s effective, which matches
  the single-core sustained DDR5 bandwidth of that platform).
- **Shallow pow2 downscales at 4K on Apple Silicon**: the 2×/4× levels
  of a 4K→1080p→540p ladder are dominated by memory traffic from the
  source and into the first output level; on M3 Ultra these run close
  to the ~60 GB/s single-core ceiling of the unified memory system.
- **Small-source workloads on CPUs with very fast memory subsystems**:
  e.g. `640×360 down:2x` on Apple Silicon completes in ~22 µs - an
  absolute time where libswscale is *also* memory-bound, so the relative
  speedup in the table (2.6×) understates how much work funnelcake is
  doing and really just reflects that both libraries are waiting on
  the same DRAM.

In these cases the kernel's job is to get out of the memory subsystem's
way, and the benchmarks above confirm that it does. The workloads where
funnelcake's speedup keeps growing with CPU improvements (e.g. deep
thirds cascades, the 1.5× upscale tail, combined down+up calls) are
all compute-bound, and those are where the op-count and register
scheduling work inside the kernels continues to pay off.


## Source frame requirements

These constraints apply to the source data passed to `fused_scaler_init` and
`fused_scaler_run` (the 8-bit SDR API). The 10-bit HDR API
(`fused_hdr_init` / `fused_hdr_run`) has its own format rules and
accepts several additional layouts - see [HDR10 support](#hdr10-support)
below for the full HDR format list.

### Format
- **YUV420 I420 planar, 8-bit unsigned.** The three planes (Y, U, V)
  must be passed separately. 4:2:2 chroma subsampling, semi-planar
  layouts (NV12), packed formats (UYVY, YUYV), and other packed
  arrangements are **not** supported on this SDR path.
- If you need **10-bit** samples, **4:2:2** chroma, or the **P010 /
  P210 semi-planar** layouts (Y plane + interleaved UV plane), use the
  HDR API instead - it handles all four of I010, P010, I210, P210 and
  can produce 10-bit HDR outputs, 8-bit SDR outputs, or both from the
  same call. You do not need to be scaling "HDR content" to use the
  HDR API: it is simply the 10-bit / wider-chroma entry point.
- **Downscaling, upscaling, or both** in a single pass over the source
  (applies to both SDR and HDR APIs).

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
A single `fused_scaler_ctx_t` may only use downscale steps from **one
family** per init. Requesting `FUSED_SCALE_3X | FUSED_SCALE_4X` (thirds
+ pow2) returns `FUSED_ERR_INVALID_FLAGS`. Use two separate contexts if
you need both downscale families.

Upscaling is independent of the downscale family selection and may be
combined with either thirds or pow2 downscale flags in the same init
call.

### Upscale constraints

Upscale flags (`FUSED_UPSCALE_2X`, `FUSED_UPSCALE_4X`, `FUSED_UPSCALE_8X`,
`FUSED_UPSCALE_16X`, `FUSED_UPSCALE_32X`) form a cascading 2× chain.
The mask set in `ctx->upscale_flags` must be a **contiguous prefix** of
the cascade - valid values are `0`, `{2x}`, `{2x,4x}`, `{2x,4x,8x}`,
`{2x,4x,8x,16x}`, or `{2x,4x,8x,16x,32x}`. Setting a non-contiguous
mask (e.g. `{4x}` alone or `{2x,8x}`) returns `FUSED_ERR_INVALID_FLAGS`.

Setting `ctx->upscale_tail_1_5x = 1` appends a single 1.5x bilinear step
on top of the deepest pow2 level, or on the source directly if
`upscale_flags == 0`. See the
[Upscale Step Flags](docs/API.md#upscale-step-flags) section of the API
reference for the full table of valid combinations.

**Size cap**: individual upscale levels are soft-rejected when their luma
output exceeds 16384×16384. For example, a 1920×1080 source with
`FUSED_UPSCALE_POW2_MASK` produces 2×, 4×, and 8× successfully; 16×
(30720×17280) and 32× (61440×34560) are rejected and `FUSED_WARN_BIT_PARTIAL`
is set in the return code.

**1.5x upscale performance**: the 1.5x tail is materially slower per
output byte than any of the 2× steps on AVX2 because it uses a weighted
85/171 bilinear blend whose inner loop is dominated by shuffle-port
throughput. On Zen 2 / Haswell and later, the 256-bit kernel is roughly
5-8× slower per byte than a straight 2× step but still substantially
faster than libswscale's bilinear upscale. On Zen 1 the gap is wider
because Zen 1 double-pumps 256-bit AVX2 instructions through its 128-bit
datapath. NEON does not have this bottleneck - the 2→3 pattern maps
cleanly onto `vld2q_u8` / `vst3q_u8`. Choose the 1.5x tail with this in
mind on compute-limited x86 targets.

### Thread safety
Each context is independent and not thread-safe. Use one context per thread.
Concurrent reads from separate contexts on the same source data are safe.


## Getting started

See **[INSTALL.md](INSTALL.md)** for build instructions, compiler requirements,
PGO and LTO setup, CPU-specific tuning recommendations, and static-library
compatibility notes for downstream consumers.

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

A combined downscale + upscale example:

```c
#include "funnelcake.h"

/* 1920×1080 source: downscale to 960×540 + upscale to 3840×2160 in one pass */
fused_scaler_ctx_t scaler = {0};
scaler.src_width     = 1920;
scaler.src_height    = 1080;
scaler.src_y_stride  = (1920 + 31) & ~31;
scaler.src_uv_stride = (960  + 31) & ~31;

scaler.requested_flags    = FUSED_SCALE_2X;                             /*  960×540  */
scaler.upscale_flags      = FUSED_UPSCALE_2X;                           /* 3840×2160 */
scaler.upscale_tail_1_5x  = 0;

int rc = fused_scaler_init(&scaler);
if (rc < 0) { /* hard error */ }

fused_scaler_run(&scaler, frame_y, frame_u, frame_v);

fused_scale_output_t *out_half = &scaler.outputs[FUSED_IDX_2X];            /*  960×540  */
fused_scale_output_t *out_4k   = &scaler.upscale_outputs[FUSED_UP_IDX_2X]; /* 3840×2160 */

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
