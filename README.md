# Funnelcake

Funnelcake is a fused multi-resolution YUV420 scaler. A single call
produces up to four downscaled outputs and up to six upscaled outputs
simultaneously in one pass over the source data, using AVX2/AVX-512
(x86-64), NEON (aarch64), or RVV 1.0 (RISC-V) SIMD kernels with a
portable scalar fallback. An HDR10 path handles 10-bit PQ and HLG input with optional
built-in tone mapping to SDR.

It is designed for video pipelines that need to derive multiple
alternate-resolution copies of each frame - thumbnail generation,
adaptive bitrate encoding ladders, preview streams, super-resolution
ladders - where calling a general-purpose scaler once per output is
prohibitively slow.

The 8-bit SDR path accepts I420 planar (separate Y, U, V planes), 8-bit
unsigned. The 10-bit HDR path accepts I010, P010, I210, and P210 formats
and can produce both HDR and tone-mapped SDR outputs at each downscale
step. Upscaling is available in both paths; HDR upscale outputs are
10-bit, with an optional tone-mapped 8-bit SDR copy per upscale level
(`upscale_sdr_flags`).


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

All measurements are single-threaded latency over ~1000 iterations per
workload. Each system was built with `make pgo LTO=1 TUNE=native` (see
[INSTALL.md](INSTALL.md)). Source frames contain pseudo-random pixel data
so the benchmark is not cache-hot from pattern repetition. libswscale is
invoked with `SWS_BILINEAR` and one `SwsContext` per output target - the
"independent" configuration a naive multi-output libswscale consumer
would use. For downscale workloads libswscale also supports a "cascade"
mode where each output feeds the next, which is roughly 1.5–2× faster
than independent mode on multi-level ladders; even against cascaded
libswscale, funnelcake remains 3–10× faster on every tested AVX2/NEON CPU
and 13–16× faster on the AVX-512 path.

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

### SDR downscale vs libswscale

**x86_64 / AVX2 & AVX-512**

| Workload | Ryzen 9955HX (Zen 5, AVX-512) | Epyc 7302 (Zen 2, AVX2) | Xeon 6132 (Skylake, AVX2) | Xeon E5v4 (Broadwell, AVX2) |
|---|---|---|---|---|
| 640×360 down:2x                   |    3 µs (16.5×) |    6 µs (18.6×) |    6 µs (20.7×) |   32 µs  (4.4×) |
| 960×540 down:1.5x,3x              |   13 µs (25.1×) |   55 µs  (9.6×) |   69 µs  (9.0×) |  182 µs  (3.8×) |
| 1280×720 down:2x,4x               |   14 µs (27.6×) |   41 µs (16.6×) |   71 µs (11.5×) |  176 µs  (5.5×) |
| 1920×1080 down:1.5x,3x,6x         |   53 µs (30.9×) |  226 µs (12.3×) |  286 µs (11.6×) |  537 µs  (6.7×) |
| 2560×1440 down:2x,4x,8x           |   71 µs (31.2×) |  192 µs (19.6×) |  364 µs (12.2×) |  479 µs (10.4×) |
| 3840×2160 down:1.5x,3x,6x,12x     |  200 µs (39.3×) | 1512 µs  (8.9×) | 1470 µs (10.7×) | 1672 µs (10.3×) |

The AVX-512 kernels require the F+BW+VL+VBMI feature set (Zen 4 and
later, Ice Lake and later) and are selected at runtime. The Xeon 6132
advertises AVX-512 but lacks VBMI, so funnelcake deliberately keeps it
on the AVX2 kernels - on that generation 512-bit code downclocks the
core and AVX2 is the faster choice anyway.

**aarch64 / NEON**

| Workload | Graviton 4 (Neoverse V2) | Apple M5 Max | Raspberry Pi 5 |
|---|---|---|---|
| 640×360 down:2x                   |    9 µs (19.2×) |     3 µs (14.2×) |    14 µs (19.3×) |
| 960×540 down:1.5x,3x              |   64 µs (11.8×) |    23 µs (11.0×) |   137 µs (10.3×) |
| 1280×720 down:2x,4x               |   46 µs (21.5×) |    16 µs (19.6×) |    86 µs (20.6×) |
| 1920×1080 down:1.5x,3x,6x         |  294 µs (21.5×) |   102 µs (12.2×) |   898 µs (14.5×) |
| 2560×1440 down:2x,4x,8x           |  216 µs (23.6×) |    78 µs (20.7×) |  1079 µs (8.7×) |
| 3840×2160 down:1.5x,3x,6x,12x     | 1346 µs (20.7×) |   435 µs (13.3×) |  4417 µs (12.8×) |

### SDR upscale vs libswscale

**x86_64 / AVX2 & AVX-512**

The upscale kernels themselves are AVX2 on every x86 part - the large
terminal upscale outputs are store-bandwidth bound, so wider vectors
have nothing to add there - but the AVX-512 column below reflects the
whole-call timing on that system.

| Workload | Ryzen 9955HX (Zen 5, AVX-512) | Epyc 7302 (Zen 2, AVX2) | Xeon 6132 (Skylake, AVX2) | Xeon E5v4 (Broadwell, AVX2) |
|---|---|---|---|---|
| 480×270 up:2x                   |    7 µs (17.7×) |   18 µs (15.5×) |   21 µs (14.5×) |   48 µs  (9.8×) |
| 480×270 up:2x,4x                |   40 µs (10.5×) |   93 µs  (9.8×) |  175 µs  (6.2×) |  226 µs  (7.0×) |
| 960×540 up:2x                   |   30 µs (16.1×) |   74 µs (13.5×) |  149 µs  (8.0×) |  178 µs  (7.8×) |
| 960×540 up:2x,3x                |  248 µs  (5.2×) |  604 µs  (4.5×) |  682 µs  (4.9×) |  838 µs  (4.7×) |
| 1920×1080 up:2x                 |  120 µs (16.0×) |  482 µs  (8.4×) |  691 µs  (6.8×) |  671 µs  (8.2×) |
| 1920×1080 up:1.5x               |  213 µs  (6.5×) |  519 µs  (5.3×) |  524 µs  (6.5×) |  706 µs  (5.7×) |
| 240×136 up:2x,4x,8x,16x         |  166 µs  (5.3×) |  706 µs  (3.2×) |  930 µs  (2.7×) |  976 µs  (3.1×) |
| 120×68 up:2x,4x,8x,16x,32x      |  169 µs  (4.3×) |  712 µs  (2.8×) |  931 µs  (2.2×) | 1202 µs  (2.1×) |

**aarch64 / NEON**

| Workload | Graviton 4 (Neoverse V2) | Apple M5 Max | Raspberry Pi 5 |
|---|---|---|---|
| 480×270 up:2x                   |   20 µs (23.5×) |    13 µs (10.2×) |    48 µs (16.5×) |
| 480×270 up:2x,4x                |  103 µs (16.8×) |    60 µs (8.0×) |   339 µs (9.3×) |
| 960×540 up:2x                   |   85 µs (23.2×) |    47 µs (11.0×) |   300 µs (10.6×) |
| 960×540 up:2x,3x                | 1025 µs  (5.3×) |   273 µs (5.2×) |  1933 µs (4.7×) |
| 1920×1080 up:2x                 |  344 µs (22.0×) |   191 µs (10.9×) |  1750 µs (7.3×) |
| 1920×1080 up:1.5x               |  924 µs  (5.8×) |   228 µs (6.3×) |  1700 µs (5.0×) |
| 240×136 up:2x,4x,8x,16x         |  470 µs (10.5×) |   259 µs (4.8×) |  3114 µs (3.2×) |
| 120×68 up:2x,4x,8x,16x,32x      |  467 µs  (9.5×) |   262 µs (4.2×) |  3139 µs (2.9×) |

On x86 the 1.5× upscale tail is slower per byte than the pure 2× steps:
AVX2 has no 3-way interleaved store, so assembling the 2-to-3 output costs
shuffle-port work the 2× kernels avoid entirely. NEON has the structural
advantage here, because the 2-to-3 bilinear maps cleanly onto `vld2q_u8` /
`vst3q_u8`. See [docs/API.md](docs/API.md#performance-notes) for a longer
discussion.

### SDR combined downscale + upscale (single pass) vs libswscale

**x86_64 / AVX2 & AVX-512**

| Workload | Ryzen 9955HX (Zen 5, AVX-512) | Epyc 7302 (Zen 2, AVX2) | Xeon 6132 (Skylake, AVX2) | Xeon E5v4 (Broadwell, AVX2) |
|---|---|---|---|---|
| 1920×1080 down:2x up:2x             |  148 µs (16.2×) |  692 µs (7.2×) |  947 µs (6.1×) |  800 µs (8.0×) |
| 1920×1080 down:1.5x,3x up:2x        |  162 µs (19.4×) |  972 µs (6.3×) | 1144 µs (6.4×) |  979 µs (8.0×) |
| 1280×720 down:2x,4x up:2x,4x        |  488 µs  (7.3×) | 2584 µs (3.0×) | 2463 µs (3.6×) | 2119 µs (4.5×) |

**aarch64 / NEON**

| Workload | Graviton 4 (Neoverse V2) | Apple M5 Max | Raspberry Pi 5 |
|---|---|---|---|
| 1920×1080 down:2x up:2x             |  426 µs (21.4×) |   223 µs (11.1×) |  2743 µs (5.5×) |
| 1920×1080 down:1.5x,3x up:2x        |  594 µs (18.1×) |   286 µs (10.7×) |  2875 µs (6.5×) |
| 1280×720 down:2x,4x up:2x,4x        |  832 µs (16.5×) |   460 µs (8.0×) |  4951 µs (4.9×) |

### HDR10 (10-bit PQ / HLG)

The bench suite does not include a libswscale HDR comparison path, so HDR
numbers are funnelcake's absolute time only. Rows marked `tone` produce
tone-mapped 8-bit SDR outputs at every ladder step, `HDR+tone` produces
both the 10-bit HDR and the tone-mapped SDR output at each step, and
`tone 1x` is a source-resolution tone map with no scaling. The `tone`
rows tone-map each output at its own resolution *after* scaling, so a
full 4K ladder tone-maps only ~59% as many pixels as `tone 1x` does -
which is why the 1:1 row costs more than a ladder despite doing no
scaling work. All of them run the full tone-mapping pipeline (PQ-domain
tone curve, BT.2020 NCL reconstruction, BT.2020-to-BT.709 gamut conversion,
BT.709 re-encode) through the SIMD kernels; AVX-512, AVX2, NEON, and RVV
each have a dedicated tone-mapping kernel that matches the scalar
reference bit for bit.

**x86_64 / AVX2 & AVX-512**

| Workload | Ryzen 9955HX (AVX-512) | Epyc 7302 (AVX2) | Xeon 6132 (AVX2) | Xeon E5v4 (AVX2) |
|---|---|---|---|---|
| 1920×1080 I010 down:1.5x,3x,6x        |   92 µs |  264 µs |  391 µs |  668 µs |
| 3840×2160 I010 down:1.5x,3x,6x,12x    |  894 µs | 2788 µs | 2930 µs | 3619 µs |
| 3840×2160 P010 down:1.5x,3x,6x,12x    | 1167 µs | 3495 µs | 3804 µs | 5513 µs |
| 1920×1080 I010 up:2x                  |  480 µs | 2055 µs | 2087 µs | 2189 µs |
| 1920×1080 I010 down:1.5x,3x up:2x     |  692 µs | 2658 µs | 2646 µs | 3305 µs |
| 1920×1080 I010 down:1.5x,3x,6x tone   |  342 µs | 3290 µs | 5132 µs | 2538 µs |
| 3840×2160 I010 down:1.5x,3x,6x,12x tone | 2024 µs | 15433 µs | 23328 µs | 12412 µs |
| 1920×1080 I010 down:1.5x,3x,6x HDR+tone |  344 µs | 3285 µs | 5149 µs | 2555 µs |
| 3840×2160 I010 tone 1x                | 2003 µs | 20753 µs | 33405 µs | 14348 µs |

**aarch64 / NEON**

| Workload | Graviton 4 | Apple M5 Max | Raspberry Pi 5 |
|---|---|---|---|
| 1920×1080 I010 down:1.5x,3x,6x        |  284 µs |   138 µs |  1816 µs |
| 3840×2160 I010 down:1.5x,3x,6x,12x    | 1585 µs |   684 µs |  8896 µs |
| 3840×2160 P010 down:1.5x,3x,6x,12x    | 1975 µs |   875 µs | 10441 µs |
| 1920×1080 I010 up:2x                  |  755 µs |   432 µs |  4046 µs |
| 1920×1080 I010 down:1.5x,3x up:2x     | 1031 µs |   580 µs |  6068 µs |
| 1920×1080 I010 down:1.5x,3x,6x tone   | 2471 µs |   744 µs |  4697 µs |
| 3840×2160 I010 down:1.5x,3x,6x,12x tone | 10229 µs |  3334 µs | 20407 µs |
| 1920×1080 I010 down:1.5x,3x,6x HDR+tone | 2465 µs |   746 µs |  4674 µs |
| 3840×2160 I010 tone 1x                | 14423 µs |  4189 µs | 19323 µs |

The P010 row uses the Y + interleaved-UV layout that most HEVC Main10
encoders emit natively; the P010 vs I010 gap on the matching 4K
workload (e.g. 3495 vs 2788 µs on Epyc 7302) is the on-the-fly UV
deinterleave cost, not a fundamental difference in scaling work.

The AVX2 and NEON HDR kernels are roughly 2–4× slower per byte than
their SDR counterparts, mostly because 10-bit samples halve the number
of pixels per SIMD register. Both express the common weighted blend as a
signed difference followed by a saturating rounding multiply-high in
16-bit lanes. AVX2 also keeps a widened multiply-add variant for the
dense 12× pipeline, where mixing the two formulations balances the
execution ports better than issuing only multiply-high instructions.
AVX-512 claws most of the lane-count penalty back - a 512-bit register
holds as many 16-bit lanes as a 256-bit register holds 8-bit ones - so
on the Zen 5 column above the HDR rows land much closer to their SDR
twins (92 µs vs 53 µs at the 1080p thirds ladder).

### RISC-V (RVV 1.0)

Tested on a SpacemiT K1 (uarch `ky,x60`, sold as the Ky X1 in the
Orange Pi RV2): full RVV 1.0, VLEN=256, DLEN=128. Kernels are
vector-length-agnostic, so the same binary should run on any V-capable
RVV chip; tuning choices (LMUL=1 with manual unrolling) target the X60
specifically.

| Workload | funnelcake | vs libswscale |
|---|---|---|
| 1920×1080 down:1.5x,3x,6x        |  3.6 ms | 59.3× / 40.2× cascade |
| 3840×2160 down:1.5x,3x,6x,12x    | 39.1 ms | 28.1× / 15.1× cascade |
| 1920×1080 up:2x                  |  3.1 ms | 138.2× |
| 1920×1080 down:2x up:2x          |  7.4 ms | 67.9× |
| 1920×1080 down:1.5x,3x up:2x     |  8.7 ms | 67.3× |
| 1920×1080 I010 down:1.5x,3x,6x   | 12.8 ms | (no HDR comparison) |
| 1920×1080 I010 up:2x             |  7.6 ms | (no HDR comparison) |
| 1920×1080 I010 down:1.5x,3x,6x tone | 19.5 ms | (no HDR comparison) |
| 3840×2160 I010 tone 1x           | 46.8 ms | (no HDR comparison) |

HDR speedups land roughly half the SDR ratio because 10-bit u16 elements
halve the per-vector throughput on the X60's 256-bit V unit.

Numbers above are from a GCC 14 build. GCC 13 produces RVV kernels
roughly 2–4× slower because its older intrinsic spec has no segment
loads and stores; see [INSTALL.md](INSTALL.md#risc-v-rvv) for the
details and for the LTO caveat on this target.

Detection requires the V extension and a non-emulated misaligned-vector
load path (queried via `riscv_hwprobe`); chips that report SLOW or
EMULATED for `RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF`, or that
advertise only the embedded `Zve*` subset, fall back to the scalar
kernel.

### A note on the memory wall

Several of the workloads in these tables profile down to one load, one
pair-average, and one store per output byte. That is the minimum useful
work per byte, so what limits them is the memory bus, not the code.
These configurations were measured sitting at the single-core bandwidth
ceiling on the machines used during development, and getting them faster
would take a wider memory bus rather than a better kernel:

- **Straight 2× upscale at 1080p on DDR5 systems**: on a Zen 5 system
  this workload is ~15 MB of source read + output write, and funnelcake
  completes it in roughly the time it takes the memory controller to
  physically move that amount of data (~82 GB/s effective, which matches
  the single-core sustained DDR5 bandwidth of that platform).
- **Shallow pow2 downscales at 4K on Apple Silicon**: the 2×/4× levels
  of a 4K→1080p→540p ladder are dominated by memory traffic from the
  source and into the first output level; on M5 Max these run close
  to the single-core ceiling of the unified memory system.
- **Small-source workloads on CPUs with very fast memory subsystems**:
  e.g. `640×360 down:2x` on Apple Silicon completes in ~4 µs - an
  absolute time where libswscale is *also* memory-bound, so the relative
  speedup in the table (11.5×) understates how much work funnelcake is
  doing and really just reflects that both libraries are waiting on
  the same DRAM.

The workloads where funnelcake's speedup keeps growing with faster CPUs
(deep thirds cascades, the 1.5× upscale tail, combined down+up calls)
are compute-bound, and those are where lower operation counts and better
register scheduling still pay off.


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
|------------------------|--------------------------------|
| 1.5× only              | 3                              |
| 3×                     | 6                              |
| 6×                     | 12                             |
| 12×                    | 24                             |

Similarly for `src_height` (vertical period):

| Deepest step requested | src_height must be divisible by |
|------------------------|---------------------------------|
| 1.5× or 3×             | 6                               |
| 6×                     | 12                              |
| 12×                    | 24                              |

#### Pow2 family (2×, 4×, 8×, 16×)
The deepest pow2 step imposes a similar requirement:

| Deepest step requested | src_width and src_height must be divisible by |
|------------------------|-----------------------------------------------|
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
family** per init. Requesting `FUSED_SCALE_3X | FUSED_SCALE_4X` (thirds + pow2)
returns `FUSED_ERR_INVALID_FLAGS`. Use two separate contexts if you need
both downscale families.

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

**1.5x upscale performance**: the 1.5x tail is slower per output byte
than any of the 2× steps on AVX2 because the 2-to-3 output pattern has no
3-way interleaved store and must be assembled with shuffles. The
weighted 85/171 blends themselves run on raw byte pairs via
`vpmaddubsw`, so on Zen 2 / Haswell and later the kernel is roughly 2×
slower per output byte than a straight 2× step, and still several times
faster than libswscale's bilinear upscale. On Zen 1 the gap is wider
because Zen 1 double-pumps 256-bit AVX2 instructions through its
128-bit datapath. NEON does not have this bottleneck; the 2-to-3 pattern
maps cleanly onto `vld2q_u8` / `vst3q_u8`. Choose the 1.5x tail with
this in mind on compute-limited x86 targets.

### Thread safety
Each context is independent and not thread-safe. Use one context per thread.
Concurrent reads from separate contexts on the same source data are safe.

### Performance: huge-page-backed source buffers (Linux)

For workloads that are bandwidth-limited rather than compute-limited (the
straight 2× upscales on DDR5 systems and the shallow pow2 downscales on
fast-memory platforms called out in [A note on the memory wall](#a-note-on-the-memory-wall)),
callers can capture a small additional speedup on Linux by allocating the
source Y/U/V planes in huge-page-backed memory:

```c
#include <sys/mman.h>

void *plane = NULL;
posix_memalign(&plane, 32, plane_size);
if (plane_size >= 2 * 1024 * 1024) {
    madvise(plane, plane_size, MADV_HUGEPAGE);
}
```

This reduces TLB pressure across the streaming row-strided read pattern and
lets the L2 hardware prefetcher (which resets at 4 KB page boundaries on
Intel and AMD) run uninterrupted across the source plane. The library
already applies the same hint internally to its own large output planes at
init, so this extension covers only the caller-owned source planes that
the library cannot allocate. The hint is a no-op on systems with
`transparent_hugepage=never` and is unnecessary or unavailable on non-Linux
platforms.


## Getting started

See **[INSTALL.md](INSTALL.md)** for build instructions, [make
targets](INSTALL.md#make-targets), [test binary
options](INSTALL.md#the-test-binary), architecture-specific compiler and
tuning advice, PGO and LTO setup, and static-library compatibility notes.

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


## Language bindings

Idiomatic bindings for the full SDR + HDR API live under
**[bindings/](bindings/)**, each with its own README covering build and usage:

| Language | Location | Notes |
|----------|----------|-------|
| Go       | [bindings/go](bindings/go/API-GO.md)         | cgo |
| Rust     | [bindings/rust](bindings/rust/API-RUST.md)   | no `bindgen` dependency |
| Python   | [bindings/python](bindings/python/API-PYTHON.md) | stdlib `ctypes`, no third-party deps |
| Java     | [bindings/java](bindings/java/API-JAVA.md)   | Foreign Function & Memory API — requires **JDK 22+** |

They are opt-in and never built by a plain `make`; see each binding's docs (and
the `bindings-<lang>` / `test-<lang>` Makefile targets).


## Releases and packaging

Cutting a release and building the FreeBSD port are covered in
[INSTALL.md](INSTALL.md#releases-and-packaging).


## Platform support

| Platform | SIMD | Notes |
|----------|------|-------|
| x86-64 with AVX-512 F+BW+VL+VBMI (Zen 4+, Ice Lake+) | AVX-512 | Detected at runtime via cpuid + xgetbv; builds with an older compiler top out at AVX2 |
| x86-64 with AVX2 (Linux, macOS, FreeBSD) | AVX2 | Detected at runtime via cpuid; also used on CPUs whose AVX-512 lacks VBMI (e.g. Skylake-SP, where 512-bit downclocking favors AVX2 anyway) |
| x86-64 without AVX2 | Scalar | Broadwell and later all have AVX2 |
| aarch64 (Apple Silicon, AWS Graviton, FreeBSD/arm64) | NEON | All aarch64 cores have NEON |
| riscv64 with RVV 1.0 (Linux) | RVV | Detected via `riscv_hwprobe`; requires the full V extension and non-emulated misaligned-vector loads |
| Other | Scalar | Portable C, no intrinsics |

The scalar fallback is correct on all platforms but significantly slower.
On hardware without AVX2, NEON, or RVV, the library logs a one-time notice
to stderr at first init.

Call `fused_simd_available()` to query this at runtime: it returns `1` when
the SIMD kernels will be used and `0` when the scalars will. It uses the same
CPU probe the scalers do (and honors `FUNNELCAKE_FORCE_SCALAR`), so callers
and test harnesses can tell *expected* whole-CPU scalar fallback apart from a
real failure. When it returns `0`, a clean init reports `FUSED_WARN_BIT_SCALAR`
rather than `FUSED_OK`.


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
| `FUSED_TONEMAP_HABLE` | Hable/Uncharted 2 filmic (default). Most highlight detail; filmic midtone dimming (~-1 stop) |
| `FUSED_TONEMAP_REINHARD` | Extended Reinhard with white point at `peak_nits`. Soft, lower contrast |
| `FUSED_TONEMAP_BT2390` | ITU-R BT.2390 EETF in PQ space (broadcast reference). Midtones pass through at correct brightness |
| `FUSED_TONEMAP_CUSTOM` | Caller-supplied 1024-entry Y LUT |

All built-in curves compress `[0, peak_nits]` smoothly onto the SDR range -
nothing below the source peak hard-clips. Chroma is reconstructed with the
exact BT.2020 non-constant-luminance inverse in the gamma domain, gamut-
converted from BT.2020 to BT.709 primaries, and re-encoded as BT.709 YCbCr.

Input and output quantization ranges are configurable via
`tonemap.src_range` / `tonemap.dst_range` (`FUSED_RANGE_LIMITED` or
`FUSED_RANGE_FULL`). The default is limited (video) range on both sides,
matching real HDR10/HLG streams; set `FUSED_RANGE_FULL` on `dst_range` if
the consumer expects PC-range 8-bit output.

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

## License

Copyright (c) 2020-2026 Kevin Day. Licensed under the BSD-2-Clause-Patent license — see [LICENSE.md](LICENSE.md) for the full text.

The core kernels were based off my hand-written assembly that were converted
to C intrinsics for easier portability and readability. AI was not used for
the core functionality, kernels or algorithms. I did use AI agents for
documentation, improving my terrible comments, fixing the build system,
and writing test cases.
