# funnelcake — Java binding

Idiomatic, thin Java bindings over the funnelcake SIMD YUV scaler and HDR
tone-mapper, built on the **Foreign Function & Memory API** (Project Panama).
No JNI, no C shim, and no generated code — just the JDK. The binding wraps the
C library's create/run/free lifecycle and hands you `Frame` / `HdrFrame` types
whose native buffers are **already aligned** the way the SIMD kernels need, so
you never compute a stride or call an aligned allocator yourself.

## Requirements

- **JDK 22 or newer** (FFM is stable since 22). Tested on JDK 25.
- The funnelcake shared libraries, built by `make bindings-java`.

## Building

```sh
make bindings-java   # builds the shared libs + compiles the classes
make test-java       # runs the smoke test
```

`make bindings-java` produces, under `bindings/java/`:

- `libfunnelcake.<dylib|so>` — the shared library (scaler API plus the binding
  helpers, including the `ctx_sizeof` layout guards)
- `classes/` — the compiled package

Both targets are opt-in — a plain `make` never touches Java. If your default
`java`/`javac` is older than 22, point the build at a newer JDK:

```sh
make test-java JAVA=/opt/homebrew/opt/openjdk@25/bin/java \
               JAVAC=/opt/homebrew/opt/openjdk@25/bin/javac
```

### Running your own code

The binding loads `libfunnelcake` from the directory named by the
`funnelcake.libdir` system property (default `"."`). FFM also requires native
access to be enabled. A typical launch:

```sh
java --enable-native-access=ALL-UNNAMED \
     -Dfunnelcake.libdir=/path/to/bindings/java \
     -cp /path/to/bindings/java/classes:your-app.jar  com.example.Main
```

> The trial loads libraries by directory and is driven from the build tree.
> Packaging the natives into a JAR (and `System.loadLibrary` discovery) is
> future work.

## Alignment, in one sentence

The SIMD kernels require 32-byte-aligned planes and 32-byte-aligned strides.
`new Frame(w, h)` / `new HdrFrame(w, h, format)` allocate exactly that from a
dedicated `Arena` (`Arena.allocate(size, 32)`), and the scaler constructors
derive matching source strides from the width — so a frame and a scaler created
with the same dimensions always line up. If you ever pass hand-rolled buffers,
misalignment silently drops the step to the scalar path (visible as
`Output.fallback() == true`).

## SDR: downscale 2x

```java
import org.your.funnelcake.*;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;

try (Frame f = new Frame(1920, 1080);                       // aligned I420 input
     Scaler s = new Scaler(new Scaler.Config(1920, 1080, Funnelcake.SCALE_2X))) {

    // Fill the planes (these are native views into the frame's memory).
    MemorySegment.copy(srcY, 0, f.y(), JAVA_BYTE, 0, srcY.length);
    MemorySegment.copy(srcU, 0, f.u(), JAVA_BYTE, 0, srcU.length);
    MemorySegment.copy(srcV, 0, f.v(), JAVA_BYTE, 0, srcV.length);

    s.run(f);

    s.output(Funnelcake.SCALE_2X).ifPresent(o -> {          // 960x540
        // o.y(), o.u(), o.v() are native segments valid until the next run/close
        use(o.y(), o.yStride(), o.width(), o.height());
    });
}
```

A single scaler can request several steps from **one** family
(`SCALE_2X | SCALE_4X | ...` or `SCALE_1_5X | SCALE_3X | ...`) and an upscale
cascade (`upscaleFlags`, a contiguous prefix such as `UPSCALE_2X | UPSCALE_4X`,
optionally with `upscaleTail15`). All requested outputs are produced in a
single `run`.

## HDR: 10-bit scale + tone-map to SDR

```java
HdrScaler.Config cfg = new HdrScaler.Config(
        3840, 2160, Funnelcake.PIX_I010, Funnelcake.TRC_PQ,
        Funnelcake.SCALE_2X,                 // requested steps
        Funnelcake.SCALE_2X,                 // produce 10-bit HDR copy
        Funnelcake.SCALE_2X,                 // produce tone-mapped 8-bit copy
        0,                                   // options
        true,                                // tonemap1x: 1:1 SDR copy
        HdrScaler.Tonemap.of(Funnelcake.TONEMAP_BT2390),
        0, false, 0, false);                 // upscale (unused here)

try (HdrFrame f = new HdrFrame(3840, 2160, Funnelcake.PIX_I010);
     HdrScaler s = new HdrScaler(cfg)) {

    // fill f.y(), f.u(), f.v() with 16-bit samples ...
    s.run(f);

    s.hdrOutput(Funnelcake.SCALE_2X).ifPresent(o -> { /* 10-bit planes */ });
    s.sdrOutput(Funnelcake.SCALE_2X).ifPresent(o -> { /* tone-mapped 8-bit */ });
    s.tonemap1xOutput().ifPresent(o -> { /* 8-bit, source resolution */ });
}
```

For semi-planar input (`PIX_P010` / `PIX_P210`), fill `f.u()` with the
interleaved UV plane; `f.v()` is `MemorySegment.NULL`. 4:2:2 formats
(`PIX_I210` / `PIX_P210`) are accepted and decimated to 4:2:0 internally.

`HdrScaler.Tonemap.defaults()` is the Hable curve, 1000-nit peak, 100-nit
target, and limited range — the same defaults as the C library. Use
`TONEMAP_CUSTOM` with a 1024-entry `customLut` byte array to supply your own.

## API reference

Everything is in package `org.your.funnelcake`. Strides are always in **bytes**
(even for 10-bit planes, where a sample is 2 bytes). Plane data is exposed as
`java.lang.foreign.MemorySegment`.

### Funnelcake (constants + capability)

```java
public static boolean Funnelcake.simdAvailable()
```
True if the vectorized kernels will run on this machine. When false, every
output's `fallback()` is true and a scalar warning is expected.

Bitmask constants (combine with `|`):
- Downscale: `SCALE_1_5X SCALE_3X SCALE_6X SCALE_12X` (thirds),
  `SCALE_2X SCALE_4X SCALE_8X SCALE_16X` (pow2), plus `SCALE_THIRDS_MASK`,
  `SCALE_POW2_MASK`. **One family per scaler.**
- Upscale: `UPSCALE_2X UPSCALE_4X UPSCALE_8X UPSCALE_16X UPSCALE_32X`
  (request a contiguous prefix only).
- Options: `OPT_NO_CROP`, `OPT_NO_FALLBACK`.

Enumerated-int constants:
- Pixel format: `PIX_I010 PIX_P010 PIX_I210 PIX_P210`.
- Transfer: `TRC_PQ TRC_HLG`.
- Range: `RANGE_LIMITED RANGE_FULL`.
- Tone-map curve: `TONEMAP_HABLE TONEMAP_REINHARD TONEMAP_BT2390 TONEMAP_CUSTOM`.

### Frame — 8-bit I420 input

```java
public Frame(int width, int height)          // even dims; throws IllegalArgumentException if <= 0
public int  width(), height(), yStride(), uvStride()
public MemorySegment y(), u(), v()           // writable; y = yStride*height bytes, u/v = uvStride*chromaH
public void close()                          // AutoCloseable; frees the native arena
```

### HdrFrame — 10-bit input

```java
public HdrFrame(int width, int height, int format)   // format = PIX_*; throws on bad args
public int  width(), height(), format(), yStride(), uvStride()
public MemorySegment y(), u(), v()                   // 16-bit samples; v() == NULL for P010/P210
public void close()
```

### Scaler — 8-bit

```java
public record Scaler.Config(int srcWidth, int srcHeight, int flags,
                            int upscaleFlags, boolean upscaleTail15, int options)
    Scaler.Config(int srcWidth, int srcHeight, int flags)   // convenience: plain downscale

public Scaler(Scaler.Config cfg)             // throws FunnelcakeException on a hard error
public Warnings warnings()
public void     run(Frame f)
public int      effectiveWidth(), effectiveHeight()
public int      achievedFlags()
public Optional<Output> output(int scaleFlag)
public Optional<Output> upscaleOutput(int upscaleFlag)
public Optional<Output> upscaleTail()
public void     close()                      // AutoCloseable; frees outputs
```

### HdrScaler — 10-bit

```java
public record HdrScaler.Tonemap(int curve, int peakNits, int targetNits,
                                int srcRange, int dstRange, byte[] customLut)
    static Tonemap defaults()                // Hable, default nits, limited range
    static Tonemap of(int curve)             // a built-in curve, default nits/range

public record HdrScaler.Config(int srcWidth, int srcHeight, int format, int transfer,
                               int flags, int hdrFlags, int sdrFlags, int options,
                               boolean tonemap1x, HdrScaler.Tonemap tonemap,
                               int upscaleFlags, boolean upscaleTail15,
                               int upscaleSdrFlags, boolean upscaleSdrTail15)
    HdrScaler.Config(int srcWidth, int srcHeight, int format, int transfer,
                     int flags, int hdrFlags, int sdrFlags)   // minimal: default tone map

public HdrScaler(HdrScaler.Config cfg)       // throws FunnelcakeException on a hard error
public Warnings warnings()
public void     run(HdrFrame f)
public int      effectiveWidth(), effectiveHeight()
public Optional<HdrOutput> hdrOutput(int scaleFlag)        // 10-bit
public Optional<Output>    sdrOutput(int scaleFlag)        // tone-mapped 8-bit
public Optional<Output>    tonemap1xOutput()               // 8-bit, source resolution
public Optional<HdrOutput> upscaleHdrOutput(int upscaleFlag)
public Optional<Output>    upscaleSdrOutput(int upscaleFlag)
public void     close()
```

### Output / HdrOutput — result views (records)

```java
public record Output(int width, int height, int yStride, int uvStride,
                     boolean fallback, MemorySegment y, MemorySegment u, MemorySegment v)

public record HdrOutput(int width, int height, int yStride, int uvStride,
                        boolean fallback, MemorySegment y, MemorySegment u, MemorySegment v)
```
`Output` planes are 8-bit; `HdrOutput` planes are 16-bit (little-endian). The
segments alias scaler-owned memory (see lifetime note below). `fallback()` is
true if the scalar kernel was used for that step.

### Errors and warnings

```java
public final class FunnelcakeException extends RuntimeException
    public int code()                         // negative FUSED_ERR_* code

public record Warnings(int bits)
    public boolean scalar(), partial(), cropped(), perfect()
```
`FunnelcakeException` is thrown by the constructors for hard failures:
`-1` invalid flags, `-2` no steps, `-3` bad dimensions, `-4` bad alignment.
`Warnings` (on `scaler.warnings()`) reports non-fatal conditions and is never
thrown.

## Output buffer lifetime

Output plane segments (`Output.y()`, `HdrOutput.y()`, …) **alias memory owned by
the scaler**. They are valid only until the scaler's next `run` or `close`. Copy
anything you need to retain (e.g. `MemorySegment.toArray`/`copy`). Input frames
are independent — you own them and free them with `close()` (or
try-with-resources).

## Safety notes

- **Frame must match the scaler.** `run` throws `IllegalArgumentException` if the
  frame's dimensions (and, for HDR, the format) differ from the scaler's
  configuration, preventing an out-of-bounds read in the native kernels.
- **Custom LUT size.** A `Tonemap.customLut` must be exactly 1024 bytes;
  otherwise the constructor throws `IllegalArgumentException`.
- **Output segments are scoped to the scaler.** They are tied to the scaler's
  native arena, so accessing one after `close()` (or after the scaler is
  garbage-collected) throws `IllegalStateException` rather than reading freed
  memory. A `Cleaner` frees the native memory if you forget `close()`, but prefer
  try-with-resources. Copy data out (`seg.toArray(JAVA_BYTE)`) to keep it past
  the scaler's lifetime.

## Concurrency

Each `Scaler` / `HdrScaler` is independent and may run on its own thread. Do not
share one scaler across threads without your own synchronization. The binding
forces the library's one-time CPU probe during class initialization, so
concurrent first uses are safe.

### Active rows and owned copies

Output and HdrOutput provide yRow(row), uRow(row), and vRow(row) read-only
segments containing only active samples. copy() returns independent, tightly
packed, read-only planes backed by Java arrays.

### Runtime diagnostics

`Funnelcake.version()` and `Funnelcake.backend()` report the loaded native
library version and preferred backend. Individual outputs can still use scalar
fallback.
