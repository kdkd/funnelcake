# funnelcake — Python binding

Idiomatic, thin Python bindings over the funnelcake SIMD YUV scaler and HDR
tone-mapper, using **stdlib `ctypes`** — no third-party dependencies, no build
step for the Python side. `ctypes.Structure` lays the context structs out per
the C ABI automatically, [`Frame`]/[`HdrFrame`] allocate input planes with the
32-byte alignment the SIMD kernels need, and outputs are exposed as zero-copy
`memoryview`s over library-owned buffers.

## Requirements

- Python 3.8+ (tested on 3.14). No pip packages required.
- The funnelcake shared libraries, built by `make bindings-python`.

## Building / running

```sh
make lib              # builds libfunnelcake.a (and is a prerequisite object set)
make bindings-python  # builds the shared libs into bindings/python/
make test-python      # runs the unittest smoke tests
```

`make bindings-python` produces `libfunnelcake.<dylib|so>` (the scaler API plus
the binding helpers) in `bindings/python/`. Both targets are opt-in; a plain
`make` never touches Python.

To use the package from your own code, put `bindings/python` on `PYTHONPATH`,
and (optionally) point `FUNNELCAKE_LIBDIR` at the directory holding the shared
library. It defaults to the directory containing the package, which is where
the Makefile puts them:

```sh
PYTHONPATH=/path/to/bindings/python \
FUNNELCAKE_LIBDIR=/path/to/bindings/python \
python3 -c "import funnelcake; print(funnelcake.simd_available())"
```

> The trial loads the libraries by directory. Packaging a `pip`-installable
> wheel with the natives bundled is future work.

## Alignment, in one sentence

The SIMD kernels require 32-byte-aligned planes and 32-byte-aligned strides.
`Frame(w, h)` / `HdrFrame(w, h, fmt)` allocate exactly that (via the C
`fused_aligned_alloc` helper), and the scalers derive matching source strides
from the width — so a frame and a scaler of the same size always line up. If a
step ever runs unaligned it silently drops to the scalar path (visible as
`output.fallback == True`).

## SDR: downscale 2x

```python
import funnelcake as fc

with fc.Frame(1920, 1080) as f:                 # aligned I420 input
    f.y()[:] = src_y                            # memoryviews (format 'B') into plane memory
    f.u()[:] = src_u
    f.v()[:] = src_v

    with fc.Scaler(fc.ScalerConfig(1920, 1080, fc.Scale.X2)) as s:
        s.run(f)
        out = s.output(fc.Scale.X2)             # 960x540, or None
        if out is not None:
            use(out.y, out.y_stride, out.width, out.height)  # out.y is a memoryview
```

A single scaler can request several steps from **one** family
(`Scale.X2 | Scale.X4 | ...` or `Scale.X1_5 | Scale.X3 | ...`) and an upscale
cascade (`upscale_flags`, a contiguous prefix such as `Upscale.X2 | Upscale.X4`,
optionally with `upscale_tail_1_5x`). All requested outputs are produced in one
`run`.

## HDR: 10-bit scale + tone-map to SDR

```python
import funnelcake as fc

with fc.HdrFrame(3840, 2160, fc.PixelFormat.I010) as f:
    f.y()[:] = src_y16          # memoryviews of uint16 samples (format 'H')
    f.u()[:] = src_u16
    f.v()[:] = src_v16

    cfg = fc.HdrConfig(
        src_width=3840, src_height=2160,
        format=fc.PixelFormat.I010, transfer=fc.Transfer.PQ,
        flags=fc.Scale.X2,
        hdr_flags=fc.Scale.X2,      # produce a 10-bit downscaled copy
        sdr_flags=fc.Scale.X2,      # and a tone-mapped 8-bit copy
        tonemap_1x=True,            # plus a 1:1 tone-mapped SDR copy
        tonemap=fc.TonemapConfig(curve=fc.TonemapCurve.BT2390),
    )
    with fc.HdrScaler(cfg) as s:
        s.run(f)
        hdr = s.hdr_output(fc.Scale.X2)     # HdrOutput (uint16 memoryviews) or None
        sdr = s.sdr_output(fc.Scale.X2)     # Output (tone-mapped 8-bit) or None
        one = s.tonemap_1x_output()         # Output (8-bit, source resolution) or None
```

For semi-planar input (`PixelFormat.P010`/`P210`), fill `f.u()` with the
interleaved UV plane; `f.v()` returns `None`. 4:2:2 formats (`I210`/`P210`) are
accepted and decimated to 4:2:0 internally. A default `TonemapConfig` is the
Hable curve, 1000-nit peak, 100-nit target, and limited range. Set
`curve=TonemapCurve.CUSTOM` with a 1024-byte `custom_lut` to supply your own.

### numpy interop (optional)

Plane views are plain `memoryview`s, so numpy can wrap them without a copy:

```python
import numpy as np
y = np.frombuffer(out.y, dtype=np.uint8).reshape(out.height, out.y_stride)
y16 = np.frombuffer(hdr.y, dtype=np.uint16).reshape(hdr.height, hdr.y_stride // 2)
```

## API reference

Everything is importable from the top-level `funnelcake` package. Strides are in
**bytes**. Plane views are `memoryview`s: format `'B'` (8-bit) or `'H'` (16-bit).

### Module

```python
funnelcake.simd_available() -> bool
```
True if the vectorized kernels will run here; when False, every `fallback` is
True and a scalar warning is expected.

### Enums

`Scale` and `Upscale` and `Option` are `IntFlag` (combine with `|`):
- `Scale.X1_5 X3 X6 X12` (thirds), `Scale.X2 X4 X8 X16` (pow2),
  `Scale.THIRDS_MASK`, `Scale.POW2_MASK`. **One family per scaler.**
- `Upscale.X2 X4 X8 X16 X32` (request a contiguous prefix only).
- `Option.NO_CROP`, `Option.NO_FALLBACK`.

`IntEnum` parameters: `PixelFormat.{I010,P010,I210,P210}`,
`Transfer.{PQ,HLG}`, `Range.{LIMITED,FULL}`,
`TonemapCurve.{HABLE,REINHARD,BT2390,CUSTOM}`.

### Frame / HdrFrame

```python
Frame(width, height)                      # raises ValueError if <= 0
  .width .height .y_stride .uv_stride      # ints
  .y() .u() .v() -> memoryview             # writable, format 'B'
  .close();  also a context manager

HdrFrame(width, height, format)
  .width .height .format .y_stride .uv_stride
  .y() .u() -> memoryview                  # format 'H' (uint16)
  .v() -> memoryview | None                # None for P010/P210
  .close();  also a context manager
```

### Scaler

```python
@dataclass ScalerConfig(src_width, src_height, flags=0,
                        upscale_flags=0, upscale_tail_1_5x=False, options=0)

Scaler(config: ScalerConfig)               # raises FunnelcakeError on hard error
  .warnings -> Warnings
  .run(frame: Frame) -> None
  .effective_width / .effective_height     # properties
  .achieved_flags                          # property
  .output(flag) -> Output | None
  .upscale_output(flag) -> Output | None
  .upscale_tail() -> Output | None
  .close();  also a context manager
```

### HdrScaler

```python
@dataclass TonemapConfig(curve=TonemapCurve.HABLE, peak_nits=0, target_nits=0,
                         src_range=Range.LIMITED, dst_range=Range.LIMITED,
                         custom_lut=None)

@dataclass HdrConfig(src_width, src_height, format=PixelFormat.I010,
                     transfer=Transfer.PQ, flags=0, hdr_flags=0, sdr_flags=0,
                     options=0, tonemap_1x=False, tonemap=TonemapConfig(),
                     upscale_flags=0, upscale_tail_1_5x=False,
                     upscale_sdr_flags=0, upscale_sdr_tail_1_5x=False)

HdrScaler(config: HdrConfig)               # raises FunnelcakeError on hard error
  .warnings -> Warnings
  .run(frame: HdrFrame) -> None
  .effective_width / .effective_height
  .hdr_output(flag) -> HdrOutput | None    # 10-bit
  .sdr_output(flag) -> Output | None       # tone-mapped 8-bit
  .tonemap_1x_output() -> Output | None     # 8-bit, source resolution
  .upscale_hdr_output(flag) -> HdrOutput | None
  .upscale_sdr_output(flag) -> Output | None
  .close();  also a context manager
```

### Output / HdrOutput (dataclasses)

```python
Output(width, height, y_stride, uv_stride, fallback, y, u, v)      # y/u/v: memoryview 'B'
HdrOutput(width, height, y_stride, uv_stride, fallback, y, u, v)   # y/u/v: memoryview 'H'
```
`fallback` is True if the scalar kernel was used. The plane memoryviews alias
scaler-owned memory and are valid only until the producing scaler's next `run`
or `close`; copy out what you need to keep.

### Errors and warnings

```python
class FunnelcakeError(Exception):  .code   # negative FUSED_ERR_* code
class Warnings:  .scalar() .partial() .cropped() .perfect()
```
`FunnelcakeError` is raised by the constructors for hard failures (`-1` invalid
flags, `-2` no steps, `-3` bad dimensions, `-4` bad alignment). `Warnings` (on
`scaler.warnings`) reports non-fatal conditions and is never raised.

## Safety notes

- **Frame must match the scaler.** `run()` raises `ValueError` if the frame's
  dimensions (and, for HDR, the format) differ from the scaler's configuration,
  preventing an out-of-bounds read in the native kernels.
- **Custom LUT size.** A `TonemapConfig.custom_lut` must be exactly 1024 bytes;
  otherwise the `HdrScaler` constructor raises `ValueError`.
- **Don't outlive the scaler with a raw memoryview.** Output memoryviews alias
  scaler-owned memory. While you hold the `Output` the scaler is kept alive, but
  a memoryview extracted and kept after the `Output` and scaler are gone
  dangles. Copy what you need to keep: `y = bytes(out.y)` (or
  `np.array(out.y)`).

## Concurrency

Each `Scaler`/`HdrScaler` is independent; use one per thread or guard with your
own lock — do not share a scaler across threads without synchronization. The
binding forces the library's one-time CPU probe at import, so concurrent first
use is safe.

### Plane ownership

Frame dimensions, format, and strides are read-only. Exported frame memoryviews
retain their native allocation, including sliced views. Output views are
read-only and also retain their allocation independently of the scaler wrapper.
Closing a frame or scaler prevents further use of that wrapper; existing views
remain valid until released. A subsequent run on an open scaler still overwrites
its borrowed output views.
