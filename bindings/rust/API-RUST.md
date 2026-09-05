# funnelcake — Rust binding

Idiomatic, thin Rust bindings over the funnelcake SIMD YUV scaler and HDR
tone-mapper. No `bindgen`/libclang — the FFI structs are `#[repr(C)]`, so the
compiler lays them out per the platform C ABI to match `funnelcake.h`
automatically. [`Frame`]/[`HdrFrame`] allocate input planes with the 32-byte
alignment the SIMD kernels need (via `std::alloc`), the scaler types wrap the
create/run/free lifecycle with `Drop`, and output views borrow the scaler so
the "valid until next run" rule is **enforced by the borrow checker**.

## Requirements

- A stable Rust toolchain (`cargo`/`rustc`). Tested on 1.96.
- The core static library, built by `make lib`.

## Building

```sh
make lib            # builds libfunnelcake.a at the repo root
make bindings-rust  # cargo build
make test-rust      # cargo test
```

`make bindings-rust`/`make test-rust` are opt-in — a plain `make` never touches
Rust. `build.rs` statically links the in-tree `libfunnelcake.a` (which contains
both the scaler API and the binding helpers); there are **no crates.io
dependencies**, so the build works offline.

> The trial links `../../libfunnelcake.a` by path from the crate. Publishing a
> standalone crate (with the native library bundled or discovered) is future
> work.

## Alignment, in one sentence

The SIMD kernels require 32-byte-aligned planes and 32-byte-aligned strides.
`Frame::new(w, h)` / `HdrFrame::new(w, h, fmt)` allocate exactly that, and the
scaler constructors derive matching source strides from the width — so a frame
and a scaler created with the same dimensions always line up. If a step ever
runs unaligned it silently drops to the scalar path (visible as
`output.fallback == true`).

## SDR: downscale 2x

```rust
use funnelcake::{Frame, Scaler, ScalerConfig, SCALE_2X};

let mut frame = Frame::new(1920, 1080);     // aligned I420 input
frame.y_mut().copy_from_slice(&src_y);
frame.u_mut().copy_from_slice(&src_u);
frame.v_mut().copy_from_slice(&src_v);

let mut scaler = Scaler::new(&ScalerConfig::new(1920, 1080, SCALE_2X))?;
scaler.run(&frame);

if let Some(out) = scaler.output(SCALE_2X) {  // 960x540
    let _y: &[u8] = out.y();                  // borrows `scaler`; valid until next run/drop
    use_planes(out.y(), out.u(), out.v(), out.y_stride());
}
# Ok::<(), funnelcake::Error>(())
```

A single scaler can request several steps from **one** family
(`SCALE_2X | SCALE_4X | ...` or `SCALE_1_5X | SCALE_3X | ...`) and an upscale
cascade (`upscale_flags`, a contiguous prefix such as `UPSCALE_2X | UPSCALE_4X`,
optionally with `upscale_tail_1_5x`). All requested outputs are produced in one
`run`.

Because `Output` borrows `&self` and `run` takes `&mut self`, this fails to
compile — you cannot hold a view across a re-run:

```rust,compile_fail
let out = scaler.output(SCALE_2X).unwrap();
scaler.run(&frame);   // error: cannot borrow `scaler` as mutable
let _ = out.y();      // ...while `out` still borrows it
```

## HDR: 10-bit scale + tone-map to SDR

```rust
use funnelcake::*;

let mut frame = HdrFrame::new(3840, 2160, PixelFormat::I010);
// fill frame.y_mut(), frame.u_mut(), frame.v_mut() (&mut [u16]) ...

let cfg = HdrConfig {
    src_width: 3840,
    src_height: 2160,
    format: PixelFormat::I010,
    transfer: Transfer::Pq,
    flags: SCALE_2X,
    hdr_flags: SCALE_2X,                 // produce a 10-bit downscaled copy
    sdr_flags: SCALE_2X,                 // and a tone-mapped 8-bit copy
    tonemap_1x: true,                    // plus a 1:1 tone-mapped SDR copy
    tonemap: TonemapConfig { curve: TonemapCurve::Bt2390, ..Default::default() },
    ..Default::default()
};
let mut scaler = HdrScaler::new(&cfg)?;
scaler.run(&frame);

let hdr = scaler.hdr_output(SCALE_2X);        // Option<HdrOutput> (&[u16] planes)
let sdr = scaler.sdr_output(SCALE_2X);        // Option<Output>    (tone-mapped &[u8])
let one = scaler.tonemap_1x_output();         // Option<Output>    (8-bit, source res)
# Ok::<(), funnelcake::Error>(())
```

For semi-planar input (`PixelFormat::P010`/`P210`), fill `u_mut()` with the
interleaved UV plane; `v_mut()` returns `None`. 4:2:2 formats (`I210`/`P210`)
are accepted and decimated to 4:2:0 internally. A default `TonemapConfig` is the
Hable curve, 1000-nit peak, 100-nit target, and limited range — matching the C
defaults. Set `curve: TonemapCurve::Custom` with a 1024-byte `custom_lut` to
supply your own.

## API reference

All items are at the crate root. Strides are in **bytes** (even for 10-bit
planes, where a sample is 2 bytes).

### Free functions / flags

```rust
pub fn simd_available() -> bool;
```
True if the vectorized kernels will run here; when false, every `fallback` is
true and a scalar warning is expected.

Bitmask `u32` consts (combine with `|`):
- Downscale: `SCALE_1_5X SCALE_3X SCALE_6X SCALE_12X` (thirds),
  `SCALE_2X SCALE_4X SCALE_8X SCALE_16X` (pow2), `SCALE_THIRDS_MASK`,
  `SCALE_POW2_MASK`. **One family per scaler.**
- Upscale: `UPSCALE_2X UPSCALE_4X UPSCALE_8X UPSCALE_16X UPSCALE_32X`
  (contiguous prefix only).
- Options: `OPT_NO_CROP`, `OPT_NO_FALLBACK`.

Enums (all `#[repr(i32)]`, `Default`):
`PixelFormat { I010, P010, I210, P210 }`, `Transfer { Pq, Hlg }`,
`Range { Limited, Full }`, `TonemapCurve { Hable, Reinhard, Bt2390, Custom }`.

### Frame — 8-bit I420 input

```rust
pub fn Frame::new(width: i32, height: i32) -> Frame;   // panics if <= 0
pub fn width(&self) -> i32; pub fn height(&self) -> i32;
pub fn y_stride(&self) -> i32; pub fn uv_stride(&self) -> i32;
pub fn y_mut(&mut self) -> &mut [u8];  // fill these before run
pub fn u_mut(&mut self) -> &mut [u8];
pub fn v_mut(&mut self) -> &mut [u8];
// dropping the Frame frees the planes
```

### HdrFrame — 10-bit input

```rust
pub fn HdrFrame::new(width: i32, height: i32, format: PixelFormat) -> HdrFrame;
pub fn width/height/format/y_stride/uv_stride(&self) -> ...;
pub fn y_mut(&mut self) -> &mut [u16];
pub fn u_mut(&mut self) -> &mut [u16];          // interleaved UV for P010/P210
pub fn v_mut(&mut self) -> Option<&mut [u16]>;  // None for P010/P210
```

### Scaler — 8-bit

```rust
pub struct ScalerConfig {           // derives Default
    pub src_width: i32, pub src_height: i32,
    pub flags: u32, pub upscale_flags: u32,
    pub upscale_tail_1_5x: bool, pub options: u32,
}
pub fn ScalerConfig::new(src_width: i32, src_height: i32, flags: u32) -> ScalerConfig;

pub fn Scaler::new(cfg: &ScalerConfig) -> Result<Scaler, Error>;
pub fn warnings(&self) -> Warnings;
pub fn run(&mut self, frame: &Frame);
pub fn effective_width(&self) -> i32; pub fn effective_height(&self) -> i32;
pub fn achieved_flags(&self) -> u32;
pub fn output(&self, flag: u32) -> Option<Output<'_>>;
pub fn upscale_output(&self, flag: u32) -> Option<Output<'_>>;
pub fn upscale_tail(&self) -> Option<Output<'_>>;
// dropping the Scaler frees its output buffers
```

### HdrScaler — 10-bit

```rust
pub struct TonemapConfig {          // derives Default (Hable, default nits, limited)
    pub curve: TonemapCurve, pub peak_nits: i32, pub target_nits: i32,
    pub src_range: Range, pub dst_range: Range, pub custom_lut: Option<Vec<u8>>,
}
pub struct HdrConfig {              // derives Default
    pub src_width: i32, pub src_height: i32,
    pub format: PixelFormat, pub transfer: Transfer,
    pub flags: u32, pub hdr_flags: u32, pub sdr_flags: u32, pub options: u32,
    pub tonemap_1x: bool, pub tonemap: TonemapConfig,
    pub upscale_flags: u32, pub upscale_tail_1_5x: bool,
    pub upscale_sdr_flags: u32, pub upscale_sdr_tail_1_5x: bool,
}

pub fn HdrScaler::new(cfg: &HdrConfig) -> Result<HdrScaler, Error>;
pub fn warnings(&self) -> Warnings;
pub fn run(&mut self, frame: &HdrFrame);
pub fn effective_width/height(&self) -> i32;
pub fn hdr_output(&self, flag: u32) -> Option<HdrOutput<'_>>;   // 10-bit
pub fn sdr_output(&self, flag: u32) -> Option<Output<'_>>;      // tone-mapped 8-bit
pub fn tonemap_1x_output(&self) -> Option<Output<'_>>;          // 8-bit, source res
pub fn upscale_hdr_output(&self, flag: u32) -> Option<HdrOutput<'_>>;
pub fn upscale_sdr_output(&self, flag: u32) -> Option<Output<'_>>;
```

### Output / HdrOutput — result views

```rust
pub struct Output<'a> {     // 8-bit
    // dimensions and strides are private; use width(), height(),
    // y_stride(), and uv_stride() getters.
    pub fallback: bool,     // scalar kernel was used for this step
}
pub fn Output::y(&self) -> &'a [u8];  // also u(), v()

pub struct HdrOutput<'a> {  // 10-bit; samples are u16 (little-endian)
    // dimensions and strides are private; use width(), height(),
    // y_stride(), and uv_stride() getters.
    pub fallback: bool,
}
pub fn HdrOutput::y(&self) -> &'a [u16];  // also u(), v()
```
The `'a` lifetime ties each view to the producing scaler: the slices cannot
outlive it, and you cannot call `run` (needs `&mut`) while a view is held.

### Errors and warnings

```rust
pub enum Error { InvalidFlags, NoSteps, BadDimensions, BadAlignment, Unknown(i32) }
// implements std::error::Error + Display

pub struct Warnings(pub u32);
pub fn Warnings::scalar(&self) -> bool;   // a step fell back to scalar
pub fn Warnings::partial(&self) -> bool;  // a requested step was rejected
pub fn Warnings::cropped(&self) -> bool;  // source was cropped
pub fn Warnings::perfect(&self) -> bool;  // none of the above
```
`Error` is returned by the constructors for hard failures; `Warnings` (on
`scaler.warnings()`) reports non-fatal conditions and is never an error.

## Safety notes

- **Frame must match the scaler.** `run` panics if the frame's dimensions (and,
  for HDR, the format) differ from the scaler's configuration, preventing an
  out-of-bounds read in the native kernels.
- **Custom LUT size.** A `TonemapConfig.custom_lut` must be exactly 1024 bytes;
  otherwise `HdrScaler::new` returns `Err(Error::CustomLutLength)`.
- **Output lifetimes are compile-time enforced.** Because each view borrows
  `&scaler`, the borrow checker already prevents use-after-free and prevents a
  `run` while a view is held. To keep data past the scaler, copy it
  (`out.y().to_vec()`).

## Concurrency

Each `Scaler`/`HdrScaler` (and `Frame`/`HdrFrame`) owns raw pointers into
library-managed memory, so the types are **not** `Send`/`Sync` and the compiler
will stop you from moving or sharing one across threads. Per-context work is
independent, so the intended pattern is to construct and use a scaler on the
thread that needs it. The binding forces the library's one-time CPU probe on
first use, so concurrent first constructions on different threads are safe.

### Active rows and owned copies

Output and HdrOutput provide y_row(row), u_row(row), and v_row(row) slices
containing only active samples. to_owned() returns an OwnedOutput with tightly
packed vectors that survive the scaler and future runs.

### Runtime diagnostics

`version()` and `backend()` report the linked native library version and
preferred backend. Individual outputs can still use scalar fallback.
