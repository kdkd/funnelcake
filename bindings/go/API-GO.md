# funnelcake — Go binding

Idiomatic, thin Go bindings over the funnelcake SIMD YUV scaler and HDR
tone-mapper. The binding wraps the C library's create/run/free lifecycle and
hands you `Frame` / `HDRFrame` types whose buffers are **already aligned** the
way the SIMD kernels need — you never compute a stride or call an aligned
allocator yourself.

## Building

The binding links the in-tree static library, so build that first:

```sh
make lib            # builds libfunnelcake.a at the repo root
make bindings-go    # cgo-builds the Go package
make test-go        # go vet + go test
```

`make bindings-go`/`make test-go` are opt-in — a plain `make` never touches Go.
You need a Go toolchain (>= 1.23) with cgo enabled (the default). To use the
package from your own code while developing, point at this module locally:

```go
import fc "your.org/funnelcake"
```

> The trial links `libfunnelcake.a` by path from the repo root. Publishing a
> standalone, `go get`-able module (with the C library bundled or discovered via
> pkg-config) is future work.

## Alignment, in one sentence

The SIMD kernels require 32-byte-aligned planes and 32-byte-aligned strides.
`NewFrame` / `NewHDRFrame` allocate exactly that, and `NewScaler` /
`NewHDRScaler` derive matching source strides from the width you pass — so a
frame and a scaler created with the same dimensions always line up. If you ever
hand-roll buffers instead, misalignment silently drops the step to the scalar
path (visible as `Output.Fallback == true`).

## SDR: downscale 2x

```go
f, err := fc.NewFrame(1920, 1080)        // aligned I420 input
if err != nil {
    log.Fatal(err)
}
defer f.Close()

// Fill the planes (these are views into the frame's memory).
copy(f.Y(), srcY)
copy(f.U(), srcU)
copy(f.V(), srcV)

s, err := fc.NewScaler(fc.ScalerConfig{
    SrcWidth:  1920,
    SrcHeight: 1080,
    Flags:     fc.Scale2X,             // one family per scaler (pow2 or thirds)
})
if err != nil {
    log.Fatal(err)
}
defer s.Close()

s.Run(f)

out, ok := s.Output(fc.Scale2X)        // 960x540
if ok {
    use(out.Y(), out.U(), out.V())     // owned []byte copies
}
```

A single scaler can request several steps from **one** family
(`Scale2X|Scale4X|Scale8X|Scale16X` or `Scale1_5X|Scale3X|Scale6X|Scale12X`)
and an upscale cascade (`UpscaleFlags`, a contiguous prefix like
`Upscale2X|Upscale4X`, optionally with `UpscaleTail15`). All requested outputs
are produced in a single `Run`.

## HDR: 10-bit scale + tone-map to SDR

```go
f, err := fc.NewHDRFrame(3840, 2160, fc.PixI010)   // 10-bit planar
if err != nil {
    log.Fatal(err)
}
defer f.Close()
copy(f.Y(), srcY16)                                 // []uint16 views
copy(f.U(), srcU16)
copy(f.V(), srcV16)

s, err := fc.NewHDRScaler(fc.HDRConfig{
    SrcWidth:  3840,
    SrcHeight: 2160,
    Format:    fc.PixI010,
    Transfer:  fc.TRCPQ,
    Flags:     fc.Scale2X,
    HDRFlags:  fc.Scale2X,         // produce a 10-bit downscaled copy
    SDRFlags:  fc.Scale2X,         // and a tone-mapped 8-bit copy
    Tonemap1x: true,               // plus a 1:1 tone-mapped SDR copy
    Tonemap:   fc.TonemapConfig{Curve: fc.TonemapBT2390},
})
if err != nil {
    log.Fatal(err)
}
defer s.Close()

s.Run(f)

hdr, _ := s.HDROutput(fc.Scale2X)   // []uint16 planes
sdr, _ := s.SDROutput(fc.Scale2X)   // []byte planes, tone-mapped
one, _ := s.Tonemap1xOutput()       // 8-bit, source resolution
```

For semi-planar input (`PixP010` / `PixP210`), fill `f.U()` with the interleaved
UV plane and leave `f.V()` unused. 4:2:2 formats (`PixI210` / `PixP210`) are
accepted and decimated to 4:2:0 internally.

A zero `TonemapConfig` means the Hable curve, 1000-nit peak, 100-nit target, and
limited (video) range — the same defaults as the C library. Set
`Curve: fc.TonemapCustom` with a 1024-entry `CustomLUT` to supply your own.

## API reference

Everything below is in package `your.org/funnelcake`. Strides are always in
**bytes** (even for 10-bit planes, where one sample is 2 bytes). All dimensions
and strides are `int`.

### Package functions

```go
func SIMDAvailable() bool
```
Reports whether the vectorized kernels will run on this machine. When `false`,
every output's `Fallback` is `true` and a scalar warning is expected. Triggers
the library's one-time CPU probe on first call (safe under concurrency).

### Input frames

```go
type Frame struct {
    Width, Height      int // luma dimensions
    YStride, UVStride  int // bytes per row (32-byte aligned)
}

func NewFrame(width, height int) (*Frame, error)
func (f *Frame) Y() []byte   // luma plane,  len == YStride*Height
func (f *Frame) U() []byte   // Cb plane,    len == UVStride*chromaHeight
func (f *Frame) V() []byte   // Cr plane,    len == UVStride*chromaHeight
func (f *Frame) Close()
```
`Frame` is an 8-bit I420 (YUV 4:2:0 planar) input. `NewFrame` allocates the
three planes with 32-byte alignment and aligned strides; `width`/`height` should
be even. It returns `ErrBadDimensions` for invalid or unrepresentable dimensions.
Storage allocation follows normal Go allocation behavior. The `Y`/`U`/`V` methods return **writable** views into
the frame's own memory; fill them before calling `Run`. `Close` releases the
frame's references and is safe to call more than once. Existing slices retain
their backing storage.

```go
type HDRFrame struct {
    Width, Height      int
    Format             PixelFormat
    YStride, UVStride  int // bytes per row (UVStride == YStride for P010/P210)
}

func NewHDRFrame(width, height int, format PixelFormat) (*HDRFrame, error)
func (f *HDRFrame) Y() []uint16  // luma plane
func (f *HDRFrame) U() []uint16  // Cb plane (I010/I210) or interleaved UV (P010/P210)
func (f *HDRFrame) V() []uint16  // Cr plane (I010/I210); nil for P010/P210
func (f *HDRFrame) Close()
```
`HDRFrame` is a 10-bit input (samples in the low 10 bits of each `uint16`).
Layout follows `Format`: planar formats use three planes; semi-planar
(`PixP010`/`PixP210`) use luma plus one interleaved-UV plane and `V()` returns
`nil`. `NewHDRFrame` returns `ErrBadDimensions` for non-positive sizes and
`ErrInvalidFlags` for an unknown format.

### SDR scaler

```go
type ScalerConfig struct {
    SrcWidth, SrcHeight int
    Flags               ScaleFlag   // downscale steps; one family only
    UpscaleFlags        UpscaleFlag // contiguous-prefix upscale cascade (optional)
    UpscaleTail15       bool        // append a 1.5x tail to the upscale cascade
    Options             Option
}

type Scaler struct {
    Warnings Warnings // non-fatal conditions reported by init
}

func NewScaler(cfg ScalerConfig) (*Scaler, error)
func (s *Scaler) Run(f *Frame)
func (s *Scaler) EffectiveWidth() int
func (s *Scaler) EffectiveHeight() int
func (s *Scaler) AchievedFlags() ScaleFlag
func (s *Scaler) Output(flag ScaleFlag) (Output, bool)
func (s *Scaler) UpscaleOutput(flag UpscaleFlag) (Output, bool)
func (s *Scaler) UpscaleTail() (Output, bool)
func (s *Scaler) Close()
```
`NewScaler` validates the config and allocates output buffers. A negative
library result becomes a typed `Error`; on success the returned scaler's
`Warnings` field carries any non-fatal conditions. `Run` processes one frame
(which must come from `NewFrame` with the same width/height) and fills every
achieved output. `Output`/`UpscaleOutput`/`UpscaleTail` look up a result by
flag and return `ok == false` if that step was not produced.
`EffectiveWidth`/`EffectiveHeight` report the source region actually used after
any cropping. `Close` frees all buffers (safe to call repeatedly).

```go
type Output struct {
    Width, Height      int
    YStride, UVStride  int
    Fallback           bool // true if the scalar kernel was used for this step
}
func (o Output) Y() []byte
func (o Output) U() []byte
func (o Output) V() []byte
```
Metadata for one 8-bit output. Plane accessors return owned copies.

### HDR scaler

```go
type TonemapConfig struct {
    Curve              TonemapCurve
    PeakNits           int     // source peak (0 = 1000)
    TargetNits         int     // SDR target (0 = 100)
    SrcRange, DstRange Range
    CustomLUT          []byte  // 1024 entries; used only when Curve == TonemapCustom
}

type HDRConfig struct {
    SrcWidth, SrcHeight int
    Format              PixelFormat
    Transfer            TransferFunc
    Flags               ScaleFlag // requested downscale steps; one family only
    HDRFlags            ScaleFlag // subset of Flags: produce 10-bit outputs
    SDRFlags            ScaleFlag // subset of Flags: produce tone-mapped 8-bit outputs
    Options             Option
    Tonemap1x           bool          // also produce a 1:1 tone-mapped SDR copy
    Tonemap             TonemapConfig
    UpscaleFlags        UpscaleFlag
    UpscaleTail15       bool
    UpscaleSDRFlags     UpscaleFlag // subset of UpscaleFlags: tone-mapped SDR copies
    UpscaleSDRTail15    bool
}

type HDRScaler struct {
    Warnings Warnings
}

func NewHDRScaler(cfg HDRConfig) (*HDRScaler, error)
func (s *HDRScaler) Run(f *HDRFrame)
func (s *HDRScaler) EffectiveWidth() int
func (s *HDRScaler) EffectiveHeight() int
func (s *HDRScaler) HDROutput(flag ScaleFlag) (HDROutput, bool)       // 10-bit
func (s *HDRScaler) SDROutput(flag ScaleFlag) (Output, bool)          // tone-mapped 8-bit
func (s *HDRScaler) Tonemap1xOutput() (Output, bool)                  // 8-bit, source res
func (s *HDRScaler) UpscaleHDROutput(flag UpscaleFlag) (HDROutput, bool)
func (s *HDRScaler) UpscaleSDROutput(flag UpscaleFlag) (Output, bool)
func (s *HDRScaler) Close()
```
`NewHDRScaler` builds the tone-mapping LUTs and allocates outputs. `HDRFlags`
and `SDRFlags` must each be a subset of `Flags`; a step may produce a 10-bit
copy, a tone-mapped 8-bit copy, or both. A `CustomLUT` is copied into the
scaler at init, so you need not retain it. Accessors return `ok == false` for
steps that were not produced; `Tonemap1xOutput` is present only when
`Tonemap1x` was set.

```go
type HDROutput struct {
    Width, Height      int
    YStride, UVStride  int  // bytes
    Fallback           bool
}
func (o HDROutput) Y() []uint16
func (o HDROutput) U() []uint16
func (o HDROutput) V() []uint16
```
Metadata for one 10-bit output. Plane accessors return owned copies.

### Constants

```go
type ScaleFlag uint32   // downscale step selector
    Scale1_5X Scale3X Scale6X Scale12X   // "thirds" family
    Scale2X   Scale4X Scale8X Scale16X   // "power-of-two" family
    ScaleThirdsMask ScalePow2Mask         // family masks
```
A single `Scaler`/`HDRScaler` must use flags from **one** family. Combine
several steps from the same family with `|` (e.g. `Scale2X | Scale4X`).

```go
type UpscaleFlag uint32  // upscale cascade level
    Upscale2X Upscale4X Upscale8X Upscale16X Upscale32X
```
Must form a contiguous prefix: `{2x}`, `{2x,4x}`, `{2x,4x,8x}`, … — not `{4x}`
alone or `{2x,8x}`.

```go
type Option uint32
    OptNoCrop      // reject steps that would need cropping (default: crop + warn)
    OptNoFallback  // reject steps that can't use SIMD (default: scalar + warn)

type PixelFormat int
    PixI010  // 4:2:0 planar
    PixP010  // 4:2:0 semi-planar (interleaved UV)
    PixI210  // 4:2:2 planar      (decimated to 4:2:0 internally)
    PixP210  // 4:2:2 semi-planar (decimated to 4:2:0 internally)

type TransferFunc int
    TRCPQ   // SMPTE ST 2084 (HDR10)
    TRCHLG  // Hybrid Log-Gamma

type Range int
    RangeLimited  // video range (default)
    RangeFull     // full / PC range

type TonemapCurve int
    TonemapHable     // filmic (default)
    TonemapReinhard  // simple, lower contrast
    TonemapBT2390    // broadcast reference
    TonemapCustom    // use TonemapConfig.CustomLUT
```

### Errors and warnings

```go
type Error int
    ErrInvalidFlags   // mixed families or unknown bits
    ErrNoSteps        // no valid steps after filtering
    ErrBadDimensions  // non-positive or too-small source
    ErrBadAlignment   // strides not 32-byte aligned
func (e Error) Error() string

type Warnings uint32
func (w Warnings) Scalar() bool   // a step fell back to scalar
func (w Warnings) Partial() bool  // a requested step was rejected
func (w Warnings) Cropped() bool  // source was cropped to fit
func (w Warnings) Perfect() bool  // none of the above
```
`Error` values are returned by the constructors for hard failures and satisfy
`errors.Is`. `Warnings` is reported on a successful scaler's field, never as an
error.

## Output buffer lifetime

Output plane accessors return independent Go-owned slices. Obtain these copies
before closing the scaler; they survive subsequent Run, Close, and garbage
collection. Input frame slices also retain their Go-owned backing storage.

## Errors and warnings

`NewScaler` / `NewHDRScaler` return a typed `Error` (wrapping the negative
`FUSED_ERR_*` codes) for hard failures; compare with `errors.Is`:

```go
if errors.Is(err, fc.ErrInvalidFlags) { /* mixed families, etc. */ }
```

Non-fatal conditions are reported on the `Warnings` field, not as an error:

```go
if !s.Warnings.Perfect() {
    _ = s.Warnings.Scalar()   // a step fell back to scalar
    _ = s.Warnings.Partial()  // a requested step was rejected
    _ = s.Warnings.Cropped()  // source was cropped to fit
}
```

`fc.SIMDAvailable()` reports whether the vectorized kernels will run at all on
this machine; when it returns false, every output's `Fallback` is true and a
scalar warning is expected.

## Safety notes

- **Frame must match the scaler.** `Run` panics if the frame's dimensions (and,
  for HDR, the format) differ from what the scaler was configured with — this
  prevents an out-of-bounds read in the native kernels. Build the frame with the
  same width/height you passed to the scaler.
- **Custom LUT size.** A `TonemapConfig.CustomLUT` must be exactly 1024 bytes;
  otherwise `NewHDRScaler` returns `ErrCustomLUTLength`.
- **Obtain outputs before Close.** Output access after closing the scaler
  panics. Previously returned plane copies remain valid.

## Concurrency

Each `Scaler` / `HDRScaler` is independent and may run on its own goroutine. Do
not share one scaler across goroutines without your own synchronization. The
package forces the library's one-time CPU probe on first use, so concurrent
first inits are safe.

### Plane ownership

Frame and HDRFrame plane slices use aligned Go-owned storage and remain valid
if the frame is closed or garbage-collected. Public dimensions and strides are
informational snapshots; changing them does not change allocation bounds or
source validation.

Output and HDROutput Y/U/V accessors return owned copies, including row padding.
Obtain copies before closing the scaler. They remain valid after another Run,
Close, or garbage collection, and modifying a copy never changes scaler output.
This replaces the earlier borrowed-output-slice behavior, since a plain Go slice
cannot keep a C allocation owner alive.
