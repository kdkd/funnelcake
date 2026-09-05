# Building Funnelcake

## Requirements

- C compiler: clang >= 6 or gcc >= 10 (gcc 10 is required for `-flto=auto`;
  older gcc still builds but LTO will run serially)
- GNU make
- No external libraries required for the library itself

The Makefile defaults to the system compiler (`cc`), which is Apple clang on
macOS and usually gcc on Linux. Both compiler families are fully supported,
including the `make pgo` pipeline which auto-detects the compiler and uses
the appropriate PGO workflow (gcc reads `.gcda` files directly; clang runs
`llvm-profdata merge` between the instrumented run and the optimized build).

Compiler choice matters on two architectures. **RISC-V should be built with
gcc >= 14** (or clang >= 18), while **AVX-512 needs gcc >= 8 or clang >= 7**
to compile the AVX-512 kernels. Older compilers still produce working
binaries, but the RISC-V path is slower and x86 builds top out at AVX2. See
[Per-architecture notes](#per-architecture-notes) for details.

## Build

    make                        # build libfunnelcake.a (O3, generic tuning)
    make test                   # build and run tests
    make bench                  # build and run benchmarks
    make visual                 # generate test output PNGs to output/
    make CC=gcc                 # pick a specific compiler
    make CC=clang-17            # any compiler binary works
    make -j8                    # parallel builds are fully supported

On FreeBSD, use `gmake` for these commands. For example:

    gmake test                  # system Clang
    gmake test CC=gcc14          # GCC installed from packages or ports
    gmake pgo CC=gcc14 LTO=1 TUNE=native

### Make targets

| Target | What it does |
|--------|--------------|
| `all` (default) | Same as `lib`. |
| `lib` | Build `libfunnelcake.a`. |
| `shared` | Build the shared library (`libfunnelcake.so.N`, or `libfunnelcake.N.dylib` on macOS). Reuses the same objects; they are always compiled `-fPIC`. |
| `install` | Install static and shared libraries, both public headers, and `funnelcake.pc` under `PREFIX`. Honors `DESTDIR`. |
| `funnelcake_test` | Build the test/benchmark binary without running it. |
| `test` | Build and run the full test suite. |
| `bench` | Run all benchmarks: SDR, libswscale comparison, and HDR. |
| `bench-sdr` | SDR benchmarks plus the libswscale comparison. |
| `bench-hdr` | HDR benchmarks (10-bit scaling and tone mapping). |
| `bench-swscale` | libswscale comparison only. Needs a build with libswscale available. |
| `visual` | Render each scale step to `output/` as PNG/MOV for visual inspection. Creates `output/` and needs ffmpeg on `PATH`. |
| `fetch-samples` | Generate synthetic PQ 10-bit test frames into `test/samples/` (gitignored). Needs ffmpeg and curl. |
| `asm` | Emit annotated `.S` listings next to the kernel sources, for reading codegen. Honors `TUNE`, e.g. `make asm TUNE=native`. |
| `pgo` | Full profile-guided build: instrument, run the benchmarks and tests, recompile the library with the profile. Combine with `TUNE=` and `LTO=1`. |
| `pgo-clean` | Delete profile data (`.gcda`, `.profraw`, `.profdata`) without touching objects. |
| `clean` | Remove all build products, including binding artifacts and the contents of `output/`. |

Language bindings are opt-in and are never built by `all` or `test`:

| Target | What it does |
|--------|--------------|
| `bindings-go` / `test-go` | Build or test the cgo binding. Override the toolchain with `GO=go1.23`. |
| `bindings-rust` / `test-rust` | Build or test the Rust binding. Override with `CARGO=`. |
| `bindings-python` / `test-python` | Build or test the ctypes binding. Override with `PYTHON=`. |
| `bindings-java` / `test-java` | Build or test the FFM binding (JDK 22+). Override with `JAVA=` / `JAVAC=`. |

### Make variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CC` | `cc` | Compiler. `make CC=gcc-14`, `make CC=clang-17`. |
| `TUNE` | (empty) | `-mtune=` target. See [Performance Tuning](#performance-tuning). |
| `LTO` | `0` | Set to `1` for link-time optimization. Auto-disabled on riscv64. |
| `SCALAR_ARCH` | `baseline` | `-march=` for the scalar fallback only (`native`, `znver2`, ...). |
| `PREFIX` | `/usr/local` | Install prefix. `LIBDIR`, `INCLUDEDIR`, and `PKGCONFDIR` derive from it and can each be overridden. |
| `DESTDIR` | (empty) | Staging root prepended to install paths, for packaging. |
| `LLVM_PROFDATA` | `llvm-profdata` | Only used by `make pgo` under clang; point it at a versioned binary if needed. |

## Performance Tuning

`TUNE=<cpu>` passes `-mtune=<cpu>` to the compiler. It changes instruction
scheduling without changing the instruction set, so the resulting binary
runs anywhere an untuned build does. It works on every architecture:

    make TUNE=znver2            # AMD Epyc 7302 / Zen 2
    make TUNE=znver1            # AMD Epyc 7301 / Zen 1
    make TUNE=broadwell         # Intel E5-2680v4
    make TUNE=skylake           # Intel Xeon 6132
    make TUNE=neoverse-v2       # AWS Graviton 4
    make TUNE=cortex-a76        # Raspberry Pi 5, RK3588 big cluster
    make TUNE=native            # the build machine's CPU

    make LTO=1                  # enable link-time optimization for in-tree binaries
    make LTO=1 TUNE=znver2      # combine LTO with CPU tuning

    make pgo                    # profile-guided optimization (compile, benchmark, recompile)

LTO automatically picks the compiler's preferred flavor: `-flto=thin` on
clang (ThinLTO, parallelized by the linker plugin) and `-flto=auto` on gcc
(parallel LTRANS across all cores). Both deliver substantially the same
optimization quality as full LTO with much faster builds and no
"using serial compilation of N LTRANS jobs" warning.

GCC LTO builds automatically select `gcc-ar` from the compiler's toolchain,
including versioned names such as `gcc-ar-14` and FreeBSD's `gcc-ar14`.
An explicit `AR` setting takes precedence. If automatic discovery fails,
install the matching GCC tools or set `AR` to a plugin-aware archiver.

    make SCALAR_ARCH=native     # optimize scalar fallback for build machine
    make SCALAR_ARCH=baseline   # maximum portability for scalar fallback (default)

### Recommended Production Build

For AMD Epyc 7302 fleet:

    make LTO=1 TUNE=znver2 SCALAR_ARCH=znver2

For mixed fleet (deploy single binary):

    make LTO=1

For a Graviton 4 fleet:

    make LTO=1 TUNE=neoverse-v2

For RISC-V (LTO is unavailable here, so PGO carries the load):

    make CC=gcc-14 pgo

For maximum performance on a specific machine:

    make pgo TUNE=native LTO=1

If you are shipping `libfunnelcake.a` to another build system, keep `LTO=0`
(the default). Both `clang -flto=thin` and `gcc -flto=auto` archives can
contain compiler-specific intermediate representation instead of standard
object files, which some downstream linkers reject unless they use the same
compiler toolchain.

## The test binary

The `test`, `bench*`, and `visual` targets run `./funnelcake_test` with common
option combinations. Run the binary directly when you need a workload filter
or want to combine options; `./funnelcake_test --help` shows the available
options.

    make funnelcake_test        # build it without running anything
    ./funnelcake_test --help

| Option | Effect |
|--------|--------|
| (none) | Run the full suite: validation, correctness, HDR, tone mapping, and scalar-vs-SIMD parity. Exits 1 if anything failed. |
| `--bench [filter]` | SDR, libswscale comparison, and HDR benchmarks. |
| `--bench-sdr [filter]` | SDR benchmarks plus the libswscale comparison. |
| `--bench-hdr [filter]` | HDR benchmarks (10-bit scaling and tone mapping). |
| `--bench-swscale [filter]` | libswscale comparison only. |
| `--skip-bench-swscale` | Never run the libswscale comparison. |
| `--visual` | Render each scale step to `output/`. Requires ffmpeg and an existing `output/` directory; `make visual` creates it. |
| `-h`, `--help` | Usage summary. |

The optional `filter` is a plain substring match against the workload
label, so `--bench-sdr 1920x1080` runs every 1080p row and
`--bench-sdr down:2x` runs the 2x downscale rows at every resolution. Run a
benchmark option with no filter to run every workload in that group. An
unrecognized option is an error (exit 2) rather than being ignored.

The swscale comparison can turn a short kernel A/B into a multi-minute run.
Skip it during kernel tuning when you only need funnelcake results:

    ./funnelcake_test --bench-sdr 'down:1.5x,3x,6x' --skip-bench-swscale

### Test environment variables

| Variable | Effect |
|----------|--------|
| `FUNNELCAKE_FORCE_SCALAR` | Skip SIMD detection entirely and use the scalar kernels. The parity tests toggle this to compare both kernel sets in one process. |
| `FUNNELCAKE_NO_AVX512` | On AVX-512 hardware, fall back to AVX2 for a same-build comparison of the two x86 kernel sets. |
| `FUNNELCAKE_SWSCALE_VERIFY` | In the libswscale comparison, print each step's row count and Y-plane checksum. |

The first two are read by the library itself and are documented in
[docs/API.md](docs/API.md#environment-variables); they apply to any
consumer, not just the test binary.

## Per-architecture notes

### x86-64 (AVX2 / AVX-512)

The AVX-512 kernels need a compiler that accepts the F+BW+VL+VBMI flags
(gcc >= 8, clang >= 7). The Makefile try-compiles those flags and says
which way it went near the top of the build:

    funnelcake: AVX-512 kernels: WILL be compiled ...

On an older compiler the AVX-512 files still build, as stubs that runtime
dispatch never selects, and AVX2 becomes the top tier. Which tier actually
runs is a cpuid decision made at runtime, so one binary covers a mixed
fleet.

**Skylake vs Broadwell tuning.** On Xeon Scalable (Skylake-SP) hardware,
`TUNE=skylake` does not always beat `TUNE=broadwell`. On a Xeon 6132,
`broadwell` was consistently 3-4% faster across all workloads: gcc's
`-mtune=skylake` adjusts prefetch distances and micro-op scheduling in ways
that conflict with the shuffle-heavy AVX2 inner loops here. Benchmark both
with `make pgo TUNE=broadwell LTO=1` and `make pgo TUNE=skylake LTO=1`
and keep whichever wins.

### aarch64 (NEON)

Every aarch64 core has NEON, so there is no `-march` to set and no feature
gate to miss - a plain `make` already gets the vector kernels. `TUNE` is the
only knob that matters, and it is worth setting: `neoverse-v2` for Graviton
4, `cortex-a76` for the Raspberry Pi 5 and the RK3588 big cluster,
`apple-m1` (or just `native`) on Apple Silicon.

**Prefer clang over gcc 11/12 if you have the choice.** The kernels use
`vst2q` interleaving stores in the 1.5x thirds and 2x-upscale paths. Apple
clang and clang 14+ schedule these well; gcc 11.4 and 12.2 generate erratic
code around interleaved stores at small vector sizes, and on A76 cores that
costs 3-10% on the thirds downscale ladders relative to the same source
built with clang. Newer gcc has not been measured. `make CC=clang` is
enough; no other changes are needed.

### RISC-V (RVV)

**Use gcc 14 or newer** (clang 18+ works equally well and emits identical
segment codegen):

    make CC=gcc-14

gcc 14 ships the v1.0 RVV intrinsic spec, including the
`vlseg2`/`vsseg2`/`vlseg3`/`vsseg3` segment loads and stores that the
kernels use for every horizontal halve, 3:1 box average, 1.5x bilinear, and
2x upsample. gcc 13 only ships v0.11 intrinsics, which have no segment ops,
so the build falls back to multiple strided accesses per chunk. On a
SpacemiT X60 that measured 1.4x to 3.7x slower, with the largest losses in
the 2x upscale rows. If gcc 13 is the system `cc`, a plain `make` still
succeeds; a `#pragma message` in the compile log is the only warning.

`LTO=1` is auto-disabled on `riscv64` and the Makefile prints a `$(warning)`
saying so. gcc 13's LTO link cannot resolve the RVV target builtins, and gcc
14's LTO partition pass hits an internal compiler error in
`riscv_vector::expand_builtin`. The build continues with `-O3` alone.
`make pgo` is unaffected and recovers most of that ground.

Leave `TUNE` unset unless your compiler actually has a scheduling model for
your chip. gcc 14 has no entry for the X60, and an unrecognized value is a
hard error rather than a fallback. The kernels are vector-length-agnostic,
so one binary runs on any V-capable chip, but the LMUL and unrolling choices
were tuned against the X60 specifically.

## Platform Notes

### macOS (Apple Silicon)

Install Xcode command line tools:

    xcode-select --install

### Ubuntu / Debian (x86_64)

    sudo apt install build-essential        # gcc (recommended on Linux)
    # or
    sudo apt install build-essential clang  # clang alternative

### Ubuntu / Debian (aarch64)

    sudo apt install build-essential
    # or
    sudo apt install build-essential clang

### Fedora / RHEL

    sudo dnf install gcc make
    # or
    sudo dnf install clang make

### Arch Linux

    sudo pacman -S base-devel  # gcc + make
    # or
    sudo pacman -S clang make

### Debian / Ubuntu (riscv64)

If `cc` resolves to gcc 13, install a newer compiler and name it explicitly.
See [RISC-V (RVV)](#risc-v-rvv) for why the version matters.

    sudo apt install build-essential gcc-14
    make CC=gcc-14

### macOS (Homebrew GCC)

If you want to build with a real GCC (not Apple clang aliased as gcc):

    brew install gcc
    make CC=gcc-15 LTO=1 TUNE=native

## Releases and packaging

### Ubuntu 20 release artifacts

To build portable Ubuntu 20.04 release tarballs for both `amd64` and `arm64`
from any Docker host with `buildx` enabled:

    ./scripts/build-linux-ubuntu20.sh

Artifacts are written to `dist/` as per-architecture tarballs containing
`libfunnelcake.a`, `include/funnelcake.h`, and basic build metadata. These
release archives are built with `CC=clang LTO=0` so the static library contains
standard object files suitable for downstream linkers that do not understand
Clang LTO bitcode.

### Cutting a new release

1. Update `VERSION` at the top of the [Makefile](Makefile). It is the single
   source of truth for `funnelcake.pc` and the FreeBSD port.
2. If the public ABI changed in a backward-incompatible way, also bump
   `SOVERSION` in the Makefile. This drives the installed `libfunnelcake.so.N`
   suffix; downstream packages will need to be rebuilt against the new
   major.
3. Commit the version bump, then tag:
   ```
   git tag -a v0.1.0 -m "Release 0.1.0"
   git push origin v0.1.0
   ```
4. GitHub auto-generates a tarball at
   `https://github.com/<owner>/funnelcake/archive/refs/tags/v0.1.0.tar.gz`
   that the FreeBSD port consumes via `USE_GITHUB`.

### Building and submitting the FreeBSD port

A port skeleton lives in [scripts/freebsd/](scripts/freebsd/). To exercise
or update the port locally:

```sh
# 1. Copy the skeleton into your ports tree.
sudo mkdir -p /usr/ports/multimedia/funnelcake
sudo cp scripts/freebsd/Makefile scripts/freebsd/pkg-descr \
        scripts/freebsd/pkg-plist /usr/ports/multimedia/funnelcake/

# 2. Update DISTVERSION in the port Makefile to match the upstream tag.

# 3. Generate the distfile checksum:
cd /usr/ports/multimedia/funnelcake
sudo make makesum

# 4. Lint, build, install, and verify the packaging list. BATCH=yes skips
#    the interactive options-config dialog (which hangs over a non-TTY
#    SSH session if you have OPTIONS_DEFINE knobs):
sudo make BATCH=yes stage check-plist
sudo make BATCH=yes package
sudo pkg add work/pkg/funnelcake-*.pkg

# 5. Run the official lint pass (portaudit-equivalent):
sudo portlint -A
```

Once the port builds and lints cleanly, submit it as a bug report against
the FreeBSD ports tree per the
[Porter's Handbook, section 3.7](https://docs.freebsd.org/en/books/porters-handbook/book/#porting-submitting).
The optional `FFMPEG` knob pulls in `multimedia/ffmpeg` for the swscale
benchmark comparison; without it the library and headers install but
`fetch-samples` / `bench-swscale` are unavailable at runtime.
