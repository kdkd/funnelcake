# Building Funnelcake

## Requirements

- C compiler: clang (recommended) or gcc
- GNU make
- No external libraries required

## Build

    make                        # build libfunnelcake.a (O3, generic tuning)
    make test                   # build and run tests
    make bench                  # build and run benchmarks
    make visual                 # generate test output PNGs to output/
    make CC=gcc                 # use gcc instead of clang

## Performance Tuning

    make TUNE=znver2            # optimize scheduling for AMD Epyc 7302 / Zen 2
    make TUNE=znver1            # optimize scheduling for AMD Epyc 7301 / Zen 1
    make TUNE=broadwell         # optimize scheduling for Intel E5-2680v4
    make TUNE=skylake           # optimize scheduling for Intel Xeon 6132
    make TUNE=native            # optimize for the build machine's CPU

> **Note - Skylake vs Broadwell tuning:** On Xeon Scalable (Skylake-SP) hardware,
> `TUNE=skylake` does not always outperform `TUNE=broadwell`.  In our benchmarks on
> a Xeon 6132, `broadwell` produced consistently better results (3-4% lower latency
> across all workloads).  GCC's `-mtune=skylake` adjusts prefetch distances and
> micro-op scheduling in ways that can conflict with the shuffle-heavy AVX2 inner
> loops used here.  When deploying to Skylake hardware, benchmark both options with
> `make pgo TUNE=broadwell LTO=1` and `make pgo TUNE=skylake LTO=1` and keep
> whichever is faster.

    make LTO=1                  # enable link-time optimization for in-tree binaries
    make LTO=1 TUNE=znver2      # combine LTO with CPU tuning

    make pgo                    # profile-guided optimization (compile, benchmark, recompile)

    make SCALAR_ARCH=native     # optimize scalar fallback for build machine
    make SCALAR_ARCH=baseline   # maximum portability for scalar fallback (default)

### Recommended Production Build

For AMD Epyc 7302 fleet:

    make LTO=1 TUNE=znver2 SCALAR_ARCH=znver2

For mixed fleet (deploy single binary):

    make LTO=1

For maximum performance on a specific machine:

    make pgo TUNE=native LTO=1

If you are shipping `libfunnelcake.a` to another build system, keep `LTO=0`
(the default). `clang -flto` archives can contain LLVM bitcode members instead
of standard ELF `.o` files, which some consumers reject.

## Platform Notes

### macOS (Apple Silicon)

Install Xcode command line tools:

    xcode-select --install

### Ubuntu / Debian (x86_64)

    sudo apt install build-essential clang

### Ubuntu / Debian (aarch64)

    sudo apt install build-essential clang

## Ubuntu 20 Release Artifacts

To build portable Ubuntu 20.04 release tarballs for both `amd64` and `arm64`
from any Docker host with `buildx` enabled:

    ./scripts/build-linux-ubuntu20.sh

Artifacts are written to `dist/` as per-architecture tarballs containing
`libfunnelcake.a`, `include/funnelcake.h`, and basic build metadata. These
release archives are built with `CC=clang LTO=0` so the static library contains
standard object files suitable for downstream linkers that do not understand
Clang LTO bitcode.
