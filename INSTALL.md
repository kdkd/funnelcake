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

## Build

    make                        # build libfunnelcake.a (O3, generic tuning)
    make test                   # build and run tests
    make bench                  # build and run benchmarks
    make visual                 # generate test output PNGs to output/
    make CC=gcc                 # pick a specific compiler
    make CC=clang-17            # any compiler binary works
    make -j8                    # parallel builds are fully supported

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

LTO automatically picks the compiler's preferred flavor: `-flto=thin` on
clang (ThinLTO, parallelized by the linker plugin) and `-flto=auto` on gcc
(parallel LTRANS across all cores). Both deliver substantially the same
optimization quality as full LTO with much faster builds and no
"using serial compilation of N LTRANS jobs" warning.

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
(the default). Both `clang -flto=thin` and `gcc -flto=auto` archives can
contain compiler-specific intermediate representation instead of standard
object files, which some downstream linkers reject unless they use the same
compiler toolchain.

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

### macOS (Homebrew GCC)

If you want to build with a real GCC (not Apple clang aliased as gcc):

    brew install gcc
    make CC=gcc-15 LTO=1 TUNE=native

## Ubuntu 20 Release Artifacts

To build portable Ubuntu 20.04 release tarballs for both `amd64` and `arm64`
from any Docker host with `buildx` enabled:

    ./scripts/build-linux-ubuntu20.sh

Artifacts are written to `dist/` as per-architecture tarballs containing
`libfunnelcake.a`, `include/funnelcake.h`, `include/funnelcake_helpers.h`, and
basic build metadata. These release archives are built with `CC=clang LTO=0`
so the static library contains standard object files suitable for downstream
linkers that do not understand Clang LTO bitcode.

## macOS Release Artifacts

To build a native macOS release archive:

    ./scripts/build-macos.sh

The script must be run on macOS with Xcode command line tools installed. It
writes `dist/funnelcake-macos-<arch>.tar.gz` containing `libfunnelcake.a`,
`include/funnelcake.h`, the README, install notes, and `BUILD_INFO`.

## Windows Release Artifacts

To build Windows release archives:

    ./scripts/build-windows.sh             # bash / MSYS2 / Git Bash
    scripts\build-windows.ps1              # native PowerShell 5.1+

By default the script builds every Windows target whose toolchain is available.
MinGW-w64 artifacts contain `libfunnelcake.a`; MSVC artifacts contain
`funnelcake.lib`. Both package layouts include `include/funnelcake.h`, the
README, install notes, and `BUILD_INFO`.

For MinGW-w64 only:

    ./scripts/build-windows.sh --mingw
    scripts\build-windows.ps1 -Mingw

For `x86_64` MinGW, install tools that provide `x86_64-w64-mingw32-gcc` and
`x86_64-w64-mingw32-ar`. For Windows on ARM64 MinGW, install tools that provide
`aarch64-w64-mingw32-gcc` and `aarch64-w64-mingw32-ar`.

For MSVC only, run from a Visual Studio developer shell where `cl.exe` and
`lib.exe` are in `PATH`:

    ./scripts/build-windows.sh --msvc
    scripts\build-windows.ps1 -Msvc

Both MinGW and MSVC builds use the normal source selection (AVX2 on `x86_64`,
NEON on `aarch64`/ARM64). The NEON kernels guard on `__aarch64__ || _M_ARM64`,
so MSVC ARM64 picks up the same SIMD coverage as the MinGW cross-compile.

### Windows ARM64 only

For a Windows-on-ARM64 build without touching the x86_64 paths, use the
dedicated PowerShell driver:

    scripts\build-windows-arm64.ps1                # MinGW + MSVC, whichever is available
    scripts\build-windows-arm64.ps1 -Mingw         # cross-compile via aarch64-w64-mingw32-gcc
    scripts\build-windows-arm64.ps1 -Msvc          # native ARM64 MSVC

The `-Msvc` path requires an "ARM64 Native Tools Command Prompt for VS" or
an equivalent Developer PowerShell with `VSCMD_ARG_TGT_ARCH=arm64`; the
script refuses to run if the shell is not configured for ARM64. Both
artifact layouts mirror `build-windows.ps1`: a per-toolchain `dist/`
package plus a `.zip.sha256`.
