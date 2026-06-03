# Compiler: defaults to the system `cc` (Apple clang on macOS, gcc on most
# Linux distros via /usr/bin/cc). Override with `make CC=gcc-15` or
# `make CC=clang-17` to pick a specific toolchain. Both gcc >= 10 and
# clang >= 6 are known to work; older gcc may not support -flto=auto.
CC ?= cc
AR ?= ar
CFLAGS_BASE = -std=c11 -D_POSIX_C_SOURCE=200112L -Wall -Wextra -Werror -fPIC -Iinclude -Isrc
LDFLAGS = -lm

# Optimization level: O3 for library, O2 for tests (tests don't need aggressive opts)
LIB_OPT = -O3
TEST_OPT = -O2

# Platform detection
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Windows_NT)
  CFLAGS_BASE += -D__USE_MINGW_ANSI_STDIO=1
endif

# Normalize FreeBSD's "amd64" to "x86_64" so the SIMD-selection blocks
# below match. FreeBSD/arm64 already reports "aarch64".
ifeq ($(UNAME_S),FreeBSD)
  ifeq ($(UNAME_M),amd64)
    UNAME_M := x86_64
  endif
endif

# --- Versioning and install layout ---
# VERSION is the project version string used in funnelcake.pc and elsewhere.
# Bump this whenever you cut a release (see README.md, "Release process").
VERSION    ?= 0.1.0

# SOVERSION is the shared-library major version. Bump only on ABI breaks.
SOVERSION  ?= 1

# Install destinations (overridable from the command line / port Makefile).
PREFIX     ?= /usr/local
LIBDIR     ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
# FreeBSD ports use $(PREFIX)/libdata/pkgconfig; Linux distros use lib/pkgconfig.
# Default to lib/pkgconfig; the FreeBSD port Makefile overrides it.
PKGCONFDIR ?= $(LIBDIR)/pkgconfig
INSTALL    ?= install

# Shared-library naming and link flags differ between Darwin and ELF systems.
ifeq ($(UNAME_S),Darwin)
  SHLIB         = libfunnelcake.$(SOVERSION).dylib
  SHLIB_LINK    = libfunnelcake.dylib
  SHLIB_LDFLAGS = -dynamiclib -install_name $(LIBDIR)/$(SHLIB)
else
  SHLIB         = libfunnelcake.so.$(SOVERSION)
  SHLIB_LINK    = libfunnelcake.so
  SHLIB_LDFLAGS = -shared -Wl,-soname,$(SHLIB)
endif

# --- Tuning options ---

# SCALAR_ARCH: controls -march for kernels_scalar.c only
# Options: baseline (default), native, sse4.2, avx2, etc.
SCALAR_ARCH ?= baseline

ifeq ($(SCALAR_ARCH),baseline)
  ifeq ($(UNAME_M),x86_64)
    SCALAR_CFLAGS = -march=x86-64
  else
    SCALAR_CFLAGS =
  endif
else ifeq ($(SCALAR_ARCH),native)
  SCALAR_CFLAGS = -march=native
else
  SCALAR_CFLAGS = -march=$(SCALAR_ARCH)
endif

# TUNE: CPU-specific instruction scheduling for SIMD kernels.
# Does not change which instructions are used, only how they are scheduled.
# Options: (empty = generic), znver1, znver2, znver3, broadwell, skylake, native
# Examples:
#   make TUNE=znver2        # optimize scheduling for Epyc 7302 / Ryzen 3000
#   make TUNE=broadwell     # optimize scheduling for E5-2680v4
#   make TUNE=native        # optimize for the build machine
TUNE ?=

ifneq ($(TUNE),)
  TUNE_CFLAGS = -mtune=$(TUNE)
else
  TUNE_CFLAGS =
endif

# LTO: Link-time optimization. Enables cross-file inlining and interprocedural
# optimization. Recommended for production builds.
# Options: 0 (off, default for dev), 1 (on)
# Example: make LTO=1
#
# Compiler-specific LTO flag selection:
#   clang: -flto=thin produces roughly the same optimization quality as
#          full LTO but parallelizes the backend via the linker plugin's
#          thread pool; build time is much faster and we avoid Apple ld64's
#          single-threaded full-LTO path.
#   gcc:   -flto=auto spawns parallel LTRANS jobs (the "lto-wrapper:
#          warning: using serial compilation of N LTRANS jobs" message
#          from plain -flto). Requires GCC 10 or newer. On older GCC,
#          -flto still works but runs serially.
LTO ?= 0

# (CC_IS_CLANG is detected below in the PGO section, after TUNE handling.
# This block picks the LTO flag once we know which compiler family we have.)
# To keep the Makefile readable we resolve it here using the same probe.
CC_FAMILY_IS_CLANG := $(shell $(CC) --version 2>/dev/null | grep -qi clang && echo 1)

ifeq ($(LTO),1)
  # LTO + RVV intrinsics is currently broken on every GCC we've tested:
  # GCC 13 lacks target-builtin info during the LTO link, and GCC 14 hits
  # an internal compiler error in riscv_vector::expand_builtin during the
  # LTO partition pass.  Auto-disable LTO on riscv64 with a clear notice
  # rather than letting the user hit the cryptic ICE.  Drop this guard
  # once a fixed compiler ships and is tested.
  ifeq ($(UNAME_M),riscv64)
    $(warning funnelcake: LTO disabled on riscv64 (GCC LTO + RVV intrinsics ICEs in tested versions). Build will use -O3 only.)
    LTO_CFLAGS =
  else ifeq ($(CC_FAMILY_IS_CLANG),1)
    LTO_CFLAGS = -flto=thin
  else
    LTO_CFLAGS = -flto=auto
  endif
else
  LTO_CFLAGS =
endif

# Assemble final CFLAGS. LTO_CFLAGS appears in both LIB_CFLAGS and
# TEST_CFLAGS so the same flag drives both compile and link (the link
# step re-invokes the compiler driver with TEST_CFLAGS and the driver
# forwards the LTO flag to the linker plugin).
LIB_CFLAGS = $(CFLAGS_BASE) $(LIB_OPT) $(TUNE_CFLAGS) $(LTO_CFLAGS)
TEST_CFLAGS = $(CFLAGS_BASE) $(TEST_OPT) $(TUNE_CFLAGS) $(LTO_CFLAGS)

# EXTRA_LDFLAGS is an injection point used by the pgo target to pass
# -fprofile-generate (and any other step-1-only link flag) into the
# final link step. Normal builds leave this empty.
EXTRA_LDFLAGS ?=

# detect.c runs exactly once at startup. Never compile it with PGO flags.
# -fprofile-generate changes register allocation inside the xgetbv inline asm,
# which can cause "=A" to capture rdx (high 32 bits) instead of rax (low 32
# bits), recording a false "XCR0 bits not set" result in the profile data.
# The -fprofile-use pass then treats the AVX2 assignment as dead code.
# This variable intentionally excludes LIB_OPT (where -fprofile-* lives).
DETECT_CFLAGS = $(CFLAGS_BASE) -O2 $(TUNE_CFLAGS)

# --- PGO compiler-family handling ---
# GCC and clang have very different PGO workflows:
#   GCC:   -fprofile-generate produces .gcda files next to the .o files;
#          -fprofile-use reads them directly. No merge step needed.
#   clang: -fprofile-generate produces default.profraw in the CWD;
#          -fprofile-use expects default.profdata, which must be produced
#          by running "llvm-profdata merge" on the .profraw files.
# CC_FAMILY_IS_CLANG is defined earlier in the LTO block.

# llvm-profdata lookup: on macOS the Xcode toolchain ships it alongside
# clang but not always in PATH, so prefer `xcrun` there. On Linux the
# tool is typically in PATH as `llvm-profdata`; if the user installed
# a versioned package (e.g. llvm-profdata-17) they can override via
# `make LLVM_PROFDATA=llvm-profdata-17 pgo`.
ifeq ($(UNAME_S),Darwin)
  LLVM_PROFDATA ?= xcrun llvm-profdata
else
  LLVM_PROFDATA ?= llvm-profdata
endif

# Suppress the "no profile data for this TU" warning that both compilers
# emit for files that weren't exercised by the PGO training run (e.g. the
# test harness, cold-path code, detect.c which is deliberately excluded).
# GCC and clang spell the flag differently.
ifeq ($(CC_FAMILY_IS_CLANG),1)
  PGO_USE_FLAGS = -fprofile-use -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date
else
  PGO_USE_FLAGS = -fprofile-use -Wno-missing-profile
endif

# --- Optional: libswscale for comparison benchmarks ---
# Try pkg-config first, then probe common paths
SWSCALE_CFLAGS := $(shell pkg-config --cflags libswscale libavutil 2>/dev/null)
SWSCALE_LIBS   := $(shell pkg-config --libs libswscale libavutil 2>/dev/null)

ifeq ($(SWSCALE_LIBS),)
  # macOS Homebrew on Apple Silicon
  ifneq ($(wildcard /opt/homebrew/include/libswscale/swscale.h),)
    SWSCALE_CFLAGS := -I/opt/homebrew/include
    SWSCALE_LIBS   := -L/opt/homebrew/lib -lswscale -lavutil
  endif
endif
ifeq ($(SWSCALE_LIBS),)
  # Intel Mac Homebrew / FreeBSD / manually installed /usr/local
  ifneq ($(wildcard /usr/local/include/libswscale/swscale.h),)
    SWSCALE_CFLAGS := -I/usr/local/include
    SWSCALE_LIBS   := -L/usr/local/lib -lswscale -lavutil
  endif
endif
ifeq ($(SWSCALE_LIBS),)
  # Debian/Ubuntu multiarch x86_64
  ifneq ($(wildcard /usr/include/x86_64-linux-gnu/libswscale/swscale.h),)
    SWSCALE_CFLAGS := -I/usr/include/x86_64-linux-gnu
    SWSCALE_LIBS   := -lswscale -lavutil
  endif
endif
ifeq ($(SWSCALE_LIBS),)
  # Debian/Ubuntu multiarch aarch64
  ifneq ($(wildcard /usr/include/aarch64-linux-gnu/libswscale/swscale.h),)
    SWSCALE_CFLAGS := -I/usr/include/aarch64-linux-gnu
    SWSCALE_LIBS   := -lswscale -lavutil
  endif
endif
ifeq ($(SWSCALE_LIBS),)
  # Arch / Fedora / Alpine - libraries live directly under /usr/include
  ifneq ($(wildcard /usr/include/libswscale/swscale.h),)
    SWSCALE_CFLAGS :=
    SWSCALE_LIBS   := -lswscale -lavutil
  endif
endif

ifneq ($(SWSCALE_LIBS),)
  SWSCALE_TEST_CFLAGS = $(SWSCALE_CFLAGS) -DHAVE_LIBSWSCALE
  SWSCALE_TEST_LDFLAGS = $(SWSCALE_LIBS)
endif

# Library sources (always compiled)
LIB_SRCS = src/funnelcake.c src/funnelcake_hdr.c src/log.c src/detect.c \
           src/kernels_scalar.c src/kernels_hdr_scalar.c src/tonemap.c \
           src/kernels_upscale_scalar.c

# Platform-specific SIMD kernels (SDR + HDR + upscale)
ifeq ($(UNAME_M),x86_64)
  LIB_SRCS += src/kernels_avx2.c src/kernels_hdr_avx2.c \
              src/kernels_upscale_avx2.c
else ifeq ($(UNAME_M),aarch64)
  LIB_SRCS += src/kernels_neon.c src/kernels_hdr_neon.c \
              src/kernels_upscale_neon.c
else ifeq ($(UNAME_M),arm64)
  LIB_SRCS += src/kernels_neon.c src/kernels_hdr_neon.c \
              src/kernels_upscale_neon.c
else ifeq ($(UNAME_M),riscv64)
  LIB_SRCS += src/kernels_rvv.c src/kernels_hdr_rvv.c \
              src/kernels_upscale_rvv.c
  # LTO link must re-process the RVV intrinsic IR with the V extension
  # available; otherwise lto1 errors out with "target specific builtin
  # not available".  Adding -march=rv64gcv to the link line satisfies it
  # without affecting non-LTO builds (where the link step doesn't touch
  # the IR at all).
  LINK_MARCH = -march=rv64gcv
endif

LIB_OBJS = $(LIB_SRCS:.c=.o)

# Test sources
TEST_SRCS = test/test_main.c test/test_validation.c test/test_correctness.c \
            test/test_patterns.c test/test_visual.c test/test_bench.c \
            test/test_hdr_validation.c test/test_hdr_correctness.c \
            test/test_hdr_bench.c test/test_swscale_bench.c \
            test/test_parity.c
TEST_OBJS = $(TEST_SRCS:.c=.o)

# Default target
.PHONY: all lib shared test bench bench-sdr bench-hdr bench-swscale visual asm fetch-samples clean pgo pgo-clean install

all: lib

lib: libfunnelcake.a

libfunnelcake.a: $(LIB_OBJS)
	$(AR) rcs $@ $^

# Shared library (built on demand, e.g. for `make install`). The library
# objects are already compiled with -fPIC, so they can be reused as-is.
shared: $(SHLIB)

$(SHLIB): $(LIB_OBJS)
	$(CC) $(SHLIB_LDFLAGS) $(LINK_MARCH) -o $@ $^ $(LDFLAGS)

# pkg-config file generated at build time so that downstream consumers
# can do `pkg-config --cflags --libs funnelcake`.
funnelcake.pc:
	@echo 'prefix=$(PREFIX)'                              >  $@
	@echo 'exec_prefix=$${prefix}'                        >> $@
	@echo 'libdir=$(LIBDIR)'                              >> $@
	@echo 'includedir=$(INCLUDEDIR)'                      >> $@
	@echo ''                                              >> $@
	@echo 'Name: funnelcake'                              >> $@
	@echo 'Description: SIMD YUV scaler with HDR/SDR tonemapping' >> $@
	@echo 'Version: $(VERSION)'                           >> $@
	@echo 'Cflags: -I$${includedir}'                      >> $@
	@echo 'Libs: -L$${libdir} -lfunnelcake'               >> $@
	@echo 'Libs.private: -lm'                             >> $@

install: lib shared funnelcake.pc
	$(INSTALL) -d $(DESTDIR)$(LIBDIR)
	$(INSTALL) -d $(DESTDIR)$(INCLUDEDIR)
	$(INSTALL) -d $(DESTDIR)$(PKGCONFDIR)
	$(INSTALL) -m 644 libfunnelcake.a $(DESTDIR)$(LIBDIR)/
	$(INSTALL) -m 755 $(SHLIB) $(DESTDIR)$(LIBDIR)/
	ln -sf $(SHLIB) $(DESTDIR)$(LIBDIR)/$(SHLIB_LINK)
	$(INSTALL) -m 644 include/funnelcake.h $(DESTDIR)$(INCLUDEDIR)/
	$(INSTALL) -m 644 funnelcake.pc $(DESTDIR)$(PKGCONFDIR)/

funnelcake_test: $(TEST_OBJS) libfunnelcake.a
	$(CC) $(TEST_CFLAGS) $(LINK_MARCH) $(EXTRA_LDFLAGS) -o $@ $(TEST_OBJS) -L. -lfunnelcake $(LDFLAGS) $(SWSCALE_TEST_LDFLAGS)

test: funnelcake_test
	./funnelcake_test

bench: funnelcake_test
	./funnelcake_test --bench

bench-sdr: funnelcake_test
	./funnelcake_test --bench-sdr

bench-hdr: funnelcake_test
	./funnelcake_test --bench-hdr

bench-swscale: funnelcake_test
	./funnelcake_test --bench-swscale

visual: funnelcake_test
	@mkdir -p output
	./funnelcake_test --visual

# --- Assembly inspection ---
# `make asm` emits annotated assembly for the AVX2 SIMD kernels next to the
# sources so optimization experiments can inspect codegen (register spills,
# intrinsic lowering) without rediscovering the compiler invocation. Built at
# -O3 with the same -mavx2/-mtune flags as the real objects but WITHOUT LTO:
# `-flto -S` emits pre-link IR, not the final per-function codegen we want to
# read. Honors TUNE, e.g. `make asm TUNE=native`.
ASM_CFLAGS = $(CFLAGS_BASE) $(LIB_OPT) $(TUNE_CFLAGS) -mavx2 -S -fverbose-asm
ASM_SRCS   = src/kernels_avx2.c src/kernels_upscale_avx2.c
ASM_OUT    = $(ASM_SRCS:.c=.S)

asm: $(ASM_OUT)

$(ASM_OUT): %.S: %.c
	$(CC) $(ASM_CFLAGS) -o $@ $<

# --- Sample HDR frames for visual testing ---
# Generates synthetic PQ-encoded 10-bit test frames using ffmpeg.
# Also downloads a small EXR reference image from the OpenEXR test suite.
# Output goes to test/samples/ (gitignored). Requires ffmpeg.
fetch-samples:
	@command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg is required for fetch-samples"; exit 1; }
	@command -v curl >/dev/null 2>&1 || { echo "Error: curl is required for fetch-samples"; exit 1; }
	@mkdir -p test/samples
	@echo ""
	@echo "=== Generating PQ-encoded 10-bit test frames ==="
	@echo "    (requires ffmpeg with zscale filter for proper PQ OETF)"
	@echo ""
	@echo "  SMPTE HD bars (1080p, PQ I010)..."
	@ffmpeg -y -f lavfi -i "smptehdbars=size=1920x1080:rate=1:duration=1" \
		-vf "zscale=tin=bt709:min=bt709:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020,format=yuv420p10le" \
		-f rawvideo -frames:v 1 \
		test/samples/bars_1080p_i010_pq.yuv 2>/dev/null \
		&& echo "    test/samples/bars_1080p_i010_pq.yuv" || echo "    (failed - zscale filter may not be available)"
	@echo "  Color gradients (1080p, PQ I010)..."
	@ffmpeg -y -f lavfi -i "gradients=size=1920x1080:rate=1:duration=1:nb_colors=6" \
		-vf "zscale=tin=bt709:min=bt709:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020,format=yuv420p10le" \
		-f rawvideo -frames:v 1 \
		test/samples/gradients_1080p_i010_pq.yuv 2>/dev/null \
		&& echo "    test/samples/gradients_1080p_i010_pq.yuv" || echo "    (failed)"
	@echo "  Mandelbrot (1080p, PQ I010)..."
	@ffmpeg -y -f lavfi -i "mandelbrot=size=1920x1080:rate=1:maxiter=500" \
		-vf "zscale=tin=bt709:min=bt709:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020,format=yuv420p10le" \
		-f rawvideo -frames:v 1 \
		test/samples/mandelbrot_1080p_i010_pq.yuv 2>/dev/null \
		&& echo "    test/samples/mandelbrot_1080p_i010_pq.yuv" || echo "    (failed)"
	@echo "  Test source v2 (1080p, PQ I010)..."
	@ffmpeg -y -f lavfi -i "testsrc2=size=1920x1080:rate=1:duration=1" \
		-vf "zscale=tin=bt709:min=bt709:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020,format=yuv420p10le" \
		-f rawvideo -frames:v 1 \
		test/samples/testsrc2_1080p_i010_pq.yuv 2>/dev/null \
		&& echo "    test/samples/testsrc2_1080p_i010_pq.yuv" || echo "    (failed)"
	@echo ""
	@echo "=== Downloading real-world HDR test content ==="
	@echo ""
	@echo "  haasn/hdr-tests colorbars (1080p HEVC HDR10, 258 KB) -> extracting 1 frame..."
	@curl -fsSL -o test/samples/hdr_colorbars.mp4 \
		"https://github.com/haasn/hdr-tests/raw/master/colorbars.mp4" 2>/dev/null \
		&& ffmpeg -y -i test/samples/hdr_colorbars.mp4 -frames:v 1 \
			-f rawvideo -pix_fmt yuv420p10le \
			test/samples/hdr_colorbars_1080p_i010_pq.yuv 2>/dev/null \
		&& rm -f test/samples/hdr_colorbars.mp4 \
		&& echo "    test/samples/hdr_colorbars_1080p_i010_pq.yuv (1080p PQ, from haasn/hdr-tests)" \
		|| { rm -f test/samples/hdr_colorbars.mp4; echo "    (skipped: download or conversion failed)"; }
	@echo "  haasn/hdr-tests snow-fades (1080p HEVC HDR10, 3.4 MB) -> extracting 1 frame..."
	@curl -fsSL -o test/samples/hdr_snow.mp4 \
		"https://github.com/haasn/hdr-tests/raw/master/snow-fades.mp4" 2>/dev/null \
		&& ffmpeg -y -i test/samples/hdr_snow.mp4 -frames:v 1 \
			-f rawvideo -pix_fmt yuv420p10le \
			test/samples/hdr_snow_i010_pq.yuv 2>/dev/null \
		&& rm -f test/samples/hdr_snow.mp4 \
		&& echo "    test/samples/hdr_snow_i010_pq.yuv (PQ, from haasn/hdr-tests)" \
		|| { rm -f test/samples/hdr_snow.mp4; echo "    (skipped: download or conversion failed)"; }
	@echo "  OpenEXR StillLife -> PQ I010 via zscale..."
	@curl -fsSL -o test/samples/StillLife.exr \
		"https://raw.githubusercontent.com/AcademySoftwareFoundation/openexr-images/main/ScanLines/StillLife.exr" 2>/dev/null \
		&& ffmpeg -y -i test/samples/StillLife.exr \
			-vf "zscale=tin=linear:min=bt709:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020,format=yuv420p10le" \
			-f rawvideo test/samples/stilllife_i010_pq.yuv 2>/dev/null \
		&& rm -f test/samples/StillLife.exr \
		&& echo "    test/samples/stilllife_i010_pq.yuv (1240x846, from OpenEXR, BSD license)" \
		|| { rm -f test/samples/StillLife.exr; echo "    (skipped: download or conversion failed)"; }
	@echo ""
	@echo "=== Converting local JPEG photographs to PQ I010 ==="
	@echo ""
	@for src in test/samples/*.jpg; do \
		[ -f "$$src" ] || continue; \
		base=$$(basename "$$src" .jpg | sed 's/[^a-zA-Z0-9]/_/g' | tr 'A-Z' 'a-z'); \
		out="test/samples/$${base}_i010_pq.yuv"; \
		if [ -f "$$out" ]; then continue; fi; \
		dims=$$(ffmpeg -i "$$src" 2>&1 | grep -oP '\d+x\d+' | head -1); \
		w=$$(echo "$$dims" | cut -dx -f1); \
		h=$$(echo "$$dims" | cut -dx -f2); \
		h=$$(( (h / 2) * 2 )); \
		echo "  $$src -> $$out ($${w}x$${h})..."; \
		ffmpeg -y -i "$$src" \
			-vf "crop=$${w}:$${h},zscale=tin=bt709:min=bt709:pin=bt709:t=smpte2084:m=bt2020nc:p=bt2020,format=yuv420p10le" \
			-f rawvideo -frames:v 1 "$$out" 2>/dev/null \
			&& echo "    $$out" || echo "    (failed)"; \
	done
	@echo ""
	@echo "=== Done. Run 'make visual' to generate output PNGs. ==="
	@echo "    Sample frames are in test/samples/ (gitignored)."
	@echo ""

clean:
	rm -f $(LIB_OBJS) $(TEST_OBJS) libfunnelcake.a funnelcake_test
	rm -f libfunnelcake.so libfunnelcake.so.* libfunnelcake.*.dylib libfunnelcake.dylib
	rm -f funnelcake.pc
	rm -f src/*.gcda test/*.gcda
	rm -f src/*.profraw default.profdata
	rm -f src/*.S
	rm -rf output/*

# --- Profile-Guided Optimization ---
# Usage: make pgo [TUNE=znver2] [LTO=1]
# This compiles with instrumentation, runs benchmarks to collect profile data,
# then recompiles the LIBRARY ONLY using the profile for optimal branch layout
# and inlining. Test code is compiled normally (no PGO).
#
# Each instrumented run is given an explicit LLVM_PROFILE_FILE so the bench
# and test profiles don't overwrite each other on clang (clang's default
# `default.profraw` would be truncated by the second run, discarding the
# bench data which is what we actually want to optimize for).
pgo: pgo-clean
	@echo "=== PGO Step 1: Compile with instrumentation ==="
	$(MAKE) clean
	$(MAKE) funnelcake_test LIB_OPT="-O3 -fprofile-generate" EXTRA_LDFLAGS="-fprofile-generate"
	@echo "=== PGO Step 2: Run benchmarks to collect profile ==="
	LLVM_PROFILE_FILE="pgo-bench.profraw" ./funnelcake_test --bench
	LLVM_PROFILE_FILE="pgo-tests.profraw" ./funnelcake_test
ifeq ($(CC_FAMILY_IS_CLANG),1)
	@echo "=== PGO Step 2a: Merge raw profiles (clang) ==="
	$(LLVM_PROFDATA) merge -output=default.profdata \
	    pgo-bench.profraw pgo-tests.profraw
endif
	@echo "=== PGO Step 3: Recompile library with profile data ==="
	rm -f $(LIB_OBJS) libfunnelcake.a funnelcake_test
	$(MAKE) funnelcake_test LIB_OPT="-O3 $(PGO_USE_FLAGS)"
	@echo "=== PGO complete. Run 'make bench' to see results. ==="

pgo-clean:
	rm -f $(LIB_SRCS:.c=.gcda) $(TEST_SRCS:.c=.gcda)
	rm -f *.profraw default.profdata
	rm -f pgo-bench.profraw pgo-tests.profraw

# --- Per-file flag overrides ---
src/kernels_scalar.o: src/kernels_scalar.c
	$(CC) $(LIB_CFLAGS) $(SCALAR_CFLAGS) -c -o $@ $<

src/kernels_avx2.o: src/kernels_avx2.c
	$(CC) $(LIB_CFLAGS) -mavx2 -c -o $@ $<

src/kernels_neon.o: src/kernels_neon.c
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

# Library source files use LIB_CFLAGS (O3)
src/funnelcake.o: src/funnelcake.c
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

src/log.o: src/log.c
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

src/detect.o: src/detect.c
	$(CC) $(DETECT_CFLAGS) -c -o $@ $<

# HDR library sources use LIB_CFLAGS (O3)
src/funnelcake_hdr.o: src/funnelcake_hdr.c
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

src/tonemap.o: src/tonemap.c
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

src/kernels_hdr_scalar.o: src/kernels_hdr_scalar.c
	$(CC) $(LIB_CFLAGS) $(SCALAR_CFLAGS) -c -o $@ $<

src/kernels_hdr_avx2.o: src/kernels_hdr_avx2.c
	$(CC) $(LIB_CFLAGS) -mavx2 -c -o $@ $<

src/kernels_hdr_neon.o: src/kernels_hdr_neon.c
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

# Upscale kernels
src/kernels_upscale_scalar.o: src/kernels_upscale_scalar.c
	$(CC) $(LIB_CFLAGS) $(SCALAR_CFLAGS) -c -o $@ $<

src/kernels_upscale_avx2.o: src/kernels_upscale_avx2.c
	$(CC) $(LIB_CFLAGS) -mavx2 -c -o $@ $<

src/kernels_upscale_neon.o: src/kernels_upscale_neon.c
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

# RVV kernels (riscv64). -march=rv64gcv enables full RVV 1.0 (the V
# extension); the kernels are written vector-length-agnostic so they run on
# any V-capable chip.
src/kernels_rvv.o: src/kernels_rvv.c
	$(CC) $(LIB_CFLAGS) -march=rv64gcv -c -o $@ $<

src/kernels_hdr_rvv.o: src/kernels_hdr_rvv.c
	$(CC) $(LIB_CFLAGS) -march=rv64gcv -c -o $@ $<

src/kernels_upscale_rvv.o: src/kernels_upscale_rvv.c
	$(CC) $(LIB_CFLAGS) -march=rv64gcv -c -o $@ $<

# Test source files use TEST_CFLAGS (O2)
test/%.o: test/%.c
	$(CC) $(TEST_CFLAGS) -c -o $@ $<

# swscale bench needs extra include paths and -DHAVE_LIBSWSCALE
test/test_swscale_bench.o: test/test_swscale_bench.c
	$(CC) $(TEST_CFLAGS) $(SWSCALE_TEST_CFLAGS) -c -o $@ $<

# Header dependencies (conservative: rebuild all on header change)
$(LIB_OBJS): include/funnelcake.h src/internal.h src/log.h src/detect.h src/tonemap.h src/upscale_chunk.h
$(TEST_OBJS): include/funnelcake.h test/test_main.h test/test_patterns.h
