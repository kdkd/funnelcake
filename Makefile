CC ?= clang
AR ?= ar
CFLAGS_BASE = -std=c11 -D_POSIX_C_SOURCE=200112L -Wall -Wextra -Werror -fPIC -Iinclude -Isrc
LDFLAGS = -lm

# Optimization level: O3 for library, O2 for tests (tests don't need aggressive opts)
LIB_OPT = -O3
TEST_OPT = -O2

# Platform detection
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

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
LTO ?= 0

ifeq ($(LTO),1)
  LTO_CFLAGS = -flto
  LTO_LDFLAGS = -flto
else
  LTO_CFLAGS =
  LTO_LDFLAGS =
endif

# Assemble final CFLAGS
LIB_CFLAGS = $(CFLAGS_BASE) $(LIB_OPT) $(TUNE_CFLAGS) $(LTO_CFLAGS)
TEST_CFLAGS = $(CFLAGS_BASE) $(TEST_OPT) $(TUNE_CFLAGS) $(LTO_CFLAGS)

# detect.c runs exactly once at startup. Never compile it with PGO flags.
# -fprofile-generate changes register allocation inside the xgetbv inline asm,
# which can cause "=A" to capture rdx (high 32 bits) instead of rax (low 32
# bits), recording a false "XCR0 bits not set" result in the profile data.
# The -fprofile-use pass then treats the AVX2 assignment as dead code.
# This variable intentionally excludes LIB_OPT (where -fprofile-* lives).
DETECT_CFLAGS = $(CFLAGS_BASE) -O2 $(TUNE_CFLAGS)

# --- Optional: libswscale for comparison benchmarks ---
# Try pkg-config first, then probe common paths
SWSCALE_CFLAGS := $(shell pkg-config --cflags libswscale libavutil 2>/dev/null)
SWSCALE_LIBS   := $(shell pkg-config --libs libswscale libavutil 2>/dev/null)

ifeq ($(SWSCALE_LIBS),)
  # macOS Homebrew
  ifneq ($(wildcard /opt/homebrew/include/libswscale/swscale.h),)
    SWSCALE_CFLAGS := -I/opt/homebrew/include
    SWSCALE_LIBS   := -L/opt/homebrew/lib -lswscale -lavutil
  endif
endif
ifeq ($(SWSCALE_LIBS),)
  # Linux x86_64
  ifneq ($(wildcard /usr/include/x86_64-linux-gnu/libswscale/swscale.h),)
    SWSCALE_CFLAGS := -I/usr/include/x86_64-linux-gnu
    SWSCALE_LIBS   := -lswscale -lavutil
  endif
endif
ifeq ($(SWSCALE_LIBS),)
  # Linux aarch64
  ifneq ($(wildcard /usr/include/aarch64-linux-gnu/libswscale/swscale.h),)
    SWSCALE_CFLAGS := -I/usr/include/aarch64-linux-gnu
    SWSCALE_LIBS   := -lswscale -lavutil
  endif
endif

ifneq ($(SWSCALE_LIBS),)
  SWSCALE_TEST_CFLAGS = $(SWSCALE_CFLAGS) -DHAVE_LIBSWSCALE
  SWSCALE_TEST_LDFLAGS = $(SWSCALE_LIBS)
endif

# Library sources (always compiled)
LIB_SRCS = src/funnelcake.c src/funnelcake_hdr.c src/log.c src/detect.c \
           src/kernels_scalar.c src/kernels_hdr_scalar.c src/tonemap.c

# Platform-specific SIMD kernels (SDR + HDR)
ifeq ($(UNAME_M),x86_64)
  LIB_SRCS += src/kernels_avx2.c src/kernels_hdr_avx2.c
else ifeq ($(UNAME_M),aarch64)
  LIB_SRCS += src/kernels_neon.c src/kernels_hdr_neon.c
else ifeq ($(UNAME_M),arm64)
  LIB_SRCS += src/kernels_neon.c src/kernels_hdr_neon.c
endif

LIB_OBJS = $(LIB_SRCS:.c=.o)

# Test sources
TEST_SRCS = test/test_main.c test/test_validation.c test/test_correctness.c \
            test/test_patterns.c test/test_visual.c test/test_bench.c \
            test/test_hdr_validation.c test/test_hdr_correctness.c \
            test/test_hdr_bench.c test/test_swscale_bench.c
TEST_OBJS = $(TEST_SRCS:.c=.o)

# Default target
.PHONY: all lib test bench bench-sdr bench-hdr bench-swscale visual fetch-samples clean pgo pgo-clean

all: lib

lib: libfunnelcake.a

libfunnelcake.a: $(LIB_OBJS)
	$(AR) rcs $@ $^

funnelcake_test: $(TEST_OBJS) libfunnelcake.a
	$(CC) $(TEST_CFLAGS) $(LTO_LDFLAGS) -o $@ $(TEST_OBJS) -L. -lfunnelcake $(LDFLAGS) $(SWSCALE_TEST_LDFLAGS)

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
		&& echo "    test/samples/bars_1080p_i010_pq.yuv" || echo "    (failed — zscale filter may not be available)"
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
	@echo "  haasn/hdr-tests colorbars (1080p HEVC HDR10, 258 KB) → extracting 1 frame..."
	@curl -fsSL -o test/samples/hdr_colorbars.mp4 \
		"https://github.com/haasn/hdr-tests/raw/master/colorbars.mp4" 2>/dev/null \
		&& ffmpeg -y -i test/samples/hdr_colorbars.mp4 -frames:v 1 \
			-f rawvideo -pix_fmt yuv420p10le \
			test/samples/hdr_colorbars_1080p_i010_pq.yuv 2>/dev/null \
		&& rm -f test/samples/hdr_colorbars.mp4 \
		&& echo "    test/samples/hdr_colorbars_1080p_i010_pq.yuv (1080p PQ, from haasn/hdr-tests)" \
		|| { rm -f test/samples/hdr_colorbars.mp4; echo "    (skipped: download or conversion failed)"; }
	@echo "  haasn/hdr-tests snow-fades (1080p HEVC HDR10, 3.4 MB) → extracting 1 frame..."
	@curl -fsSL -o test/samples/hdr_snow.mp4 \
		"https://github.com/haasn/hdr-tests/raw/master/snow-fades.mp4" 2>/dev/null \
		&& ffmpeg -y -i test/samples/hdr_snow.mp4 -frames:v 1 \
			-f rawvideo -pix_fmt yuv420p10le \
			test/samples/hdr_snow_i010_pq.yuv 2>/dev/null \
		&& rm -f test/samples/hdr_snow.mp4 \
		&& echo "    test/samples/hdr_snow_i010_pq.yuv (PQ, from haasn/hdr-tests)" \
		|| { rm -f test/samples/hdr_snow.mp4; echo "    (skipped: download or conversion failed)"; }
	@echo "  OpenEXR StillLife → PQ I010 via zscale..."
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
		echo "  $$src → $$out ($${w}x$${h})..."; \
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
	rm -f src/*.gcda test/*.gcda
	rm -f src/*.profraw default.profdata
	rm -rf output/*

# --- Profile-Guided Optimization ---
# Usage: make pgo [TUNE=znver2] [LTO=1]
# This compiles with instrumentation, runs benchmarks to collect profile data,
# then recompiles the LIBRARY ONLY using the profile for optimal branch layout
# and inlining. Test code is compiled normally (no PGO).
pgo: pgo-clean
	@echo "=== PGO Step 1: Compile with instrumentation ==="
	$(MAKE) clean
	$(MAKE) funnelcake_test LIB_OPT="-O3 -fprofile-generate" LTO_LDFLAGS="-fprofile-generate"
	@echo "=== PGO Step 2: Run benchmarks to collect profile ==="
	./funnelcake_test --bench
	./funnelcake_test
	@echo "=== PGO Step 3: Recompile library with profile data ==="
	rm -f $(LIB_OBJS) libfunnelcake.a funnelcake_test
	$(MAKE) funnelcake_test LIB_OPT="-O3 -fprofile-use -Wno-missing-profile"
	@echo "=== PGO complete. Run 'make bench' to see results. ==="

pgo-clean:
	rm -f $(LIB_SRCS:.c=.gcda) $(TEST_SRCS:.c=.gcda)
	rm -f *.profraw default.profdata

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

# Test source files use TEST_CFLAGS (O2)
test/%.o: test/%.c
	$(CC) $(TEST_CFLAGS) -c -o $@ $<

# swscale bench needs extra include paths and -DHAVE_LIBSWSCALE
test/test_swscale_bench.o: test/test_swscale_bench.c
	$(CC) $(TEST_CFLAGS) $(SWSCALE_TEST_CFLAGS) -c -o $@ $<

# Header dependencies (conservative: rebuild all on header change)
$(LIB_OBJS): include/funnelcake.h src/internal.h src/log.h src/detect.h src/tonemap.h
$(TEST_OBJS): include/funnelcake.h test/test_main.h test/test_patterns.h
