CC ?= clang
AR ?= ar
CFLAGS_BASE = -std=c11 -D_POSIX_C_SOURCE=200112L -Wall -Wextra -Werror -Iinclude -Isrc
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

# Library sources (always compiled)
LIB_SRCS = src/funnelcake.c src/log.c src/detect.c src/kernels_scalar.c

# Platform-specific SIMD kernel
ifeq ($(UNAME_M),x86_64)
  LIB_SRCS += src/kernels_avx2.c
else ifeq ($(UNAME_M),aarch64)
  LIB_SRCS += src/kernels_neon.c
else ifeq ($(UNAME_M),arm64)
  LIB_SRCS += src/kernels_neon.c
endif

LIB_OBJS = $(LIB_SRCS:.c=.o)

# Test sources
TEST_SRCS = test/test_main.c test/test_validation.c test/test_correctness.c \
            test/test_patterns.c test/test_visual.c test/test_bench.c
TEST_OBJS = $(TEST_SRCS:.c=.o)

# Default target
.PHONY: all lib test bench visual clean pgo pgo-clean

all: lib

lib: libfunnelcake.a

libfunnelcake.a: $(LIB_OBJS)
	$(AR) rcs $@ $^

funnelcake_test: $(TEST_OBJS) libfunnelcake.a
	$(CC) $(TEST_CFLAGS) $(LTO_LDFLAGS) -o $@ $(TEST_OBJS) -L. -lfunnelcake $(LDFLAGS)

test: funnelcake_test
	./funnelcake_test

bench: funnelcake_test
	./funnelcake_test --bench

visual: funnelcake_test
	@mkdir -p output
	./funnelcake_test --visual

clean:
	rm -f $(LIB_OBJS) $(TEST_OBJS) libfunnelcake.a funnelcake_test
	rm -f $(LIB_SRCS:.c=.gcda) $(TEST_SRCS:.c=.gcda)
	rm -f $(LIB_SRCS:.c=.profraw) default.profdata
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

# Test source files use TEST_CFLAGS (O2)
test/%.o: test/%.c
	$(CC) $(TEST_CFLAGS) -c -o $@ $<

# Header dependencies (conservative: rebuild all on header change)
$(LIB_OBJS): include/funnelcake.h src/internal.h src/log.h src/detect.h
$(TEST_OBJS): include/funnelcake.h test/test_main.h test/test_patterns.h
