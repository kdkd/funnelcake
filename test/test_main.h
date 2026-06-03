/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <stdio.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Test result counters
 * -------------------------------------------------------------------------- */

typedef struct {
    int passed;
    int failed;
    int skipped;
} test_results_t;

extern test_results_t g_results;

/* --------------------------------------------------------------------------
 * Assertion macros
 * -------------------------------------------------------------------------- */

/* TEST_ASSERT: check a boolean condition */
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("  FAIL [%s:%d] %s\n", __func__, __LINE__, (msg)); \
            g_results.failed++; \
            return; \
        } \
    } while (0)

/* TEST_ASSERT_EQ: check two integer values are equal */
#define TEST_ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            printf("  FAIL [%s:%d] %s: got %d, expected %d\n", \
                   __func__, __LINE__, (msg), (int)(a), (int)(b)); \
            g_results.failed++; \
            return; \
        } \
    } while (0)

/* TEST_PASS: record a successful test */
#define TEST_PASS() \
    do { \
        g_results.passed++; \
    } while (0)

/* TEST_SKIP: skip a test with a reason message */
#define TEST_SKIP(msg) \
    do { \
        printf("  SKIP [%s:%d] %s\n", __func__, __LINE__, (msg)); \
        g_results.skipped++; \
        return; \
    } while (0)

/* RUN_TEST: print test name and invoke the function */
#define RUN_TEST(fn) \
    do { \
        printf("  %-50s", #fn); \
        fflush(stdout); \
        fn(); \
        /* If test passed (no FAIL/SKIP printed), we already incremented \
         * passed inside the test via TEST_PASS(). Print a newline. */ \
        printf("\n"); \
    } while (0)

/* --------------------------------------------------------------------------
 * Test suite entry points (defined in their respective .c files)
 * -------------------------------------------------------------------------- */

void run_validation_tests(void);
void run_correctness_tests(void);
void run_parity_tests(void);
void run_visual_tests(void);
void run_bench_tests(const char *filter);
void run_hdr_validation_tests(void);
void run_hdr_correctness_tests(void);
void run_hdr_bench_tests(const char *filter);
void run_swscale_bench_tests(const char *filter);

/* Shared benchmark results for comparison table */
#define BENCH_MAX_CONFIGS 32
typedef struct {
    const char *label;
    double      funnelcake_med;     /* funnelcake median µs (0 = not run) */
    double      swscale_indep_med;  /* swscale independent median µs */
    double      swscale_cascade_med;/* swscale cascaded median µs */
} bench_comparison_t;

extern bench_comparison_t g_bench_comparison[BENCH_MAX_CONFIGS];
extern int                g_bench_comparison_count;

void print_bench_comparison_table(void);

/* --------------------------------------------------------------------------
 * Options parsed from command-line arguments
 * -------------------------------------------------------------------------- */

typedef struct {
    int         run_bench;         /* --bench: runs SDR + HDR + swscale benches */
    int         run_bench_sdr;     /* --bench-sdr: SDR benchmarks only */
    int         run_bench_hdr;     /* --bench-hdr: HDR benchmarks only */
    int         run_bench_swscale; /* --bench-swscale: libswscale comparison only */
    int         run_visual;
    const char *bench_filter;
} test_options_t;

#endif /* TEST_MAIN_H */
