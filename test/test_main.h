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
void run_visual_tests(void);
void run_bench_tests(const char *filter);

/* --------------------------------------------------------------------------
 * Options parsed from command-line arguments
 * -------------------------------------------------------------------------- */

typedef struct {
    int         run_bench;
    int         run_visual;
    const char *bench_filter;
} test_options_t;

#endif /* TEST_MAIN_H */
