#include "test_main.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
 * Global test results
 * -------------------------------------------------------------------------- */

test_results_t g_results = { 0, 0, 0 };

/* --------------------------------------------------------------------------
 * main
 * -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    test_options_t opts;
    memset(&opts, 0, sizeof(opts));

    /* Parse command-line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            opts.run_bench = 1;
            /* Optional filter argument: next arg that doesn't start with '--' */
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opts.bench_filter = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--visual") == 0) {
            opts.run_visual = 1;
        }
    }

    if (opts.run_bench) {
        printf("=== Benchmark tests ===\n");
        run_bench_tests(opts.bench_filter);
    } else if (opts.run_visual) {
        printf("=== Visual tests ===\n");
        run_visual_tests();
    } else {
        printf("=== Validation tests ===\n");
        run_validation_tests();

        printf("\n=== Correctness tests ===\n");
        run_correctness_tests();
    }

    /* Print summary */
    printf("\n--- Results: %d passed, %d failed, %d skipped ---\n",
           g_results.passed, g_results.failed, g_results.skipped);

    return (g_results.failed > 0) ? 1 : 0;
}
