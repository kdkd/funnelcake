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
        } else if (strcmp(argv[i], "--bench-sdr") == 0) {
            opts.run_bench_sdr = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opts.bench_filter = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--bench-hdr") == 0) {
            opts.run_bench_hdr = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opts.bench_filter = argv[i + 1];
                i++;
            }
        } else if (strcmp(argv[i], "--visual") == 0) {
            opts.run_visual = 1;
        }
    }

    if (opts.run_visual) {
        printf("=== Visual tests ===\n");
        run_visual_tests();
    }

    if (!opts.run_bench && !opts.run_bench_sdr && !opts.run_bench_hdr && !opts.run_visual) {
        /* Default: run all validation and correctness tests */
        printf("=== Validation tests ===\n");
        run_validation_tests();

        printf("\n=== Correctness tests ===\n");
        run_correctness_tests();

        printf("\n=== HDR Validation tests ===\n");
        run_hdr_validation_tests();

        printf("\n=== HDR Correctness tests ===\n");
        run_hdr_correctness_tests();
    }

    if (opts.run_bench || opts.run_bench_sdr) {
        printf("\n=== SDR Benchmarks ===\n\n");
        run_bench_tests(opts.bench_filter);
    }

    if (opts.run_bench || opts.run_bench_hdr) {
        printf("\n=== HDR Benchmarks ===\n\n");
        run_hdr_bench_tests(opts.bench_filter);
    }

    /* Print summary */
    printf("\n--- Results: %d passed, %d failed, %d skipped ---\n",
           g_results.passed, g_results.failed, g_results.skipped);

    return (g_results.failed > 0) ? 1 : 0;
}
