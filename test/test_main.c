/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "test_main.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
 * Global test results
 * -------------------------------------------------------------------------- */

test_results_t g_results = { 0, 0, 0 };

bench_comparison_t g_bench_comparison[BENCH_MAX_CONFIGS];
int                g_bench_comparison_count = 0;

void print_bench_comparison_table(void)
{
    if (g_bench_comparison_count == 0) return;

    int has_swscale = 0;
    for (int i = 0; i < g_bench_comparison_count; i++)
        if (g_bench_comparison[i].swscale_indep_med > 0) { has_swscale = 1; break; }
    if (!has_swscale) return;

    printf("\n  %-44s %12s %12s %12s %10s %10s\n",
           "Workload", "funnelcake", "sws indep", "sws cascade",
           "vs indep", "vs cascade");
    printf("  %-44s %12s %12s %12s %10s %10s\n",
           "--------------------------------------------",
           "------------", "------------", "------------",
           "----------", "----------");

    for (int i = 0; i < g_bench_comparison_count; i++) {
        bench_comparison_t *c = &g_bench_comparison[i];
        if (c->funnelcake_med <= 0 || c->swscale_indep_med <= 0) continue;

        char fc_str[16], si_str[16], sc_str[16];
        char speedup_indep[16], speedup_cascade[16];

        snprintf(fc_str, sizeof(fc_str), "%.0f us", c->funnelcake_med);
        snprintf(si_str, sizeof(si_str), "%.0f us", c->swscale_indep_med);

        if (c->swscale_cascade_med > 0) {
            snprintf(sc_str, sizeof(sc_str), "%.0f us", c->swscale_cascade_med);
            snprintf(speedup_cascade, sizeof(speedup_cascade), "%.1fx",
                     c->swscale_cascade_med / c->funnelcake_med);
        } else {
            snprintf(sc_str, sizeof(sc_str), "-");
            snprintf(speedup_cascade, sizeof(speedup_cascade), "-");
        }

        snprintf(speedup_indep, sizeof(speedup_indep), "%.1fx",
                 c->swscale_indep_med / c->funnelcake_med);

        printf("  %-44s %12s %12s %12s %10s %10s\n",
               c->label, fc_str, si_str, sc_str,
               speedup_indep, speedup_cascade);
    }
    printf("\n");
}

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
        } else if (strcmp(argv[i], "--bench-swscale") == 0) {
            opts.run_bench_swscale = 1;
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

    if (!opts.run_bench && !opts.run_bench_sdr && !opts.run_bench_hdr
        && !opts.run_bench_swscale && !opts.run_visual) {
        /* Default: run all validation and correctness tests */
        printf("=== Validation tests ===\n");
        run_validation_tests();

        printf("\n=== Correctness tests ===\n");
        run_correctness_tests();

        printf("\n=== HDR Validation tests ===\n");
        run_hdr_validation_tests();

        printf("\n=== HDR Correctness tests ===\n");
        run_hdr_correctness_tests();

        printf("\n=== Parity tests (scalar vs SIMD) ===\n");
        run_parity_tests();
    }

    if (opts.run_bench || opts.run_bench_sdr) {
        printf("\n=== SDR Benchmarks ===\n\n");
        run_bench_tests(opts.bench_filter);

        printf("\n=== libswscale Comparison ===\n\n");
        run_swscale_bench_tests(opts.bench_filter);

        print_bench_comparison_table();
    }

    if (opts.run_bench_swscale) {
        printf("\n=== libswscale Comparison ===\n\n");
        run_swscale_bench_tests(opts.bench_filter);
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
