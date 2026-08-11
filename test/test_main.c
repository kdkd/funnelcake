/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

/* sysctlbyname needs the BSD types that strict _POSIX_C_SOURCE hides */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#endif

#include "test_main.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "detect.h"

#include <sys/utsname.h>

#if defined(__x86_64__)
#include <cpuid.h>
int fused_avx512_compiled(void);   /* internal.h pulls too much in here */
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

/* --------------------------------------------------------------------------
 * Benchmark system info header (best effort, no exotic OS hooks)
 * -------------------------------------------------------------------------- */

static void bench_rtrim(char *s)
{
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\n'))
        s[--n] = '\0';
}

static void bench_cpu_name(char *out, size_t out_size)
{
    snprintf(out, out_size, "unknown");

#if defined(__x86_64__)
    /* CPUID brand string - works on any OS */
    unsigned int a, b, c, d;
    if (__get_cpuid(0x80000000u, &a, &b, &c, &d) && a >= 0x80000004u) {
        char brand[49];
        unsigned int *p = (unsigned int *)brand;
        for (unsigned int leaf = 0; leaf < 3; leaf++) {
            __get_cpuid(0x80000002u + leaf, &a, &b, &c, &d);
            *p++ = a; *p++ = b; *p++ = c; *p++ = d;
        }
        brand[48] = '\0';
        const char *s = brand;
        while (*s == ' ') s++;
        snprintf(out, out_size, "%s", s);
        return;
    }
#elif defined(__APPLE__)
    size_t len = out_size;
    if (sysctlbyname("machdep.cpu.brand_string", out, &len, NULL, 0) == 0)
        return;
    snprintf(out, out_size, "unknown");
#elif defined(__linux__)
    /* "model name" covers riscv and some arm; "Hardware" some others */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0 ||
                strncmp(line, "Hardware", 8) == 0) {
                const char *colon = strchr(line, ':');
                if (colon) {
                    colon++;
                    while (*colon == ' ' || *colon == '\t') colon++;
                    snprintf(out, out_size, "%s", colon);
                    out[strcspn(out, "\n")] = '\0';
                    fclose(f);
                    return;
                }
            }
        }
        fclose(f);
    }
    /* Board name beats "unknown" (e.g. Raspberry Pi, Orange Pi) */
    f = fopen("/proc/device-tree/model", "r");
    if (f) {
        size_t n = fread(out, 1, out_size - 1, f);
        out[n] = '\0';
        fclose(f);
        if (n > 0) return;
        snprintf(out, out_size, "unknown");
    }
#endif
}

static void print_bench_system_info(void)
{
#if defined(__x86_64__)
    const char *platform = "x86_64";
#elif defined(__aarch64__)
    const char *platform = "aarch64";
#elif defined(__riscv) && (__riscv_xlen == 64)
    const char *platform = "riscv64";
#else
    const char *platform = "unknown";
#endif

    char cpu[256];
    bench_cpu_name(cpu, sizeof(cpu));
    bench_rtrim(cpu);   /* CPUID brand strings carry trailing padding */

    /* OS: uname is POSIX; prefer the friendlier names where one file
     * read or sysctl gets them. */
    char os[256] = "unknown";
    struct utsname un;
    if (uname(&un) == 0)
        snprintf(os, sizeof(os), "%s %s", un.sysname, un.release);
#if defined(__APPLE__)
    {
        char ver[64];
        size_t len = sizeof(ver);
        if (sysctlbyname("kern.osproductversion", ver, &len, NULL, 0) == 0)
            snprintf(os, sizeof(os), "macOS %s", ver);
    }
#elif defined(__linux__)
    {
        FILE *f = fopen("/etc/os-release", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "PRETTY_NAME=\"", 13) == 0) {
                    char *val = line + 13;
                    val[strcspn(val, "\"")] = '\0';
                    snprintf(os, sizeof(os), "%.180s (%.64s)", val,
                             uname(&un) == 0 ? un.release : "?");
                    break;
                }
            }
            fclose(f);
        }
    }
#endif

    /* Compiler: compile-time macros, so this reports whatever built the
     * test binary (and the library - they build together). */
#if defined(__clang__)
    const char *cc_name = "clang " __clang_version__;
#elif defined(__GNUC__)
    const char *cc_name = "gcc " __VERSION__;
#else
    const char *cc_name = "unknown";
#endif

    /* Mirrors the dispatchers' kernel selection (env overrides like
     * FUNNELCAKE_FORCE_SCALAR / FUNNELCAKE_NO_AVX512 are reflected
     * because they act on the caps themselves). */
    const fused_cpu_caps_t *caps = fused_detect_cpu();
    const char *kernels = "scalar";
#if defined(__x86_64__)
    if (caps->has_avx512 && fused_avx512_compiled()) kernels = "AVX-512";
    else if (caps->has_avx2)                         kernels = "AVX2";
#elif defined(__aarch64__)
    if (caps->has_neon) kernels = "NEON";
#elif defined(__riscv) && (__riscv_xlen == 64)
    if (caps->has_rvv) kernels = "RVV";
#endif

    printf("Platform: %s | CPU: %s | Kernels: %s\n", platform, cpu, kernels);
    printf("OS: %s | Compiler: %s\n", os, cc_name);
}

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
 * Usage
 * -------------------------------------------------------------------------- */

static void print_usage(FILE *out, const char *argv0)
{
    fprintf(out,
"Usage: %s [option]...\n"
"\n"
"With no options, runs the full test suite: validation, correctness, HDR,\n"
"tone mapping, and scalar-vs-SIMD parity. It exits with status 1 if any\n"
"test fails, or 0 if they all pass.\n"
"\n"
"Test options:\n"
"  --visual                 Write PNG/MOV renderings of each scale step to\n"
"                           output/. Needs ffmpeg on PATH and an existing\n"
"                           output/ directory. Use make visual, which\n"
"                           creates it for you.\n"
"\n"
"Benchmark options (each accepts an optional workload filter):\n"
"  --bench [filter]         SDR, libswscale comparison, and HDR benchmarks\n"
"  --bench-sdr [filter]     SDR benchmarks, plus the libswscale comparison\n"
"  --bench-hdr [filter]     HDR benchmarks (10-bit scaling and tone mapping)\n"
"  --bench-swscale [filter] libswscale comparison only\n"
"  --skip-bench-swscale     Skip the libswscale comparison. Useful when the\n"
"                           build lacks libswscale or for a shorter A/B run.\n"
"\n"
"  -h, --help               This message\n"
"\n"
"The filter is a plain substring match against the workload label. For\n"
"example, --bench-sdr 1920x1080 runs every 1080p SDR row, while\n"
"--bench-sdr down:2x runs the 2x downscale rows at every resolution. With\n"
"no filter, a benchmark option runs every workload in its group.\n"
"\n"
"Environment:\n"
"  FUNNELCAKE_FORCE_SCALAR    Skip SIMD detection and use the scalar kernels.\n"
"  FUNNELCAKE_NO_AVX512       On AVX-512 hardware, fall back to AVX2 for a\n"
"                             same-build comparison of the two kernel sets.\n"
"  FUNNELCAKE_SWSCALE_VERIFY  In the libswscale comparison, print each step's\n"
"                             row count and Y-plane checksum.\n"
"\n"
"Common make targets: test, bench, bench-sdr, bench-hdr, bench-swscale,\n"
"and visual.\n",
    argv0);
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
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(stdout, argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--bench") == 0) {
            opts.run_bench = 1;
            /* Optional filter argument: next arg that does not start with a dash */
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
        } else if (strcmp(argv[i], "--skip-bench-swscale") == 0) {
            opts.skip_bench_swscale = 1;
        } else if (strcmp(argv[i], "--visual") == 0) {
            opts.run_visual = 1;
        } else {
            /* Reject typos instead of silently running the full test suite. */
            fprintf(stderr, "%s: unrecognized option '%s'\n\n", argv[0], argv[i]);
            print_usage(stderr, argv[0]);
            return 2;
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

        printf("\n=== Tone mapping tests ===\n");
        run_tonemap_tests();

        printf("\n=== Parity tests (scalar vs SIMD) ===\n");
        run_parity_tests();
    }

    if (opts.run_bench || opts.run_bench_sdr || opts.run_bench_hdr ||
        opts.run_bench_swscale) {
        print_bench_system_info();
    }

    if (opts.run_bench || opts.run_bench_sdr) {
        printf("\n=== SDR Benchmarks ===\n\n");
        run_bench_tests(opts.bench_filter);

        if (!opts.skip_bench_swscale) {
            printf("\n=== libswscale Comparison ===\n\n");
            run_swscale_bench_tests(opts.bench_filter);

            print_bench_comparison_table();
        }
    }

    if (opts.run_bench_swscale && !opts.skip_bench_swscale) {
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
