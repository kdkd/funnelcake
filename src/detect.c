#include "detect.h"
#include <stdint.h>

/* Static cache — zero-initialised at startup */
static fused_cpu_caps_t g_caps = {0, 0};
static int              g_detected = 0;


/* --------------------------------------------------------------------------
 * Platform-specific detection helpers
 * -------------------------------------------------------------------------- */

#if defined(__x86_64__)

#include <cpuid.h>

/*
 * Detect AVX2 on x86_64.
 *
 * A full AVX2 check requires three things:
 *  1. The CPU advertises OSXSAVE (cpuid eax=1, ecx bit 27) — OS has enabled
 *     the XSAVE instruction.
 *  2. The CPU advertises AVX support (cpuid eax=1, ecx bit 28).
 *  3. The OS has enabled YMM state in XCR0 (xgetbv xcr0, bits 1 and 2 set).
 *  4. The CPU advertises AVX2 (cpuid eax=7 ecx=0, ebx bit 5).
 *
 * Skipping any of these can lead to illegal-instruction faults on machines
 * where the OS has not enabled AVX context saving.
 */
static void detect_x86(void)
{
    unsigned int eax, ebx, ecx, edx;

    /* Step 1+2: check OSXSAVE and AVX in cpuid leaf 1 */
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return; /* cpuid not supported — very old CPU */
    }

    /* ecx bit 27 = OSXSAVE, ecx bit 28 = AVX */
    if (((ecx >> 27) & 1) == 0 || ((ecx >> 28) & 1) == 0) {
        return;
    }

    /* Step 3: confirm OS has enabled YMM state via xgetbv(0) */
    {
#if defined(__GNUC__) || defined(__clang__)
        /*
         * Use explicit "=a"/"=d" constraints rather than "=A" for uint64_t.
         * In x86-64, "=A" is ambiguous for 64-bit values: the compiler may
         * choose either rax or rdx (not the edx:eax pair as in 32-bit mode).
         * xgetbv writes XCR0[31:0] into eax and XCR0[63:32] into edx; bits 1
         * and 2 (SSE/AVX state) live in the low 32 bits (eax).  If the
         * compiler picks rdx for "=A", xcr0 gets the high 32 bits and the
         * bit check silently fails on every call.  Using "=a" and "=d"
         * separately is unambiguous and immune to register-allocation changes
         * that PGO instrumentation can cause.
         */
        uint32_t xcr0_lo, xcr0_hi;
        __asm__ volatile ("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        (void)xcr0_hi;  /* high 32 bits not needed; bits 1 and 2 are in eax */
        /* bit 1 = SSE state, bit 2 = AVX/YMM state — both must be set */
        if ((xcr0_lo & 0x6) != 0x6) {
            return;
        }
#else
        return; /* can't check without inline asm — be conservative */
#endif
    }

    /* Step 4: check AVX2 in cpuid leaf 7, sub-leaf 0 */
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return;
    }

    /* ebx bit 5 = AVX2 */
    if ((ebx >> 5) & 1) {
        g_caps.has_avx2 = 1;
    }
}

#endif /* __x86_64__ */


#if defined(__aarch64__)

#if defined(__APPLE__)

/*
 * On aarch64 macOS (Apple Silicon), NEON is architecturally mandatory and
 * always available. No runtime detection needed.
 */
static void detect_aarch64(void)
{
    g_caps.has_neon = 1;
}

#else /* aarch64 Linux (and other non-Apple aarch64) */

#include <stdio.h>
#include <string.h>

/*
 * On Linux aarch64, check /proc/cpuinfo for "neon" or "asimd" in the
 * Features line. ASIMD is the AArch64 name for NEON-equivalent capability.
 */
static void detect_aarch64(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f == NULL) {
        return;
    }

    char line[1024];
    while (fgets(line, sizeof(line), f) != NULL) {
        /* Look for a line that starts with "Features" */
        if (strncmp(line, "Features", 8) != 0) {
            continue;
        }
        /* Search the feature list for "neon" or "asimd" */
        if (strstr(line, "neon") != NULL || strstr(line, "asimd") != NULL) {
            g_caps.has_neon = 1;
        }
        break; /* Features line found — no need to keep reading */
    }

    fclose(f);
}

#endif /* __APPLE__ */

#endif /* __aarch64__ */


/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

const fused_cpu_caps_t *fused_detect_cpu(void)
{
    if (!g_detected) {
#if defined(__x86_64__)
        detect_x86();
#elif defined(__aarch64__)
        detect_aarch64();
#endif
        g_detected = 1;
    }

    return &g_caps;
}
