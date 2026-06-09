/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include "detect.h"
#include "funnelcake.h"   /* for the public fused_simd_available() prototype */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Static cache - zero-initialised at startup */
static fused_cpu_caps_t g_caps = {0, 0, 0, 0};
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
 *  1. The CPU advertises OSXSAVE (cpuid eax=1, ecx bit 27) - OS has enabled
 *     the XSAVE instruction.
 *  2. The CPU advertises AVX support (cpuid eax=1, ecx bit 28).
 *  3. The OS has enabled YMM state in XCR0 (xgetbv xcr0, bits 1 and 2 set).
 *  4. The CPU advertises AVX2 (cpuid eax=7 ecx=0, ebx bit 5).
 *
 * Skipping any of these can lead to illegal-instruction faults on machines
 * where the OS has not enabled AVX context saving.
 */
/*
 * Probe the last-level data cache size.
 *
 * Intel publishes deterministic cache parameters in CPUID leaf 4; AMD
 * (Zen and newer) publishes the identical layout at leaf 0x8000001D.
 * Both are walked subleaf by subleaf and the largest data or unified
 * cache wins.  On AMD this reports the per-CCX L3 share - which is the
 * right number for a single thread, since that is the capacity its
 * working set actually competes for.  Older AMD parts fall back to leaf
 * 0x80000006 (L3 in 512KB units, L2 in KB).  Returns 0 when nothing can
 * be determined.
 */
static size_t probe_llc_bytes(void)
{
    unsigned int eax, ebx, ecx, edx;
    unsigned int max_ext = 0;
    size_t best = 0;

    if (__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx))
        max_ext = eax;

    unsigned int leaves[2] = { 4, 0 };
    if (max_ext >= 0x8000001du)
        leaves[1] = 0x8000001du;

    for (int l = 0; l < 2 && best == 0; l++) {
        if (leaves[l] == 0) continue;
        for (unsigned int sub = 0; sub < 8; sub++) {
            if (__get_cpuid_count(leaves[l], sub, &eax, &ebx, &ecx, &edx) == 0)
                break;
            unsigned int cache_type = eax & 0x1f;
            if (cache_type == 0)
                break;                          /* no more cache levels */
            if (cache_type != 1 && cache_type != 3)
                continue;                       /* data or unified only */
            size_t ways  = ((ebx >> 22) & 0x3ff) + 1;
            size_t parts = ((ebx >> 12) & 0x3ff) + 1;
            size_t line  = (ebx & 0xfff) + 1;
            size_t sets  = (size_t)ecx + 1;
            size_t size  = ways * parts * line * sets;
            if (size > best) best = size;
        }
    }

    if (best == 0 && max_ext >= 0x80000006u &&
        __get_cpuid(0x80000006u, &eax, &ebx, &ecx, &edx)) {
        size_t l3 = (size_t)((edx >> 18) & 0x3fffu) * 512u * 1024u;
        size_t l2 = (size_t)((ecx >> 16) & 0xffffu) * 1024u;
        best = l3 ? l3 : l2;
    }

    return best;
}

static void detect_x86(void)
{
    unsigned int eax, ebx, ecx, edx;

    /* Cache geometry is independent of the SIMD checks below (and their
     * early returns), so probe it first. */
    g_caps.llc_bytes = probe_llc_bytes();

    /* Step 1+2: check OSXSAVE and AVX in cpuid leaf 1 */
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return; /* cpuid not supported - very old CPU */
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
        /* bit 1 = SSE state, bit 2 = AVX/YMM state - both must be set */
        if ((xcr0_lo & 0x6) != 0x6) {
            return;
        }
#else
        return; /* can't check without inline asm - be conservative */
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

#elif defined(__FreeBSD__)

#include <sys/auxv.h>
#include <machine/elf.h>

/*
 * On FreeBSD aarch64, query AT_HWCAP via elf_aux_info() (analogous to
 * Linux's getauxval). HWCAP_ASIMD indicates Advanced SIMD (NEON-equivalent).
 */
static void detect_aarch64(void)
{
    unsigned long hwcap = 0;
    if (elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap)) != 0) {
        return;
    }
    if (hwcap & HWCAP_ASIMD) {
        g_caps.has_neon = 1;
    }
}

#else /* aarch64 Linux (and other non-Apple aarch64) */

#include <stdio.h>

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
        break; /* Features line found - no need to keep reading */
    }

    fclose(f);
}

#endif /* __APPLE__ */

#endif /* __aarch64__ */


#if defined(__riscv) && (__riscv_xlen == 64)

#include <stdio.h>
#include <sys/syscall.h>

/* syscall(2) is not declared under _POSIX_C_SOURCE=200112L (it's gated
 * behind _GNU_SOURCE / _DEFAULT_SOURCE).  Declare it locally rather than
 * widening the feature-test macros for the whole TU. */
extern long syscall(long number, ...);

/*
 * Detect RVV 1.0 (the "V" extension) on riscv64 Linux.
 *
 * Two-tier fallback:
 *   1. riscv_hwprobe() syscall (kernel 6.5+) - the canonical interface,
 *      reports both "is V supported" and a perf hint for misaligned vector
 *      access.  We require V supported AND - when the kernel reports a
 *      definite perf class - misaligned vector access NOT marked SLOW or
 *      EMULATED.  Chips that emulate misaligned vector loads via kernel
 *      trap are slower with vectors than without.
 *   2. Parse /proc/cpuinfo "isa:" line for the V extension.  Last resort.
 *      Reject zve* (embedded vector subset) - the kernels assume full V.
 *
 * Does not depend on <asm/hwprobe.h> being installed: the syscall numbers
 * and struct layout are stable kernel ABI, so we declare them inline.
 */

/* SYS_riscv_hwprobe = 258 has been the riscv64 syscall number since Linux
 * 6.5 (the kernel that introduced the syscall); the values below mirror
 * the upstream uapi definition. */
#ifndef SYS_riscv_hwprobe
# define SYS_riscv_hwprobe 258
#endif

struct fused_riscv_hwprobe {
    int64_t  key;
    uint64_t value;
};

#define FUSED_HWPROBE_KEY_IMA_EXT_0               4
#define FUSED_HWPROBE_KEY_MISALIGNED_VECTOR_PERF  11
#define FUSED_HWPROBE_IMA_V                       (1ULL << 2)
/* Misaligned-vector-perf return classes (from upstream uapi):
 *   0 = UNKNOWN, 2 = SLOW, 3 = FAST, 4 = EMULATED.
 * SLOW and EMULATED disqualify; UNKNOWN and FAST allow RVV. */
#define FUSED_HWPROBE_MISALIGNED_VECTOR_SLOW      2
#define FUSED_HWPROBE_MISALIGNED_VECTOR_EMULATED  4

static int detect_rvv_via_hwprobe(int *out_present, int *out_misaligned_ok)
{
    struct fused_riscv_hwprobe pairs[2] = {
        { FUSED_HWPROBE_KEY_IMA_EXT_0,              0 },
        { FUSED_HWPROBE_KEY_MISALIGNED_VECTOR_PERF, 0 },
    };

    /* args: pairs, pair_count, cpu_set_size, cpu_set, flags */
    long rc = syscall(SYS_riscv_hwprobe, pairs, (long)2, (long)0,
                      (void *)0, (long)0);
    if (rc < 0) {
        return -1; /* syscall not available - fall through to /proc/cpuinfo */
    }

    *out_present = (pairs[0].value & FUSED_HWPROBE_IMA_V) ? 1 : 0;

    uint64_t mv = pairs[1].value;
    *out_misaligned_ok = (mv != FUSED_HWPROBE_MISALIGNED_VECTOR_SLOW &&
                          mv != FUSED_HWPROBE_MISALIGNED_VECTOR_EMULATED);
    return 0;
}

static int detect_rvv_via_cpuinfo(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f == NULL) {
        return 0;
    }

    int found = 0;
    char line[2048];
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, "isa", 3) != 0) {
            continue;
        }
        const char *colon = strchr(line, ':');
        if (colon == NULL) {
            continue;
        }
        const char *isa = colon + 1;

        /* The base ISA spelling packs extension letters together
         * (e.g. "rv64imafdcv"), and named extensions are appended with
         * underscores (e.g. "_zicsr_zifencei").  Walk the inline letters
         * looking for a standalone 'v', then also check for "_v_"/"_v "
         * tokens.  Reject zve*-only chips: an embedded vector subset is
         * not enough for these kernels. */
        const char *rv = strstr(isa, "rv64");
        if (rv != NULL) {
            const char *p = rv + 4;
            while (*p && *p != '_' && *p != ' ' && *p != '\t' && *p != '\n') {
                if (*p == 'v') { found = 1; break; }
                p++;
            }
        }
        if (!found && (strstr(isa, "_v_") != NULL ||
                       strstr(isa, "_v\n") != NULL ||
                       strstr(isa, "_v ")  != NULL)) {
            found = 1;
        }
        break;
    }

    fclose(f);
    return found;
}

static void detect_riscv(void)
{
    int present = 0, misaligned_ok = 0;
    if (detect_rvv_via_hwprobe(&present, &misaligned_ok) == 0) {
        g_caps.has_rvv = (present && misaligned_ok) ? 1 : 0;
        return;
    }

    /* hwprobe unavailable - fall through to /proc/cpuinfo parsing. */
    if (detect_rvv_via_cpuinfo()) {
        g_caps.has_rvv = 1;
    }
}

#endif /* __riscv && __riscv_xlen == 64 */


/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

const fused_cpu_caps_t *fused_detect_cpu(void)
{
    if (!g_detected) {
        /* Test/diagnostic override: setting FUNNELCAKE_FORCE_SCALAR=<non-empty>
         * skips every per-arch SIMD probe, leaving caps zeroed.  Used by the
         * parity test to compare scalar output against SIMD output without
         * running two separate processes. */
        const char *force_scalar = getenv("FUNNELCAKE_FORCE_SCALAR");
        if (force_scalar == NULL || force_scalar[0] == '\0') {
#if defined(__x86_64__)
            detect_x86();
#elif defined(__aarch64__)
            detect_aarch64();
#elif defined(__riscv) && (__riscv_xlen == 64)
            detect_riscv();
#endif
        }
        g_detected = 1;
    }

    return &g_caps;
}

void fused_detect_cpu_reset(void)
{
    memset(&g_caps, 0, sizeof(g_caps));
    g_detected = 0;
}

/*
 * Public capability query. This is the single source of truth for "will the
 * scalers vectorize?": funnelcake.c and funnelcake_hdr.c gate their internal
 * has_simd flag on these exact same caps fields, per the same arch #if blocks,
 * so this stays in lock-step with the kernels they actually select.
 */
int fused_simd_available(void)
{
    const fused_cpu_caps_t *caps = fused_detect_cpu();
#if defined(__x86_64__)
    return caps->has_avx2 ? 1 : 0;
#elif defined(__aarch64__)
    return caps->has_neon ? 1 : 0;
#elif defined(__riscv) && (__riscv_xlen == 64)
    return caps->has_rvv ? 1 : 0;
#else
    (void)caps;
    return 0;
#endif
}
