/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#ifndef FUNNELCAKE_DETECT_H
#define FUNNELCAKE_DETECT_H

#include <stddef.h>  /* size_t */

/*
 * CPU capability detection for funnelcake.
 *
 * Results are cached after the first call. The function is safe to call from
 * multiple threads after the first initialisation: worst case two threads both
 * write the same values (the detection is idempotent).
 *
 * Setting the environment variable FUNNELCAKE_FORCE_SCALAR=1 (any non-empty
 * value) before the first call disables all SIMD detection across every
 * platform.  Used by the parity test to compare scalar against SIMD output.
 */

typedef struct {
    int has_avx2;   /* x86_64 only - 1 if AVX2 + OS XSAVE for YMM is available */
    int has_neon;   /* aarch64 only - 1 if NEON is available                    */
    int has_rvv;    /* riscv64 only - 1 if RVV 1.0 (V extension) is available
                     * AND misaligned vector access is fast on the chip        */
    size_t llc_bytes; /* x86_64 only - last-level data cache size in bytes as
                       * CPUID reports it for one core (per-CCX on AMD, per
                       * socket on Intel).  0 = unknown.  Kernels use this to
                       * judge whether an output plane can stay cache-resident
                       * or should be written with streaming stores.          */
} fused_cpu_caps_t;

/*
 * fused_detect_cpu - detect and cache CPU capabilities.
 *
 * Returns a pointer to the static capability struct. Never returns NULL.
 * The pointed-to memory is read-only after first initialisation.
 */
const fused_cpu_caps_t *fused_detect_cpu(void);

/*
 * fused_detect_cpu_reset - clear the cached detection result.
 *
 * Test-only entry point: lets the parity test toggle FUNNELCAKE_FORCE_SCALAR
 * between successive scaler initializations and have the new env value take
 * effect.  Not intended for production use; not declared in the public
 * include/funnelcake.h header.
 */
void fused_detect_cpu_reset(void);

#endif /* FUNNELCAKE_DETECT_H */
