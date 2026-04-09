#ifndef FUNNELCAKE_DETECT_H
#define FUNNELCAKE_DETECT_H

/*
 * CPU capability detection for funnelcake.
 *
 * Results are cached after the first call. The function is safe to call from
 * multiple threads after the first initialisation: worst case two threads both
 * write the same values (the detection is idempotent).
 */

typedef struct {
    int has_avx2;   /* x86_64 only - 1 if AVX2 + OS XSAVE for YMM is available */
    int has_neon;   /* aarch64 only - 1 if NEON is available                    */
} fused_cpu_caps_t;

/*
 * fused_detect_cpu - detect and cache CPU capabilities.
 *
 * Returns a pointer to the static capability struct. Never returns NULL.
 * The pointed-to memory is read-only after first initialisation.
 */
const fused_cpu_caps_t *fused_detect_cpu(void);

#endif /* FUNNELCAKE_DETECT_H */
