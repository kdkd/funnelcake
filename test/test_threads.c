/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#include "funnelcake.h"
#include <pthread.h>
#include <stdatomic.h>
#include <assert.h>

static atomic_int start;
static void *probe(void *result)
{
    while (!atomic_load_explicit(&start, memory_order_acquire)) {}
    int first = fused_simd_available();
    for (int i = 0; i < 1000; ++i) assert(fused_simd_available() == first);
    *(int *)result = first;
    return NULL;
}
int main(void)
{
    pthread_t threads[16];
    int results[16];
    for (int i = 0; i < 16; ++i) assert(!pthread_create(&threads[i], NULL, probe, &results[i]));
    atomic_store_explicit(&start, 1, memory_order_release);
    for (int i = 0; i < 16; ++i) assert(!pthread_join(threads[i], NULL));
    for (int i = 1; i < 16; ++i) assert(results[i] == results[0]);
    return 0;
}
