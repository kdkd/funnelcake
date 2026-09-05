/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#ifndef TEST_ALLOC_HOOKS_H
#define TEST_ALLOC_HOOKS_H
/* Force-included only in the two test copies of the initialization sources.
 * Include libc before renaming calls so its declarations stay untouched. */
#include <stdlib.h>
void *test_calloc(size_t count, size_t size);
int test_posix_memalign(void **out, size_t alignment, size_t size);
void test_free(void *ptr);
#define calloc test_calloc
#define posix_memalign test_posix_memalign
#define free test_free
#endif
