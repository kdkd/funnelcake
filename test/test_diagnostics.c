/* Copyright (c) 2020-2026 Kevin Day
 * SPDX-License-Identifier: BSD-2-Clause-Patent */
#include "funnelcake.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>
int main(void) {
    assert(strlen(fused_version()) > 0);
    const char *b=fused_backend();
    assert(!strcmp(b,"scalar") || !strcmp(b,"avx2") || !strcmp(b,"avx512") || !strcmp(b,"neon") || !strcmp(b,"rvv"));
    const char *forced=getenv("FUNNELCAKE_FORCE_SCALAR");
    if(forced && forced[0]) assert(!strcmp(b,"scalar"));
    if(!strcmp(b,"scalar")) assert(!fused_simd_available());
    return 0;
}
